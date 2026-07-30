"""
사용법:
python -m analysis.diagnose_temporal_usage \
--teacher_ckpt /home/DAHS1/gangmin/my_research/clinical_multimodal_learning/checkpoints_duett/teacher/20260720_163718_aug_mask=0.15_aug_noise=0.05_aux_img_alpha=0.3_aux_ts_alpha=0.7_eval_train_batches=100_freeze_duett=True_label_weights=1.0,1.0,1.0_perceiver_type=dual_patch/best.pt \
--split val --bootstrap 2000
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

import sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from training_duett.data_processing import (AnchorConfig, build_datasets, duett_kd_collate)
from training_duett.engine import _move_lists
from models.main_architecture_duett import (
    CXREncoder,
    DualPathologyPerceiver,
    PatchDualPathologyPerceiver,
    TeacherModel,
    load_duett_backbone,
)


CONDITIONS = (
    "full",
    "patient_shuffle",
    "ts_shuffle",
    "time_reverse",
    "time_permute",
    # "image_shuffle",   # positive control — 필요 시 주석 해제
)


class _SubjectAnnotatedDataset(Dataset):
    """Attach subject_id while leaving the training dataset implementation intact."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        item["_subject_id"] = int(self.base.anchor_df.iloc[index]["subject_id"])
        return item


def _diagnostic_collate(batch: list[dict]) -> dict:
    out = duett_kd_collate(batch, mode="teacher")
    out["_subject_id"] = torch.tensor(
        [item["_subject_id"] for item in batch], dtype=torch.long)
    return out


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        return torch.load(path, map_location="cpu")


def _saved_args(checkpoint: dict) -> dict:
    saved = checkpoint.get("args", {})
    if isinstance(saved, dict):
        return saved
    if hasattr(saved, "__dict__"):
        return vars(saved)
    return {}


def _pick(cli_value, saved: dict, key: str, default=None):
    if cli_value is not None:
        return cli_value
    return saved.get(key, default)


def _parse_labels(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(x.strip() for x in value.split(",") if x.strip())
    return tuple(value)


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def _different_subject_permutation(subject_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a within-batch permutation maximizing cross-subject pairing."""
    n = len(subject_ids)
    if n <= 1:
        return np.arange(n)

    for _ in range(100):
        perm = rng.permutation(n)
        if np.all(subject_ids[perm] != subject_ids):
            return perm

    # Repeated CXRs can make a perfect derangement impossible in a batch.
    # Choose the cyclic shift with the fewest same-subject pairs.
    best_perm = np.roll(np.arange(n), 1)
    best_matches = int(np.sum(subject_ids[best_perm] == subject_ids))
    for shift in range(2, n):
        candidate = np.roll(np.arange(n), shift)
        matches = int(np.sum(subject_ids[candidate] == subject_ids))
        if matches < best_matches:
            best_perm, best_matches = candidate, matches
            if matches == 0:
                break
    return best_perm


def _permute_tuple(values: tuple[torch.Tensor, ...], perm: np.ndarray):
    return tuple(values[int(j)] for j in perm)


def _encode_ts(model: TeacherModel,
               x_ts: tuple[torch.Tensor, ...],
               x_static: tuple[torch.Tensor, ...],
               bin_ends: tuple[torch.Tensor, ...]
               ) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(x_ts)
    duett_in = model.duett.feats_to_input((x_ts, x_static, bin_ends), batch_size)
    return model.duett.encode(duett_in)


def _encode_image_patches(model: TeacherModel,
                          pixel_values: torch.Tensor) -> torch.Tensor:
    """dual_patch용: raw patches → img_proj → [B, N, D_latent].

    Training(teacher forward)의 dual_patch 분기와 반드시 일치해야 함.
    현재 세팅: 37×37=1369 patches raw pass-through (spatial pooling 없음).
    """
    _, img_patches = model.cxr(pixel_values)     # [B, N, D_img]
    return model.img_proj(img_patches)           # [B, N, D_latent]


def _encode_image_logits(model: TeacherModel,
                         pixel_values: torch.Tensor) -> torch.Tensor:
    """dual용: CXR CLS → frozen pretrained head → keep_idx로 K개 pathology logit 추출."""
    cls = model.cxr(pixel_values)
    if isinstance(cls, tuple):
        cls = cls[0]
    pretrained_logits = model.pretrained_cxr_head(cls)      # [B, C_pretrain]
    return pretrained_logits[:, model.cxr_head_keep_idx]    # [B, K]


def _image_branch_input(model: TeacherModel, pixel_values: torch.Tensor,
                        perceiver_type: str) -> torch.Tensor:
    """Perceiver의 두 번째 인자로 넘길 image 입력을 perceiver_type에 맞게 계산."""
    if perceiver_type == "dual":
        return _encode_image_logits(model, pixel_values)
    if perceiver_type == "dual_patch":
        return _encode_image_patches(model, pixel_values)
    raise ValueError(f"unsupported perceiver_type={perceiver_type!r}")


def _time_permuted(x_ts: tuple[torch.Tensor, ...],
                   rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
    out = []
    for x in x_ts:
        perm = torch.as_tensor(rng.permutation(x.shape[0]),
                               dtype=torch.long, device=x.device)
        out.append(x.index_select(0, perm))
    return tuple(out)


def _safe_metric(fn: Callable, y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(fn(y, p))
    except ValueError:
        return float("nan")


def _prob(logits: np.ndarray) -> np.ndarray:
    x = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _metrics_by_label(logits: np.ndarray,
                      y: np.ndarray,
                      mask: np.ndarray,
                      labels: tuple[str, ...]) -> list[dict]:
    probs = _prob(logits)
    rows = []
    for k, label in enumerate(labels):
        valid = mask[:, k].astype(bool)
        yk, pk = y[valid, k], probs[valid, k]
        rows.append({
            "label": label,
            "n": int(valid.sum()),
            "pos_frac": float(yk.mean()) if len(yk) else float("nan"),
            "auroc": _safe_metric(roc_auc_score, yk, pk),
            "auprc": _safe_metric(average_precision_score, yk, pk),
        })
    return rows


def _cluster_bootstrap_delta(subject_ids: np.ndarray,
                             y: np.ndarray,
                             valid: np.ndarray,
                             p_full: np.ndarray,
                             p_ablated: np.ndarray,
                             metric_fn: Callable,
                             n_boot: int,
                             seed: int) -> tuple[float, float, float, int]:
    """Subject-cluster bootstrap CI for metric(full) - metric(ablated)."""
    ids = np.unique(subject_ids)
    groups = {sid: np.flatnonzero(subject_ids == sid) for sid in ids}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([groups[sid] for sid in sampled])
        idx = idx[valid[idx]]
        if len(idx) == 0 or np.unique(y[idx]).size < 2:
            continue
        full_m = _safe_metric(metric_fn, y[idx], p_full[idx])
        abl_m = _safe_metric(metric_fn, y[idx], p_ablated[idx])
        if np.isfinite(full_m) and np.isfinite(abl_m):
            deltas.append(full_m - abl_m)
    if not deltas:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.asarray(deltas)
    return (float(arr.mean()), float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)), int(len(arr)))


@torch.no_grad()
def collect_predictions(model: TeacherModel,
                        loader: DataLoader,
                        device: torch.device,
                        perceiver_type: str,
                        seed: int) -> dict:
    """Dual / dual_patch teacher에 대해 조건별 fusion/ts logit + full 조건 img logit 수집.

    Image branch는 TS와 무관하므로 조건별로 재계산하지 않고 batch당 1회만 계산.
    """
    model.eval()
    fus = {c: [] for c in CONDITIONS}
    ts = {c: [] for c in CONDITIONS}
    img_full = []
    y_all, mask_all, attn_all, subject_all = [], [], [], []
    shuffled_same_subject = 0
    shuffled_total = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="temporal diagnostics")):
        batch_subject_ids = batch["_subject_id"].numpy()
        subject_all.append(batch["_subject_id"].clone())
        b = _move_lists(batch, device)
        batch_size = len(b["x_ts"])

        rng = np.random.default_rng(seed + 10007 * batch_idx)
        cross_perm = _different_subject_permutation(batch_subject_ids, rng)
        shuffled_same_subject += int(
            np.sum(batch_subject_ids[cross_perm] == batch_subject_ids))
        shuffled_total += batch_size

        img_input = _image_branch_input(model, b["pixel_values"], perceiver_type)
        ts_full = _encode_ts(model, b["x_ts"], b["x_static"], b["bin_ends"])

        ts_inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {"full": ts_full}

        # Shuffle the complete EHR package (dynamic + static + time-bin metadata).
        ts_inputs["patient_shuffle"] = _encode_ts(
            model,
            _permute_tuple(b["x_ts"], cross_perm),
            _permute_tuple(b["x_static"], cross_perm),
            _permute_tuple(b["bin_ends"], cross_perm),
        )

        # Shuffle dynamic measurements only, keeping the original static data.
        ts_inputs["ts_shuffle"] = _encode_ts(
            model,
            _permute_tuple(b["x_ts"], cross_perm),
            b["x_static"], b["bin_ends"],
        )

        reversed_x = tuple(torch.flip(x, dims=(0,)) for x in b["x_ts"])
        ts_inputs["time_reverse"] = _encode_ts(
            model, reversed_x, b["x_static"], b["bin_ends"])

        permuted_x = _time_permuted(b["x_ts"], rng)
        ts_inputs["time_permute"] = _encode_ts(
            model, permuted_x, b["x_static"], b["bin_ends"])

        # img_perm = torch.as_tensor(cross_perm, dtype=torch.long,
        #                            device=img_input.device)
        # img_shuffled = img_input.index_select(0, img_perm)
        # (image_shuffle 필요 시 여기서 별도 perceiver 호출)

        outputs = {}
        for cond, (ts_tokens, ts_event_grid) in ts_inputs.items():
            need_attn = (cond == "full")
            if perceiver_type == "dual_patch":
                outputs[cond] = model.perceiver(
                    ts_event_grid, img_input, return_attn=need_attn
                )
            else:
                outputs[cond] = model.perceiver(
                    ts_tokens, img_input,
                    return_attn=need_attn, ts_ablation="hourly_only")

        for cond, out in outputs.items():
            fus[cond].append(out["fusion_logits"].cpu())
            ts[cond].append(out["ts_logits"].cpu())
        img_full.append(outputs["full"]["img_logits"].cpu())
        y_all.append(b["y_multi"].cpu())
        mask_all.append(b["y_multi_mask"].cpu())
        attention_key = "event_attn" if perceiver_type == "dual_patch" else "ts_attn"
        attn_all.append(outputs["full"][attention_key].cpu())

    return {
        "fus":    {c: torch.cat(v).float().numpy() for c, v in fus.items()},
        "ts":     {c: torch.cat(v).float().numpy() for c, v in ts.items()},
        "img":    torch.cat(img_full).float().numpy(),  # invariant across conds
        "y":      torch.cat(y_all).float().numpy(),
        "mask":   torch.cat(mask_all).float().numpy(),
        "subject_ids": torch.cat(subject_all).numpy(),
        "attention":   torch.cat(attn_all).float().numpy(),
        "shuffle_same_subject": shuffled_same_subject,
        "shuffle_total": shuffled_total,
    }


def _print_results(pred: dict,
                   labels: tuple[str, ...],
                   n_boot: int,
                   seed: int,
                   attention_axis: str):
    y, mask = pred["y"], pred["mask"]
    subject_ids = pred["subject_ids"]

    # ── [1] Full 조건 baseline: img / ts / fus ─────────────────────
    img_full_m = _metrics_by_label(pred["img"],           y, mask, labels)
    ts_full_m  = _metrics_by_label(pred["ts"]["full"],    y, mask, labels)
    fus_full_m = _metrics_by_label(pred["fus"]["full"],   y, mask, labels)

    print("\n[1] Original checkpoint under FULL input: img / ts / fus per-label")
    print(f"{'label':<24s} {'n':>6s} {'pos':>7s} "
          f"{'img_roc':>8s} {'ts_roc':>8s} {'fus_roc':>8s}  "
          f"{'img_prc':>8s} {'ts_prc':>8s} {'fus_prc':>8s}")
    for ri, rt, rf in zip(img_full_m, ts_full_m, fus_full_m):
        print(f"{ri['label']:<24s} {ri['n']:>6d} {ri['pos_frac']:>7.4f} "
              f"{ri['auroc']:>8.4f} {rt['auroc']:>8.4f} {rf['auroc']:>8.4f}  "
              f"{ri['auprc']:>8.4f} {rt['auprc']:>8.4f} {rf['auprc']:>8.4f}")

    # ── [2] Fusion logit AUROC/AUPRC vs full baseline per condition ─
    fus_by_cond = {c: _metrics_by_label(pred["fus"][c], y, mask, labels)
                   for c in CONDITIONS}
    print("\n[2] Fusion logits under counterfactual TS inputs "
          "(delta = ablated - full; negative = 성능 하락)")
    print(f"{'condition':<18s} {'label':<24s} {'AUROC':>9s} {'d_ROC':>9s} "
          f"{'AUPRC':>9s} {'d_PRC':>9s}")
    for condition in CONDITIONS:
        for k, row in enumerate(fus_by_cond[condition]):
            base = fus_by_cond["full"][k]
            print(f"{condition:<18s} {row['label']:<24s} "
                  f"{row['auroc']:>9.4f} {row['auroc'] - base['auroc']:>+9.4f} "
                  f"{row['auprc']:>9.4f} {row['auprc'] - base['auprc']:>+9.4f}")

    # ── [3] TS logit sensitivity + img invariance sanity ─────────────
    print("\n[3] Sensitivity to TS corruption (label=Edema, idx 0)")
    print(f"{'condition':<18s} {'mean|dp fus|':>14s} {'corr fus':>10s} "
          f"{'mean|dp ts|':>13s} {'max|d img|':>12s}")
    p_full_fus = _prob(pred["fus"]["full"][:, 0])
    p_full_ts  = _prob(pred["ts"]["full"][:, 0])
    for condition in CONDITIONS[1:]:
        p_cond_fus = _prob(pred["fus"][condition][:, 0])
        p_cond_ts  = _prob(pred["ts"][condition][:, 0])
        corr = np.corrcoef(p_full_fus, p_cond_fus)[0, 1]
        # img_logits는 TS 무관 → 조건별 저장 안 하고 full 1회. 항상 0.
        img_diff = 0.0
        print(f"{condition:<18s} {np.mean(np.abs(p_full_fus - p_cond_fus)):>14.6f} "
              f"{corr:>10.6f} {np.mean(np.abs(p_full_ts - p_cond_ts)):>13.6f} "
              f"{img_diff:>12.6g}")
    print("img logits는 TS 무관 → invariance는 항상 0 (perceiver 입력에서 image branch가 TS와 분리)")

    # ── [4] Temporal attention entropy (full 조건) ───────────────────
    attn = pred["attention"]             # [N, K, V] or [N, K, T*]
    if attn.size > 0 and attn.ndim == 3 and attn.shape[-1] > 0:
        normalized_attn = attn / np.clip(attn.sum(axis=-1, keepdims=True), 1e-12, None)
        entropy = -(normalized_attn * np.log(np.clip(normalized_attn, 1e-12, None))).sum(axis=-1)
        axis_size = normalized_attn.shape[-1]
        entropy = entropy / max(math.log(axis_size), 1e-12)
        print(f"\n[4] Full-condition TS attention ({attention_axis}, N={axis_size}): "
              "normalized entropy per label")
        print(f"{'label':<24s} {'mean entropy':>14s}")
        for k, label in enumerate(labels):
            print(f"{label:<24s} {entropy[:, k].mean():>14.6f}")

    same = pred["shuffle_same_subject"]
    total = pred["shuffle_total"]
    print(f"\nCross-patient shuffle audit: same-subject pairs={same}/{total} "
          f"({same / max(total, 1):.4%})")

    if n_boot <= 0:
        return

    print(f"\n[5] Main label (idx 0) subject-cluster paired bootstrap "
          f"({n_boot} replicates)")
    print("Delta = full - ablated; positive → 원본 TS 사용이 유리")
    print(f"{'condition':<18s} {'metric':<7s} {'mean delta':>11s} "
          f"{'95% CI':>24s} {'valid':>7s}")
    valid = mask[:, 0].astype(bool)
    y0 = y[:, 0]
    for ci, condition in enumerate(CONDITIONS[1:], start=1):
        pc = _prob(pred["fus"][condition][:, 0])
        for metric_name, metric_fn in (
            ("AUROC", roc_auc_score),
            ("AUPRC", average_precision_score),
        ):
            mean_d, lo, hi, n_valid = _cluster_bootstrap_delta(
                subject_ids, y0, valid, p_full_fus, pc, metric_fn,
                n_boot=n_boot,
                seed=seed + 1000 * ci + (0 if metric_name == "AUROC" else 1),
            )
            print(f"{condition:<18s} {metric_name:<7s} {mean_d:>+11.5f} "
                  f"[{lo:>+9.5f}, {hi:>+9.5f}] {n_valid:>7d}")


def build_model_and_loader(args, checkpoint: dict, device: torch.device):
    """Dual / dual_patch teacher를 checkpoint에서 복원 + diagnostic loader 준비.

    Returns:
        teacher, loader, pathology_labels, perceiver_type
    """
    saved = _saved_args(checkpoint)
    perceiver_type = _pick(args.perceiver_type, saved, "perceiver_type")
    if perceiver_type not in ("dual", "dual_patch"):
        raise NotImplementedError(
            f"perceiver_type={perceiver_type!r} 미지원. dual / dual_patch 만 지원.")

    final_df_path = _pick(args.final_df_path, saved, "final_df_path")
    static_path = _pick(args.static_path, saved, "static_path")
    duett_ckpt = _pick(args.duett_ckpt, saved, "duett_ckpt")
    if not all((final_df_path, static_path, duett_ckpt)):
        raise ValueError(
            "final_df_path/static_path/duett_ckpt must be in checkpoint args "
            "or supplied on the CLI")

    meta_path = _pick(args.meta_path, saved, "meta_path",
                      os.path.join(os.path.dirname(duett_ckpt), "meta_with_stats.pkl"))
    label_col = _pick(args.label_col, saved, "label_col", "label_edema")
    pathology_labels = _parse_labels(
        _pick(args.pathology_labels, saved, "pathology_labels",
              "label_edema,label_cardiomegaly,label_effusion"))
    n_timesteps = int(_pick(args.n_timesteps, saved, "n_timesteps", 24))
    split_seed = int(saved.get("split_seed", 42))
    cxr_model_name = _pick(args.cxr_model_name, saved, "cxr_model_name",
                           "microsoft/rad-dino")
    pretrained_cxr_head_ckpt = saved.get("pretrained_cxr_head_ckpt", None)

    processor = AutoImageProcessor.from_pretrained(cxr_model_name)
    cfg = AnchorConfig(
        final_df_path=final_df_path,
        static_path=static_path,
        meta_path=meta_path,
        label_col=label_col,
        n_timesteps=n_timesteps,
        split_seed=split_seed,
        pathology_labels=pathology_labels,
    )
    bundle = build_datasets(cfg, image_processor=processor, include_cxr=True)
    base_dataset = bundle["datasets"][args.split]
    if args.cxr_root is not None:
        base_dataset.cxr_root = args.cxr_root
    dataset = _SubjectAnnotatedDataset(base_dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        # Randomized once with a fixed generator so within-batch EHR shuffling
        # is drawn from the whole split rather than neighboring table rows.
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_diagnostic_collate,
    )

    state = _strip_module_prefix(checkpoint["model"])
    d_latent = int(saved.get("d_latent", 256))
    n_heads = int(saved.get("n_perceiver_heads", 4))
    dropout = float(saved.get("perceiver_dropout", 0.1))
    head_hidden = int(saved.get("head_hidden", 64))
    head_dropout = float(saved.get("head_dropout", 0.1))

    backbone = load_duett_backbone(
        ckpt_path=duett_ckpt,
        d_static_num=int(bundle["meta"]["D_STATIC"]),
        d_time_series_num=len(bundle["ts_vars"]),
        n_timesteps=n_timesteps,
        freeze=True,
        aug_noise=0.0,
        aug_mask=0.0,
        transformer_dropout=0.0,
    )
    K = len(pathology_labels)
    if perceiver_type == "dual":
        cxr = CXREncoder(model_name=cxr_model_name, freeze=True, return_patches=False)
        perceiver = DualPathologyPerceiver(
            n_pathologies=K, d_ts=backbone.d_representation,
            d_latent=d_latent, n_heads=n_heads, dropout=dropout)
        teacher = TeacherModel(
            backbone, cxr, perceiver,
            head_hidden=head_hidden, head_dropout=head_dropout,
            cxr_return_patches=False, d_img=cxr.d_out,
            use_aux_cxr=False,
            dual_pathology_mode=True,
            pretrained_cxr_head_ckpt=pretrained_cxr_head_ckpt,
            pathology_labels=pathology_labels,
        )
    else:  # dual_patch
        cxr = CXREncoder(model_name=cxr_model_name, freeze=True, return_patches=True)
        perceiver = PatchDualPathologyPerceiver(
            n_pathologies=K,
            d_latent=d_latent, n_heads=n_heads, dropout=dropout,
            n_timesteps=n_timesteps,
            d_event_embedding=backbone.d_embedding)
        teacher = TeacherModel(
            backbone, cxr, perceiver,
            head_hidden=head_hidden, head_dropout=head_dropout,
            cxr_return_patches=True, d_img=cxr.d_out,
            use_aux_cxr=False,
            patch_dual_pathology_mode=True,
        )

    teacher.load_state_dict(state, strict=True)
    teacher.eval().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    return teacher, loader, pathology_labels, perceiver_type


def parse_args():
    p = argparse.ArgumentParser("Dual/DualPatch teacher temporal-usage diagnostics")
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="subject-cluster paired bootstrap replicates; 0 disables")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default=None,
                   help="e.g. cuda, cuda:1, cpu; default=cuda if available")
    p.add_argument("--output_npz", default=None,
                   help="optional path for raw aligned predictions")

    # Optional path/config overrides for a checkpoint copied to another server.
    p.add_argument("--final_df_path", default=None)
    p.add_argument("--static_path", default=None)
    p.add_argument("--duett_ckpt", default=None)
    p.add_argument("--meta_path", default=None)
    p.add_argument("--cxr_root", default=None)
    p.add_argument("--cxr_model_name", default=None)
    p.add_argument("--label_col", default=None)
    p.add_argument("--pathology_labels", default=None)
    p.add_argument("--n_timesteps", type=int, default=None)
    p.add_argument("--perceiver_type", default=None,
                   choices=("dual", "dual_patch"),
                   help="checkpoint args와 다를 때만 override. 미지정 시 checkpoint에서 읽음.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.split == "test":
        print("WARNING: use val for model diagnosis/selection. "
              "Reserve test for the final locked evaluation.")
    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  split={args.split}")

    checkpoint = _torch_load(args.teacher_ckpt)
    model, loader, labels, perceiver_type = build_model_and_loader(
        args, checkpoint, device)
    print(f"perceiver_type={perceiver_type}  labels={labels}")
    pred = collect_predictions(model, loader, device, perceiver_type, args.seed)
    _print_results(
        pred,
        labels,
        args.bootstrap,
        args.seed,
        attention_axis=("variables" if perceiver_type == "dual_patch"
                        else "time tokens"),
    )

    if args.output_npz:
        output_dir = os.path.dirname(os.path.abspath(args.output_npz))
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "subject_ids": pred["subject_ids"],
            "labels": np.asarray(labels),
            "y": pred["y"],
            "mask": pred["mask"],
            "img_full": pred["img"],
            "ts_attention_full": pred["attention"],
        }
        for condition in CONDITIONS:
            payload[f"fus_{condition}"] = pred["fus"][condition]
            payload[f"ts_{condition}"]  = pred["ts"][condition]
        np.savez_compressed(args.output_npz, **payload)
        print(f"saved raw predictions: {args.output_npz}")


if __name__ == "__main__":
    main()
