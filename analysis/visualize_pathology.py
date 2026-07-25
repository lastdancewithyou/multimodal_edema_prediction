"""Visualize a trained PathologyPerceiver teacher checkpoint.

Loads a `best.pt` produced by `training_duett.trainer.train_teacher` (with
`--use_pathology_perceiver`) and generates 5 figures:

    1. patch_attention_overlay.png — per-sample × per-pathology CXR + attention heatmap
    2. ts_attention_heatmap.png    — per-sample K × (T+1) TS attention heatmap
    3. query_similarity.png        — K×K cosine similarity of learned pathology queries
    4. stage4_token_umap.png       — UMAP (or t-SNE fallback) of stage4 tokens, colored by pathology
    5. gap_summary_test.png/csv    — bar chart of stage2 vs stage4 AUROC per pathology (+ table)

Usage:
    python analysis/visualize_pathology.py \\
        --ckpt <path>/best.pt \\
        --outdir analysis/figs_<name>/ \\
        --n_samples 8
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoImageProcessor

from training_duett.data_processing import (AnchorConfig, DEFAULT_PATHOLOGY_LABELS,
                                              build_datasets, duett_kd_collate)
from training_duett.engine import _move_lists
from training_duett.evaluator import (evaluate_pathology, format_pathology_gap_table,
                                        evaluate_dual_pathology, format_dual_pathology_gap_table)
from models.main_architecture_duett import (
    CXREncoder, DualPathologyPerceiver, PatchDualPathologyPerceiver,
    TeacherModel, load_duett_backbone,
)
try:
    from models.main_architecture_duett import PathologyPerceiver
except ImportError:
    PathologyPerceiver = None


# =============================================================================
# Setup
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser("Visualize trained PathologyPerceiver")
    p.add_argument("--ckpt",       type=str, required=True, help="best.pt path")
    p.add_argument("--outdir",     type=str, required=True)
    p.add_argument("--n_samples",  type=int, default=8, help="attention viz sample count")
    p.add_argument("--split",      type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dim_reduce", type=str, default="auto", choices=["auto", "umap", "tsne"],
                   help="stage4 token 투영 방식. auto = umap 있으면 사용, 없으면 tsne")
    return p.parse_args()


def _loader(ds, batch_size, num_workers, mode="teacher"):
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=lambda b: duett_kd_collate(b, mode))


def _detect_mode(ck_args: dict) -> str:
    """Ckpt args에서 perceiver 모드 추론. 'single' | 'dual' | 'dual_patch'."""
    ptype = ck_args.get("perceiver_type")
    if ptype in ("dual", "dual_patch", "single"):
        return ptype
    # 레거시: perceiver_type 이 없던 시절 ckpt는 use_pathology_perceiver 로 판단
    if ck_args.get("use_pathology_perceiver", False):
        return "single"
    raise RuntimeError("Ckpt가 query-based perceiver로 학습되지 않았습니다 "
                       "(perceiver_type ∈ {single,dual,dual_patch} 이어야 함).")


def load_teacher(ckpt_path: str, device):
    """Ckpt에서 args를 복원해서 teacher 재구성 → state_dict 로드.
    Returns: (teacher, bundle, processor, args, pathology_labels, mode)
        mode ∈ {"single", "dual"}
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = state["args"]
    args = SimpleNamespace(**ck_args)
    mode = _detect_mode(ck_args)

    # 학습 시 pathology_labels 는 args에 저장되지 않고 DEFAULT_PATHOLOGY_LABELS 상수에서 가져옴.
    # (구버전 ckpt 대비 args.pathology_labels 도 폴백으로 시도.)
    raw = ck_args.get("pathology_labels")
    if isinstance(raw, str):
        pathology_labels = tuple(s.strip() for s in raw.split(","))
    elif isinstance(raw, (list, tuple)) and len(raw) > 0:
        pathology_labels = tuple(raw)
    else:
        pathology_labels = tuple(DEFAULT_PATHOLOGY_LABELS)
    # 무결성 체크: labels[0] 은 반드시 main task label_col 이어야 함
    if pathology_labels[0] != args.label_col:
        raise ValueError(f"pathology_labels[0]={pathology_labels[0]!r} != "
                         f"label_col={args.label_col!r}")
    cfg = AnchorConfig(
        final_df_path=args.final_df_path,
        static_path=args.static_path,
        meta_path=args.meta_path,
        label_col=args.label_col,
        n_timesteps=args.n_timesteps,
        split_seed=args.split_seed,
        pathology_labels=pathology_labels,
    )
    processor = AutoImageProcessor.from_pretrained(args.cxr_model_name)
    bundle = build_datasets(cfg, image_processor=processor, include_cxr=True)

    meta = bundle["meta"]
    backbone = load_duett_backbone(
        ckpt_path=args.duett_ckpt,
        d_static_num=int(meta["D_STATIC"]),
        d_time_series_num=len(bundle["ts_vars"]),
        n_timesteps=args.n_timesteps,
        freeze=False,
    )
    cxr_enc = CXREncoder(model_name=args.cxr_model_name, freeze=True, return_patches=True)

    if mode == "dual":
        perceiver = DualPathologyPerceiver(
            n_pathologies=len(pathology_labels),
            d_ts=backbone.d_representation,
            d_latent=args.d_latent,
            n_heads=args.n_perceiver_heads,
            dropout=0.0,
        )
    elif mode == "dual_patch":
        perceiver = PatchDualPathologyPerceiver(
            n_pathologies=len(pathology_labels),
            d_ts=backbone.d_representation,
            d_latent=args.d_latent,
            n_heads=args.n_perceiver_heads,
            dropout=0.0,
        )
    else:  # "single"
        if PathologyPerceiver is None:
            raise ImportError("Single(PathologyPerceiver) ckpt인데 class가 주석 처리됨. "
                              "models/main_architecture_duett.py에서 un-comment 필요.")
        perceiver = PathologyPerceiver(
            n_pathologies=len(pathology_labels),
            d_ts=backbone.d_representation,
            d_latent=args.d_latent,
            n_heads=args.n_perceiver_heads,
            dropout=0.0,
        )

    # dual (residual) 는 pretrained CXR head ckpt가 필요
    pretrained_cxr_head_ckpt = None
    ckpt_pathology_labels = None
    if mode == "dual":
        pretrained_cxr_head_ckpt = getattr(args, "pretrained_cxr_head_ckpt", None)
        ckpt_pathology_labels = pathology_labels

    teacher = TeacherModel(
        backbone, cxr_enc, perceiver,
        head_hidden=args.head_hidden, head_dropout=0.0,
        cxr_return_patches=True,
        d_img=cxr_enc.d_out,
        use_aux_cxr=False,
        pathology_mode=(mode == "single"),
        dual_pathology_mode=(mode == "dual"),
        patch_dual_pathology_mode=(mode == "dual_patch"),
        pretrained_cxr_head_ckpt=pretrained_cxr_head_ckpt,
        pathology_labels=ckpt_pathology_labels,
    )
    teacher.load_state_dict(state["model"])
    teacher.to(device).eval()
    return teacher, bundle, processor, args, pathology_labels, mode


def _unnormalize(pv: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=pv.device).view(3, 1, 1)
    s = torch.tensor(std, device=pv.device).view(3, 1, 1)
    return (pv * s + m).clamp(0, 1)


def _short(name: str) -> str:
    return name.replace("label_", "")


# =============================================================================
# Fig 1: Per-sample × per-pathology patch attention overlay on CXR
# =============================================================================
def viz_patch_attention(teacher, loader, device, processor, pathology_labels,
                        n_samples: int, outpath: str):
    """K rows(pathology) × n_samples cols. 각 row는 그 pathology 에 대해 y=1 인 sample 만.

    각 셀 = "pathology k query가, 실제 병변 있는 환자의 CXR 어디에 attend 했는가".
    positive sample이 없어서 채워지지 않은 행은 남은 셀을 빈 상태로 둠.
    """
    K = len(pathology_labels)
    per_path = [[] for _ in range(K)]  # per_path[k] = list of (pv, attn_k) tuples

    with torch.no_grad():
        for batch in loader:
            b = _move_lists(batch, device)
            out = teacher(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"],
                          return_attn=True)
            pvs   = b["pixel_values"].cpu()      # [B, 3, H, W]
            attn  = out["img_attn"].cpu()         # [B, K, N]
            y     = b["y_multi"].cpu()            # [B, K]
            mask  = b["y_multi_mask"].cpu()       # [B, K]
            for i in range(pvs.shape[0]):
                for k in range(K):
                    if len(per_path[k]) >= n_samples:
                        continue
                    if bool(mask[i, k].item()) and int(y[i, k].item()) == 1:
                        per_path[k].append((pvs[i], attn[i, k]))
            if all(len(pp) >= n_samples for pp in per_path):
                break

    filled_per_k = [len(pp) for pp in per_path]
    ncols = n_samples
    nrows = K
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    if nrows == 1: axes = axes[None, :]
    if ncols == 1: axes = axes[:, None]

    for k in range(K):
        for col in range(ncols):
            ax = axes[k, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= filled_per_k[k]:
                ax.axis("off")
                continue
            pv, at_k = per_path[k][col]
            img = _unnormalize(pv, processor.image_mean, processor.image_std)
            img_np = img.permute(1, 2, 0).numpy()
            H, W = img_np.shape[:2]
            N_patch = at_k.shape[-1]
            side = int(N_patch ** 0.5)
            if side * side != N_patch:
                raise RuntimeError(f"img_attn patch count가 정사각형이 아님: N={N_patch}")
            a_map = at_k.reshape(side, side)
            a_up = F.interpolate(a_map[None, None].float(), size=(H, W),
                                 mode="bicubic", align_corners=False)[0, 0].numpy()
            a_up = (a_up - a_up.min()) / (a_up.max() - a_up.min() + 1e-8)
            ax.imshow(img_np, cmap="gray")
            ax.imshow(a_up, cmap="jet", alpha=0.4)
            if col == 0:
                ax.set_ylabel(_short(pathology_labels[k]), fontsize=11)
            if k == 0:
                ax.set_title(f"pos #{col}", fontsize=9)

    for k in range(K):
        if filled_per_k[k] < n_samples:
            print(f"[viz] warn: {_short(pathology_labels[k])} 에서 y=1 sample "
                  f"{filled_per_k[k]}/{n_samples} 개만 확보")

    fig.suptitle("Per-pathology patch attention on y=1 samples "
                 f"(patch grid = {int(at_k.shape[-1] ** 0.5)}×{int(at_k.shape[-1] ** 0.5)})",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved: {outpath}")


# =============================================================================
# Fig 2: TS attention heatmap per sample (K × (T+1))
# =============================================================================
def viz_ts_attention(teacher, loader, device, pathology_labels,
                     n_samples: int, outpath: str):
    attns, labels_multi = [], []
    with torch.no_grad():
        for batch in loader:
            b = _move_lists(batch, device)
            out = teacher(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"],
                          return_attn=True)
            for i in range(b["pixel_values"].shape[0]):
                attns.append(out["ts_attn"][i].cpu().numpy())  # [K, T+1]
                labels_multi.append(b["y_multi"][i].cpu().numpy())
                if len(attns) >= n_samples:
                    break
            if len(attns) >= n_samples:
                break

    K = len(pathology_labels)
    label_short = [_short(n) for n in pathology_labels]
    ncols = 2
    nrows = (len(attns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 2.4))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for i, (a, y) in enumerate(zip(attns, labels_multi)):
        ax = axes[i]
        im = ax.imshow(a, aspect="auto", cmap="viridis")
        ax.set_yticks(range(K)); ax.set_yticklabels(label_short, fontsize=8)
        T_plus_1 = a.shape[1]
        # x축 마지막 위치는 [REP] 토큰 (DuETT 관례)
        xticks = list(range(0, T_plus_1 - 1, max(1, (T_plus_1 - 1) // 6))) + [T_plus_1 - 1]
        xlabels = [str(x) for x in xticks[:-1]] + ["REP"]
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel("time bin (h)", fontsize=8)
        y_str = "|".join(f"{int(v)}" for v in y)
        ax.set_title(f"sample {i}  y=({y_str})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    for j in range(len(attns), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-pathology TS attention (row = pathology, col = time bin + REP)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved: {outpath}")


# =============================================================================
# Fig 3: Learned query cosine similarity (K × K)
# =============================================================================
def _plot_sim_matrix(ax, sim, labels, title):
    K = len(labels)
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=9)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{sim[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(sim[i, j]) > 0.5 else "black", fontsize=8)
    ax.set_title(title, fontsize=11)
    return im


def viz_query_similarity(teacher, pathology_labels, outpath: str, mode: str = "single"):
    """single: 1개 query bank → 1 heatmap.
    dual_patch: image_queries / temporal_queries 각각 K×K + cross similarity → 3 heatmap.
    dual(residual): temporal_queries 1개만 존재 → 1 heatmap."""
    labels = [_short(n) for n in pathology_labels]

    if mode == "dual_patch":
        qi = teacher.perceiver.image_queries.detach().cpu()
        qt = teacher.perceiver.temporal_queries.detach().cpu()
        qi_n = qi / (qi.norm(dim=1, keepdim=True) + 1e-8)
        qt_n = qt / (qt.norm(dim=1, keepdim=True) + 1e-8)
        sim_i = (qi_n @ qi_n.T).numpy()
        sim_t = (qt_n @ qt_n.T).numpy()
        sim_x = (qi_n @ qt_n.T).numpy()   # cross: image queries vs temporal queries

        fig = plt.figure(figsize=(15.5, 4.4))
        gs = fig.add_gridspec(
            1, 4,
            width_ratios=[1.0, 1.0, 1.0, 0.05],
            wspace=0.45,
            left=0.05, right=0.94, top=0.90, bottom=0.18,
        )
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        cax = fig.add_subplot(gs[0, 3])
        _plot_sim_matrix(ax0, sim_i, labels, "Image queries (K×K)")
        _plot_sim_matrix(ax1, sim_t, labels, "Temporal queries (K×K)")
        im = _plot_sim_matrix(ax2, sim_x, labels, "Cross (image ↔ temporal)")
        fig.colorbar(im, cax=cax)
        fig.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] saved: {outpath}")
        return

    if mode == "dual":
        # residual DualPathologyPerceiver: temporal_queries 만 존재
        qt = teacher.perceiver.temporal_queries.detach().cpu()
        qt_n = qt / (qt.norm(dim=1, keepdim=True) + 1e-8)
        sim = (qt_n @ qt_n.T).numpy()
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = _plot_sim_matrix(ax, sim, labels, "Temporal query cosine similarity (dual/residual)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    else:  # single
        q = teacher.perceiver.queries.detach().cpu()
        q_n = q / (q.norm(dim=1, keepdim=True) + 1e-8)
        sim = (q_n @ q_n.T).numpy()
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = _plot_sim_matrix(ax, sim, labels, "Learned pathology query cosine similarity")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved: {outpath}")


# =============================================================================
# Fig 4: Stage-4 token dim-reduction (UMAP or t-SNE)
# =============================================================================
def viz_stage4_projection(teacher, loader, device, pathology_labels,
                          dim_reduce: str, outpath: str, mode: str = "single"):
    """single: stage4_tokens. dual_patch: fusion_tokens. dual(residual): ts_tokens."""
    if mode == "dual_patch":
        token_key = "fusion_tokens"
    elif mode == "dual":
        token_key = "ts_tokens"
    else:
        token_key = "stage4_tokens"
    all_tokens = []
    with torch.no_grad():
        for batch in loader:
            b = _move_lists(batch, device)
            out = teacher(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"],
                          return_attn=True)
            all_tokens.append(out[token_key].cpu())          # [B, K, d]
    tokens = torch.cat(all_tokens, dim=0).numpy()             # [N, K, d]
    N, K, d = tokens.shape
    color_ids = np.tile(np.arange(K), N)

    # raw 와 per-sample centered 두 벌 준비.
    # centered: 각 sample의 K 평균을 빼서 sample-level 성분 제거
    #           → 남은 신호가 pathology-specific 이어야 클러스터가 분리됨
    tokens_centered = tokens - tokens.mean(axis=1, keepdims=True)   # [N, K, d]
    flat_raw = tokens.reshape(N * K, d)
    flat_cen = tokens_centered.reshape(N * K, d)

    reducer_name = dim_reduce
    if reducer_name == "auto":
        try:
            import umap  # noqa: F401
            reducer_name = "umap"
        except ImportError:
            reducer_name = "tsne"
    print(f"[viz] projection method: {reducer_name}  (N×K = {flat_raw.shape[0]}, d={d})")

    def _fit(flat):
        if reducer_name == "umap":
            import umap
            r = umap.UMAP(n_components=2, random_state=42)
        else:
            from sklearn.manifold import TSNE
            perp = min(30, max(5, flat.shape[0] // 4 - 1))
            r = TSNE(n_components=2, random_state=42, init="pca", perplexity=perp)
        return r.fit_transform(flat)

    proj_raw = _fit(flat_raw)
    proj_cen = _fit(flat_cen)

    labels = [_short(n) for n in pathology_labels]
    cmap = plt.get_cmap("tab10")
    token_title = {"fusion_tokens": "Fusion tokens",
                   "ts_tokens":     "TS tokens",
                   "stage4_tokens": "Stage-4 tokens"}[token_key]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_i, (proj, tag) in enumerate([(proj_raw, "raw"),
                                         (proj_cen, "per-sample centered")]):
        ax = axes[ax_i]
        for k in range(K):
            mask = color_ids == k
            ax.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.5,
                       color=cmap(k % 10), label=labels[k])
        ax.legend(fontsize=9, loc="best")
        ax.set_title(f"{token_title} — {tag} ({reducer_name.upper()})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved: {outpath}")


# =============================================================================
# Fig 5: Gap summary (bar chart + CSV)
# =============================================================================
def viz_gap_summary(result: dict, pathology_labels,
                    outpath_png: str, outpath_csv: str, mode: str = "single"):
    """single: stage2 vs stage4 bar. dual / dual_patch: img vs ts vs fusion 3-way bar."""
    per = result["per_label"]
    labels = [_short(r["name"]) for r in per]
    K = len(per)
    x = np.arange(K)

    if mode in ("dual", "dual_patch"):
        img = [r["img_auroc"] for r in per]
        ts  = [r["ts_auroc"]  for r in per]
        fus = [r["fus_auroc"] for r in per]
        gap_i2f = [r["gap_i2f"] for r in per]
        gap_t2f = [r["gap_t2f"] for r in per]
        w = 0.25

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(x - w, img, w, label="image-only",  color="#6ca0dc")
        ax.bar(x,     ts,  w, label="TS-only",     color="#7fbf7b")
        ax.bar(x + w, fus, w, label="fusion",       color="#d9776b")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("AUROC")
        ax.set_title("Per-pathology img / ts / fusion AUROC (test)")
        # gap 값 라벨링 (fusion 위)
        for i in range(K):
            top = max(img[i], ts[i], fus[i]) + 0.015
            ax.text(i, top, f"i2f{gap_i2f[i]:+.3f}\nt2f{gap_t2f[i]:+.3f}",
                    ha="center", fontsize=7,
                    color="green" if gap_i2f[i] >= 0 and gap_t2f[i] >= 0 else "red")
        ax.legend(fontsize=9); ax.set_ylim(0, 1.05)
        plt.tight_layout()
        fig.savefig(outpath_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] saved: {outpath_png}")

        with open(outpath_csv, "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(["name", "n_valid", "pos_frac",
                            "img_auroc", "ts_auroc", "fus_auroc",
                            "gap_i2f", "gap_t2f",
                            "img_auprc", "ts_auprc", "fus_auprc"])
            for r in per:
                w_csv.writerow([r["name"], r["n_valid"], f"{r['pos_frac']:.4f}",
                                f"{r['img_auroc']:.4f}", f"{r['ts_auroc']:.4f}", f"{r['fus_auroc']:.4f}",
                                f"{r['gap_i2f']:+.4f}", f"{r['gap_t2f']:+.4f}",
                                f"{r['img_auprc']:.4f}", f"{r['ts_auprc']:.4f}", f"{r['fus_auprc']:.4f}"])
        print(f"[viz] saved: {outpath_csv}")
        return

    # single mode (기존)
    s2 = [r["stage2_auroc"] for r in per]
    s4 = [r["stage4_auroc"] for r in per]
    gap = [r["gap_auroc"] for r in per]
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, s2, w, label="stage2 (image-only)", color="#6ca0dc")
    ax.bar(x + w / 2, s4, w, label="stage4 (multimodal)", color="#d9776b")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-pathology stage2 vs stage4 AUROC (test)")
    for i, g in enumerate(gap):
        ax.text(i, max(s2[i], s4[i]) + 0.01, f"{g:+.3f}",
                ha="center", fontsize=9,
                color="green" if g >= 0 else "red")
    ax.legend(fontsize=9); ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(outpath_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved: {outpath_png}")

    with open(outpath_csv, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["name", "n_valid", "pos_frac",
                        "stage2_auroc", "stage4_auroc", "gap_auroc",
                        "stage2_auprc", "stage4_auprc", "gap_auprc"])
        for r in per:
            w_csv.writerow([r["name"], r["n_valid"], f"{r['pos_frac']:.4f}",
                            f"{r['stage2_auroc']:.4f}", f"{r['stage4_auroc']:.4f}",
                            f"{r['gap_auroc']:+.4f}",
                            f"{r['stage2_auprc']:.4f}", f"{r['stage4_auprc']:.4f}",
                            f"{r['gap_auprc']:+.4f}"])
    print(f"[viz] saved: {outpath_csv}")


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[viz] device={device}")

    teacher, bundle, processor, ck_args, pathology_labels, mode = load_teacher(args.ckpt, device)
    print(f"[viz] loaded ckpt: {args.ckpt}")
    print(f"[viz] perceiver mode: {mode}")
    print(f"[viz] pathology_labels: {pathology_labels}")
    print(f"[viz] main task (index 0): {ck_args.label_col}")

    split_ds = bundle["datasets"][args.split]
    full_loader = _loader(split_ds, args.batch_size, args.num_workers)

    n_attn = min(args.n_samples, len(split_ds))
    small_ds = Subset(split_ds, list(range(n_attn)))
    small_loader = _loader(small_ds, args.batch_size, args.num_workers)

    if mode == "dual_patch":
        proj_fname = "fusion_token_umap.png"
    elif mode == "dual":
        proj_fname = "ts_token_umap.png"
    else:
        proj_fname = "stage4_token_umap.png"
    csv_fname = "gap_summary_test.csv"
    gap_png   = "gap_summary_test.png"

    # Fig 1 (img_attn — 두 모드 공통, y=1 positive 만 사용하므로 full_loader 필요)
    viz_patch_attention(teacher, full_loader, device, processor, pathology_labels,
                        args.n_samples,
                        os.path.join(args.outdir, "patch_attention_overlay.png"))
    # Fig 2 (ts_attn — 두 모드 공통)
    viz_ts_attention(teacher, small_loader, device, pathology_labels,
                     args.n_samples,
                     os.path.join(args.outdir, "ts_attention_heatmap.png"))
    # Fig 3 (dual: 3 heatmap / single: 1 heatmap)
    viz_query_similarity(teacher, pathology_labels,
                         os.path.join(args.outdir, "query_similarity.png"),
                         mode=mode)
    # Fig 4 (dual: fusion_tokens / single: stage4_tokens)
    viz_stage4_projection(teacher, full_loader, device, pathology_labels, args.dim_reduce,
                          os.path.join(args.outdir, proj_fname),
                          mode=mode)
    # Fig 5 (full split eval — mode별 evaluator + bar chart)
    if mode in ("dual", "dual_patch"):
        result = evaluate_dual_pathology(teacher, full_loader, device, pathology_labels)
        print("\n" + format_dual_pathology_gap_table(result))
    else:
        result = evaluate_pathology(teacher, full_loader, device, pathology_labels)
        print("\n" + format_pathology_gap_table(result))
    viz_gap_summary(result, pathology_labels,
                    os.path.join(args.outdir, gap_png),
                    os.path.join(args.outdir, csv_fname),
                    mode=mode)

    print(f"\n[viz] all figures written to: {args.outdir}")


if __name__ == "__main__":
    main()
