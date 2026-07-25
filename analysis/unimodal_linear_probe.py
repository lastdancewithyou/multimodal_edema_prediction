"""
사용법:
(CXR)
python analysis/unimodal_linear_probe.py --modality cxr

(Time-series)
python analysis/unimodal_linear_probe.py --modality duett_rep && \
python analysis/unimodal_linear_probe.py --modality duett_hourly_mean && \
python analysis/unimodal_linear_probe.py --modality duett_multiscale && \
python analysis/unimodal_linear_probe.py --modality duett_attn_pool
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.main_architecture_duett import CXREncoder, load_duett_backbone
from training_duett.data_processing import AnchorConfig, build_datasets, duett_kd_collate

import warnings
warnings.filterwarnings("ignore", message=r".*rotary embedding dimension.*")


DEFAULTS = {
    "final_df_path": "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/subject_data/mimic_3_1/final_df_20260713",
    "static_path":   "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/mimic_3_1/static_full.ftr",
    "duett_ckpt":    "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/duett/checkpoints/mimic_ssl/n24_s12/pretrain-epoch=130-val_loss=0.2717.ckpt",
    "cxr_model_name": "microsoft/rad-dino",
}


# =============================================================================
# Feature extractors — forward once through frozen backbone
# =============================================================================
@torch.no_grad()
def _extract_cxr(loader, model, device):
    model.eval()
    feats, labels = [], []
    for batch in tqdm(loader, desc="[cxr]"):
        pixel = batch["pixel_values"].to(device, non_blocking=True)
        out = model(pixel)
        cls = out[0] if isinstance(out, tuple) else out       # [B, 768]
        feats.append(cls.detach().float().cpu())
        labels.append(batch["y"].float())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def _pool_duett_tokens(tokens: torch.Tensor, feature_type: str) -> torch.Tensor:
    """DuETT 출력 tokens [B, T+1, d_rep]에서 feature_type에 따라 pooling.

    tokens 순서: [hour_0, hour_1, ..., hour_{T-1}, REP]  (T=n_timesteps, 마지막이 REP).
    """
    if feature_type == "rep":
        return tokens[:, -1, :]                                    # [B, d]
    if feature_type == "hourly_mean":
        return tokens[:, :-1, :].mean(dim=1)                       # [B, d]
    if feature_type == "multiscale":
        hourly = tokens[:, :-1, :]                                 # [B, T, d]
        T = hourly.shape[1]
        # 24h 기준 3-window 분할; T!=24면 비례 분할
        b1, b2 = T // 4, T // 2                                    # T=24 → 6, 12
        w1 = hourly[:, :b1, :].mean(dim=1)                         # 0~6h (T=24 기준)
        w2 = hourly[:, b1:b2, :].mean(dim=1)                       # 6~12h
        w3 = hourly[:, b2:, :].mean(dim=1)                         # 12~24h
        rep = tokens[:, -1, :]
        return torch.cat([w1, w2, w3, rep], dim=-1)                # [B, 4d]
    if feature_type == "attn_pool":
        # 실제 pooling은 LinearHead에서 학습 가능한 attention으로 수행.
        # 여기선 hourly token만 반환 (REP 제외) → [B, T, d]
        return tokens[:, :-1, :]
    raise ValueError(f"unknown feature_type={feature_type!r}; "
                     "expected one of {'rep', 'hourly_mean', 'multiscale', 'attn_pool'}")


@torch.no_grad()
def _extract_duett(loader, backbone, device, feature_type: str):
    backbone.eval()
    feats, labels = [], []
    for batch in tqdm(loader, desc=f"[duett-{feature_type}]"):
        x = (
            tuple(t.to(device, non_blocking=True) for t in batch["x_ts"]),
            tuple(t.to(device, non_blocking=True) for t in batch["x_static"]),
            tuple(t.to(device, non_blocking=True) for t in batch["bin_ends"]),
        )
        y = batch["y"].float()
        duett_in = backbone.feats_to_input(x, y.shape[0])
        tokens = backbone.encode(duett_in)                    # [B, T+1, d_rep]
        feat = _pool_duett_tokens(tokens, feature_type)
        feats.append(feat.detach().float().cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


# =============================================================================
# Linear probe (PyTorch joint multi-label — matches cxr_linear_training.ipynb /
#                                            train_cxr_linear_head_icu.py)
# =============================================================================
class LinearHead(nn.Module):
    """CXR head 학습과 동일 구조 (Dropout → Linear).

    use_attn_pool=True 이면 입력이 [B, T, d]로 오고 학습 가능한 query 기반
    attention pooling으로 [B, d]로 축소한 뒤 head 적용.
    """
    def __init__(self, d_in: int, num_labels: int, dropout: float = 0.1,
                 use_attn_pool: bool = False):
        super().__init__()
        self.use_attn_pool = use_attn_pool
        if use_attn_pool:
            # query-based scalar attention: params ≈ d_in
            self.attn_query = nn.Parameter(torch.randn(d_in) * 0.02)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attn_pool:
            # x: [B, T, d]
            scores = (x @ self.attn_query) / (x.shape[-1] ** 0.5)   # [B, T]
            w = torch.softmax(scores, dim=1)                         # [B, T]
            x = (x * w.unsqueeze(-1)).sum(dim=1)                     # [B, d]
        return self.head(x)


def masked_bce(logits: torch.Tensor, labels: torch.Tensor,
               label_mask: torch.Tensor) -> torch.Tensor:
    """CXR head 학습과 동일 — NaN label은 mask로 배제, pos_weight 없음."""
    l = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    mf = label_mask.float()
    vc = mf.sum()
    if vc.item() == 0:
        return logits.sum() * 0.0
    return (l * mf).sum() / vc


@torch.no_grad()
def _eval_multi(model, X: torch.Tensor, Y: torch.Tensor, M: torch.Tensor,
                label_names: list[str], device) -> dict:
    """model(X) → per-label AUROC/AUPRC + macro. NaN(mask=0) 제외."""
    model.eval()
    logits = model(X.to(device)).cpu().numpy()
    y = Y.numpy(); m = M.numpy().astype(bool)
    per_label = {}
    aurocs, auprcs = [], []
    for i, name in enumerate(label_names):
        mk = m[:, i]
        yk = y[mk, i]
        pk = 1.0 / (1.0 + np.exp(-logits[mk, i]))
        if mk.sum() < 2 or len(np.unique(yk)) < 2:
            per_label[name] = {"auroc": float("nan"), "auprc": float("nan"),
                               "n": int(mk.sum()), "pos": int(yk.sum())}
            continue
        au = roc_auc_score(yk, pk); pr = average_precision_score(yk, pk)
        per_label[name] = {"auroc": au, "auprc": pr,
                           "n": int(mk.sum()), "pos": int(yk.sum())}
        aurocs.append(au); auprcs.append(pr)
    return {
        "per_label":   per_label,
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
    }


def train_linear_head(X_tr, Y_tr, M_tr, X_va, Y_va, M_va,
                       label_names: list[str], device,
                       epochs: int = 100, batch_size: int = 128,
                       lr: float = 1e-4, weight_decay: float = 1e-4,
                       dropout: float = 0.1, verbose: bool = True,
                       use_attn_pool: bool = False):
    """Joint multi-label linear head 학습 → best-val macro AUROC state 반환.

    X_*: [N, d_feat] (기본) 또는 [N, T, d_feat] (use_attn_pool=True)
    Y_*: [N, L] torch.Tensor (NaN → 0 채움, mask로 배제)
    M_*: [N, L] torch.Tensor (1=valid, 0=NaN)
    """
    d_feat = X_tr.shape[-1]   # 2D/3D 모두 마지막 축이 feature dim
    num_labels = len(label_names)
    model = LinearHead(d_feat, num_labels, dropout=dropout,
                       use_attn_pool=use_attn_pool).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(X_tr, Y_tr, M_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    best_val = -float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        run_l, run_v = 0.0, 0.0
        for xb, yb, mb in train_loader:
            xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = masked_bce(logits, yb, mb)
            loss.backward()
            optimizer.step()
            v = mb.float().sum().item()
            run_l += loss.item() * v; run_v += v

        val_metrics = _eval_multi(model, X_va, Y_va, M_va, label_names, device)
        if verbose:
            print(f"  epoch {epoch:>2d}  train_loss={run_l/max(run_v,1):.4f}  "
                  f"val_macro_AUROC={val_metrics['macro_auroc']:.4f}")

        if val_metrics["macro_auroc"] > best_val:
            best_val = val_metrics["macro_auroc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_epoch, best_val


# =============================================================================
# Loader helper
# =============================================================================
def _make_loader(ds, batch_size, num_workers, mode):
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=lambda b: duett_kd_collate(b, mode))


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="CXR-only / DuETT-only linear probe")
    p.add_argument("--modality",
                   choices=["cxr", "duett", "duett_rep", "duett_hourly_mean",
                            "duett_multiscale", "duett_attn_pool"],
                   required=True,
                   help="cxr: RAD-DINO CLS / duett_rep: DuETT [REP] token / "
                        "duett_hourly_mean: mean of 24 hourly tokens / "
                        "duett_multiscale: concat of 0-6h, 6-12h, 12-24h, REP means / "
                        "duett_attn_pool: 24 hourly tokens + query-based attention pooling "
                        "(head와 함께 학습). "
                        "'duett'은 duett_rep의 alias.")
    p.add_argument("--final_df_path", default=DEFAULTS["final_df_path"])
    p.add_argument("--static_path",   default=DEFAULTS["static_path"])
    p.add_argument("--duett_ckpt",    default=DEFAULTS["duett_ckpt"])
    p.add_argument("--cxr_model_name", default=DEFAULTS["cxr_model_name"])
    p.add_argument("--label_col",     default="label_edema",
                   help="anchor 필터링용 라벨 컬럼 (build_datasets 트리거용). 실제 평가 대상은 --labels로 지정")
    p.add_argument("--labels",        default="all",
                   help="평가할 label 컬럼들(쉼표 구분) 또는 'all'로 label_* 전부. label별로 NaN 필터 후 별도 LR")
    p.add_argument("--n_timesteps",   type=int, default=24)
    p.add_argument("--split_seed",    type=int, default=42)
    # Aligned split(pretrained head와 정합) 사용으로 val_size/test_size는 미사용.
    # 스플릿 비율은 pretrained head의 70/15/15가 자동 상속됨.
    # p.add_argument("--val_size",      type=float, default=0.10)
    # p.add_argument("--test_size",     type=float, default=0.10)
    p.add_argument("--batch_size",    type=int, default=64,
                   help="feature extraction용 (frozen backbone forward)")
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--save_features", type=str, default="", help="비우지 않으면 이 폴더에 X_*.npy / y_*.npy 저장 (재실험용)")

    # Linear head 학습 hyperparameters
    p.add_argument("--epochs",           type=int,   default=300,
                   help="linear head 학습 epoch 수")
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--train_batch_size", type=int,   default=128,
                   help="linear head 학습 batch size (feature-space 미니배치)")
    p.add_argument("--dropout",          type=float, default=0.1)

    args = p.parse_args()
    args.meta_path = os.path.join(os.path.dirname(args.duett_ckpt), "meta_with_stats.pkl")
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[modality] {args.modality.upper()}")

    # duett → duett_rep 별칭 흡수
    if args.modality == "duett":
        args.modality = "duett_rep"

    # CXR probe는 이미지 로딩 필요, DuETT probe는 필요 없음
    include_cxr = (args.modality == "cxr")
    is_duett = args.modality.startswith("duett_")
    duett_feature_type = args.modality[len("duett_"):] if is_duett else None
    processor = AutoImageProcessor.from_pretrained(args.cxr_model_name) if include_cxr else None

    cfg = AnchorConfig(
        final_df_path=args.final_df_path,
        static_path=args.static_path,
        meta_path=args.meta_path,
        label_col=args.label_col,
        n_timesteps=args.n_timesteps,
        split_seed=args.split_seed,
        # Aligned split 사용 — val_size/test_size는 pretrained head의 70/15/15 상속
    )
    bundle = build_datasets(cfg, image_processor=processor, include_cxr=include_cxr)
    ds = bundle["datasets"]
    meta = bundle["meta"]

    mode = "teacher" if include_cxr else "student"
    dl_tr = _make_loader(ds["train"], args.batch_size, args.num_workers, mode)
    dl_va = _make_loader(ds["val"],   args.batch_size, args.num_workers, mode)
    dl_te = _make_loader(ds["test"],  args.batch_size, args.num_workers, mode)

    # Feature extraction (frozen backbone forward)
    print(f"\n[extract] frozen backbone forward on train/val/test")
    if args.modality == "cxr":
        cxr = CXREncoder(model_name=args.cxr_model_name,
                          freeze=True, return_patches=False).to(device)
        X_tr, y_tr = _extract_cxr(dl_tr, cxr, device)
        X_va, y_va = _extract_cxr(dl_va, cxr, device)
        X_te, y_te = _extract_cxr(dl_te, cxr, device)
    else:
        backbone = load_duett_backbone(
            ckpt_path=args.duett_ckpt,
            d_static_num=int(meta["D_STATIC"]),
            d_time_series_num=len(bundle["ts_vars"]),
            n_timesteps=args.n_timesteps,
            freeze=True,
        ).to(device)
        X_tr, y_tr = _extract_duett(dl_tr, backbone, device, duett_feature_type)
        X_va, y_va = _extract_duett(dl_va, backbone, device, duett_feature_type)
        X_te, y_te = _extract_duett(dl_te, backbone, device, duett_feature_type)

    print(f"[shape] X_tr={X_tr.shape}  X_va={X_va.shape}  X_te={X_te.shape}")

    if args.save_features:
        os.makedirs(args.save_features, exist_ok=True)
        for name, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
            np.save(os.path.join(args.save_features, f"X_{args.modality}_{name}.npy"), X)
            np.save(os.path.join(args.save_features, f"y_{args.modality}_{name}.npy"), y)
        print(f"[save] features → {args.save_features}")

    # ─── Multi-label targets: anchor 순서 유지한 채 final_df에서 label 컬럼 lookup ───
    print(f"\n[multi-label] loading final_df to fetch label columns ...")
    full_df = pd.read_feather(args.final_df_path)
    if args.labels.strip().lower() == "all":
        label_cols = [c for c in full_df.columns if c.startswith("label_")]
    else:
        label_cols = [c.strip() for c in args.labels.split(",") if c.strip()]
        missing = [c for c in label_cols if c not in full_df.columns]
        if missing:
            raise ValueError(f"final_df에 없는 라벨 컬럼: {missing}")
    print(f"[multi-label] evaluating {len(label_cols)} labels: {label_cols}")

    key_cols = ["subject_id", "study_id", "dicom_id"]
    lookup = full_df[key_cols + label_cols].drop_duplicates(subset=key_cols)

    def _fetch_Y(split_name, expected_n):
        idx = bundle["splits"][split_name]
        sub = bundle["anchor_df"].iloc[idx].reset_index(drop=True)
        merged = sub.merge(lookup, on=key_cols, how="left")
        assert len(merged) == expected_n, \
            f"[{split_name}] merge {len(merged)} rows vs expected {expected_n} — key 중복/누락 확인 필요"
        return merged[label_cols].values.astype(float)   # [N, L]  NaN 유지

    Y_tr = _fetch_Y("train", len(X_tr))
    Y_va = _fetch_Y("val",   len(X_va))
    Y_te = _fetch_Y("test",  len(X_te))

    # ─── Joint multi-label linear head 학습 (CXR head training과 동일 프로토콜) ───
    # NaN → 0 placeholder + mask=0, valid label → 1
    def _to_tensor(Y):
        m = (~np.isnan(Y)).astype(np.float32)
        y = np.nan_to_num(Y, nan=0.0).astype(np.float32)
        return torch.from_numpy(y), torch.from_numpy(m)

    X_tr_t = torch.from_numpy(X_tr).float()
    X_va_t = torch.from_numpy(X_va).float()
    X_te_t = torch.from_numpy(X_te).float()
    Y_tr_t, M_tr_t = _to_tensor(Y_tr)
    Y_va_t, M_va_t = _to_tensor(Y_va)
    Y_te_t, M_te_t = _to_tensor(Y_te)

    use_attn_pool = (args.modality == "duett_attn_pool")
    d_feat = X_tr_t.shape[-1]
    pool_info = f", pool=attn (T={X_tr_t.shape[1]})" if use_attn_pool else ""
    print(f"\n[train] joint multi-label linear head "
          f"(d_feat={d_feat}, L={len(label_cols)}{pool_info}, "
          f"epochs={args.epochs}, lr={args.lr}, wd={args.weight_decay}, "
          f"batch={args.train_batch_size}, dropout={args.dropout}, no pos_weight)")
    model, best_epoch, best_val_macro = train_linear_head(
        X_tr_t, Y_tr_t, M_tr_t, X_va_t, Y_va_t, M_va_t,
        label_names=label_cols, device=device,
        epochs=args.epochs, batch_size=args.train_batch_size,
        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
        use_attn_pool=use_attn_pool,
    )
    print(f"[train] best epoch={best_epoch}  best val macro AUROC={best_val_macro:.4f}")

    # ─── Per-label 결과 리포트 (train / val / test) ─────────────────
    tr_res = _eval_multi(model, X_tr_t, Y_tr_t, M_tr_t, label_cols, device)
    va_res = _eval_multi(model, X_va_t, Y_va_t, M_va_t, label_cols, device)
    te_res = _eval_multi(model, X_te_t, Y_te_t, M_te_t, label_cols, device)

    print(f"\n[result] {args.modality.upper()}-only linear probe (joint multi-label)")
    hdr = (f"  {'label':<25} {'train AUROC':>11} {'val AUROC':>10} {'test AUROC':>11} "
           f"{'test AUPRC':>11} {'n_tr':>7} {'n_pos_tr':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name in label_cols:
        tr_s = tr_res["per_label"][name]
        va_s = va_res["per_label"][name]
        te_s = te_res["per_label"][name]
        print(f"  {name:<25} {tr_s['auroc']:>11.4f} {va_s['auroc']:>10.4f} "
              f"{te_s['auroc']:>11.4f} {te_s['auprc']:>11.4f} "
              f"{tr_s['n']:>7d} {tr_s['pos']:>10d}")

    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'macro average':<25} {tr_res['macro_auroc']:>11.4f} "
          f"{va_res['macro_auroc']:>10.4f} {te_res['macro_auroc']:>11.4f} "
          f"{te_res['macro_auprc']:>11.4f}")


if __name__ == "__main__":
    main()
