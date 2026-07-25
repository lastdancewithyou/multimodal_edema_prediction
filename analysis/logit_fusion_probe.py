"""Logit-level fusion probe.

Frozen unimodal backbones (CXR = RAD-DINO, TS = DuETT)에서 각각 linear probe를 학습한 뒤,
그 unimodal logits를 단순 concat → 소형 fusion head 학습 → image-only 대비 개선 여부 확인.

두 backbone 모두 frozen. 학습되는 건 각 unimodal linear head + fusion head 뿐.
Fusion에 넣는 신호는 오직 unimodal logits(스칼라 · 병리 개수만큼)뿐이므로,
성능 향상이 있다면 그건 "logit calibration + cross-modal linear/nonlinear 결합" 만의 기여.

사용법:
    python analysis/logit_fusion_probe.py --ts_modality duett_multiscale --fusion_type linear

--fusion_type:
    per_label : pathology별 독립 2→1 결합 (img 가중치=1로 init → image-only에서 출발)
    linear    : [2L → L] 단일 linear layer
    mlp       : [2L → hidden → L] 소형 MLP

Options:
    --features_dir <dir>  : unimodal_linear_probe.py --save_features로 미리 저장한 feature 사용.
                            없거나 파일 없으면 backbone forward 재실행.
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
from transformers import AutoImageProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.main_architecture_duett import CXREncoder, load_duett_backbone
from training_duett.data_processing import AnchorConfig, build_datasets, duett_kd_collate

from analysis.unimodal_linear_probe import (
    DEFAULTS,
    LinearHead,
    _extract_cxr,
    _extract_duett,
    _eval_multi,
    _make_loader,
    masked_bce,
    train_linear_head,
)

import warnings
warnings.filterwarnings("ignore", message=r".*rotary embedding dimension.*")


# =============================================================================
# Fusion head
# =============================================================================
class LogitFusionHead(nn.Module):
    """Unimodal logits → fused logits.

    Input:  img_logits, ts_logits  each ∈ ℝ[B, L]
    Output: fused logits ∈ ℝ[B, L]
    """
    def __init__(self, n_labels: int, fusion_type: str = "linear",
                 hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.fusion_type = fusion_type
        self.n_labels = n_labels
        d_in = 2 * n_labels

        if fusion_type == "linear":
            self.head = nn.Linear(d_in, n_labels)
        elif fusion_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, n_labels),
            )
        elif fusion_type == "per_label":
            # per-label 독립 2→1: fused[k] = w_img[k]·img[k] + w_ts[k]·ts[k] + b[k]
            # img=1, ts=0으로 init → 학습 시작점이 image-only와 동일.
            # 성능이 image-only 대비 개선/저하되는지 gradient signal만으로 학습.
            self.per_label_w = nn.Parameter(torch.zeros(n_labels, 2))
            self.per_label_b = nn.Parameter(torch.zeros(n_labels))
            with torch.no_grad():
                self.per_label_w[:, 0] = 1.0   # img weight
        else:
            raise ValueError(f"unknown fusion_type={fusion_type!r}")

    def forward(self, logits_img: torch.Tensor, logits_ts: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "per_label":
            pair = torch.stack([logits_img, logits_ts], dim=-1)        # [B, K, 2]
            return (pair * self.per_label_w).sum(dim=-1) + self.per_label_b
        x = torch.cat([logits_img, logits_ts], dim=-1)                 # [B, 2K]
        return self.head(x)


def train_fusion_head(img_tr, ts_tr, Y_tr, M_tr,
                       img_va, ts_va, Y_va, M_va,
                       label_names, device,
                       fusion_type: str = "linear",
                       hidden: int = 32, dropout: float = 0.1,
                       epochs: int = 300, batch_size: int = 128,
                       lr: float = 1e-3, weight_decay: float = 1e-4,
                       verbose: bool = True):
    """Concat logits 위에 fusion head 학습 (best val macro AUROC 기준 early-select)."""
    L = len(label_names)
    model = LogitFusionHead(n_labels=L, fusion_type=fusion_type,
                             hidden=hidden, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(img_tr, ts_tr, Y_tr, M_tr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    img_va_d, ts_va_d = img_va.to(device), ts_va.to(device)
    Y_va_d, M_va_d = Y_va.to(device), M_va.to(device)

    best_val, best_state, best_epoch = -float("inf"), None, -1
    for ep in range(1, epochs + 1):
        model.train()
        run_l, run_v = 0.0, 0.0
        for xb, tb, yb, mb in dl:
            xb, tb = xb.to(device), tb.to(device)
            yb, mb = yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, tb)
            loss = masked_bce(logits, yb, mb)
            loss.backward()
            optimizer.step()
            v = mb.float().sum().item()
            run_l += loss.item() * v; run_v += v

        # val eval — 별도 헬퍼로 로짓 계산해서 metrics 산출
        model.eval()
        with torch.no_grad():
            va_logits = model(img_va_d, ts_va_d).cpu().numpy()
        va_res = _eval_from_logits(va_logits, Y_va_d.cpu().numpy(),
                                    M_va_d.cpu().numpy(), label_names)
        if verbose:
            print(f"  ep {ep:>2d}  train_loss={run_l/max(run_v,1):.4f}  "
                  f"val_macro_AUROC={va_res['macro_auroc']:.4f}")
        if va_res["macro_auroc"] > best_val:
            best_val, best_epoch = va_res["macro_auroc"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_epoch, best_val


# =============================================================================
# Eval helpers
# =============================================================================
def _eval_from_logits(logits: np.ndarray, Y: np.ndarray, M: np.ndarray,
                       label_names: list[str]) -> dict:
    """logits/Y/M: [N, L] numpy. NaN(mask=0) 제외한 per-label AUROC/AUPRC."""
    m = M.astype(bool)
    per_label = {}
    aurocs, auprcs = [], []
    for i, name in enumerate(label_names):
        mk = m[:, i]
        yk = Y[mk, i]
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


@torch.no_grad()
def _head_logits(model, X: torch.Tensor, device, batch_size: int = 512) -> np.ndarray:
    """LinearHead를 미니배치로 forward해 [N, L] logits 반환."""
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch_size):
        chunk = X[i:i+batch_size].to(device)
        outs.append(model(chunk).cpu())
    return torch.cat(outs, dim=0).numpy()


# =============================================================================
# Feature loading — cache aware
# =============================================================================
def _load_or_extract_features(args, device):
    """CXR / DuETT feature를 dict로 반환. features_dir가 있으면 캐시 로드, 없으면 재추출.

    Returns: {
        "img_tr", "img_va", "img_te",  # np.ndarray [N, d_img]
        "ts_tr",  "ts_va",  "ts_te",   # np.ndarray [N, d_ts] (or [N, T, d] for attn_pool)
        "y_tr",   "y_va",   "y_te",    # single-label anchor targets (not used for multi-label)
        "bundle",                        # build_datasets bundle (final_df에서 multi-label 재조회용)
    }
    """
    fd = args.features_dir
    img_cache = fd and all(os.path.exists(os.path.join(fd, f"X_cxr_{sp}.npy")) for sp in ["train","val","test"])
    ts_cache  = fd and all(os.path.exists(os.path.join(fd, f"X_{args.ts_modality}_{sp}.npy")) for sp in ["train","val","test"])

    # bundle은 항상 필요 (multi-label 재조회 + split 인덱스 참조)
    processor = AutoImageProcessor.from_pretrained(args.cxr_model_name) if not img_cache else None
    cfg = AnchorConfig(
        final_df_path=args.final_df_path,
        static_path=args.static_path,
        meta_path=args.meta_path,
        label_col=args.label_col,
        n_timesteps=args.n_timesteps,
        split_seed=args.split_seed,
    )
    bundle = build_datasets(cfg, image_processor=processor,
                              include_cxr=(not img_cache))
    ds = bundle["datasets"]
    meta = bundle["meta"]

    out = {"bundle": bundle}

    # ── CXR ──
    if img_cache:
        print(f"[cache] loading CXR features from {fd}")
        out["img_tr"] = np.load(os.path.join(fd, "X_cxr_train.npy"))
        out["img_va"] = np.load(os.path.join(fd, "X_cxr_val.npy"))
        out["img_te"] = np.load(os.path.join(fd, "X_cxr_test.npy"))
    else:
        print(f"[extract] CXR (RAD-DINO CLS) — frozen forward")
        mode = "teacher"
        dl_tr = _make_loader(ds["train"], args.batch_size, args.num_workers, mode)
        dl_va = _make_loader(ds["val"],   args.batch_size, args.num_workers, mode)
        dl_te = _make_loader(ds["test"],  args.batch_size, args.num_workers, mode)
        cxr = CXREncoder(model_name=args.cxr_model_name,
                          freeze=True, return_patches=False).to(device)
        out["img_tr"], _ = _extract_cxr(dl_tr, cxr, device)
        out["img_va"], _ = _extract_cxr(dl_va, cxr, device)
        out["img_te"], _ = _extract_cxr(dl_te, cxr, device)
        del cxr; torch.cuda.empty_cache()

    # ── DuETT ──
    ts_feature_type = args.ts_modality[len("duett_"):] if args.ts_modality.startswith("duett_") else args.ts_modality
    if ts_cache:
        print(f"[cache] loading TS features ({args.ts_modality}) from {fd}")
        out["ts_tr"] = np.load(os.path.join(fd, f"X_{args.ts_modality}_train.npy"))
        out["ts_va"] = np.load(os.path.join(fd, f"X_{args.ts_modality}_val.npy"))
        out["ts_te"] = np.load(os.path.join(fd, f"X_{args.ts_modality}_test.npy"))
    else:
        print(f"[extract] DuETT ({args.ts_modality}) — frozen forward")
        mode = "student"     # pixel_values 불필요
        # ts extract는 pixel 없이 돌리기 위해 include_cxr=False로 build했으면 이상적이지만,
        # cache 없을 때 include_cxr는 이미 True로 잡혀있으므로 그냥 같은 dataset 사용.
        dl_tr = _make_loader(ds["train"], args.batch_size, args.num_workers, mode)
        dl_va = _make_loader(ds["val"],   args.batch_size, args.num_workers, mode)
        dl_te = _make_loader(ds["test"],  args.batch_size, args.num_workers, mode)
        backbone = load_duett_backbone(
            ckpt_path=args.duett_ckpt,
            d_static_num=int(meta["D_STATIC"]),
            d_time_series_num=len(bundle["ts_vars"]),
            n_timesteps=args.n_timesteps,
            freeze=True,
        ).to(device)
        out["ts_tr"], _ = _extract_duett(dl_tr, backbone, device, ts_feature_type)
        out["ts_va"], _ = _extract_duett(dl_va, backbone, device, ts_feature_type)
        out["ts_te"], _ = _extract_duett(dl_te, backbone, device, ts_feature_type)
        del backbone; torch.cuda.empty_cache()

    return out


def _fetch_multilabel_Y(bundle, split_name: str, expected_n: int,
                        label_cols: list[str], final_df: pd.DataFrame) -> np.ndarray:
    """bundle의 split 순서에 맞춰 final_df에서 label 컬럼들 lookup. NaN 유지."""
    key_cols = ["subject_id", "study_id", "dicom_id"]
    lookup = final_df[key_cols + label_cols].drop_duplicates(subset=key_cols)
    idx = bundle["splits"][split_name]
    sub = bundle["anchor_df"].iloc[idx].reset_index(drop=True)
    merged = sub.merge(lookup, on=key_cols, how="left")
    assert len(merged) == expected_n, \
        f"[{split_name}] merge {len(merged)} vs expected {expected_n}"
    return merged[label_cols].values.astype(float)


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Logit-level fusion probe (frozen unimodal backbones)")
    # Data paths
    p.add_argument("--final_df_path", default=DEFAULTS["final_df_path"])
    p.add_argument("--static_path",   default=DEFAULTS["static_path"])
    p.add_argument("--duett_ckpt",    default=DEFAULTS["duett_ckpt"])
    p.add_argument("--cxr_model_name", default=DEFAULTS["cxr_model_name"])
    p.add_argument("--label_col",     default="label_edema")
    p.add_argument("--labels",        default="all",
                   help="평가할 label 컬럼들 (쉼표 구분) 또는 'all'")
    p.add_argument("--n_timesteps",   type=int, default=24)
    p.add_argument("--split_seed",    type=int, default=42)
    # Extract / loader
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--features_dir",  default="",
                   help="unimodal_linear_probe.py --save_features 폴더. 있으면 재사용, 없으면 재추출.")
    # Modality choice
    p.add_argument("--ts_modality", default="duett_multiscale",
                   choices=["duett_rep", "duett_hourly_mean", "duett_multiscale", "duett_attn_pool"],
                   help="TS unimodal probe feature 방식")
    # Unimodal head 학습
    p.add_argument("--uni_epochs",       type=int,   default=300)
    p.add_argument("--uni_lr",           type=float, default=1e-4)
    p.add_argument("--uni_weight_decay", type=float, default=1e-4)
    p.add_argument("--uni_batch_size",   type=int,   default=128)
    p.add_argument("--uni_dropout",      type=float, default=0.1)
    # Fusion head
    p.add_argument("--fusion_type", default="linear",
                   choices=["per_label", "linear", "mlp"])
    p.add_argument("--fusion_hidden",  type=int,   default=32)
    p.add_argument("--fusion_dropout", type=float, default=0.1)
    p.add_argument("--fus_epochs",       type=int,   default=300)
    p.add_argument("--fus_lr",           type=float, default=1e-3)
    p.add_argument("--fus_weight_decay", type=float, default=1e-4)
    p.add_argument("--fus_batch_size",   type=int,   default=128)
    # 진행 로그 억제
    p.add_argument("--quiet", action="store_true",
                   help="epoch별 loss 로그 억제 (요약만 출력)")

    args = p.parse_args()
    args.meta_path = os.path.join(os.path.dirname(args.duett_ckpt), "meta_with_stats.pkl")
    return args


def _to_tensor(Y: np.ndarray):
    m = (~np.isnan(Y)).astype(np.float32)
    y = np.nan_to_num(Y, nan=0.0).astype(np.float32)
    return torch.from_numpy(y), torch.from_numpy(m)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[config] ts_modality={args.ts_modality}  fusion_type={args.fusion_type}")

    # ── Features + bundle ──
    feats = _load_or_extract_features(args, device)
    bundle = feats["bundle"]

    # ── Multi-label targets ──
    full_df = pd.read_feather(args.final_df_path)
    if args.labels.strip().lower() == "all":
        label_cols = [c for c in full_df.columns if c.startswith("label_")]
    else:
        label_cols = [c.strip() for c in args.labels.split(",") if c.strip()]
        missing = [c for c in label_cols if c not in full_df.columns]
        if missing:
            raise ValueError(f"final_df에 없는 라벨 컬럼: {missing}")
    print(f"[multi-label] L={len(label_cols)} labels: {label_cols}")

    Y_tr_np = _fetch_multilabel_Y(bundle, "train", len(feats["img_tr"]), label_cols, full_df)
    Y_va_np = _fetch_multilabel_Y(bundle, "val",   len(feats["img_va"]), label_cols, full_df)
    Y_te_np = _fetch_multilabel_Y(bundle, "test",  len(feats["img_te"]), label_cols, full_df)

    Y_tr, M_tr = _to_tensor(Y_tr_np)
    Y_va, M_va = _to_tensor(Y_va_np)
    Y_te, M_te = _to_tensor(Y_te_np)

    X_img_tr = torch.from_numpy(feats["img_tr"]).float()
    X_img_va = torch.from_numpy(feats["img_va"]).float()
    X_img_te = torch.from_numpy(feats["img_te"]).float()
    X_ts_tr  = torch.from_numpy(feats["ts_tr"]).float()
    X_ts_va  = torch.from_numpy(feats["ts_va"]).float()
    X_ts_te  = torch.from_numpy(feats["ts_te"]).float()

    print(f"[shape] img feat d={X_img_tr.shape[-1]}   ts feat d={X_ts_tr.shape[-1]}"
          + (f" (T={X_ts_tr.shape[1]})" if X_ts_tr.ndim == 3 else ""))

    verbose = not args.quiet

    # ── Stage 1: CXR linear probe ──
    print("\n[stage 1] CXR linear probe")
    cxr_head, ep_c, val_c = train_linear_head(
        X_img_tr, Y_tr, M_tr, X_img_va, Y_va, M_va,
        label_names=label_cols, device=device,
        epochs=args.uni_epochs, batch_size=args.uni_batch_size,
        lr=args.uni_lr, weight_decay=args.uni_weight_decay,
        dropout=args.uni_dropout, verbose=verbose, use_attn_pool=False,
    )
    print(f"[stage 1] best epoch={ep_c}  val macro AUROC={val_c:.4f}")

    img_logits_tr = _head_logits(cxr_head, X_img_tr, device)
    img_logits_va = _head_logits(cxr_head, X_img_va, device)
    img_logits_te = _head_logits(cxr_head, X_img_te, device)
    img_res_te = _eval_from_logits(img_logits_te, Y_te.numpy(), M_te.numpy(), label_cols)

    # ── Stage 2: DuETT linear probe ──
    print(f"\n[stage 2] DuETT linear probe ({args.ts_modality})")
    use_attn_pool = (args.ts_modality == "duett_attn_pool")
    ts_head, ep_t, val_t = train_linear_head(
        X_ts_tr, Y_tr, M_tr, X_ts_va, Y_va, M_va,
        label_names=label_cols, device=device,
        epochs=args.uni_epochs, batch_size=args.uni_batch_size,
        lr=args.uni_lr, weight_decay=args.uni_weight_decay,
        dropout=args.uni_dropout, verbose=verbose, use_attn_pool=use_attn_pool,
    )
    print(f"[stage 2] best epoch={ep_t}  val macro AUROC={val_t:.4f}")

    ts_logits_tr = _head_logits(ts_head, X_ts_tr, device)
    ts_logits_va = _head_logits(ts_head, X_ts_va, device)
    ts_logits_te = _head_logits(ts_head, X_ts_te, device)
    ts_res_te = _eval_from_logits(ts_logits_te, Y_te.numpy(), M_te.numpy(), label_cols)

    # ── Stage 3: fusion head on concat logits ──
    print(f"\n[stage 3] fusion head ({args.fusion_type}) on concat logits")
    img_tr_t = torch.from_numpy(img_logits_tr).float()
    img_va_t = torch.from_numpy(img_logits_va).float()
    img_te_t = torch.from_numpy(img_logits_te).float()
    ts_tr_t  = torch.from_numpy(ts_logits_tr).float()
    ts_va_t  = torch.from_numpy(ts_logits_va).float()
    ts_te_t  = torch.from_numpy(ts_logits_te).float()

    fusion, ep_f, val_f = train_fusion_head(
        img_tr_t, ts_tr_t, Y_tr, M_tr,
        img_va_t, ts_va_t, Y_va, M_va,
        label_names=label_cols, device=device,
        fusion_type=args.fusion_type, hidden=args.fusion_hidden, dropout=args.fusion_dropout,
        epochs=args.fus_epochs, batch_size=args.fus_batch_size,
        lr=args.fus_lr, weight_decay=args.fus_weight_decay, verbose=verbose,
    )
    print(f"[stage 3] best epoch={ep_f}  val macro AUROC={val_f:.4f}")

    with torch.no_grad():
        fus_logits_te = fusion(img_te_t.to(device), ts_te_t.to(device)).cpu().numpy()
    fus_res_te = _eval_from_logits(fus_logits_te, Y_te.numpy(), M_te.numpy(), label_cols)

    # ── Report ──
    print(f"\n[result] logit-fusion probe   (fusion_type={args.fusion_type})")
    hdr = (f"  {'label':<22} {'n':>6} {'pos':>6}   "
           f"{'img_roc':>8} {'ts_roc':>8} {'fus_roc':>8}   "
           f"{'img_prc':>8} {'ts_prc':>8} {'fus_prc':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in label_cols:
        i = img_res_te["per_label"][name]
        t = ts_res_te["per_label"][name]
        f = fus_res_te["per_label"][name]
        print(f"  {name:<22} {i['n']:>6d} {i['pos']:>6d}   "
              f"{i['auroc']:>8.4f} {t['auroc']:>8.4f} {f['auroc']:>8.4f}   "
              f"{i['auprc']:>8.4f} {t['auprc']:>8.4f} {f['auprc']:>8.4f}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'macro':<22} {'':>6} {'':>6}   "
          f"{img_res_te['macro_auroc']:>8.4f} {ts_res_te['macro_auroc']:>8.4f} "
          f"{fus_res_te['macro_auroc']:>8.4f}   "
          f"{img_res_te['macro_auprc']:>8.4f} {ts_res_te['macro_auprc']:>8.4f} "
          f"{fus_res_te['macro_auprc']:>8.4f}")

    # per_label 모드는 학습된 결합 계수도 같이 출력 (해석용)
    if args.fusion_type == "per_label":
        print(f"\n[per_label weights]  (init: w_img=1, w_ts=0, b=0)")
        w = fusion.per_label_w.detach().cpu().numpy()
        b = fusion.per_label_b.detach().cpu().numpy()
        print(f"  {'label':<22} {'w_img':>8} {'w_ts':>8} {'bias':>8}")
        for i, name in enumerate(label_cols):
            print(f"  {name:<22} {w[i,0]:>8.4f} {w[i,1]:>8.4f} {b[i]:>8.4f}")


if __name__ == "__main__":
    main()
