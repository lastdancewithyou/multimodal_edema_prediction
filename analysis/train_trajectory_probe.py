"""Standalone TS-only probe for the variable-first trajectory hypothesis.

This script intentionally excludes CXR, fusion, correction, and distillation.
It answers one bounded question: can a variable-first temporal encoder extract
more CXR-label signal from the same 24 h input than the current TS baseline?

Example
-------
python analysis/train_trajectory_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.main_architecture_duett import LocalTrajectoryEncoder  # noqa: E402
from training_duett.data_processing import (  # noqa: E402
    AnchorConfig,
    DEFAULT_PATHOLOGY_LABELS,
    build_datasets,
)
from training_duett.run import REPO_DEFAULTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Standalone variable-first trajectory probe")
    p.add_argument("--final_df_path", default=REPO_DEFAULTS["final_df_path"])
    p.add_argument("--static_path", default=REPO_DEFAULTS["static_path"])
    p.add_argument("--duett_ckpt", default=REPO_DEFAULTS["duett_ckpt"],
                   help="Used only to locate meta_with_stats.pkl")
    p.add_argument("--meta_path", default="",
                   help="default: <duett_ckpt directory>/meta_with_stats.pkl")
    p.add_argument("--label_col", default="label_edema")
    p.add_argument("--n_timesteps", type=int, default=24)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--trajectory_windows", default="6,12,24")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true",
                   help="Use CUDA bfloat16 autocast when supported")
    p.add_argument("--limit_train_batches", type=int, default=0,
                   help="Development only; 0 uses the complete training split")

    p.add_argument(
        "--reference_aurocs", default="0.641,0.634,0.609,0.604",
        help="Existing shared-query DuETT TS AUROCs, in pathology label order. "
             "Use an empty string to suppress delta reporting.",
    )
    p.add_argument("--output_root", default="analysis_outputs/trajectory_probe")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_ts(batch: list[dict]) -> dict:
    return {
        "x_ts": tuple(item["x_ts"] for item in batch),
        "y": torch.stack([item["y_multi"] for item in batch]),
        "mask": torch.stack([item["y_multi_mask"] for item in batch]),
    }


class TrajectoryPathologyProbe(nn.Module):
    """Variable-first trajectory encoder plus a small pathology-query readout."""

    def __init__(
        self,
        n_vars: int,
        n_pathologies: int,
        n_timesteps: int,
        d_model: int,
        gru_layers: int,
        n_heads: int,
        dropout: float,
        recency_windows: tuple[int, ...],
    ):
        super().__init__()
        self.encoder = LocalTrajectoryEncoder(
            n_vars=n_vars,
            n_timesteps=n_timesteps,
            d_model=d_model,
            n_layers=gru_layers,
            dropout=dropout,
            recency_windows=recency_windows,
        )
        self.pathology_queries = nn.Parameter(
            torch.randn(n_pathologies, d_model) * 0.02
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.label_bias = nn.Parameter(torch.zeros(n_pathologies))

    def forward(self, x_ts_list, return_attn: bool = False):
        tokens, padding_mask = self.encoder(
            x_ts_list, return_padding_mask=True
        )
        # The final REP token is not used by this probe.  Every remaining token
        # maps to one explicit (variable, recency-window) pair.
        tokens = tokens[:, :-1]
        padding_mask = padding_mask[:, :-1]
        B = tokens.size(0)
        q = self.pathology_queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, attn = self.cross_attn(
            self.norm_q(q),
            self.norm_kv(tokens),
            self.norm_kv(tokens),
            key_padding_mask=padding_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        logits = self.head(q).squeeze(-1) + self.label_bias.unsqueeze(0)
        return (logits, attn) if return_attn else logits


def masked_bce(logits: torch.Tensor, y: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def make_loader(dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        drop_last=shuffle,
        collate_fn=collate_ts,
    )


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "x_ts": tuple(x.to(device, non_blocking=True) for x in batch["x_ts"]),
        "y": batch["y"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
    }


def train_epoch(model, loader, optimizer, device, args, amp_enabled: bool) -> float:
    model.train()
    total_loss, total_batches = 0.0, 0
    for step, raw_batch in enumerate(loader):
        if args.limit_train_batches > 0 and step >= args.limit_train_batches:
            break
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            logits = model(batch["x_ts"])
            loss = masked_bce(logits, batch["y"], batch["mask"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += float(loss.detach())
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.inference_mode()
def evaluate(model, loader, device, labels: tuple[str, ...]) -> dict:
    model.eval()
    logits_all, y_all, mask_all = [], [], []
    losses = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        logits = model(batch["x_ts"])
        losses.append(float(masked_bce(logits, batch["y"], batch["mask"])))
        logits_all.append(logits.float().cpu())
        y_all.append(batch["y"].cpu())
        mask_all.append(batch["mask"].cpu())

    logits = torch.cat(logits_all).numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    y = torch.cat(y_all).numpy()
    mask = torch.cat(mask_all).numpy().astype(bool)

    per_label = []
    for k, label in enumerate(labels):
        valid = mask[:, k]
        yk, pk = y[valid, k], probs[valid, k]
        roc = float("nan")
        prc = float("nan")
        if valid.sum() and np.unique(yk).size == 2:
            roc = float(roc_auc_score(yk, pk))
            prc = float(average_precision_score(yk, pk))
        per_label.append({
            "label": label,
            "n": int(valid.sum()),
            "pos": int(yk.sum()),
            "auroc": roc,
            "auprc": prc,
        })
    rocs = [row["auroc"] for row in per_label if np.isfinite(row["auroc"])]
    prcs = [row["auprc"] for row in per_label if np.isfinite(row["auprc"])]
    return {
        "loss": float(np.mean(losses)),
        "macro_auroc": float(np.mean(rocs)) if rocs else float("nan"),
        "macro_auprc": float(np.mean(prcs)) if prcs else float("nan"),
        "per_label": per_label,
    }


def print_metrics(title: str, metrics: dict,
                  reference: list[float] | None = None) -> None:
    print(f"\n=== {title} ===")
    print(f"BCE={metrics['loss']:.5f}  macro_AUROC={metrics['macro_auroc']:.4f}  "
          f"macro_AUPRC={metrics['macro_auprc']:.4f}")
    print("label                     n   pos   AUROC   AUPRC    dROC")
    print("-" * 65)
    for k, row in enumerate(metrics["per_label"]):
        delta = "--"
        if reference is not None and k < len(reference) and np.isfinite(row["auroc"]):
            delta = f"{row['auroc'] - reference[k]:+.4f}"
        print(f"{row['label']:<24s} {row['n']:>5d} {row['pos']:>5d}  "
              f"{row['auroc']:.4f}  {row['auprc']:.4f}  {delta:>7s}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda" and torch.cuda.is_bf16_supported())
    labels = tuple(DEFAULT_PATHOLOGY_LABELS)
    if labels[0] != args.label_col:
        raise ValueError(f"labels[0]={labels[0]!r} != label_col={args.label_col!r}")

    windows = tuple(int(x.strip()) for x in args.trajectory_windows.split(",") if x.strip())
    if args.d_model % args.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    meta_path = args.meta_path or os.path.join(
        os.path.dirname(args.duett_ckpt), "meta_with_stats.pkl"
    )
    cfg = AnchorConfig(
        final_df_path=args.final_df_path,
        static_path=args.static_path,
        meta_path=meta_path,
        label_col=args.label_col,
        n_timesteps=args.n_timesteps,
        split_seed=args.split_seed,
        pathology_labels=labels,
    )
    print(f"[probe] device={device} amp_bf16={amp_enabled} seed={args.seed}")
    print("[probe] CXR/fusion/correction are disabled by design")
    bundle = build_datasets(cfg, image_processor=None, include_cxr=False)
    loaders = {
        split: make_loader(bundle["datasets"][split], args, shuffle=(split == "train"))
        for split in ("train", "val", "test")
    }

    model = TrajectoryPathologyProbe(
        n_vars=len(bundle["ts_vars"]),
        n_pathologies=len(labels),
        n_timesteps=args.n_timesteps,
        d_model=args.d_model,
        gru_layers=args.gru_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        recency_windows=windows,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[probe] labels={labels}")
    print(f"[probe] V={len(bundle['ts_vars'])} windows={windows} "
          f"tokens={len(bundle['ts_vars']) * len(windows)} params={n_params:,}")

    run_dir = os.path.join(
        args.output_root,
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}",
    )
    os.makedirs(run_dir, exist_ok=False)
    best_path = os.path.join(run_dir, "best.pt")
    best_roc, stale = -float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, device, args, amp_enabled)
        val = evaluate(model, loaders["val"], device, labels)
        scheduler.step()
        print(f"epoch={epoch:02d} train_BCE={train_loss:.5f} "
              f"val_BCE={val['loss']:.5f} val_macroROC={val['macro_auroc']:.4f} "
              f"val_macroPRC={val['macro_auprc']:.4f}")

        if val["macro_auroc"] > best_roc + 1e-6:
            best_roc = val["macro_auroc"]
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val": val,
                "args": vars(args),
                "labels": labels,
                "ts_vars": list(bundle["ts_vars"]),
            }, best_path)
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"[probe] early stop at epoch={epoch}; best val macro AUROC={best_roc:.4f}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    reference = None
    if args.reference_aurocs.strip():
        reference = [float(x) for x in args.reference_aurocs.split(",")]
        if len(reference) != len(labels):
            raise ValueError("reference_aurocs length must equal pathology label count")
    test_metrics = evaluate(model, loaders["test"], device, labels)
    print_metrics(
        f"TEST — best validation epoch {checkpoint['epoch']}",
        test_metrics,
        reference,
    )
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2, allow_nan=True)
    print(f"\n[probe] saved {best_path}")
    print(f"[probe] saved {os.path.join(run_dir, 'test_metrics.json')}")


if __name__ == "__main__":
    main()
