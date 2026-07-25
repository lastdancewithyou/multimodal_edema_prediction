"""ICU-adapted CXR-only linear head (4-label) trained on DuETT train split.

목적:
  Tier 2 CXR-only baseline. Frozen RAD-DINO 위에 4-label linear head를
  DuETT train subject의 CXR만으로 학습. 완료 후 DuETT val/test에서 평가한
  숫자가 논문 표의 CXR-only baseline이 됨.
  학습된 ckpt는 DuETT teacher의 `--pretrained_cxr_head_ckpt`에 그대로
  plug-in 가능 (num_classes=4, label_cols == PATHOLOGY_LABELS).

Leakage 방지:
  - split_anchors(seed=42, val=0.10, test=0.10)로 얻은 train subject만 학습에 사용
  - val/test subject는 학습에 절대 노출 X
  - assert로 disjoint 확인
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from training_duett.data_processing import (
    CXR_JPG_ROOT, AnchorConfig, build_anchors, dicom_to_jpg_path,
    load_duett_meta, load_static_df, split_anchors,
)

# =============================================================================
# Config
# =============================================================================
FINAL_DF_PATH = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/subject_data/mimic_3_1/final_df_20260713"
STATIC_PATH   = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/mimic_3_1/static_full.ftr"
DUETT_CKPT    = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/duett/checkpoints/mimic_ssl/n24_s12/pretrain-epoch=130-val_loss=0.2717.ckpt"
META_PATH     = os.path.join(os.path.dirname(DUETT_CKPT), "meta_with_stats.pkl")

PATHOLOGY_LABELS = ["label_edema", "label_cardiomegaly", "label_effusion", "label_pneumonia"]
NUM_CLASSES = len(PATHOLOGY_LABELS)

SAVE_DIR = Path("/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/checkpoints/cxr_linear_head")

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 128
NUM_WORKERS  = 8
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 30
DROPOUT      = 0.1


# =============================================================================
# Dataset — DuETT anchor row → CXR pixel_values + multi-label
# =============================================================================
class CXRAnchorDataset(Dataset):
    def __init__(self, anchor_df: pd.DataFrame, processor, label_cols: list[str],
                 cxr_root: str = CXR_JPG_ROOT):
        self.df = anchor_df.reset_index(drop=True)
        self.processor = processor
        self.pathology_cols = [f"_y_{c}" for c in label_cols]
        self.cxr_root = cxr_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = dicom_to_jpg_path(row["subject_id"], row["study_id"], row["dicom_id"],
                                 self.cxr_root)
        img = Image.open(path).convert("RGB")
        px = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        raw = np.array([row[c] for c in self.pathology_cols], dtype=np.float32)
        m = ~np.isnan(raw)
        y = np.nan_to_num(raw, nan=0.0)
        return {
            "pixel_values": px,
            "labels":       torch.from_numpy(y),
            "label_mask":   torch.from_numpy(m.astype(np.float32)),
        }


# =============================================================================
# Model — cxr_linear_training.ipynb RadDinoClassifier 그대로
# =============================================================================
class RadDinoClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/rad-dino")
        hidden = self.encoder.config.hidden_size
        # Sequential(Dropout, Linear) → state_dict keys: "1.weight"/"1.bias"
        # (DuETT teacher의 pretrained_cxr_head 로딩 규약과 호환)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.encoder(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


def masked_bce(logits, labels, label_mask):
    l = nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
    mf = label_mask.float()
    vc = mf.sum()
    if vc.item() == 0:
        return logits.sum() * 0.0
    return (l * mf).sum() / vc


# =============================================================================
# Eval helper
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, label_cols, device, desc="eval"):
    model.eval()
    all_p, all_y, all_m = [], [], []
    tot_l, tot_v = 0.0, 0.0
    for batch in tqdm(loader, desc=desc, leave=False):
        px = batch["pixel_values"].to(device, non_blocking=True)
        y  = batch["labels"].to(device, non_blocking=True)
        m  = batch["label_mask"].to(device, non_blocking=True)
        logits = model(px)
        loss = masked_bce(logits, y, m)
        v = m.float().sum().item()
        tot_l += loss.item() * v
        tot_v += v
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(y.cpu().numpy())
        all_m.append(m.cpu().numpy())

    probs  = np.concatenate(all_p)
    labels = np.concatenate(all_y)
    masks  = np.concatenate(all_m).astype(bool)

    per_label = {}
    aurocs, auprcs = [], []
    for i, name in enumerate(label_cols):
        mm = masks[:, i]
        yk = labels[mm, i]
        pk = probs[mm, i]
        if len(np.unique(yk)) < 2:
            per_label[name] = {"auroc": float("nan"), "auprc": float("nan"),
                               "n": int(mm.sum()), "pos": int(yk.sum())}
            continue
        au = roc_auc_score(yk, pk)
        pr = average_precision_score(yk, pk)
        per_label[name] = {"auroc": au, "auprc": pr,
                           "n": int(mm.sum()), "pos": int(yk.sum())}
        aurocs.append(au); auprcs.append(pr)

    return {
        "loss":        tot_l / max(tot_v, 1.0),
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "per_label":   per_label,
    }


def print_metrics(m, title):
    print(f"\n[{title}]  loss={m['loss']:.4f}  "
          f"macro-AUROC={m['macro_auroc']:.4f}  macro-AUPRC={m['macro_auprc']:.4f}")
    print(f"  {'label':<25}{'n':>8}{'pos':>8}{'AUROC':>10}{'AUPRC':>10}")
    for name, s in m["per_label"].items():
        print(f"  {name:<25}{s['n']:>8}{s['pos']:>8}"
              f"{s['auroc']:>10.4f}{s['auprc']:>10.4f}")


# =============================================================================
# Main
# =============================================================================
def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[1] Building DuETT anchor_df + subject-level split ...")
    cfg = AnchorConfig(
        final_df_path=FINAL_DF_PATH,
        static_path=STATIC_PATH,
        meta_path=META_PATH,
        label_col="label_edema",
        n_timesteps=24,
        split_seed=42,
        val_size=0.10,
        test_size=0.10,
        pathology_labels=tuple(PATHOLOGY_LABELS),
    )
    meta = load_duett_meta(META_PATH)
    final_df = pd.read_feather(FINAL_DF_PATH)
    static_df = load_static_df(STATIC_PATH)
    anchor_df, _, _ = build_anchors(cfg, meta, final_df, static_df)
    splits = split_anchors(anchor_df, seed=cfg.split_seed,
                           val_size=cfg.val_size, test_size=cfg.test_size)

    train_df = anchor_df.iloc[splits["train"]].reset_index(drop=True)
    val_df   = anchor_df.iloc[splits["val"]  ].reset_index(drop=True)
    test_df  = anchor_df.iloc[splits["test"] ].reset_index(drop=True)

    # ── Leakage 검증 ─────────────────────────────────────────────
    train_subj = set(train_df["subject_id"].astype(int))
    val_subj   = set(val_df["subject_id"].astype(int))
    test_subj  = set(test_df["subject_id"].astype(int))
    assert train_subj.isdisjoint(val_subj),  "LEAKAGE: train ∩ val subject non-empty"
    assert train_subj.isdisjoint(test_subj), "LEAKAGE: train ∩ test subject non-empty"
    assert val_subj.isdisjoint(test_subj),   "LEAKAGE: val ∩ test subject non-empty"
    print(f"    train: n_samples={len(train_df):>6d}  n_subj={len(train_subj):>5d}")
    print(f"    val:   n_samples={len(val_df):>6d}  n_subj={len(val_subj):>5d}")
    print(f"    test:  n_samples={len(test_df):>6d}  n_subj={len(test_subj):>5d}")
    print("    ✓ no subject leakage across splits")

    print("\n    Positive prevalence per split (NaN 무시):")
    print(f"    {'label':<25}{'train':>10}{'val':>10}{'test':>10}")
    for lbl in PATHOLOGY_LABELS:
        col = f"_y_{lbl}"
        print(f"    {lbl:<25}"
              f"{train_df[col].mean(skipna=True):>10.4f}"
              f"{val_df[col].mean(skipna=True):>10.4f}"
              f"{test_df[col].mean(skipna=True):>10.4f}")

    # ── Dataloaders ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[2] Building dataloaders ...")
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    train_ds = CXRAnchorDataset(train_df, processor, PATHOLOGY_LABELS)
    val_ds   = CXRAnchorDataset(val_df,   processor, PATHOLOGY_LABELS)
    test_ds  = CXRAnchorDataset(test_df,  processor, PATHOLOGY_LABELS)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model + optimizer ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("[3] Building model ...")
    model = RadDinoClassifier(NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"    trainable params: {n_train:,}  "
          f"(expected {model.encoder.config.hidden_size} * {NUM_CLASSES} + {NUM_CLASSES})")
    optimizer = AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    # ── Train loop ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[4] Training ...")
    best_val = -float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.encoder.eval()  # frozen: 항상 eval

        run_l, run_v = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            px = batch["pixel_values"].to(DEVICE, non_blocking=True)
            y  = batch["labels"].to(DEVICE, non_blocking=True)
            m  = batch["label_mask"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(px)
            loss = masked_bce(logits, y, m)
            loss.backward()
            optimizer.step()
            v = m.float().sum().item()
            run_l += loss.item() * v
            run_v += v
            pbar.set_postfix(loss=run_l / max(run_v, 1.0))

        train_loss = run_l / max(run_v, 1.0)
        val_metrics = evaluate(model, val_loader, PATHOLOGY_LABELS, DEVICE, desc=f"val {epoch}")
        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}")
        print_metrics(val_metrics, f"val (epoch {epoch})")

        if val_metrics["macro_auroc"] > best_val:
            best_val   = val_metrics["macro_auroc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.classifier.state_dict().items()}
            print(f"  ** new best macro-AUROC: {best_val:.4f}  (epoch {best_epoch})")

    # ── Final test with best ckpt ───────────────────────────────
    print("\n" + "=" * 80)
    print(f"[5] Restoring best (epoch {best_epoch}, val macro-AUROC={best_val:.4f}) and evaluating on test ...")
    model.classifier.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, PATHOLOGY_LABELS, DEVICE, desc="test")
    print_metrics(test_metrics, f"TEST (best-val ckpt @ epoch {best_epoch})")

    # ── Save ckpt (DuETT teacher와 호환되는 포맷) ─────────────────
    print("\n" + "=" * 80)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = SAVE_DIR / f"raddino_linear_head_icu4_{tag}.pt"
    meta_path = SAVE_DIR / f"raddino_linear_head_icu4_{tag}.json"

    torch.save({
        "classifier_state_dict": best_state,       # keys "1.weight" / "1.bias"
        "label_cols":            PATHOLOGY_LABELS,  # 순서 == DuETT PATHOLOGY_LABELS
        "encoder_name":          "microsoft/rad-dino",
        "hidden_size":           model.encoder.config.hidden_size,
        "num_classes":           NUM_CLASSES,
        "freeze_encoder":        True,
        "best_val_macro_auroc":  best_val,
        "best_epoch":            best_epoch,
        "test_metrics":          {
            "macro_auroc": test_metrics["macro_auroc"],
            "macro_auprc": test_metrics["macro_auprc"],
            "per_label":   test_metrics["per_label"],
        },
        "trained_on": "DuETT train subject anchor CXRs (ICU-adjacent)",
    }, ckpt_path)

    with open(meta_path, "w") as f:
        json.dump({
            "ckpt":                 str(ckpt_path),
            "label_cols":           PATHOLOGY_LABELS,
            "best_val_macro_auroc": best_val,
            "best_epoch":           best_epoch,
            "test_macro_auroc":     test_metrics["macro_auroc"],
            "test_macro_auprc":     test_metrics["macro_auprc"],
            "test_per_label":       test_metrics["per_label"],
        }, f, indent=2)

    print(f"[6] Saved:")
    print(f"    ckpt: {ckpt_path}")
    print(f"    meta: {meta_path}")
    print()
    print(f"    → DuETT teacher에 plug-in:")
    print(f"      --pretrained_cxr_head_ckpt {ckpt_path}")


if __name__ == "__main__":
    main()
