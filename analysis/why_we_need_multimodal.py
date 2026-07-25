"""ICU-hardness ablation on the pretrained head's own TEST set.

같은 pretrained head + 같은 evaluation 파이프라인으로 mutually exclusive
4-way group 비교 → "ICU CXR이 general population보다 어렵다"를 정량 증명.

Groups (모두 pretrained TEST subject 안, subject leakage 없음):
  G0_FULL         = pretrained TEST 전체 (28,960장) — 논문 published RAD-DINO context
  G1_NON-ICU      = subject가 ICU anchor에 없음 (일반 인구)
  G2_ICU-non-anch = subject는 ICU set에 있지만 이 CXR은 anchor 아님
                    (ICU 환자의 안정기/외래 CXR — subject 요인만)
  G3_ICU-anchor   = dicom이 ICU anchor에 있음 (= 멀티모달 TEST 자체)
                    (subject + 촬영상황 요인)
  → G1 ∪ G2 ∪ G3 = G0  (mutually exclusive decomposition)

산출물 (analysis/figs_test/):
  - icu_hardness_summary.json          — 모든 numerical 결과
  - icu_hardness_table_3label.csv      — 본문 표 (3-label)
  - icu_hardness_table_7label.csv      — 부록 표 (7-label)
  - icu_hardness_macro.png             — macro AUROC bar chart
  - icu_hardness_per_label_3.png       — 3-label × 4-group grouped bar
  - icu_hardness_per_label_7.png       — 7-label × 4-group grouped bar
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning")

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

from training_duett.data_processing import (
    AnchorConfig, DEFAULT_PATHOLOGY_LABELS,
    build_anchors, load_duett_meta, load_static_df,
)

# =============================================================================
# Config
# =============================================================================
CXR_JPG_ROOT  = "/home/DAHS1/mimic-cxr-jpg-2.0.0"
FTR_PATH      = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/subject_data/final_cxr_df_20260713.ftr"
HEAD_CKPT     = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/checkpoints/cxr_linear_head/raddino_linear_head_20260715_053311.pt"
FINAL_DF_PATH = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/subject_data/mimic_3_1/final_df_20260713"
STATIC_PATH   = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/mimic_3_1/static_full.ftr"
DUETT_CKPT    = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/duett/checkpoints/mimic_ssl/n24_s12/pretrain-epoch=130-val_loss=0.2717.ckpt"
META_PATH     = os.path.join(os.path.dirname(DUETT_CKPT), "meta_with_stats.pkl")

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 128
WORKERS  = 8

# Pretrained head가 학습한 라벨 순서 (7개 그대로 — 부록 리포트용)
LABEL_COLS = [
    "label_cardiomegaly", "label_pneumonia", "label_atelectasis",
    "label_opacity", "label_consolidation", "label_edema", "label_effusion",
]
# 본문 리포트용 label 집합 = 멀티모달 target
MAIN_LABELS = list(DEFAULT_PATHOLOGY_LABELS)  # ["label_edema","label_cardiomegaly","label_effusion"]

# 출력 위치
OUT_DIR = Path("/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/analysis/figs_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dataset — cxr_linear_training.ipynb의 CXRDataset과 완전 동일
# =============================================================================
class CXRDataset(Dataset):
    def __init__(self, df, processor, label_cols, image_root=CXR_JPG_ROOT):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.label_cols = label_cols
        self.image_root = image_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_root, row["image_path"])
        img = Image.open(path).convert("RGB")
        px = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        raw = row[self.label_cols].to_numpy(dtype=np.float32)
        mask = ~np.isnan(raw)
        y = np.nan_to_num(raw, nan=0.0)
        return {
            "pixel_values": px,
            "labels":       torch.from_numpy(y),
            "label_mask":   torch.from_numpy(mask),
        }


def eval_head(backbone, classifier, df, processor, label_cols, tag):
    print(f"    [eval] {tag}  n={len(df)}")
    if len(df) == 0:
        return None
    ds = CXRDataset(df, processor, label_cols)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=WORKERS, pin_memory=True)

    all_logits, all_y, all_m = [], [], []
    for batch in loader:
        px = batch["pixel_values"].to(DEVICE, non_blocking=True)
        with torch.no_grad():
            cls = backbone(pixel_values=px).last_hidden_state[:, 0, :]
            logits = classifier(cls)
        all_logits.append(logits.cpu().numpy())
        all_y.append(batch["labels"].numpy())
        all_m.append(batch["label_mask"].numpy())

    logits = np.concatenate(all_logits)
    y = np.concatenate(all_y)
    m = np.concatenate(all_m).astype(bool)
    probs = 1.0 / (1.0 + np.exp(-logits))

    per_label = {}
    aurocs, auprcs = [], []
    for i, lbl in enumerate(label_cols):
        mk = m[:, i]
        yk, pk = y[mk, i], probs[mk, i]
        if mk.sum() < 2 or len(np.unique(yk)) < 2:
            per_label[lbl] = {"n": int(mk.sum()), "pos": int(yk.sum()),
                              "auroc": float("nan"), "auprc": float("nan")}
            continue
        au = roc_auc_score(yk, pk)
        pr = average_precision_score(yk, pk)
        per_label[lbl] = {"n": int(mk.sum()), "pos": int(yk.sum()),
                          "auroc": au, "auprc": pr}
        aurocs.append(au); auprcs.append(pr)

    return {
        "n_img":       len(df),
        "n_subj":      df["subject_id"].nunique(),
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "per_label":   per_label,
    }


# =============================================================================
# (1) pretrained test split 재현
# =============================================================================
print("=" * 90)
print("[1] Reproducing pretrained head TEST split (cxr_linear_training.ipynb 셀 6과 동일) ...")
cxr_full = pd.read_feather(FTR_PATH)
cxr_df = cxr_full[cxr_full[LABEL_COLS].notna().any(axis=1)]
cxr_df = cxr_df.drop_duplicates(subset=["dicom_id"]).reset_index(drop=True)
subject_ids = cxr_df["subject_id"].unique()
train_ids, temp_ids = train_test_split(subject_ids, test_size=0.30, random_state=42)
val_ids, test_ids   = train_test_split(temp_ids,   test_size=0.50, random_state=42)
test_df = cxr_df[cxr_df["subject_id"].isin(test_ids)].reset_index(drop=True)
print(f"    pretrained TEST: n_img={len(test_df)}  n_subj={test_df['subject_id'].nunique()}")


# =============================================================================
# (2) DuETT ICU subject / dicom set
# =============================================================================
print("\n" + "=" * 90)
print("[2] Building DuETT anchor set (ICU-adjacent CXR 목록) ...")
cfg = AnchorConfig(
    final_df_path=FINAL_DF_PATH, static_path=STATIC_PATH, meta_path=META_PATH,
    label_col="label_edema", n_timesteps=24, split_seed=42,
    # Aligned split 사용 후 val_size/test_size 미사용
    pathology_labels=DEFAULT_PATHOLOGY_LABELS,
)
meta = load_duett_meta(META_PATH)
final_df = pd.read_feather(FINAL_DF_PATH)
static_df = load_static_df(STATIC_PATH)
anchor_df, _, _ = build_anchors(cfg, meta, final_df, static_df)

duett_subj  = set(anchor_df["subject_id"].astype(int).unique())
duett_dicom = set(anchor_df["dicom_id"].astype(str).unique())
print(f"    DuETT anchors: n_subj={len(duett_subj)}  n_dicom={len(duett_dicom)}")


# =============================================================================
# (3) Mutually exclusive slice: G0 ⊃ G1 ⊔ G2 ⊔ G3
# =============================================================================
print("\n" + "=" * 90)
print("[3] Slicing pretrained TEST into mutually exclusive ICU groups ...")
test_df["subject_id_int"] = test_df["subject_id"].astype(int)
test_df["is_icu_subj"]    = test_df["subject_id_int"].isin(duett_subj)
test_df["is_icu_dicom"]   = test_df["dicom_id"].astype(str).isin(duett_dicom)

slices = {
    "G0_FULL":         test_df.reset_index(drop=True),
    "G1_NON-ICU":      test_df[~test_df["is_icu_subj"]].reset_index(drop=True),
    "G2_ICU-non-anch": test_df[test_df["is_icu_subj"] & ~test_df["is_icu_dicom"]].reset_index(drop=True),
    "G3_ICU-anchor":   test_df[test_df["is_icu_dicom"]].reset_index(drop=True),
}
for name, df in slices.items():
    print(f"    {name:<18s}  n_img={len(df):>6d}  n_subj={df['subject_id'].nunique():>5d}")

# Decomposition 검증: G1 + G2 + G3 = G0
assert (len(slices["G1_NON-ICU"]) + len(slices["G2_ICU-non-anch"])
        + len(slices["G3_ICU-anchor"])) == len(slices["G0_FULL"]), \
    "mutually exclusive decomposition failed"
print(f"    ✓ G1+G2+G3 = G0 ({len(slices['G0_FULL'])}장) decomposition 확정")


# =============================================================================
# (4) RAD-DINO + pretrained head 로드
# =============================================================================
print("\n" + "=" * 90)
print("[4] Loading RAD-DINO + pretrained head ...")
processor  = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
backbone   = AutoModel.from_pretrained("microsoft/rad-dino").eval().to(DEVICE)
for p in backbone.parameters():
    p.requires_grad = False

state = torch.load(HEAD_CKPT, map_location="cpu", weights_only=False)
hidden = backbone.config.hidden_size
classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, state["num_classes"]))
classifier.load_state_dict(state["classifier_state_dict"])
classifier.eval().to(DEVICE)


# =============================================================================
# (5) 4개 subset 각각 evaluation
# =============================================================================
print("\n" + "=" * 90)
print("[5] Running evaluations ...")
results = {}
for name, df in slices.items():
    results[name] = eval_head(backbone, classifier, df, processor, LABEL_COLS, name)


# =============================================================================
# (6) Label-subset macro 계산 (본문 3-label + 부록 7-label)
# =============================================================================
def compute_macro(per_label: dict, label_subset: list[str]) -> tuple[float, float]:
    aus, prs = [], []
    for lbl in label_subset:
        s = per_label.get(lbl)
        if s is None or not np.isfinite(s["auroc"]):
            continue
        aus.append(s["auroc"]); prs.append(s["auprc"])
    return (float(np.mean(aus)) if aus else float("nan"),
            float(np.mean(prs)) if prs else float("nan"))


LABEL_SETS = {
    "3-label (Research Label)":     MAIN_LABELS,      # 본문
    "7-label": LABEL_COLS,       # 부록
}

# 각 group × 각 label set의 macro 저장
macros = {}
for group_name, r in results.items():
    macros[group_name] = {}
    for set_name, lbls in LABEL_SETS.items():
        macros[group_name][set_name] = compute_macro(r["per_label"], lbls)


# =============================================================================
# (7) Console 표 출력 (각 label set별)
# =============================================================================
for set_name, lbls in LABEL_SETS.items():
    print("\n" + "=" * 90)
    print(f"[7] Group × Label — {set_name}")
    header = (f"{'group':<18s} {'n_img':>7s} {'n_subj':>7s} {'label':<22s} "
              f"{'pos%':>7s} {'AUROC':>8s} {'AUPRC':>8s}")
    print(header)
    print("-" * len(header))
    for name in ["G0_FULL", "G1_NON-ICU", "G2_ICU-non-anch", "G3_ICU-anchor"]:
        r = results[name]
        for lbl in lbls:
            s = r["per_label"][lbl]
            pos_pct = 100.0 * s["pos"] / s["n"] if s["n"] else float("nan")
            print(f"{name:<18s} {r['n_img']:>7d} {r['n_subj']:>7d} {lbl:<22s} "
                  f"{pos_pct:>7.2f} {s['auroc']:>8.4f} {s['auprc']:>8.4f}")
        au, pr = macros[name][set_name]
        print(f"{name:<18s} {'':>7s} {'':>7s} {'-- MACRO --':<22s} "
              f"{'':>7s} {au:>8.4f} {pr:>8.4f}")
        print()


# =============================================================================
# (8) 저장: JSON / CSV / seaborn figures
# =============================================================================
print("=" * 90)
print(f"[8] Saving artifacts → {OUT_DIR}")

# 8a. JSON — 전체 raw 결과
summary = {
    "groups": {
        name: {
            "n_img":  r["n_img"], "n_subj": r["n_subj"],
            "per_label": r["per_label"],
            "macro":  macros[name],
        } for name, r in results.items()
    },
    "label_sets": LABEL_SETS,
    "notes": "same pretrained head + same pipeline; only subset filter varies.",
}
with open(OUT_DIR / "icu_hardness_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(f"    - icu_hardness_summary.json")

# 8b. CSV — 각 label set별 표
for set_name, lbls in LABEL_SETS.items():
    rows = []
    for group in ["G0_FULL", "G1_NON-ICU", "G2_ICU-non-anch", "G3_ICU-anchor"]:
        r = results[group]
        for lbl in lbls:
            s = r["per_label"][lbl]
            rows.append({
                "group":   group, "label": lbl,
                "n_img":   r["n_img"], "n_subj": r["n_subj"],
                "n_valid": s["n"],  "n_pos":  s["pos"],
                "pos_pct": 100 * s["pos"] / s["n"] if s["n"] else np.nan,
                "AUROC":   s["auroc"], "AUPRC": s["auprc"],
            })
        au, pr = macros[group][set_name]
        rows.append({
            "group": group, "label": "MACRO",
            "n_img": r["n_img"], "n_subj": r["n_subj"],
            "n_valid": np.nan, "n_pos": np.nan, "pos_pct": np.nan,
            "AUROC": au, "AUPRC": pr,
        })
    csv_name = f"icu_hardness_table_{set_name.split()[0].replace('-','')}.csv"
    pd.DataFrame(rows).to_csv(OUT_DIR / csv_name, index=False)
    print(f"    - {csv_name}")


# 8c. Seaborn figures
sns.set_theme(style="white", context="paper")
GROUP_ORDER  = ["G0_FULL", "G1_NON-ICU", "G2_ICU-non-anch", "G3_ICU-anchor"]
GROUP_LABELS = ["All studies",
                "No ICU history",
                "Patients with\nhistory of ICU",
                "ICU imaging\n(Research cohort)"]
GROUP_COLORS = ["#808080", "#4CAF50", "#FF9800", "#E53935"]

# AUROC / AUPRC를 한 figure에 좌우 subplot으로 배치
# AUROC: 0.5~1.0 (0.1 tick),  AUPRC: 0.0~1.0 (0.1 tick — pneumonia처럼 낮은 값 잘리지 않도록)
METRIC_SPECS = [("AUROC", 0, (0.5, 1.0)), ("AUPRC", 1, (0.0, 1.0))]

# --- Figure 1: macro AUROC + AUPRC across groups (main) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
x = np.arange(len(GROUP_ORDER))
width = 0.35
for metric_name, metric_idx, (ylo, yhi) in METRIC_SPECS:
    ax = axes[metric_idx]
    for i, (set_name, _) in enumerate(LABEL_SETS.items()):
        vals = [macros[g][set_name][metric_idx] for g in GROUP_ORDER]
        off  = -width/2 + i*width
        # label set 구분: 3-label = solid, 7-label = 대각선 무늬
        hatch = "" if i == 0 else "///"
        bars = ax.bar(x + off, vals, width,
                       color=[GROUP_COLORS[j] for j in range(len(GROUP_ORDER))],
                       edgecolor="black", linewidth=0.8,
                       hatch=hatch)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(GROUP_LABELS, fontsize=9)
    ax.set_ylabel(metric_name)
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(np.arange(ylo, yhi + 1e-9, 0.1))
    # ax.set_title(f"Macro {metric_name} by Patient Cohort", fontsize=11)
    # 커스텀 legend: 무늬로 label set 구분 (bar 색은 x축 group을 가리킴)
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="3-label"),
        Patch(facecolor="white", edgecolor="black", hatch="///",
              label="7-label"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
fig.suptitle("Img Encoder Classification Performance by Patient Cohort",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "icu_hardness_macro.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"    - icu_hardness_macro.png")

# --- Figure 2 & 3: per-label × group grouped bar per label set (AUROC + AUPRC) ---
DISPLAY_NAMES = dict(zip(GROUP_ORDER, [s.replace("\n", " ") for s in GROUP_LABELS]))
for set_name, lbls in LABEL_SETS.items():
    tag = "3" if "3-label" in set_name else "7"
    data = []
    for g in GROUP_ORDER:
        for lbl in lbls:
            data.append({"group": DISPLAY_NAMES[g],
                         "label": lbl.replace("label_", ""),
                         "AUROC": results[g]["per_label"][lbl]["auroc"],
                         "AUPRC": results[g]["per_label"][lbl]["auprc"]})
    df_plot = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(lbls) * 1.9), 4.5))
    for metric_name, metric_idx, (ylo, yhi) in METRIC_SPECS:
        ax = axes[metric_idx]
        sns.barplot(data=df_plot, x="label", y=metric_name, hue="group",
                    palette=GROUP_COLORS, edgecolor="black", ax=ax)
        ax.set_ylim(ylo, yhi)
        ax.set_yticks(np.arange(ylo, yhi + 1e-9, 0.1))
        ax.set_ylabel(metric_name); ax.set_xlabel("")
        ax.set_title(f"Per-label {metric_name})", fontsize=11)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="", fontsize=8, loc="upper right", ncol=2)
    plt.tight_layout()
    fname = f"icu_hardness_per_label_{tag}.png"
    plt.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    - {fname}")