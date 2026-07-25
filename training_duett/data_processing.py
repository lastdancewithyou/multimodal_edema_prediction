from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from duett.mimic_dataset import build_stay_tensor, encode_static


# =============================================================================
# Multi-label pathology config
# =============================================================================
# Index 0은 반드시 main target(Edema). PathologyPerceiver의 query 순서와 일치.
DEFAULT_PATHOLOGY_LABELS = (
    "label_edema",
    "label_cardiomegaly",
    "label_effusion",
    "label_pneumonia",
)


# =============================================================================
# Pretrained CXR head alignment
# =============================================================================
# 멀티모달 subject split을 pretrained CXR head의 subject split에 정합시키기 위한
# 참조. 이 두 값이 cxr_linear_training.ipynb에서 사용된 값과 정확히 일치해야
# subject-disjoint 정합이 성립.
# 나중에 pretrained head를 다른 데이터로 재학습하면 여기만 갱신하면 됨.
PRETRAIN_CXR_FTR_PATH = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/subject_data/final_cxr_df_20260713.ftr"
PRETRAIN_LABEL_COLS = [
    "label_cardiomegaly", "label_pneumonia", "label_atelectasis",
    "label_opacity", "label_consolidation", "label_edema", "label_effusion",
]


# =============================================================================
# Meta / static loaders
# =============================================================================
REQUIRED_META_KEYS = (
    "ALL_VARS", "ALL_COUNTS", "ONEHOT_STATIC", "D_STATIC", "LABEL_COL",
    "means", "stds", "age_mean", "age_std", "N_TIMESTEPS",
)


def load_duett_meta(meta_path: str) -> dict:
    """meta_with_stats.pkl 로드 + 정규화 통계 sanity check.

    SSL 스크립트가 만든 meta에는 스키마 + train split 통계가 모두 있어야 함.
    없으면 KeyError로 나중에 터지느니 여기서 명확한 메시지로 실패시킴.
    """
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise KeyError(
            f"meta.pkl에 필수 키 누락: {missing}\n"
            f"  파일: {meta_path}\n"
            f"  → SSL 학습(train_duett_ssl.py)을 새로 돌리면 meta_with_stats.pkl이 "
            f"ckpt 폴더에 자동 저장됩니다."
        )

    all_vars = list(meta["ALL_VARS"])
    means, stds = meta["means"], meta["stds"]

    # ─── 1. 변수 커버리지: ALL_VARS 전체가 means/stds에 있어야 ───
    miss_m = [v for v in all_vars if v not in means]
    miss_s = [v for v in all_vars if v not in stds]
    if miss_m or miss_s:
        raise KeyError(
            f"stats에 빠진 변수 있음:\n"
            f"  means 누락: {miss_m}\n  stds  누락: {miss_s}"
        )

    # ─── 2. 값 sanity: NaN/Inf, std==0 (분모 폭발 위험) ───
    import math
    bad_mean = [v for v in all_vars if not math.isfinite(float(means[v]))]
    bad_std  = [v for v in all_vars if not math.isfinite(float(stds[v]))]
    zero_std = [v for v in all_vars if float(stds[v]) == 0.0]
    if bad_mean or bad_std:
        raise ValueError(
            f"stats에 NaN/Inf:\n  means: {bad_mean}\n  stds: {bad_std}"
        )
    if zero_std:
        print(f"[meta][WARN] std==0 인 변수: {zero_std} (정규화 시 1e-7 fallback)")

    if not math.isfinite(float(meta["age_mean"])) or not math.isfinite(float(meta["age_std"])):
        raise ValueError(f"age_mean/age_std 값 이상: "
                          f"{meta['age_mean']}, {meta['age_std']}")

    # ─── 3. 요약 출력 (사람이 눈으로 확인) ───
    print(f"[meta] loaded: {meta_path}")
    print(f"  N_TIMESTEPS = {meta['N_TIMESTEPS']}   "
          f"n_ts_vars = {len(all_vars)}   D_STATIC = {meta['D_STATIC']}")
    print(f"  age: mean={float(meta['age_mean']):.2f}  std={float(meta['age_std']):.2f}")
    mean_vals = [float(means[v]) for v in all_vars]
    std_vals  = [float(stds[v])  for v in all_vars]
    print(f"  ts means: min={min(mean_vals):.3g}  max={max(mean_vals):.3g}  "
          f"median={sorted(mean_vals)[len(mean_vals)//2]:.3g}")
    print(f"  ts stds:  min={min(std_vals):.3g}   max={max(std_vals):.3g}   "
          f"median={sorted(std_vals)[len(std_vals)//2]:.3g}")
    for split in ("train_ids", "val_ids", "test_ids"):
        if split in meta:
            print(f"  {split}: n={len(meta[split])}")

    return meta


def load_static_df(static_path: str) -> pd.DataFrame:
    return pd.read_feather(static_path).drop_duplicates("stay_id").set_index("stay_id")


# =============================================================================
# Anchor construction
# =============================================================================
@dataclass
class AnchorConfig:
    final_df_path: str
    static_path: str
    meta_path: str
    label_col: str = "label_edema"
    n_timesteps: int = 24        # DuETT pretrain N_TIMESTEPS
    min_history_slots: int = 1   # minimum observed slots inside [e-K, e)
    split_seed: int = 42
    # Aligned split(pretrained head와 subject-level 정합) 사용 시 아래 두 값은
    # 무의미. 스플릿 비율은 pretrained head의 70/15/15가 자동 상속됨.
    # val_size: float = 0.1
    # test_size: float = 0.1
    # Multi-label aux (PathologyPerceiver용). 순서 = query 순서. 반드시 index 0 = label_col.
    pathology_labels: tuple[str, ...] = field(default_factory=lambda: DEFAULT_PATHOLOGY_LABELS)


def build_anchors(cfg: AnchorConfig,
                  meta: dict,
                  final_df: pd.DataFrame,
                  static_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Filter final_df to anchor rows and return (anchor_df, ts_vars, ts_counts).

    ts_vars follows meta['ALL_VARS'] intersected with final_df columns.
    """
    all_vars_meta = list(meta["ALL_VARS"])
    all_counts_meta = list(meta["ALL_COUNTS"])
    keep_pairs = [(v, c) for v, c in zip(all_vars_meta, all_counts_meta)
                  if v in final_df.columns and c in final_df.columns]
    ts_vars = [v for v, _ in keep_pairs]
    ts_counts = [c for _, c in keep_pairs]
    dropped = set(all_vars_meta) - set(ts_vars)
    if dropped:
        print(f"[anchors] meta vars missing from final_df (skipped): {sorted(dropped)}")
    print(f"[anchors] using {len(ts_vars)} ts vars from meta order")

    df = final_df
    if "cxr_flag" not in df.columns:
        raise ValueError("final_df missing 'cxr_flag'")
    if cfg.label_col not in df.columns:
        raise ValueError(f"final_df missing label col '{cfg.label_col}'")

    cxr_rows = df[df["cxr_flag"] == 1].copy()
    print(f"[anchors] cxr_flag==1 rows: {len(cxr_rows)}")

    lab = cxr_rows[cfg.label_col]
    y = pd.Series(np.nan, index=cxr_rows.index, dtype="float32")
    y[lab == 1.0] = 1.0
    y[lab == 0.0] = 0.0
    y[lab == -1.0] = 0.0  # U → 0
    keep = y.notna()
    cxr_rows = cxr_rows[keep].copy()
    cxr_rows["y_e"] = y[keep].values.astype(np.float32)
    print(f"[anchors] after Edema label filter: {len(cxr_rows)}  "
          f"(pos={int(cxr_rows['y_e'].sum())})")

    cxr_rows = cxr_rows[cxr_rows["slot_idx"] >= cfg.n_timesteps].copy()
    print(f"[anchors] after slot_idx >= {cfg.n_timesteps} filter: {len(cxr_rows)}")

    stay_ids = cxr_rows["stay_id"].unique()
    have_static = static_df.index.intersection(stay_ids)
    cxr_rows = cxr_rows[cxr_rows["stay_id"].isin(have_static)].copy()
    print(f"[anchors] after static-join filter: {len(cxr_rows)}")

    # ─── Multi-label pathology 라벨을 [row, K] 로 materialize ───────────────
    # index 0 = main label_col (Edema). 나머지는 aux.
    if cfg.pathology_labels[0] != cfg.label_col:
        raise ValueError(
            f"pathology_labels[0] must equal label_col. "
            f"got pathology_labels[0]={cfg.pathology_labels[0]!r}, label_col={cfg.label_col!r}"
        )
    missing_cols = [c for c in cfg.pathology_labels if c not in cxr_rows.columns]
    if missing_cols:
        raise ValueError(f"final_df missing pathology label cols: {missing_cols}")

    # 0/1만 있는 데이터 (사용자 확인). NaN은 그대로 두고 mask 처리.
    pathology_cols_out = []
    print(f"[anchors] pathology labels (index 0 = main):")
    for i, col in enumerate(cfg.pathology_labels):
        out_col = f"_y_{col}"
        cxr_rows[out_col] = pd.to_numeric(cxr_rows[col], errors="coerce").astype("float32")
        pathology_cols_out.append(out_col)
        vals = cxr_rows[out_col]
        n_valid = int(vals.notna().sum())
        n_pos = int((vals == 1.0).sum())
        n_neg = int((vals == 0.0).sum())
        pos_frac = n_pos / n_valid if n_valid > 0 else 0.0
        print(f"  [{i}] {col:25s}  valid={n_valid:6d}  pos={n_pos:5d}  neg={n_neg:5d}  "
              f"pos_frac={pos_frac:.4f}  nan={len(vals) - n_valid}")

    anchor_df = cxr_rows[[
        "subject_id", "stay_id", "hadm_id", "study_id", "dicom_id",
        "slot_idx", "y_e",
    ] + pathology_cols_out].reset_index(drop=True)
    return anchor_df, ts_vars, ts_counts


def split_anchors(anchor_df: pd.DataFrame,
                  seed: int = 42,
                  pretrained_ftr_path: str = PRETRAIN_CXR_FTR_PATH,
                  pretrained_label_cols: list[str] | None = None
                  ) -> dict[str, np.ndarray]:
    """Subject-level split aligned with pretrained CXR head's 70/15/15 split.

    Pretrained head가 cxr_linear_training.ipynb에서 사용한 split(seed=42)을
    그대로 재현한 뒤, ICU anchor의 각 subject를 그 split에 강제 매핑.
    결과 비율은 pretrained의 70/15/15를 자연 상속하며 ±1%p 편차 이내.

    이렇게 하면 어떤 subject도 pretrained TRAIN과 ICU TEST에 동시에 속하지
    않음이 물리적으로 보장됨 → subject-leakage 배제.
    """
    if pretrained_label_cols is None:
        pretrained_label_cols = PRETRAIN_LABEL_COLS

    # (1) Pretrained head의 subject split 재현
    cxr_full = pd.read_feather(pretrained_ftr_path)
    cxr = cxr_full[cxr_full[pretrained_label_cols].notna().any(axis=1)]
    cxr = cxr.drop_duplicates(subset=["dicom_id"])
    subj_all = cxr["subject_id"].unique()
    train_ids, temp_ids = train_test_split(subj_all, test_size=0.30, random_state=seed)
    val_ids, test_ids   = train_test_split(temp_ids, test_size=0.50, random_state=seed)
    pre_train = {int(x) for x in train_ids}
    pre_val   = {int(x) for x in val_ids}
    pre_test  = {int(x) for x in test_ids}

    # (2) ICU anchor row를 pretrained split에 매핑
    subj_col = anchor_df["subject_id"].astype(int).values
    idx = np.arange(len(anchor_df))
    train_idx = idx[np.isin(subj_col, list(pre_train))]
    val_idx   = idx[np.isin(subj_col, list(pre_val))]
    test_idx  = idx[np.isin(subj_col, list(pre_test))]

    # (3) 배정 실패 검증 — pretrained head가 label을 못 붙였던 subject 감지
    assigned = len(train_idx) + len(val_idx) + len(test_idx)
    if assigned != len(anchor_df):
        never = len(anchor_df) - assigned
        raise RuntimeError(
            f"{never} anchor rows not assigned to any pretrained split. "
            f"pretrained_ftr_path에 이 subject들의 label이 없음. "
            f"pretrained head 재학습 필요 또는 PRETRAIN_LABEL_COLS 확인."
        )

    # (4) subject-disjoint 검증
    assert set(train_idx).isdisjoint(val_idx) and set(train_idx).isdisjoint(test_idx) \
        and set(val_idx).isdisjoint(test_idx), "subject leakage detected"

    # (5) 스플릿 비율 출력 (소수 첫째 자리)
    total = len(anchor_df)
    print(f"[split] aligned with pretrained head (seed={seed}, target 70/15/15)")
    print(f"[split] TRAIN n={len(train_idx):>6d}  ratio={100*len(train_idx)/total:>4.1f}%  "
          f"n_subj={len(np.unique(subj_col[train_idx])):>5d}")
    print(f"[split] VAL   n={len(val_idx):>6d}  ratio={100*len(val_idx)/total:>4.1f}%  "
          f"n_subj={len(np.unique(subj_col[val_idx])):>5d}")
    print(f"[split] TEST  n={len(test_idx):>6d}  ratio={100*len(test_idx)/total:>4.1f}%  "
          f"n_subj={len(np.unique(subj_col[test_idx])):>5d}")

    return {"train": train_idx, "val": val_idx, "test": test_idx}


# =============================================================================
# TS window builder
# =============================================================================
def build_window_slots_df(
    final_df_group: pd.DataFrame,
    slot_e: int,
    n_timesteps: int) -> pd.DataFrame:
    lo = slot_e - n_timesteps
    win = final_df_group[(final_df_group["slot_idx"] >= lo) & (final_df_group["slot_idx"] < slot_e)].copy()
    win["slot_idx"] = (win["slot_idx"] - lo).astype(int)
    return win


# =============================================================================
# Dataset
# =============================================================================
CXR_JPG_ROOT = "/home/DAHS1/mimic-cxr-jpg-2.0.0/files"


def dicom_to_jpg_path(subject_id, study_id, dicom_id, root: str = CXR_JPG_ROOT) -> str:
    sid = str(int(subject_id))
    return os.path.join(root, f"p{sid[:2]}", f"p{sid}", f"s{int(study_id)}", f"{dicom_id}.jpg")


class DuettAnchorDataset(Dataset):
    """One CXR event = one sample.

    Args:
        mode: "teacher" (loads CXR pixel_values) or "student" (skips it).
        final_df: full merged DF, indexed by stay_id externally.
        anchor_df: filtered anchor rows.
        static_df: static features indexed by stay_id.
        meta: DuETT ckpt meta.
        ts_vars / ts_counts: variable order (must match backbone).
        cfg: AnchorConfig.
        image_processor: HF AutoImageProcessor for RAD-DINO (teacher only).
    """

    def __init__(self,
                 mode: str,
                 final_df_by_stay: dict[int, pd.DataFrame],
                 anchor_df: pd.DataFrame,
                 static_df: pd.DataFrame,
                 meta: dict,
                 ts_vars: list[str],
                 ts_counts: list[str],
                 cfg: AnchorConfig,
                 image_processor=None,
                 cxr_root: str = CXR_JPG_ROOT):
        assert mode in {"teacher", "student"}
        if mode == "teacher" and image_processor is None:
            raise ValueError("teacher mode requires image_processor")
        self.mode = mode
        self.final_by_stay = final_df_by_stay
        self.anchor_df = anchor_df.reset_index(drop=True)
        self.static_df = static_df
        self.meta = meta
        self.ts_vars = ts_vars
        self.ts_counts = ts_counts
        self.cfg = cfg
        self.image_processor = image_processor
        self.cxr_root = cxr_root

        self.K = cfg.n_timesteps
        self.bin_ends = torch.arange(1, self.K + 1).float() / 24.0

        self.onehot_static = list(meta["ONEHOT_STATIC"])
        self.age_mean = float(meta["age_mean"])
        self.age_std = float(meta["age_std"])
        self.means = meta["means"]
        self.stds = meta["stds"]

        # Multi-label pathology 컬럼 (anchor_df에 _y_{col} 로 미리 붙어 있음)
        self.pathology_labels = tuple(cfg.pathology_labels)
        self.pathology_cols = [f"_y_{c}" for c in self.pathology_labels]

    def __len__(self):
        return len(self.anchor_df)

    def __getitem__(self, i: int):
        row = self.anchor_df.iloc[i]
        stay_id = row["stay_id"]
        slot_e = int(row["slot_idx"])
        y = float(row["y_e"])

        stay_df = self.final_by_stay[stay_id]
        win = build_window_slots_df(stay_df, slot_e, self.K)

        x_ts = build_stay_tensor(win, self.means, self.stds,
                                  self.K, self.ts_vars, self.ts_counts)

        sta_row = self.static_df.loc[stay_id]
        x_static = encode_static(sta_row, self.age_mean, self.age_std, self.onehot_static)

        # Multi-label pathology: NaN → mask=0, y=0 placeholder / 그 외 → mask=1
        raw = np.array([row[c] for c in self.pathology_cols], dtype=np.float32)
        mask = (~np.isnan(raw)).astype(np.float32)
        y_multi = np.where(mask > 0, raw, 0.0).astype(np.float32)

        item = {
            "x_ts": x_ts,
            "x_static": x_static,
            "bin_ends": self.bin_ends,
            "y": torch.tensor(y, dtype=torch.float32),
            "y_multi": torch.from_numpy(y_multi),          # [K]
            "y_multi_mask": torch.from_numpy(mask),        # [K]
        }
        if self.mode == "teacher":
            jpg = dicom_to_jpg_path(row["subject_id"], row["study_id"], row["dicom_id"], self.cxr_root)
            img = Image.open(jpg).convert("RGB")
            proc = self.image_processor(images=img, return_tensors="pt")
            item["pixel_values"] = proc["pixel_values"].squeeze(0)  # [3, H, W]
        return item


def duett_kd_collate(batch: list[dict], mode: str) -> dict:
    """Collate that matches DuETT's `feats_to_input` list-of-tensors format."""
    x_ts_list = tuple(b["x_ts"] for b in batch)
    x_static_list = tuple(b["x_static"] for b in batch)
    bin_ends_list = tuple(b["bin_ends"] for b in batch)
    y = torch.stack([b["y"] for b in batch])
    out = {
        "x_ts": x_ts_list,
        "x_static": x_static_list,
        "bin_ends": bin_ends_list,
        "y": y,
    }
    if "y_multi" in batch[0]:
        out["y_multi"]      = torch.stack([b["y_multi"]      for b in batch])  # [B, K]
        out["y_multi_mask"] = torch.stack([b["y_multi_mask"] for b in batch])  # [B, K]
    if mode == "teacher":
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    return out


# =============================================================================
# One-shot builder
# =============================================================================
def build_datasets(cfg: AnchorConfig,
                   image_processor=None,
                   include_cxr: bool = True) -> dict:
    """Read raw files → anchor df → per-split DuettAnchorDataset.

    Returns:
        dict with keys: meta, ts_vars, ts_counts, anchor_df, splits (idx dict),
        datasets (per-split Dataset objects), pos_frac (train label mean).
    """
    print(f"[build] loading meta: {cfg.meta_path}")
    meta = load_duett_meta(cfg.meta_path)
    print(f"[build] loading static: {cfg.static_path}")
    static_df = load_static_df(cfg.static_path)
    print(f"[build] loading final_df: {cfg.final_df_path}")
    final_df = pd.read_feather(cfg.final_df_path)

    anchor_df, ts_vars, ts_counts = build_anchors(cfg, meta, final_df, static_df)
    splits = split_anchors(anchor_df, seed=cfg.split_seed)

    for name, idx in splits.items():
        pos = anchor_df.iloc[idx]["y_e"].mean() if len(idx) else 0.0
        print(f"[split] {name}: n={len(idx)}  pos_frac={pos:.4f}")

    stay_ids_used = set(anchor_df["stay_id"].unique())
    used_final = final_df[final_df["stay_id"].isin(stay_ids_used)]
    print(f"[build] filtering final_df: {len(final_df)} → {len(used_final)}")
    final_by_stay = {sid: g.sort_values("slot_idx")
                     for sid, g in used_final.groupby("stay_id")}
    del final_df, used_final

    mode = "teacher" if include_cxr else "student"
    datasets = {}
    for name, idx in splits.items():
        sub_df = anchor_df.iloc[idx].reset_index(drop=True)
        datasets[name] = DuettAnchorDataset(
            mode=mode,
            final_df_by_stay=final_by_stay,
            anchor_df=sub_df,
            static_df=static_df,
            meta=meta,
            ts_vars=ts_vars,
            ts_counts=ts_counts,
            cfg=cfg,
            image_processor=image_processor,
        )

    pos_frac = float(anchor_df.iloc[splits["train"]]["y_e"].mean())

    # Per-pathology pos_frac (train split, NaN 제외) — pos_weight 계산에 사용
    train_df = anchor_df.iloc[splits["train"]]
    pathology_pos_frac = []
    print(f"[build] per-pathology stats on train split:")
    for i, col in enumerate(cfg.pathology_labels):
        v = train_df[f"_y_{col}"]
        n_valid = int(v.notna().sum())
        n_pos = int((v == 1.0).sum())
        pf = (n_pos / n_valid) if n_valid > 0 else 0.0
        pathology_pos_frac.append(pf)
        print(f"  [{i}] {col:25s}  n_valid={n_valid:6d}  pos_frac={pf:.4f}")

    return {
        "meta": meta,
        "ts_vars": ts_vars,
        "ts_counts": ts_counts,
        "anchor_df": anchor_df,
        "splits": splits,
        "datasets": datasets,
        "pos_frac": pos_frac,
        "pathology_labels": tuple(cfg.pathology_labels),
        "pathology_pos_frac": pathology_pos_frac,
        "mode": mode,
    }
