"""Test whether raw pre-CXR trajectories add information beyond the image.

This is a diagnostic probe, not a replacement fusion model.  It uses exactly
the same [CXR - K hours, CXR) windows as ``DuettAnchorDataset`` and summarizes
each dynamic variable before any DuETT embedding is computed.

For every variable the summaries are split into three scientifically useful
blocks:

    level
        last, count-weighted mean/std, min, max
    trajectory
        last-first delta, 24 h slope, recent-window slope,
        recent mean - earlier mean
    observation
        observed-hour fraction, log measurement count, measurement recency,
        recent observed-hour fraction, recent log measurement count

The script calibrates the checkpoint image logit on a probe-training split,
freezes that score, and fits only an additive raw-TS correction.  An exact
zero-correction candidate is always included, so an unhelpful TS block falls
back to the calibrated image predictor instead of damaging it. Hyperparameters
are selected only by cross-validation inside the probe-training split; the
test split is evaluated once. No Youden threshold is used.

Example
-------
python analysis/raw_trajectory_conditional_probe.py \
    --ckpt <run>/best.pt \
    --outdir analysis/raw_trajectory_probe_<run> \
    --labels label_edema

Interpretation
--------------
* raw positive, DuETT probe negative: the representation/readout is the likely
  bottleneck;
* raw and DuETT positive, trained fusion negative: fusion optimization is the
  likely bottleneck;
* raw negative: the current cohort/window has little easily decodable
  image-conditional signal (it is not proof that no possible signal exists).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.special import expit


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_duett.data_processing import build_window_slots_df  # noqa: E402


LEVEL_STATS = ("last", "mean", "std", "min", "max")
TRAJECTORY_STATS = ("delta", "slope24", "slope_recent", "recent_shift")
OBSERVATION_STATS = (
    "observed_fraction",
    "log_total_count",
    "time_since_last",
    "recent_observed_fraction",
    "log_recent_count",
)
DEFAULT_BLOCKS = ("level", "trajectory", "observation", "physiologic", "all")
ALLOWED_MODELS = ("offset_logistic", "logistic")
DEFAULT_MODELS = ("offset_logistic",)


def _resolve_label_indices(
    requested: str, pathology_labels: Sequence[str]
) -> tuple[int, ...]:
    if requested.strip().lower() == "all":
        return tuple(range(len(pathology_labels)))
    normalized = {name.lower(): i for i, name in enumerate(pathology_labels)}
    for i, name in enumerate(pathology_labels):
        normalized[name.lower().removeprefix("label_")] = i
    names = [item.strip().lower() for item in requested.split(",") if item.strip()]
    unknown = [name for name in names if name not in normalized]
    if unknown:
        raise ValueError(
            f"Unknown labels {unknown}; checkpoint labels={list(pathology_labels)}"
        )
    indices = tuple(normalized[name] for name in names)
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate labels requested: {requested!r}")
    return indices


def _safe_metrics(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    metrics = {
        "bce": float(log_loss(y, probability, labels=[0, 1])),
        "auroc": float("nan"),
        "auprc": float("nan"),
    }
    if np.unique(y).size >= 2:
        metrics["auroc"] = float(roc_auc_score(y, probability))
        metrics["auprc"] = float(average_precision_score(y, probability))
    return metrics


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _image_risk_bins(image_logit: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1:
        return np.zeros(len(image_logit), dtype=np.int64)
    edges = np.unique(
        np.quantile(image_logit, np.linspace(0.0, 1.0, int(n_bins) + 1))
    )
    if len(edges) <= 2:
        return np.zeros(len(image_logit), dtype=np.int64)
    return np.digitize(image_logit, edges[1:-1], right=True).astype(np.int64)


def _conditional_shuffle_indices(
    bins: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    shuffled = np.arange(len(bins))
    for value in np.unique(bins):
        members = np.flatnonzero(bins == value)
        if len(members) > 1:
            shuffled[members] = rng.permutation(members)
    return shuffled


def _slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_") or "label"


def _fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "--"
    prefix = "+" if signed else ""
    return f"{value:{prefix}.{digits}f}"


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Probe image-conditional information in raw pre-CXR trajectories"
    )
    parser.add_argument("--ckpt", required=True, help="dual/dual_patch best.pt")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--labels",
        default="label_edema",
        help="comma-separated checkpoint labels, with or without 'label_'",
    )
    parser.add_argument("--probe_train_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--blocks",
        default=",".join(DEFAULT_BLOCKS),
        help=f"subset of {DEFAULT_BLOCKS}",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=(
            "fixed-image correction model. 'logistic' is retained as an alias "
            "for 'offset_logistic'; joint logistic/HGB are intentionally disabled"
        ),
    )
    parser.add_argument(
        "--recent_hours",
        type=int,
        default=6,
        help="recent window used for slope/shift/count summaries",
    )
    parser.add_argument(
        "--c_grid",
        default="0.001,0.01,0.1,1,10",
        help="image-calibration inverse-L2 grid selected by inner CV",
    )
    parser.add_argument(
        "--correction_l2_grid",
        default="0.0001,0.001,0.01,0.1,1,10,100",
        help="L2 strengths for TS-only offset correction; exact null is added automatically",
    )
    parser.add_argument(
        "--null_tolerance",
        type=float,
        default=5e-4,
        help=(
            "select exact zero correction when its inner-CV BCE is within this "
            "amount of the best non-null correction"
        ),
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--perm_repeats", type=int, default=100)
    parser.add_argument("--perm_bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_choice_list(raw: str, allowed: Sequence[str], name: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {name}={unknown}; allowed={list(allowed)}")
    if not values:
        raise ValueError(f"At least one {name} must be selected")
    return tuple(dict.fromkeys(values))


def _parse_float_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("--c_grid must contain positive comma-separated values")
    return values


@torch.no_grad()
def _gather_image_predictions(teacher, loader, device: torch.device) -> Dict[str, np.ndarray]:
    from training_duett.engine import _move_lists

    image_logits, labels, masks = [], [], []
    teacher.eval()
    for batch in loader:
        moved = _move_lists(batch, device)
        output = teacher(
            moved["x_ts"],
            moved["x_static"],
            moved["bin_ends"],
            moved["pixel_values"],
        )
        if "img_logits" not in output:
            raise RuntimeError("Teacher output does not contain img_logits")
        image_logits.append(output["img_logits"].detach().float().cpu())
        labels.append(moved["y_multi"].detach().float().cpu())
        masks.append(moved["y_multi_mask"].detach().float().cpu())
    if not image_logits:
        raise RuntimeError("Probe loader yielded no batches")
    return {
        "img": torch.cat(image_logits).numpy(),
        "y": torch.cat(labels).numpy(),
        "mask": torch.cat(masks).numpy(),
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    mean = _weighted_mean(values, weights)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return float(np.std(values, ddof=0))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return float(math.sqrt(max(variance, 0.0)))


def _weighted_slope(times: np.ndarray, values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted least-squares slope in raw value units per hour."""
    if values.size < 2 or np.unique(times).size < 2:
        return float("nan")
    weights = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        weights = np.ones_like(values, dtype=np.float64)
    t_mean = float(np.average(times, weights=weights))
    y_mean = float(np.average(values, weights=weights))
    centered_t = times - t_mean
    denominator = float(np.sum(weights * centered_t**2))
    if denominator <= 0:
        return float("nan")
    numerator = float(np.sum(weights * centered_t * (values - y_mean)))
    return numerator / denominator


def _summarize_one_variable(
    window: pd.DataFrame,
    variable: str,
    count_column: str,
    n_timesteps: int,
    recent_hours: int,
) -> tuple[list[float], list[float], list[float]]:
    values_all = pd.to_numeric(window[variable], errors="coerce").to_numpy(
        dtype=np.float64
    )
    counts_all = pd.to_numeric(window[count_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    times_all = pd.to_numeric(window["slot_idx"], errors="coerce").to_numpy(
        dtype=np.float64
    )

    observed = np.isfinite(counts_all) & (counts_all > 0) & np.isfinite(times_all)
    valid = observed & np.isfinite(values_all)
    times = times_all[valid]
    values = values_all[valid]
    weights = counts_all[valid]

    if values.size:
        order = np.argsort(times, kind="stable")
        times, values, weights = times[order], values[order], weights[order]
        last = float(values[-1])
        mean = _weighted_mean(values, weights)
        std = _weighted_std(values, weights)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        delta = float(values[-1] - values[0]) if values.size >= 2 else float("nan")
        slope24 = _weighted_slope(times, values, weights)
    else:
        last = mean = std = minimum = maximum = float("nan")
        delta = slope24 = float("nan")

    recent_start = max(int(n_timesteps) - int(recent_hours), 0)
    recent_valid = valid & (times_all >= recent_start)
    earlier_valid = valid & (times_all < recent_start)
    recent_times = times_all[recent_valid]
    recent_values = values_all[recent_valid]
    recent_weights = counts_all[recent_valid]
    slope_recent = _weighted_slope(recent_times, recent_values, recent_weights)
    if recent_values.size and earlier_valid.any():
        recent_shift = _weighted_mean(recent_values, recent_weights) - _weighted_mean(
            values_all[earlier_valid], counts_all[earlier_valid]
        )
    else:
        recent_shift = float("nan")

    observed_times = times_all[observed]
    observed_fraction = float(np.unique(observed_times).size / n_timesteps)
    total_count = float(np.sum(counts_all[observed])) if observed.any() else 0.0
    if observed_times.size:
        time_since_last = float(max((n_timesteps - 1) - np.max(observed_times), 0.0))
    else:
        time_since_last = float(n_timesteps)
    recent_observed = observed & (times_all >= recent_start)
    recent_denominator = max(n_timesteps - recent_start, 1)
    recent_observed_fraction = float(
        np.unique(times_all[recent_observed]).size / recent_denominator
    )
    recent_count = (
        float(np.sum(counts_all[recent_observed])) if recent_observed.any() else 0.0
    )

    level = [last, mean, std, minimum, maximum]
    trajectory = [delta, slope24, slope_recent, recent_shift]
    observation = [
        observed_fraction,
        float(np.log1p(max(total_count, 0.0))),
        time_since_last,
        recent_observed_fraction,
        float(np.log1p(max(recent_count, 0.0))),
    ]
    return level, trajectory, observation


def _raw_summary_blocks(
    dataset,
    ts_vars: Sequence[str],
    ts_counts: Sequence[str],
    recent_hours: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, tuple[str, ...]], np.ndarray]:
    """Extract aligned raw summaries without loading images or running DuETT."""
    if len(ts_vars) != len(ts_counts):
        raise ValueError("ts_vars and ts_counts must have the same length")
    n_timesteps = int(dataset.K)
    if recent_hours < 1 or recent_hours > n_timesteps:
        raise ValueError(
            f"recent_hours must be within [1, {n_timesteps}], got {recent_hours}"
        )

    names_level = tuple(f"{var}__{stat}" for var in ts_vars for stat in LEVEL_STATS)
    names_trajectory = tuple(
        f"{var}__{stat}" for var in ts_vars for stat in TRAJECTORY_STATS
    )
    names_observation = tuple(
        f"{var}__{stat}" for var in ts_vars for stat in OBSERVATION_STATS
    )

    rows_level, rows_trajectory, rows_observation, subjects = [], [], [], []
    anchors = dataset.anchor_df
    for index, row in anchors.iterrows():
        stay_id = row["stay_id"]
        slot_e = int(row["slot_idx"])
        window = build_window_slots_df(
            dataset.final_by_stay[stay_id], slot_e, n_timesteps
        )
        level_row, trajectory_row, observation_row = [], [], []
        for variable, count_column in zip(ts_vars, ts_counts):
            level, trajectory, observation = _summarize_one_variable(
                window,
                variable,
                count_column,
                n_timesteps,
                recent_hours,
            )
            level_row.extend(level)
            trajectory_row.extend(trajectory)
            observation_row.extend(observation)
        rows_level.append(level_row)
        rows_trajectory.append(trajectory_row)
        rows_observation.append(observation_row)
        subjects.append(row["subject_id"])
        if (index + 1) % 500 == 0 or index + 1 == len(anchors):
            print(f"  raw summaries: {index + 1}/{len(anchors)}")

    level = np.asarray(rows_level, dtype=np.float64)
    trajectory = np.asarray(rows_trajectory, dtype=np.float64)
    observation = np.asarray(rows_observation, dtype=np.float64)
    physiologic = np.column_stack([level, trajectory])
    all_features = np.column_stack([physiologic, observation])
    blocks = {
        "level": level,
        "trajectory": trajectory,
        "observation": observation,
        "physiologic": physiologic,
        "all": all_features,
    }
    names = {
        "level": names_level,
        "trajectory": names_trajectory,
        "observation": names_observation,
        "physiologic": names_level + names_trajectory,
        "all": names_level + names_trajectory + names_observation,
    }
    for block, matrix in blocks.items():
        if matrix.shape != (len(anchors), len(names[block])):
            raise RuntimeError(
                f"Raw block alignment error for {block}: matrix={matrix.shape}, "
                f"names={len(names[block])}, anchors={len(anchors)}"
            )
    return blocks, names, np.asarray(subjects)


def _design_frame(
    image_logit: np.ndarray,
    raw_features: np.ndarray | None,
    raw_names: Sequence[str] = (),
) -> pd.DataFrame:
    image_logit = np.asarray(image_logit, dtype=np.float64).reshape(-1)
    data: Dict[str, np.ndarray] = {"image_logit": image_logit}
    if raw_features is not None:
        raw_features = np.asarray(raw_features, dtype=np.float64)
        if raw_features.shape != (len(image_logit), len(raw_names)):
            raise ValueError(
                f"Feature/name mismatch: X={raw_features.shape}, "
                f"n={len(image_logit)}, names={len(raw_names)}"
            )
        data.update({name: raw_features[:, i] for i, name in enumerate(raw_names)})
    return pd.DataFrame(data)


def _cv_splitter(y: np.ndarray, requested_folds: int, seed: int) -> StratifiedKFold:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
    folds = min(int(requested_folds), int(counts.min()))
    if folds < 2:
        raise ValueError(f"Not enough samples in both classes for CV: counts={counts}")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def _fit_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    c_grid: Sequence[float],
    cv_folds: int,
    max_iter: int,
    n_jobs: int,
    seed: int,
) -> GridSearchCV:
    cv = _cv_splitter(y_train, cv_folds, seed)
    if model_name != "logistic":
        raise ValueError(f"Unknown model_name={model_name!r}")
    estimator = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight=None,
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )
    parameter_grid = {"model__C": list(c_grid)}

    search = GridSearchCV(
        estimator,
        parameter_grid,
        scoring="neg_log_loss",
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    return search.fit(x_train, y_train)


def _bce_from_scores(y: np.ndarray, score: np.ndarray) -> float:
    """Stable mean binary cross-entropy from logits."""
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, score) - y * score))


def _fit_offset_weights(
    features: np.ndarray,
    y: np.ndarray,
    fixed_offset: np.ndarray,
    l2_strength: float,
    max_iter: int,
) -> np.ndarray:
    """Fit only w in sigmoid(fixed_offset + X @ w)."""
    features = np.asarray(features, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fixed_offset = np.asarray(fixed_offset, dtype=np.float64)
    n_samples, n_features = features.shape
    if n_features == 0:
        return np.zeros(0, dtype=np.float64)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        score = fixed_offset + features @ weights
        loss = _bce_from_scores(y, score)
        loss += 0.5 * float(l2_strength) * float(weights @ weights)
        gradient = features.T @ (expit(score) - y) / n_samples
        gradient += float(l2_strength) * weights
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(n_features, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(
            "Offset correction optimization failed: "
            f"status={result.status}, message={result.message}"
        )
    return np.asarray(result.x, dtype=np.float64)


@dataclass
class OffsetCorrectionModel:
    """Preprocessed TS correction added to an externally fixed image score."""

    imputer: SimpleImputer
    scaler: StandardScaler
    weights: np.ndarray
    input_names: tuple[str, ...]
    transformed_names: tuple[str, ...]
    selected_l2: float | None
    cv_bce: float
    cv_results: Mapping[str, float]

    @property
    def null_selected(self) -> bool:
        return self.selected_l2 is None

    @property
    def best_params_(self) -> Dict[str, object]:
        if self.null_selected:
            return {"correction": "null", "correction_l2": None}
        return {
            "correction": "offset_logistic",
            "correction_l2": float(self.selected_l2),
        }

    def _transform(self, raw_features: pd.DataFrame) -> np.ndarray:
        transformed = self.imputer.transform(raw_features)
        return np.asarray(self.scaler.transform(transformed), dtype=np.float64)

    def decision_function(
        self, fixed_image_score: np.ndarray, raw_features: pd.DataFrame
    ) -> np.ndarray:
        fixed_image_score = np.asarray(fixed_image_score, dtype=np.float64)
        correction = self._transform(raw_features) @ self.weights
        return fixed_image_score + correction

    def predict(
        self, fixed_image_score: np.ndarray, raw_features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        score = self.decision_function(fixed_image_score, raw_features)
        return expit(score).astype(np.float64), score

    def standardized_coefficients(self) -> list[tuple[str, float]]:
        return sorted(
            [
                (name, float(coefficient))
                for name, coefficient in zip(self.transformed_names, self.weights)
            ],
            key=lambda item: abs(item[1]),
            reverse=True,
        )


def _fit_offset_correction(
    raw_train: pd.DataFrame,
    y_train: np.ndarray,
    fixed_image_score: np.ndarray,
    l2_grid: Sequence[float],
    cv_folds: int,
    max_iter: int,
    null_tolerance: float,
    seed: int,
) -> OffsetCorrectionModel:
    """Inner-CV selection with an exact zero-correction candidate.

    Image scores are never re-estimated here and are never regularized.  The
    null candidate therefore reproduces the calibrated image predictor exactly.
    """
    if null_tolerance < 0:
        raise ValueError("null_tolerance must be non-negative")
    cv = _cv_splitter(y_train, cv_folds, seed)
    candidate_names = ["null"] + [f"l2={value:g}" for value in l2_grid]
    fold_losses: Dict[str, list[float]] = {name: [] for name in candidate_names}

    for train_index, valid_index in cv.split(raw_train, y_train):
        fold_train = raw_train.iloc[train_index]
        fold_valid = raw_train.iloc[valid_index]
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        x_train = imputer.fit_transform(fold_train)
        x_valid = imputer.transform(fold_valid)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        y_fold = y_train[train_index]
        offset_train = fixed_image_score[train_index]
        offset_valid = fixed_image_score[valid_index]
        fold_losses["null"].append(
            _bce_from_scores(y_train[valid_index], offset_valid)
        )
        for l2_strength in l2_grid:
            weights = _fit_offset_weights(
                x_train,
                y_fold,
                offset_train,
                l2_strength,
                max_iter,
            )
            fold_score = offset_valid + x_valid @ weights
            fold_losses[f"l2={l2_strength:g}"].append(
                _bce_from_scores(y_train[valid_index], fold_score)
            )

    mean_losses = {
        name: float(np.mean(values)) for name, values in fold_losses.items()
    }
    best_non_null = min(
        (name for name in candidate_names if name != "null"),
        key=mean_losses.__getitem__,
    )
    if mean_losses["null"] <= mean_losses[best_non_null] + null_tolerance:
        selected_name = "null"
        selected_l2 = None
    else:
        selected_name = best_non_null
        selected_l2 = float(selected_name.split("=", 1)[1])

    final_imputer = SimpleImputer(strategy="median", add_indicator=True)
    transformed = final_imputer.fit_transform(raw_train)
    final_scaler = StandardScaler()
    transformed = final_scaler.fit_transform(transformed)
    transformed_names = tuple(
        str(name)
        for name in final_imputer.get_feature_names_out(
            np.asarray(raw_train.columns, dtype=object)
        )
    )
    if selected_l2 is None:
        weights = np.zeros(transformed.shape[1], dtype=np.float64)
    else:
        weights = _fit_offset_weights(
            transformed,
            y_train,
            fixed_image_score,
            selected_l2,
            max_iter,
        )
    return OffsetCorrectionModel(
        imputer=final_imputer,
        scaler=final_scaler,
        weights=weights,
        input_names=tuple(str(name) for name in raw_train.columns),
        transformed_names=transformed_names,
        selected_l2=selected_l2,
        cv_bce=mean_losses[selected_name],
        cv_results=mean_losses,
    )


def _predict(model: GridSearchCV, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probability = model.predict_proba(features)[:, 1].astype(np.float64)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(features), dtype=np.float64)
    else:
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        score = np.log(clipped / (1 - clipped))
    return probability, score


def _cluster_bootstrap_differences(
    y: np.ndarray,
    base_probability: np.ndarray,
    probe_probability: np.ndarray,
    subject_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    """Paired patient-cluster bootstrap; positive values favor the probe."""
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    unique_subjects = np.unique(subject_ids)
    members = {subject: np.flatnonzero(subject_ids == subject) for subject in unique_subjects}
    rng = np.random.default_rng(seed)
    samples = {"bce_gain": [], "auroc_gain": [], "auprc_gain": []}
    for _ in range(max(int(n_bootstrap), 0)):
        drawn = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        index = np.concatenate([members[subject] for subject in drawn])
        y_b = y[index]
        base_b = base_probability[index]
        probe_b = probe_probability[index]
        samples["bce_gain"].append(
            log_loss(y_b, base_b, labels=[0, 1])
            - log_loss(y_b, probe_b, labels=[0, 1])
        )
        if np.unique(y_b).size >= 2:
            samples["auroc_gain"].append(
                roc_auc_score(y_b, probe_b) - roc_auc_score(y_b, base_b)
            )
            samples["auprc_gain"].append(
                average_precision_score(y_b, probe_b)
                - average_precision_score(y_b, base_b)
            )
    output: Dict[str, float] = {}
    for metric, values in samples.items():
        if values:
            low, high = np.percentile(np.asarray(values), [2.5, 97.5])
        else:
            low = high = float("nan")
        output[f"{metric}_ci_low"] = float(low)
        output[f"{metric}_ci_high"] = float(high)
    return output


def _conditional_permutation_offset(
    model: OffsetCorrectionModel,
    y: np.ndarray,
    image_logit: np.ndarray,
    fixed_image_score: np.ndarray,
    raw_features: np.ndarray,
    raw_names: Sequence[str],
    repeats: int,
    n_bins: int,
    seed: int,
) -> Dict[str, float]:
    """Shuffle the whole raw-TS feature row among similar image-risk samples."""
    bins = _image_risk_bins(image_logit, n_bins)
    rng = np.random.default_rng(seed)
    samples = {"bce": [], "auroc": [], "auprc": []}
    for _ in range(max(int(repeats), 0)):
        shuffled_index = _conditional_shuffle_indices(bins, rng)
        shuffled = pd.DataFrame(
            raw_features[shuffled_index], columns=list(raw_names)
        )
        probability, _ = model.predict(fixed_image_score, shuffled)
        metrics = _safe_metrics(y, probability)
        for metric in samples:
            samples[metric].append(metrics[metric])
    output: Dict[str, float] = {}
    for metric, values in samples.items():
        finite = np.asarray([value for value in values if np.isfinite(value)])
        output[f"perm_{metric}_mean"] = (
            float(finite.mean()) if finite.size else float("nan")
        )
        if finite.size:
            low, high = np.percentile(finite, [2.5, 97.5])
        else:
            low = high = float("nan")
        output[f"perm_{metric}_low"] = float(low)
        output[f"perm_{metric}_high"] = float(high)
    return output


def main() -> None:
    args = parse_args()
    if args.probe_train_split == args.test_split:
        raise SystemExit("probe_train_split and test_split must differ")
    blocks_selected = _parse_choice_list(args.blocks, DEFAULT_BLOCKS, "blocks")
    requested_models = _parse_choice_list(args.models, ALLOWED_MODELS, "models")
    # Backward-compatible alias: old commands used ``--models logistic``.
    models_selected = tuple(
        dict.fromkeys(
            "offset_logistic" if name == "logistic" else name
            for name in requested_models
        )
    )
    c_grid = _parse_float_grid(args.c_grid)
    correction_l2_grid = _parse_float_grid(args.correction_l2_grid)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Keep model-only dependencies (transformers, torchmetrics, x-transformers)
    # out of module import so feature/statistical utilities remain testable alone.
    from analysis.visualize_pathology import _loader, load_teacher

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[raw-trajectory-probe] device={device}")
    teacher, bundle, _, _, pathology_labels, mode = load_teacher(args.ckpt, device)
    datasets = bundle["datasets"]
    for split in (args.probe_train_split, args.test_split):
        if split not in datasets:
            raise SystemExit(f"Unknown split={split!r}; available={list(datasets)}")
    selected_labels = _resolve_label_indices(args.labels, pathology_labels)

    print(f"[raw-trajectory-probe] extracting raw {args.probe_train_split} summaries")
    train_blocks, block_names, train_subjects = _raw_summary_blocks(
        datasets[args.probe_train_split],
        bundle["ts_vars"],
        bundle["ts_counts"],
        args.recent_hours,
    )
    print(f"[raw-trajectory-probe] extracting raw {args.test_split} summaries")
    test_blocks, test_names, test_subjects = _raw_summary_blocks(
        datasets[args.test_split],
        bundle["ts_vars"],
        bundle["ts_counts"],
        args.recent_hours,
    )
    if block_names != test_names:
        raise RuntimeError("Train/test raw feature schemas differ")

    print(f"[raw-trajectory-probe] running checkpoint on {args.probe_train_split}")
    train_predictions = _gather_image_predictions(
        teacher,
        _loader(datasets[args.probe_train_split], args.batch_size, args.num_workers),
        device,
    )
    print(f"[raw-trajectory-probe] running checkpoint on {args.test_split}")
    test_predictions = _gather_image_predictions(
        teacher,
        _loader(datasets[args.test_split], args.batch_size, args.num_workers),
        device,
    )
    if len(train_subjects) != len(train_predictions["y"]):
        raise RuntimeError("Probe-train raw/prediction order is misaligned")
    if len(test_subjects) != len(test_predictions["y"]):
        raise RuntimeError("Test raw/prediction order is misaligned")

    rows: list[Mapping[str, object]] = []
    coefficient_rows: list[Mapping[str, object]] = []
    prediction_archive: Dict[str, np.ndarray] = {
        "test_img_logits": test_predictions["img"],
        "test_y": test_predictions["y"],
        "test_mask": test_predictions["mask"],
        "test_subject_ids": test_subjects,
    }
    summary: Dict[str, object] = {
        "checkpoint": os.path.abspath(args.ckpt),
        "mode": mode,
        "probe_train_split": args.probe_train_split,
        "test_split": args.test_split,
        "configuration": vars(args),
        "ts_variables": list(bundle["ts_vars"]),
        "feature_counts": {block: len(names) for block, names in block_names.items()},
        "labels": {},
    }

    for label_index in selected_labels:
        label = pathology_labels[label_index]
        train_mask = train_predictions["mask"][:, label_index].astype(bool)
        test_mask = test_predictions["mask"][:, label_index].astype(bool)
        y_train = train_predictions["y"][train_mask, label_index].astype(np.int64)
        y_test = test_predictions["y"][test_mask, label_index].astype(np.int64)
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            print(f"[raw-trajectory-probe] skip {label}: one split has one class")
            continue
        image_train = train_predictions["img"][train_mask, label_index]
        image_test = test_predictions["img"][test_mask, label_index]
        subject_test = test_subjects[test_mask]

        base_train = _design_frame(image_train, None)
        base_test = _design_frame(image_test, None)
        base_model = _fit_model(
            "logistic",
            base_train,
            y_train,
            c_grid,
            args.cv_folds,
            args.max_iter,
            args.n_jobs,
            args.seed + label_index * 1000,
        )
        _, base_train_score = _predict(base_model, base_train)
        base_probability, base_score = _predict(base_model, base_test)
        base_metrics = _safe_metrics(y_test, base_probability)
        print(
            f"\n[{label}] n={len(y_test)} pos={int(y_test.sum())} "
            f"image-cal BCE={base_metrics['bce']:.5f} "
            f"AUROC={base_metrics['auroc']:.4f} "
            f"AUPRC={base_metrics['auprc']:.4f}"
        )
        print(
            "model     block         choice      BCE   BCEgain [cluster 95% CI]   "
            "AUROC  dROC    AUPRC  dPRC   corr_r  perm_dBCE evidence"
        )
        print("-" * 135)

        label_summary: Dict[str, object] = {
            "n_test": int(len(y_test)),
            "n_positive": int(y_test.sum()),
            "prevalence": float(y_test.mean()),
            "image_cal": {
                **base_metrics,
                "best_params": base_model.best_params_,
                "inner_cv_bce": float(-base_model.best_score_),
            },
            "probes": {},
        }

        probe_offset = 0
        for model_name in models_selected:
            for block in blocks_selected:
                names = block_names[block]
                raw_train = train_blocks[block][train_mask]
                raw_test = test_blocks[block][test_mask]
                train_frame = pd.DataFrame(raw_train, columns=list(names))
                test_frame = pd.DataFrame(raw_test, columns=list(names))
                fitted = _fit_offset_correction(
                    train_frame,
                    y_train,
                    base_train_score,
                    correction_l2_grid,
                    args.cv_folds,
                    args.max_iter,
                    args.null_tolerance,
                    args.seed + label_index * 1000 + probe_offset + 1,
                )
                probability, score = fitted.predict(base_score, test_frame)
                metrics = _safe_metrics(y_test, probability)
                gains = {
                    "bce_gain": base_metrics["bce"] - metrics["bce"],
                    "auroc_gain": metrics["auroc"] - base_metrics["auroc"],
                    "auprc_gain": metrics["auprc"] - base_metrics["auprc"],
                }
                confidence = _cluster_bootstrap_differences(
                    y_test,
                    base_probability,
                    probability,
                    subject_test,
                    args.bootstrap,
                    args.seed + label_index * 10000 + probe_offset,
                )
                corr_residual = _pearson(
                    score - base_score,
                    y_test.astype(np.float64) - base_probability,
                )
                permutation = _conditional_permutation_offset(
                    fitted,
                    y_test,
                    image_test,
                    base_score,
                    raw_test,
                    names,
                    args.perm_repeats,
                    args.perm_bins,
                    args.seed + label_index * 100000 + probe_offset,
                )
                perm_bce_increase = permutation["perm_bce_mean"] - metrics["bce"]
                perm_auroc_drop = metrics["auroc"] - permutation["perm_auroc_mean"]
                if (
                    gains["bce_gain"] > 0
                    and confidence["bce_gain_ci_low"] > 0
                    and perm_bce_increase > 0
                ):
                    evidence = "supported"
                elif gains["bce_gain"] > 0:
                    evidence = "suggestive"
                else:
                    evidence = "not_detected"

                probe_name = f"{model_name}_{block}"
                row = {
                    "label": label,
                    "model": model_name,
                    "block": block,
                    "n_test": int(len(y_test)),
                    "n_positive": int(y_test.sum()),
                    "prevalence": float(y_test.mean()),
                    "n_input_features": int(train_frame.shape[1]),
                    "image_cal_bce": base_metrics["bce"],
                    "image_cal_auroc": base_metrics["auroc"],
                    "image_cal_auprc": base_metrics["auprc"],
                    "probe_bce": metrics["bce"],
                    "probe_auroc": metrics["auroc"],
                    "probe_auprc": metrics["auprc"],
                    **gains,
                    **confidence,
                    "corr_residual": corr_residual,
                    **permutation,
                    "perm_bce_increase": perm_bce_increase,
                    "perm_auroc_drop": perm_auroc_drop,
                    "inner_cv_bce": fitted.cv_bce,
                    "best_params": json.dumps(fitted.best_params_, sort_keys=True),
                    "correction_cv_results": json.dumps(
                        fitted.cv_results, sort_keys=True
                    ),
                    "null_selected": fitted.null_selected,
                    "evidence": evidence,
                }
                rows.append(row)
                label_summary["probes"][probe_name] = row
                prediction_archive[
                    f"{_slug(label)}_{probe_name}_probability"
                ] = probability.astype(np.float32)

                coefficients = fitted.standardized_coefficients()
                for rank, (feature, coefficient) in enumerate(coefficients, start=1):
                    coefficient_rows.append(
                        {
                            "label": label,
                            "block": block,
                            "rank_abs": rank,
                            "feature": feature,
                            "standardized_coefficient": coefficient,
                            "null_selected": fitted.null_selected,
                            "selected_l2": fitted.selected_l2,
                        }
                    )

                choice = (
                    "null"
                    if fitted.null_selected
                    else f"l2={fitted.selected_l2:g}"
                )
                print(
                    f"{'offset':<9} {block:<13} {choice:<10} "
                    f"{metrics['bce']:.5f} "
                    f"{gains['bce_gain']:+.5f} "
                    f"[{confidence['bce_gain_ci_low']:+.5f},"
                    f"{confidence['bce_gain_ci_high']:+.5f}] "
                    f"{metrics['auroc']:.4f} {gains['auroc_gain']:+.4f} "
                    f"{metrics['auprc']:.4f} {gains['auprc_gain']:+.4f} "
                    f"{_fmt(corr_residual, 3, signed=True):>7} "
                    f"{_fmt(perm_bce_increase, 5, signed=True):>10} "
                    f"{evidence}"
                )
                probe_offset += 1

        summary["labels"][label] = label_summary

    _write_csv(rows, outdir / "raw_trajectory_probe_results.csv")
    _write_csv(coefficient_rows, outdir / "logistic_coefficients.csv")
    with (outdir / "raw_trajectory_probe_summary.json").open("w") as handle:
        json.dump(_json_ready(summary), handle, indent=2, sort_keys=True)
    np.savez_compressed(outdir / "raw_trajectory_probe_predictions.npz", **prediction_archive)
    print(f"\n[raw-trajectory-probe] wrote outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
