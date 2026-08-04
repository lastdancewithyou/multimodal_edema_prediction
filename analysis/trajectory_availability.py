"""Audit whether the 24 h pre-CXR inputs contain genuine per-variable trajectories.

This is deliberately a data audit, not another predictive model.  A variable can
only contribute a learned slope/shape when it is observed at least twice inside
the window; three or more observed hours are a stricter notion of trajectory.

Example
-------
python analysis/trajectory_availability.py \
  --split train \
  --out_csv trajectory_availability_train.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training_duett.data_processing import (  # noqa: E402
    AnchorConfig,
    DEFAULT_PATHOLOGY_LABELS,
    build_datasets,
)
from training_duett.run import REPO_DEFAULTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Audit pre-CXR trajectory availability")
    p.add_argument("--final_df_path", default=REPO_DEFAULTS["final_df_path"])
    p.add_argument("--static_path", default=REPO_DEFAULTS["static_path"])
    p.add_argument("--duett_ckpt", default=REPO_DEFAULTS["duett_ckpt"])
    p.add_argument("--meta_path", default="",
                   help="default: <duett_ckpt directory>/meta_with_stats.pkl")
    p.add_argument("--label_col", default="label_edema")
    p.add_argument("--n_timesteps", type=int, default=24)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    p.add_argument("--max_samples", type=int, default=0,
                   help="0=all samples; positive values are useful for a quick check")
    p.add_argument("--out_csv", default="trajectory_availability.csv")
    return p.parse_args()


def _nanmedian(x: np.ndarray) -> float:
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


def audit_dataset(dataset, variable_names: list[str], n_timesteps: int,
                  max_samples: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    v = len(variable_names)
    obs_hours = np.zeros((n, v), dtype=np.int16)
    total_measurements = np.zeros((n, v), dtype=np.float32)
    recency_h = np.full((n, v), np.nan, dtype=np.float32)
    within_std = np.full((n, v), np.nan, dtype=np.float32)
    endpoint_change = np.full((n, v), np.nan, dtype=np.float32)

    for i in range(n):
        x = dataset[i]["x_ts"].numpy()
        values = x[:, :v]
        counts = x[:, v:]
        observed = counts > 0
        obs_hours[i] = observed.sum(axis=0)
        total_measurements[i] = counts.sum(axis=0)

        for j in np.flatnonzero(observed.any(axis=0)):
            idx = np.flatnonzero(observed[:, j])
            recency_h[i, j] = float(n_timesteps - idx[-1])
            vals = values[idx, j]
            if len(idx) >= 2:
                within_std[i, j] = float(np.std(vals, ddof=0))
                endpoint_change[i, j] = float(vals[-1] - vals[0])

    rows = []
    for j, name in enumerate(variable_names):
        k = obs_hours[:, j]
        rows.append({
            "variable": name,
            "n_samples": n,
            "any_observed_rate": float(np.mean(k >= 1)),
            "trajectory_2plus_rate": float(np.mean(k >= 2)),
            "trajectory_3plus_rate": float(np.mean(k >= 3)),
            "median_observed_hours": float(np.median(k)),
            "mean_observed_hours": float(np.mean(k)),
            "median_total_measurements": float(np.median(total_measurements[:, j])),
            "median_recency_h_if_observed": _nanmedian(recency_h[:, j]),
            "median_within_patient_std_if_2plus": _nanmedian(within_std[:, j]),
            "median_abs_endpoint_change_if_2plus": _nanmedian(np.abs(endpoint_change[:, j])),
        })
    per_variable = pd.DataFrame(rows).sort_values(
        ["trajectory_2plus_rate", "any_observed_rate"], ascending=False
    )

    per_sample = pd.DataFrame({
        "sample_index": np.arange(n),
        "n_variables_observed": (obs_hours >= 1).sum(axis=1),
        "n_variables_with_trajectory_2plus": (obs_hours >= 2).sum(axis=1),
        "n_variables_with_trajectory_3plus": (obs_hours >= 3).sum(axis=1),
        "n_observed_variable_hours": obs_hours.sum(axis=1),
    })
    return per_variable, per_sample


def _print_summary(per_variable: pd.DataFrame, per_sample: pd.DataFrame, split: str) -> None:
    print(f"\n=== 24 h trajectory availability: split={split}, n={len(per_sample)} ===")
    print("Definition: >=2 observed hours can express a change; >=3 can express a shape.")
    print(
        "Per patient median: "
        f"observed variables={per_sample['n_variables_observed'].median():.0f}, "
        f">=2h variables={per_sample['n_variables_with_trajectory_2plus'].median():.0f}, "
        f">=3h variables={per_sample['n_variables_with_trajectory_3plus'].median():.0f}"
    )

    cols = [
        "variable", "any_observed_rate", "trajectory_2plus_rate",
        "trajectory_3plus_rate", "median_observed_hours",
        "median_recency_h_if_observed",
    ]
    print("\nTop variables with usable trajectories")
    print(per_variable[cols].head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nVariables with little/no usable trajectory")
    print(per_variable[cols].tail(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    median_two = float(per_sample["n_variables_with_trajectory_2plus"].median())
    if median_two < 3:
        verdict = "VERY SPARSE: most inputs contain levels/missingness, not multivariable trajectories."
    elif median_two < 8:
        verdict = "SPARSE: trajectory modeling is plausible for only a small variable subset."
    else:
        verdict = "TRAJECTORY-RICH: an encoder that preserves variable-wise temporal structure is justified."
    print(f"\nVerdict: {verdict}")


def main() -> None:
    args = parse_args()
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
        pathology_labels=DEFAULT_PATHOLOGY_LABELS,
    )
    bundle = build_datasets(cfg, image_processor=None, include_cxr=False)
    if args.split == "all":
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset([bundle["datasets"][s] for s in ("train", "val", "test")])
    else:
        dataset = bundle["datasets"][args.split]

    per_variable, per_sample = audit_dataset(
        dataset, list(bundle["ts_vars"]), args.n_timesteps, args.max_samples
    )
    _print_summary(per_variable, per_sample, args.split)
    per_variable.to_csv(args.out_csv, index=False)
    sample_path = os.path.splitext(args.out_csv)[0] + "_per_sample.csv"
    per_sample.to_csv(sample_path, index=False)
    print(f"\nSaved: {args.out_csv}")
    print(f"Saved: {sample_path}")


if __name__ == "__main__":
    main()
