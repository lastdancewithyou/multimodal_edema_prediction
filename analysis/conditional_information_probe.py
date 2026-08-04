"""Threshold-free conditional-information probes for a trained dual teacher.

This script asks a different question from ``complementarity.py``:

    Does the temporal branch contain information about the label *after* the
    image prediction is already known?

For each pathology it fits four small post-hoc models on a probe-training
split and evaluates them on an untouched test split:

    image_cal          sigmoid(a * image_logit + b)
    logit_add          sigmoid(a * image_logit + b * ts_logit + c)
    logit_interaction  logit_add + d * image_logit * ts_logit
    token_linear       sigmoid(a * image_logit + w^T temporal_token + b)

All comparisons use continuous probabilities; no classification threshold or
Youden index is involved.  The image-calibration model is the control, so an
apparent gain caused only by rescaling the image logit is not attributed to TS.

The report additionally contains:

* paired bootstrap confidence intervals for test-set metric differences;
* correlation between the learned logit correction and the image residual;
* conditional permutation, where TS is shuffled only among patients with
  similar image logits (image-risk quantile bins).

A positive held-out BCE gain with a confidence interval above zero, followed
by loss of that gain after conditional permutation, is evidence that the
*current TS representation* contains decodable image-conditional information.
A negative result does not prove that the raw TS data contain no such signal.

Example:
    python analysis/conditional_information_probe.py \
        --ckpt <run>/best.pt \
        --outdir analysis/conditional_probe_<run> \
        --labels label_edema
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.visualize_pathology import _loader, load_teacher  # noqa: E402
from training_duett.engine import _move_lists  # noqa: E402


PROBE_NAMES = ("logit_add", "logit_interaction", "token_linear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Probe image-conditional information in TS logits and pathology tokens"
    )
    parser.add_argument("--ckpt", required=True, help="dual_patch best.pt path")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--labels",
        default="all",
        help="comma-separated checkpoint labels, with or without 'label_'; default: all",
    )
    parser.add_argument("--probe_train_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--logit_c",
        type=float,
        default=100.0,
        help="inverse L2 strength for image/logit probes",
    )
    parser.add_argument(
        "--token_c",
        type=float,
        default=1.0,
        help="inverse L2 strength for the high-dimensional token probe",
    )
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--perm_repeats", type=int, default=100)
    parser.add_argument("--perm_bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def _gather(teacher, loader, device: torch.device) -> Dict[str, np.ndarray]:
    image_logits, ts_logits, fusion_logits = [], [], []
    temporal_tokens, labels, masks = [], [], []
    teacher.eval()
    for batch in loader:
        moved = _move_lists(batch, device)
        output = teacher(
            moved["x_ts"],
            moved["x_static"],
            moved["bin_ends"],
            moved["pixel_values"],
            return_attn=True,
        )
        required = {"img_logits", "ts_logits", "fusion_logits", "ts_tokens"}
        missing = sorted(required.difference(output))
        if missing:
            raise RuntimeError(
                "Conditional probe requires a dual teacher exposing "
                f"{sorted(required)} with return_attn=True; missing={missing}"
            )
        tokens = output["ts_tokens"]
        if tokens.ndim != 3:
            raise ValueError(
                "Expected pathology-wise ts_tokens [B, K, D], got "
                f"shape={tuple(tokens.shape)}"
            )
        image_logits.append(output["img_logits"].detach().float().cpu())
        ts_logits.append(output["ts_logits"].detach().float().cpu())
        fusion_logits.append(output["fusion_logits"].detach().float().cpu())
        temporal_tokens.append(tokens.detach().float().cpu())
        labels.append(moved["y_multi"].detach().float().cpu())
        masks.append(moved["y_multi_mask"].detach().float().cpu())

    if not image_logits:
        raise RuntimeError("Probe loader yielded no batches")
    return {
        "img": torch.cat(image_logits).numpy(),
        "ts": torch.cat(ts_logits).numpy(),
        "fus": torch.cat(fusion_logits).numpy(),
        "token": torch.cat(temporal_tokens).numpy(),
        "y": torch.cat(labels).numpy(),
        "mask": torch.cat(masks).numpy(),
    }


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


def _fit_probe(
    features: np.ndarray,
    y: np.ndarray,
    c_value: float,
    max_iter: int,
) -> Pipeline:
    if features.ndim != 2:
        raise ValueError(f"Probe features must be rank 2, got {features.shape}")
    if len(np.unique(y)) < 2:
        raise ValueError("Probe-training labels contain only one class")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=int(max_iter),
                    class_weight=None,
                    random_state=0,
                ),
            ),
        ]
    )
    return model.fit(features, y)


def _predict(model: Pipeline, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = model.predict_proba(features)[:, 1].astype(np.float64)
    score = np.asarray(model.decision_function(features), dtype=np.float64)
    return probability, score


def _safe_metrics(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    metrics = {
        "bce": float(log_loss(y, probability, labels=[0, 1])),
        "auroc": float("nan"),
        "auprc": float("nan"),
    }
    if len(np.unique(y)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y, probability))
        metrics["auprc"] = float(average_precision_score(y, probability))
    return metrics


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _bootstrap_differences(
    y: np.ndarray,
    base_probability: np.ndarray,
    probe_probability: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    """Paired bootstrap; positive differences mean the probe is better."""
    rng = np.random.default_rng(seed)
    n = len(y)
    values = {"bce_gain": [], "auroc_gain": [], "auprc_gain": []}
    for _ in range(max(int(n_bootstrap), 0)):
        index = rng.integers(0, n, size=n)
        y_b = y[index]
        base_b = base_probability[index]
        probe_b = probe_probability[index]
        values["bce_gain"].append(
            log_loss(y_b, base_b, labels=[0, 1])
            - log_loss(y_b, probe_b, labels=[0, 1])
        )
        if len(np.unique(y_b)) >= 2:
            values["auroc_gain"].append(
                roc_auc_score(y_b, probe_b) - roc_auc_score(y_b, base_b)
            )
            values["auprc_gain"].append(
                average_precision_score(y_b, probe_b)
                - average_precision_score(y_b, base_b)
            )

    output: Dict[str, float] = {}
    for name, samples in values.items():
        if not samples:
            output[f"{name}_ci_low"] = float("nan")
            output[f"{name}_ci_high"] = float("nan")
            continue
        low, high = np.percentile(np.asarray(samples), [2.5, 97.5])
        output[f"{name}_ci_low"] = float(low)
        output[f"{name}_ci_high"] = float(high)
    return output


def _image_risk_bins(image_logit: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1:
        return np.zeros(len(image_logit), dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(image_logit, quantiles))
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


def _features(
    probe_name: str,
    image_logit: np.ndarray,
    ts_logit: np.ndarray,
    temporal_token: np.ndarray,
) -> np.ndarray:
    image_logit = np.asarray(image_logit).reshape(-1)
    ts_logit = np.asarray(ts_logit).reshape(-1)
    if probe_name == "image_cal":
        return image_logit[:, None]
    if probe_name == "logit_add":
        return np.column_stack([image_logit, ts_logit])
    if probe_name == "logit_interaction":
        return np.column_stack([image_logit, ts_logit, image_logit * ts_logit])
    if probe_name == "token_linear":
        if temporal_token.ndim != 2:
            raise ValueError(
                f"temporal_token for one pathology must be [N, D], got {temporal_token.shape}"
            )
        return np.column_stack([image_logit, temporal_token])
    raise ValueError(f"Unknown probe_name={probe_name!r}")


def _conditional_permutation(
    model: Pipeline,
    probe_name: str,
    y: np.ndarray,
    image_logit: np.ndarray,
    ts_logit: np.ndarray,
    temporal_token: np.ndarray,
    repeats: int,
    n_bins: int,
    seed: int,
) -> Dict[str, float]:
    """Shuffle TS within image-risk bins and summarize resulting metrics."""
    bins = _image_risk_bins(image_logit, n_bins)
    rng = np.random.default_rng(seed)
    samples = {"bce": [], "auroc": [], "auprc": []}
    for _ in range(max(int(repeats), 0)):
        index = _conditional_shuffle_indices(bins, rng)
        shuffled_ts_logit = ts_logit[index]
        shuffled_token = temporal_token[index]
        shuffled_features = _features(
            probe_name, image_logit, shuffled_ts_logit, shuffled_token
        )
        probability, _ = _predict(model, shuffled_features)
        metrics = _safe_metrics(y, probability)
        for name in samples:
            samples[name].append(metrics[name])

    output: Dict[str, float] = {}
    for name, values in samples.items():
        finite = np.asarray([value for value in values if np.isfinite(value)])
        output[f"perm_{name}_mean"] = (
            float(finite.mean()) if finite.size else float("nan")
        )
        if finite.size:
            low, high = np.percentile(finite, [2.5, 97.5])
            output[f"perm_{name}_low"] = float(low)
            output[f"perm_{name}_high"] = float(high)
        else:
            output[f"perm_{name}_low"] = float("nan")
            output[f"perm_{name}_high"] = float("nan")
    return output


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


def main() -> None:
    args = parse_args()
    if args.probe_train_split == args.test_split:
        raise SystemExit("probe_train_split and test_split must be different")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[conditional-probe] device={device}")
    teacher, bundle, _, _, pathology_labels, mode = load_teacher(args.ckpt, device)
    if mode not in ("dual", "dual_patch"):
        raise SystemExit(f"Expected a dual teacher checkpoint, got mode={mode!r}")
    datasets = bundle["datasets"]
    for split in (args.probe_train_split, args.test_split):
        if split not in datasets:
            raise SystemExit(f"Unknown split={split!r}; available={list(datasets)}")

    selected = _resolve_label_indices(args.labels, pathology_labels)
    print(
        f"[conditional-probe] mode={mode} labels="
        f"{[pathology_labels[i] for i in selected]}"
    )
    print(f"[conditional-probe] gathering probe-train={args.probe_train_split} ...")
    train_data = _gather(
        teacher,
        _loader(
            datasets[args.probe_train_split], args.batch_size, args.num_workers
        ),
        device,
    )
    print(f"[conditional-probe] gathering test={args.test_split} ...")
    test_data = _gather(
        teacher,
        _loader(datasets[args.test_split], args.batch_size, args.num_workers),
        device,
    )

    rows = []
    summary: Dict[str, object] = {
        "checkpoint": os.path.abspath(args.ckpt),
        "mode": mode,
        "probe_train_split": args.probe_train_split,
        "test_split": args.test_split,
        "configuration": vars(args),
        "labels": {},
    }
    prediction_archive: Dict[str, np.ndarray] = {
        "test_img_logits": test_data["img"],
        "test_ts_logits": test_data["ts"],
        "test_fusion_logits": test_data["fus"],
        "test_y": test_data["y"],
        "test_mask": test_data["mask"],
    }

    for label_index in selected:
        label = pathology_labels[label_index]
        train_mask = train_data["mask"][:, label_index].astype(bool)
        test_mask = test_data["mask"][:, label_index].astype(bool)
        y_train = train_data["y"][train_mask, label_index].astype(np.int64)
        y_test = test_data["y"][test_mask, label_index].astype(np.int64)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"[conditional-probe] skip {label}: one split has only one class")
            continue

        image_train = train_data["img"][train_mask, label_index]
        ts_train = train_data["ts"][train_mask, label_index]
        token_train = train_data["token"][train_mask, label_index, :]
        image_test = test_data["img"][test_mask, label_index]
        ts_test = test_data["ts"][test_mask, label_index]
        token_test = test_data["token"][test_mask, label_index, :]

        image_model = _fit_probe(
            _features("image_cal", image_train, ts_train, token_train),
            y_train,
            args.logit_c,
            args.max_iter,
        )
        base_probability, base_score = _predict(
            image_model,
            _features("image_cal", image_test, ts_test, token_test),
        )
        base_metrics = _safe_metrics(y_test, base_probability)
        label_summary: Dict[str, object] = {
            "n_test": int(len(y_test)),
            "n_positive": int(y_test.sum()),
            "prevalence": float(y_test.mean()),
            "image_cal": base_metrics,
            "probes": {},
        }

        print(
            f"\n[{label}] n={len(y_test)} pos={int(y_test.sum())} "
            f"image-cal BCE={base_metrics['bce']:.5f} "
            f"AUROC={base_metrics['auroc']:.4f} "
            f"AUPRC={base_metrics['auprc']:.4f}"
        )
        print(
            "probe                 BCE   BCEgain [95% CI]       "
            "AUROC  dROC    AUPRC  dPRC   corr_r  perm_dBCE  evidence"
        )
        print("-" * 117)

        for probe_offset, probe_name in enumerate(PROBE_NAMES):
            c_value = args.token_c if probe_name == "token_linear" else args.logit_c
            train_features = _features(
                probe_name, image_train, ts_train, token_train
            )
            test_features = _features(probe_name, image_test, ts_test, token_test)
            model = _fit_probe(train_features, y_train, c_value, args.max_iter)
            probability, score = _predict(model, test_features)
            metrics = _safe_metrics(y_test, probability)
            gains = {
                "bce_gain": base_metrics["bce"] - metrics["bce"],
                "auroc_gain": metrics["auroc"] - base_metrics["auroc"],
                "auprc_gain": metrics["auprc"] - base_metrics["auprc"],
            }
            confidence = _bootstrap_differences(
                y_test,
                base_probability,
                probability,
                args.bootstrap,
                args.seed + 1000 * label_index + probe_offset,
            )
            correction = score - base_score
            image_residual = y_test.astype(np.float64) - base_probability
            corr_residual = _pearson(correction, image_residual)
            permutation = _conditional_permutation(
                model,
                probe_name,
                y_test,
                image_test,
                ts_test,
                token_test,
                args.perm_repeats,
                args.perm_bins,
                args.seed + 10000 * label_index + probe_offset,
            )
            perm_bce_increase = permutation["perm_bce_mean"] - metrics["bce"]
            perm_auroc_drop = metrics["auroc"] - permutation["perm_auroc_mean"]
            ci_low = confidence["bce_gain_ci_low"]
            if gains["bce_gain"] > 0 and ci_low > 0 and perm_bce_increase > 0:
                evidence = "supported"
            elif gains["bce_gain"] > 0:
                evidence = "suggestive"
            else:
                evidence = "not_detected"

            row = {
                "label": label,
                "probe": probe_name,
                "n_test": int(len(y_test)),
                "n_positive": int(y_test.sum()),
                "prevalence": float(y_test.mean()),
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
                "evidence": evidence,
            }
            rows.append(row)
            label_summary["probes"][probe_name] = row
            prediction_archive[
                f"{_slug(label)}_{probe_name}_probability"
            ] = probability.astype(np.float32)

            print(
                f"{probe_name:<20} "
                f"{metrics['bce']:.5f} "
                f"{_fmt(gains['bce_gain'], 5, True):>8} "
                f"[{_fmt(confidence['bce_gain_ci_low'], 5, True):>8},"
                f"{_fmt(confidence['bce_gain_ci_high'], 5, True):>8}] "
                f"{metrics['auroc']:.4f} "
                f"{_fmt(gains['auroc_gain'], 4, True):>7} "
                f"{metrics['auprc']:.4f} "
                f"{_fmt(gains['auprc_gain'], 4, True):>7} "
                f"{_fmt(corr_residual, 3, True):>7} "
                f"{_fmt(perm_bce_increase, 5, True):>10}  "
                f"{evidence}"
            )

        summary["labels"][label] = label_summary

    csv_path = outdir / "conditional_probe.csv"
    json_path = outdir / "conditional_probe.json"
    npz_path = outdir / "conditional_probe_predictions.npz"
    _write_csv(rows, csv_path)
    with json_path.open("w") as handle:
        json.dump(_json_ready(summary), handle, indent=2, ensure_ascii=False)
    np.savez_compressed(npz_path, **prediction_archive)

    print(f"\n[conditional-probe] CSV  -> {csv_path}")
    print(f"[conditional-probe] JSON -> {json_path}")
    print(f"[conditional-probe] NPZ  -> {npz_path}")
    print(
        "[interpretation] BCEgain > 0 is better. 'supported' requires a paired "
        "bootstrap BCE-gain CI above zero and worse BCE after within-image-risk "
        "TS permutation. This is evidence about the current representation, not "
        "proof that the raw TS data do or do not contain all possible signal."
    )


if __name__ == "__main__":
    main()
