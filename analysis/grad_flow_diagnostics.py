"""Gradient diagnostics for the dual-patch pathology-query teacher.

This module is deliberately read-only with respect to training:

* it does not call ``backward()`` on model parameters,
* it never calls ``optimizer.step()``,
* it uses ``torch.autograd.grad`` and discards the diagnostic graph, and
* it restores the model's original train/eval state on exit.

The main entry point is :func:`run_dual_gradient_diagnostics`; the CLI at
the bottom loads a trained checkpoint, rebuilds ``DualPathologyLoss`` from
its saved args, constructs a deterministic subset loader, and prints/saves
the report.

Usage:
    python analysis/grad_flow_diagnostics.py \
        --ckpt <path>/best.pt \
        --outdir analysis/grad_diag_<name>/ \
        --split train \
        --max_batches 16 \
        --batch_size 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.visualize_pathology import _loader, load_teacher
from loss.losses_duett import DualPathologyLoss


Tensor = torch.Tensor


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move the fields used by TeacherModel to ``device``."""
    moved: Dict[str, Any] = {
        "x_ts": tuple(t.to(device) for t in batch["x_ts"]),
        "x_static": tuple(t.to(device) for t in batch["x_static"]),
        "bin_ends": tuple(t.to(device) for t in batch["bin_ends"]),
        "pixel_values": batch["pixel_values"].to(device),
        "y_multi": batch["y_multi"].to(device),
        "y_multi_mask": batch["y_multi_mask"].to(device),
    }
    if "y" in batch:
        moved["y"] = batch["y"].to(device)
    return moved


def _unwrap_model(model: torch.nn.Module, accelerator=None) -> torch.nn.Module:
    if accelerator is not None:
        return accelerator.unwrap_model(model)
    while hasattr(model, "module"):
        model = model.module
    return model


def _find_pathology_query_banks(
    model: torch.nn.Module,
) -> Tuple[Tuple[str, ...], Tuple[Tensor, ...]]:
    """Find independent modality queries or the legacy shared query."""
    named = dict(model.named_parameters())
    image = [(n, p) for n, p in named.items() if n.endswith("image_queries")]
    temporal = [(n, p) for n, p in named.items() if n.endswith("temporal_queries")]
    if len(image) == 1 and len(temporal) == 1:
        pairs = (image[0], temporal[0])
        return tuple(n for n, _ in pairs), tuple(p for _, p in pairs)

    shared = [(n, p) for n, p in named.items() if n.endswith("pathology_queries")]
    if len(shared) == 1:
        return (shared[0][0],), (shared[0][1],)

    raise RuntimeError(
        "Expected image_queries + temporal_queries, or one legacy "
        f"pathology_queries; found image={len(image)}, temporal={len(temporal)}, "
        f"shared={len(shared)}"
    )


def _per_pathology_bce(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    pos_weight: Optional[Tensor],
    eps: float,
) -> Tensor:
    """Masked BCE identical to DualPathologyLoss, without detach."""
    losses = []
    for k in range(logits.shape[1]):
        pw = None if pos_weight is None else pos_weight[k : k + 1]
        elementwise = F.binary_cross_entropy_with_logits(
            logits[:, k],
            target[:, k],
            reduction="none",
            pos_weight=pw,
        )
        valid = mask[:, k].to(elementwise.dtype)
        losses.append((elementwise * valid).sum() / (valid.sum() + eps))
    return torch.stack(losses)


def _branch_losses(
    output: Mapping[str, Tensor],
    target: Tensor,
    mask: Tensor,
    loss_fn,
) -> Dict[str, Tensor]:
    pos_weight = getattr(loss_fn, "pos_weight", None)
    eps = float(getattr(loss_fn, "eps", 1e-6))
    label_weights = loss_fn.label_weights.to(
        device=target.device, dtype=output["fusion_logits"].dtype
    )

    img_per = _per_pathology_bce(
        output["img_logits"], target, mask, pos_weight, eps
    )
    ts_per = _per_pathology_bce(
        output["ts_logits"], target, mask, pos_weight, eps
    )
    fus_per = _per_pathology_bce(
        output["fusion_logits"], target, mask, pos_weight, eps
    )
    return {
        "img_per": img_per,
        "ts_per": ts_per,
        "fus_per": fus_per,
        "img": (label_weights * img_per).sum(),
        "ts": (label_weights * ts_per).sum(),
        "fus": (label_weights * fus_per).sum(),
        "label_weights": label_weights,
    }


def _grad(loss: Tensor, target: Tensor) -> Tensor:
    grad = torch.autograd.grad(
        loss,
        target,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if grad is None:
        grad = torch.zeros_like(target)
    return grad.detach().float()


def _grads(loss: Tensor, targets: Sequence[Tensor]) -> Tuple[Tensor, ...]:
    values = torch.autograd.grad(
        loss,
        tuple(targets),
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(target).float()
        if value is None
        else value.detach().float()
        for value, target in zip(values, targets)
    )


def _norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value.reshape(-1))


def _cosine(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denominator = _norm(a_flat) * _norm(b_flat)
    if float(denominator.detach().cpu()) <= eps:
        return torch.zeros((), device=a.device, dtype=torch.float32)
    return torch.dot(a_flat, b_flat) / denominator.clamp_min(eps)


def _sample_token_sensitivity(grad: Tensor, token: Tensor) -> Tuple[Tensor, Tensor]:
    """Return sum of raw and scale-normalized sensitivity over samples."""
    grad_f = grad.float().flatten(1)
    token_f = token.detach().float().flatten(1)
    grad_norm = torch.linalg.vector_norm(grad_f, dim=1)
    token_norm = torch.linalg.vector_norm(token_f, dim=1)
    raw = grad_norm.sum()
    normalized = (grad_norm * token_norm).sum()
    return raw, normalized


def _reduce_sum(value: Tensor, accelerator=None) -> Tensor:
    if accelerator is not None and accelerator.num_processes > 1:
        return accelerator.reduce(value, reduction="sum")
    return value


def _cosine_matrix(rows: Tensor, eps: float = 1e-12) -> Tensor:
    rows = rows.float()
    rows = rows / torch.linalg.vector_norm(rows, dim=-1, keepdim=True).clamp_min(eps)
    return rows @ rows.transpose(0, 1)


def _effective_queries(block: torch.nn.Module, prototypes: Tensor) -> Tensor:
    """Apply a _PerceiverBlock's LayerNorm and MHA query projection."""
    normalized = block.norm_q(prototypes)
    attention = block.attn
    d_model = prototypes.shape[-1]
    if attention.in_proj_weight is None:
        raise RuntimeError(
            "The diagnostic currently expects MultiheadAttention with "
            "a packed in_proj_weight."
        )
    q_weight = attention.in_proj_weight[:d_model]
    q_bias = (
        None
        if attention.in_proj_bias is None
        else attention.in_proj_bias[:d_model]
    )
    return F.linear(normalized, q_weight, q_bias)


def _float(value: Tensor) -> float:
    return float(value.detach().cpu())


def run_dual_gradient_diagnostics(
    teacher: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    loss_fn,
    device: torch.device,
    *,
    max_batches: int = 8,
    label_names: Optional[Sequence[str]] = None,
    accelerator=None,
) -> Dict[str, Any]:
    """Measure objective gradients and fusion-token sensitivity.

    Parameters
    ----------
    teacher:
        Prepared or unprepared TeacherModel in ``dual_patch`` mode.
    loader:
        Prefer a deterministic, non-shuffled subset of the training set.
    loss_fn:
        The existing DualPathologyLoss instance. Its label weights,
        positive weights, branch alphas, and masking convention are reused.
    device:
        Device that receives the diagnostic batches.
    max_batches:
        Number of fixed batches to aggregate. Eight is a useful first pass.
    label_names:
        Optional pathology names in model output order.
    accelerator:
        Optional Hugging Face Accelerator. All ranks run the diagnostic and
        the small accumulated tensors are reduced across ranks.
    """
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")

    model = _unwrap_model(teacher, accelerator)
    query_names, query_parameters = _find_pathology_query_banks(model)
    for query_name, parameter in zip(query_names, query_parameters):
        if not parameter.requires_grad:
            raise RuntimeError(f"{query_name} does not require gradients")

    n_labels, d_model = query_parameters[0].shape
    if any(parameter.shape != (n_labels, d_model) for parameter in query_parameters):
        raise RuntimeError("All pathology query banks must have the same [K, D] shape")
    n_query_banks = len(query_parameters)
    query_shape = (n_query_banks, n_labels, d_model)
    if label_names is None:
        label_names = [f"label_{k}" for k in range(n_labels)]
    if len(label_names) != n_labels:
        raise ValueError(
            f"label_names has {len(label_names)} entries but queries have {n_labels} rows"
        )

    branch_names = ("img", "ts", "fus")
    grad_sums = {
        name: torch.zeros(query_shape, dtype=torch.float32, device=device)
        for name in branch_names
    }
    per_label_grad_sums = {
        name: torch.zeros(
            n_labels, *query_shape, dtype=torch.float32, device=device
        )
        for name in branch_names
    }

    batch_cos_sum = torch.zeros((), dtype=torch.float32, device=device)
    batch_cos_negative = torch.zeros((), dtype=torch.float32, device=device)
    batch_count = torch.zeros((), dtype=torch.float32, device=device)
    sample_count = torch.zeros((), dtype=torch.float32, device=device)
    valid_label_count = torch.zeros(
        n_labels, dtype=torch.float32, device=device
    )

    fusion_sensitivity = {
        "img_raw": torch.zeros((), dtype=torch.float32, device=device),
        "ts_raw": torch.zeros((), dtype=torch.float32, device=device),
        "img_scaled": torch.zeros((), dtype=torch.float32, device=device),
        "ts_scaled": torch.zeros((), dtype=torch.float32, device=device),
    }
    per_label_fusion_sensitivity = {
        key: torch.zeros(n_labels, dtype=torch.float32, device=device)
        for key in ("img_raw", "ts_raw", "img_scaled", "ts_scaled")
    }
    loss_sums = {
        name: torch.zeros((), dtype=torch.float32, device=device)
        for name in branch_names
    }

    was_training = model.training
    model.eval()
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            moved = _move_batch(batch, device)
            autocast_context = (
                accelerator.autocast()
                if accelerator is not None
                else nullcontext()
            )
            with torch.enable_grad(), autocast_context:
                output = model(
                    moved["x_ts"],
                    moved["x_static"],
                    moved["bin_ends"],
                    moved["pixel_values"],
                    return_attn=True,
                )
                required = {
                    "img_logits",
                    "ts_logits",
                    "fusion_logits",
                    "img_tokens",
                    "ts_tokens",
                }
                missing = sorted(required.difference(output))
                if missing:
                    raise RuntimeError(
                        "TeacherModel diagnostic output is missing "
                        f"{missing}. In dual_patch mode, return_attn=True must "
                        "also expose img_tokens and ts_tokens."
                    )

                losses = _branch_losses(
                    output,
                    moved["y_multi"],
                    moved["y_multi_mask"],
                    loss_fn,
                )

                batch_grads = {
                    name: torch.stack(
                        _grads(losses[name], query_parameters), dim=0
                    )
                    for name in branch_names
                }
                for name in branch_names:
                    grad_sums[name] += batch_grads[name]
                    loss_sums[name] += losses[name].detach().float()

                img_ts_cos = _cosine(batch_grads["img"], batch_grads["ts"])
                batch_cos_sum += img_ts_cos
                batch_cos_negative += (img_ts_cos < 0).float()

                label_weights = losses["label_weights"]
                for k in range(n_labels):
                    for name, per_key in (
                        ("img", "img_per"),
                        ("ts", "ts_per"),
                        ("fus", "fus_per"),
                    ):
                        label_loss = label_weights[k] * losses[per_key][k]
                        per_label_grad_sums[name][k] += torch.stack(
                            _grads(label_loss, query_parameters), dim=0
                        )

                img_tokens = output["img_tokens"]
                ts_tokens = output["ts_tokens"]
                fus_img_grad, fus_ts_grad = _grads(
                    losses["fus"], (img_tokens, ts_tokens)
                )
                img_raw, img_scaled = _sample_token_sensitivity(
                    fus_img_grad, img_tokens
                )
                ts_raw, ts_scaled = _sample_token_sensitivity(
                    fus_ts_grad, ts_tokens
                )
                fusion_sensitivity["img_raw"] += img_raw
                fusion_sensitivity["ts_raw"] += ts_raw
                fusion_sensitivity["img_scaled"] += img_scaled
                fusion_sensitivity["ts_scaled"] += ts_scaled

                for k in range(n_labels):
                    label_fus_loss = label_weights[k] * losses["fus_per"][k]
                    label_img_grad, label_ts_grad = _grads(
                        label_fus_loss, (img_tokens, ts_tokens)
                    )
                    li_raw, li_scaled = _sample_token_sensitivity(
                        label_img_grad, img_tokens
                    )
                    lt_raw, lt_scaled = _sample_token_sensitivity(
                        label_ts_grad, ts_tokens
                    )
                    per_label_fusion_sensitivity["img_raw"][k] += li_raw
                    per_label_fusion_sensitivity["ts_raw"][k] += lt_raw
                    per_label_fusion_sensitivity["img_scaled"][k] += li_scaled
                    per_label_fusion_sensitivity["ts_scaled"][k] += lt_scaled

                batch_size = float(moved["y_multi"].shape[0])
                batch_count += 1.0
                sample_count += batch_size
                valid_label_count += moved["y_multi_mask"].float().sum(dim=0)

            del output, losses, batch_grads
    finally:
        model.train(was_training)

    if _float(batch_count) == 0:
        raise RuntimeError("The diagnostic loader yielded no batches")

    # Aggregate the small tensors across Accelerator processes.
    batch_count = _reduce_sum(batch_count, accelerator)
    sample_count = _reduce_sum(sample_count, accelerator)
    valid_label_count = _reduce_sum(valid_label_count, accelerator)
    batch_cos_sum = _reduce_sum(batch_cos_sum, accelerator)
    batch_cos_negative = _reduce_sum(batch_cos_negative, accelerator)
    for name in branch_names:
        grad_sums[name] = _reduce_sum(grad_sums[name], accelerator)
        per_label_grad_sums[name] = _reduce_sum(
            per_label_grad_sums[name], accelerator
        )
        loss_sums[name] = _reduce_sum(loss_sums[name], accelerator)
    for key in fusion_sensitivity:
        fusion_sensitivity[key] = _reduce_sum(
            fusion_sensitivity[key], accelerator
        )
        per_label_fusion_sensitivity[key] = _reduce_sum(
            per_label_fusion_sensitivity[key], accelerator
        )

    mean_grads = {
        name: grad_sums[name] / batch_count.clamp_min(1.0)
        for name in branch_names
    }
    mean_label_grads = {
        name: per_label_grad_sums[name] / batch_count.clamp_min(1.0)
        for name in branch_names
    }

    alphas = {
        "img": float(getattr(loss_fn, "alpha_img", 1.0)),
        "ts": float(getattr(loss_fn, "alpha_ts", 1.0)),
        "fus": float(getattr(loss_fn, "alpha_fus", 1.0)),
    }
    weighted_grads = {
        name: alphas[name] * mean_grads[name] for name in branch_names
    }
    total_weighted_grad = sum(weighted_grads.values())

    branch_report: Dict[str, Dict[str, float]] = {}
    for name in branch_names:
        branch_report[name] = {
            "loss": _float(loss_sums[name] / batch_count.clamp_min(1.0)),
            "alpha": alphas[name],
            "raw_grad_norm": _float(_norm(mean_grads[name])),
            "weighted_grad_norm": _float(_norm(weighted_grads[name])),
            "cos_to_total_update": _float(
                _cosine(weighted_grads[name], total_weighted_grad)
            ),
        }

    eps = 1e-12
    sensitivity_report = {
        key: _float(value / sample_count.clamp_min(1.0))
        for key, value in fusion_sensitivity.items()
    }
    sensitivity_report["raw_img_over_ts"] = (
        sensitivity_report["img_raw"]
        / max(sensitivity_report["ts_raw"], eps)
    )
    sensitivity_report["scaled_img_over_ts"] = (
        sensitivity_report["img_scaled"]
        / max(sensitivity_report["ts_scaled"], eps)
    )

    per_label_report = []
    for k, label_name in enumerate(label_names):
        img_grad = mean_label_grads["img"][k]
        ts_grad = mean_label_grads["ts"][k]
        fus_grad = mean_label_grads["fus"][k]
        total_grad = (
            alphas["img"] * img_grad
            + alphas["ts"] * ts_grad
            + alphas["fus"] * fus_grad
        )
        # Independent model: image supervision owns bank 0; TS and residual
        # fusion own the temporal bank. Legacy shared-query models use bank 0.
        own_bank = {"img": 0, "ts": n_query_banks - 1, "fus": n_query_banks - 1}
        own_row_norm = {
            name: _float(_norm(mean_label_grads[name][k, own_bank[name], k]))
            for name in branch_names
        }
        full_norm = {
            name: _float(_norm(mean_label_grads[name][k]))
            for name in branch_names
        }
        label_sensitivity = {
            key: _float(
                per_label_fusion_sensitivity[key][k]
                / valid_label_count[k].clamp_min(1.0)
            )
            for key in per_label_fusion_sensitivity
        }
        label_sensitivity["scaled_img_over_ts"] = (
            label_sensitivity["img_scaled"]
            / max(label_sensitivity["ts_scaled"], eps)
        )
        per_label_report.append(
            {
                "label": str(label_name),
                "valid_samples": int(round(_float(valid_label_count[k]))),
                "img_grad_norm": full_norm["img"],
                "ts_grad_norm": full_norm["ts"],
                "fus_grad_norm": full_norm["fus"],
                "img_ts_cos": _float(_cosine(img_grad, ts_grad)),
                "img_fus_cos": _float(_cosine(img_grad, fus_grad)),
                "ts_fus_cos": _float(_cosine(ts_grad, fus_grad)),
                "weighted_total_grad_norm": _float(_norm(total_grad)),
                "img_own_query_fraction": own_row_norm["img"]
                / max(full_norm["img"], eps),
                "ts_own_query_fraction": own_row_norm["ts"]
                / max(full_norm["ts"], eps),
                "fus_own_query_fraction": own_row_norm["fus"]
                / max(full_norm["fus"], eps),
                "fusion_token_sensitivity": label_sensitivity,
            }
        )

    # Query geometry is deterministic at the checkpoint, so no loader averaging
    # is necessary. The two effective spaces have independent W_Q matrices;
    # compare their within-modality Gram matrices rather than treating direct
    # cross-modal coordinates as semantically aligned.
    with torch.no_grad():
        perceiver = model.perceiver
        image_prototypes = query_parameters[0].detach()
        temporal_prototypes = query_parameters[-1].detach()
        raw_gram = _cosine_matrix(temporal_prototypes)
        img_effective = _effective_queries(
            perceiver.img_cross, image_prototypes
        )
        if hasattr(perceiver, "event_query_proj"):
            ts_effective = perceiver.event_query_norm(
                perceiver.event_query_proj(temporal_prototypes)
            )
        else:
            ts_effective = _effective_queries(
                perceiver.ts_cross, temporal_prototypes
            )
        img_gram = _cosine_matrix(img_effective)
        ts_gram = _cosine_matrix(ts_effective)
        geometry_gap = torch.linalg.vector_norm(img_gram - ts_gram) / n_labels

    report: Dict[str, Any] = {
        "query_parameter": "+".join(query_names),
        "query_layout": "independent" if n_query_banks == 2 else "shared",
        "batches": int(round(_float(batch_count))),
        "samples": int(round(_float(sample_count))),
        "branch": branch_report,
        "pairwise_gradient_cosine": {
            "img_ts": _float(_cosine(mean_grads["img"], mean_grads["ts"])),
            "img_fus": _float(_cosine(mean_grads["img"], mean_grads["fus"])),
            "ts_fus": _float(_cosine(mean_grads["ts"], mean_grads["fus"])),
            "img_ts_batch_mean": _float(
                batch_cos_sum / batch_count.clamp_min(1.0)
            ),
            "img_ts_negative_batch_fraction": _float(
                batch_cos_negative / batch_count.clamp_min(1.0)
            ),
        },
        "weighted_img_over_ts": branch_report["img"]["weighted_grad_norm"]
        / max(branch_report["ts"]["weighted_grad_norm"], eps),
        "fusion_token_sensitivity": sensitivity_report,
        "per_label": per_label_report,
        "query_geometry": {
            "prototype_norms": [
                _float(value)
                for value in torch.linalg.vector_norm(
                    torch.stack([p.detach().float() for p in query_parameters]),
                    dim=-1,
                ).flatten()
            ],
            "raw_cosine": raw_gram.detach().cpu().tolist(),
            "image_effective_cosine": img_gram.detach().cpu().tolist(),
            "ts_effective_cosine": ts_gram.detach().cpu().tolist(),
            "image_ts_gram_gap": _float(geometry_gap),
        },
    }
    return report


def format_gradient_diagnostics(report: Mapping[str, Any]) -> str:
    """Human-readable console summary."""
    lines = [
        (
            f"[grad-diag] parameter={report['query_parameter']} "
            f"layout={report.get('query_layout', 'shared')} "
            f"batches={report['batches']} samples={report['samples']}"
        ),
        "",
        "branch      loss    alpha    ||g raw||   ||alpha*g||   cos(g,total)",
        "-------------------------------------------------------------------",
    ]
    for name in ("img", "ts", "fus"):
        item = report["branch"][name]
        lines.append(
            f"{name:<7} {item['loss']:>9.5f} {item['alpha']:>7.3f} "
            f"{item['raw_grad_norm']:>12.6g} "
            f"{item['weighted_grad_norm']:>13.6g} "
            f"{item['cos_to_total_update']:>14.5f}"
        )

    cosine = report["pairwise_gradient_cosine"]
    sensitivity = report["fusion_token_sensitivity"]
    if report.get("query_layout") == "independent":
        lines.extend([
            "",
            "img–ts query-gradient cosine is expected to be 0: the modality "
            "queries occupy disjoint parameter banks.",
        ])
    lines.extend(
        [
            "",
            (
                "gradient cosine: "
                f"img-ts={cosine['img_ts']:+.5f}  "
                f"img-fus={cosine['img_fus']:+.5f}  "
                f"ts-fus={cosine['ts_fus']:+.5f}"
            ),
            (
                "batch img-ts cosine: "
                f"mean={cosine['img_ts_batch_mean']:+.5f}  "
                f"negative_fraction="
                f"{cosine['img_ts_negative_batch_fraction']:.3f}"
            ),
            (
                "weighted gradient dominance: "
                f"img/ts={report['weighted_img_over_ts']:.4f}"
            ),
            (
                "fusion token sensitivity: "
                f"raw img/ts={sensitivity['raw_img_over_ts']:.4f}  "
                f"scale-normalized img/ts="
                f"{sensitivity['scaled_img_over_ts']:.4f}"
            ),
            "",
            (
                "label                         ||g_img||   ||g_ts||  "
                "cos(i,t)  fusSens(i/t)  ownQ(img/ts/fus)"
            ),
            "-" * 100,
        ]
    )
    for item in report["per_label"]:
        token_sensitivity = item["fusion_token_sensitivity"]
        lines.append(
            f"{item['label']:<28} "
            f"{item['img_grad_norm']:>10.5g} "
            f"{item['ts_grad_norm']:>10.5g} "
            f"{item['img_ts_cos']:>+9.4f} "
            f"{token_sensitivity['scaled_img_over_ts']:>13.4f} "
            f"{item['img_own_query_fraction']:.2f}/"
            f"{item['ts_own_query_fraction']:.2f}/"
            f"{item['fus_own_query_fraction']:.2f}"
        )

    geometry = report["query_geometry"]
    lines.extend(
        [
            "",
            (
                "query geometry: prototype norms="
                + ", ".join(f"{value:.4f}" for value in geometry["prototype_norms"])
            ),
            (
                "effective image-vs-TS Gram gap="
                f"{geometry['image_ts_gram_gap']:.6f}"
            ),
        ]
    )
    return "\n".join(lines)


def gradient_diagnostics_to_log_dict(
    report: Mapping[str, Any],
    prefix: str = "grad_diag",
) -> Dict[str, float]:
    """Flatten the most useful values for WandB/TensorBoard."""
    values: Dict[str, float] = {}
    for name, item in report["branch"].items():
        values[f"{prefix}/{name}/loss"] = float(item["loss"])
        values[f"{prefix}/{name}/raw_grad_norm"] = float(
            item["raw_grad_norm"]
        )
        values[f"{prefix}/{name}/weighted_grad_norm"] = float(
            item["weighted_grad_norm"]
        )
        values[f"{prefix}/{name}/cos_to_total"] = float(
            item["cos_to_total_update"]
        )

    cosine = report["pairwise_gradient_cosine"]
    for key, value in cosine.items():
        values[f"{prefix}/cosine/{key}"] = float(value)

    values[f"{prefix}/dominance/weighted_img_over_ts"] = float(
        report["weighted_img_over_ts"]
    )
    sensitivity = report["fusion_token_sensitivity"]
    values[f"{prefix}/fusion_sensitivity/raw_img_over_ts"] = float(
        sensitivity["raw_img_over_ts"]
    )
    values[f"{prefix}/fusion_sensitivity/scaled_img_over_ts"] = float(
        sensitivity["scaled_img_over_ts"]
    )
    values[f"{prefix}/query_geometry/image_ts_gram_gap"] = float(
        report["query_geometry"]["image_ts_gram_gap"]
    )

    for item in report["per_label"]:
        safe_label = item["label"].replace("/", "_")
        base = f"{prefix}/label/{safe_label}"
        values[f"{base}/img_grad_norm"] = float(item["img_grad_norm"])
        values[f"{base}/ts_grad_norm"] = float(item["ts_grad_norm"])
        values[f"{base}/fus_grad_norm"] = float(item["fus_grad_norm"])
        values[f"{base}/img_ts_cos"] = float(item["img_ts_cos"])
        values[f"{base}/fusion_scaled_img_over_ts"] = float(
            item["fusion_token_sensitivity"]["scaled_img_over_ts"]
        )
    return values


# =============================================================================
# CLI: ckpt 로딩 + loss 재구성 + subset loader → run diagnostic
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser("Gradient flow diagnostics for dual_patch teacher")
    p.add_argument("--ckpt",        type=str, required=True, help="best.pt path")
    p.add_argument("--outdir",      type=str, required=True)
    p.add_argument("--split",       type=str, default="train",
                   choices=["train", "val", "test"])
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=16,
                   help="deterministic subset size in batches")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[grad-diag] device={device}")

    teacher, bundle, processor, ck_args, pathology_labels, mode = load_teacher(
        args.ckpt, device)
    print(f"[grad-diag] ckpt: {args.ckpt}")
    print(f"[grad-diag] mode={mode}  K={len(pathology_labels)}  labels={pathology_labels}")

    if mode != "dual_patch":
        raise RuntimeError(f"perceiver_type={mode} 은 지원하지 않음. dual_patch 만 대상.")

    label_weights = torch.tensor(
        [float(w) for w in ck_args.label_weights.split(",")], dtype=torch.float32)
    if label_weights.numel() != len(pathology_labels):
        raise ValueError(f"label_weights len ({label_weights.numel()}) != "
                         f"pathology_labels len ({len(pathology_labels)})")
    loss_fn = DualPathologyLoss(
        label_weights=label_weights,
        pos_weight=None,
        alpha_img=ck_args.aux_img_alpha,
        alpha_ts=ck_args.aux_ts_alpha,
        alpha_fus=ck_args.aux_fus_alpha,
    ).to(device)
    print(f"[grad-diag] loss alphas: img={ck_args.aux_img_alpha} "
          f"ts={ck_args.aux_ts_alpha} fus={ck_args.aux_fus_alpha}")

    split_ds = bundle["datasets"][args.split]
    n_needed = min(args.max_batches * args.batch_size, len(split_ds))
    subset = Subset(split_ds, list(range(n_needed)))
    loader = _loader(subset, args.batch_size, args.num_workers)
    print(f"[grad-diag] split={args.split}: first {n_needed}/{len(split_ds)} samples "
          f"→ up to {args.max_batches} batches of {args.batch_size}")

    report = run_dual_gradient_diagnostics(
        teacher, loader, loss_fn, device,
        max_batches=args.max_batches,
        label_names=list(pathology_labels),
    )

    txt = format_gradient_diagnostics(report)
    print("\n" + txt)

    txt_path = os.path.join(args.outdir, "grad_flow_report.txt")
    with open(txt_path, "w") as f:
        f.write(txt + "\n")
    json_path = os.path.join(args.outdir, "grad_flow_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[grad-diag] saved: {txt_path}")
    print(f"[grad-diag] saved: {json_path}")


if __name__ == "__main__":
    main()
