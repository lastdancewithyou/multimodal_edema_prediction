"""Binary AUROC/AUPRC evaluation for teacher/student."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@torch.no_grad()
def evaluate_binary(model, loader, device, forward_fn):
    """Aggregate logits/labels across a loader and compute AUROC/AUPRC.

    forward_fn(model, batch, device) -> dict with keys `logits`, `y`.
    Handles both teacher (with pixel_values) and student loaders through the
    forward_fn indirection.
    """
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        out = forward_fn(model, batch, device)
        logits_all.append(out["logits"].cpu())
        y_all.append(out["y"].cpu())
    logits = torch.cat(logits_all).float()
    y = torch.cat(y_all).float().numpy()
    probs = torch.sigmoid(logits).numpy()

    try:
        auroc = float(roc_auc_score(y, probs))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y, probs))
    except ValueError:
        auprc = float("nan")

    return {"auroc": auroc, "auprc": auprc, "n": len(y), "pos_frac": float(y.mean())}


def make_teacher_forward():
    """Main head 기준 evaluation.
    - Pathology mode: dict → main_logit (= stage4_logits[:, 0])
    - Legacy + aux  : tuple → [0]
    - Legacy no-aux : Tensor
    """
    from .engine import _move_lists

    @torch.no_grad()
    def _fwd(teacher, batch, device):
        b = _move_lists(batch, device)
        out = teacher(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"])
        if isinstance(out, dict):
            z = out["main_logit"]
        elif isinstance(out, tuple):
            z = out[0]
        else:
            z = out
        return {"logits": z, "y": b["y"]}

    return _fwd


def make_teacher_aux_forward():
    """Auxiliary CXR-only head 기준 evaluation. Teacher가 tuple 반환할 때만 유효."""
    from .engine import _move_lists

    @torch.no_grad()
    def _fwd(teacher, batch, device):
        b = _move_lists(batch, device)
        out = teacher(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"])
        if not isinstance(out, tuple):
            raise RuntimeError("aux forward는 TeacherModel(use_aux_cxr=True)에서만 사용 가능")
        return {"logits": out[1], "y": b["y"]}

    return _fwd


def make_student_forward():
    from .engine import _move_lists

    @torch.no_grad()
    def _fwd(student, batch, device):
        b = _move_lists(batch, device)
        z = student(b["x_ts"], b["x_static"], b["bin_ends"])
        return {"logits": z, "y": b["y"]}

    return _fwd


# =============================================================================
# Pathology-mode evaluation (per-pathology stage2/stage4 AUROC/AUPRC + gap)
# =============================================================================
def _safe(fn, y, p):
    try:
        return float(fn(y, p))
    except ValueError:
        return float("nan")


@torch.no_grad()
def evaluate_pathology(model, loader, device, pathology_labels: tuple[str, ...]) -> dict:
    """PathologyPerceiver teacher 평가.

    Loader 전체에서 stage2/stage4 logits와 multi-label을 모아 pathology마다
    valid mask를 적용해 AUROC/AUPRC + gap을 계산.

    Returns dict with keys:
        labels, n, main_auroc, main_auprc, per_label(list of dict)
    """
    from .engine import _move_lists

    model.eval()
    s2_all, s4_all, y_all, mask_all = [], [], [], []
    for batch in loader:
        b = _move_lists(batch, device)
        out = model(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"])
        if not isinstance(out, dict):
            raise RuntimeError("evaluate_pathology 는 pathology_mode teacher 전용")
        s2_all.append(out["stage2_logits"].cpu())
        s4_all.append(out["stage4_logits"].cpu())
        y_all.append(b["y_multi"].cpu())
        mask_all.append(b["y_multi_mask"].cpu())

    s2 = torch.cat(s2_all).float().numpy()   # [N, K]
    s4 = torch.cat(s4_all).float().numpy()
    y  = torch.cat(y_all).float().numpy()
    mk = torch.cat(mask_all).float().numpy()

    K = len(pathology_labels)
    per_label = []
    for k in range(K):
        m = mk[:, k].astype(bool)
        yk = y[m, k]
        p2 = 1.0 / (1.0 + np.exp(-s2[m, k]))
        p4 = 1.0 / (1.0 + np.exp(-s4[m, k]))
        a2 = _safe(roc_auc_score, yk, p2)
        a4 = _safe(roc_auc_score, yk, p4)
        r2 = _safe(average_precision_score, yk, p2)
        r4 = _safe(average_precision_score, yk, p4)
        per_label.append({
            "name":         pathology_labels[k],
            "n_valid":      int(m.sum()),
            "pos_frac":     float(yk.mean()) if len(yk) > 0 else float("nan"),
            "stage2_auroc": a2, "stage4_auroc": a4, "gap_auroc": a4 - a2,
            "stage2_auprc": r2, "stage4_auprc": r4, "gap_auprc": r4 - r2,
        })

    return {
        "labels":     list(pathology_labels),
        "n":          int(len(y)),
        "main_auroc": per_label[0]["stage4_auroc"],   # Edema stage4 = main task
        "main_auprc": per_label[0]["stage4_auprc"],
        "per_label":  per_label,
    }


def format_pathology_gap_table(result: dict) -> str:
    """콘솔 출력용 gap 표. per_label의 각 항목을 한 줄씩."""
    header = (f"{'label':<22s} {'n':>6s} {'pos':>7s} "
              f"{'s2_auroc':>10s} {'s4_auroc':>10s} {'gap_ro':>8s} "
              f"{'s2_auprc':>10s} {'s4_auprc':>10s} {'gap_pr':>8s}")
    lines = [header]
    for r in result["per_label"]:
        lines.append(
            f"{r['name']:<22s} {r['n_valid']:>6d} {r['pos_frac']:>7.4f} "
            f"{r['stage2_auroc']:>10.4f} {r['stage4_auroc']:>10.4f} {r['gap_auroc']:>+8.4f} "
            f"{r['stage2_auprc']:>10.4f} {r['stage4_auprc']:>10.4f} {r['gap_auprc']:>+8.4f}"
        )
    return "\n".join(lines)


# =============================================================================
# Dual pathology mode evaluation (3-branch: image / ts / fusion per pathology)
# =============================================================================
@torch.no_grad()
def evaluate_dual_pathology(model, loader, device, pathology_labels: tuple[str, ...]) -> dict:
    """DualPathologyPerceiver teacher 평가.

    loader 전체에서 img_logits/ts_logits/fusion_logits 및 multi-label을 모아
    pathology 마다 valid mask 적용해 AUROC/AUPRC + gap 계산.

    gap_i2f = fusion_auroc - img_auroc  (fusion이 image-only 이겼나)
    gap_t2f = fusion_auroc - ts_auroc   (fusion이 ts-only 이겼나)

    main_auroc = per_label[0]["fus_auroc"]  (Edema fusion)   ← best ckpt 선정 기준
    """
    from .engine import _move_lists

    model.eval()
    img_all, ts_all, fus_all, y_all, mask_all = [], [], [], [], []
    for batch in loader:
        b = _move_lists(batch, device)
        out = model(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"])
        if not isinstance(out, dict) or "fusion_logits" not in out:
            raise RuntimeError("evaluate_dual_pathology 는 dual_pathology_mode teacher 전용")
        img_all.append(out["img_logits"].cpu())
        ts_all.append(out["ts_logits"].cpu())
        fus_all.append(out["fusion_logits"].cpu())
        y_all.append(b["y_multi"].cpu())
        mask_all.append(b["y_multi_mask"].cpu())

    img = torch.cat(img_all).float().numpy()
    ts  = torch.cat(ts_all).float().numpy()
    fus = torch.cat(fus_all).float().numpy()
    y   = torch.cat(y_all).float().numpy()
    mk  = torch.cat(mask_all).float().numpy()

    K = len(pathology_labels)
    per_label = []
    for k in range(K):
        m = mk[:, k].astype(bool)
        yk = y[m, k]
        pi = 1.0 / (1.0 + np.exp(-img[m, k]))
        pt = 1.0 / (1.0 + np.exp(-ts[m, k]))
        pf = 1.0 / (1.0 + np.exp(-fus[m, k]))
        ai, at, af = _safe(roc_auc_score, yk, pi), _safe(roc_auc_score, yk, pt), _safe(roc_auc_score, yk, pf)
        ri, rt, rf = _safe(average_precision_score, yk, pi), _safe(average_precision_score, yk, pt), _safe(average_precision_score, yk, pf)
        per_label.append({
            "name":       pathology_labels[k],
            "n_valid":    int(m.sum()),
            "pos_frac":   float(yk.mean()) if len(yk) else float("nan"),
            "img_auroc":  ai, "ts_auroc":  at, "fus_auroc":  af,
            "gap_i2f":    af - ai,  "gap_t2f": af - at,
            "img_auprc":  ri, "ts_auprc":  rt, "fus_auprc":  rf,
            "gap_i2f_pr": rf - ri,  "gap_t2f_pr": rf - rt,
        })

    return {
        "labels":     list(pathology_labels),
        "n":          int(len(y)),
        "main_auroc": per_label[0]["fus_auroc"],   # Edema fusion = main task
        "main_auprc": per_label[0]["fus_auprc"],
        "per_label":  per_label,
    }


def format_dual_pathology_gap_table(result: dict) -> str:
    """3-way 성능표: img / ts / fusion AUROC · AUPRC."""
    header = (f"{'label':<22s} {'n':>6s} {'pos':>7s} "
              f"{'img_roc':>8s} {'ts_roc':>8s} {'fus_roc':>8s}  "
              f"{'img_prc':>8s} {'ts_prc':>8s} {'fus_prc':>8s}")
    lines = [header]
    for r in result["per_label"]:
        lines.append(
            f"{r['name']:<22s} {r['n_valid']:>6d} {r['pos_frac']:>7.4f} "
            f"{r['img_auroc']:>8.4f} {r['ts_auroc']:>8.4f} {r['fus_auroc']:>8.4f}  "
            f"{r['img_auprc']:>8.4f} {r['ts_auprc']:>8.4f} {r['fus_auprc']:>8.4f}"
        )
    return "\n".join(lines)
