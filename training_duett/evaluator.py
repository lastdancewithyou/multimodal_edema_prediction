"""Binary AUROC/AUPRC evaluation for teacher/student."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
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
def _bce_per_sample(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Numerically-stable element-wise BCE: max(l,0) - l*y + log(1+exp(-|l|))."""
    return np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Population Pearson correlation. NaN if either has zero variance."""
    if a.size < 2:
        return float("nan")
    va = a.std()
    vb = b.std()
    if va == 0 or vb == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate_dual_pathology(model, loader, device, pathology_labels: tuple[str, ...],
                             *, query_ref: torch.Tensor | None = None) -> dict:
    """DualPathologyPerceiver / PatchDualPathologyPerceiver teacher 평가.

    loader 전체에서 img_logits/ts_logits/fusion_logits + (residual mode 시)
    scaled_correction 을 모아 pathology 마다 valid mask 적용해 지표 계산.

    지표:
        - AUROC / AUPRC (per branch: img, ts, fus) + gain (fus - img)
        - BCE  (per branch) + delta_BCE = fus_BCE - img_BCE
        - mean_abs_correction : |scaled_correction| 의 sample-mean (residual mode 만)
        - correction_residual_corr : Pearson(scaled_correction, y - sigmoid(img_logit))
        - beta[k] : perceiver.beta 파라미터 값
        - query_cos[k] : cos(current temporal_queries[k], query_ref[k])
        - modal_query_cos[k] : cos(image_queries[k], temporal_queries[k])

    main_auroc = per_label[0]["fus_auroc"]  (index 0 label의 fusion AUROC).

    Parameters
    ----------
    query_ref : Optional[Tensor of shape [K, D]]
        학습 시작 시점의 temporal_queries 스냅샷. 없으면 query_cos 는 NaN.
    """
    from .engine import _move_lists

    model.eval()
    img_all, ts_all, fus_all, y_all, mask_all, corr_all = [], [], [], [], [], []
    has_correction = None
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
        if has_correction is None:
            has_correction = "scaled_correction" in out
        if has_correction:
            corr_all.append(out["scaled_correction"].cpu())

    img = torch.cat(img_all).float().numpy()
    ts  = torch.cat(ts_all).float().numpy()
    fus = torch.cat(fus_all).float().numpy()
    y   = torch.cat(y_all).float().numpy()
    mk  = torch.cat(mask_all).float().numpy()
    corr = torch.cat(corr_all).float().numpy() if has_correction else None

    # perceiver 에서 beta / current query 추출 (available 하면)
    unwrapped = model.module if hasattr(model, "module") else model
    perceiver = getattr(unwrapped, "perceiver", None)
    beta_vec = None
    if perceiver is not None and hasattr(perceiver, "beta"):
        beta_vec = perceiver.beta.detach().float().cpu().numpy()
    query_cos_vec = None
    if (query_ref is not None and perceiver is not None
            and hasattr(perceiver, "temporal_queries")):
        cur = perceiver.temporal_queries.detach().float()
        ref = query_ref.detach().float().to(cur.device)
        if cur.shape == ref.shape:
            query_cos_vec = F.cosine_similarity(cur, ref, dim=-1).cpu().numpy()
    modal_query_cos_vec = None
    if (perceiver is not None
            and hasattr(perceiver, "image_queries")
            and hasattr(perceiver, "temporal_queries")):
        img_q = perceiver.image_queries.detach().float()
        ts_q = perceiver.temporal_queries.detach().float()
        if img_q.shape == ts_q.shape:
            modal_query_cos_vec = F.cosine_similarity(
                img_q, ts_q, dim=-1
            ).cpu().numpy()

    K = len(pathology_labels)
    per_label = []
    for k in range(K):
        m = mk[:, k].astype(bool)
        yk = y[m, k]
        li, lt, lf = img[m, k], ts[m, k], fus[m, k]
        pi = 1.0 / (1.0 + np.exp(-li))
        pt = 1.0 / (1.0 + np.exp(-lt))
        pf = 1.0 / (1.0 + np.exp(-lf))
        ai, at, af = _safe(roc_auc_score, yk, pi), _safe(roc_auc_score, yk, pt), _safe(roc_auc_score, yk, pf)
        ri, rt, rf = _safe(average_precision_score, yk, pi), _safe(average_precision_score, yk, pt), _safe(average_precision_score, yk, pf)

        # BCE per branch (mean over valid samples)
        img_bce = float(_bce_per_sample(li, yk).mean()) if yk.size else float("nan")
        ts_bce  = float(_bce_per_sample(lt, yk).mean()) if yk.size else float("nan")
        fus_bce = float(_bce_per_sample(lf, yk).mean()) if yk.size else float("nan")

        # residual mode 전용 지표
        if corr is not None and yk.size:
            ck = corr[m, k]
            mean_abs_corr = float(np.abs(ck).mean())
            corr_r = _pearson(ck, yk - pi)
        else:
            mean_abs_corr = float("nan")
            corr_r = float("nan")

        beta_k     = float(beta_vec[k])      if beta_vec      is not None else float("nan")
        q_cos_k    = float(query_cos_vec[k]) if query_cos_vec is not None else float("nan")
        modal_q_cos_k = (float(modal_query_cos_vec[k])
                         if modal_query_cos_vec is not None else float("nan"))

        per_label.append({
            "name":         pathology_labels[k],
            "n_valid":      int(m.sum()),
            "pos_frac":     float(yk.mean()) if len(yk) else float("nan"),
            "img_auroc":    ai, "ts_auroc":  at, "fus_auroc":  af,
            "gap_i2f":      af - ai,  "gap_t2f": af - at,
            "img_auprc":    ri, "ts_auprc":  rt, "fus_auprc":  rf,
            "gap_i2f_pr":   rf - ri,  "gap_t2f_pr": rf - rt,
            "img_bce":      img_bce,
            "ts_bce":       ts_bce,
            "fus_bce":      fus_bce,
            "delta_bce":    fus_bce - img_bce,        # 음수면 fusion 개선
            "mean_abs_corr": mean_abs_corr,           # residual 사용량
            "corr_residual": corr_r,                  # 방향성 (양수=오류 수정 방향)
            "beta":         beta_k,
            "query_cos":    q_cos_k,
            "modal_query_cos": modal_q_cos_k,
        })

    return {
        "labels":     list(pathology_labels),
        "n":          int(len(y)),
        "main_auroc": per_label[0]["fus_auroc"],
        "main_auprc": per_label[0]["fus_auprc"],
        "per_label":  per_label,
    }


def _fmt(v, spec="7.3f"):
    """NaN-safe formatter — returns '--' for NaN."""
    import math
    width = spec.split(".")[0].lstrip("+")
    try:
        if math.isnan(float(v)):
            return f"{'--':>{width}s}"
    except (TypeError, ValueError):
        return f"{'--':>{width}s}"
    return f"{v:{spec}}"


def format_dual_pathology_gap_table(result: dict) -> str:
    """Residual fusion 종합 지표 표.

    label  imgROC  tsROC  fusROC  gain   imgPR  fusPR  dBCE  |corr|  corr_r  beta  q_cos
    """
    header = (
        f"{'label':<12s} "
        f"{'imgROC':>7s} {'tsROC':>7s} {'fusROC':>7s} {'gain':>7s}  "
        f"{'imgPRC':>6s} {'fusPRC':>6s}  "
        f"{'dBCE':>7s}  "
        f"{'|corr|':>7s} {'corr_r':>7s}  "
        f"{'beta':>6s} {'q_cos':>6s} {'i-t_cos':>7s}"
    )
    lines = [header, "-" * len(header)]
    for r in result["per_label"]:
        short = r["name"].replace("label_", "")
        lines.append(
            f"{short:<12s} "
            f"{_fmt(r['img_auroc'], '7.3f')} {_fmt(r['ts_auroc'], '7.3f')} "
            f"{_fmt(r['fus_auroc'], '7.3f')} {_fmt(r['gap_i2f'], '+7.3f')}  "
            f"{_fmt(r['img_auprc'], '6.3f')} {_fmt(r['fus_auprc'], '6.3f')}  "
            f"{_fmt(r['delta_bce'], '+7.4f')}  "
            f"{_fmt(r['mean_abs_corr'], '7.4f')} {_fmt(r['corr_residual'], '+7.3f')}  "
            f"{_fmt(r['beta'], '6.3f')} {_fmt(r['query_cos'], '6.3f')} "
            f"{_fmt(r['modal_query_cos'], '7.3f')}"
        )
    return "\n".join(lines)
