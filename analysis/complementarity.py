"""Modality complementarity analysis for dual/dual_patch teacher.

각 pathology 마다 세 head (image / temporal / fusion) 가 같은 환자를 맞추고
틀리는 패턴을 정량화한다. 두 층으로 결과가 나온다:

  Level 1 — 2x2 (image vs TS)  → 표현 상에 상보성이 존재하는가?
      both / img_only / ts_only / both_wrong
      → ts_unique_gain, ts_redundancy, kappa

  Level 2 — 3-way (image x TS x fusion)  → fusion 이 그 상보성을 활용하는가?
      핵심 지표: ts_gain_retention  = ts_only_and_fus_ok / (ts_only_and_fus_ok + ts_only_but_fus_lost_it)
                  fusion_harm_rate  = image_only_but_fus_lost_it / (image_only_and_fus_ok + image_only_but_fus_lost_it)
                  emergent_gain     = both_wrong_but_fus_saved   / (both_wrong_but_fus_saved  + all_three_wrong)
      셀 이름 (4 그룹 × fusion 성공 여부 = 8 cells):
          image ✓ AND TS ✓  →  both_correct_and_fus_ok       / both_correct_but_fus_broke_it
          image ✓ AND TS ✗  →  image_only_and_fus_ok         / image_only_but_fus_lost_it
          image ✗ AND TS ✓  →  ts_only_and_fus_ok            / ts_only_but_fus_lost_it
          image ✗ AND TS ✗  →  both_wrong_but_fus_saved      / all_three_wrong

Threshold 는 val split 에서 Youden's J 로 pathology × head 별 각자 학습.

Usage:
    python analysis/complementarity.py \\
        --ckpt <path>/best.pt \\
        --outdir analysis/figs_<name>/complementarity
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training_duett.engine import _move_lists                         # noqa: E402
from analysis.visualize_pathology import _loader, load_teacher        # noqa: E402

try:
    from matplotlib_venn import venn3
    HAS_VENN = True
except ImportError:
    HAS_VENN = False


def parse_args():
    p = argparse.ArgumentParser("Modality complementarity analysis")
    p.add_argument("--ckpt", type=str, required=True, help="best.pt path")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--val_split", type=str, default="val",
                   help="threshold 계산용 split")
    p.add_argument("--test_split", type=str, default="test",
                   help="contingency 계산용 split")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=str, default="youden",
                   choices=["youden", "fixed"],
                   help="youden: val split 에서 TPR-FPR 최대점; "
                        "fixed: logit 0 (=prob 0.5)")
    p.add_argument("--labels", type=str, default="",
                   help="분석할 pathology 이름들 (콤마 구분). 비우면 전체. 예: 'edema'. "
                        "threshold 계산은 전체 label 로 이뤄지고 리포트/Venn 만 필터됨.")
    return p.parse_args()


# =============================================================================
# Data collection
# =============================================================================
@torch.no_grad()
def _gather(teacher, loader, device):
    imgs, tss, fuss, ys, mks = [], [], [], [], []
    for batch in loader:
        b = _move_lists(batch, device)
        out = teacher(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"])
        if not isinstance(out, dict) or "fusion_logits" not in out:
            raise RuntimeError("complementarity 는 dual/dual_patch 모드 전용 "
                               "(img_logits/ts_logits/fusion_logits 필요)")
        imgs.append(out["img_logits"].cpu())
        tss.append(out["ts_logits"].cpu())
        fuss.append(out["fusion_logits"].cpu())
        ys.append(b["y_multi"].cpu())
        mks.append(b["y_multi_mask"].cpu())
    return {
        "img":  torch.cat(imgs).float().numpy(),
        "ts":   torch.cat(tss).float().numpy(),
        "fus":  torch.cat(fuss).float().numpy(),
        "y":    torch.cat(ys).float().numpy(),
        "mask": torch.cat(mks).float().numpy(),
    }


# =============================================================================
# Threshold derivation
# =============================================================================
def _youden_threshold(logits, y):
    """Return logit threshold maximizing TPR-FPR. NaN if one class only."""
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, thr = roc_curve(y, logits)
    return float(thr[int(np.argmax(tpr - fpr))])


def _derive_thresholds(val_data, K, method):
    if method == "fixed":
        return {mod: np.zeros(K) for mod in ("img", "ts", "fus")}
    out = {}
    for mod in ("img", "ts", "fus"):
        thr = np.full(K, np.nan)
        for k in range(K):
            m = val_data["mask"][:, k].astype(bool)
            if m.sum() < 2:
                continue
            thr[k] = _youden_threshold(val_data[mod][m, k], val_data["y"][m, k])
        out[mod] = thr
    return out


def _binarize(data, thresholds, K):
    out = {}
    for mod in ("img", "ts", "fus"):
        thr = thresholds[mod]
        pred = np.zeros_like(data[mod], dtype=bool)
        for k in range(K):
            if not np.isnan(thr[k]):
                pred[:, k] = data[mod][:, k] > thr[k]
        out[mod] = pred
    return out


# =============================================================================
# Per-pathology analysis
# =============================================================================
def _cohens_kappa(x_bool, y_bool):
    n = len(x_bool)
    if n == 0:
        return float("nan")
    po = float((x_bool == y_bool).mean())
    px, py = float(x_bool.mean()), float(y_bool.mean())
    pe = px * py + (1 - px) * (1 - py)
    if 1 - pe == 0:
        return float("nan")
    return (po - pe) / (1 - pe)


def _pearson(a, b):
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _analyze_pathology(k, label, data, preds):
    m = data["mask"][:, k].astype(bool)
    y = data["y"][m, k].astype(bool)
    if m.sum() == 0:
        empty = {"label": label, "n": 0, "pos_frac": float("nan")}
        for key in ("img_acc", "ts_acc", "fus_acc",
                    "ts_unique_gain", "ts_redundancy", "coverage_gain",
                    "kappa_img_ts", "err_corr",
                    "ts_gain_retention", "fusion_harm_rate",
                    "emergent_gain", "both_agree_broken"):
            empty[key] = float("nan")
        for key in ("both_correct", "image_only_correct",
                    "ts_only_correct", "both_wrong",
                    "ts_only_and_fus_ok", "ts_only_but_fus_lost_it",
                    "image_only_and_fus_ok", "image_only_but_fus_lost_it",
                    "both_wrong_but_fus_saved", "all_three_wrong",
                    "both_correct_and_fus_ok", "both_correct_but_fus_broke_it"):
            empty[key] = 0
        return empty

    ip = preds["img"][m, k]
    tp = preds["ts"][m, k]
    fp = preds["fus"][m, k]

    ic = (ip == y)   # image correct
    tc = (tp == y)   # TS correct
    fc = (fp == y)   # fusion correct

    n = int(m.sum())
    both_correct       = int(( ic &  tc).sum())
    image_only_correct = int(( ic & ~tc).sum())
    ts_only_correct    = int((~ic &  tc).sum())
    both_wrong         = int((~ic & ~tc).sum())

    ts_only_and_fus_ok            = int((~ic &  tc &  fc).sum())
    ts_only_but_fus_lost_it       = int((~ic &  tc & ~fc).sum())
    image_only_and_fus_ok         = int(( ic & ~tc &  fc).sum())
    image_only_but_fus_lost_it    = int(( ic & ~tc & ~fc).sum())
    both_wrong_but_fus_saved      = int((~ic & ~tc &  fc).sum())
    all_three_wrong               = int((~ic & ~tc & ~fc).sum())
    both_correct_and_fus_ok       = int(( ic &  tc &  fc).sum())
    both_correct_but_fus_broke_it = int(( ic &  tc & ~fc).sum())

    def _ratio(num, den):
        return num / den if den > 0 else float("nan")

    return {
        "label": label, "n": n, "pos_frac": float(y.mean()),
        "img_acc": float(ic.mean()), "ts_acc": float(tc.mean()),
        "fus_acc": float(fc.mean()),
        # Level 1
        "both_correct": both_correct,
        "image_only_correct": image_only_correct,
        "ts_only_correct": ts_only_correct,
        "both_wrong": both_wrong,
        "ts_unique_gain":    ts_only_correct / n,
        "ts_redundancy":     _ratio(both_correct, both_correct + ts_only_correct),
        "coverage_gain":     (both_correct + image_only_correct + ts_only_correct) / n,
        "kappa_img_ts":      _cohens_kappa(ic, tc),
        "err_corr":          _pearson((~ic).astype(float), (~tc).astype(float)),
        # Level 2 — 8 cells
        "ts_only_and_fus_ok":            ts_only_and_fus_ok,
        "ts_only_but_fus_lost_it":       ts_only_but_fus_lost_it,
        "image_only_and_fus_ok":         image_only_and_fus_ok,
        "image_only_but_fus_lost_it":    image_only_but_fus_lost_it,
        "both_wrong_but_fus_saved":      both_wrong_but_fus_saved,
        "all_three_wrong":               all_three_wrong,
        "both_correct_and_fus_ok":       both_correct_and_fus_ok,
        "both_correct_but_fus_broke_it": both_correct_but_fus_broke_it,
        # Ratios
        "ts_gain_retention":     _ratio(ts_only_and_fus_ok,
                                        ts_only_and_fus_ok + ts_only_but_fus_lost_it),
        "fusion_harm_rate":      _ratio(image_only_but_fus_lost_it,
                                        image_only_and_fus_ok + image_only_but_fus_lost_it),
        "emergent_gain":         _ratio(both_wrong_but_fus_saved,
                                        both_wrong_but_fus_saved + all_three_wrong),
        "both_agree_broken_rate": _ratio(both_correct_but_fus_broke_it,
                                         both_correct_and_fus_ok + both_correct_but_fus_broke_it),
    }


# =============================================================================
# Reporting
# =============================================================================
def _fmt(v, spec="7.3f"):
    import math
    width = spec.split(".")[0].lstrip("+")
    try:
        if math.isnan(float(v)):
            return f"{'--':>{width}s}"
    except (TypeError, ValueError):
        return f"{'--':>{width}s}"
    return f"{v:{spec}}"


def _print_report(rows, thresholds, pathology_labels):
    print("\n=== Per-pathology thresholds (logit units) ===")
    print(f"{'':>4s}  " + "  ".join(f"{lbl[:12]:>12s}" for lbl in pathology_labels))
    for mod in ("img", "ts", "fus"):
        line = "  ".join(_fmt(t, "12.4f") for t in thresholds[mod])
        print(f"{mod:>4s}  {line}")

    print("\n=== Level 1: image vs TS  (does complementarity exist?) ===")
    print(f"{'label':<14s} {'n':>5s} {'img_acc':>7s} {'ts_acc':>7s} "
          f"{'both_ok':>7s} {'img_only':>8s} {'ts_only':>7s} {'both_wr':>7s} "
          f"{'ts_gain':>7s} {'ts_redun':>8s} {'kappa':>6s} {'err_r':>6s}")
    for r in rows:
        print(f"{r['label'][:14]:<14s} {r['n']:>5d} "
              f"{_fmt(r['img_acc'], '7.3f')} {_fmt(r['ts_acc'], '7.3f')} "
              f"{r['both_correct']:>7d} {r['image_only_correct']:>8d} "
              f"{r['ts_only_correct']:>7d} {r['both_wrong']:>7d} "
              f"{_fmt(r['ts_unique_gain'], '7.3f')} "
              f"{_fmt(r['ts_redundancy'], '8.3f')} "
              f"{_fmt(r['kappa_img_ts'], '6.3f')} "
              f"{_fmt(r['err_corr'], '6.3f')}")

    print("\n=== Level 2: 3-way with fusion  (does fusion capture it?) ===")
    print("cells: fus_ok / fus_bad")
    print(f"{'label':<14s} {'fus_acc':>7s} "
          f"{'ts_retain':>9s} {'fus_harm':>8s} {'emergent':>8s} "
          f"{'ts_only':>9s} {'img_only':>9s} {'both_wr':>9s} {'both_ok':>9s}")
    for r in rows:
        print(f"{r['label'][:14]:<14s} {_fmt(r['fus_acc'], '7.3f')} "
              f"{_fmt(r['ts_gain_retention'], '9.3f')} "
              f"{_fmt(r['fusion_harm_rate'], '8.3f')} "
              f"{_fmt(r['emergent_gain'], '8.3f')} "
              f"{r['ts_only_and_fus_ok']:>4d}/{r['ts_only_but_fus_lost_it']:<4d} "
              f"{r['image_only_and_fus_ok']:>4d}/{r['image_only_but_fus_lost_it']:<4d} "
              f"{r['both_wrong_but_fus_saved']:>4d}/{r['all_three_wrong']:<4d} "
              f"{r['both_correct_and_fus_ok']:>4d}/{r['both_correct_but_fus_broke_it']:<4d}")


def _save_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[complementarity] csv → {path}")


def _slug(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", label).strip("_") or "label"


def _plot_venn(k, label, data, preds, out_png):
    if not HAS_VENN:
        return False
    m = data["mask"][:, k].astype(bool)
    y_pos = (data["y"][:, k] == 1) & m
    n_pos = int(y_pos.sum())
    if n_pos == 0:
        return False
    idx = np.where(y_pos)[0]
    img_set = set(int(i) for i in idx[preds["img"][idx, k]])
    ts_set  = set(int(i) for i in idx[preds["ts"][idx, k]])
    fus_set = set(int(i) for i in idx[preds["fus"][idx, k]])

    counts = {
        "100": len(img_set - ts_set - fus_set),
        "010": len(ts_set - img_set - fus_set),
        "110": len((img_set & ts_set) - fus_set),
        "001": len(fus_set - img_set - ts_set),
        "101": len((img_set & fus_set) - ts_set),
        "011": len((ts_set & fus_set) - img_set),
        "111": len(img_set & ts_set & fus_set),
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    # 세 원을 동일 크기로 그리기 위해 subset area 를 모두 1 로 두고,
    # 실제 count 는 label 로 덮어쓴다.
    v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1),
              set_labels=("image", "TS", "fusion"), ax=ax)
    for region_id, count in counts.items():
        lbl = v.get_label_by_id(region_id)
        if lbl is not None:
            lbl.set_text(str(count))
    ax.set_title(f"{label} — positives caught (n_pos={n_pos})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[complementarity] device={device}")

    teacher, bundle, processor, ck_args, pathology_labels, mode = load_teacher(
        args.ckpt, device
    )
    print(f"[complementarity] loaded ckpt: {args.ckpt}  mode={mode}")
    print(f"[complementarity] pathology_labels: {pathology_labels}")
    if mode not in ("dual", "dual_patch"):
        raise SystemExit("complementarity 는 dual/dual_patch teacher 전용")

    K = len(pathology_labels)
    if args.val_split not in bundle["datasets"]:
        raise SystemExit(f"val_split '{args.val_split}' not in datasets: "
                         f"{list(bundle['datasets'])}")
    if args.test_split not in bundle["datasets"]:
        raise SystemExit(f"test_split '{args.test_split}' not in datasets: "
                         f"{list(bundle['datasets'])}")

    val_loader  = _loader(bundle["datasets"][args.val_split],
                          args.batch_size, args.num_workers)
    test_loader = _loader(bundle["datasets"][args.test_split],
                          args.batch_size, args.num_workers)

    print(f"[complementarity] gathering {args.val_split} for thresholds...")
    val_data  = _gather(teacher, val_loader, device)
    print(f"[complementarity] gathering {args.test_split} for evaluation...")
    test_data = _gather(teacher, test_loader, device)

    print(f"[complementarity] threshold method: {args.threshold}")
    thresholds = _derive_thresholds(val_data, K, args.threshold)
    preds = _binarize(test_data, thresholds, K)

    label_to_idx = {lbl.lower(): i for i, lbl in enumerate(pathology_labels)}
    requested = [s.strip().lower() for s in args.labels.split(",") if s.strip()]
    unknown = [n for n in requested if n not in label_to_idx]
    if unknown:
        raise SystemExit(f"--labels 에 알 수 없는 pathology: {unknown}. "
                         f"사용 가능: {list(label_to_idx)}")
    analyze_idx = tuple(label_to_idx[n] for n in requested) if requested else tuple(range(K))
    if requested:
        print(f"[complementarity] --labels 필터: {[pathology_labels[i] for i in analyze_idx]}")

    filtered_labels     = tuple(pathology_labels[i] for i in analyze_idx)
    filtered_thresholds = {mod: thresholds[mod][list(analyze_idx)]
                           for mod in ("img", "ts", "fus")}
    rows = [_analyze_pathology(k, pathology_labels[k], test_data, preds)
            for k in analyze_idx]

    _print_report(rows, filtered_thresholds, filtered_labels)
    _save_csv(rows, os.path.join(args.outdir, "complementarity.csv"))

    if HAS_VENN:
        n_drawn = 0
        for k in analyze_idx:
            out_png = os.path.join(args.outdir, f"venn_pos_{_slug(pathology_labels[k])}.png")
            if _plot_venn(k, pathology_labels[k], test_data, preds, out_png):
                n_drawn += 1
        print(f"[complementarity] venn diagrams: {n_drawn}/{len(analyze_idx)} → {args.outdir}/venn_pos_*.png")
    else:
        print("[complementarity] matplotlib_venn 미설치 → Venn diagram 스킵. "
              "설치: pip install matplotlib-venn")

    print(f"\n[complementarity] done → {args.outdir}")


if __name__ == "__main__":
    main()
