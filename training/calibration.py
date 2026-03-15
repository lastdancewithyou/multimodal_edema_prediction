import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn


class ExpectedCalibrationError:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def compute(self, y_true, y_prob):
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_prob, bins[1:-1])

        ece = 0.0
        bin_stats = []

        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue

            bin_acc = y_true[mask].mean()      # Actual accuracy in this bin
            bin_conf = y_prob[mask].mean()     # Average predicted probability
            bin_size = mask.sum()

            ece += (bin_size / len(y_true)) * abs(bin_acc - bin_conf)

            bin_stats.append({
                'bin_index': i,
                'bin_lower': bins[i],
                'bin_upper': bins[i+1],
                'accuracy': bin_acc,
                'confidence': bin_conf,
                'count': int(bin_size),
                'gap': bin_conf - bin_acc  # Positive = overconfident
            })

        return ece, bin_stats


def analyze_calibration(y_true_dict, y_prob_dict, save_dir='./output/calibration', prefix='analysis'):
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    print("\n" + "="*80)
    print("📊 Calibration Analysis")
    print("="*80)

    for task_name in y_true_dict.keys():
        y_true = y_true_dict[task_name]
        y_prob = y_prob_dict[task_name]

        print(f"\n[{task_name}]")

        if "subtype" in task_name.lower():
            class_names = ("NCPE", "CPE")
        else:
            class_names = ("Negative", "Positive")

        # Calculate ECE
        ece_calc = ExpectedCalibrationError(n_bins=10)
        ece, bin_stats = ece_calc.compute(y_true, y_prob)

        # Determine calibration quality
        if ece < 0.05:
            quality = "✅ Well-calibrated"
        else:
            quality = "❌ Poorly calibrated"

        print(f"  ECE: {ece:.4f} {quality}")

        # Plot calibration curve
        task_dir = os.path.join(save_dir, task_name.lower().replace(' ', '_'))
        os.makedirs(task_dir, exist_ok=True)

        plot_calibration_curve(
            y_true, y_prob,
            task_name=task_name,
            class_names=class_names,
            n_bins=10,
            save_path=f"{task_dir}/{prefix}_calibration_curve.png"
        )

        plot_reliability_diagram(
            y_true, y_prob,
            task_name=task_name,
            n_bins=10,
            save_path=f"{task_dir}/{prefix}_reliability_gaps.png"
        )

        results[task_name] = {
            'ece': ece,
            'mean_confidence': y_prob.mean(),
            'positive_rate': y_true.mean(),
            'bin_stats': bin_stats,
            'quality': quality
        }

    print("\n" + "="*80)

    return results


def plot_calibration_curve(y_true, y_prob, task_name='Binary', class_names=('Negative', 'Positive'), n_bins=10, save_path=None):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')

    ece_calc = ExpectedCalibrationError(n_bins=n_bins)
    ece, bin_stats = ece_calc.compute(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ==================== Left: Reliability Diagram ====================
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    ax1.plot(prob_pred, prob_true, 's-', linewidth=2, markersize=8,
            color='#2E86AB', label=f'{task_name}')


    ax1.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Probability', fontsize=12, fontweight='bold')
    ax1.set_title(f'Calibration Curve\n{task_name} - ECE={ece:.4f}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # ==================== Right: Confidence Distribution ====================
    pos_mask = (y_true == 1) # 실제 환자의 윈도우 positive label 
    neg_mask = (y_true == 0) # 실제 환자의 윈도우 negative label

    pos_probs = y_prob[pos_mask]
    neg_probs = y_prob[neg_mask]

    bins = np.linspace(0, 1, 21)

    # Negative Histogram
    neg_name, pos_name = class_names
    ax2.hist(
        neg_probs,
        bins=bins,
        range=(0, 1),
        density=True,
        alpha=0.35,
        color='#64B5F6',
        edgecolor='#1E88E5',
        linewidth=1,
        label=f'{neg_name} (n={len(neg_probs):,})'
    )

    # Positive Histogram
    ax2.hist(
        pos_probs,
        bins=bins,
        range=(0, 1),
        density=True,
        alpha=0.35,
        color='#E57373',
        edgecolor='#C62828',
        linewidth=1,
        label=f'{pos_name} (n={len(pos_probs):,})'
    )

    # ==================== Median Lines ====================
    pos_median = np.median(pos_probs)
    neg_median = np.median(neg_probs)

    ax2.axvline(
        neg_median,
        color='#0D47A1',
        linestyle='--',
        linewidth=2,
        label=f'{neg_name} Median = {neg_median:.3f}'
    )

    ax2.axvline(
        pos_median,
        color='#B71C1C',
        linestyle='--',
        linewidth=2,
        label=f'{pos_name} Median = {pos_median:.3f}'
    )

    # ==================== Formatting ====================
    ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, frameon=True, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, 1])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Calibration plot saved: {save_path}")

    plt.close()

    return fig, ece, bin_stats


def plot_reliability_diagram(y_true, y_prob, task_name='Binary', n_bins=10, save_path=None):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')

    ece_calc = ExpectedCalibrationError(n_bins=n_bins)
    ece, bin_stats = ece_calc.compute(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect', alpha=0.7) # Perfect calibration

    # Calibration curve
    ax.plot(prob_pred, prob_true, 'o-', linewidth=3, markersize=8, color='#2E86AB', label=f'{task_name}')
    ax.set_xlabel('Predicted Probability', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Probability', fontsize=14, fontweight='bold')
    ax.set_title(f'{task_name} - Reliability Diagram\nECE = {ece:.4f}',
                fontsize=15, fontweight='bold')

    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Reliability diagram saved: {save_path}")

    plt.close()

    return fig, bin_stats





# def compute_hierarchical_calibration(edema_true, edema_prob, subtype_true, subtype_prob_ncpe, subtype_prob_cpe):
#     """
#     Compute calibration for hierarchical 3-class prediction

#     Args:
#         edema_true: [N] - binary edema labels (0/1)
#         edema_prob: [N] - P(edema=1)
#         subtype_true: [N] - subtype labels (0=NCPE, 1=CPE) for positive edema samples
#         subtype_prob_ncpe: [N] - P(NCPE | edema=1)
#         subtype_prob_cpe: [N] - P(CPE | edema=1)

#     Returns:
#         class_probs: dict - 3-class probabilities
#         class_true: dict - 3-class true labels
#     """
#     # Compute 3-class probabilities
#     p_negative = 1 - edema_prob
#     p_ncpe = edema_prob * subtype_prob_ncpe
#     p_cpe = edema_prob * subtype_prob_cpe

#     # Convert to 3-class labels
#     class_labels = np.zeros(len(edema_true), dtype=int)
#     class_labels[edema_true == 0] = 0  # Negative

#     # For positive samples, use subtype
#     positive_mask = edema_true == 1
#     if positive_mask.sum() > 0:
#         class_labels[positive_mask] = subtype_true[positive_mask] + 1  # 1=NCPE, 2=CPE

#     return {
#         'Negative': p_negative,
#         'NCPE': p_ncpe,
#         'CPE': p_cpe,
#         'labels': class_labels
#     }
