import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def net_benefit_from_counts(tp: int, fp: int, n: int, pt: float) -> float:
    """
    Calculate Net Benefit from confusion matrix counts.

    Net Benefit quantifies clinical utility:
        NB = (TP/N) - (FP/N) * (pt/(1-pt))
    """
    if n <= 0:
        return np.nan
    if pt <= 0 or pt >= 1:
        return np.nan

    w = pt / (1 - pt)  # Harm-to-benefit ratio
    return (tp / n) - (fp / n) * w


def compute_dca(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    assert y_true.ndim == 1 and y_prob.ndim == 1, "y_true and y_prob must be 1D arrays"
    assert len(y_true) == len(y_prob), "y_true and y_prob must have same length"
    assert set(y_true).issubset({0, 1}), "y_true must be binary (0 or 1)"

    n = len(y_true)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    # Treat-none baseline: no interventions → TP=0, FP=0 → NB=0
    nb_none = np.zeros_like(thresholds, dtype=float)

    # Treat-all baseline: intervene for everyone
    tp_all = int((y_true == 1).sum())
    fp_all = int((y_true == 0).sum())
    nb_all = np.array(
        [net_benefit_from_counts(tp_all, fp_all, n, pt) for pt in thresholds],
        dtype=float
    )

    # Model-based strategy: compute for each threshold
    rows = []
    for pt in thresholds:
        alert = (y_prob >= pt)  # Binary alert decisions
        tp = int(((alert) & (y_true == 1)).sum())
        fp = int(((alert) & (y_true == 0)).sum())
        nb_model = net_benefit_from_counts(tp, fp, n, pt)
        rows.append((pt, nb_model, tp, fp, int(alert.sum())))

    dca_df = pd.DataFrame(rows, columns=["threshold", "nb_model", "tp", "fp", "n_alerts"])
    dca_df["nb_all"] = nb_all
    dca_df["nb_none"] = nb_none

    return dca_df


def apply_cooldown_alerts(
    df: pd.DataFrame,
    prob_col: str,
    threshold: float,
    patient_col: str,
    time_col: str,
    cooldown_hours: float = 0.0,
) -> np.ndarray:
    """
    Apply patient-level cooldown policy to suppress repeated alerts.

    Cooldown logic:
    - Sort windows by (patient, time)
    - If an alert triggers at time t, suppress further alerts for that patient
      until time >= t + cooldown_hours

    This is useful for clinical applications where multiple alerts within
    a short time window may be redundant.

    Args:
        df: DataFrame containing predictions
        prob_col: Column name for predicted probabilities
        threshold: Probability threshold for alerting
        patient_col: Column name for patient identifier
        time_col: Column name for time (numeric hours or datetime)
        cooldown_hours: Minimum time between alerts for same patient

    Returns:
        Boolean array indicating which windows trigger alerts (length = len(df))

    Example:
        >>> alerts = apply_cooldown_alerts(
        ...     df=test_df,
        ...     prob_col='edema_prob',
        ...     threshold=0.5,
        ...     patient_col='stay_id',
        ...     time_col='hours_since_admission',
        ...     cooldown_hours=12.0
        ... )
    """
    assert 0 < threshold < 1, "threshold must be in (0, 1)"

    d = df[[patient_col, time_col, prob_col]].copy()
    d["_idx"] = np.arange(len(d))

    # Sort by patient, then time
    d = d.sort_values([patient_col, time_col], kind="mergesort")

    alerts = np.zeros(len(d), dtype=bool)

    # Determine time type
    is_datetime = np.issubdtype(d[time_col].dtype, np.datetime64)

    last_alert_time = {}  # patient -> last alert time

    for row in d.itertuples(index=False):
        pid = getattr(row, patient_col)
        t = getattr(row, time_col)
        p = getattr(row, prob_col)
        idx = getattr(row, "_idx")

        # Skip if probability below threshold
        if p < threshold:
            continue

        # First alert for this patient
        if pid not in last_alert_time:
            alerts[idx] = True
            last_alert_time[pid] = t
            continue

        # No cooldown
        if cooldown_hours <= 0:
            alerts[idx] = True
            last_alert_time[pid] = t
            continue

        # Check cooldown period
        prev_t = last_alert_time[pid]
        if is_datetime:
            delta_hours = (t - prev_t) / np.timedelta64(1, "h")
        else:
            delta_hours = float(t) - float(prev_t)

        if delta_hours >= cooldown_hours:
            alerts[idx] = True
            last_alert_time[pid] = t

    return alerts


def compute_dca_with_cooldown(
    df: pd.DataFrame,
    y_col: str,
    prob_col: str,
    patient_col: str,
    time_col: str,
    cooldown_hours: float,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute DCA with patient-level cooldown policy.

    This variant is more clinically realistic as it avoids counting
    multiple alerts for the same patient within a short time window.

    Treat-all/none baselines are still defined over all windows (standard practice).
    Model curve uses cooldown-adjusted alerts.

    Args:
        df: DataFrame with predictions (one row per window)
        y_col: Column name for ground truth labels
        prob_col: Column name for predicted probabilities
        patient_col: Column name for patient identifier
        time_col: Column name for time (numeric hours or datetime)
        cooldown_hours: Minimum hours between alerts for same patient
        thresholds: Array of threshold values to evaluate

    Returns:
        DataFrame with same columns as compute_dca()

    Example:
        >>> dca_df = compute_dca_with_cooldown(
        ...     df=test_df,
        ...     y_col='edema_label',
        ...     prob_col='edema_prob',
        ...     patient_col='stay_id',
        ...     time_col='hours_since_admission',
        ...     cooldown_hours=12.0
        ... )
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    y_true = df[y_col].to_numpy().astype(int)
    y_prob = df[prob_col].to_numpy().astype(float)
    n = len(df)

    # Treat-none baseline
    nb_none = np.zeros_like(thresholds, dtype=float)

    # Treat-all baseline
    tp_all = int((y_true == 1).sum())
    fp_all = int((y_true == 0).sum())
    nb_all = np.array(
        [net_benefit_from_counts(tp_all, fp_all, n, pt) for pt in thresholds],
        dtype=float
    )

    # Model with cooldown
    rows = []
    for pt in thresholds:
        alert = apply_cooldown_alerts(
            df=df,
            prob_col=prob_col,
            threshold=pt,
            patient_col=patient_col,
            time_col=time_col,
            cooldown_hours=cooldown_hours,
        )
        tp = int(((alert) & (y_true == 1)).sum())
        fp = int(((alert) & (y_true == 0)).sum())
        nb_model = net_benefit_from_counts(tp, fp, n, pt)
        rows.append((pt, nb_model, tp, fp, int(alert.sum())))

    dca_df = pd.DataFrame(rows, columns=["threshold", "nb_model", "tp", "fp", "n_alerts"])
    dca_df["nb_all"] = nb_all
    dca_df["nb_none"] = nb_none

    return dca_df


def plot_dca_curve(
    dca_df: pd.DataFrame,
    title: str = "Decision Curve Analysis",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 7),
    show_alerts: bool = False,
    policy_threshold: Optional[float] = None,
) -> Optional[str]:
    """
    Plot Decision Curve Analysis results.

    Args:
        dca_df: DataFrame from compute_dca() or compute_dca_with_cooldown()
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size (width, height)
        show_alerts: If True, add secondary y-axis showing number of alerts
        policy_threshold: If provided, draw vertical line at this threshold value

    Returns:
        Path to saved figure if save_path is provided, otherwise None

    Example:
        >>> plot_dca_curve(
        ...     dca_df=dca_results,
        ...     title="DCA: Edema Detection at 8-hour Horizon",
        ...     save_path="./output/dca/edema_dca.png",
        ...     show_alerts=True,
        ...     policy_threshold=0.4
        ... )
        >>> # Then display in notebook:
        >>> from IPython.display import Image, display
        >>> display(Image("./output/dca/edema_dca.png"))
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # Net Benefit curves
    ax1.plot(dca_df["threshold"], dca_df["nb_model"],
             label="Model", linewidth=2.5, color="#ff0000")
    ax1.plot(dca_df["threshold"], dca_df["nb_all"],
             label="Alert All", linewidth=2.5, linestyle="--", color="#2ca02c")
    ax1.plot(dca_df["threshold"], dca_df["nb_none"],
             label="Alert None", linewidth=2.5, linestyle=":", color="#0616eb")

    ax1.set_xlabel("Threshold Probability", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Net Benefit", fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # ⭐ Set y-axis limits for better visibility
    clinical_range = dca_df[dca_df['threshold'] <= 0.7] # to avoid extreme negative values from Treat All at high thresholds

    # Find y-axis range from Model and Treat All in clinical range
    y_min_model = clinical_range["nb_model"].min()
    y_max_model = clinical_range["nb_model"].max()
    y_min_all = clinical_range["nb_all"].min()
    y_max_all = clinical_range["nb_all"].max()

    y_min = min(y_min_model, y_min_all, 0)  # Include 0 (Treat None)
    y_max = max(y_max_model, y_max_all)

    # Add padding (15% on each side for better visibility)
    y_range = y_max - y_min
    y_padding = y_range * 0.15
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)

    # Optional: Add policy threshold vertical line
    if policy_threshold is not None:
        ax1.axvline(x=policy_threshold, color='gray', linestyle='--', linewidth=2.5, alpha=0.8, label=f'Policy Threshold ({policy_threshold:.1f})')

    # Optional: show number of alerts on secondary y-axis
    if show_alerts and 'n_alerts' in dca_df.columns:
        ax2 = ax1.twinx()
        alert_line = ax2.plot(dca_df["threshold"], dca_df["n_alerts"], color='red', alpha=0.3, linewidth=2.5, linestyle='-.', label='Number of Alerts')
        ax2.set_ylabel("Number of Alerts", fontsize=11, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.spines['right'].set_color('red')

        # Set secondary y-axis to start from 0
        ax2.set_ylim(0, dca_df["n_alerts"].max() * 1.1)

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11, framealpha=0.9)
    else:
        # Only primary axis legend
        ax1.legend(loc="upper right", fontsize=11, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📈 DCA plot saved: {save_path}")

    plt.close(fig)

    return save_path if save_path else None


def analyze_optimal_threshold(dca_df: pd.DataFrame, verbose: bool = True) -> dict:
    # Find threshold with maximum net benefit
    idx_max = dca_df["nb_model"].idxmax()
    optimal_row = dca_df.iloc[idx_max]

    # Find range where model outperforms both baselines
    better_than_all = dca_df["nb_model"] > dca_df["nb_all"]
    better_than_none = dca_df["nb_model"] > dca_df["nb_none"]
    useful_range = better_than_all & better_than_none

    if useful_range.any():
        useful_thresholds = dca_df.loc[useful_range, "threshold"]
        range_min = useful_thresholds.min()
        range_max = useful_thresholds.max()
    else:
        range_min = range_max = np.nan

    results = {
        "threshold": optimal_row["threshold"],
        "max_nb": optimal_row["nb_model"],
        "tp": optimal_row["tp"],
        "fp": optimal_row["fp"],
        "n_alerts": optimal_row["n_alerts"],
        "useful_range_min": range_min,
        "useful_range_max": range_max,
    }

    if verbose:
        print("\n" + "="*60)
        print("📊 Optimal Threshold Analysis")
        print("="*60)
        print(f"Optimal threshold: {results['threshold']:.3f}")
        print(f"Maximum net benefit: {results['max_nb']:.4f}")
        print(f"True positives: {results['tp']}")
        print(f"False positives: {results['fp']}")
        print(f"Number of alerts: {results['n_alerts']}")
        if not np.isnan(range_min):
            print(f"\nClinically useful range: [{range_min:.3f}, {range_max:.3f}]")
        else:
            print("\nWarning: Model does not outperform baselines at any threshold")
        print("="*60 + "\n")

    return results