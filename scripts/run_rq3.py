"""RQ3 analysis: signal value, confidence calibration, predictive model.

Usage:
    python scripts/run_rq3.py

Reads  : data/tweets_validated.csv
Writes : figures/rq3_calibration_curve.png
         figures/rq3_feature_importance.png
Prints : (a) calibration table + correlation statistics
         (b) model AUCs vs baseline
         (c) top features from both LR and GB
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.stats_rq3 import compute_rq3, build_feature_matrix

DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
FIGURES_DIR = pathlib.Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

INPUT_CSV   = DATA_DIR / "tweets_validated.csv"
FIG_CALIB   = FIGURES_DIR / "rq3_calibration_curve.png"
FIG_IMPORT  = FIGURES_DIR / "rq3_feature_importance.png"


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_calibration(res: dict) -> None:
    calib = res["calibration"]
    corr  = res["correlation"]

    print()
    print("=" * 72)
    print("RQ3 — Confidence calibration")
    print("=" * 72)
    header = (
        f"  {'Bin range':>14}  {'Midpoint':>8}  {'n':>6}  "
        f"{'Actual acc':>10}  {'95% CI':^18}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in calib.iterrows():
        bin_str = f"[{row['bin_lower']:.0f}, {row['bin_upper']:.0f})"
        ci_str  = f"[{row['ci_low']*100:.1f}%, {row['ci_high']*100:.1f}%]"
        print(
            f"  {bin_str:>14}  {row['bin_mid']:>8.1f}  "
            f"{int(row['n_predictions']):>6}  "
            f"{row['actual_accuracy']*100:>9.1f}%  {ci_str:^18}"
        )

    print()
    print("  Pearson correlation (confidence vs outcome):")
    print(f"    vs prediction_correct : r = {corr['r_correct']:+.4f}  "
          f"p = {corr['p_correct']:.4f}  "
          f"95% CI [{corr['ci_correct_low']:+.4f}, {corr['ci_correct_high']:+.4f}]")
    print(f"    vs beats_market       : r = {corr['r_beats']:+.4f}  "
          f"p = {corr['p_beats']:.4f}  "
          f"95% CI [{corr['ci_beats_low']:+.4f}, {corr['ci_beats_high']:+.4f}]")
    print(f"    n = {corr['n']:,}")


def print_model_results(res: dict) -> None:
    print()
    print("=" * 72)
    print("RQ3 — Predictive model results (AUC-ROC, 5-fold stratified CV)")
    print("=" * 72)
    print(f"  Baseline (no-skill) AUC : {res['baseline_auc']:.3f}")
    print(f"  Majority-class accuracy : {res['baseline_accuracy']*100:.1f}%")
    print()
    print(f"  Logistic regression AUC : {res['lr']['auc_cv']:.4f}  "
          f"(vs baseline {res['baseline_auc']:.3f}, "
          f"delta = {res['lr']['auc_cv'] - res['baseline_auc']:+.4f})")
    print(f"  Gradient boosting AUC   : {res['gb']['auc_cv']:.4f}  "
          f"(vs baseline {res['baseline_auc']:.3f}, "
          f"delta = {res['gb']['auc_cv'] - res['baseline_auc']:+.4f})")

    print()
    print("  Top 10 LR standardised coefficients (by |coef|):")
    lr_top = res["lr"]["coefs"].head(10)
    for feat, coef in lr_top.items():
        or_  = res["lr"]["odds_ratios"][feat]
        print(f"    {feat:<30}  coef={coef:+.4f}  OR={or_:.4f}")

    print()
    print("  Top 10 GB feature importances:")
    gb_top = res["gb"]["importances"].head(10)
    for feat, imp in gb_top.items():
        print(f"    {feat:<30}  importance={imp:.4f}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_calibration_figure(res: dict, out_path: pathlib.Path) -> None:
    calib  = res["calibration"]
    corr   = res["correlation"]
    mids   = calib["bin_mid"].values
    accs   = calib["actual_accuracy"].values
    ns     = calib["n_predictions"].values
    ci_lo  = calib["ci_low"].values
    ci_hi  = calib["ci_high"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    # Error bars
    yerr = np.array([accs - ci_lo, ci_hi - accs])
    ax.errorbar(
        mids, accs, yerr=yerr,
        fmt="o", color="#2F5496", markersize=6, capsize=4,
        linewidth=1.4, elinewidth=1.0, label="Actual accuracy (Wilson CI)",
    )

    # Scale dot size by n (cosmetic)
    sizes = np.clip(ns / ns.max() * 200, 30, 200)
    ax.scatter(mids, accs, s=sizes, color="#2F5496", zorder=5, alpha=0.5)

    # Overall accuracy reference line — weighted mean matches Table 8.1 / §6.1
    overall_acc = np.average(calib["actual_accuracy"].values, weights=ns)
    ax.axhline(
        overall_acc, color="#ED7D31", linestyle="--", linewidth=1.2,
        label=f"Overall accuracy ≈ {overall_acc*100:.1f}%",
    )

    # Perfect calibration (y = x/100)
    x_line = np.linspace(0, 100, 200)
    ax.plot(
        x_line, x_line / 100.0,
        color="gray", linestyle=":", linewidth=1.0, alpha=0.8,
        label="Perfect calibration (y = x/100)",
    )

    r_str = f"r = {corr['r_correct']:+.4f} (p = {corr['p_correct']:.3f})"
    ax.set_xlabel("Predicted confidence score", fontsize=11)
    ax.set_ylabel("Actual prediction_correct rate", fontsize=11)
    ax.set_title(
        f"RQ3 — Reliability curve: LLM confidence vs actual accuracy\n{r_str}",
        fontsize=11,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")


def make_importance_figure(res: dict, out_path: pathlib.Path) -> None:
    top_n = 12

    lr_coefs  = res["lr"]["coefs"].head(top_n)
    gb_imps   = res["gb"]["importances"].head(top_n)

    fig, (ax_lr, ax_gb) = plt.subplots(1, 2, figsize=(13, 5))

    # --- LR coefficients ---
    colors_lr = ["#2F5496" if v >= 0 else "#C00000" for v in lr_coefs.values]
    y_lr = np.arange(len(lr_coefs))
    ax_lr.barh(y_lr, lr_coefs.values, color=colors_lr, edgecolor="white", linewidth=0.5)
    ax_lr.set_yticks(y_lr)
    ax_lr.set_yticklabels(lr_coefs.index, fontsize=8.5)
    ax_lr.axvline(0, color="black", linewidth=0.8)
    ax_lr.set_xlabel("Standardised coefficient", fontsize=10)
    ax_lr.set_title(
        f"Logistic Regression\nTop {top_n} features by |coef|  "
        f"(AUC = {res['lr']['auc_cv']:.3f})",
        fontsize=10,
    )
    ax_lr.grid(axis="x", linestyle=":", alpha=0.35)
    ax_lr.spines[["top", "right"]].set_visible(False)

    # --- GB importances ---
    y_gb = np.arange(len(gb_imps))
    ax_gb.barh(y_gb, gb_imps.values, color="#2F5496", edgecolor="white", linewidth=0.5)
    ax_gb.set_yticks(y_gb)
    ax_gb.set_yticklabels(gb_imps.index, fontsize=8.5)
    ax_gb.set_xlabel("Feature importance (impurity decrease)", fontsize=10)
    ax_gb.set_title(
        f"Gradient Boosting\nTop {top_n} features  "
        f"(AUC = {res['gb']['auc_cv']:.3f})",
        fontsize=10,
    )
    ax_gb.grid(axis="x", linestyle=":", alpha=0.35)
    ax_gb.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        "RQ3 — Feature importance: what predicts prediction accuracy?",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(
            f"ERROR: {INPUT_CSV} not found.\n"
            "Run scripts/validate_predictions.py first."
        )

    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print("Running RQ3 analysis (this may take 1-2 minutes for gradient boosting CV) ...")
    res = compute_rq3(df)

    print_calibration(res)
    print_model_results(res)
    make_calibration_figure(res, FIG_CALIB)
    make_importance_figure(res, FIG_IMPORT)

    print("\nDone.")


if __name__ == "__main__":
    main()
