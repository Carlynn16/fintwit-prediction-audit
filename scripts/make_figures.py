"""Generate all new report figures for the FinTwit Prediction Audit.

Usage:
    python scripts/make_figures.py

Produces 10 PNG files in figures/:
    data_tweets_per_month.png
    data_predictions_per_account.png
    data_top_tickers.png
    rq1_accuracy_by_year.png
    rq1_excess_return_hist.png
    rq2_multiple_testing_bar.png
    rq3_calibration_curve.png
    rq3_roc_curves.png
    rq3_confidence_hist.png
    rq4_regime_scatter.png
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

FIGURES_DIR = pathlib.Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── shared style constants ────────────────────────────────────────────────────
BLUE = "#2F5496"
ORANGE = "#ED7D31"
GREEN = "#70AD47"
CRIMSON = "#C00000"


def _clean_spines(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Monthly tweet volume
# ─────────────────────────────────────────────────────────────────────────────

def make_tweets_per_month(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    dates = pd.to_datetime(df["created_date"], errors="coerce")
    monthly = dates.dt.to_period("M").value_counts().sort_index()
    x = monthly.index.to_timestamp()
    y = monthly.values

    mean_count = y.mean()

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(x, y, color=BLUE, linewidth=1.5)
    ax.axhline(mean_count, color="gray", linestyle="--", alpha=0.6, label="Monthly mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Tweet count per month")
    ax.set_title("Monthly tweet volume (all accounts, 2021–2025)")
    ax.legend()
    ax.grid(linestyle=":", alpha=0.4)
    _clean_spines(ax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Validated predictions per account
# ─────────────────────────────────────────────────────────────────────────────

def make_predictions_per_account(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    val = df[df["validated"] == True]
    counts = val.groupby("author").size().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    ax.barh(counts.index, counts.values, color=BLUE, alpha=0.8)
    ax.axvline(30, color="crimson", linestyle="--", alpha=0.8, label="n=30 threshold")
    ax.set_xlabel("Number of validated predictions")
    ax.set_title("Validated predictions per account (n≥1)")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend()
    _clean_spines(ax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Top 15 most frequently predicted tickers
# ─────────────────────────────────────────────────────────────────────────────

def make_top_tickers(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    val = df[df["validated"] == True]
    top15 = val["prediction_ticker"].value_counts().head(15).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.barh(top15.index, top15.values, color=BLUE, alpha=0.85)
    ax.set_xlabel("Number of validated predictions")
    ax.set_title("Top 15 most frequently predicted tickers")
    ax.tick_params(axis="y", labelsize=9)
    _clean_spines(ax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Accuracy by year (RQ1)
# ─────────────────────────────────────────────────────────────────────────────

def make_accuracy_by_year(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    from src.stats_rq1 import wilson_ci

    val = df[df["validated"] == True].copy()

    def _as_bool(s):
        return s.astype(str).str.strip().str.lower().eq("true")

    val["prediction_correct_bool"] = _as_bool(val["prediction_correct"])
    val["beats_market_bool"] = _as_bool(val["beats_market"])
    val["year"] = pd.to_datetime(val["entry_date"], errors="coerce").dt.year

    years = [2021, 2022, 2023, 2024, 2025]
    acc_vals, acc_lo, acc_hi = [], [], []
    bm_vals, bm_lo, bm_hi = [], [], []

    for yr in years:
        sub = val[val["year"] == yr]
        n = len(sub)
        k_c = int(sub["prediction_correct_bool"].sum())
        k_b = int(sub["beats_market_bool"].sum())

        acc = k_c / n if n > 0 else 0.0
        ci_c = wilson_ci(k_c, n) if n > 0 else (0.0, 1.0)
        bm = k_b / n if n > 0 else 0.0
        ci_b = wilson_ci(k_b, n) if n > 0 else (0.0, 1.0)

        acc_vals.append(acc)
        acc_lo.append(acc - ci_c[0])
        acc_hi.append(ci_c[1] - acc)
        bm_vals.append(bm)
        bm_lo.append(bm - ci_b[0])
        bm_hi.append(ci_b[1] - bm)

    x = np.arange(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.bar(x - width / 2, acc_vals, width, color=BLUE, label="Prediction correct",
           yerr=[acc_lo, acc_hi], capsize=4, error_kw={"elinewidth": 1})
    ax.bar(x + width / 2, bm_vals, width, color=ORANGE, label="Beats market",
           yerr=[bm_lo, bm_hi], capsize=4, error_kw={"elinewidth": 1})
    ax.axhline(0.5, color="crimson", linestyle="--", linewidth=1.2, label="50% (no skill)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Accuracy / beats-market rate")
    ax.set_title("RQ1 — Directional accuracy and beats_market by calendar year")
    ax.legend()
    _clean_spines(ax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Excess return distribution (RQ1)
# ─────────────────────────────────────────────────────────────────────────────

def make_excess_return_hist(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    val = df[df["validated"] == True].copy()
    er = pd.to_numeric(val["excess_return"], errors="coerce").dropna()
    er_clipped = er.clip(-0.5, 0.5)
    mean_er = float(er.mean())

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.hist(er_clipped, bins=60, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=1.2, label="zero")
    ax.axvline(mean_er, color=ORANGE, linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_er:.3f}")
    ax.set_xlabel("Excess return (stock – SPY, clipped at ±0.5)")
    ax.set_ylabel("Count")
    ax.set_title("RQ1 — Distribution of per-prediction excess return (n = 10,690)")
    ax.legend()
    _clean_spines(ax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multiple-testing bar chart (RQ2)
# ─────────────────────────────────────────────────────────────────────────────

def make_multiple_testing_bar(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    from src.stats_rq2 import compute_rq2, MIN_N

    rq2 = compute_rq2(df)
    acc = rq2["account_df"]

    # Raw (p < 0.05)
    raw_above = int(((acc["pval_beats"] < 0.05) & (acc["beats_rate"] > 0.5)).sum())
    raw_below = int(((acc["pval_beats"] < 0.05) & (acc["beats_rate"] < 0.5)).sum())

    # BH-FDR
    bh_above = int((acc["reject_bh"] & (acc["beats_rate"] > 0.5)).sum())
    bh_below = int((acc["reject_bh"] & (acc["beats_rate"] < 0.5)).sum())

    # Bonferroni
    bonf_above = int((acc["reject_bonf"] & (acc["beats_rate"] > 0.5)).sum())
    bonf_below = int((acc["reject_bonf"] & (acc["beats_rate"] < 0.5)).sum())

    groups = ["Raw\n(p<0.05)", "BH-FDR\n(α=0.05)", "Bonferroni\n(α=0.05)"]
    above_counts = [raw_above, bh_above, bonf_above]
    below_counts = [raw_below, bh_below, bonf_below]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    bars_above = ax.bar(x - width / 2, above_counts, width,
                        color=GREEN, label="Beats market (above 50%)")
    bars_below = ax.bar(x + width / 2, below_counts, width,
                        color=CRIMSON, label="Below market (below 50%)")

    # Value labels above each bar
    for bar in bars_above:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.05,
                str(int(h)), ha="center", va="bottom", fontsize=10)
    for bar in bars_below:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.05,
                str(int(h)), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Number of significant accounts")
    ax.set_title("RQ2 — Multiple-testing correction: significant accounts above/below 50%")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    _clean_spines(ax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Calibration curve (RQ3) — reliability curve for LLM confidence score
# ─────────────────────────────────────────────────────────────────────────────

def make_calibration_curve(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    from src.stats_rq3 import calibration_table, confidence_correlation

    calib = calibration_table(df)
    corr  = confidence_correlation(df)

    mids  = calib["bin_mid"].values
    accs  = calib["actual_accuracy"].values
    ns    = calib["n_predictions"].values
    ci_lo = calib["ci_low"].values
    ci_hi = calib["ci_high"].values

    # Weighted mean matches overall accuracy in Table 8.1 (= Σk_i / Σn_i)
    overall_acc = np.average(accs, weights=ns)

    yerr = np.array([accs - ci_lo, ci_hi - accs])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.errorbar(
        mids, accs, yerr=yerr,
        fmt="o", color=BLUE, markersize=6, capsize=4,
        linewidth=1.4, elinewidth=1.0, label="Actual accuracy (Wilson CI)",
    )
    sizes = np.clip(ns / ns.max() * 200, 30, 200)
    ax.scatter(mids, accs, s=sizes, color=BLUE, zorder=5, alpha=0.5)

    ax.axhline(
        overall_acc, color=ORANGE, linestyle="--", linewidth=1.2,
        label=f"Overall accuracy ≈ {overall_acc*100:.1f}%",
    )
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
    import matplotlib.ticker as _mticker
    ax.yaxis.set_major_formatter(_mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.35)
    _clean_spines(ax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. ROC curves (RQ3)
# ─────────────────────────────────────────────────────────────────────────────

def make_roc_curves(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    from src.stats_rq3 import build_feature_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_curve, auc

    X, y, _ = build_feature_matrix(df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    lr_pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42),
    )
    lr_proba = cross_val_predict(lr_pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y, lr_proba)
    auc_lr = auc(fpr_lr, tpr_lr)

    # Gradient Boosting
    gb_pipe = make_pipeline(
        GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        )
    )
    gb_proba = cross_val_predict(gb_pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    fpr_gb, tpr_gb, _ = roc_curve(y, gb_proba)
    auc_gb = auc(fpr_gb, tpr_gb)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot(fpr_lr, tpr_lr, color=BLUE, linewidth=1.8,
            label=f"Logistic Regression (AUC = {auc_lr:.3f})")
    ax.plot(fpr_gb, tpr_gb, color=ORANGE, linewidth=1.8,
            label=f"Gradient Boosting (AUC = {auc_gb:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0,
            alpha=0.7, label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("RQ3 — ROC curves (5-fold stratified CV out-of-fold predictions)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(linestyle=":", alpha=0.35)
    _clean_spines(ax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Confidence score distribution (RQ3)
# ─────────────────────────────────────────────────────────────────────────────

def make_confidence_hist(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    val = df[df["validated"] == True].copy()
    conf = pd.to_numeric(val["confidence"], errors="coerce").dropna()
    mean_conf = float(conf.mean())

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.hist(conf, bins=20, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(mean_conf, color=ORANGE, linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_conf:.1f}")
    ax.set_xlabel("LLM confidence score (0–100)")
    ax.set_ylabel("Count")
    ax.set_title("RQ3 — Distribution of LLM confidence scores (n = 10,690 validated predictions)")
    ax.legend()
    _clean_spines(ax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Regime scatter (Phase 7 — robustness)
# ─────────────────────────────────────────────────────────────────────────────

def make_regime_scatter(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """P1 vs P2 beats_market scatter for all accounts qualifying in both sub-periods."""
    from src.stats_regime import compute_persistence, SKILLED_ACCOUNTS

    pers = compute_persistence(df, min_n_per_period=20)
    account_df = pers["account_df"]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    if account_df.empty:
        ax.text(0.5, 0.5,
                "Fewer than 2 accounts qualify\n(min 20 predictions per sub-period)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
    else:
        skilled_set  = set(SKILLED_ACCOUNTS)
        regular_mask = ~account_df.index.isin(skilled_set)
        skilled_mask = account_df.index.isin(skilled_set)

        if regular_mask.any():
            ax.scatter(
                account_df.loc[regular_mask, "p1_beats_rate"],
                account_df.loc[regular_mask, "p2_beats_rate"],
                color=BLUE, alpha=0.75, s=60, zorder=3,
                label="Other qualifying accounts",
            )
        if skilled_mask.any():
            ax.scatter(
                account_df.loc[skilled_mask, "p1_beats_rate"],
                account_df.loc[skilled_mask, "p2_beats_rate"],
                color=CRIMSON, s=90, zorder=4, marker="D",
                label="§7 'skilled' accounts (CI above 50%)",
            )

        for acct, row in account_df.iterrows():
            ax.annotate(
                acct, (row["p1_beats_rate"], row["p2_beats_rate"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7.5, color="#333333",
            )

        note = ("Pearson r = " +
                (f"{pers['pearson_r']:+.3f} (p = {pers['pearson_p']:.3f})"
                 if pers["computable"] else "not computable (n < 4 qualifying accounts)"))
        ax.text(0.03, 0.97, note, transform=ax.transAxes,
                fontsize=8, va="top", color="gray")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0,
            alpha=0.7, label="y = x  (perfect persistence)")
    ax.axvline(0.5, color=CRIMSON, linestyle=":", linewidth=0.9, alpha=0.55)
    ax.axhline(0.5, color=CRIMSON, linestyle=":", linewidth=0.9, alpha=0.55,
               label="50% reference")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    import matplotlib.ticker as _mt
    ax.xaxis.set_major_formatter(_mt.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(_mt.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("P1 beats_market rate  (Jan 2021 – Dec 2022)", fontsize=10)
    ax.set_ylabel("P2 beats_market rate  (Jan 2023 – Feb 2025)", fontsize=10)
    ax.set_title(
        "Regime persistence: P1 vs P2 beats_market rate per account\n"
        "(≥20 validated predictions required in each sub-period)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle=":", alpha=0.35)
    _clean_spines(ax)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    data_path = pathlib.Path(__file__).parent.parent / "data" / "tweets_validated.csv"
    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} rows.")

    t0 = time.time()

    print("\n[1/8] Monthly tweet volume ...")
    make_tweets_per_month(df, FIGURES_DIR / "data_tweets_per_month.png")

    print("[2/8] Predictions per account ...")
    make_predictions_per_account(df, FIGURES_DIR / "data_predictions_per_account.png")

    print("[3/8] Top tickers ...")
    make_top_tickers(df, FIGURES_DIR / "data_top_tickers.png")

    print("[4/8] Accuracy by year (RQ1) ...")
    make_accuracy_by_year(df, FIGURES_DIR / "rq1_accuracy_by_year.png")

    print("[5/8] Excess return distribution (RQ1) ...")
    make_excess_return_hist(df, FIGURES_DIR / "rq1_excess_return_hist.png")

    print("[6/10] Multiple-testing bar chart (RQ2) ...")
    make_multiple_testing_bar(df, FIGURES_DIR / "rq2_multiple_testing_bar.png")

    print("[7/10] Calibration curve (RQ3) ...")
    make_calibration_curve(df, FIGURES_DIR / "rq3_calibration_curve.png")

    print("[8/10] ROC curves (RQ3) — this may take ~2 minutes ...")
    make_roc_curves(df, FIGURES_DIR / "rq3_roc_curves.png")

    print("[9/10] Confidence score histogram (RQ3) ...")
    make_confidence_hist(df, FIGURES_DIR / "rq3_confidence_hist.png")

    print("[10/10] Regime persistence scatter (Phase 7) ...")
    make_regime_scatter(df, FIGURES_DIR / "rq4_regime_scatter.png")

    elapsed = time.time() - t0
    print(f"\nAll figures saved to {FIGURES_DIR}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
