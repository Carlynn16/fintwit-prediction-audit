"""RQ1 analysis: accuracy vs chance and vs market benchmark.

Usage:
    python scripts/run_rq1.py

Reads  : data/tweets_validated.csv
Writes : figures/rq1_accuracy_by_segment.png
Prints : full results table
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

from src.stats_rq1 import compute_rq1

DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
FIGURES_DIR = pathlib.Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

INPUT_CSV  = DATA_DIR / "tweets_validated.csv"
FIGURE_OUT = FIGURES_DIR / "rq1_accuracy_by_segment.png"

# Friendly labels for display
HORIZON_LABELS = {
    "short_term": "Short\n(21d)",
    "medium_term": "Medium\n(63d)",
    "long_term": "Long\n(126d)",
    "unknown": "Unknown\n(->21d)",
}
TRADE_LABELS = {
    "analysis":          "Analysis",
    "trade_suggestion":  "Trade\nsuggestion",
    "general_discussion":"General\ndiscussion",
    "news":              "News",
}


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt_pct(p: float, lo: float, hi: float) -> str:
    return f"{p*100:.1f}%  [{lo*100:.1f}%, {hi*100:.1f}%]"


def _fmt_pval(p: float) -> str:
    if p < 1e-100:
        return "< 1e-100"
    if p < 0.001:
        return f"< 0.001  ({p:.2e})"
    return f"{p:.4f}"


def print_results(res: dict) -> None:
    n = res["n_validated"]
    oc = res["overall"]["correct"]
    ob = res["overall"]["beats"]

    print()
    print("=" * 70)
    print("RQ1 — Accuracy vs chance and vs market benchmark")
    print("=" * 70)
    print(f"  Validated predictions : {n:,}")
    print()
    print("  Overall (95% Wilson CI):")
    print(f"    Directional correct : {_fmt_pct(oc['p_hat'], oc['ci_low'], oc['ci_high'])}")
    print(f"    Beats market (SPY)  : {_fmt_pct(ob['p_hat'], ob['ci_low'], ob['ci_high'])}")
    print()
    print("  Significance vs 50% (two-tailed proportion z-test):")
    print(f"    Correct  z = {oc['z_stat']:+.2f}  p {_fmt_pval(oc['p_value'])}")
    print(f"    Beats    z = {ob['z_stat']:+.2f}  p {_fmt_pval(ob['p_value'])}")

    print()
    print("  Mean returns over validated horizon windows:")
    ret = res["returns"]
    print(f"    Mean stock return  : {ret['mean_stock_return']*100:+.2f}%")
    print(f"    Mean SPY return    : {ret['mean_spy_return']*100:+.2f}%")
    print(f"    Mean excess return : {ret['mean_excess_return']*100:+.2f}%")

    print()
    print("  Sensitivity check (excluding horizon_was_unknown rows):")
    s = res["sensitivity"]
    c_ci = s["ci_correct"]
    b_ci = s["ci_beats"]
    print(f"    n = {s['n']:,}  (dropped {n - s['n']:,} unknown-horizon rows)")
    print(f"    Correct  : {_fmt_pct(s['acc_correct'], c_ci[0], c_ci[1])}  p {_fmt_pval(s['p_value_correct'])}")
    print(f"    Beats    : {_fmt_pct(s['acc_beats'],   b_ci[0], b_ci[1])}  p {_fmt_pval(s['p_value_beats'])}")

    for seg_name, seg_df in res["segments"].items():
        print()
        print(f"  By {seg_name}:")
        header = f"  {'Group':<22} {'n':>6}  {'Correct':>22}  {'Beats market':>22}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for group, row in seg_df.iterrows():
            c_str = _fmt_pct(row["acc_correct"], row["ci_correct_low"], row["ci_correct_high"])
            b_str = _fmt_pct(row["acc_beats"],   row["ci_beats_low"],   row["ci_beats_high"])
            print(f"  {group:<22} {row['n']:>6,}  {c_str:>22}  {b_str:>22}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

COLORS = {"correct": "#2F5496", "beats": "#ED7D31"}

def _segment_panel(
    ax: plt.Axes,
    seg_df: pd.DataFrame,
    title: str,
    label_map: dict | None = None,
) -> None:
    groups  = list(seg_df.index)
    n_g     = len(groups)
    x       = np.arange(n_g)
    width   = 0.35

    labels = [label_map.get(g, g) if label_map else g for g in groups]

    for offset, metric, color, display in [
        (-width / 2, "correct", COLORS["correct"], "Directional correct"),
        (+width / 2, "beats",   COLORS["beats"],   "Beats market (SPY)"),
    ]:
        accs  = seg_df[f"acc_{metric}"].values
        lows  = seg_df[f"ci_{metric}_low"].values
        highs = seg_df[f"ci_{metric}_high"].values
        errs  = np.array([accs - lows, highs - accs])

        bars = ax.bar(x + offset, accs, width, label=display,
                      color=color, alpha=0.85, zorder=3)
        ax.errorbar(x + offset, accs, yerr=errs, fmt="none",
                    color="black", capsize=4, linewidth=1.2, zorder=4)

    ax.axhline(0.5, color="crimson", linewidth=1.2, linestyle="--",
               alpha=0.8, label="50% (chance)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (95% CI)", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(res: dict, out_path: pathlib.Path) -> None:
    segs = res["segments"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "RQ1 — Prediction accuracy by segment\n"
        "Blue = directional correct vs stock; Orange = beats SPY; Red dashed = 50% chance",
        fontsize=10, y=1.01,
    )

    _segment_panel(axes[0], segs["sentiment"],    "By sentiment")
    _segment_panel(axes[1], segs["time_horizon"], "By prediction horizon",
                   label_map=HORIZON_LABELS)
    _segment_panel(axes[2], segs["trade_type"],   "By trade type",
                   label_map=TRADE_LABELS)

    # Shared legend on the first panel only
    axes[0].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(
            f"ERROR: {INPUT_CSV} not found.\n"
            "Run scripts/validate_predictions.py first."
        )

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    res = compute_rq1(df)
    print_results(res)
    make_figure(res, FIGURE_OUT)


if __name__ == "__main__":
    main()
