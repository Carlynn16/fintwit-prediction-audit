"""RQ2 analysis: per-account skill vs luck.

Usage:
    python scripts/run_rq2.py

Reads  : data/tweets_validated.csv
Writes : figures/rq2_skill_caterpillar.png
Prints : (a) before/after correction significance counts
         (b) ranked table of all qualifying accounts
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

from src.stats_rq2 import compute_rq2, MIN_N

DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
FIGURES_DIR = pathlib.Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

INPUT_CSV  = DATA_DIR / "tweets_validated.csv"
FIGURE_OUT = FIGURES_DIR / "rq2_skill_caterpillar.png"


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _sig_star(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "   "


def print_correction_summary(res: dict) -> None:
    corr = res["corrections"]
    acc  = res["account_df"]
    n    = res["n_qualifying"]

    # Direction: how many significant are above vs below 50%?
    n_raw_above = int(
        ((acc["pval_beats"] < 0.05) & (acc["beats_rate"] > 0.5)).sum()
    )
    n_raw_below = int(
        ((acc["pval_beats"] < 0.05) & (acc["beats_rate"] < 0.5)).sum()
    )
    n_bh_above  = int((acc["reject_bh"]   & (acc["beats_rate"] > 0.5)).sum())
    n_bh_below  = int((acc["reject_bh"]   & (acc["beats_rate"] < 0.5)).sum())
    n_bo_above  = int((acc["reject_bonf"] & (acc["beats_rate"] > 0.5)).sum())
    n_bo_below  = int((acc["reject_bonf"] & (acc["beats_rate"] < 0.5)).sum())

    print()
    print("=" * 60)
    print("RQ2 — Multiple-testing correction summary")
    print("=" * 60)
    print(f"  Qualifying accounts (n >= {res['min_n']}): {n}")
    print()
    print(f"  Before correction (raw p < 0.05) : {corr['raw_significant']:>3}  "
          f"[above 50%: {n_raw_above}, below 50%: {n_raw_below}]")
    print(f"  After BH-FDR (alpha = 0.05)      : {corr['bh_significant']:>3}  "
          f"[above 50%: {n_bh_above}, below 50%: {n_bh_below}]")
    print(f"  After Bonferroni (alpha = 0.05)   : {corr['bonf_significant']:>3}  "
          f"[above 50%: {n_bo_above}, below 50%: {n_bo_below}]")
    print()

    prior = res["prior"]
    print(f"  Empirical-Bayes prior: Beta({prior['alpha']:.2f}, {prior['beta']:.2f})"
          f"  —  prior mean = {prior['prior_mean']:.3f}")
    print(f"  Accounts credible interval entirely above 50%: "
          f"{res['n_credible_above_50']}")


def print_account_table(res: dict) -> None:
    acc = res["account_df"].copy()
    print()
    print("=" * 100)
    print("RQ2 — Per-account results (sorted by shrunk beats_market estimate, descending)")
    print("=" * 100)
    header = (
        f"  {'Account':<14} {'n':>5}  {'Raw %':>7}  {'Shrunk %':>9}  "
        f"{'95% CI':^18}  {'p_raw':>8}  {'p_BH':>8}  {'p_Bonf':>8}  Sig"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for author, row in acc.iterrows():
        ci_str = f"[{row['credible_low']*100:.1f}%, {row['credible_high']*100:.1f}%]"
        above  = " <-- above 50%" if row["credible_above_50"] else ""
        print(
            f"  {author:<14} {int(row['n']):>5}  "
            f"{row['beats_rate']*100:>6.1f}%  "
            f"{row['shrunk_beats']*100:>8.1f}%  "
            f"{ci_str:^18}  "
            f"{row['pval_beats']:>8.4f}  "
            f"{row['pval_beats_bh']:>8.4f}  "
            f"{row['pval_beats_bonf']:>8.4f}  "
            f"{_sig_star(row['pval_beats'])}{above}"
        )


# ---------------------------------------------------------------------------
# Forest / caterpillar plot
# ---------------------------------------------------------------------------

def make_caterpillar(res: dict, out_path: pathlib.Path) -> None:
    acc = res["account_df"].copy()
    # Sort bottom-to-top by shrunk estimate for the plot
    acc = acc.sort_values("shrunk_beats", ascending=True)
    labels = [str(a) for a in acc.index]

    y = np.arange(len(acc))
    shrunk = acc["shrunk_beats"].values
    raw    = acc["beats_rate"].values
    ci_lo  = acc["credible_low"].values
    ci_hi  = acc["credible_high"].values
    above  = acc["credible_above_50"].values

    fig, ax = plt.subplots(figsize=(9, max(6, len(acc) * 0.28)))

    # Credible intervals
    for i in range(len(acc)):
        color = "#C00000" if above[i] else "#2F5496"
        ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]],
                color=color, linewidth=1.2, alpha=0.6, zorder=2)

    # Raw rate (hollow circles)
    ax.scatter(raw, y, marker="o", s=22, facecolors="none",
               edgecolors="#777777", linewidths=0.8, zorder=3,
               label="Raw beats_market rate")

    # Shrunk estimate (filled circles)
    colors_fill = ["#C00000" if a else "#2F5496" for a in above]
    ax.scatter(shrunk, y, marker="o", s=28, c=colors_fill,
               zorder=4, label="Shrunk estimate (posterior mean)")

    # 50% reference line
    ax.axvline(0.5, color="crimson", linestyle="--", linewidth=1.2,
               alpha=0.9, label="50% (no skill)", zorder=1)

    # Prior mean
    ax.axvline(res["prior"]["prior_mean"], color="darkorange",
               linestyle=":", linewidth=1.2, alpha=0.8,
               label=f"Prior mean ({res['prior']['prior_mean']*100:.1f}%)", zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("beats_market rate", fontsize=10)
    ax.set_title(
        "RQ2 — Per-account beats_market rate\n"
        "Filled = shrunk (posterior mean);  Hollow = raw;  "
        "Bars = 95% credible interval;  Red = CI entirely above 50%",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

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

    df  = pd.read_csv(INPUT_CSV, low_memory=False)
    res = compute_rq2(df, min_n=MIN_N)

    print_correction_summary(res)
    print_account_table(res)
    make_caterpillar(res, FIGURE_OUT)


if __name__ == "__main__":
    main()
