"""Phase 7 robustness analysis: skill persistence across market regimes.

Usage:
    python scripts/run_regime.py

Reads  : data/tweets_validated.csv
Writes : figures/rq4_regime_scatter.png
Prints : (a) sub-period n-counts and accuracy
         (b) persistence test result (or n/a if too few qualifying accounts)
         (c) §7 skilled-account P1 vs P2 table
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

from src.stats_regime import (
    SKILLED_ACCOUNTS,
    compute_regime,
)

DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
FIGURES_DIR = pathlib.Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

INPUT_CSV  = DATA_DIR / "tweets_validated.csv"
FIG_REGIME = FIGURES_DIR / "rq4_regime_scatter.png"


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_split(res: dict) -> None:
    split = res["split"]
    p1, p2 = split["p1"], split["p2"]
    print()
    print("=" * 70)
    print("Regime split — sub-period overview")
    print("=" * 70)
    print(f"  {split['p1_label']}")
    print(f"    beats_market : {p1['beats_rate']*100:.1f}%  "
          f"[{p1['beats_ci'][0]*100:.1f}%, {p1['beats_ci'][1]*100:.1f}%]")
    print(f"    correct      : {p1['correct_rate']*100:.1f}%  "
          f"[{p1['correct_ci'][0]*100:.1f}%, {p1['correct_ci'][1]*100:.1f}%]")
    print()
    print(f"  {split['p2_label']}")
    print(f"    beats_market : {p2['beats_rate']*100:.1f}%  "
          f"[{p2['beats_ci'][0]*100:.1f}%, {p2['beats_ci'][1]*100:.1f}%]")
    print(f"    correct      : {p2['correct_rate']*100:.1f}%  "
          f"[{p2['correct_ci'][0]*100:.1f}%, {p2['correct_ci'][1]*100:.1f}%]")


def print_persistence(res: dict) -> None:
    p = res["persistence"]
    print()
    print("=" * 70)
    print("Persistence test (split-half correlation)")
    print("=" * 70)
    print(f"  Min predictions per period : {p['min_n_per_period']}")
    print(f"  Accounts qualifying (both) : {p['n_qualifying']}")
    if p["computable"]:
        print(f"  Pearson  r = {p['pearson_r']:+.3f}  "
              f"(p = {p['pearson_p']:.4f})  "
              f"95% CI [{p['pearson_ci'][0]:+.3f}, {p['pearson_ci'][1]:+.3f}]")
        print(f"  Spearman r = {p['spearman_r']:+.3f}  "
              f"(p = {p['spearman_p']:.4f})")
    else:
        print("  RESULT: Not computable — fewer than 4 accounts qualify.")
        print("  NOTE:   99.2% of validated predictions fall in P2 (2023-2025);")
        print("          only 2 accounts have >=20 predictions in P1.")
    if not p["account_df"].empty:
        print()
        print("  Qualifying account breakdown:")
        print(f"  {'Account':<15} {'P1 n':>5} {'P1 rate':>8} {'P2 n':>5} {'P2 rate':>8}")
        print("  " + "-" * 44)
        for acct, row in p["account_df"].iterrows():
            print(f"  {acct:<15} {int(row['p1_n']):>5} {row['p1_beats_rate']*100:>7.1f}% "
                  f"{int(row['p2_n']):>5} {row['p2_beats_rate']*100:>7.1f}%")


def print_skilled_table(res: dict) -> None:
    tbl = res["skilled"]
    print()
    print("=" * 70)
    print("§7 skilled accounts — P1 vs P2 beats_market rates")
    print("=" * 70)
    print(f"  {'Account':<15} {'P1 n':>5} {'P1 rate':>8} {'P1 95% CI':>16}  "
          f"{'P2 n':>5} {'P2 rate':>8} {'P2 95% CI':>16}  {'Both>50%':>8}")
    print("  " + "-" * 85)
    for acct, row in tbl.iterrows():
        p1_rate_s = f"{row['p1_beats_rate']*100:.1f}%" if not np.isnan(row["p1_beats_rate"]) else "N/A"
        p1_ci_s   = (f"[{row['p1_ci_low']*100:.1f}%,{row['p1_ci_high']*100:.1f}%]"
                     if not np.isnan(row["p1_ci_low"]) else "N/A")
        p2_rate_s = f"{row['p2_beats_rate']*100:.1f}%"
        p2_ci_s   = f"[{row['p2_ci_low']*100:.1f}%,{row['p2_ci_high']*100:.1f}%]"
        both_s    = "Yes*" if row["above_50_both"] else ("No (P1 absent)" if row["p1_n"] == 0 else "No")
        print(f"  {acct:<15} {int(row['p1_n']):>5} {p1_rate_s:>8} {p1_ci_s:>16}  "
              f"{int(row['p2_n']):>5} {p2_rate_s:>8} {p2_ci_s:>16}  {both_s:>8}")
    print("  * Both periods have P1 and P2 rate > 50%.")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_regime_scatter(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """P1 vs P2 beats_market scatter for all accounts qualifying in both sub-periods."""
    from src.stats_regime import compute_persistence, SKILLED_ACCOUNTS
    import matplotlib.ticker as mticker

    pers = compute_persistence(df, min_n_per_period=20)
    account_df = pers["account_df"]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    if account_df.empty:
        ax.text(0.5, 0.5, "Fewer than 2 accounts qualify\n(min 20 predictions per sub-period)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
    else:
        skilled_set = set(SKILLED_ACCOUNTS)
        regular_mask  = ~account_df.index.isin(skilled_set)
        skilled_mask  = account_df.index.isin(skilled_set)

        if regular_mask.any():
            ax.scatter(
                account_df.loc[regular_mask, "p1_beats_rate"],
                account_df.loc[regular_mask, "p2_beats_rate"],
                color="#2F5496", alpha=0.75, s=60, zorder=3, label="Other qualifying accounts",
            )
        if skilled_mask.any():
            ax.scatter(
                account_df.loc[skilled_mask, "p1_beats_rate"],
                account_df.loc[skilled_mask, "p2_beats_rate"],
                color="#C00000", s=90, zorder=4, marker="D",
                label="§7 'skilled' accounts (CI above 50%)",
            )

        for acct, row in account_df.iterrows():
            ax.annotate(
                acct, (row["p1_beats_rate"], row["p2_beats_rate"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7.5, color="#333333",
            )

        note = ("Pearson r = " + (f"{pers['pearson_r']:+.3f} (p = {pers['pearson_p']:.3f})"
                                  if pers["computable"] else "not computable (n < 4)"))
        ax.text(0.03, 0.97, note, transform=ax.transAxes,
                fontsize=8, va="top", color="gray")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0,
            alpha=0.7, label="y = x  (perfect persistence)")
    ax.axvline(0.5, color="crimson", linestyle=":", linewidth=0.9, alpha=0.55)
    ax.axhline(0.5, color="crimson", linestyle=":", linewidth=0.9, alpha=0.55,
               label="50% reference")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("P1 beats_market rate  (Jan 2021 – Dec 2022)", fontsize=10)
    ax.set_ylabel("P2 beats_market rate  (Jan 2023 – Feb 2025)", fontsize=10)
    ax.set_title(
        "Regime persistence: P1 vs P2 beats_market rate per account\n"
        "(≥20 validated predictions required in each sub-period)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"ERROR: {INPUT_CSV} not found.")

    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    res = compute_regime(df)

    print_split(res)
    print_persistence(res)
    print_skilled_table(res)

    make_regime_scatter(df, FIG_REGIME)
    print("\nDone.")


if __name__ == "__main__":
    main()
