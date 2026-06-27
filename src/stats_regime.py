"""Statistical analysis for the regime robustness check — Phase 7.

Public API
----------
compute_regime_split(df)                       -> dict  (n + accuracy per sub-period)
compute_persistence(df, min_n_per_period)      -> dict  (correlation + qualifying account table)
skilled_regime_table(df, skilled_accounts)     -> DataFrame  (P1/P2 for each skilled account)
compute_regime(df, min_n_per_period, skilled)  -> dict  (all results combined)

Sub-period definitions
----------------------
P1 = entry_date in [2021-01-01, 2022-12-31]   — includes the 2022 bear market
P2 = entry_date in [2023-01-01, 2025-02-28]   — post-bear recovery through corpus end
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.stats_rq1 import wilson_ci

# Sub-period boundaries (inclusive)
P1_START = "2021-01-01"
P1_END   = "2022-12-31"
P2_START = "2023-01-01"
P2_END   = "2025-02-28"

# Accounts flagged as credibly skilled in §7 (CI entirely above 50%)
SKILLED_ACCOUNTS = [
    "account_19", "account_37", "account_45", "account_03",
    "account_09", "account_13", "account_39", "account_29", "account_06",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().eq("true")


def _period_mask(dates: pd.Series, start: str, end: str) -> pd.Series:
    return (dates >= start) & (dates <= end)


def _pearson_ci(r: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Fisher z-transform 95% CI on Pearson r. Returns (nan, nan) when n <= 3."""
    if abs(r) >= 1.0 or n <= 3:
        return (float("nan"), float("nan"))
    z_r  = np.arctanh(float(r))
    se   = 1.0 / np.sqrt(n - 3)
    z_c  = float(scipy_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    return (float(np.tanh(z_r - z_c * se)), float(np.tanh(z_r + z_c * se)))


def _period_beats_stats(sub: pd.DataFrame) -> dict:
    """Compute beats_market and directional-correct stats for a sub-period slice."""
    n  = len(sub)
    kb = int(sub["beats"].sum())
    kc = int(sub["correct"].sum())
    return {
        "n":            n,
        "k_beats":      kb,
        "beats_rate":   kb / n if n > 0 else float("nan"),
        "beats_ci":     wilson_ci(kb, n) if n > 0 else (float("nan"), float("nan")),
        "k_correct":    kc,
        "correct_rate": kc / n if n > 0 else float("nan"),
        "correct_ci":   wilson_ci(kc, n) if n > 0 else (float("nan"), float("nan")),
    }


# ---------------------------------------------------------------------------
# A) Sub-period overview
# ---------------------------------------------------------------------------

def compute_regime_split(df: pd.DataFrame) -> dict:
    """Overall beats_market / directional accuracy for P1 and P2.

    Returns dict with keys: p1 (stats dict), p2 (stats dict), p1_label, p2_label.
    """
    val = df[_as_bool(df["validated"])].copy()
    dates = pd.to_datetime(val["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    val["beats"]   = _as_bool(val["beats_market"])
    val["correct"] = _as_bool(val["prediction_correct"])

    p1 = val[_period_mask(dates, P1_START, P1_END)]
    p2 = val[_period_mask(dates, P2_START, P2_END)]

    return {
        "p1":       _period_beats_stats(p1),
        "p2":       _period_beats_stats(p2),
        "p1_label": f"P1: Jan 2021 – Dec 2022 (n = {len(p1):,})",
        "p2_label": f"P2: Jan 2023 – Feb 2025 (n = {len(p2):,})",
    }


# ---------------------------------------------------------------------------
# B) Split-half persistence correlation
# ---------------------------------------------------------------------------

def compute_persistence(
    df: pd.DataFrame,
    min_n_per_period: int = 20,
) -> dict:
    """Pearson and Spearman correlation between P1 and P2 beats_market rates.

    Restricted to accounts with >= min_n_per_period validated predictions in
    BOTH sub-periods.  Returns dict with keys:

        n_qualifying, min_n_per_period,
        pearson_r, pearson_p, pearson_ci,
        spearman_r, spearman_p,
        computable  (bool — False when n_qualifying < 4),
        account_df  (DataFrame with per-account P1/P2 data for all qualifying accounts)
    """
    val = df[_as_bool(df["validated"])].copy()
    dates = pd.to_datetime(val["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    val["beats"] = _as_bool(val["beats_market"])

    p1 = val[_period_mask(dates, P1_START, P1_END)]
    p2 = val[_period_mask(dates, P2_START, P2_END)]

    def _per_account(sub: pd.DataFrame) -> pd.DataFrame:
        agg = sub.groupby("author").agg(
            n=("beats", "count"),
            k_beats=("beats", "sum"),
        )
        agg["beats_rate"] = agg["k_beats"] / agg["n"]
        return agg[agg["n"] >= min_n_per_period]

    p1_acc = _per_account(p1)
    p2_acc = _per_account(p2)
    common = p1_acc.index.intersection(p2_acc.index)
    nq = len(common)

    # Build account-level DataFrame for all qualifying accounts
    if nq > 0:
        account_df = pd.DataFrame({
            "p1_n":          p1_acc.loc[common, "n"].values,
            "p1_k_beats":    p1_acc.loc[common, "k_beats"].values,
            "p1_beats_rate": p1_acc.loc[common, "beats_rate"].values,
            "p2_n":          p2_acc.loc[common, "n"].values,
            "p2_k_beats":    p2_acc.loc[common, "k_beats"].values,
            "p2_beats_rate": p2_acc.loc[common, "beats_rate"].values,
        }, index=common)
    else:
        account_df = pd.DataFrame(
            columns=["p1_n","p1_k_beats","p1_beats_rate",
                     "p2_n","p2_k_beats","p2_beats_rate"]
        )

    # Need >= 4 points for a meaningful correlation
    if nq < 4:
        return {
            "n_qualifying":     nq,
            "min_n_per_period": min_n_per_period,
            "computable":       False,
            "pearson_r":        float("nan"),
            "pearson_p":        float("nan"),
            "pearson_ci":       (float("nan"), float("nan")),
            "spearman_r":       float("nan"),
            "spearman_p":       float("nan"),
            "account_df":       account_df,
        }

    r1 = p1_acc.loc[common, "beats_rate"].values
    r2 = p2_acc.loc[common, "beats_rate"].values

    pearson_r,  pearson_p  = scipy_stats.pearsonr(r1, r2)
    spearman_r, spearman_p = scipy_stats.spearmanr(r1, r2)
    pearson_ci = _pearson_ci(float(pearson_r), nq)

    return {
        "n_qualifying":     nq,
        "min_n_per_period": min_n_per_period,
        "computable":       True,
        "pearson_r":        float(pearson_r),
        "pearson_p":        float(pearson_p),
        "pearson_ci":       pearson_ci,
        "spearman_r":       float(spearman_r),
        "spearman_p":       float(spearman_p),
        "account_df":       account_df,
    }


# ---------------------------------------------------------------------------
# C) Skilled-account regime table
# ---------------------------------------------------------------------------

def skilled_regime_table(
    df: pd.DataFrame,
    skilled_accounts: list[str] | None = None,
) -> pd.DataFrame:
    """Beats_market rate + 95% Wilson CI in P1 and P2 for the §7 skilled accounts.

    Returns a DataFrame indexed by author with columns:
        p1_n, p1_k_beats, p1_beats_rate, p1_ci_low, p1_ci_high,
        p2_n, p2_k_beats, p2_beats_rate, p2_ci_low, p2_ci_high,
        above_50_p1, above_50_p2, above_50_both
    """
    if skilled_accounts is None:
        skilled_accounts = SKILLED_ACCOUNTS

    val = df[_as_bool(df["validated"])].copy()
    dates = pd.to_datetime(val["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    val["beats"] = _as_bool(val["beats_market"])

    p1 = val[_period_mask(dates, P1_START, P1_END)]
    p2 = val[_period_mask(dates, P2_START, P2_END)]

    rows = []
    for acct in skilled_accounts:
        p1s = p1[p1["author"] == acct]
        p2s = p2[p2["author"] == acct]

        p1_n  = len(p1s)
        p1_kb = int(p1s["beats"].sum()) if p1_n > 0 else 0
        p2_n  = len(p2s)
        p2_kb = int(p2s["beats"].sum()) if p2_n > 0 else 0

        p1_rate = p1_kb / p1_n if p1_n > 0 else float("nan")
        p2_rate = p2_kb / p2_n if p2_n > 0 else float("nan")

        p1_ci = wilson_ci(p1_kb, p1_n) if p1_n > 0 else (float("nan"), float("nan"))
        p2_ci = wilson_ci(p2_kb, p2_n) if p2_n > 0 else (float("nan"), float("nan"))

        above_p1   = (p1_rate > 0.5)    if p1_n > 0 else False
        above_p2   = (p2_rate > 0.5)    if p2_n > 0 else False
        above_both = above_p1 and above_p2 and (p1_n > 0) and (p2_n > 0)

        rows.append({
            "author":        acct,
            "p1_n":          p1_n,
            "p1_k_beats":    p1_kb,
            "p1_beats_rate": p1_rate,
            "p1_ci_low":     p1_ci[0],
            "p1_ci_high":    p1_ci[1],
            "p2_n":          p2_n,
            "p2_k_beats":    p2_kb,
            "p2_beats_rate": p2_rate,
            "p2_ci_low":     p2_ci[0],
            "p2_ci_high":    p2_ci[1],
            "above_50_p1":   above_p1,
            "above_50_p2":   above_p2,
            "above_50_both": above_both,
        })

    return pd.DataFrame(rows).set_index("author")


# ---------------------------------------------------------------------------
# D) Top-level entry point
# ---------------------------------------------------------------------------

def compute_regime(
    df: pd.DataFrame,
    min_n_per_period: int = 20,
    skilled_accounts: list[str] | None = None,
) -> dict:
    """Run the full regime split analysis.

    Returns dict with keys: split, persistence, skilled.
    """
    return {
        "split":       compute_regime_split(df),
        "persistence": compute_persistence(df, min_n_per_period=min_n_per_period),
        "skilled":     skilled_regime_table(df, skilled_accounts=skilled_accounts),
    }
