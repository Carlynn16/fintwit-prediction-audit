"""Statistical analysis for RQ1 — accuracy vs chance and vs market benchmark.

Public API
----------
wilson_ci(k, n, confidence)        -> (lower, upper)
proportion_ztest(k, n, p0)         -> dict
segment_stats(df, group_col)       -> DataFrame  (one row per group, both metrics)
compute_rq1(df)                    -> dict       (all results; df must be tweets_validated)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# CI and hypothesis-test primitives
# ---------------------------------------------------------------------------

def wilson_ci(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a proportion k / n.

    Returns (lower, upper) clipped to [0, 1].
    Returns (0.0, 1.0) when n == 0 (maximum uncertainty).
    """
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2))
    return (float(np.clip(centre - margin, 0.0, 1.0)),
            float(np.clip(centre + margin, 0.0, 1.0)))


def proportion_ztest(
    k: int, n: int, p0: float = 0.5
) -> dict:
    """Two-tailed proportion z-test of H0: p = p0.

    Returns a dict with keys: p_hat, z_stat, p_value, ci_low, ci_high, k, n.
    Uses the Wilson CI for the confidence interval.
    """
    if n == 0:
        return dict(p_hat=float("nan"), z_stat=float("nan"),
                    p_value=float("nan"), ci_low=0.0, ci_high=1.0, k=k, n=n)
    p_hat = k / n
    se = np.sqrt(p0 * (1.0 - p0) / n)
    z = (p_hat - p0) / se
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    ci_low, ci_high = wilson_ci(k, n)
    return dict(p_hat=float(p_hat), z_stat=float(z), p_value=p_value,
                ci_low=ci_low, ci_high=ci_high, k=int(k), n=int(n))


# ---------------------------------------------------------------------------
# Segmented statistics
# ---------------------------------------------------------------------------

def segment_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute accuracy + Wilson 95% CIs for prediction_correct and beats_market
    broken down by *group_col*.

    Input *df* must already be filtered to validated rows and have boolean columns
    'correct' and 'beats' (True/False, no NAs).

    Returns a DataFrame indexed by group value with columns:
        n, k_correct, acc_correct, ci_correct_low, ci_correct_high,
        k_beats,   acc_beats,   ci_beats_low,   ci_beats_high
    """
    rows = []
    for group, sub in df.groupby(group_col, sort=True):
        n = len(sub)
        k_c = int(sub["correct"].sum())
        k_b = int(sub["beats"].sum())
        ci_c = wilson_ci(k_c, n)
        ci_b = wilson_ci(k_b, n)
        rows.append({
            "group":          str(group),
            "n":              n,
            "k_correct":      k_c,
            "acc_correct":    k_c / n,
            "ci_correct_low": ci_c[0],
            "ci_correct_high":ci_c[1],
            "k_beats":        k_b,
            "acc_beats":      k_b / n,
            "ci_beats_low":   ci_b[0],
            "ci_beats_high":  ci_b[1],
        })
    return pd.DataFrame(rows).set_index("group")


# ---------------------------------------------------------------------------
# Top-level RQ1 computation
# ---------------------------------------------------------------------------

def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a CSV-read column to bool (handles 'True'/'False' strings)."""
    return series.astype(str).str.strip().str.lower().eq("true")


def compute_rq1(df: pd.DataFrame) -> dict:
    """Compute all RQ1 statistics from *tweets_validated.csv*.

    *df* may contain all 18 071 rows; validated rows are filtered internally.

    Returns a nested dict with keys:
        n_validated, overall, segments, returns, sensitivity
    """
    # ── filter to validated predictions ────────────────────────────────────
    val = df[_as_bool(df["validated"])].copy()
    val["correct"] = _as_bool(val["prediction_correct"])
    val["beats"]   = _as_bool(val["beats_market"])
    val["unk_hz"]  = _as_bool(val["horizon_was_unknown"])

    n  = len(val)
    kc = int(val["correct"].sum())
    kb = int(val["beats"].sum())

    # ── overall tests vs 50% ───────────────────────────────────────────────
    overall = {
        "correct": proportion_ztest(kc, n),
        "beats":   proportion_ztest(kb, n),
    }

    # ── segmented accuracy ─────────────────────────────────────────────────
    segments = {
        "sentiment":    segment_stats(val, "sentiment"),
        "time_horizon": segment_stats(val, "time_horizon"),
        "trade_type":   segment_stats(val, "trade_type"),
    }

    # ── mean returns ───────────────────────────────────────────────────────
    ret = {
        "mean_stock_return":  float(pd.to_numeric(val["stock_return"],  errors="coerce").mean()),
        "mean_spy_return":    float(pd.to_numeric(val["spy_return"],    errors="coerce").mean()),
        "mean_excess_return": float(pd.to_numeric(val["excess_return"], errors="coerce").mean()),
    }

    # ── sensitivity: exclude horizon_was_unknown rows ──────────────────────
    val_known = val[~val["unk_hz"]]
    n_k  = len(val_known)
    kc_k = int(val_known["correct"].sum())
    kb_k = int(val_known["beats"].sum())
    sensitivity = {
        "n":               n_k,
        "k_correct":       kc_k,
        "acc_correct":     kc_k / n_k,
        "ci_correct":      wilson_ci(kc_k, n_k),
        "p_value_correct": proportion_ztest(kc_k, n_k)["p_value"],
        "k_beats":         kb_k,
        "acc_beats":       kb_k / n_k,
        "ci_beats":        wilson_ci(kb_k, n_k),
        "p_value_beats":   proportion_ztest(kb_k, n_k)["p_value"],
    }

    return {
        "n_validated": n,
        "overall":     overall,
        "segments":    segments,
        "returns":     ret,
        "sensitivity": sensitivity,
    }
