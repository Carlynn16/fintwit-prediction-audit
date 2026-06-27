"""Statistical analysis for RQ2 — per-account skill vs luck.

Public API
----------
per_account_stats(df, min_n)        -> DataFrame  (one row per qualifying account)
bh_bonferroni(p_values)             -> dict        (raw/BH/Bonferroni results)
fit_beta_prior(p_hats, ns)          -> (alpha, beta)
add_shrinkage(account_df, alpha, beta) -> DataFrame  (adds shrunk_beats, credible_*)
compute_rq2(df, min_n)              -> dict         (all RQ2 results)

Method notes
------------
Primary skill metric  : beats_market vs p0 = 0.50 (two-sided binomial exact test).
Secondary metric       : prediction_correct (same test, reported for comparison).
Threshold              : >= 30 validated predictions per account (documented below).
Multiple-testing       : Benjamini-Hochberg FDR and Bonferroni, 47 tests (N_QUALIFYING).
Shrinkage              : Empirical-Bayes Beta prior estimated by method of moments from
                         the pooled per-account rates, then per-account Beta posterior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Minimum number of validated predictions required for an account to be included.
# Below this, per-account binomial tests have very low power and wide CIs;
# 30 is a standard minimum for single-proportion inference.
MIN_N = 30


# ---------------------------------------------------------------------------
# Per-account raw statistics
# ---------------------------------------------------------------------------

def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def per_account_stats(df: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    """Compute per-account n, accuracy, and binomial p-values.

    Filters to validated rows (validated == True) and to accounts with
    at least *min_n* validated predictions.  Returns a DataFrame indexed
    by author with columns:

        n, k_beats, beats_rate, k_correct, correct_rate,
        pval_beats, pval_correct

    p-values are two-sided exact binomial tests vs p0 = 0.50.
    """
    val = df[_as_bool(df["validated"])].copy()
    val["beats"]   = _as_bool(val["beats_market"])
    val["correct"] = _as_bool(val["prediction_correct"])

    agg = (
        val.groupby("author")
        .agg(n=("beats", "count"), k_beats=("beats", "sum"), k_correct=("correct", "sum"))
        .query(f"n >= {min_n}")
        .copy()
    )

    agg["beats_rate"]   = agg["k_beats"]   / agg["n"]
    agg["correct_rate"] = agg["k_correct"] / agg["n"]

    # Two-sided exact binomial tests vs 50%
    agg["pval_beats"]   = agg.apply(
        lambda r: stats.binomtest(int(r.k_beats),   int(r.n), 0.5, alternative="two-sided").pvalue, axis=1
    )
    agg["pval_correct"] = agg.apply(
        lambda r: stats.binomtest(int(r.k_correct), int(r.n), 0.5, alternative="two-sided").pvalue, axis=1
    )

    return agg.sort_values("beats_rate", ascending=False)


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def bh_bonferroni(p_values: np.ndarray) -> dict:
    """Apply BH-FDR and Bonferroni corrections to an array of p-values.

    Returns a dict with keys:
        raw_significant   : int  (p < 0.05 before any correction)
        raw_above_50_sig  : int  (needs caller to supply direction info; set to None here)
        bh_reject         : ndarray[bool]
        bh_pvals_adj      : ndarray[float]
        bh_significant    : int
        bonf_reject       : ndarray[bool]
        bonf_pvals_adj    : ndarray[float]
        bonf_significant  : int
    """
    p = np.asarray(p_values, dtype=float)

    raw_sig = int((p < 0.05).sum())

    bh_reject, bh_adj, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    bf_reject, bf_adj, _, _ = multipletests(p, alpha=0.05, method="bonferroni")

    return {
        "raw_significant":  raw_sig,
        "bh_reject":        bh_reject,
        "bh_pvals_adj":     bh_adj,
        "bh_significant":   int(bh_reject.sum()),
        "bonf_reject":      bf_reject,
        "bonf_pvals_adj":   bf_adj,
        "bonf_significant": int(bf_reject.sum()),
    }


# ---------------------------------------------------------------------------
# Empirical-Bayes Beta prior (method of moments)
# ---------------------------------------------------------------------------

def fit_beta_prior(p_hats: np.ndarray, ns: np.ndarray) -> tuple[float, float]:
    """Estimate Beta(alpha, beta) prior from per-account observed rates.

    Uses the method of moments on the marginal Beta-Binomial distribution:
      1. mu  = mean(p_hat_i)          — estimates alpha / (alpha + beta)
      2. var_obs = var(p_hat_i)       — variance of observed rates
      3. var_noise = mean(p_i*(1-p_i)/n_i)   — expected binomial noise
      4. var_signal = max(var_obs - var_noise, epsilon)
      5. phi = mu*(1-mu) / var_signal - 1    — concentration (alpha + beta)
      6. alpha = mu * phi,  beta = (1-mu) * phi

    Returns (alpha, beta) with both > 0.
    """
    p = np.asarray(p_hats, dtype=float)
    n = np.asarray(ns, dtype=float)

    mu = float(np.mean(p))
    mu = np.clip(mu, 1e-6, 1.0 - 1e-6)

    var_obs   = float(np.var(p, ddof=1))
    var_noise = float(np.mean(p * (1.0 - p) / n))

    var_signal = max(var_obs - var_noise, 1e-6)

    phi = mu * (1.0 - mu) / var_signal - 1.0
    phi = max(phi, 1.0)   # concentration must be >= 1 (weakest meaningful prior)

    alpha = float(mu * phi)
    beta  = float((1.0 - mu) * phi)

    return alpha, beta


# ---------------------------------------------------------------------------
# Shrinkage via Beta posterior
# ---------------------------------------------------------------------------

def add_shrinkage(
    account_df: pd.DataFrame,
    alpha_prior: float,
    beta_prior: float,
) -> pd.DataFrame:
    """Add empirical-Bayes shrinkage columns to *account_df* (in-place copy).

    New columns:
        shrunk_beats      — posterior mean  = (alpha_prior + k_beats) / (alpha_prior + beta_prior + n)
        credible_low      — 2.5th percentile of Beta posterior
        credible_high     — 97.5th percentile of Beta posterior
        credible_above_50 — True if entire 95% credible interval is above 0.50
    """
    df = account_df.copy()

    alpha_post = alpha_prior + df["k_beats"]
    beta_post  = beta_prior  + (df["n"] - df["k_beats"])

    df["shrunk_beats"]  = alpha_post / (alpha_post + beta_post)
    df["credible_low"]  = stats.beta.ppf(0.025, alpha_post, beta_post)
    df["credible_high"] = stats.beta.ppf(0.975, alpha_post, beta_post)
    df["credible_above_50"] = df["credible_low"] > 0.50

    return df


# ---------------------------------------------------------------------------
# Top-level RQ2 computation
# ---------------------------------------------------------------------------

def compute_rq2(df: pd.DataFrame, min_n: int = MIN_N) -> dict:
    """Run the full RQ2 analysis on *tweets_validated.csv* data.

    Returns a dict with keys:
        min_n, n_total_accounts, n_qualifying,
        account_df  (DataFrame with all per-account results),
        corrections (dict from bh_bonferroni),
        prior       (dict: alpha, beta, prior_mean),
        n_credible_above_50
    """
    # Total accounts (regardless of threshold)
    val = df[_as_bool(df["validated"])]
    n_total = val["author"].nunique()

    # Per-account stats (filtered to >= min_n)
    acc = per_account_stats(df, min_n=min_n)
    n_qual = len(acc)

    # Multiple-testing correction on beats_market p-values
    corr = bh_bonferroni(acc["pval_beats"].values)

    # Merge correction results back
    acc["pval_beats_bh"]   = corr["bh_pvals_adj"]
    acc["pval_beats_bonf"] = corr["bonf_pvals_adj"]
    acc["reject_bh"]       = corr["bh_reject"]
    acc["reject_bonf"]     = corr["bonf_reject"]

    # Empirical-Bayes prior from pooled accounts
    alpha_p, beta_p = fit_beta_prior(acc["beats_rate"].values, acc["n"].values)

    # Shrinkage
    acc = add_shrinkage(acc, alpha_p, beta_p)

    # Sort by shrunk estimate (descending) for presentation
    acc = acc.sort_values("shrunk_beats", ascending=False)

    n_above_50 = int(acc["credible_above_50"].sum())

    return {
        "min_n":              min_n,
        "n_total_accounts":   n_total,
        "n_qualifying":       n_qual,
        "account_df":         acc,
        "corrections":        corr,
        "prior": {
            "alpha":      alpha_p,
            "beta":       beta_p,
            "prior_mean": alpha_p / (alpha_p + beta_p),
        },
        "n_credible_above_50": n_above_50,
    }
