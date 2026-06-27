"""Unit tests for src/stats_rq2.py — deterministic synthetic-data tests."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.stats_rq2 import (
    MIN_N,
    per_account_stats,
    bh_bonferroni,
    fit_beta_prior,
    add_shrinkage,
    compute_rq2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_validated_df(accounts: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Build a synthetic tweets_validated DataFrame.

    accounts : {author: (n_total, n_beats)}
    """
    rows = []
    for author, (n, k) in accounts.items():
        for i in range(n):
            rows.append({
                "author":            author,
                "validated":         "True",
                "beats_market":      str(i < k),        # first k rows are True
                "prediction_correct": str(i < k),        # same for simplicity
            })
    # Add a few non-validated rows (must be ignored)
    rows.append({"author": "account_99", "validated": "False",
                 "beats_market": "True", "prediction_correct": "True"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# per_account_stats
# ---------------------------------------------------------------------------

class TestPerAccountStats:
    def test_only_validated_rows_counted(self):
        df = _make_validated_df({"account_01": (50, 30)})
        result = per_account_stats(df, min_n=30)
        assert result.loc["account_01", "n"] == 50   # non-validated row excluded

    def test_min_n_threshold_applied(self):
        df = _make_validated_df({
            "account_01": (50, 25),
            "account_02": (29, 15),  # below threshold
        })
        result = per_account_stats(df, min_n=30)
        assert "account_01" in result.index
        assert "account_02" not in result.index

    def test_beats_rate_computed(self):
        df = _make_validated_df({"account_01": (100, 60)})
        result = per_account_stats(df, min_n=30)
        assert abs(result.loc["account_01", "beats_rate"] - 0.60) < 1e-9

    def test_correct_rate_computed(self):
        df = _make_validated_df({"account_01": (100, 40)})
        result = per_account_stats(df, min_n=30)
        assert abs(result.loc["account_01", "correct_rate"] - 0.40) < 1e-9

    def test_pval_beats_exactly_half(self):
        # k=50, n=100 → two-sided binomial p = 1.0
        df = _make_validated_df({"account_01": (100, 50)})
        result = per_account_stats(df, min_n=30)
        assert abs(result.loc["account_01", "pval_beats"] - 1.0) < 1e-9

    def test_pval_beats_significant_low(self):
        # k=5, n=100 → clearly below 50%, p very small
        df = _make_validated_df({"account_01": (100, 5)})
        result = per_account_stats(df, min_n=30)
        assert result.loc["account_01", "pval_beats"] < 0.001

    def test_pval_beats_significant_high(self):
        # k=95, n=100 → clearly above 50%, p very small
        df = _make_validated_df({"account_01": (100, 95)})
        result = per_account_stats(df, min_n=30)
        assert result.loc["account_01", "pval_beats"] < 0.001

    def test_multiple_accounts_all_present(self):
        df = _make_validated_df({
            "account_A": (50, 25),
            "account_B": (60, 35),
            "account_C": (40, 20),
        })
        result = per_account_stats(df, min_n=30)
        assert set(result.index) == {"account_A", "account_B", "account_C"}

    def test_pval_two_sided_symmetric(self):
        # k=80, n=100 should give same p-value as k=20, n=100 (symmetric around 50%)
        df_hi = _make_validated_df({"account_hi": (100, 80)})
        df_lo = _make_validated_df({"account_lo": (100, 20)})
        p_hi = per_account_stats(df_hi, min_n=30).loc["account_hi", "pval_beats"]
        p_lo = per_account_stats(df_lo, min_n=30).loc["account_lo", "pval_beats"]
        assert abs(p_hi - p_lo) < 1e-10


# ---------------------------------------------------------------------------
# bh_bonferroni
# ---------------------------------------------------------------------------

class TestBhBonferroni:
    def test_raw_significant_count(self):
        p = np.array([0.001, 0.04, 0.06, 0.5, 0.9])
        res = bh_bonferroni(p)
        assert res["raw_significant"] == 2   # 0.001 and 0.04

    def test_bonferroni_is_stricter_than_raw(self):
        p = np.array([0.04, 0.04, 0.04, 0.04, 0.04])
        res = bh_bonferroni(p)
        # 0.04 * 5 = 0.20 > 0.05 → none survive Bonferroni
        assert res["bonf_significant"] == 0

    def test_bh_rejects_obvious_case(self):
        # All p-values tiny → all should be rejected
        p = np.array([1e-10, 1e-9, 1e-8, 1e-7])
        res = bh_bonferroni(p)
        assert res["bh_significant"] == 4
        assert res["bonf_significant"] == 4

    def test_all_high_p_nothing_survives(self):
        p = np.array([0.3, 0.5, 0.7, 0.9])
        res = bh_bonferroni(p)
        assert res["raw_significant"] == 0
        assert res["bh_significant"]  == 0
        assert res["bonf_significant"] == 0

    def test_bh_less_strict_than_bonferroni(self):
        # p-values where BH passes some but Bonferroni rejects all
        p = np.array([0.001, 0.002, 0.01, 0.04, 0.5])
        res = bh_bonferroni(p)
        # Bonferroni threshold: 0.05/5 = 0.01 → only first two survive
        assert res["bonf_significant"] <= res["bh_significant"]

    def test_output_arrays_same_length_as_input(self):
        p = np.array([0.01, 0.05, 0.1])
        res = bh_bonferroni(p)
        assert len(res["bh_reject"])     == 3
        assert len(res["bonf_reject"])   == 3
        assert len(res["bh_pvals_adj"])  == 3
        assert len(res["bonf_pvals_adj"]) == 3

    def test_known_bonferroni_adjusted_value(self):
        # Bonferroni adjusted p = min(raw_p * n, 1)
        p = np.array([0.01, 0.04])
        res = bh_bonferroni(p)
        # 0.01 * 2 = 0.02; 0.04 * 2 = 0.08 (clipped to 0.08)
        assert abs(res["bonf_pvals_adj"][0] - 0.02) < 1e-9
        assert abs(res["bonf_pvals_adj"][1] - 0.08) < 1e-9


# ---------------------------------------------------------------------------
# fit_beta_prior
# ---------------------------------------------------------------------------

class TestFitBetaPrior:
    def test_returns_positive_alpha_and_beta(self):
        p = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
        n = np.array([100, 100, 100, 100, 100])
        alpha, beta = fit_beta_prior(p, n)
        assert alpha > 0
        assert beta > 0

    def test_prior_mean_near_data_mean(self):
        # Prior mean should be alpha/(alpha+beta), roughly equal to mean(p_hats)
        p = np.array([0.45, 0.46, 0.47, 0.48, 0.49])
        n = np.array([200, 200, 200, 200, 200])
        alpha, beta = fit_beta_prior(p, n)
        prior_mean = alpha / (alpha + beta)
        assert abs(prior_mean - np.mean(p)) < 0.05

    def test_high_variance_gives_lower_phi(self):
        # High between-account variance → weaker prior (smaller phi = alpha+beta)
        p_low_var  = np.array([0.47, 0.48, 0.49, 0.50, 0.51])
        p_high_var = np.array([0.2,  0.3,  0.5,  0.7,  0.8])
        n = np.array([100, 100, 100, 100, 100])
        a1, b1 = fit_beta_prior(p_low_var,  n)
        a2, b2 = fit_beta_prior(p_high_var, n)
        phi1 = a1 + b1
        phi2 = a2 + b2
        assert phi1 > phi2   # low variance → more concentrated (stronger) prior

    def test_no_crash_on_homogeneous_rates(self):
        # All same rate → var_signal clips to epsilon, phi = 1 (minimum)
        p = np.array([0.5, 0.5, 0.5, 0.5])
        n = np.array([100, 100, 100, 100])
        alpha, beta = fit_beta_prior(p, n)
        assert alpha > 0 and beta > 0


# ---------------------------------------------------------------------------
# add_shrinkage
# ---------------------------------------------------------------------------

class TestAddShrinkage:
    def _make_acc_df(self):
        return pd.DataFrame({
            "n":       [30, 300],
            "k_beats": [20, 160],
        }, index=["account_small", "account_large"])

    def test_shrunk_beats_is_between_prior_and_raw(self):
        acc = self._make_acc_df()
        alpha, beta = 5.0, 5.0   # prior mean = 0.5
        result = add_shrinkage(acc, alpha, beta)

        for idx in result.index:
            raw    = acc.loc[idx, "k_beats"] / acc.loc[idx, "n"]
            shrunk = result.loc[idx, "shrunk_beats"]
            prior  = alpha / (alpha + beta)
            # Shrunk must lie between raw and prior
            lo, hi = min(raw, prior), max(raw, prior)
            assert lo - 1e-9 <= shrunk <= hi + 1e-9, f"{idx}: shrunk={shrunk}, raw={raw}"

    def test_small_n_shrinks_more_toward_prior(self):
        # account_small (n=30) should be pulled further toward prior=0.5
        # than account_large (n=300) since raw rates are the same fraction
        acc = pd.DataFrame({
            "n":       [30,  300],
            "k_beats": [21, 210],   # both ~70%
        }, index=["small", "large"])
        alpha, beta = 5.0, 5.0   # prior mean = 0.5
        result = add_shrinkage(acc, alpha, beta)

        dist_small = abs(result.loc["small", "shrunk_beats"] - 0.5)
        dist_large = abs(result.loc["large", "shrunk_beats"] - 0.5)
        assert dist_small < dist_large   # small account pulled closer to 0.5

    def test_credible_interval_contains_shrunk_estimate(self):
        acc = self._make_acc_df()
        result = add_shrinkage(acc, 5.0, 5.0)
        for idx in result.index:
            assert result.loc[idx, "credible_low"] <= result.loc[idx, "shrunk_beats"]
            assert result.loc[idx, "shrunk_beats"] <= result.loc[idx, "credible_high"]

    def test_credible_above_50_flag(self):
        # Account with raw rate 90% and large n: CI should be above 0.5
        acc = pd.DataFrame({
            "n":       [500],
            "k_beats": [450],   # 90%
        }, index=["account_A"])
        result = add_shrinkage(acc, 2.0, 2.0)  # very weak prior
        assert result.loc["account_A", "credible_above_50"] is True \
            or result.loc["account_A", "credible_above_50"] == True

    def test_credible_above_50_false_for_mediocre(self):
        acc = pd.DataFrame({
            "n":       [100],
            "k_beats": [50],    # exactly 50% → CI straddles 0.5
        }, index=["account_B"])
        result = add_shrinkage(acc, 5.0, 5.0)
        assert not result.loc["account_B", "credible_above_50"]


# ---------------------------------------------------------------------------
# compute_rq2 — integration
# ---------------------------------------------------------------------------

class TestComputeRQ2:
    def _make_df(self):
        accounts = {f"account_{i:02d}": (100, 45) for i in range(1, 40)}
        accounts["account_40"] = (25, 20)  # below threshold
        accounts["account_41"] = (300, 270)  # very high rate
        return _make_validated_df(accounts)

    def test_all_keys_present(self):
        df = self._make_df()
        result = compute_rq2(df, min_n=30)
        for k in ("min_n", "n_total_accounts", "n_qualifying",
                  "account_df", "corrections", "prior", "n_credible_above_50"):
            assert k in result

    def test_n_qualifying_excludes_below_threshold(self):
        df = self._make_df()
        result = compute_rq2(df, min_n=30)
        # account_40 has n=25, should be excluded; account_41 (n=300) is in
        assert result["n_qualifying"] == 40   # 39 regular + account_41

    def test_prior_keys_present(self):
        df = self._make_df()
        result = compute_rq2(df, min_n=30)
        for k in ("alpha", "beta", "prior_mean"):
            assert k in result["prior"]

    def test_account_df_has_shrinkage_columns(self):
        df = self._make_df()
        result = compute_rq2(df, min_n=30)
        for col in ("shrunk_beats", "credible_low", "credible_high", "credible_above_50"):
            assert col in result["account_df"].columns

    def test_account_df_has_correction_columns(self):
        df = self._make_df()
        result = compute_rq2(df, min_n=30)
        for col in ("pval_beats_bh", "pval_beats_bonf", "reject_bh", "reject_bonf"):
            assert col in result["account_df"].columns

    def test_very_high_account_credible_above_50(self):
        # account_41 has 270/300 = 90% beats — should be credibly above 50%
        df = self._make_df()
        result = compute_rq2(df, min_n=30)
        acc_df = result["account_df"]
        row = acc_df.loc["account_41"]
        assert row["credible_above_50"]

    def test_n_credible_above_50_matches_column(self):
        df = self._make_df()
        result = compute_rq2(df, min_n=30)
        acc_df = result["account_df"]
        assert result["n_credible_above_50"] == acc_df["credible_above_50"].sum()
