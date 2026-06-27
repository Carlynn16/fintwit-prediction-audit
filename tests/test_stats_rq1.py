"""Unit tests for src/stats_rq1.py — all deterministic, no network calls."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.stats_rq1 import (
    wilson_ci,
    proportion_ztest,
    segment_stats,
    compute_rq1,
)


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_zero_n_returns_full_interval(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0 and hi == 1.0

    def test_symmetric_at_50_pct(self):
        lo, hi = wilson_ci(50, 100)
        # Wilson is symmetric around p_hat = 0.5 by construction
        assert abs((lo + hi) / 2 - 0.5) < 1e-9

    def test_bounds_are_within_zero_one(self):
        for k, n in [(0, 10), (10, 10), (3, 7), (1, 100)]:
            lo, hi = wilson_ci(k, n)
            assert 0.0 <= lo <= hi <= 1.0, f"k={k}, n={n}"

    def test_all_successes_upper_bound_lt_one(self):
        # Wilson does not return exactly 1.0 when k == n for n > 1
        lo, hi = wilson_ci(100, 100)
        assert hi <= 1.0

    def test_zero_successes_lower_bound_ge_zero(self):
        lo, hi = wilson_ci(0, 100)
        assert lo >= 0.0

    def test_interval_width_shrinks_with_n(self):
        _, hi_small = wilson_ci(50, 100)
        lo_small, _  = wilson_ci(50, 100)
        lo_large, hi_large = wilson_ci(500, 1000)
        width_small = hi_small - lo_small
        width_large = hi_large - lo_large
        assert width_large < width_small

    def test_90pct_narrower_than_95pct(self):
        lo95, hi95 = wilson_ci(45, 100, confidence=0.95)
        lo90, hi90 = wilson_ci(45, 100, confidence=0.90)
        assert (hi90 - lo90) < (hi95 - lo95)

    def test_approximate_values_known_case(self):
        # p_hat = 0.45, n = 100 → Wilson CI ≈ (0.355, 0.548)
        lo, hi = wilson_ci(45, 100)
        assert 0.34 < lo < 0.37
        assert 0.53 < hi < 0.56

    def test_p_hat_inside_ci(self):
        # Skip k=0 and k=100: Wilson CI lower/upper bounds are slightly off-boundary
        # at p=0 and p=1 by float precision (O(1e-18)), not a meaningful error.
        for k in range(10, 100, 10):
            lo, hi = wilson_ci(k, 100)
            p = k / 100
            assert lo <= p <= hi, f"p={p}, ci=({lo},{hi})"


# ---------------------------------------------------------------------------
# proportion_ztest
# ---------------------------------------------------------------------------

class TestProportionZtest:
    def test_zero_n_returns_nan(self):
        result = proportion_ztest(0, 0)
        assert np.isnan(result["z_stat"])
        assert np.isnan(result["p_value"])

    def test_null_is_true_at_50_pct(self):
        # k=50, n=100, p0=0.5 → z=0, p_value=1.0
        result = proportion_ztest(50, 100)
        assert abs(result["z_stat"]) < 1e-10
        assert abs(result["p_value"] - 1.0) < 1e-10

    def test_p_hat_computed_correctly(self):
        result = proportion_ztest(40, 100)
        assert abs(result["p_hat"] - 0.40) < 1e-10

    def test_z_stat_direction(self):
        # p_hat < p0 → z_stat negative
        result = proportion_ztest(40, 100)
        assert result["z_stat"] < 0

    def test_z_stat_known_value(self):
        # k=40, n=100, p0=0.5 → z = -0.1/0.05 = -2.0
        result = proportion_ztest(40, 100)
        assert abs(result["z_stat"] - (-2.0)) < 1e-9

    def test_p_value_two_tailed(self):
        # |z|=2.0 → p ≈ 0.0455
        result = proportion_ztest(40, 100)
        assert 0.044 < result["p_value"] < 0.047

    def test_significant_at_large_n(self):
        # 45% accuracy with n=10000 is extremely significant
        result = proportion_ztest(4500, 10000)
        assert result["p_value"] < 1e-20

    def test_ci_returned_as_floats(self):
        result = proportion_ztest(50, 100)
        assert isinstance(result["ci_low"],  float)
        assert isinstance(result["ci_high"], float)

    def test_k_and_n_in_result(self):
        result = proportion_ztest(40, 100)
        assert result["k"] == 40
        assert result["n"] == 100

    def test_custom_p0(self):
        # k=60, n=100, p0=0.6 → z=0, p=1
        result = proportion_ztest(60, 100, p0=0.6)
        assert abs(result["z_stat"]) < 1e-9


# ---------------------------------------------------------------------------
# segment_stats
# ---------------------------------------------------------------------------

class TestSegmentStats:
    def _make_df(self):
        return pd.DataFrame({
            "sentiment": ["bullish"] * 10 + ["bearish"] * 10,
            "correct":   [True] * 8 + [False] * 2 + [True] * 3 + [False] * 7,
            "beats":     [True] * 6 + [False] * 4 + [True] * 4 + [False] * 6,
        })

    def test_returns_one_row_per_group(self):
        df = self._make_df()
        result = segment_stats(df, "sentiment")
        assert len(result) == 2

    def test_index_is_group_values(self):
        df = self._make_df()
        result = segment_stats(df, "sentiment")
        assert "bullish" in result.index
        assert "bearish" in result.index

    def test_accuracy_correct_computed(self):
        df = self._make_df()
        result = segment_stats(df, "sentiment")
        assert abs(result.loc["bullish", "acc_correct"] - 0.8) < 1e-10
        assert abs(result.loc["bearish", "acc_correct"] - 0.3) < 1e-10

    def test_accuracy_beats_computed(self):
        df = self._make_df()
        result = segment_stats(df, "sentiment")
        assert abs(result.loc["bullish", "acc_beats"] - 0.6) < 1e-10
        assert abs(result.loc["bearish", "acc_beats"] - 0.4) < 1e-10

    def test_n_correct(self):
        df = self._make_df()
        result = segment_stats(df, "sentiment")
        assert result.loc["bullish", "n"] == 10
        assert result.loc["bearish", "n"] == 10

    def test_k_correct_values(self):
        df = self._make_df()
        result = segment_stats(df, "sentiment")
        assert result.loc["bullish", "k_correct"] == 8
        assert result.loc["bearish", "k_correct"] == 3

    def test_ci_bounds_present(self):
        df = self._make_df()
        result = segment_stats(df, "sentiment")
        for col in ["ci_correct_low", "ci_correct_high", "ci_beats_low", "ci_beats_high"]:
            assert col in result.columns

    def test_ci_contains_p_hat(self):
        df = self._make_df()
        result = segment_stats(df, "sentiment")
        for group in result.index:
            acc = result.loc[group, "acc_correct"]
            lo  = result.loc[group, "ci_correct_low"]
            hi  = result.loc[group, "ci_correct_high"]
            assert lo <= acc <= hi, f"group={group}: {lo} <= {acc} <= {hi}"


# ---------------------------------------------------------------------------
# compute_rq1
# ---------------------------------------------------------------------------

class TestComputeRQ1:
    def _make_validated_df(self, n_correct=45, n_beats=40, n_total=100):
        """Synthetic tweets_validated.csv with n_total rows, all validated."""
        import math
        correct = [True] * n_correct + [False] * (n_total - n_correct)
        beats   = [True] * n_beats   + [False] * (n_total - n_beats)
        # Categorical columns scale proportionally with n_total
        n_bull = math.floor(n_total * 0.70)
        n_st   = math.floor(n_total * 0.50)
        n_mt   = math.floor(n_total * 0.30)
        n_lt   = n_total - n_st - n_mt
        n_ana  = math.floor(n_total * 0.60)
        n_unk  = math.floor(n_total * 0.20)
        return pd.DataFrame({
            "validated":          ["True"]  * n_total,
            "prediction_correct": correct,
            "beats_market":       beats,
            "stock_return":       [0.02]  * n_total,
            "spy_return":         [0.015] * n_total,
            "excess_return":      [0.005] * n_total,
            "sentiment":          ["bullish"] * n_bull + ["bearish"] * (n_total - n_bull),
            "time_horizon":       ["short_term"] * n_st + ["medium_term"] * n_mt + ["long_term"] * n_lt,
            "trade_type":         ["analysis"] * n_ana + ["trade_suggestion"] * (n_total - n_ana),
            "horizon_was_unknown":["True"] * n_unk + ["False"] * (n_total - n_unk),
        })

    def test_all_keys_present(self):
        df = self._make_validated_df()
        result = compute_rq1(df)
        for key in ("n_validated", "overall", "segments", "returns", "sensitivity"):
            assert key in result

    def test_n_validated_correct(self):
        df = self._make_validated_df(n_total=100)
        result = compute_rq1(df)
        assert result["n_validated"] == 100

    def test_overall_correct_p_hat(self):
        df = self._make_validated_df(n_correct=45, n_total=100)
        result = compute_rq1(df)
        assert abs(result["overall"]["correct"]["p_hat"] - 0.45) < 1e-9

    def test_overall_beats_p_hat(self):
        df = self._make_validated_df(n_beats=40, n_total=100)
        result = compute_rq1(df)
        assert abs(result["overall"]["beats"]["p_hat"] - 0.40) < 1e-9

    def test_segments_all_three_present(self):
        df = self._make_validated_df()
        result = compute_rq1(df)
        for key in ("sentiment", "time_horizon", "trade_type"):
            assert key in result["segments"]

    def test_returns_computed(self):
        df = self._make_validated_df()
        result = compute_rq1(df)
        assert abs(result["returns"]["mean_stock_return"]  - 0.02)  < 1e-9
        assert abs(result["returns"]["mean_spy_return"]    - 0.015) < 1e-9
        assert abs(result["returns"]["mean_excess_return"] - 0.005) < 1e-9

    def test_sensitivity_excludes_unknown_horizon(self):
        df = self._make_validated_df(n_total=100)  # 20 have horizon_was_unknown=True
        result = compute_rq1(df)
        assert result["sensitivity"]["n"] == 80

    def test_sensitivity_has_required_keys(self):
        df = self._make_validated_df()
        result = compute_rq1(df)
        sens = result["sensitivity"]
        for key in ("n", "acc_correct", "ci_correct", "p_value_correct",
                    "acc_beats",   "ci_beats",   "p_value_beats"):
            assert key in sens

    def test_non_validated_rows_excluded(self):
        # Add 50 non-validated rows; only the 100 validated rows should count
        df_val = self._make_validated_df(n_total=100)
        df_nonval = df_val.copy()
        df_nonval["validated"] = "False"
        df = pd.concat([df_val, df_nonval], ignore_index=True)
        result = compute_rq1(df)
        assert result["n_validated"] == 100

    def test_significant_below_50pct(self):
        # 45% accuracy with n=10000 must be significant vs 50%
        df = self._make_validated_df(n_correct=4500, n_beats=4400, n_total=10000)
        result = compute_rq1(df)
        assert result["overall"]["correct"]["p_value"] < 0.001
        assert result["overall"]["beats"]["p_value"]   < 0.001
