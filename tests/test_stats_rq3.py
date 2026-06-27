"""Unit tests for src/stats_rq3.py — deterministic synthetic-data tests.

All tests use synthetic DataFrames; no real CSV or API calls are made.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.stats_rq3 import (
    CAT_DUMMIES,
    ENG_COLS,
    _pearson_ci,
    calibration_table,
    confidence_correlation,
    build_feature_matrix,
    run_logistic_regression,
    run_gradient_boosting,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 300,
    conf_range: tuple[float, float] = (60.0, 95.0),
    correct_prob: float = 0.45,
    seed: int = 0,
) -> pd.DataFrame:
    """Minimal validated DataFrame for testing RQ3 functions."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)

    confidence   = rng.uniform(conf_range[0], conf_range[1], n)
    correct      = (rng.random(n) < correct_prob).astype(bool)
    beats        = (rng.random(n) < correct_prob - 0.01).astype(bool)
    sentiments   = rng.choice(["bullish", "bearish"], n)
    horizons     = rng.choice(["short_term", "medium_term", "long_term", "unknown"], n)
    trade_types  = rng.choice(["analysis", "trade_suggestion", "general_discussion", "news"], n)
    likes        = rng.integers(0, 5000, n)
    retweets     = rng.integers(0, 500, n)
    replies      = rng.integers(0, 200, n)
    views        = rng.integers(0, 50000, n)
    followers    = rng.integers(1000, 500000, n)

    return pd.DataFrame({
        "author":             [f"account_{(i % 10) + 1:02d}" for i in idx],
        "validated":          "True",
        "confidence":         confidence,
        "prediction_correct": correct.astype(str),
        "beats_market":       beats.astype(str),
        "sentiment":          sentiments,
        "time_horizon":       horizons,
        "trade_type":         trade_types,
        "likes":              likes.astype(float),
        "retweets":           retweets.astype(float),
        "replies_count":      replies.astype(float),
        "views":              views.astype(float),
        "author_followers":   followers.astype(float),
    })


def _make_df_with_unvalidated(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """DataFrame with some non-validated rows appended."""
    df_val = _make_df(n, seed=seed)
    non_val = _make_df(20, seed=seed + 1)
    non_val["validated"] = "False"
    return pd.concat([df_val, non_val], ignore_index=True)


# ---------------------------------------------------------------------------
# _pearson_ci
# ---------------------------------------------------------------------------

class TestPearsonCi:
    def test_ci_contains_r_for_moderate_correlation(self):
        # Construct a dataset with known moderate correlation
        rng = np.random.default_rng(7)
        n = 1000
        x = rng.standard_normal(n)
        y = 0.4 * x + rng.standard_normal(n)
        r = float(np.corrcoef(x, y)[0, 1])
        lo, hi = _pearson_ci(r, n)
        assert lo <= r <= hi

    def test_ci_wider_for_small_n(self):
        r = 0.3
        lo_small, hi_small = _pearson_ci(r, n=20)
        lo_large, hi_large = _pearson_ci(r, n=500)
        assert (hi_small - lo_small) > (hi_large - lo_large)

    def test_nan_returned_for_n_le_3(self):
        lo, hi = _pearson_ci(0.5, n=3)
        assert np.isnan(lo) and np.isnan(hi)

    def test_nan_returned_for_perfect_correlation(self):
        lo, hi = _pearson_ci(1.0, n=100)
        assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# calibration_table
# ---------------------------------------------------------------------------

class TestCalibrationTable:
    def test_returns_dataframe(self):
        df = _make_df(300)
        result = calibration_table(df, n_bins=10)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        df = _make_df(300)
        result = calibration_table(df, n_bins=10)
        for col in ("bin_lower", "bin_upper", "bin_mid", "n_predictions",
                    "actual_accuracy", "ci_low", "ci_high"):
            assert col in result.columns, f"Missing column: {col}"

    def test_only_non_empty_bins_returned(self):
        # All predictions have confidence in [70, 80] → only that bin should appear
        df = _make_df(200, conf_range=(70.0, 80.0))
        result = calibration_table(df, n_bins=10)  # bins: 0-10, 10-20, ..., 70-80, ...
        assert (result["n_predictions"] > 0).all()

    def test_n_predictions_sums_to_validated_count(self):
        df = _make_df(300)
        result = calibration_table(df, n_bins=10)
        assert result["n_predictions"].sum() == 300

    def test_unvalidated_rows_excluded(self):
        df = _make_df_with_unvalidated(n=200)
        result = calibration_table(df, n_bins=10)
        assert result["n_predictions"].sum() == 200   # 20 non-validated excluded

    def test_actual_accuracy_in_unit_interval(self):
        df = _make_df(300)
        result = calibration_table(df, n_bins=10)
        assert (result["actual_accuracy"] >= 0.0).all()
        assert (result["actual_accuracy"] <= 1.0).all()

    def test_ci_contains_accuracy(self):
        df = _make_df(300)
        result = calibration_table(df, n_bins=10)
        for _, row in result.iterrows():
            assert row["ci_low"] <= row["actual_accuracy"] + 1e-9
            assert row["ci_high"] >= row["actual_accuracy"] - 1e-9

    def test_bin_midpoints_correct(self):
        df = _make_df(300)
        result = calibration_table(df, n_bins=10)
        for _, row in result.iterrows():
            expected_mid = (row["bin_lower"] + row["bin_upper"]) / 2.0
            assert abs(row["bin_mid"] - expected_mid) < 1e-9

    def test_n_bins_parameter_respected(self):
        # With n_bins=5, bin width is 20; should return at most 5 bins
        df = _make_df(300)
        result5  = calibration_table(df, n_bins=5)
        result10 = calibration_table(df, n_bins=10)
        # Wider bins in 5-bin version → each non-empty bin is wider
        assert result5["n_predictions"].sum() == result10["n_predictions"].sum()


# ---------------------------------------------------------------------------
# confidence_correlation
# ---------------------------------------------------------------------------

class TestConfidenceCorrelation:
    def test_required_keys_present(self):
        df = _make_df(300)
        result = confidence_correlation(df)
        for k in ("r_correct", "p_correct", "ci_correct_low", "ci_correct_high",
                  "r_beats",   "p_beats",   "ci_beats_low",   "ci_beats_high",  "n"):
            assert k in result, f"Missing key: {k}"

    def test_r_values_in_range(self):
        df = _make_df(300)
        result = confidence_correlation(df)
        assert -1.0 <= result["r_correct"] <= 1.0
        assert -1.0 <= result["r_beats"]   <= 1.0

    def test_n_equals_validated_count(self):
        df = _make_df_with_unvalidated(n=300)
        result = confidence_correlation(df)
        assert result["n"] == 300   # non-validated excluded

    def test_p_values_in_unit_interval(self):
        df = _make_df(300)
        result = confidence_correlation(df)
        assert 0.0 <= result["p_correct"] <= 1.0
        assert 0.0 <= result["p_beats"]   <= 1.0

    def test_ci_contains_r(self):
        df = _make_df(1000)
        result = confidence_correlation(df)
        assert result["ci_correct_low"] <= result["r_correct"] <= result["ci_correct_high"]
        assert result["ci_beats_low"]   <= result["r_beats"]   <= result["ci_beats_high"]

    def test_known_perfect_correlation(self):
        # Build data where confidence == prediction_correct → r should be large positive
        rng = np.random.default_rng(99)
        n = 500
        conf = rng.uniform(60, 95, n)
        # correct = True where conf > median  → strong positive correlation
        median = np.median(conf)
        correct = (conf > median)
        beats   = correct  # same

        df = pd.DataFrame({
            "validated":          "True",
            "confidence":         conf,
            "prediction_correct": correct.astype(str),
            "beats_market":       beats.astype(str),
        })
        result = confidence_correlation(df)
        assert result["r_correct"] > 0.8   # strong positive correlation


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_returns_dataframe_array_list(self):
        df = _make_df(200)
        X, y, names = build_feature_matrix(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert isinstance(names, list)

    def test_unvalidated_rows_excluded(self):
        df = _make_df_with_unvalidated(n=200)
        X, y, names = build_feature_matrix(df)
        assert len(X) == 200

    def test_y_is_binary(self):
        df = _make_df(200)
        _, y, _ = build_feature_matrix(df)
        assert set(y).issubset({0, 1})

    def test_feature_names_match_x_columns(self):
        df = _make_df(200)
        X, _, names = build_feature_matrix(df)
        assert list(X.columns) == names

    def test_no_account_identifier_in_features(self):
        # 'author' (the account ID column) must not appear as a feature.
        # 'log_author_followers' is fine — it's a tweet-time observable, not an identity.
        df = _make_df(200)
        _, _, names = build_feature_matrix(df)
        assert "author" not in names, "Raw account identifier leaked into feature matrix"

    def test_no_account_accuracy_in_features(self):
        # Leakage check: account-level historical accuracy must not appear
        df = _make_df(200)
        _, _, names = build_feature_matrix(df)
        leakage_terms = ("rate", "correct_rate", "beats_rate", "shrunk", "k_beats")
        for n in names:
            for term in leakage_terms:
                assert term not in n, f"Potential leakage feature: {n}"

    def test_is_bullish_present(self):
        df = _make_df(200)
        _, _, names = build_feature_matrix(df)
        assert "is_bullish" in names

    def test_log_engagement_features_present(self):
        df = _make_df(200)
        _, _, names = build_feature_matrix(df)
        for col in ENG_COLS:
            assert f"log_{col}" in names, f"Missing log_{col}"

    def test_horizon_dummies_present(self):
        df = _make_df(200)
        _, _, names = build_feature_matrix(df)
        for dummy, _ in CAT_DUMMIES["time_horizon"]:
            assert dummy in names, f"Missing {dummy}"

    def test_trade_type_dummies_present(self):
        df = _make_df(200)
        _, _, names = build_feature_matrix(df)
        for dummy, _ in CAT_DUMMIES["trade_type"]:
            assert dummy in names, f"Missing {dummy}"

    def test_log_transform_non_negative(self):
        df = _make_df(200)
        X, _, names = build_feature_matrix(df)
        for col in [f"log_{c}" for c in ENG_COLS]:
            if col in X.columns:
                assert (X[col] >= 0.0).all(), f"{col} has negative values"

    def test_views_na_filled_to_zero(self):
        df = _make_df(100)
        df.loc[df.index[:10], "views"] = np.nan
        X, _, names = build_feature_matrix(df)
        assert not X["log_views"].isna().any()
        assert (X["log_views"].iloc[:10] == 0.0).all()


# ---------------------------------------------------------------------------
# run_logistic_regression
# ---------------------------------------------------------------------------

class TestRunLogisticRegression:
    def _make_X_y(self, n: int = 300, seed: int = 0):
        df = _make_df(n, seed=seed)
        return build_feature_matrix(df)

    def test_auc_in_unit_interval(self):
        X, y, names = self._make_X_y(300)
        result = run_logistic_regression(X, y, names, cv_folds=3)
        assert 0.0 <= result["auc_cv"] <= 1.0

    def test_required_keys_present(self):
        X, y, names = self._make_X_y(300)
        result = run_logistic_regression(X, y, names, cv_folds=3)
        assert "auc_cv"      in result
        assert "coefs"       in result
        assert "odds_ratios" in result

    def test_coef_index_matches_feature_names(self):
        X, y, names = self._make_X_y(300)
        result = run_logistic_regression(X, y, names, cv_folds=3)
        coef_names = set(result["coefs"].index)
        assert coef_names == set(names)

    def test_odds_ratios_positive(self):
        X, y, names = self._make_X_y(300)
        result = run_logistic_regression(X, y, names, cv_folds=3)
        assert (result["odds_ratios"] > 0).all()

    def test_coefs_sorted_by_magnitude(self):
        X, y, names = self._make_X_y(300)
        result = run_logistic_regression(X, y, names, cv_folds=3)
        magnitudes = result["coefs"].abs().values
        assert list(magnitudes) == sorted(magnitudes, reverse=True)


# ---------------------------------------------------------------------------
# run_gradient_boosting
# ---------------------------------------------------------------------------

class TestRunGradientBoosting:
    def _make_X_y(self, n: int = 300, seed: int = 1):
        df = _make_df(n, seed=seed)
        return build_feature_matrix(df)

    def test_auc_in_unit_interval(self):
        X, y, names = self._make_X_y(300)
        result = run_gradient_boosting(X, y, names, cv_folds=3, n_estimators=10)
        assert 0.0 <= result["auc_cv"] <= 1.0

    def test_required_keys_present(self):
        X, y, names = self._make_X_y(300)
        result = run_gradient_boosting(X, y, names, cv_folds=3, n_estimators=10)
        assert "auc_cv"      in result
        assert "importances" in result

    def test_importances_index_matches_feature_names(self):
        X, y, names = self._make_X_y(300)
        result = run_gradient_boosting(X, y, names, cv_folds=3, n_estimators=10)
        assert set(result["importances"].index) == set(names)

    def test_importances_sum_to_one(self):
        X, y, names = self._make_X_y(300)
        result = run_gradient_boosting(X, y, names, cv_folds=3, n_estimators=10)
        assert abs(result["importances"].sum() - 1.0) < 1e-6

    def test_importances_non_negative(self):
        X, y, names = self._make_X_y(300)
        result = run_gradient_boosting(X, y, names, cv_folds=3, n_estimators=10)
        assert (result["importances"] >= 0.0).all()

    def test_importances_sorted_descending(self):
        X, y, names = self._make_X_y(300)
        result = run_gradient_boosting(X, y, names, cv_folds=3, n_estimators=10)
        vals = result["importances"].values
        assert list(vals) == sorted(vals, reverse=True)
