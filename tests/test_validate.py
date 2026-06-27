"""Unit tests for src/validate.py — yfinance calls are fully mocked."""

from __future__ import annotations

import pathlib
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.validate import (
    HORIZON_WINDOWS,
    VALIDATION_COLS,
    _first_ticker,
    _nearest_on_or_after,
    _offset_trading_day,
    _not_validated,
    validate_prediction,
    validate_batch,
    get_price_series,
)


# ---------------------------------------------------------------------------
# Synthetic price series helpers
# ---------------------------------------------------------------------------

def _make_series(start: str = "2021-01-04", periods: int = 300, step: float = 1.0) -> pd.Series:
    """Business-day price series starting at 100 + step increments."""
    dates = pd.bdate_range(start, periods=periods)
    prices = [100.0 + i * step for i in range(periods)]
    return pd.Series(prices, index=pd.DatetimeIndex(dates), name="TEST", dtype=float)


def _spy_flat(start: str = "2021-01-04", periods: int = 300) -> pd.Series:
    """SPY series that is perfectly flat (all prices = 200.0)."""
    dates = pd.bdate_range(start, periods=periods)
    return pd.Series([200.0] * periods, index=pd.DatetimeIndex(dates), name="SPY", dtype=float)


# ---------------------------------------------------------------------------
# _first_ticker
# ---------------------------------------------------------------------------

class TestFirstTicker:
    def test_returns_first_from_list(self):
        assert _first_ticker("['AAPL', 'TSLA']") == "AAPL"

    def test_single_ticker(self):
        assert _first_ticker("['MSFT']") == "MSFT"

    def test_empty_list_returns_none(self):
        assert _first_ticker("[]") is None

    def test_malformed_string_returns_none(self):
        assert _first_ticker("not a list") is None

    def test_nan_returns_none(self):
        assert _first_ticker(float("nan")) is None


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------

class TestIndexUtils:
    def setup_method(self):
        self.idx = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=10))

    def test_nearest_on_or_after_exact(self):
        result = _nearest_on_or_after(self.idx, self.idx[0])
        assert result == self.idx[0]

    def test_nearest_on_or_after_weekend(self):
        # Saturday 2021-01-02 → should return first Monday 2021-01-04
        sat = pd.Timestamp("2021-01-02")
        result = _nearest_on_or_after(self.idx, sat)
        assert result == self.idx[0]

    def test_nearest_on_or_after_beyond_end_is_none(self):
        future = pd.Timestamp("2030-01-01")
        assert _nearest_on_or_after(self.idx, future) is None

    def test_offset_trading_day_correct(self):
        result = _offset_trading_day(self.idx, self.idx[0], 3)
        assert result == self.idx[3]

    def test_offset_beyond_end_is_none(self):
        result = _offset_trading_day(self.idx, self.idx[0], 100)
        assert result is None

    def test_offset_entry_not_in_index_is_none(self):
        not_in_idx = pd.Timestamp("2021-01-02")  # Saturday, not in bdate range
        result = _offset_trading_day(self.idx, not_in_idx, 1)
        assert result is None


# ---------------------------------------------------------------------------
# validate_prediction — happy path
# ---------------------------------------------------------------------------

class TestValidatePrediction:
    def setup_method(self):
        # stock rises 1 point per day: entry=100, target@21=121 → +21%
        self.stock = _make_series(periods=200)
        # SPY flat: spy_return = 0, excess_return = stock_return
        self.spy = _spy_flat(periods=200)
        # Use a past date so target is safely in the past
        self.entry_date = pd.Timestamp("2021-01-04")

    def test_bullish_correct_when_stock_rises(self):
        result = validate_prediction(
            created_date=self.entry_date, sentiment="bullish",
            horizon="short_term", stock_series=self.stock, spy_series=self.spy,
        )
        assert result["validated"] is True
        assert result["prediction_correct"] is True

    def test_bearish_wrong_when_stock_rises(self):
        result = validate_prediction(
            created_date=self.entry_date, sentiment="bearish",
            horizon="short_term", stock_series=self.stock, spy_series=self.spy,
        )
        assert result["validated"] is True
        assert result["prediction_correct"] is False

    def test_beats_market_bullish_with_flat_spy(self):
        # stock rises → excess_return > 0 → beats market for bullish
        result = validate_prediction(
            created_date=self.entry_date, sentiment="bullish",
            horizon="short_term", stock_series=self.stock, spy_series=self.spy,
        )
        assert result["beats_market"] is True

    def test_beats_market_bearish_stock_rises_spy_flat(self):
        # stock rises → excess_return > 0 → bearish does NOT beat market
        result = validate_prediction(
            created_date=self.entry_date, sentiment="bearish",
            horizon="short_term", stock_series=self.stock, spy_series=self.spy,
        )
        assert result["beats_market"] is False

    def test_stock_return_computed_correctly(self):
        # entry_price = 100 (index 0), target at +21 days = 121 → return = 21/100
        result = validate_prediction(
            created_date=self.entry_date, sentiment="bullish",
            horizon="short_term", stock_series=self.stock, spy_series=self.spy,
        )
        assert abs(result["stock_return"] - 0.21) < 1e-5

    def test_spy_return_zero_for_flat_spy(self):
        result = validate_prediction(
            created_date=self.entry_date, sentiment="bullish",
            horizon="short_term", stock_series=self.stock, spy_series=self.spy,
        )
        assert abs(result["spy_return"]) < 1e-9

    def test_excess_return_equals_stock_return_with_flat_spy(self):
        result = validate_prediction(
            created_date=self.entry_date, sentiment="bullish",
            horizon="short_term", stock_series=self.stock, spy_series=self.spy,
        )
        assert abs(result["excess_return"] - result["stock_return"]) < 1e-9

    def test_horizon_windows_applied(self):
        for horizon, n_days in HORIZON_WINDOWS.items():
            result = validate_prediction(
                created_date=self.entry_date, sentiment="bullish",
                horizon=horizon, stock_series=self.stock, spy_series=self.spy,
            )
            # entry_price=100, target_price=100+n_days
            expected_return = n_days / 100.0
            assert abs(result["stock_return"] - expected_return) < 1e-5, horizon

    def test_unknown_horizon_sets_flag(self):
        result = validate_prediction(
            created_date=self.entry_date, sentiment="bullish",
            horizon="unknown", stock_series=self.stock, spy_series=self.spy,
        )
        assert result["horizon_was_unknown"] is True

    def test_known_horizon_flag_false(self):
        result = validate_prediction(
            created_date=self.entry_date, sentiment="bullish",
            horizon="short_term", stock_series=self.stock, spy_series=self.spy,
        )
        assert result["horizon_was_unknown"] is False

    def test_target_date_future_returns_not_validated(self):
        # Series only has 10 entries so +21 day offset falls off the end.
        # _offset_trading_day returns None → reason = "target_date_future".
        short_stock = _make_series(start="2021-01-04", periods=10)
        result = validate_prediction(
            created_date=pd.Timestamp("2021-01-04"), sentiment="bullish",
            horizon="short_term", stock_series=short_stock, spy_series=self.spy,
        )
        assert result["validated"] is False
        assert result["not_validated_reason"] == "target_date_future"

    def test_no_entry_date_when_series_exhausted(self):
        tiny = _make_series(periods=5)
        very_late = pd.Timestamp("2030-01-01")
        result = validate_prediction(
            created_date=very_late, sentiment="bullish",
            horizon="short_term", stock_series=tiny, spy_series=self.spy,
        )
        assert result["validated"] is False
        assert result["not_validated_reason"] == "no_entry_date"

    def test_all_validation_cols_present(self):
        result = validate_prediction(
            created_date=self.entry_date, sentiment="bullish",
            horizon="short_term", stock_series=self.stock, spy_series=self.spy,
        )
        for col in VALIDATION_COLS:
            if col != "prediction_ticker":
                assert col in result, f"missing column: {col}"


# ---------------------------------------------------------------------------
# get_price_series — caching
# ---------------------------------------------------------------------------

class TestGetPriceSeries:
    def test_not_found_sentinel_prevents_retry(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.validate.PRICE_CACHE_DIR", tmp_path)
        (tmp_path / "FAKE.notfound").touch()
        with patch("src.validate._download_prices") as mock_dl:
            result = get_price_series("FAKE")
        mock_dl.assert_not_called()
        assert result is None

    def test_download_called_when_no_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.validate.PRICE_CACHE_DIR", tmp_path)
        fake_series = _make_series()
        with patch("src.validate._download_prices", return_value=fake_series) as mock_dl:
            result = get_price_series("AAPL")
        mock_dl.assert_called_once_with("AAPL")
        assert result is not None
        assert len(result) == len(fake_series)

    def test_csv_cache_written_on_first_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.validate.PRICE_CACHE_DIR", tmp_path)
        fake_series = _make_series()
        with patch("src.validate._download_prices", return_value=fake_series):
            get_price_series("AAPL")
        assert (tmp_path / "AAPL.csv").exists()

    def test_csv_cache_read_on_second_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.validate.PRICE_CACHE_DIR", tmp_path)
        fake_series = _make_series()
        with patch("src.validate._download_prices", return_value=fake_series) as mock_dl:
            get_price_series("AAPL")
            get_price_series("AAPL")
        assert mock_dl.call_count == 1

    def test_none_download_writes_notfound(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.validate.PRICE_CACHE_DIR", tmp_path)
        with patch("src.validate._download_prices", return_value=None):
            result = get_price_series("BOGUS")
        assert result is None
        assert (tmp_path / "BOGUS.notfound").exists()


# ---------------------------------------------------------------------------
# validate_batch — integration
# ---------------------------------------------------------------------------

class TestValidateBatch:
    def _make_df(self):
        return pd.DataFrame({
            "tweet_id":          ["1", "2", "3", "4", "5"],
            "tickers_mentioned": ["['AAPL']", "['MSFT']", "['TSLA']", "['GME']", "[]"],
            "has_ticker":        [True, True, True, True, False],
            # Row 3 (TSLA) is bullish so it reaches ticker resolution (mocked → None)
            # Row 4 (GME)  is neutral to test not_directional path
            "sentiment":         ["bullish", "bearish", "bullish", "neutral", "bullish"],
            "time_horizon":      ["short_term", "medium_term", "long_term", "short_term", "short_term"],
            "created_date":      ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08"],
        })

    def _mock_get_price(self, ticker):
        if ticker == "SPY":
            return _spy_flat()
        if ticker in ("AAPL", "MSFT"):
            return _make_series()
        return None  # TSLA (and any other ticker) treated as unresolvable

    def test_non_ticker_rows_marked_no_ticker(self):
        df = self._make_df()
        with patch("src.validate.get_price_series", side_effect=self._mock_get_price):
            out = validate_batch(df)
        row = out[out["has_ticker"] == False].iloc[0]
        assert row["not_validated_reason"] == "no_ticker"
        assert str(row["validated"]).lower() == "false"

    def test_neutral_rows_marked_not_directional(self):
        df = self._make_df()
        with patch("src.validate.get_price_series", side_effect=self._mock_get_price):
            out = validate_batch(df)
        # GME row (row 4) has sentiment=neutral and has_ticker=True
        row = out[(out["sentiment"] == "neutral") & (out["has_ticker"] == True)].iloc[0]
        assert row["not_validated_reason"] == "not_directional"

    def test_unresolvable_ticker_marked_not_found(self):
        df = self._make_df()
        with patch("src.validate.get_price_series", side_effect=self._mock_get_price):
            out = validate_batch(df)
        # TSLA mocked to return None (unresolvable)
        row = out[out["tickers_mentioned"] == "['TSLA']"].iloc[0]
        assert row["not_validated_reason"] == "ticker_not_found"

    def test_validated_rows_have_correct_columns(self):
        df = self._make_df()
        with patch("src.validate.get_price_series", side_effect=self._mock_get_price):
            out = validate_batch(df)
        validated = out[out["validated"].astype(str).str.lower() == "true"]
        assert len(validated) >= 1
        for col in VALIDATION_COLS:
            assert col in out.columns

    def test_original_columns_preserved(self):
        df = self._make_df()
        with patch("src.validate.get_price_series", side_effect=self._mock_get_price):
            out = validate_batch(df)
        for col in df.columns:
            assert col in out.columns

    def test_prediction_ticker_is_first_ticker(self):
        df = self._make_df()
        with patch("src.validate.get_price_series", side_effect=self._mock_get_price):
            out = validate_batch(df)
        aapl_row = out[out["tickers_mentioned"] == "['AAPL']"].iloc[0]
        assert aapl_row["prediction_ticker"] == "AAPL"

    def test_empty_dataframe_returns_without_error(self):
        df = pd.DataFrame(columns=[
            "tweet_id", "tickers_mentioned", "has_ticker",
            "sentiment", "time_horizon", "created_date",
        ])
        out = validate_batch(df)
        assert len(out) == 0
