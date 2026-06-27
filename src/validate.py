"""Price validation and market-benchmark computation for FinTwit predictions (Phase 3).

For each directional prediction (bullish or bearish, has_ticker == True):
  1. Fetch adjusted-close prices for the first named ticker and SPY.
  2. Find the entry trading day (first trading day on or after created_date).
  3. Find the target trading day (entry + N trading days per horizon mapping).
  4. Compute stock_return, spy_return, excess_return.
  5. Determine prediction_correct (direction vs stock) and beats_market (vs SPY).

Look-ahead safety: predictions whose target date has not yet passed are
excluded (validated=False, not_validated_reason="target_date_future").

Horizon mapping (trading days):
  short_term   → 21
  medium_term  → 63
  long_term    → 126
  unknown      → 21  (same as short_term; flagged via horizon_was_unknown=True)

Prices are cached on disk under cache/prices/. Tickers that return an empty
yfinance response receive a .notfound sentinel so they are never retried.
The pipeline never crashes on bad tickers — they are counted and skipped.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import re
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZON_WINDOWS: dict[str, int] = {
    "short_term":  21,
    "medium_term": 63,
    "long_term":  126,
    "unknown":     21,
}

SPY_TICKER = "SPY"
PRICE_START = "2020-01-01"

PRICE_CACHE_DIR = pathlib.Path(__file__).parent.parent / "cache" / "prices"

VALIDATION_COLS = [
    "prediction_ticker",
    "entry_date",
    "entry_price",
    "target_date",
    "target_price",
    "stock_return",
    "spy_return",
    "excess_return",
    "prediction_correct",
    "beats_market",
    "validated",
    "not_validated_reason",
    "horizon_was_unknown",
]


# ---------------------------------------------------------------------------
# Price fetching and caching
# ---------------------------------------------------------------------------

def get_price_series(ticker: str) -> Optional[pd.Series]:
    """Return adjusted-close daily prices for *ticker*, or None if unavailable.

    Caches to PRICE_CACHE_DIR/{safe_ticker}.csv. A .notfound sentinel file
    prevents repeated network calls for tickers yfinance cannot resolve.
    """
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(ticker)
    csv_path = PRICE_CACHE_DIR / f"{safe}.csv"
    nf_path  = PRICE_CACHE_DIR / f"{safe}.notfound"

    if nf_path.exists():
        return None

    if csv_path.exists():
        try:
            s = pd.read_csv(csv_path, index_col=0, parse_dates=True).squeeze("columns")
            s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
            s.name = ticker
            return s.astype(float)
        except Exception:
            csv_path.unlink(missing_ok=True)

    prices = _download_prices(ticker)
    if prices is None or prices.empty:
        nf_path.touch()
        return None

    prices.to_frame("Close").to_csv(csv_path)
    return prices


def _download_prices(ticker: str) -> Optional[pd.Series]:
    """Fetch adjusted-close price series via yfinance. Returns None on any failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(start=PRICE_START, auto_adjust=True)
        if hist.empty:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        close = hist["Close"].dropna()
        # Normalise index to tz-naive midnight so comparisons against
        # created_date (also tz-naive) work without ambiguity.
        close.index = pd.DatetimeIndex(pd.to_datetime(close.index.date))
        close.name = ticker
        return close.astype(float)
    except Exception as exc:
        log.warning("yfinance download failed for %s: %s", ticker, exc)
        return None


def _safe_filename(ticker: str) -> str:
    return re.sub(r"[^A-Z0-9]", "_", ticker.upper())


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------

def _nearest_on_or_after(
    index: pd.DatetimeIndex, date: pd.Timestamp
) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(date)
    return index[pos] if pos < len(index) else None


def _offset_trading_day(
    index: pd.DatetimeIndex, entry_date: pd.Timestamp, n: int
) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(entry_date)
    if pos >= len(index) or index[pos] != entry_date:
        return None
    target_pos = pos + n
    return index[target_pos] if target_pos < len(index) else None


# ---------------------------------------------------------------------------
# Single-prediction validation
# ---------------------------------------------------------------------------

def validate_prediction(
    created_date: pd.Timestamp,
    sentiment: str,
    horizon: str,
    stock_series: pd.Series,
    spy_series: pd.Series,
) -> dict:
    """Validate one directional prediction and return a result dict.

    Does not raise; returns a not_validated dict on any data gap.
    The dict contains all VALIDATION_COLS except 'prediction_ticker'
    (added by the caller).
    """
    horizon_was_unknown = horizon == "unknown"
    n_days = HORIZON_WINDOWS.get(horizon, HORIZON_WINDOWS["unknown"])
    today = pd.Timestamp.today().normalize()

    stock_idx = pd.DatetimeIndex(stock_series.index)

    entry_date = _nearest_on_or_after(stock_idx, created_date)
    if entry_date is None:
        return _not_validated("no_entry_date", horizon_was_unknown)

    target_date = _offset_trading_day(stock_idx, entry_date, n_days)
    if target_date is None or target_date > today:
        return _not_validated("target_date_future", horizon_was_unknown)

    entry_price = float(stock_series.loc[entry_date])
    target_price = float(stock_series.loc[target_date])
    if entry_price <= 0:
        return _not_validated("zero_entry_price", horizon_was_unknown)

    stock_return = (target_price - entry_price) / entry_price

    spy_idx = pd.DatetimeIndex(spy_series.index)
    spy_entry  = _nearest_on_or_after(spy_idx, entry_date)
    spy_target = _nearest_on_or_after(spy_idx, target_date)
    if spy_entry is None or spy_target is None:
        return _not_validated("spy_data_missing", horizon_was_unknown)

    spy_entry_price  = float(spy_series.loc[spy_entry])
    spy_target_price = float(spy_series.loc[spy_target])
    if spy_entry_price <= 0:
        return _not_validated("spy_data_missing", horizon_was_unknown)

    spy_return    = (spy_target_price - spy_entry_price) / spy_entry_price
    excess_return = stock_return - spy_return

    if sentiment == "bullish":
        prediction_correct = stock_return > 0
        beats_market       = excess_return > 0
    else:  # bearish
        prediction_correct = stock_return < 0
        beats_market       = excess_return < 0

    return {
        "entry_date":        entry_date,
        "entry_price":       round(entry_price, 4),
        "target_date":       target_date,
        "target_price":      round(target_price, 4),
        "stock_return":      round(stock_return, 6),
        "spy_return":        round(spy_return, 6),
        "excess_return":     round(excess_return, 6),
        "prediction_correct": bool(prediction_correct),
        "beats_market":      bool(beats_market),
        "validated":         True,
        "not_validated_reason": pd.NA,
        "horizon_was_unknown": horizon_was_unknown,
    }


def _not_validated(reason: str, horizon_was_unknown: bool) -> dict:
    return {
        "entry_date": pd.NaT, "entry_price": pd.NA,
        "target_date": pd.NaT, "target_price": pd.NA,
        "stock_return": pd.NA, "spy_return": pd.NA, "excess_return": pd.NA,
        "prediction_correct": pd.NA, "beats_market": pd.NA,
        "validated": False, "not_validated_reason": reason,
        "horizon_was_unknown": horizon_was_unknown,
    }


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def _first_ticker(tickers_str: object) -> Optional[str]:
    """Return the first ticker from a tickers_mentioned string repr, or None."""
    try:
        lst = ast.literal_eval(str(tickers_str))
        if lst and isinstance(lst, list):
            return str(lst[0])
    except Exception:
        pass
    return None


def validate_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Validate all predictions in *df* and return the augmented DataFrame.

    New columns (VALIDATION_COLS) are added in place. Non-directional rows
    receive validated=False with a descriptive not_validated_reason.
    """
    df = df.copy()

    # Initialise output columns
    for col in VALIDATION_COLS:
        df[col] = pd.NA
    df["validated"] = False
    df["horizon_was_unknown"] = False

    # Identify row categories
    has_ticker    = df["has_ticker"].astype(str).str.lower() == "true"
    is_directional = df["sentiment"].isin({"bullish", "bearish"})

    df.loc[~has_ticker,                     "not_validated_reason"] = "no_ticker"
    df.loc[has_ticker & ~is_directional,    "not_validated_reason"] = "not_directional"

    directional_mask = has_ticker & is_directional
    if not directional_mask.any():
        return df

    # Fetch SPY once
    log.info("Loading SPY price series …")
    spy_series = get_price_series(SPY_TICKER)
    if spy_series is None:
        raise RuntimeError(
            "Could not fetch SPY prices — check internet connection and yfinance."
        )

    # Resolve first ticker for each directional row
    dir_df = df.loc[directional_mask].copy()
    dir_df["prediction_ticker"] = dir_df["tickers_mentioned"].apply(_first_ticker)

    no_pt = dir_df["prediction_ticker"].isna()
    df.loc[dir_df.index[no_pt], "not_validated_reason"] = "no_first_ticker"

    # Pre-load all price series (one yfinance call per unique ticker)
    unique_tickers = dir_df["prediction_ticker"].dropna().unique().tolist()
    log.info("Loading prices for %d unique tickers …", len(unique_tickers))
    price_cache: dict[str, Optional[pd.Series]] = {}
    for i, ticker in enumerate(unique_tickers, 1):
        if i % 100 == 0:
            log.info("  %d / %d", i, len(unique_tickers))
        price_cache[ticker] = get_price_series(ticker)

    # Row-by-row validation (all data already in memory — fast)
    for idx, row in dir_df.iterrows():
        pt = row["prediction_ticker"]
        if pd.isna(pt):
            continue

        df.at[idx, "prediction_ticker"] = pt

        stock_series = price_cache.get(pt)
        if stock_series is None:
            df.at[idx, "not_validated_reason"] = "ticker_not_found"
            df.at[idx, "horizon_was_unknown"]  = (row["time_horizon"] == "unknown")
            continue

        result = validate_prediction(
            created_date  = pd.Timestamp(row["created_date"]).normalize(),
            sentiment     = row["sentiment"],
            horizon       = str(row["time_horizon"]),
            stock_series  = stock_series,
            spy_series    = spy_series,
        )
        for col, val in result.items():
            df.at[idx, col] = val

    return df
