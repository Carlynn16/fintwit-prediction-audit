"""Validate FinTwit predictions against market prices and write tweets_validated.csv.

Usage:
    python scripts/validate_predictions.py

Reads  : data/tweets_classified.csv
Writes : data/tweets_validated.csv

Prints a funnel showing how many predictions survive each filter stage
and the overall accuracy / beats-market rates for those that are validated.

Prices are cached under cache/prices/ so re-runs are fast after the first
download pass.
"""

from __future__ import annotations

import pathlib
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import logging
import pandas as pd

from src.validate import validate_batch, VALIDATION_COLS

logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR   = pathlib.Path(__file__).parent.parent / "data"
INPUT_CSV  = DATA_DIR / "tweets_classified.csv"
OUTPUT_CSV = DATA_DIR / "tweets_validated.csv"


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(
            f"ERROR: {INPUT_CSV} not found.\n"
            "Run scripts/classify_tweets.py first."
        )

    print(f"Reading {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV, dtype={"tweet_id": str, "conversation_id": str})

    total = len(df)
    has_ticker = (df["has_ticker"].astype(str).str.lower() == "true").sum()
    directional = (
        (df["has_ticker"].astype(str).str.lower() == "true")
        & df["sentiment"].isin({"bullish", "bearish"})
    ).sum()

    print(f"Rows: {total:,}  |  has_ticker: {has_ticker:,}  "
          f"|  directional: {directional:,}")
    print("Running validation (first run downloads prices; cached afterwards) …")
    print()

    df_val = validate_batch(df)

    df_val.to_csv(OUTPUT_CSV, index=False)

    # ── Funnel ────────────────────────────────────────────────────────────────
    reasons = df_val["not_validated_reason"]
    validated_mask = df_val["validated"].astype(str).str.lower() == "true"

    n_validated         = validated_mask.sum()
    n_no_ticker         = (reasons == "no_ticker").sum()
    n_not_directional   = (reasons == "not_directional").sum()
    n_ticker_not_found  = (reasons == "ticker_not_found").sum()
    n_target_future     = (reasons == "target_date_future").sum()
    n_no_entry          = (reasons == "no_entry_date").sum()
    n_other             = directional - n_validated - n_ticker_not_found - n_target_future - n_no_entry

    val_df = df_val[validated_mask]
    n_correct     = val_df["prediction_correct"].astype(str).str.lower().eq("true").sum()
    n_beats       = val_df["beats_market"].astype(str).str.lower().eq("true").sum()
    n_unk_horizon = val_df["horizon_was_unknown"].astype(str).str.lower().eq("true").sum()

    pct = lambda n, d: f"{n/d*100:.1f}%" if d else "—"

    print("=" * 60)
    print("Validation funnel")
    print("=" * 60)
    print(f"  Total rows                     : {total:>7,}")
    print(f"  Has ticker                     : {has_ticker:>7,}  ({pct(has_ticker, total)})")
    print(f"  Directional (bull/bear)        : {directional:>7,}  ({pct(directional, total)})")
    print()
    print(f"  Excluded — ticker not in yfinance : {n_ticker_not_found:>5,}")
    print(f"  Excluded — target date future     : {n_target_future:>5,}")
    print(f"  Excluded — no entry date          : {n_no_entry:>5,}")
    if n_other > 0:
        print(f"  Excluded — other                  : {n_other:>5,}")
    print()
    print(f"  Validated predictions          : {n_validated:>7,}  ({pct(n_validated, directional)} of directional)")
    print(f"    - horizon was unknown (->21d) : {n_unk_horizon:>7,}  ({pct(n_unk_horizon, n_validated)} of validated)")
    print()
    print(f"  Prediction correct (vs stock)  : {n_correct:>7,}  ({pct(n_correct, n_validated)} of validated)")
    print(f"  Beats market      (vs SPY)     : {n_beats:>7,}  ({pct(n_beats, n_validated)} of validated)")

    print()
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
