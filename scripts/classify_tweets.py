"""Classify ticker-bearing tweets with gpt-4o-mini and write tweets_classified.csv.

Usage:
    python scripts/classify_tweets.py [--no-llm]

--no-llm  skips API calls (useful for offline/CI runs); classified rows get fallback
          values so the output file still has all columns.

Only rows where has_ticker == True are sent to the LLM. The rest receive null/NA for
all four classification columns.
"""

from __future__ import annotations

import os
import pathlib
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
from src.classify import classify_batch, FALLBACK

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
INPUT_CSV = DATA_DIR / "tweets_with_tickers.csv"
OUTPUT_CSV = DATA_DIR / "tweets_classified.csv"

CLASS_COLS = ["sentiment", "time_horizon", "trade_type", "confidence"]


def main(use_llm: bool = True) -> None:
    df = pd.read_csv(INPUT_CSV)
    api_key = os.getenv("OPENAI_API_KEY", "") if use_llm else ""

    if use_llm and not api_key:
        raise SystemExit(
            "ERROR: OPENAI_API_KEY not found.\n"
            "Create a .env file at the repo root with OPENAI_API_KEY=sk-...\n"
            "or run with --no-llm."
        )

    # Initialise all classification columns to NA
    for col in CLASS_COLS:
        df[col] = pd.NA

    ticker_mask = df["has_ticker"].astype(str).str.lower() == "true"
    ticker_idx = df.index[ticker_mask].tolist()
    texts = df.loc[ticker_mask, "text"].tolist()

    n_to_classify = len(texts)
    print(f"Classifying {n_to_classify:,} ticker-bearing tweets …")

    results, n_api, n_cache = classify_batch(
        texts, api_key=api_key if use_llm else "", verbose=True
    )

    for i, (idx, res) in enumerate(zip(ticker_idx, results)):
        for col in CLASS_COLS:
            df.at[idx, col] = res[col]

    # confidence → nullable Int64 so NAs are preserved
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").astype("Int64")

    df.to_csv(OUTPUT_CSV, index=False)

    # --- Summary ---
    classified = df[ticker_mask]
    print()
    print("=" * 54)
    print("Classification summary")
    print("=" * 54)
    print(f"  Rows classified          : {n_to_classify:>7,}")
    print(f"  New API calls            : {n_api:>7,}")
    print(f"  Cache hits               : {n_cache:>7,}")
    print()

    print("  Sentiment distribution:")
    for val, cnt in classified["sentiment"].value_counts().items():
        print(f"    {val:<22} {cnt:>6,}  ({cnt/n_to_classify*100:.1f}%)")

    print()
    print("  Time-horizon distribution:")
    for val, cnt in classified["time_horizon"].value_counts().items():
        print(f"    {val:<22} {cnt:>6,}  ({cnt/n_to_classify*100:.1f}%)")

    print()
    print("  Trade-type distribution:")
    for val, cnt in classified["trade_type"].value_counts().items():
        print(f"    {val:<22} {cnt:>6,}  ({cnt/n_to_classify*100:.1f}%)")

    conf = classified["confidence"].dropna()
    print()
    print(f"  Confidence — mean: {conf.mean():.1f}  median: {conf.median():.0f}  "
          f"min: {conf.min()}  max: {conf.max()}")
    print(f"\nOutput: {OUTPUT_CSV}")


if __name__ == "__main__":
    no_llm = "--no-llm" in sys.argv
    main(use_llm=not no_llm)
