"""Run ticker extraction over data/tweets_raw.csv and write data/tweets_with_tickers.csv.

Usage:
    python scripts/extract_tickers.py [--no-llm]

The --no-llm flag skips the LLM fallback (useful for offline runs / CI).
Coverage statistics are printed on completion.
"""

from __future__ import annotations

import os
import pathlib
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
from src.tickers import extract_regex, extract_llm, _normalise_list

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
INPUT_CSV = DATA_DIR / "tweets_raw.csv"
OUTPUT_CSV = DATA_DIR / "tweets_with_tickers.csv"


def main(use_llm: bool = True) -> None:
    df = pd.read_csv(INPUT_CSV)
    api_key = os.getenv("OPENAI_API_KEY", "") if use_llm else ""

    if use_llm and not api_key:
        raise SystemExit(
            "ERROR: OPENAI_API_KEY not found.\n"
            "Create a .env file at the repo root containing:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "Or run with --no-llm to use regex only."
        )

    regex_tickers: list[list[str]] = []
    llm_tickers: list[list[str]] = []
    combined_tickers: list[list[str]] = []

    n = len(df)
    for i, text in enumerate(df["text"]):
        if i % 1000 == 0:
            print(f"  {i}/{n} tweets processed…")

        r_hits = extract_regex(text if isinstance(text, str) else "")
        regex_tickers.append(r_hits)

        if r_hits or not use_llm or not api_key:
            l_hits: list[str] = []
        else:
            l_hits = extract_llm(text, api_key=api_key)
        llm_tickers.append(l_hits)

        combined = _normalise_list(r_hits + l_hits)
        combined_tickers.append(combined)

    df["regex_tickers"] = [str(t) for t in regex_tickers]
    df["llm_tickers"] = [str(t) for t in llm_tickers]
    df["tickers_mentioned"] = [str(t) for t in combined_tickers]
    df["has_ticker"] = [bool(t) for t in combined_tickers]

    df.to_csv(OUTPUT_CSV, index=False)

    # --- Coverage summary ---
    n_regex = sum(1 for t in regex_tickers if t)
    n_llm_only = sum(1 for r, l in zip(regex_tickers, llm_tickers) if not r and l)
    n_none = sum(1 for t in combined_tickers if not t)
    n_any = n - n_none
    pct = n_any / n * 100

    print()
    print("=" * 50)
    print("Ticker extraction coverage")
    print("=" * 50)
    print(f"  Total tweets        : {n:>7,}")
    print(f"  Regex hit           : {n_regex:>7,}  ({n_regex/n*100:.1f}%)")
    print(f"  LLM-only hit        : {n_llm_only:>7,}  ({n_llm_only/n*100:.1f}%)")
    print(f"  No ticker found     : {n_none:>7,}  ({n_none/n*100:.1f}%)")
    print(f"  >= 1 ticker (total) : {n_any:>7,}  ({pct:.1f}%)")
    print(f"\nOutput: {OUTPUT_CSV}")


if __name__ == "__main__":
    no_llm = "--no-llm" in sys.argv
    main(use_llm=not no_llm)
