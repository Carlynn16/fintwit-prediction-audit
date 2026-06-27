"""Evaluate ticker extraction precision, recall, and F1 against hand-labels.

Usage:
    python scripts/eval_tickers.py

Reads data/ticker_eval_sample.csv — the `true_tickers` column must be filled
with the gold-standard ticker lists before running (format: Python list literal,
e.g. ['AAPL', 'MSFT'] or []).

Prints a comparison table: regex-only vs combined (regex + LLM).
"""

from __future__ import annotations

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd

EVAL_CSV = pathlib.Path(__file__).parent.parent / "data" / "ticker_eval_sample.csv"


def _parse_list(val: object) -> set[str]:
    if pd.isna(val) or str(val).strip() in ("", "[]"):
        return set()
    try:
        parsed = ast.literal_eval(str(val))
        return {str(t).upper().strip() for t in parsed if t}
    except (ValueError, SyntaxError):
        return set()


def _prf(pred_sets: list[set[str]], true_sets: list[set[str]]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for pred, true in zip(pred_sets, true_sets):
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def main() -> None:
    if not EVAL_CSV.exists():
        print(f"ERROR: {EVAL_CSV} not found. Run scripts/extract_tickers.py first.")
        sys.exit(1)

    df = pd.read_csv(EVAL_CSV)

    required = {"regex_tickers", "llm_tickers", "combined_tickers", "true_tickers"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(1)

    unlabelled = df["true_tickers"].isna() | (df["true_tickers"].astype(str).str.strip() == "")
    if unlabelled.all():
        print("true_tickers column is entirely empty — fill it in before running eval.")
        sys.exit(1)

    # Drop rows without labels
    df = df[~unlabelled].copy()
    print(f"Evaluating on {len(df)} labelled rows (of {len(pd.read_csv(EVAL_CSV))} total).")

    true_sets = [_parse_list(v) for v in df["true_tickers"]]
    regex_sets = [_parse_list(v) for v in df["regex_tickers"]]
    combined_sets = [_parse_list(v) for v in df["combined_tickers"]]

    r_p, r_r, r_f = _prf(regex_sets, true_sets)
    c_p, c_r, c_f = _prf(combined_sets, true_sets)

    print()
    print(f"{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Regex only':<20} {r_p:>10.3f} {r_r:>10.3f} {r_f:>10.3f}")
    print(f"{'Regex + LLM':<20} {c_p:>10.3f} {c_r:>10.3f} {c_f:>10.3f}")


if __name__ == "__main__":
    main()
