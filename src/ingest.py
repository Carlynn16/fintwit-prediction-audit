"""Build tweets_raw.csv from the validated_predictions.csv source export.

DATA-QUALITY NOTE — ID precision:
  The source export stored tweet_id, conversation_id, and reply_to_tweet_id as
  float64 (scientific notation, e.g. 1.8924e+18).  Converting float→string loses
  the least-significant digits, so exact tweet IDs may have lost precision.  These
  columns are kept as opaque string identifiers only; do not use them as exact join
  keys across external datasets.

PSEUDONYMISATION:
  Every unique author handle is mapped to account_01 … account_NN, assigned by
  alphabetical sort order.  The same mapping is applied to reply_to_user where the
  target is a known author; non-author reply targets are left unchanged.  The
  mapping file is written to data/account_mapping.csv (gitignored).  tweets_raw.csv
  contains only pseudonyms — never real handles.
"""

from __future__ import annotations

import pathlib
import pandas as pd

SOURCE_PATH = pathlib.Path(__file__).parent.parent / "OLD project" / "validated_predictions.csv"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RAW_OUTPUT = DATA_DIR / "tweets_raw.csv"
MAPPING_OUTPUT = DATA_DIR / "account_mapping.csv"

RAW_COLUMNS = [
    "tweet_id",
    "conversation_id",
    "tweet_type",
    "author",
    "text",
    "created_at",
    "created_date",
    "reply_to_tweet_id",
    "reply_to_user",
    "likes",
    "retweets",
    "replies_count",
    "views",
    "author_followers",
    "author_following",
    "author_verified",
    "author_blue_verified",
]


def _build_pseudonym_map(authors: pd.Series) -> dict[str, str]:
    """Map each unique author handle → account_NN (alphabetical assignment)."""
    unique = sorted(authors.dropna().unique())
    return {handle: f"account_{i+1:02d}" for i, handle in enumerate(unique)}


def build_raw(source: pathlib.Path = SOURCE_PATH) -> pd.DataFrame:
    """Load source CSV, keep raw columns, clean types, pseudonymise, and write output."""
    df = pd.read_csv(source, dtype=str)  # read everything as str first for safety

    # Keep only raw columns
    df = df[RAW_COLUMNS].copy()

    # --- ID columns: store as str (already str from read_csv)
    # The source stored these as float; precision may be lost — treat as opaque.
    for id_col in ("tweet_id", "conversation_id", "reply_to_tweet_id"):
        # Strip trailing '.0' that pandas would have written for float values
        df[id_col] = (
            df[id_col]
            .str.rstrip("0")
            .str.rstrip(".")
            .where(df[id_col].notna() & (df[id_col] != "nan"), other=pd.NA)
        )

    # --- Datetime columns
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce").dt.date

    # --- Numeric engagement / account columns → Int64 (nullable integer)
    int_cols = [
        "likes", "retweets", "replies_count", "views",
        "author_followers", "author_following",
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # --- Boolean columns
    for col in ("author_verified", "author_blue_verified"):
        df[col] = df[col].map({"True": True, "False": False, "true": True, "false": False})
        df[col] = df[col].astype("boolean")

    # --- Pseudonymisation
    pseudonym_map = _build_pseudonym_map(df["author"])
    df["author"] = df["author"].map(pseudonym_map)
    df["reply_to_user"] = df["reply_to_user"].apply(
        lambda v: pseudonym_map.get(v, v) if pd.notna(v) and v != "nan" else pd.NA
    )

    # Write mapping (gitignored)
    DATA_DIR.mkdir(exist_ok=True)
    mapping_df = pd.DataFrame(
        list(pseudonym_map.items()), columns=["handle", "pseudonym"]
    ).sort_values("pseudonym")
    mapping_df.to_csv(MAPPING_OUTPUT, index=False)

    # Write tweets_raw.csv
    df.to_csv(RAW_OUTPUT, index=False)
    return df


if __name__ == "__main__":
    df = build_raw()
    print(f"tweets_raw.csv written: {len(df):,} rows, {df['author'].nunique()} pseudonymised accounts")
