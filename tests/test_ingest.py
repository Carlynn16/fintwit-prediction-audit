"""Tests for src/ingest.py."""

import pathlib
import pandas as pd
import pytest

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RAW_CSV = DATA_DIR / "tweets_raw.csv"
MAPPING_CSV = DATA_DIR / "account_mapping.csv"

EXPECTED_COLUMNS = [
    "tweet_id", "conversation_id", "tweet_type", "author", "text",
    "created_at", "created_date", "reply_to_tweet_id", "reply_to_user",
    "likes", "retweets", "replies_count", "views",
    "author_followers", "author_following",
    "author_verified", "author_blue_verified",
]

# Real handles from the source (sampled — used to verify no leakage)
KNOWN_REAL_HANDLES = [
    "Jake__Wujastyk", "ScorpionFund", "alphatrends", "CompoundinGirl",
    "kintsugiinvest", "Errecck", "hftquant_", "TheLongInvest",
    "dampedspring", "Mr_Derivatives",
]


@pytest.fixture(scope="module")
def df():
    if not RAW_CSV.exists():
        pytest.skip("tweets_raw.csv not yet generated — run src/ingest.py first")
    return pd.read_csv(RAW_CSV, dtype={"tweet_id": str, "conversation_id": str,
                                       "reply_to_tweet_id": str})


@pytest.fixture(scope="module")
def mapping():
    if not MAPPING_CSV.exists():
        pytest.skip("account_mapping.csv not yet generated")
    return pd.read_csv(MAPPING_CSV)


class TestColumns:
    def test_expected_columns_present(self, df):
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_no_derived_columns(self, df):
        forbidden = {
            "time_horizon", "trade_type", "sentiment", "confidence",
            "tickers_mentioned", "stocks", "has_ticker", "prediction_date",
            "price_change_pct", "prediction_correct", "actual_return",
            "validated_ticker", "company_names",
        }
        assert not forbidden.intersection(df.columns)


class TestRowCount:
    def test_row_count(self, df):
        assert len(df) == 18071


class TestPseudonymisation:
    def test_no_real_handles_in_author(self, df):
        for handle in KNOWN_REAL_HANDLES:
            assert handle not in df["author"].values, f"Real handle found in author: {handle}"

    def test_no_real_handles_in_reply_to_user(self, df):
        non_null = df["reply_to_user"].dropna()
        for handle in KNOWN_REAL_HANDLES:
            assert handle not in non_null.values, f"Real handle found in reply_to_user: {handle}"

    def test_author_pseudonym_format(self, df):
        pattern = r"^account_\d{2}$"
        assert df["author"].str.match(pattern).all()

    def test_pseudonym_count_matches_unique_authors(self, df, mapping):
        assert df["author"].nunique() == len(mapping)
        assert df["author"].nunique() == 51

    def test_mapping_pseudonyms_sequential(self, mapping):
        pseudonyms = sorted(mapping["pseudonym"].tolist())
        expected = [f"account_{i+1:02d}" for i in range(len(pseudonyms))]
        assert pseudonyms == expected


class TestDtypes:
    def test_created_at_is_datetime(self, df):
        col = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        assert col.notna().sum() > 0
        # pandas ≥2.0 uses microsecond resolution (us); older uses ns — accept both
        assert "UTC" in str(col.dtype)

    def test_created_date_parseable(self, df):
        col = pd.to_datetime(df["created_date"], errors="coerce")
        assert col.notna().sum() > 0

    def test_numeric_columns_castable_to_int(self, df):
        int_cols = ["likes", "retweets", "replies_count", "views",
                    "author_followers", "author_following"]
        for col in int_cols:
            numeric = pd.to_numeric(df[col], errors="coerce")
            # At least some values should be numeric (not all NaN)
            assert numeric.notna().sum() > 0, f"{col} has no numeric values"

    def test_boolean_columns_castable(self, df):
        for col in ("author_verified", "author_blue_verified"):
            vals = df[col].dropna().unique()
            assert set(vals).issubset({"True", "False", True, False, "true", "false"})


class TestDataIntegrity:
    def test_tweet_type_values(self, df):
        # Source uses "parent" for original posts, "reply" for replies
        assert df["tweet_type"].isin(["parent", "reply"]).all()

    def test_text_not_all_null(self, df):
        assert df["text"].notna().sum() > 0

    def test_date_range(self, df):
        dates = pd.to_datetime(df["created_date"], errors="coerce")
        assert dates.min() >= pd.Timestamp("2021-01-01")
        assert dates.max() <= pd.Timestamp("2025-03-01")
