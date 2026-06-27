"""Unit tests for src/tickers.py — all LLM calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.tickers import (
    extract_regex,
    extract_llm,
    extract_tickers,
    _normalise_list,
    _STOPLIST,
)


# ---------------------------------------------------------------------------
# Regex extraction
# ---------------------------------------------------------------------------

class TestExtractRegex:
    def test_single_cashtag(self):
        assert extract_regex("Bullish on $AAPL today") == ["AAPL"]

    def test_multiple_cashtags(self):
        result = extract_regex("$TSLA and $NVDA looking strong")
        assert result == ["TSLA", "NVDA"]

    def test_lowercase_cashtag_normalised(self):
        assert extract_regex("long $tsla here") == ["TSLA"]

    def test_mixed_case_normalised(self):
        assert extract_regex("$Aapl is a buy") == ["AAPL"]

    def test_dollar_number_excluded(self):
        assert extract_regex("Stock up $100 today") == []

    def test_dollar_number_mixed(self):
        # $100 should be excluded; $AAPL should be kept
        result = extract_regex("$AAPL gained $50 today")
        assert result == ["AAPL"]

    def test_stoplist_usd_excluded(self):
        assert extract_regex("Paid in $USD") == []

    def test_stoplist_eur_excluded(self):
        assert extract_regex("$EUR strengthening") == []

    def test_all_stoplist_entries_excluded(self):
        for token in _STOPLIST:
            result = extract_regex(f"Trading ${token} pairs")
            assert token not in result, f"Stoplist token {token} leaked through"

    def test_deduplication(self):
        result = extract_regex("$AAPL is great, $AAPL will moon")
        assert result == ["AAPL"]
        assert len(result) == 1

    def test_no_ticker_tweet(self):
        assert extract_regex("Good morning everyone! Markets looking interesting.") == []

    def test_five_letter_ticker(self):
        assert extract_regex("$GOOGL breaking out") == ["GOOGL"]

    def test_six_letter_not_matched(self):
        # 6-letter strings should NOT be matched as tickers
        result = extract_regex("$TOOLONG is not a ticker")
        assert result == []

    def test_cashtag_at_start_of_text(self):
        assert extract_regex("$TSLA to the moon!") == ["TSLA"]

    def test_empty_string(self):
        assert extract_regex("") == []

    def test_non_string_input(self):
        assert extract_regex(None) == []  # type: ignore
        assert extract_regex(42) == []  # type: ignore

    def test_ticker_followed_by_letters_not_matched(self):
        # $AAPLisgreat should NOT match because letters follow immediately
        result = extract_regex("Check $AAPLisgreat out")
        assert result == []

    def test_order_preserved(self):
        result = extract_regex("$MSFT then $AAPL then $NVDA")
        assert result == ["MSFT", "AAPL", "NVDA"]


# ---------------------------------------------------------------------------
# Normalise list helper
# ---------------------------------------------------------------------------

class TestNormaliseList:
    def test_upper_cases(self):
        assert _normalise_list(["tsla", "aapl"]) == ["TSLA", "AAPL"]

    def test_deduplicates(self):
        assert _normalise_list(["AAPL", "aapl", "AAPL"]) == ["AAPL"]

    def test_filters_stoplist(self):
        assert _normalise_list(["USD", "AAPL", "EUR"]) == ["AAPL"]

    def test_filters_non_alpha(self):
        assert _normalise_list(["AA11", "AAPL"]) == ["AAPL"]

    def test_filters_too_long(self):
        assert _normalise_list(["TOOLONG", "AAPL"]) == ["AAPL"]

    def test_empty_list(self):
        assert _normalise_list([]) == []

    def test_ignores_non_string_entries(self):
        assert _normalise_list([None, 123, "AAPL"]) == ["AAPL"]  # type: ignore


# ---------------------------------------------------------------------------
# LLM extraction (mocked — no network)
# ---------------------------------------------------------------------------

class TestExtractLLM:
    def test_returns_empty_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = extract_llm("Tesla is going up", api_key=None)
        assert result == []

    def test_returns_cached_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tickers.CACHE_DIR", tmp_path)
        from src.tickers import _cache_path, _write_cache
        text = "Tesla is going to the moon"
        cache = _cache_path.__wrapped__(text) if hasattr(_cache_path, "__wrapped__") else None
        # Write a cache entry manually
        import json, hashlib
        digest = hashlib.sha256(text.encode()).hexdigest()
        cache_file = tmp_path / f"{digest}.json"
        cache_file.write_text(json.dumps({"tickers": ["TSLA"]}), encoding="utf-8")
        # Now extract_llm should return from cache without calling the API
        with patch("src.tickers.CACHE_DIR", tmp_path):
            result = extract_llm(text, api_key="fake-key")
        assert result == ["TSLA"]

    def test_calls_openai_and_caches(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tickers.CACHE_DIR", tmp_path)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"tickers": ["NVDA"]}'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("src.tickers.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI", return_value=mock_client):
                result = extract_llm("Nvidia is breaking out", api_key="fake-key")

        assert result == ["NVDA"]
        mock_client.chat.completions.create.assert_called_once()

    def test_llm_empty_response_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tickers.CACHE_DIR", tmp_path)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"tickers": []}'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("src.tickers.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI", return_value=mock_client):
                result = extract_llm("Good morning everyone!", api_key="fake-key")

        assert result == []


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------

class TestExtractTickers:
    def test_regex_hit_no_llm_call(self):
        with patch("src.tickers.extract_llm") as mock_llm:
            tickers, has = extract_tickers("$AAPL is bullish")
        mock_llm.assert_not_called()
        assert tickers == ["AAPL"]
        assert has is True

    def test_no_cashtag_triggers_llm(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tickers.CACHE_DIR", tmp_path)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"tickers": ["TSLA"]}'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("src.tickers.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI", return_value=mock_client):
                tickers, has = extract_tickers("Tesla is going to the moon", api_key="fake-key")

        assert tickers == ["TSLA"]
        assert has is True

    def test_has_ticker_false_when_empty(self):
        with patch("src.tickers.extract_llm", return_value=[]):
            tickers, has = extract_tickers("Good morning!", api_key="fake-key")
        assert tickers == []
        assert has is False

    def test_has_ticker_true_iff_nonempty(self):
        with patch("src.tickers.extract_llm", return_value=[]):
            tickers1, has1 = extract_tickers("Nothing here", api_key="fake-key")
            tickers2, has2 = extract_tickers("$MSFT up 3%")
        assert has1 == (len(tickers1) > 0)
        assert has2 == (len(tickers2) > 0)

    def test_dedup_across_regex_and_llm(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tickers.CACHE_DIR", tmp_path)
        # Regex finds AAPL; if LLM were called it would also return AAPL — but LLM
        # should NOT be called when regex already found something.
        with patch("src.tickers.extract_llm") as mock_llm:
            tickers, has = extract_tickers("$AAPL $AAPL double mention")
        mock_llm.assert_not_called()
        assert tickers.count("AAPL") == 1
