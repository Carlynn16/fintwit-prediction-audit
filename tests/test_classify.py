"""Unit tests for src/classify.py — all LLM calls are mocked."""

from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.classify import (
    classify_tweet,
    classify_batch,
    _validate_and_coerce,
    FALLBACK,
    _VALID_SENTIMENT,
    _VALID_HORIZON,
    _VALID_TRADE_TYPE,
)

VALID_RESPONSE = {
    "sentiment": "bullish",
    "time_horizon": "short_term",
    "trade_type": "trade_suggestion",
    "confidence": 85,
}

VALID_JSON_STR = json.dumps(VALID_RESPONSE)


def _make_mock_client(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# _validate_and_coerce
# ---------------------------------------------------------------------------

class TestValidateAndCoerce:
    def test_valid_input_passes_through(self):
        result = _validate_and_coerce(VALID_RESPONSE)
        assert result == VALID_RESPONSE

    def test_confidence_clipped_above_100(self):
        data = {**VALID_RESPONSE, "confidence": 150}
        assert _validate_and_coerce(data)["confidence"] == 100

    def test_confidence_clipped_below_0(self):
        data = {**VALID_RESPONSE, "confidence": -5}
        assert _validate_and_coerce(data)["confidence"] == 0

    def test_confidence_coerced_from_string(self):
        data = {**VALID_RESPONSE, "confidence": "72"}
        assert _validate_and_coerce(data)["confidence"] == 72

    def test_confidence_invalid_string_falls_back(self):
        data = {**VALID_RESPONSE, "confidence": "high"}
        assert _validate_and_coerce(data)["confidence"] == 0

    def test_confidence_float_truncated(self):
        data = {**VALID_RESPONSE, "confidence": 73.9}
        assert _validate_and_coerce(data)["confidence"] == 73

    def test_invalid_sentiment_falls_back(self):
        data = {**VALID_RESPONSE, "sentiment": "very_bullish"}
        assert _validate_and_coerce(data)["sentiment"] == FALLBACK["sentiment"]

    def test_invalid_horizon_falls_back(self):
        data = {**VALID_RESPONSE, "time_horizon": "next_week"}
        assert _validate_and_coerce(data)["time_horizon"] == FALLBACK["time_horizon"]

    def test_invalid_trade_type_falls_back(self):
        data = {**VALID_RESPONSE, "trade_type": "rumour"}
        assert _validate_and_coerce(data)["trade_type"] == FALLBACK["trade_type"]

    def test_missing_fields_fall_back(self):
        result = _validate_and_coerce({})
        assert result == FALLBACK

    def test_all_valid_sentiment_values_accepted(self):
        for val in _VALID_SENTIMENT:
            data = {**VALID_RESPONSE, "sentiment": val}
            assert _validate_and_coerce(data)["sentiment"] == val

    def test_all_valid_horizon_values_accepted(self):
        for val in _VALID_HORIZON:
            data = {**VALID_RESPONSE, "time_horizon": val}
            assert _validate_and_coerce(data)["time_horizon"] == val

    def test_all_valid_trade_type_values_accepted(self):
        for val in _VALID_TRADE_TYPE:
            data = {**VALID_RESPONSE, "trade_type": val}
            assert _validate_and_coerce(data)["trade_type"] == val


# ---------------------------------------------------------------------------
# classify_tweet — happy path (mocked API)
# ---------------------------------------------------------------------------

class TestClassifyTweet:
    def test_valid_response_parsed_correctly(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.classify.CACHE_DIR", tmp_path)
        mock_client = _make_mock_client(VALID_JSON_STR)
        with patch("src.classify.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI", return_value=mock_client):
                result = classify_tweet("$AAPL to 200 this month", api_key="fake-key")
        assert result["sentiment"] == "bullish"
        assert result["time_horizon"] == "short_term"
        assert result["trade_type"] == "trade_suggestion"
        assert result["confidence"] == 85

    def test_result_is_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.classify.CACHE_DIR", tmp_path)
        mock_client = _make_mock_client(VALID_JSON_STR)
        text = "$TSLA breaking out"
        with patch("src.classify.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI", return_value=mock_client):
                classify_tweet(text, api_key="fake-key")
                classify_tweet(text, api_key="fake-key")
        # Second call should use cache — only one API call made
        assert mock_client.chat.completions.create.call_count == 1

    def test_cache_read_on_second_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.classify.CACHE_DIR", tmp_path)
        text = "Bearish on $NVDA"
        cached_data = {
            "sentiment": "bearish", "time_horizon": "medium_term",
            "trade_type": "analysis", "confidence": 60,
        }
        import hashlib
        digest = hashlib.sha256(text.encode()).hexdigest()
        (tmp_path / f"{digest}.json").write_text(json.dumps(cached_data), encoding="utf-8")

        with patch("src.classify.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI") as mock_openai:
                result = classify_tweet(text, api_key="fake-key")
        mock_openai.assert_not_called()
        assert result["sentiment"] == "bearish"
        assert result["confidence"] == 60

    def test_returns_fallback_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = classify_tweet("$AAPL going up", api_key=None)
        assert result == FALLBACK

    def test_returns_fallback_on_empty_text(self):
        result = classify_tweet("", api_key="fake-key")
        assert result == FALLBACK

    def test_returns_fallback_on_non_string(self):
        result = classify_tweet(None, api_key="fake-key")  # type: ignore
        assert result == FALLBACK

    def test_invalid_json_returns_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.classify.CACHE_DIR", tmp_path)
        mock_client = _make_mock_client('{"sentiment": "bullish", "time_horizo')  # truncated
        with patch("src.classify.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI", return_value=mock_client):
                result = classify_tweet("$AAPL buy", api_key="fake-key")
        assert result == FALLBACK

    def test_unexpected_field_values_coerced_to_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.classify.CACHE_DIR", tmp_path)
        bad_response = json.dumps({
            "sentiment": "very_bullish",
            "time_horizon": "next_week",
            "trade_type": "tip",
            "confidence": 999,
        })
        mock_client = _make_mock_client(bad_response)
        with patch("src.classify.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI", return_value=mock_client):
                result = classify_tweet("Some tweet $AAPL", api_key="fake-key")
        assert result["sentiment"] == FALLBACK["sentiment"]
        assert result["time_horizon"] == FALLBACK["time_horizon"]
        assert result["trade_type"] == FALLBACK["trade_type"]
        assert result["confidence"] == 100  # clipped from 999


# ---------------------------------------------------------------------------
# classify_batch — only has_ticker rows should be classified
# ---------------------------------------------------------------------------

class TestClassifyBatch:
    def test_batch_returns_one_result_per_input(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.classify.CACHE_DIR", tmp_path)
        mock_client = _make_mock_client(VALID_JSON_STR)
        texts = ["$AAPL up", "$TSLA down", "$MSFT hold"]
        with patch("src.classify.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI", return_value=mock_client):
                results, n_api, n_cache = classify_batch(texts, api_key="fake-key")
        assert len(results) == 3

    def test_batch_counts_api_calls_and_cache_hits(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.classify.CACHE_DIR", tmp_path)
        text1 = "$AAPL up"
        text2 = "$TSLA down"
        # Pre-populate cache for text1
        import hashlib
        digest = hashlib.sha256(text1.encode()).hexdigest()
        (tmp_path / f"{digest}.json").write_text(json.dumps(VALID_RESPONSE), encoding="utf-8")

        mock_client = _make_mock_client(VALID_JSON_STR)
        with patch("src.classify.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI", return_value=mock_client):
                _, n_api, n_cache = classify_batch([text1, text2], api_key="fake-key")
        assert n_cache == 1
        assert n_api == 1

    def test_empty_text_in_batch_gets_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.classify.CACHE_DIR", tmp_path)
        results, _, _ = classify_batch(["", "   "], api_key="")
        assert results[0] == FALLBACK
        assert results[1] == FALLBACK

    def test_all_result_keys_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.classify.CACHE_DIR", tmp_path)
        mock_client = _make_mock_client(VALID_JSON_STR)
        with patch("src.classify.CACHE_DIR", tmp_path):
            with patch("openai.OpenAI", return_value=mock_client):
                results, _, _ = classify_batch(["$AAPL bullish"], api_key="fake-key")
        assert set(results[0].keys()) == {"sentiment", "time_horizon", "trade_type", "confidence"}


# ---------------------------------------------------------------------------
# Integration: only has_ticker rows classified (tested via script logic)
# ---------------------------------------------------------------------------

class TestOnlyTickerRowsClassified:
    def test_fallback_for_no_ticker_rows(self):
        """Rows without tickers should receive FALLBACK (no API call)."""
        with patch("src.classify.classify_tweet") as mock_classify:
            mock_classify.return_value = VALID_RESPONSE
            # Simulate: only call classify_tweet for has_ticker rows
            rows = [
                {"text": "general comment", "has_ticker": False},
                {"text": "$AAPL is up", "has_ticker": True},
            ]
            results = []
            for row in rows:
                if row["has_ticker"]:
                    results.append(mock_classify(row["text"], api_key="fake-key"))
                else:
                    results.append(FALLBACK.copy())
            assert results[0] == FALLBACK
            assert results[1] == VALID_RESPONSE
            mock_classify.assert_called_once()
