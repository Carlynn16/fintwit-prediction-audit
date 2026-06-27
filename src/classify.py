"""LLM classification of tweet predictions.

For each ticker-bearing tweet, calls gpt-4o-mini to produce:
  sentiment     : "bullish" | "bearish" | "neutral"
  time_horizon  : "short_term" | "medium_term" | "long_term" | "unknown"
  trade_type    : "trade_suggestion" | "analysis" | "news" | "general_discussion"
  confidence    : int 0-100  (how confident the tweet *sounds* — not a calibrated
                               probability; calibration is assessed separately in Phase 6)

Only tweets with has_ticker == True are classified; others receive null/NA.

Results are cached on disk (cache/llm_classify/) keyed by SHA-256 of the tweet text
so re-runs cost zero and the pipeline is fully resumable after interruption.

On any failure (API error, malformed JSON, unexpected field values) the row falls back
to safe defaults and the error is logged — the pipeline never crashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import time
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CACHE_DIR = pathlib.Path(__file__).parent.parent / "cache" / "llm_classify"

# ---------------------------------------------------------------------------
# Valid field values
# ---------------------------------------------------------------------------

_VALID_SENTIMENT = {"bullish", "bearish", "neutral"}
_VALID_HORIZON = {"short_term", "medium_term", "long_term", "unknown"}
_VALID_TRADE_TYPE = {"trade_suggestion", "analysis", "news", "general_discussion"}

# ---------------------------------------------------------------------------
# Defaults (used on any failure)
# ---------------------------------------------------------------------------

FALLBACK: dict = {
    "sentiment": "neutral",
    "time_horizon": "unknown",
    "trade_type": "general_discussion",
    "confidence": 0,
}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a financial prediction classifier. "
    "Given a tweet from a stock-market influencer, classify it and return ONLY a JSON "
    "object with exactly these four fields:\n\n"
    "  \"sentiment\"    : the tweet's directional view on the stock(s) mentioned. "
    "One of: \"bullish\" (positive/long bias), \"bearish\" (negative/short bias), "
    "\"neutral\" (no clear directional view).\n\n"
    "  \"time_horizon\" : the implied prediction horizon. One of: "
    "\"short_term\" (days to ~1 month), \"medium_term\" (~1-6 months), "
    "\"long_term\" (>6 months), \"unknown\" (no horizon implied).\n\n"
    "  \"trade_type\"   : the nature of the tweet. One of: "
    "\"trade_suggestion\" (explicit buy/sell recommendation or entry/exit point), "
    "\"analysis\" (reasoned opinion or chart reading, no explicit trade call), "
    "\"news\" (relaying factual information or earnings), "
    "\"general_discussion\" (everything else).\n\n"
    "  \"confidence\"   : an integer 0-100 representing how confident or assertive "
    "the tweet sounds (0 = very tentative, 100 = extremely certain).\n\n"
    "Rules:\n"
    "- Output ONLY the JSON object, no extra text.\n"
    "- Every field is required.\n"
    "- Use only the exact string values listed above for the categorical fields.\n"
    "- confidence must be an integer between 0 and 100 inclusive."
)

_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_tweet(text: str, api_key: Optional[str] = None) -> dict:
    """Return classification dict for a single tweet text.

    Returns FALLBACK (never raises) on any failure.
    """
    if not isinstance(text, str) or not text.strip():
        return FALLBACK.copy()

    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        return FALLBACK.copy()

    cache_path = _cache_path(text)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return _validate_and_coerce(cached)
        except (json.JSONDecodeError, OSError):
            pass

    from openai import OpenAI, RateLimitError, APIError

    client = OpenAI(api_key=key)
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=128,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            result = _validate_and_coerce(data)
            _write_cache(cache_path, result)
            return result
        except json.JSONDecodeError:
            log.warning("JSONDecodeError for text: %.80s", text)
            _write_cache(cache_path, FALLBACK)
            return FALLBACK.copy()
        except (RateLimitError, APIError) as exc:
            if attempt == _MAX_RETRIES - 1:
                log.error("API error after %d retries: %s", _MAX_RETRIES, exc)
                return FALLBACK.copy()
            wait = _BACKOFF_BASE ** (attempt + 1)
            log.warning("API error (attempt %d), retrying in %.1fs: %s", attempt + 1, wait, exc)
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected error classifying tweet: %s", exc)
            return FALLBACK.copy()

    return FALLBACK.copy()


def classify_batch(
    texts: list[str],
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> tuple[list[dict], int, int]:
    """Classify a list of tweet texts.

    Returns (results, n_api_calls, n_cache_hits).
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    n_api = 0
    n_cache = 0
    results = []

    for i, text in enumerate(texts):
        if verbose and i % 500 == 0:
            print(f"  classify: {i}/{len(texts)}  (api={n_api}, cache={n_cache})")

        if not isinstance(text, str) or not text.strip():
            results.append(FALLBACK.copy())
            continue

        cache_path = _cache_path(text)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                results.append(_validate_and_coerce(cached))
                n_cache += 1
                continue
            except (json.JSONDecodeError, OSError):
                pass

        result = classify_tweet(text, api_key=key)
        results.append(result)
        n_api += 1

    return results, n_api, n_cache


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_and_coerce(data: dict) -> dict:
    """Return a clean classification dict, falling back field-by-field."""
    sentiment = data.get("sentiment", "")
    if sentiment not in _VALID_SENTIMENT:
        sentiment = FALLBACK["sentiment"]

    horizon = data.get("time_horizon", "")
    if horizon not in _VALID_HORIZON:
        horizon = FALLBACK["time_horizon"]

    trade_type = data.get("trade_type", "")
    if trade_type not in _VALID_TRADE_TYPE:
        trade_type = FALLBACK["trade_type"]

    try:
        confidence = max(0, min(100, int(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0

    return {
        "sentiment": sentiment,
        "time_horizon": horizon,
        "trade_type": trade_type,
        "confidence": confidence,
    }


def _cache_path(text: str) -> pathlib.Path:
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{digest}.json"


def _write_cache(path: pathlib.Path, result: dict) -> None:
    try:
        path.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass
