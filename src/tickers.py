"""Ticker extraction for FinTwit tweets.

Two-stage pipeline:
  1. Regex baseline  — finds $TICKER cashtags directly in the tweet text.
  2. LLM fallback    — called ONLY when regex finds nothing; uses gpt-4o-mini to
                       resolve company name mentions to ticker symbols.

The LLM results are cached on disk (cache/llm_tickers/) keyed by SHA-256 of the
tweet text so re-runs are free and the pipeline is fully resumable.

Caller receives:
  tickers_mentioned : list[str]  — deduplicated, upper-cased tickers (may be [])
  has_ticker        : bool       — True iff tickers_mentioned is non-empty
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import time
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Regex configuration
# ---------------------------------------------------------------------------

# Cashtag pattern: $ followed by 1–5 ASCII letters (word boundary after)
_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})(?![A-Za-z])")

# Tokens that match the pattern but are not stock tickers.
# Currencies, common abbreviations, and other false positives.
_STOPLIST: frozenset[str] = frozenset({
    # Fiat currencies
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "NZD",
    "SEK", "NOK", "DKK", "HKD", "SGD", "MXN", "BRL", "INR", "KRW",
    # Common non-ticker fragments seen in finance tweets
    "YOY", "QOQ", "IPO", "ETF", "NFT", "CEO", "CFO", "CTO", "COO",
    "US", "UK", "EU", "AI",
})

# ---------------------------------------------------------------------------
# LLM cache configuration
# ---------------------------------------------------------------------------

CACHE_DIR = pathlib.Path(__file__).parent.parent / "cache" / "llm_tickers"

_LLM_SYSTEM_PROMPT = (
    "You are a financial ticker extraction assistant. "
    "Given a tweet, identify every stock ticker that the tweet is making a directional "
    "prediction about. "
    "Rules:\n"
    "- Return strict JSON: {\"tickers\": [\"TICKER1\", \"TICKER2\"]}\n"
    "- Resolve unambiguous company names to their primary US exchange ticker "
    "  (e.g. 'Tesla' -> 'TSLA', 'Apple' -> 'AAPL').\n"
    "- Include cashtags already present in the text.\n"
    "- Return [] if the tweet makes no prediction about a specific stock.\n"
    "- Do NOT invent tickers. If you are uncertain, omit the ticker.\n"
    "- Tickers must be 1–5 uppercase letters only.\n"
    "- Output only the JSON object, nothing else."
)

_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0  # seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_regex(text: str) -> list[str]:
    """Return deduplicated, upper-cased cashtag tickers found by regex."""
    if not isinstance(text, str):
        return []
    raw = _CASHTAG_RE.findall(text)
    seen: dict[str, None] = {}
    result = []
    for t in raw:
        t_up = t.upper()
        if t_up not in _STOPLIST and t_up not in seen:
            seen[t_up] = None
            result.append(t_up)
    return result


def extract_llm(text: str, api_key: Optional[str] = None) -> list[str]:
    """Call gpt-4o-mini to extract tickers. Results are cached on disk.

    Returns [] if the API key is not available (graceful degradation).
    """
    if not isinstance(text, str) or not text.strip():
        return []

    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        return []

    cache_path = _cache_path(text)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8")).get("tickers", [])
        except (json.JSONDecodeError, OSError):
            pass

    from openai import OpenAI, RateLimitError, APIError

    client = OpenAI(api_key=key)
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            tickers = _normalise_list(data.get("tickers", []))
            _write_cache(cache_path, tickers)
            return tickers
        except json.JSONDecodeError:
            # Truncated or malformed JSON from the model — cache empty result and give up.
            _write_cache(cache_path, [])
            return []
        except (RateLimitError, APIError) as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BACKOFF_BASE ** (attempt + 1)
            time.sleep(wait)

    return []  # unreachable, but satisfies type checkers


def extract_tickers(text: str, api_key: Optional[str] = None) -> tuple[list[str], bool]:
    """Full two-stage extraction for a single tweet.

    Returns (tickers_mentioned, has_ticker).
    LLM fallback is invoked only when regex finds nothing.
    """
    regex_hits = extract_regex(text)
    if regex_hits:
        return regex_hits, True

    llm_hits = extract_llm(text, api_key=api_key)
    combined = _normalise_list(regex_hits + llm_hits)
    return combined, bool(combined)


def extract_batch(
    texts: list[str],
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> list[tuple[list[str], bool]]:
    """Extract tickers for a list of tweet texts."""
    results = []
    n = len(texts)
    for i, text in enumerate(texts):
        if verbose and i % 500 == 0:
            print(f"  tickers: {i}/{n}")
        results.append(extract_tickers(text, api_key=api_key))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_list(tickers: list) -> list[str]:
    """Upper-case, filter stoplist, deduplicate, preserve order."""
    seen: dict[str, None] = {}
    result = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        t_up = t.upper().strip()
        if (
            t_up
            and t_up not in _STOPLIST
            and t_up not in seen
            and re.fullmatch(r"[A-Z]{1,5}", t_up)
        ):
            seen[t_up] = None
            result.append(t_up)
    return result


def _cache_path(text: str) -> pathlib.Path:
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{digest}.json"


def _write_cache(path: pathlib.Path, tickers: list[str]) -> None:
    try:
        path.write_text(json.dumps({"tickers": tickers}), encoding="utf-8")
    except OSError:
        pass
