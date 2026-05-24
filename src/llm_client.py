"""
Base LLM client wrapper with retry, rate-limit handling, and token counting.

Usage logging side-channel
--------------------------
If env var ``MASTE_USAGE_LOG`` is set to a file path, every successful
``chat_complete`` call appends one JSON line containing the model name and
prompt/completion/total tokens. Existing callers are unaffected when the
variable is unset.
"""

import os
import time
import json
import logging
import threading
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_USAGE_LOCK = threading.Lock()


def _log_usage(model: str, usage: Optional[object]) -> None:
    """Append one JSON line to the usage log file if MASTE_USAGE_LOG is set."""
    log_path = os.environ.get("MASTE_USAGE_LOG")
    if not log_path or usage is None:
        return
    record = {
        "ts": time.time(),
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "tag": os.environ.get("MASTE_USAGE_TAG", ""),
    }
    try:
        with _USAGE_LOCK:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover - best-effort logging
        logger.debug(f"Usage logging failed: {e}")


def get_client():
    """Return an OpenAI-compatible client with hard httpx-level timeouts.

    Uses httpx.Timeout to enforce transport-level timeouts, which prevents
    TCP-level hangs that SDK-level timeouts alone cannot catch.
    """
    try:
        import httpx
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Export it before running experiments."
            )
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        # Hard transport-level timeout: connect=10s, read=45s, write=10s, pool=5s
        http_client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)
        )
        kwargs = dict(api_key=api_key, http_client=http_client)
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)
    except ImportError as e:
        raise ImportError(f"Missing package: {e}. Run: pip install openai httpx")


_CLIENT = None


def get_cached_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = get_client()
    return _CLIENT


def reset_client():
    """Force-recreate the client (e.g. after a timeout to clear stale connections)."""
    global _CLIENT
    _CLIENT = None


def chat_complete(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 5,
    response_format: Optional[dict] = None,
    timeout: float = 60.0,
    min_tokens: int = 16,
) -> str:
    """
    Call the chat completion API with retries and a per-request timeout.
    Returns the text content of the first choice.
    """
    client = get_cached_client()
    wait = 2.0
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max(max_tokens, min_tokens),
        timeout=timeout,
    )
    if response_format:
        kwargs["response_format"] = response_format

    # OpenRouter routing preferences. When MASTE_PROVIDER_ORDER is set
    # (comma-separated list, e.g. "OpenAI" or "OpenAI,Together"), pin the
    # provider order so concurrent requests stay on a single backend and
    # do not get silently rerouted to a region-restricted provider.
    provider_order = os.environ.get("MASTE_PROVIDER_ORDER", "").strip()
    if provider_order:
        providers = [p.strip() for p in provider_order.split(",") if p.strip()]
        kwargs["extra_body"] = {
            "provider": {"order": providers, "allow_fallbacks": False}
        }

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            _log_usage(model, getattr(resp, "usage", None))
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                logger.warning(f"Rate limit hit, waiting {wait}s …")
                time.sleep(wait)
                wait = min(wait * 2, 60)
            elif any(k in err.lower() for k in ("timeout", "timed out", "read timeout", "connect timeout")):
                logger.warning(f"Request timed out (attempt {attempt+1}/{max_retries}), resetting client and retrying …")
                reset_client()   # recreate httpx client to clear stale connection
                time.sleep(5)
            elif attempt < max_retries - 1:
                logger.warning(f"API error ({err}), retry {attempt+1} …")
                time.sleep(2)
            else:
                raise
    raise RuntimeError("Max retries exceeded")
