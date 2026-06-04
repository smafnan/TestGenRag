"""
Caching layer for generated test suites.

Uses Redis when REDIS_URL is set, otherwise falls back to a process-local
in-memory dict. The interface is identical either way, so the rest of the app
never needs to know which backend is active. Caching identical requirements
avoids re-running the (relatively expensive) agent pipeline.
"""

import hashlib
import json
import os
from typing import Optional

_TTL_SECONDS = int(os.getenv("CACHE_TTL", "3600"))

_redis = None
_memory: dict[str, str] = {}


def _client():
    """Lazily connect to Redis; return None if unavailable/unconfigured."""
    global _redis
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    if _redis is None:
        try:
            import redis  # redis-py
            _redis = redis.Redis.from_url(url, decode_responses=True)
            _redis.ping()
        except Exception as exc:  # connection refused, bad URL, etc.
            print(f"[WARN] Redis unavailable ({exc}); using in-memory cache.")
            _redis = False  # sentinel: tried and failed
    return _redis or None


def _key(provider: str, requirement: str) -> str:
    digest = hashlib.sha256(f"{provider}::{requirement}".encode()).hexdigest()[:32]
    return f"testgenrag:suite:{digest}"


def get_cached(provider: str, requirement: str) -> Optional[dict]:
    key = _key(provider, requirement)
    client = _client()
    raw = client.get(key) if client else _memory.get(key)
    return json.loads(raw) if raw else None


def set_cached(provider: str, requirement: str, payload: dict) -> None:
    key = _key(provider, requirement)
    raw = json.dumps(payload, ensure_ascii=False)
    client = _client()
    if client:
        client.setex(key, _TTL_SECONDS, raw)
    else:
        _memory[key] = raw


def backend_name() -> str:
    return "redis" if _client() else "memory"
