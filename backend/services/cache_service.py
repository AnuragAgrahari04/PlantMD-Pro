"""
Cache Service — Redis with in-memory fallback.
"""
import json
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

_memory_cache: dict = {}


async def get_cache(key: str) -> Optional[dict]:
    try:
        import redis.asyncio as aioredis
        from core.config import settings
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        val = await r.get(key)
        await r.aclose()
        if val:
            parsed = json.loads(val)
            # Validate cached result has required fields before returning
            if parsed.get("display_name") and parsed.get("class_key"):
                return parsed
            # Stale/invalid cache entry — delete it
            await delete_cache(key)
            return None
    except Exception:
        val = _memory_cache.get(key)
        if val and val.get("display_name") and val.get("class_key"):
            return val
        _memory_cache.pop(key, None)
        return None
    return None


async def set_cache(key: str, value: dict, ttl: int = 3600) -> None:
    try:
        import redis.asyncio as aioredis
        from core.config import settings
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        await r.setex(key, ttl, json.dumps(value))
        await r.aclose()
    except Exception:
        _memory_cache[key] = value
        if len(_memory_cache) > 500:
            oldest_key = next(iter(_memory_cache))
            del _memory_cache[oldest_key]


async def delete_cache(key: str) -> None:
    try:
        import redis.asyncio as aioredis
        from core.config import settings
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        await r.delete(key)
        await r.aclose()
    except Exception:
        _memory_cache.pop(key, None)


async def clear_all_cache() -> None:
    """Clear all cached predictions — useful after model updates."""
    global _memory_cache
    _memory_cache = {}
    try:
        import redis.asyncio as aioredis
        from core.config import settings
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        await r.flushdb()
        await r.aclose()
        logger.info("Redis cache cleared")
    except Exception:
        logger.info("Memory cache cleared")