"""Health check endpoints — used by Kubernetes liveness/readiness probes."""
import time

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from core.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter()

_start_time = time.time()


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # fixes Pydantic model_ warning

    status: str
    version: str
    environment: str
    uptime_seconds: float
    model_loaded: bool


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Liveness probe — is the service alive?"""
    from ml.model import get_model
    try:
        model = get_model()
        model_loaded = model._is_loaded
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        uptime_seconds=round(time.time() - _start_time, 1),
        model_loaded=model_loaded,
    )


@router.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe — is the service ready to accept traffic?"""
    from ml.model import get_model
    model = get_model()
    if not model._is_loaded:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}


@router.get("/health/live", tags=["Health"])
async def liveness_check():
    """Simple liveness check."""
    return {"status": "alive"}