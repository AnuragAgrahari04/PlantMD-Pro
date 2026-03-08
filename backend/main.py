"""
PlantMD Pro - Industry-Level Plant Disease Prediction System
Main FastAPI Application Entry Point
"""
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.logging import setup_logging
from api.routes import predict, health, auth
from core.exceptions import PlantMDException

# Setup structured logging
setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info("Starting PlantMD Pro API", version=settings.APP_VERSION, env=settings.ENVIRONMENT)
    # Preload model on startup
    from ml.model import get_model
    get_model()
    logger.info("Model loaded and ready")
    yield
    logger.info("Shutting down PlantMD Pro API")


app = FastAPI(
    title="PlantMD Pro API",
    description="Industry-grade plant disease detection using deep learning",
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan,
)

# ── Middleware ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    """Attach request ID and log request timing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration_ms, 2),
        request_id=request_id,
    )
    return response


# ── Exception Handlers ──────────────────────────────────────────────────────
@app.exception_handler(PlantMDException)
async def plantmd_exception_handler(request: Request, exc: PlantMDException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "message": exc.message, "request_id": getattr(request.state, "request_id", None)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", error=str(exc), path=request.url.path, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
    )


# ── Routes ──────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])


@app.get("/", tags=["Root"])
async def root():
    return {"service": "PlantMD Pro", "version": settings.APP_VERSION, "status": "operational"}
