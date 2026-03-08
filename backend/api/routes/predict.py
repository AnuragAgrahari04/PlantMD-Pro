"""
Prediction Route — /api/v1/predict
"""
import base64
import time
from typing import Optional

import structlog
from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from core.config import settings
from core.exceptions import LowConfidenceError
from ml.model import get_model
from ml.preprocessor import create_heatmap_overlay, validate_and_load_image
from services.cache_service import get_cache, set_cache

logger = structlog.get_logger(__name__)
router = APIRouter()


class TreatmentPlan(BaseModel):
    immediate_action: str
    fungicide: str
    prevention: str


class TopPrediction(BaseModel):
    class_key: str
    display_name: str
    confidence: float


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    request_id: str
    display_name: str
    class_key: str
    confidence: float = Field(..., ge=0, le=1)
    is_healthy: bool
    severity: str
    affected_crop: str
    treatment: TreatmentPlan
    top3: list[TopPrediction]
    heatmap_base64: Optional[str] = None
    inference_ms: float
    model_version: str
    cached: bool = False
    demo_mode: bool = False


@router.post("/predict", response_model=PredictionResponse, summary="Predict plant disease from leaf image")
async def predict_disease(
    request: Request,
    file: UploadFile = File(...),
    generate_heatmap: bool = Form(False),
):
    request_id = getattr(request.state, "request_id", "unknown")
    t0 = time.perf_counter()

    file_bytes = await file.read()
    image_array, image_hash = validate_and_load_image(
        file_bytes, filename=file.filename or "upload.jpg", content_type=file.content_type or ""
    )

    # Cache check — but validate cached result has all required fields
    cache_key = f"predict:{image_hash}:{generate_heatmap}"
    cached = await get_cache(cache_key)
    if cached and cached.get("display_name") and cached.get("class_key"):
        cached["cached"] = True
        cached["request_id"] = request_id
        logger.info("Cache hit", hash=image_hash[:8])
        return PredictionResponse(**cached)

    # Run inference
    model = get_model()
    result = model.predict(image_array)

    # Confidence gate — skip for demo mode
    if not result.get("demo_mode") and result["confidence"] < settings.CONFIDENCE_THRESHOLD:
        raise LowConfidenceError(result["confidence"])

    # Optional Grad-CAM
    heatmap_b64 = None
    if generate_heatmap:
        cam = model.generate_gradcam(image_array)
        if cam is not None:
            heatmap_bytes = create_heatmap_overlay(image_array, cam)
            heatmap_b64 = base64.b64encode(heatmap_bytes).decode()

    inference_ms = (time.perf_counter() - t0) * 1000

    response_data = {
        "request_id": request_id,
        "display_name": result["display_name"],
        "class_key": result["class_key"],
        "confidence": result["confidence"],
        "is_healthy": result["is_healthy"],
        "severity": result["severity"],
        "affected_crop": result["affected_crop"],
        "treatment": result["treatment"],
        "top3": result.get("top3", []),
        "heatmap_base64": heatmap_b64,
        "inference_ms": round(inference_ms, 2),
        "model_version": settings.MODEL_VERSION,
        "cached": False,
        "demo_mode": result.get("demo_mode", False),
    }

    cacheable = {k: v for k, v in response_data.items() if k != "request_id"}
    await set_cache(cache_key, cacheable, ttl=settings.CACHE_TTL_SECONDS)

    logger.info("Prediction complete", disease=result["display_name"], confidence=result["confidence"], inference_ms=round(inference_ms, 2))
    return PredictionResponse(**response_data)


@router.get("/predict/classes", summary="List all supported disease classes")
async def list_classes():
    from ml.model import DISEASE_METADATA
    return {
        "total": len(DISEASE_METADATA),
        "classes": [
            {"key": k, "display_name": v["display_name"], "crop": v["affected_crop"]}
            for k, v in DISEASE_METADATA.items()
        ],
    }