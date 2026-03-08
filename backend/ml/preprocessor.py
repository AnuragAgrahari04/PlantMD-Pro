"""
Image Preprocessing & Validation Service
Handles upload validation, sanitization, and preprocessing pipeline.
"""
import hashlib
import io
from pathlib import Path
from typing import Tuple

import numpy as np
import structlog
from PIL import Image, UnidentifiedImageError

from core.config import settings
from core.exceptions import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedFormatError,
)

logger = structlog.get_logger(__name__)

# Minimum dimensions for meaningful leaf detection
MIN_WIDTH = 64
MIN_HEIGHT = 64
MAX_WIDTH = 4096
MAX_HEIGHT = 4096


def validate_and_load_image(
    file_bytes: bytes,
    filename: str = "",
    content_type: str = "",
) -> Tuple[np.ndarray, str]:
    """
    Validate, sanitize, and load image from raw bytes.

    Returns:
        (image_array, image_hash) tuple
    Raises:
        InvalidImageError, ImageTooLargeError, UnsupportedFormatError
    """
    # 1. Size check
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.UPLOAD_MAX_SIZE_MB:
        raise ImageTooLargeError(settings.UPLOAD_MAX_SIZE_MB)

    # 2. Extension check
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(ext)

    # 3. Load and decode with PIL (PIL validates format internally)
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        pil_image.verify()  # Detect truncated files
        pil_image = Image.open(io.BytesIO(file_bytes))  # Re-open after verify
        pil_image = pil_image.convert("RGB")  # Normalize to RGB
    except (UnidentifiedImageError, Exception) as e:
        raise InvalidImageError(f"Cannot decode image: {e}")

    # 5. Dimension check
    w, h = pil_image.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        raise InvalidImageError(f"Image too small (min {MIN_WIDTH}x{MIN_HEIGHT}px, got {w}x{h})")
    if w > MAX_WIDTH or h > MAX_HEIGHT:
        # Resize large images instead of rejecting
        pil_image = pil_image.resize((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)
        logger.info("Large image resized", original_size=(w, h))

    # 6. Convert to numpy array
    image_array = np.array(pil_image, dtype=np.float32)

    # 7. Compute hash for caching
    image_hash = hashlib.sha256(file_bytes).hexdigest()

    logger.info("Image validated", size_mb=round(size_mb, 2), dimensions=(w, h), hash=image_hash[:8])
    return image_array, image_hash


def _validate_magic_bytes(data: bytes, claimed_ext: str) -> bool:
    """Validate file magic bytes against claimed extension."""
    if len(data) < 12:
        return False

    magic_map = {
        "jpg":  [b"\xff\xd8\xff"],
        "jpeg": [b"\xff\xd8\xff"],
        "png":  [b"\x89PNG\r\n\x1a\n"],
        "gif":  [b"GIF87a", b"GIF89a"],
    }

    # WebP: bytes 0-3 == "RIFF" AND bytes 8-11 == "WEBP"
    if claimed_ext == "webp":
        return data[:4] == b"RIFF" and data[8:12] == b"WEBP"

    signatures = magic_map.get(claimed_ext, [])
    if not signatures:
        return True  # Unknown ext — let PIL handle it
    return any(data.startswith(sig) for sig in signatures)


def create_heatmap_overlay(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> bytes:
    """
    Overlay Grad-CAM heatmap on original image and return as PNG bytes.
    """
    import cv2

    # Normalize cam to 0-255
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize original if needed
    orig = original_image.astype(np.uint8)
    if orig.shape[:2] != (224, 224):
        orig = np.array(Image.fromarray(orig).resize((224, 224)))

    overlay = (alpha * heatmap + (1 - alpha) * orig).astype(np.uint8)
    pil_out = Image.fromarray(overlay)

    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    return buf.getvalue()