"""Custom domain exceptions for PlantMD Pro."""
from fastapi import status


class PlantMDException(Exception):
    def __init__(self, message: str, error_code: str, status_code: int = 400):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(message)


class InvalidImageError(PlantMDException):
    def __init__(self, message: str = "Invalid or corrupted image file"):
        super().__init__(message, "INVALID_IMAGE", status.HTTP_422_UNPROCESSABLE_ENTITY)


class ImageTooLargeError(PlantMDException):
    def __init__(self, max_mb: int = 10):
        super().__init__(f"Image exceeds maximum size of {max_mb}MB", "IMAGE_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)


class UnsupportedFormatError(PlantMDException):
    def __init__(self, ext: str):
        super().__init__(f"File format '{ext}' is not supported", "UNSUPPORTED_FORMAT", status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)


class ModelNotLoadedError(PlantMDException):
    def __init__(self):
        super().__init__("ML model is not loaded", "MODEL_NOT_LOADED", status.HTTP_503_SERVICE_UNAVAILABLE)


class LowConfidenceError(PlantMDException):
    def __init__(self, confidence: float):
        super().__init__(
            f"Prediction confidence ({confidence:.1%}) is below threshold. Please provide a clearer leaf image.",
            "LOW_CONFIDENCE",
            status.HTTP_200_OK,
        )


class AuthenticationError(PlantMDException):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTH_ERROR", status.HTTP_401_UNAUTHORIZED)


class RateLimitError(PlantMDException):
    def __init__(self):
        super().__init__("Rate limit exceeded. Please wait before making more requests.", "RATE_LIMIT", status.HTTP_429_TOO_MANY_REQUESTS)
