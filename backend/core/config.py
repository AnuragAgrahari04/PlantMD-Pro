"""
Application Configuration — Pydantic BaseSettings
All values can be overridden via environment variables or .env file.
"""
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "PlantMD Pro"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"  # development | staging | production
    DEBUG: bool = True
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_USE_SECRETS_MANAGER"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]

    # Auth
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    API_KEY_HEADER: str = "X-API-Key"

    # ML Model
    MODEL_PATH: str = "models/plantmd_efficientnet.h5"
    MODEL_VERSION: str = "2.0.0"
    LABELS_PATH: str = "models/labels.json"
    IMG_SIZE: int = 224
    BATCH_SIZE: int = 1
    CONFIDENCE_THRESHOLD: float = 0.5
    GRADCAM_LAYER: str = "top_conv"

    # Storage
    UPLOAD_MAX_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]
    UPLOAD_DIR: str = "uploads"
    USE_S3: bool = False
    S3_BUCKET: str = ""
    AWS_REGION: str = "us-east-1"

    # Redis (caching)
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600

    # Database
    DATABASE_URL: str = "sqlite:///./plantmd.db"

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 30

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
