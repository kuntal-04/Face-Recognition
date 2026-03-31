from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # ── Model settings ──────────────────────────────────────────────
    DISTANCE_METRIC: str = "cosine"                # Most reliable for Facenet

    # ── Verification thresholds ──────────────────────────────────────
    # Cosine distance: lower = more similar (0 = identical, 1 = completely different)
    MATCH_THRESHOLD: float = 0.60                  # InsightFace calibrated threshold 
    # Minimum fraction of selfie angles that must match to pass
    MIN_MATCH_RATIO: float = 0.40                  # ≥60% of pairs must match (as documented in API)

    # ── Image constraints ────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list[str] = ["jpg", "jpeg", "png"]
    MIN_REFERENCE_IMAGES: int = 1
    MAX_REFERENCE_IMAGES: int = 2
    MIN_SELFIE_IMAGES: int = 2
    MAX_SELFIE_IMAGES: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()