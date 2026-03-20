from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # ── Model settings ──────────────────────────────────────────────
    RECOGNITION_MODEL: str = "Facenet512"          # Best accuracy/speed balance
    DETECTOR_BACKEND: str = "retinaface"           # Best for angled mobile selfies
    DISTANCE_METRIC: str = "cosine"                # Most reliable for Facenet512

    # ── Verification thresholds ──────────────────────────────────────
    # Cosine distance: lower = more similar (0 = identical, 1 = completely different)
    MATCH_THRESHOLD: float = 0.40                  # Strict for dating app identity verify
    # Minimum fraction of selfie angles that must match to pass
    MIN_MATCH_RATIO: float = 0.50                  # e.g., 3 of 5 angles must match

    # ── Image constraints ────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list[str] = ["jpg", "jpeg", "png"]
    MIN_REFERENCE_IMAGES: int = 1
    MAX_REFERENCE_IMAGES: int = 2
    MIN_SELFIE_IMAGES: int = 2
    MAX_SELFIE_IMAGES: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()