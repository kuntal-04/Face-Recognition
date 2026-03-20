from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"


class PairResult(BaseModel):
    reference_index: int = Field(..., description="Index of the reference image (0-based)")
    selfie_index: int = Field(..., description="Index of the selfie image (0-based)")
    is_match: bool = Field(..., description="Whether this pair is the same person")
    confidence_score: float = Field(..., description="Confidence score 0.0–1.0 (higher = more confident match)")
    distance: float = Field(..., description="Raw cosine distance (lower = more similar)")


class VerificationResponse(BaseModel):
    status: VerificationStatus = Field(..., description="'verified' or 'rejected'")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence score (0.0–1.0). Weighted average across all pairs.",
    )
    matched_pairs: int = Field(..., description="Number of reference-selfie pairs that matched")
    total_pairs: int = Field(..., description="Total number of pairs compared")
    match_ratio: float = Field(..., description="Fraction of pairs that matched (matched/total)")
    pair_results: list[PairResult] = Field(..., description="Detailed results per image pair")
    message: str = Field(..., description="Human-readable result summary")


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None