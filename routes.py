from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Annotated
import logging

from config import settings
from schemas import VerificationResponse, ErrorResponse
from service import decode_image, verify_faces

logger = logging.getLogger(__name__)

router = APIRouter()


# ── /verify ───────────────────────────────────────────────────────────────────

@router.post(
    "/verify",
    response_model=VerificationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input (bad file type, size, count)"},
        422: {"model": ErrorResponse, "description": "No face detected in one or more images"},
        500: {"model": ErrorResponse, "description": "Internal model error"},
    },
    summary="Verify identity from reference images + selfies",
    description="""
Upload **1–2 reference images** and **2–5 selfie images** (JPG/PNG).

The API will:
1. Detect faces in every image using **RetinaFace**.
2. Compute **Facenet512** embeddings for each face.
3. Compare all reference–selfie pairs via **cosine distance**.
4. Return `verified` if ≥60% of pairs match, otherwise `rejected`.
    """,
)
async def verify_identity(
    reference_images: Annotated[
        list[UploadFile],
        File(description=f"1–2 reference photos (JPG/PNG, max 10MB each)"),
    ],
    selfie_images: Annotated[
        list[UploadFile],
        File(description=f"2–5 selfie photos (JPG/PNG, max 10MB each)"),
    ],
):
    # ── Validate counts ───────────────────────────────────────────────────────
    if not (settings.MIN_REFERENCE_IMAGES <= len(reference_images) <= settings.MAX_REFERENCE_IMAGES):
        raise HTTPException(
            status_code=400,
            detail=f"Provide between {settings.MIN_REFERENCE_IMAGES} and {settings.MAX_REFERENCE_IMAGES} reference images. Got {len(reference_images)}.",
        )

    if not (settings.MIN_SELFIE_IMAGES <= len(selfie_images) <= settings.MAX_SELFIE_IMAGES):
        raise HTTPException(
            status_code=400,
            detail=f"Provide between {settings.MIN_SELFIE_IMAGES} and {settings.MAX_SELFIE_IMAGES} selfie images. Got {len(selfie_images)}.",
        )

    # ── Read & decode reference images ────────────────────────────────────────
    ref_arrays = []
    ref_filenames = []
    for upload in reference_images:
        raw = await upload.read()
        img = decode_image(raw, upload.filename or "reference.jpg")
        ref_arrays.append(img)
        ref_filenames.append(upload.filename or "reference.jpg")

    # ── Read & decode selfie images ───────────────────────────────────────────
    selfie_arrays = []
    selfie_filenames = []
    for upload in selfie_images:
        raw = await upload.read()
        img = decode_image(raw, upload.filename or "selfie.jpg")
        selfie_arrays.append(img)
        selfie_filenames.append(upload.filename or "selfie.jpg")

    # ── Run verification ──────────────────────────────────────────────────────
    logger.info(f"Verifying {len(ref_arrays)} reference(s) against {len(selfie_arrays)} selfie(s)")

    result = verify_faces(
        reference_images=ref_arrays,
        selfie_images=selfie_arrays,
        reference_filenames=ref_filenames,
        selfie_filenames=selfie_filenames,
    )

    logger.info(f"Result: {result.status} | confidence={result.confidence_score} | ratio={result.match_ratio}")
    return result


# ── /verify/quick ─────────────────────────────────────────────────────────────

@router.post(
    "/verify/quick",
    response_model=VerificationResponse,
    summary="Quick 1-vs-1 face comparison",
    description="Compare a single reference image against a single selfie. Useful for quick spot-checks.",
)
async def verify_quick(
    reference_image: Annotated[UploadFile, File(description="Single reference photo (JPG/PNG)")],
    selfie_image: Annotated[UploadFile, File(description="Single selfie photo (JPG/PNG)")],
):
    ref_raw = await reference_image.read()
    ref_img = decode_image(ref_raw, reference_image.filename or "reference.jpg")

    selfie_raw = await selfie_image.read()
    selfie_img = decode_image(selfie_raw, selfie_image.filename or "selfie.jpg")

    return verify_faces(
        reference_images=[ref_img],
        selfie_images=[selfie_img],
        reference_filenames=[reference_image.filename or "reference.jpg"],
        selfie_filenames=[selfie_image.filename or "selfie.jpg"],
    )