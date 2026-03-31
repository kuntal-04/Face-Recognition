import cv2
import numpy as np
from PIL import Image
import io
import insightface
from insightface.app import FaceAnalysis
from fastapi import HTTPException
import logging
from config import settings
from schemas import PairResult, VerificationResponse, VerificationStatus

logger = logging.getLogger(__name__)


# ── InsightFace Setup ─────────────────────────────────────────────────────────

def load_insight_face():
    """Load InsightFace ArcFace model — downloads once, cached locally."""
    app = FaceAnalysis(
        name="buffalo_sc",           # lightweight model — good accuracy, fast
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("InsightFace loaded successfully.")
    return app


try:
    face_app = load_insight_face()
except Exception as e:
    logger.error(f"InsightFace failed to load: {e}")
    face_app = None


# ── Image Reading ─────────────────────────────────────────────────────────────

def decode_image(image_bytes: bytes, filename: str) -> np.ndarray:
    """Decode uploaded image bytes into OpenCV BGR array."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' has unsupported format. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' exceeds {settings.MAX_FILE_SIZE_MB}MB limit.",
        )

    # Try OpenCV first
    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    # Fallback to PIL — handles any format or encoding
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(pil_img)[:, :, ::-1]  # RGB to BGR
        return img
    except Exception:
        pass

    raise HTTPException(
        status_code=400,
        detail=f"Could not read image '{filename}'. Please try a different photo.",
    )


# ── Face Detection + Embedding ────────────────────────────────────────────────

def get_face_embedding(img: np.ndarray, filename: str) -> np.ndarray:
    """
    Detect face and get ArcFace embedding in one step using InsightFace.
    Returns 512-dimensional embedding vector.
    """
    if face_app is None:
        raise HTTPException(
            status_code=500,
            detail="Face recognition model not loaded. Please restart the server.",
        )

    # InsightFace expects RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        faces = face_app.get(img_rgb)
    except Exception as e:
        logger.error(f"InsightFace error on '{filename}': {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Face detection failed for '{filename}': {str(e)}",
        )

    if not faces:
        raise HTTPException(
            status_code=422,
            detail=f"No face detected in '{filename}'. Please ensure the face is clearly visible and well-lit.",
        )

    if len(faces) > 1:
        logger.warning(f"Multiple faces in '{filename}', using largest.")
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

    return faces[0].embedding


# ── Distance & Matching ───────────────────────────────────────────────────────

def cosine_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine distance between two embeddings. 0 = identical, 1 = completely different."""
    a = np.array(emb_a)
    b = np.array(emb_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return float(1.0 - dot / norm)


def distance_to_confidence(distance: float, threshold: float) -> float:
    """Convert cosine distance to 0-1 confidence score."""
    score = 1.0 - (distance / (threshold * 2))
    return float(max(0.0, min(1.0, score)))


# ── Main Verification ─────────────────────────────────────────────────────────

def verify_faces(
    reference_images: list[np.ndarray],
    selfie_images: list[np.ndarray],
    reference_filenames: list[str],
    selfie_filenames: list[str],
) -> VerificationResponse:
    """
    Compare each reference image against every selfie.
    Returns verified if match_ratio >= MIN_MATCH_RATIO.
    """

    # Step 1 — get reference embeddings
    ref_embeddings = []
    for img, fname in zip(reference_images, reference_filenames):
        emb = get_face_embedding(img, fname)
        ref_embeddings.append(emb)

    # Step 2 — get selfie embeddings
    selfie_embeddings = []
    for img, fname in zip(selfie_images, selfie_filenames):
        emb = get_face_embedding(img, fname)
        selfie_embeddings.append(emb)

    # Step 3 — compare all pairs
    pair_results: list[PairResult] = []
    matched = 0

    for ref_idx, ref_emb in enumerate(ref_embeddings):
        for sel_idx, sel_emb in enumerate(selfie_embeddings):
            dist = cosine_distance(ref_emb, sel_emb)
            is_match = dist < settings.MATCH_THRESHOLD
            confidence = distance_to_confidence(dist, settings.MATCH_THRESHOLD)

            if is_match:
                matched += 1

            pair_results.append(
                PairResult(
                    reference_index=ref_idx,
                    selfie_index=sel_idx,
                    is_match=is_match,
                    confidence_score=round(confidence, 4),
                    distance=round(dist, 4),
                )
            )

    total_pairs = len(pair_results)
    match_ratio = matched / total_pairs if total_pairs > 0 else 0.0

    # Step 4 — overall confidence
    if matched > 0:
        matched_confidences = [p.confidence_score for p in pair_results if p.is_match]
        overall_confidence = round(sum(matched_confidences) / len(matched_confidences), 4)
    else:
        overall_confidence = round(max(p.confidence_score for p in pair_results), 4)

    # Step 5 — verdict
    is_verified = match_ratio >= settings.MIN_MATCH_RATIO
    status = VerificationStatus.VERIFIED if is_verified else VerificationStatus.REJECTED

    if is_verified:
        message = (
            f"Identity verified. {matched}/{total_pairs} image pairs matched "
            f"with {overall_confidence * 100:.1f}% confidence."
        )
    else:
        message = (
            f"Identity rejected. Only {matched}/{total_pairs} pairs matched "
            f"(minimum required ratio: {settings.MIN_MATCH_RATIO:.0%}). "
            "Please retake clearer selfies in good lighting."
        )

    return VerificationResponse(
        status=status,
        confidence_score=overall_confidence,
        matched_pairs=matched,
        total_pairs=total_pairs,
        match_ratio=round(match_ratio, 4),
        pair_results=pair_results,
        message=message,
    )