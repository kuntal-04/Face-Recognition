import cv2
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace
from fastapi import HTTPException
import logging
from config import settings
from schemas import PairResult, VerificationResponse, VerificationStatus

logger = logging.getLogger(__name__)


# ── Image validation ──────────────────────────────────────────────────────────

def decode_image(image_bytes: bytes, filename: str) -> np.ndarray:
    """Decode raw bytes into an OpenCV BGR image."""
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

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image '{filename}'. File may be corrupted.",
        )
    return img


# ── Face detection ────────────────────────────────────────────────────────────

def detect_faces(img: np.ndarray, filename: str) -> list[np.ndarray]:
    """
    Use RetinaFace to detect all faces in an image.
    Returns a list of cropped face arrays.
    Raises HTTPException if no face is found.
    """
    try:
        detections = RetinaFace.detect_faces(img)
    except Exception as e:
        logger.error(f"RetinaFace error on '{filename}': {e}")
        raise HTTPException(status_code=422, detail=f"Face detection failed for '{filename}': {str(e)}")

    if not detections or not isinstance(detections, dict):
        raise HTTPException(
            status_code=422,
            detail=f"No face detected in '{filename}'. Ensure the image is well-lit and the face is clearly visible.",
        )

    faces = []
    for key, face_data in detections.items():
        x1, y1, x2, y2 = face_data["facial_area"]
        h, w = img.shape[:2]
        margin = 20
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
        faces.append(img[y1:y2, x1:x2])

    return faces


def get_primary_face(img: np.ndarray, filename: str) -> np.ndarray:
    """
    Extract exactly one face from an image.
    If multiple faces are found, pick the largest (most prominent).
    """
    faces = detect_faces(img, filename)

    if len(faces) > 1:
        logger.warning(f"Multiple faces detected in '{filename}', using the largest one.")
        faces.sort(key=lambda f: f.shape[0] * f.shape[1], reverse=True)

    return faces[0]


# ── Embedding extraction ──────────────────────────────────────────────────────

def get_embedding(face_img: np.ndarray) -> list[float]:
    """
    Compute a Facenet512 face embedding for a cropped face image.
    """
    try:
        result = DeepFace.represent(
            img_path=face_img,
            model_name=settings.RECOGNITION_MODEL,
            detector_backend="skip",
            enforce_detection=False,
        )
        return result[0]["embedding"]
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compute face embedding: {str(e)}")


# ── Distance & matching ───────────────────────────────────────────────────────

def cosine_distance(emb_a: list[float], emb_b: list[float]) -> float:
    """Compute cosine distance between two embeddings (0 = identical, 1 = opposite)."""
    a = np.array(emb_a)
    b = np.array(emb_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return float(1.0 - dot / norm)


def distance_to_confidence(distance: float, threshold: float) -> float:
    """
    Convert cosine distance to a 0–1 confidence score.
    At distance=0   → confidence=1.0 (perfect match)
    At distance=threshold → confidence=0.5 (decision boundary)
    At distance=1   → confidence=0.0 (no similarity)
    """
    score = 1.0 - (distance / (threshold * 2))
    return float(max(0.0, min(1.0, score)))


# ── Main verification logic ───────────────────────────────────────────────────

def verify_faces(
    reference_images: list[np.ndarray],
    selfie_images: list[np.ndarray],
    reference_filenames: list[str],
    selfie_filenames: list[str],
) -> VerificationResponse:
    """
    Compare each reference image against every selfie image.
    """

    # Step 1 – extract reference embeddings
    ref_embeddings = []
    for img, fname in zip(reference_images, reference_filenames):
        face = get_primary_face(img, fname)
        emb = get_embedding(face)
        ref_embeddings.append(emb)

    # Step 2 – extract selfie embeddings
    selfie_embeddings = []
    for img, fname in zip(selfie_images, selfie_filenames):
        face = get_primary_face(img, fname)
        emb = get_embedding(face)
        selfie_embeddings.append(emb)

    # Step 3 – compare all pairs
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

    # Step 4 – overall confidence
    if matched > 0:
        matched_confidences = [p.confidence_score for p in pair_results if p.is_match]
        overall_confidence = round(sum(matched_confidences) / len(matched_confidences), 4)
    else:
        overall_confidence = round(max(p.confidence_score for p in pair_results), 4)

    # Step 5 – determine status
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