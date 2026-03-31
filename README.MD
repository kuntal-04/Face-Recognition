# Face Recognition API

A FastAPI-based identity verification service for dating apps.
Compares reference photos against multiple selfie angles to confirm it's the same person using **Facenet** + **RetinaFace**.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Local Setup](#local-setup)
- [API Endpoints](#api-endpoints)
- [Request & Response Format](#request--response-format)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [App Flow](#app-flow)
- [Mentor Integration Guide](#mentor-integration-guide)
- [Tips for Better Accuracy](#tips-for-better-accuracy)

---

## Project Structure

```
face_recognition_api/
├── main.py             # FastAPI app entry point, CORS, lifespan
├── routes.py           # API endpoints
├── service.py          # Core logic: RetinaFace + Facenet512 + cosine distance
├── schemas.py          # Pydantic models for request/response
├── config.py           # All thresholds and settings
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| API Framework | FastAPI |
| Face Detection | RetinaFace |
| Face Recognition | DeepFace (Facenet) |
| Image Processing | OpenCV |
| Distance Metric | Cosine similarity |

---

## Local Setup

### Step 1 — Check Python version

This project requires **Python 3.10 or higher**. Check your version:

```bash
python --version
```

If you don't have Python 3.10+ installed, download and install Python 3.13 from the official site:

```
https://www.python.org/downloads/
```

After installing, verify it:

```bash
python --version
# Should show: Python 3.13.x
```

> On some systems you may need to use `python3` instead of `python`.

---

### Step 2 — Clone the repository

```bash
git clone <your-repo-url>
cd face_recognition_api
```

---

### Step 3 — Create virtual environment

```bash
# Windows
python -m venv venv

# Mac / Linux
python3 -m venv venv
```

---

### Step 4 — Activate virtual environment

```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line after activation.

---

### Step 5 — Install dependencies

```bash
pip install -r requirements.txt
```

> This will take a few minutes on first run. TensorFlow alone is ~500MB.
> Make sure you have a stable internet connection.

---

### Step 6 — Run the server

```bash
python main.py
```

You should see this output:

```
INFO:     Started server process
INFO:     Waiting for application startup.
🚀 Face Recognition API starting up...
   Model: Facenet512
   Detector: retinaface
   Threshold: 0.3
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Step 7 — Verify it's running

Open your browser and visit:

```
http://localhost:8000/health
```

You should see:

```json
{ "status": "healthy" }
```

Interactive API docs are available at:

```
http://localhost:8000/docs
```

---

### Stopping the server

Press `Ctrl + C` in the terminal to stop the server.

### Restarting the server

```bash
python main.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/verify` | Full verification — 1–2 reference images + 2–5 selfies |
| POST | `/api/v1/verify/quick` | Quick 1-vs-1 face comparison |
| GET | `/` | Service info |
| GET | `/health` | Health check |

---

## Request & Response Format

### POST `/api/v1/verify`

**Request type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reference_images` | File(s) | Yes | 1–2 reference photos (JPG/PNG, max 10MB each) |
| `selfie_images` | File(s) | Yes | 2–5 selfie photos (JPG/PNG, max 10MB each) |

**Success Response — `200 OK`**

```json
{
  "status": "verified",
  "confidence_score": 0.875,
  "matched_pairs": 10,
  "total_pairs": 12,
  "match_ratio": 0.83,
  "pair_results": [
    {
      "reference_index": 0,
      "selfie_index": 0,
      "is_match": true,
      "confidence_score": 0.87,
      "distance": 0.195
    }
  ],
  "message": "Identity verified. 10/12 image pairs matched with 87.5% confidence."
}
```

**Response fields explained**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `verified` or `rejected` |
| `confidence_score` | float | Overall confidence 0.0–1.0 |
| `matched_pairs` | int | Number of pairs that matched |
| `total_pairs` | int | Total pairs compared |
| `match_ratio` | float | matched / total |
| `pair_results` | array | Per-pair breakdown with distance and score |
| `message` | string | Human-readable result summary |

---

### POST `/api/v1/verify/quick`

Quick 1-vs-1 comparison. Same response format as above.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reference_image` | File | Yes | Single reference photo (JPG/PNG) |
| `selfie_image` | File | Yes | Single selfie photo (JPG/PNG) |

---

## Configuration

All settings are in `config.py` and can be overridden via a `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `RECOGNITION_MODEL` | `Facenet512` | Face embedding model |
| `DETECTOR_BACKEND` | `retinaface` | Face detection backend |
| `MATCH_THRESHOLD` | `0.30` | Cosine distance cutoff per pair (lower = stricter) |
| `MIN_MATCH_RATIO` | `0.60` | Minimum % of pairs that must match for verified result |
| `MAX_FILE_SIZE_MB` | `10` | Max image size per uploaded file |
| `MAX_SELFIE_IMAGES` | `5` | Max number of selfie images allowed |

---

## Error Handling

| HTTP Code | Meaning | Cause |
|-----------|---------|-------|
| `400` | Bad Request | Wrong file type, file too large, or wrong image count |
| `422` | Unprocessable | No face detected in one or more images |
| `500` | Server Error | Internal model or embedding failure |

All error responses follow this format:

```json
{
  "error": "string",
  "detail": "string"
}
```

---

## App Flow

This is the intended user flow for the dating app:

```
Step 1 — User picks 1–2 photos from their device gallery
              ↓
Step 2 — App opens camera, user captures 2–5 selfies
         at different angles (front, left, right, up, down, smile)
              ↓
Step 3 — App sends all images to POST /api/v1/verify
              ↓
Step 4 — API returns verified or rejected with confidence score
              ↓
Step 5 — App shows result to the user
```

**Important notes for mobile integration:**
- Send images as `multipart/form-data` — not base64, not JSON
- Use field name `reference_images` for gallery photos
- Use field name `selfie_images` for camera captures
- Set request timeout to at least 60 seconds — the first request loads the model
- Full body photos are supported — the API automatically detects and crops the face

---

## Mentor Integration Guide

Follow these steps to set up and run the API on a new system.

### Step 1 — Prerequisites

Make sure the following are installed on your system:

| Tool | Minimum Version | Download |
|------|----------------|----------|
| Python | 3.10 or higher | https://www.python.org/downloads/ |
| Git | Any | https://git-scm.com/downloads |
| pip | Comes with Python | — |

Verify Python is installed:

```bash
python --version
```

If Python is not installed or version is below 3.10, install Python 3.13:

```bash
# Windows — download installer from:
https://www.python.org/downloads/release/python-3130/

# Mac — using Homebrew
brew install python@3.13

# Ubuntu / Linux
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-pip
```

---

### Step 2 — Clone the repository

```bash
git clone <your-repo-url>
cd face_recognition_api
```

---

### Step 3 — Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### Step 4 — Install all dependencies

```bash
pip install -r requirements.txt
```

This installs everything automatically including FastAPI, DeepFace, RetinaFace, TensorFlow, and OpenCV.

> First install takes 5–10 minutes depending on internet speed.
> TensorFlow is ~500MB. Make sure you have at least 2GB of free disk space.

---

### Step 5 — Start the API server

```bash
python main.py
```

Expected output:

```
INFO:     Started server process
INFO:     Waiting for application startup.
🚀 Face Recognition API starting up...
   Model: Facenet
   Detector: retinaface
   Threshold: 0.3
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Step 6 — Test the API is working

Open browser and visit:

```
http://localhost:8000/health
```

Expected response:

```json
{ "status": "healthy" }
```

Full interactive docs:

```
http://localhost:8000/docs
```

---

### Step 7 — Call the API from your app

The API accepts `multipart/form-data` POST requests. Here is the base URL:

```
http://localhost:8000/api/v1/verify
```

Send two fields:
- `reference_images` — gallery photos selected by user (1–2 images)
- `selfie_images` — selfies captured from camera (2–5 images)

The API will return `verified` or `rejected` with a confidence score.

---

### Troubleshooting

**Problem: `ModuleNotFoundError` for any package**
```bash
pip install -r requirements.txt
```

**Problem: Port 8000 already in use**
```bash
# Run on a different port
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**Problem: First request is very slow (30–60 seconds)**

This is normal. DeepFace downloads the Facenet512 model weights (~90MB) on the very first request. After that all requests are fast.

**Problem: `No face detected` error (422)**

The uploaded image does not have a clearly visible face. Ask the user to upload a clearer, well-lit photo where the face is visible.

**Problem: Camera not working in test page**

Open the test HTML file in Chrome or Edge. Firefox may block camera access for local files.

---

## Tips for Better Accuracy

- Reference photos should clearly show the face — avoid group photos or far-away shots
- Selfie angles to capture: **front, left, right, up, down, smile**
- Keep angle tilts subtle — extreme angles (fully sideways) reduce match accuracy
- Good lighting matters — faces lit from the front perform best
- Avoid backlighting (e.g. sitting in front of a bright window)
- Minimum face size in any image should be at least 80×80 pixels for reliable detection
- Glasses are handled automatically — the model works with or without glasses
- Full body photos work fine — the API automatically crops the face region