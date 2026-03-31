from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from routes import router
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Face Recognition API starting up...")
    print("   Model: InsightFace (buffalo_sc)")
    print(f"   Threshold: {settings.MATCH_THRESHOLD}")
    yield
    print("🛑 Face Recognition API shutting down...")


app = FastAPI(
    title="Face Recognition API",
    description="Dating app face verification using InsightFace ArcFace.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Face Recognition API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)