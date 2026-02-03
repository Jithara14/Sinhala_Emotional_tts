from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modules.sinta_emotional_tts import router as sinta_emotional_tts_router

app = FastAPI(
    title="Sinhala + Tamil Emotion & TTS API",
    version="1.0.0",
    description="Emotion classification and expressive TTS for Sinhala and Tamil"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Sinhala + Tamil Emotion & TTS API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(sinta_emotional_tts_router)
