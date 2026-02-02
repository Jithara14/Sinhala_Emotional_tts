from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# IMPORT ROUTER + MODEL LOADER
from modules.sinta_emotional_tts import router as sinta_emotional_tts_router
from modules.sinta_emotional_tts import load_models

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Sinhala + Tamil Emotion & TTS API",
    version="1.0.0",
    description="Emotion classification and expressive TTS for Sinhala and Tamil"
)

# ============================================================
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🔥 STARTUP WARM-UP (CRITICAL FIX FOR 502)
# ============================================================
@app.on_event("startup")
def warmup():
    print("🔥 Warming up emotion models...")
    load_models()
    print("✅ Models ready")

# ============================================================
# ROOT
# ============================================================
@app.get("/")
def root():
    return {
        "message": "Sinhala + Tamil Emotion & TTS API is running",
        "docs": "/docs",
        "health": "/health"
    }

# ============================================================
# HEALTH
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}

# ============================================================
# ROUTERS
# ============================================================
app.include_router(sinta_emotional_tts_router)
