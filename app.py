from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# IMPORT MODULE ROUTERS
from modules.sinta_emotional_tts import router as sinta_emotional_tts_router

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
    allow_origins=["*"],   # OK for research/demo
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ROOT ROUTE (IMPORTANT FOR RENDER)
# ============================================================
@app.get("/")
def root():
    return {
        "message": "Sinhala + Tamil Emotion & TTS API is running",
        "docs": "/docs",
        "health": "/health"
    }

# ============================================================
# HEALTH CHECK (USED BY RENDER / MONITORING)
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}

# ============================================================
# REGISTER ROUTERS
# ============================================================
app.include_router(sinta_emotional_tts_router)
