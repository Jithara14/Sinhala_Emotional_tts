from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# IMPORT MODULE ROUTERS
from modules.sinta_emotional_tts import router as sinta_emotional_tts_router

app = FastAPI(
    title="Research Backend – Common API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# REGISTER ROUTERS
app.include_router(sinta_emotional_tts_router)


