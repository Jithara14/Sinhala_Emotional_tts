import os
import re
import json
import uuid
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from openai import AsyncOpenAI

# =====================================================
# 1️⃣ Load environment variables
# =====================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")

# =====================================================
# 2️⃣ Configure OpenAI
# =====================================================
openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# =====================================================
# 3️⃣ Create FastAPI App
# =====================================================
app = FastAPI(
    title="OpenAI Emotion → OpenAI TTS API",
    version="2.0.0",
    description="Sinhala & Tamil emotion classification using OpenAI and expressive TTS using OpenAI (MP3 Stable)"
)

# =====================================================
# 4️⃣ CORS
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 5️⃣ Output directory
# =====================================================
TTS_DIR = "tts_outputs"
os.makedirs(TTS_DIR, exist_ok=True)

# =====================================================
# 6️⃣ Emotion Metadata
# =====================================================
EMOTION_META = {

    "happy": {
        "voice_affect": "Warm, cheerful, bright emotional color with genuine joy and a natural smile in the voice",
        "tone": "Lively, friendly, uplifting tone with smooth expressive energy",
        "pacing": "Energetic, rhythmic pacing with natural flow",
        "emotion_description": "Expressing happiness, excitement, comfort, and positive feelings",
        "personality": "Kind, optimistic, joyful, and emotionally open",
        "pauses": "Short, soft pauses that feel playful and natural"
    },

    "sad": {
        "voice_affect": "Low, soft tone with a gentle, heavy-hearted emotional color",
        "tone": "Muted, reflective, slightly trembling tone carrying emotional weight",
        "pacing": "Slow pacing with long emotional pauses and softened delivery",
        "emotion_description": "Deep sadness, disappointment, grief, or emotional pain",
        "personality": "Sensitive, calm, introspective, emotionally delicate",
        "pauses": "Long, deep pauses showing heaviness and emotional reflection"
    },

    "fear": {
        "voice_affect": "Shaky, tense emotional color with audible nervousness",
        "tone": "Hesitant, unstable tone with anxious fluctuations",
        "pacing": "Uneven pacing with sudden short pauses showing tension",
        "emotion_description": "Fear, stress, uncertainty, or nervousness",
        "personality": "Alert, worried, cautious, easily startled",
        "pauses": "Irregular, broken pauses that express fear or hesitation"
    },

    "anger": {
        "voice_affect": "Strong, forceful emotional color with heated energy",
        "tone": "Sharp, firm, intense tone with clear irritation",
        "pacing": "Fast, pressured, and forceful speaking rhythm",
        "emotion_description": "Anger, frustration, conflict, or strong disagreement",
        "personality": "Bold, assertive, direct, and intense",
        "pauses": "Short tight pauses that emphasize strong emotions"
    },

    "surprise": {
        "voice_affect": "Excited, bright emotional color with high alertness",
        "tone": "High-pitched and expressive tone",
        "pacing": "Fast bursts with rising pitch patterns",
        "emotion_description": "Unexpected shock, amazement, confusion, or discovery",
        "personality": "Curious, expressive, reactive",
        "pauses": "Quick dramatic pauses that highlight surprise"
    },

    "neutral": {
        "voice_affect": "Calm, balanced emotional color with steady clarity",
        "tone": "Natural, clear, and even tone without emotional bias",
        "pacing": "Smooth, steady pacing with clear articulation",
        "emotion_description": "Neutral and objective with no emotional load",
        "personality": "Professional, calm, composed",
        "pauses": "Natural pauses with consistent timing"
    }
}

# =====================================================
# 7️⃣ OpenAI Emotion Classifier
# =====================================================
async def classify_emotion(text: str):

    prompt = f"""
You are an expert Sinhala and Tamil emotion classifier.

Emotion classes:
happy, sad, fear, anger, surprise, neutral.

Return ONLY JSON:

{{
"text": "{text}",
"emotion": "<one_of:[happy,sad,fear,anger,surprise,neutral]>"
}}
"""

    response = await openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You classify emotions in Sinhala and Tamil sentences."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except:
        result = {"text": text, "emotion": "neutral"}

    emotion = result.get("emotion", "neutral").lower()

    if emotion not in EMOTION_META:
        emotion = "neutral"

    meta = EMOTION_META[emotion]

    return {
        "text": text,
        "emotion": emotion,
        **meta
    }

# =====================================================
# 8️⃣ OpenAI TTS (MP3 Streaming)
# =====================================================
async def generate_tts_audio(text, instructions):
    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(TTS_DIR, filename)

    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="ballad",
        input=text,
        instructions=instructions,
        response_format="mp3",
    ) as response:

        with open(output_path, "wb") as f:
            async for chunk in response.iter_bytes():
                f.write(chunk)

    return filename

# =====================================================
# 9️⃣ API: Emotion → TTS
# =====================================================
@app.post("/classify-emotion-tts")
async def classify_emotion_tts(text: str = Form(...)):

    emotion_result = await classify_emotion(text)

    tts_instructions = (
        f"Affect: {emotion_result.get('voice_affect','')}\n"
        f"Tone: {emotion_result.get('tone','')}\n"
        f"Pacing: {emotion_result.get('pacing','')}\n"
        f"Personality: {emotion_result.get('personality','')}\n"
        f"Pauses: {emotion_result.get('pauses','')}\n"
        f"Emotion: {emotion_result.get('emotion_description','')}"
    )

    audio_file = await generate_tts_audio(
        emotion_result.get("text", text),
        tts_instructions
    )

    return {
        "success": True,
        "emotion_result": emotion_result,
        "audio_url": f"/audio/{audio_file}"
    }

# =====================================================
# 🔟 Audio serving
# =====================================================
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = os.path.join(TTS_DIR, filename)

    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename=filename
    )

# =====================================================
# 11️⃣ Root + Health
# =====================================================
@app.get("/")
def root():
    return {
        "message": "Emotion → TTS API is running (MP3 Stable)",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok"}