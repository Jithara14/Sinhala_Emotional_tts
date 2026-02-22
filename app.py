import os
import re
import json
import uuid
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
import google.generativeai as genai
from openai import AsyncOpenAI

# =====================================================
# 1️⃣ Load environment variables
# =====================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_API_KEY or not GEMINI_MODEL:
    raise ValueError("❌ Missing GEMINI_API_KEY or GEMINI_MODEL in .env")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")

# =====================================================
# 2️⃣ Configure APIs
# =====================================================
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# =====================================================
# 3️⃣ Create FastAPI App
# =====================================================
app = FastAPI(
    title="Gemini Emotion → OpenAI TTS API",
    version="2.0.0",
    description="Sinhala & Tamil emotion classification using Gemini and expressive TTS using OpenAI (MP3 Stable)"
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
# 6️⃣ Helper: Safe JSON Parsing for Gemini
# =====================================================
def parse_gemini_json(response_text: str):
    cleaned = re.sub(r'```(?:json)?', '', response_text, flags=re.I)
    cleaned = cleaned.strip('`\n ')
    cleaned = re.sub(r'""(.*)""', r'"\1"', cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r'\{.*\}', cleaned, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

    return {"text": cleaned, "emotion": "neutral"}

# =====================================================
# 7️⃣ Gemini Emotion Classifier
# =====================================================
def classify_emotion(text: str):
    prompt = f"""
You are an expert Sinhala and Tamil emotion classifier.

Emotion classes:
happy, sad, fear, anger, surprise, neutral.

Return ONLY valid JSON:
{{
  "text": "{text}",
  "emotion": "<one_of:[happy,sad,fear,anger,surprise,neutral]>",
  "voice_affect": "<voice style>",
  "tone": "<tone details>",
  "pacing": "<pacing style>",
  "emotion_description": "<explanation>",
  "personality": "<personality traits>",
  "pauses": "<pause style>"
}}

Sentence: "{text}"
"""
    response = gemini_model.generate_content(prompt)
    return parse_gemini_json(response.text)

# =====================================================
# 8️⃣ OpenAI TTS (MP3 Streaming - FIXED)
# =====================================================
async def generate_tts_audio(text, instructions):
    filename = f"{uuid.uuid4().hex}.mp3"   # ✅ MP3 instead of WAV
    output_path = os.path.join(TTS_DIR, filename)

    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="ballad",
        input=text,
        instructions=instructions,
        response_format="mp3",   # ✅ IMPORTANT CHANGE
    ) as response:

        with open(output_path, "wb") as f:
            async for chunk in response.iter_bytes():
                f.write(chunk)

    return filename

# =====================================================
# 9️⃣ API: Gemini → OpenAI TTS
# =====================================================
@app.post("/classify-emotion-tts")
async def classify_emotion_tts(text: str = Form(...)):
    gemini_result = classify_emotion(text)

    tts_instructions = (
        f"Affect: {gemini_result.get('voice_affect','')}\n"
        f"Tone: {gemini_result.get('tone','')}\n"
        f"Pacing: {gemini_result.get('pacing','')}\n"
        f"Personality: {gemini_result.get('personality','')}\n"
        f"Pauses: {gemini_result.get('pauses','')}\n"
        f"Emotion: {gemini_result.get('emotion_description','')}"
    )

    audio_file = await generate_tts_audio(
        gemini_result.get("text", text),
        tts_instructions
    )

    return {
        "success": True,
        "emotion_result": gemini_result,
        "audio_url": f"/audio/{audio_file}"
    }

# =====================================================
# 🔟 Audio serving (Correct Media Type)
# =====================================================
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = os.path.join(TTS_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(
        path,
        media_type="audio/mpeg",  # ✅ Correct for MP3
        filename=filename
    )

# =====================================================
# 11️⃣ Root + Health
# =====================================================
@app.get("/")
def root():
    return {
        "message": "Gemini Emotion → OpenAI TTS API is running (MP3 Stable)",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok"}