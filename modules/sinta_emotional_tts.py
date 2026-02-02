###############################################################
#  ONE FASTAPI BACKEND FOR SINHALA + TAMIL EMOTION + TTS
###############################################################

import os
import re
import emoji
import torch
from uuid import uuid4
from dotenv import load_dotenv
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

torch.set_num_threads(1)

# ============================================
# LOAD ENV
# ============================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# CLEAN TEXT
# ============================================
def clean_text(text: str, lang: str):
    text = emoji.demojize(str(text))
    text = re.sub(r"http\S+|www\S+", "", text)

    if lang == "si":
        text = re.sub(r"[^0-9A-Za-z\u0D80-\u0DFF.,!?\s]", " ", text)
    else:
        text = re.sub(r"[^0-9A-Za-z\u0B80-\u0BFF.,!?\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()

# ============================================
# CONSTANTS
# ============================================
SINHALA_MODEL_DIR = "jithara/sinbert_sinhala_best"
TAMIL_MODEL_DIR = "jithara/best_emotion_model"

SINHALA_MAX_LEN = 96
TAMIL_MAX_LEN = 128

SINHALA_TTS_DIR = "tts_outputs_sinhala"
TAMIL_TTS_DIR = "tts_outputs"

os.makedirs(SINHALA_TTS_DIR, exist_ok=True)
os.makedirs(TAMIL_TTS_DIR, exist_ok=True)

# ============================================
# GLOBAL MODELS
# ============================================
si_tokenizer = None
si_model = None
ta_tokenizer = None
ta_model = None
si_id2label = None
ta_id2label = None

# ============================================
# 🔥 MODEL LOADER (UNCHANGED LOGIC)
# ============================================
def load_models():
    global si_tokenizer, si_model, ta_tokenizer, ta_model, si_id2label, ta_id2label

    if si_model is None:
        print("📌 Loading Sinhala model...")
        si_tokenizer = AutoTokenizer.from_pretrained(SINHALA_MODEL_DIR)
        si_model = AutoModelForSequenceClassification.from_pretrained(
            SINHALA_MODEL_DIR
        ).to(DEVICE).eval()
        si_id2label = si_model.config.id2label

    if ta_model is None:
        print("📌 Loading Tamil model...")
        ta_tokenizer = AutoTokenizer.from_pretrained(TAMIL_MODEL_DIR)
        ta_model = AutoModelForSequenceClassification.from_pretrained(
            TAMIL_MODEL_DIR
        ).to(DEVICE).eval()
        ta_id2label = ta_model.config.id2label

# ============================================
# EMOTION META (UNCHANGED)
# ============================================
SI_EMOTION_META = {
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

TA_EMOTION_META = SI_EMOTION_META

# ============================================
# EMOTION PREDICT (UNCHANGED)
# ============================================
def predict_emotion(text, lang):
    load_models()
    cleaned = clean_text(text, lang)

    if lang == "si":
        tokenizer, model, max_len, label_map, meta_map = (
            si_tokenizer, si_model, SINHALA_MAX_LEN, si_id2label, SI_EMOTION_META
        )
    else:
        tokenizer, model, max_len, label_map, meta_map = (
            ta_tokenizer, ta_model, TAMIL_MAX_LEN, ta_id2label, TA_EMOTION_META
        )

    enc = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_len
    )

    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE)
        ).logits

    pred_id = torch.argmax(logits, dim=1).item()
    raw = label_map[pred_id].strip().lower()

    emotion_map = {
        "anger": "anger", "ang": "anger",
        "fear": "fear",
        "sad": "sad", "sadness": "sad",
        "happy": "happy", "happ": "happy",
        "surprise": "surprise",
        "neutral": "neutral"
    }

    emotion = emotion_map.get(raw, "neutral")
    meta = meta_map[emotion]
    meta["emotion_name"] = emotion

    return {"text": text, "emotion": emotion, **meta}

# ============================================
# TTS (UNCHANGED)
# ============================================
def generate_tts(text, meta, lang):
    instructions = (
        f"Affect: {meta['voice_affect']}\n"
        f"Tone: {meta['tone']}\n"
        f"Pacing: {meta['pacing']}\n"
        f"Personality: {meta['personality']}\n"
        f"Pauses: {meta['pauses']}\n"
        f"Emotion: {meta['emotion_description']}"
    )

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="ballad",
        input=text,
        instructions=instructions
    )

    filename = f"{uuid4().hex}.wav"
    folder = SINHALA_TTS_DIR if lang == "si" else TAMIL_TTS_DIR
    path = os.path.join(folder, filename)

    with open(path, "wb") as f:
        f.write(response.read())

    return filename

# ============================================
# ROUTES (UNCHANGED)
# ============================================
router = APIRouter()

@router.post("/sinhala/predict-emotion")
async def sinhala_predict(text: str = Form(...)):
    return JSONResponse(predict_emotion(text, "si"))

@router.post("/sinhala/predict-emotion-tts")
async def sinhala_predict_tts(text: str = Form(...)):
    result = predict_emotion(text, "si")
    audio = generate_tts(result["text"], result, "si")
    return {"success": True, "classification": result, "audio_url": f"/audio/sinhala/{audio}"}

@router.get("/audio/sinhala/{filename}")
async def sinhala_audio(filename: str):
    return FileResponse(os.path.join(SINHALA_TTS_DIR, filename))

@router.post("/tamil/predict-emotion")
async def tamil_predict(text: str = Form(...)):
    return JSONResponse(predict_emotion(text, "ta"))

@router.post("/tamil/predict-emotion-tts")
async def tamil_predict_tts(text: str = Form(...)):
    result = predict_emotion(text, "ta")
    audio = generate_tts(result["text"], result, "ta")
    return {"success": True, "classification": result, "audio_url": f"/audio/tamil/{audio}"}

@router.get("/audio/tamil/{filename}")
async def tamil_audio(filename: str):
    return FileResponse(os.path.join(TAMIL_TTS_DIR, filename))
