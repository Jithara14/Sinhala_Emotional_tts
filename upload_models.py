import os
from huggingface_hub import upload_folder

token = os.getenv("HF_TOKEN")
assert token is not None, "HF_TOKEN not found"

print("🚀 Uploading Sinhala model...")
upload_folder(
    folder_path="sinbert_sinhala_best",
    repo_id="Jithara/sinbert_sinhala_best",
    repo_type="model",
    token=token,
)

print("🚀 Uploading Tamil model...")
upload_folder(
    folder_path="best_emotion_model",
    repo_id="Jithara/best_emotion_model",
    repo_type="model",
    token=token,
)

print("✅ ALL UPLOADS COMPLETE")
