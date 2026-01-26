

import os
import shutil
import torch
from speechbrain.inference.speaker import SpeakerRecognition

# --- PATHS ---
SOURCE_DIR = '/Users/insights-trainee/Desktop/Cursor/Voice_notes/voice-notes/Phone_2022/2_Converted_WAVs'
ANCHOR_FILE = '/Users/insights-trainee/Desktop/Cursor/Voice_notes/my_voice_anchor.wav'
BASE_OUTPUT = '/Users/insights-trainee/Desktop/Cursor/Voice_notes/voice-notes/Phone_2022/3_Filtered_Results'

# --- THRESHOLDS ---
HIGH_CONFIDENCE = 0.5
LOW_CONFIDENCE = 0.3

# Create folders
folders = {
    "correct": os.path.join(BASE_OUTPUT, "1_Correct"),
    "review": os.path.join(BASE_OUTPUT, "2_Need_Review"),
    "rejected": os.path.join(BASE_OUTPUT, "3_Rejected")
}

for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

# Load the model
print("Loading Speaker Recognition model...")
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models"
)

def get_score(file_to_check):
    score, prediction = verification.verify_files(ANCHOR_FILE, file_to_check)
    return score.item()

print(f"Starting three-tier filtering...")

for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".wav"):
        file_path = os.path.join(SOURCE_DIR, filename)
        
        try:
            score = get_score(file_path)
            
            if score >= HIGH_CONFIDENCE:
                status = "✅ CORRECT"
                target_folder = folders["correct"]
            elif LOW_CONFIDENCE <= score < HIGH_CONFIDENCE:
                status = "⚠️ REVIEW"
                target_folder = folders["review"]
            else:
                status = "❌ REJECTED"
                target_folder = folders["rejected"]
            
            print(f"{status} ({score:.2f}): {filename}")
            shutil.copy2(file_path, os.path.join(target_folder, filename))
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"\nProcessing complete! Check your results in: {BASE_OUTPUT}")