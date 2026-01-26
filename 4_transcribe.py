import os
import csv
from faster_whisper import WhisperModel

# --- SETTINGS ---
# Path to your clean voice notes
AUDIO_DIR = '/Users/insights-trainee/Desktop/Cursor/Voice_notes/Filtered_Results/1_Correct_Test'
OUTPUT_CSV = '/Users/insights-trainee/Desktop/Cursor/Voice_notes/metadata.csv'

# Use "large-v3" for best Jordanian dialect results
# Use "base" if you want to test quickly first
MODEL_SIZE = "large-v3"

# Load Model (Using CPU for Mac; if you have M1/M2/M3 it will use the Neural Engine)
print(f"Loading Whisper {MODEL_SIZE} model... this may take a minute.")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

# --- TRANSCRIPTION LOGIC ---
print("Starting transcription. This will take some time...")

with open(OUTPUT_CSV, mode='w', encoding='utf-8-sig') as f:
    writer = csv.writer(f, delimiter='|')
    
    # Sort files to keep things organized
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])

    for filename in audio_files:
        file_path = os.path.join(AUDIO_DIR, filename)
        
        # We provide a "initial_prompt" to help Whisper with Jordanian dialect
        segments, info = model.transcribe(
            file_path, 
            beam_size=5, 
            language="ar",
            initial_prompt="هذا تسجيل بلهجة أردنية عامية" # "This is a recording in Jordanian dialect"
        )

        full_text = ""
        for segment in segments:
            full_text += segment.text + " "
        
        # Clean up the text (remove extra spaces)
        full_text = full_text.strip()
        
        # Write to our metadata file
        writer.writerow([filename, full_text])
        print(f"Done: {filename} -> {full_text[:50]}...")

print(f"\n✅ All done! Your metadata for TTS training is at: {OUTPUT_CSV}")