import os
from pydub import AudioSegment

# --- YOUR SPECIFIC PATHS ---
SOURCE = '/Users/insights-trainee/Desktop/Cursor/Voice_notes/WhatsApp Voice Notes'
# Note: We point the conversion script to the FLATTENED folder from Step 1
FLATTENED_DIR = '/Users/insights-trainee/Desktop/Cursor/Voice_notes/voice-notes/Phone_2022/1_Flattened_Raw'
OUTPUT_WAV_DIR = '/Users/insights-trainee/Desktop/Cursor/Voice_notes/voice-notes/Phone_2022/2_Converted_WAVs'

# Ensure the output directory exists
os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)

print(f"Starting conversion of files from: {FLATTENED_DIR}")

# Counters for tracking
converted_count = 0
error_count = 0

for filename in os.listdir(FLATTENED_DIR):
    if filename.endswith(".opus"):
        try:
            input_path = os.path.join(FLATTENED_DIR, filename)
            # Change extension to .wav
            output_filename = filename.replace(".opus", ".wav")
            output_path = os.path.join(OUTPUT_WAV_DIR, output_filename)

            # Load and process
            audio = AudioSegment.from_file(input_path, codec="opus")
            
            # ML Optimization: 16k sample rate, 1 channel (Mono)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            audio.export(output_path, format="wav")
            converted_count += 1
            if converted_count % 10 == 0:
                print(f"Converted {converted_count} files...")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1

print(f"\n--- Process Complete ---")
print(f"Successfully converted: {converted_count}")
print(f"Errors encountered: {error_count}")
print(f"Files saved to: {OUTPUT_WAV_DIR}")