import os
import shutil
from pathlib import Path

def flatten_whatsapp_voice_notes(source_path, output_path):
    # Convert string paths to Path objects for easier handling
    src_dir = Path(source_path)
    dest_dir = Path(output_path)
    
    # Create the destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    files_moved = 0
    
    print(f"Scanning {src_dir} for voice notes...")
    
    # rglob stands for "recursive glob" - it finds all .opus files in subfolders
    for file_path in src_dir.rglob('*.opus'):
        # Get the folder name (e.g., 202445) to use as a prefix for uniqueness
        parent_folder = file_path.parent.name
        new_filename = f"{parent_folder}_{file_path.name}"
        
        # Define the final destination
        destination = dest_dir / new_filename
        
        # Copy the file
        shutil.copy2(file_path, destination)
        files_moved += 1
        
    print(f"Success! {files_moved} files flattened into: {dest_dir}")

# --- SETUP ---
# Update these paths to match your local computer
# Example: 'C:/Users/Intern/Downloads/WhatsApp Voice Notes'
SOURCE = '/Users/insights-trainee/Desktop/Cursor/Voice_notes/voice-notes/Phone_2022/0_Raw_Extracted_From_WhatsApp' 
OUTPUT = '/Users/insights-trainee/Desktop/Cursor/Voice_notes/voice-notes/Phone_2022/1_Flattened_Raw'

flatten_whatsapp_voice_notes(SOURCE, OUTPUT)