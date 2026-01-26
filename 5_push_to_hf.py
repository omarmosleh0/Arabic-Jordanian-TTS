#/opt/homebrew/bin/python3.11 -m venv .venv source .venv/bin/activate
  

#pip install --upgrade pip                 pip install \            
#  torch==2.1.2 \
#  torchaudio==2.1.2 \
#  speechbrain==1.0.3 \
#  huggingface_hub==0.19.4

from datasets import Dataset, Audio
import pandas as pd
from huggingface_hub import login
import os

# Optional: login with token (or just do `hf auth login` in terminal)
# login("YOUR_HF_TOKEN")

# Path to your dataset folder
dataset_folder = "/Users/insights-trainee/Desktop/Cursor/Voice_notes/voice-notes/Phone_2022/3_Filtered_Results/1_Correct"

# Load metadata
metadata_path = os.path.join(dataset_folder, "metadata.csv")
df = pd.read_csv(metadata_path)

# Add full path for each wav file (datasets library will handle it)
df["audio"] = df["audio"].apply(lambda x: os.path.join(dataset_folder, x))

# Convert to HF Dataset
dataset = Dataset.from_pandas(df)

# Cast the audio column
dataset = dataset.cast_column("audio", Audio())
# Push to Hugging Face hub (private)
dataset.push_to_hub("omarmosleh/fourth-dataset-mapped", private=True)

print("✅ Dataset uploaded successfully!")
