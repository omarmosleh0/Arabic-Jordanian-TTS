%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    import torch; v = re.match(r"[0-9]{1,}\.[0-9]{1,}", str(torch.__version__)).group(0)
    xformers = "xformers==" + ("0.0.33.post1" if v=="2.9" else "0.0.32.post2" if v=="2.8" else "0.0.29.post3")
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
!git clone https://github.com/SparkAudio/Spark-TTS
!pip install omegaconf einx torchcodec "datasets>=3.4.1,<4.0.0"

!uv pip install torchvision>=0.24.0 --system
!uv pip install wandb --upgrade --system



#####

from unsloth import FastModel
import torch
from huggingface_hub import snapshot_download

max_seq_length = 2048 # Choose any for long context!

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    # Qwen3 new models
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    # Other very popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

# Download model and code
snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir = "Spark-TTS-0.5B")

model, tokenizer = FastModel.from_pretrained(
    model_name = "Spark-TTS-0.5B/LLM",
    max_seq_length = max_seq_length,
    dtype = torch.float32, # Spark seems to only work on float32 for now
    full_finetuning = True, # We support full finetuning now!
    load_in_4bit = False,
    #token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


!huggingface-cli login --token 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
!wandb login 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




# 2. Load the dataset from the local folder
dataset = load_dataset(
    "csv",
    data_files=os.path.join(local_data_path, "metadata .csv"),
    split='train'
)
def add_full_path(example):
    example["audio"] = os.path.join(local_data_path, example["audio"])
    return example

dataset = dataset.map(add_full_path)
# 3. Select the 'train' split

dataset = dataset.cast_column("audio", Audio())
print(f"Total samples in dataset: {len(dataset)}")
print(f"Column names: {dataset.column_names}")

# 4. Verify the structure
print(f"\nProcessed {len(dataset)} samples")
print(f"Sample data structure:")
print(f"  - Text: {dataset[0]['text'][:50]}...")
print(f"  - Audio keys: {dataset[0]['audio'].keys()}")
print(f"  - Audio sampling rate: {dataset[0]['audio']['sampling_rate']}")

print(f"\nDataset ready for TTS training!")



## Push just incase we need to rerun again
dataset.push_to_hub("omarmosleh/second-dataset-mapped", private=True)


# View the first entry
sample = dataset[0]
print("\nFirst sample row structure:")
print(sample)

# Specifically check the audio column format
if "audio" in sample:
    print("\nAudio column content sample:", sample["audio"])


    from datasets import load_dataset, Audio
from huggingface_hub import snapshot_download
import os

# 1. Download the entire dataset to a local directory first
local_data_path = "./my_local_dataset"

if not os.path.exists(local_data_path):
    print("Downloading dataset...")
    snapshot_download(
        repo_id="omarmosleh/third-dataset",
        repo_type="dataset",
        local_dir=local_data_path
    )
    print(f"Download complete!")
else:
    print(f"Using existing dataset at: {local_data_path}")

dataset = load_dataset(local_data_path, split="train")  # <-- no CSV here

# Check columns
print(dataset.column_names)
import locale
import torchaudio.transforms as T
import os
import torch
import sys
import numpy as np
sys.path.append('Spark-TTS')
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize


audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", "cuda")
def extract_wav2vec2_features( wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""

        if wavs.shape[0] != 1:

             raise ValueError(f"Expected batch size 1, but got shape {wavs.shape}")
        wav_np = wavs.squeeze(0).cpu().numpy()

        processed = audio_tokenizer.processor(
            wav_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = processed.input_values

        input_values = input_values.to(audio_tokenizer.feature_extractor.device)

        model_output = audio_tokenizer.feature_extractor(
            input_values,
        )


        if model_output.hidden_states is None:
             raise ValueError("Wav2Vec2Model did not return hidden states. Ensure config `output_hidden_states=True`.")

        num_layers = len(model_output.hidden_states)
        required_layers = [11, 14, 16]
        if any(l >= num_layers for l in required_layers):
             raise IndexError(f"Requested hidden state indices {required_layers} out of range for model with {num_layers} layers.")

        feats_mix = (
            model_output.hidden_states[11] + model_output.hidden_states[14] + model_output.hidden_states[16]
        ) / 3
        return feats_mix


def formatting_audio_func(example):
    # 1. Use raw text directly from your dataset
    # We use example["text"] since that is what your metadata.csv loading created
    raw_text = example["text"]

    # 2. Process Audio
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    target_sr = audio_tokenizer.config['sample_rate']
def formatting_audio_func(example):
    # Use torch.no_grad() to save massive amounts of memory
    with torch.no_grad():
        raw_text = example["text"]

        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        target_sr = audio_tokenizer.config['sample_rate']

        # 1. Resample on CPU (Don't use GPU for this!)
        if sampling_rate != target_sr:
            resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
            audio_tensor_temp = torch.from_numpy(audio_array).float()
            audio_array = resampler(audio_tensor_temp).numpy()

        if audio_tokenizer.config["volume_normalize"]:
            audio_array = audio_volume_normalize(audio_array)

        # 2. Tokenize on GPU
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to("cuda")
        ref_wav_np = audio_tokenizer.get_ref_clip(audio_array)
        ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to("cuda")

        feat = extract_wav2vec2_features(audio_tensor)
        batch = {"wav": audio_tensor, "ref_wav": ref_wav_tensor, "feat": feat}

        semantic_token_ids, global_token_ids = audio_tokenizer.model.tokenize(batch)

        # 3. IMMEDIATELY turn them into strings and clear tensors
        global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze().cpu().numpy()])
        semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze().cpu().numpy()])

        # Clean up GPU memory for this specific sample
        del audio_tensor, ref_wav_tensor, feat, batch

        inputs = f"<|task_tts|><|start_content|>{raw_text}<|end_content|><|start_global_token|>{global_tokens}<|end_global_token|><|start_semantic_token|>{semantic_tokens}<|end_semantic_token|><|im_end|>"

        return {"text": inputs}
    feat = extract_wav2vec2_features(audio_tensor)
    batch = {
        "wav": audio_tensor,
        "ref_wav": ref_wav_tensor,
        "feat": feat.to(audio_tokenizer.device),
    }

    semantic_token_ids, global_token_ids = audio_tokenizer.model.tokenize(batch)

    # 4. Format for the LLM
    global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze().cpu().numpy()])
    semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze().cpu().numpy()])

    # This creates the final prompt string the model learns from
    inputs = [
        "<|task_tts|>",
        "<|start_content|>",
        raw_text,  # Clean, raw text from your metadata
        "<|end_content|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        semantic_tokens,
        "<|end_semantic_token|>",
        "<|im_end|>"
    ]
    inputs = "".join(inputs)
    return {"text": inputs}

model.cpu() # Move the 0.5B model to RAM
torch.cuda.empty_cache() # Clear the "ghost" memory

# Apply to your dataset
dataset = dataset.map(formatting_audio_func, remove_columns=["audio"])

print("Moving Bicodec model and Wav2Vec2Model to cpu.")
audio_tokenizer.model.cpu()
audio_tokenizer.feature_extractor.cpu()
torch.cuda.empty_cache()

## Push just incase we need to rerun again
dataset.push_to_hub("omarmosleh/second-dataset-tokenized", private=True)