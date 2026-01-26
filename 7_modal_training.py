# this is for modal.com training

import modal

app = modal.App("spark-tts-unsloth-training")

# ------------------------------------------------------------
# Image
# ------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "datasets>=3.4.1,<4.0.0",
        "trl",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "huggingface_hub>=0.34.0",
        "omegaconf",
        "einops",
        "wandb",
        "unsloth",
    )
    .run_commands(
        "git clone https://github.com/SparkAudio/Spark-TTS"
    )
)

volume = modal.Volume.from_name("spark-tts-data", create_if_missing=True)




# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6, # 6 hours
    volumes={"/data": volume},
)
def train():
    # -----------------------------
    # TOKENS (INLINE)
    # -----------------------------
    import os
    # NOTE: In a production Wiley environment, we would use modal.Secret
    os.environ["HF_TOKEN"] = "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"
    os.environ["WANDB_API_KEY"] = "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"
    os.environ["WANDB_PROJECT"] = "spark-tts-incremental"
    os.environ["HF_HOME"] = "/data/hf"
    
    # -----------------------------
    # Imports
    # -----------------------------
    import torch
    from unsloth import FastModel
    from huggingface_hub import snapshot_download, HfApi
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import wandb
    import glob

    wandb.login()

    print("CUDA:", torch.cuda.get_device_name(0))

    # -----------------------------
    # Define Paths
    # -----------------------------
    # This is the directory where your PREVIOUS training finished
    PREVIOUS_OUTPUT_DIR = "/data/spark-tts-normalized-da7ee7-8k-lm-egy"
    
    # This is where we will save the NEW incremental run
    NEW_OUTPUT_DIR = "/data/spark-tts-incremental-run"
    
    # This is the repo on Hugging Face where we want to push the results
    HF_REPO_ID = "omarmosleh/spark-tts-2nd-dataset" 

    # -----------------------------
    # Incremental Model Loading
    # -----------------------------
    max_seq_length = 2048
    
    # Check if we have the previous checkpoint in the volume
    if os.path.exists(PREVIOUS_OUTPUT_DIR) and os.listdir(PREVIOUS_OUTPUT_DIR):
        print(f"🔄 Found previous training data at: {PREVIOUS_OUTPUT_DIR}")
        
        # We try to find the latest checkpoint folder inside that directory
        # (e.g., /checkpoint-1000) or use the root if the model was saved at the end
        checkpoints = sorted(glob.glob(f"{PREVIOUS_OUTPUT_DIR}/checkpoint-*"))
        if checkpoints:
            model_path_to_load = checkpoints[-1] # Take the latest checkpoint
            print(f"📍 Resuming from latest checkpoint: {model_path_to_load}")
        else:
            model_path_to_load = PREVIOUS_OUTPUT_DIR
            print(f"📍 Loading from root output dir: {model_path_to_load}")
            
    else:
        print("⚠️ Previous output directory not found or empty. Downloading base model instead.")
        snapshot_download(
            "SparkAudio/Spark-TTS-0.5B",
            local_dir="/data/Spark-TTS-0.5B",
            token=os.environ["HF_TOKEN"],
        )
        model_path_to_load = "/data/Spark-TTS-0.5B/LLM"

    # Load the model with the weights from the previous run
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path_to_load,
        max_seq_length=max_seq_length,
        dtype=torch.float32,
        load_in_4bit=False,
        full_finetuning=True,
    )

    # -----------------------------
    # Dataset (The NEW Data)
    # -----------------------------
    print("📚 Loading dataset: omarmosleh/second-dataset-tokenized")
    dataset = load_dataset(
        "omarmosleh/second-dataset-tokenized",
        split="train",
        token=os.environ["HF_TOKEN"],
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=6,
            learning_rate=2e-4, 
            fp16=False,
            bf16=False,
            logging_steps=50,
            
            # Save strategy
            save_steps=500,
            save_total_limit=2,
            output_dir=NEW_OUTPUT_DIR,
            
            # Hugging Face Hub Integration
            push_to_hub=True,
            hub_model_id=HF_REPO_ID,
            hub_strategy="every_save", # Important: Push every time a checkpoint is saved
            hub_token=os.environ["HF_TOKEN"],
            
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="wandb",
        ),
    )

    # -----------------------------
    # Execution & Safety Push
    # -----------------------------
    try:
        print("🚀 Starting Training...")
        trainer.train()
        print("✅ Training complete.")
        
    except Exception as e:
        print(f"❌ Training interrupted/failed: {e}")
        
    finally:
        # This block runs whether training succeeds OR fails
        print("💾 Starting Safety Push Procedure...")
        
        # 1. Try standard trainer push
        try:
            trainer.push_to_hub("Final push (post-script execution)")
            print("✅ Trainer push_to_hub successful.")
        except Exception as e:
            print(f"⚠️ Trainer push failed: {e}")
            
            # 2. Manual Fallback via HfApi
            print("🔄 Attempting manual folder upload via HfApi...")
            try:
                api = HfApi(token=os.environ["HF_TOKEN"])
                api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)
                api.upload_folder(
                    folder_path=NEW_OUTPUT_DIR,
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                )
                print("✅ Manual HfApi upload successful.")
            except Exception as inner_e:
                print(f"❌ Manual upload also failed: {inner_e}")

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
with app.run():
    train.remote()