"""
Merge LoRA Checkpoint Script (No AWQ)

Merges LoRA adapter with base model and saves as fp16.
Run AWQ quantization separately afterwards.

Usage:
    python merge_lora.py
"""

from pathlib import Path

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# Configuration - UPDATE THESE PATHS
# =============================================================================
CONFIG = {
    # Input: LoRA checkpoint from fine-tuning
    "lora_checkpoint_path": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-nemo-minimal-lora/checkpoint-2400",
    # Base model
    "base_model": "mistralai/Mistral-Nemo-Instruct-2407",
    # Output: Merged model (fp16)
    "merged_model_path": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-nemo-merged",
}


def merge_lora_checkpoint(config: dict):
    """Merge LoRA adapter with base model and save."""
    print("=" * 60)
    print("MERGE LORA CHECKPOINT")
    print("=" * 60)

    lora_path = config["lora_checkpoint_path"]
    base_model_name = config["base_model"]
    merged_path = config["merged_model_path"]

    print(f"\n📂 Base model: {base_model_name}")
    print(f"📂 LoRA checkpoint: {lora_path}")
    print(f"📂 Output path: {merged_path}")

    # Load tokenizer
    print("\n🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model in fp16 (NOT quantized - needed for clean merge)
    print("🔧 Loading base model in fp16 (this may take a while)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load and apply LoRA adapter
    print("🔧 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    # Merge LoRA weights into base model
    print("🔧 Merging LoRA weights...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"\n💾 Saving merged model to: {merged_path}")
    Path(merged_path).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    print("\n" + "=" * 60)
    print("✅ MERGE COMPLETE!")
    print("=" * 60)
    print(f"\nMerged model saved to: {merged_path}")
    print("\nNext: Quantize to AWQ using llm-compressor (vLLM's tool)")

    return merged_path


if __name__ == "__main__":
    merge_lora_checkpoint(CONFIG)
