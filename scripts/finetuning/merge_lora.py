"""
Merge LoRA checkpoint with base model - FIXED for fp16
"""

from pathlib import Path

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG = {
    "base_model": "mistralai/Ministral-8B-Instruct-2410",
    "lora_checkpoint": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-ministral_8b-minimal-lora/checkpoint-358",
    "output_dir": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-ministral_8b-minimal-merged",
}


def merge_lora_checkpoint(config: dict):
    """Merge LoRA adapter with base model."""

    print("=" * 60)
    print("MERGE LORA CHECKPOINT")
    print("=" * 60)
    print()
    print(f"📂 Base model: {config['base_model']}")
    print(f"📂 LoRA checkpoint: {config['lora_checkpoint']}")
    print(f"📂 Output path: {config['output_dir']}")
    print()

    # Load tokenizer from checkpoint (has any custom tokens)
    print("🔧 Loading tokenizer from LoRA checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(config["lora_checkpoint"], trust_remote_code=True)

    # Load base model in fp16 - NO quantization config
    print("🔧 Loading base model in fp16 (this may take a while)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.float16,  # Use fp16, not quantized
        device_map="auto",  # Auto distribute across GPUs
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Efficient loading
    )

    # Load LoRA adapter and merge
    print("🔧 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        config["lora_checkpoint"],
        torch_dtype=torch.float16,
    )

    print("🔧 Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Save merged model
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving merged model to {output_path}...")
    model.save_pretrained(
        output_path,
        safe_serialization=True,  # Save as safetensors
        max_shard_size="5GB",  # Shard size
    )

    print("💾 Saving tokenizer...")
    tokenizer.save_pretrained(output_path)

    print()
    print("=" * 60)
    print("✅ MERGE COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Merged model saved to: {output_path}")
    print("\nYou can now use this model with:")
    print("  - Transformers (via AutoModelForCausalLM)")
    print("  - vLLM (recommended for inference)")
    print("  - Text Generation Inference")
    print()

    return output_path


if __name__ == "__main__":
    merged_path = merge_lora_checkpoint(CONFIG)
    print(f"🎉 Done! Model ready at: {merged_path}")
