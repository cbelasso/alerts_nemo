"""
OPTIMIZED Multi-GPU QLoRA Fine-Tuning Script

Designed for large datasets (50K+ samples) with multiple GPUs.

Key optimizations:
- Multi-GPU training with Accelerate
- Larger batch sizes
- Flash Attention 2
- Gradient checkpointing
- 1 epoch (sufficient for large datasets)

Usage:
    # Single command (uses all available GPUs)
    accelerate launch --multi_gpu finetune_optimized.py

    # Or specify GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu finetune_optimized.py
"""

from datetime import datetime
import json
from pathlib import Path

from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# =============================================================================
# Configuration - OPTIMIZED FOR SPEED
# =============================================================================
CONFIG = {
    # Paths - UPDATE THESE
    "training_data_path": "/home/clyde/workspace/alerts_detection_llama/scripts/finetuning/training_data/alerts_training_20251202_224001.jsonl",
    "output_dir": "/home/clyde/workspace/alerts_detection_llama/models/alerts-ministral_8b-minimal-lora",
    "merged_output_dir": "/home/clyde/workspace/alerts_detection_llama/models/alerts-ministral_8b-minimal-merged",
    # Model
    "base_model": "mistralai/Ministral-8B-Instruct-2410",
    "max_seq_length": 2048,  # Reduced from 4096 - most alerts are short
    # LoRA - slightly reduced for speed
    "lora_r": 16,  # Reduced from 32
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    # Training - OPTIMIZED
    "num_train_epochs": 1,  # 1 epoch is enough for large datasets
    "per_device_train_batch_size": 8,  # Increased from 4
    "gradient_accumulation_steps": 2,  # Reduced - with 8 GPUs, effective = 8*8*2 = 128
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,  # Use ratio instead of steps
    "weight_decay": 0.01,
    # Logging - less frequent for speed
    "logging_steps": 25,
    "save_steps": 200,
    "eval_steps": 200,
    # Validation
    "val_split": 0.02,  # 2% validation
    # SAMPLE SIZE - Set to limit training data
    "max_train_samples": 35000,  # Use 25K samples (plenty for good results!)
}


# =============================================================================
# Data Loading
# =============================================================================
def load_training_data(path: str, max_samples: int = None) -> list:
    """Load training data from JSONL file."""
    import os
    import random

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if is_main:
        print(f"📂 Total examples in file: {len(data)}")

    # Shuffle and sample if max_samples specified
    if max_samples and len(data) > max_samples:
        random.seed(42)  # Reproducible sampling
        random.shuffle(data)
        data = data[:max_samples]
        if is_main:
            print(f"📊 Sampled {max_samples} examples (shuffled)")

    if is_main:
        print(f"✅ Using {len(data)} training examples")
    return data


def format_minimal(examples: list) -> Dataset:
    """Format with zero prompting."""
    formatted_texts = []
    for ex in examples:
        text = f"<s>[INST] {ex['input']} [/INST]{ex['output']}</s>"
        formatted_texts.append({"text": text})
    return Dataset.from_list(formatted_texts)


# =============================================================================
# Model Setup
# =============================================================================
def setup_model(config: dict):
    """Load model optimized for multi-GPU training - FIXED for CUDA compatibility."""
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        print(f"\n🔧 Loading model: {config['base_model']}")

    # Check GPU capability
    if is_main:
        capability = torch.cuda.get_device_capability()
        print(f"📊 GPU Compute Capability: {capability}")
        if capability[0] < 7:
            print("⚠️  GPU doesn't support bfloat16, using float16")

    # 4-bit quantization - USE FLOAT16 for compatibility
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # ← FIXED: Use float16
        bnb_4bit_use_double_quant=True,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model kwargs - FIXED: Use float16
    model_kwargs = {
        "quantization_config": bnb_config,
        "torch_dtype": torch.float16,  # ← FIXED: Use float16
        "trust_remote_code": True,
        "device_map": {"": local_rank},  # Load to current GPU
    }

    if is_main:
        print(f"📍 Loading model to GPU {local_rank}")

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        **model_kwargs,
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA
    if is_main:
        print("🔧 Adding LoRA adapters...")

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# Training
# =============================================================================
def train(config: dict):
    """Main training function - optimized for multi-GPU."""
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        print("=" * 60)
        print("OPTIMIZED MULTI-GPU FINE-TUNING")
        print("=" * 60)

    # Check GPU count
    gpu_count = torch.cuda.device_count()
    if is_main:
        print(f"\n🖥️  Available GPUs: {gpu_count}")

    # Load data
    if is_main:
        print(f"\n📂 Loading: {config['training_data_path']}")

    raw_data = load_training_data(
        config["training_data_path"],
        max_samples=config.get("max_train_samples"),
    )

    # Setup model
    model, tokenizer = setup_model(config)

    # Format data
    if is_main:
        print("\n📝 Formatting data...")
    dataset = format_minimal(raw_data)

    # Split
    if config["val_split"] > 0:
        split = dataset.train_test_split(test_size=config["val_split"], seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        if is_main:
            print(f"   Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None

    # Pre-tokenize
    if is_main:
        print("\n🔧 Pre-tokenizing...")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["max_seq_length"],
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4,  # Parallel tokenization
        desc="Tokenizing train" if is_main else None,
    )

    if eval_dataset:
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=2,
            desc="Tokenizing eval" if is_main else None,
        )

    # Output dir
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate steps
    total_samples = len(train_dataset)
    effective_batch = (
        config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
        * gpu_count
    )
    steps_per_epoch = total_samples // effective_batch
    total_steps = steps_per_epoch * config["num_train_epochs"]

    if is_main:
        print("\n📊 Training stats:")
        print(f"   Total samples: {total_samples}")
        print(f"   Per-device batch: {config['per_device_train_batch_size']}")
        print(f"   Gradient accumulation: {config['gradient_accumulation_steps']}")
        print(f"   GPUs: {gpu_count}")
        print(f"   Effective batch size: {effective_batch}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Total steps: {total_steps}")
        print(f"   Epochs: {config['num_train_epochs']}")

    # Training args - optimized
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type="cosine",
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=2,  # Only keep 2 checkpoints
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config["eval_steps"] if eval_dataset else None,
        load_best_model_at_end=False,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Estimate time
    estimated_seconds_per_step = 1.5  # Rough estimate with optimizations
    estimated_hours = (total_steps * estimated_seconds_per_step) / 3600

    if is_main:
        print("\n🚀 Starting training...")
        print(f"   Estimated time: ~{estimated_hours:.1f} hours")
        print()

    start = datetime.now()
    trainer.train()
    elapsed = datetime.now() - start

    if is_main:
        print(f"\n✅ Training completed in {elapsed}")

    # Save only from main process
    if is_main:
        print(f"\n💾 Saving LoRA adapter: {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        print("\nNext: Run merge_lora.py to merge adapter with base model")

        return output_dir

    return None


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    merged_dir = train(CONFIG)

    if local_rank == 0 and merged_dir:
        print(f"\n🎉 Done! Model saved to: {merged_dir}")
        print("\nNext step: Run quantize_to_awq.py to convert to AWQ for vLLM")
