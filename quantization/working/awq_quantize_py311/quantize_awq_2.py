"""
AWQ Quantization using AutoAWQ with config patching
"""

import json
from pathlib import Path
import random
import shutil

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

CONFIG = {
    "merged_model_path": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-ministral_8b-minimal-merged",
    "awq_output_path": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-ministral_8b-awq-native",
    "n_samples": 128,
    "max_seq_length": 512,
    "zero_prompt": True,
}


def patch_model_config(model_path: str):
    """Temporarily change model_type from 'ministral' to 'mistral'."""
    config_path = Path(model_path) / "config.json"
    backup_path = Path(model_path) / "config.json.backup"

    shutil.copy(config_path, backup_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    original_type = config.get("model_type")
    print(f"📝 Original model_type: {original_type}")

    if original_type == "ministral":
        config["model_type"] = "mistral"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("📝 Patched model_type: mistral")

    return backup_path


def restore_model_config(model_path: str, backup_path: Path):
    """Restore original config."""
    config_path = Path(model_path) / "config.json"
    if backup_path.exists():
        shutil.move(backup_path, config_path)
        print("📝 Restored original config")


def get_calibration_data(tokenizer, config: dict):
    """Generate calibration data for AWQ quantization."""
    n_samples = config["n_samples"]
    zero_prompt = config["zero_prompt"]

    calibration_texts = [
        # Alerts
        "He called me a fucking chink in front of everyone and nobody did anything about it.",
        "My manager has been submitting fake expense reports for trips he never took.",
        "Ever since I filed that complaint, I've been excluded from all important meetings.",
        "She implied that a promotion might happen faster if we spent some private time together.",
        "I don't see the point anymore. Everything feels hopeless and I've been thinking about not waking up.",
        "The emergency exits have been blocked with boxes for weeks and nobody cares.",
        "He slammed me against the wall in the break room when no one was looking.",
        "The procurement guy is steering all contracts to his brother's company.",
        "Someone left all the customer credit card info on an unsecured shared drive.",
        "My coworker keeps a flask in his desk and is clearly drunk by lunch every day.",
        # Non-alerts
        "The new software update is buggy and crashes frequently.",
        "The project timeline seems aggressive given our current resources.",
        "It would be more efficient if we had a shared calendar for meeting room bookings.",
        "We really need a second printer on this floor to reduce wait times.",
        "The training was fantastic! The instructor really knew their stuff.",
        "My team lead never meets deadlines and it delays everyone else's work.",
        "Things just aren't the same since the reorganization.",
        "The onboarding process could be improved by adding a mentor system.",
        "The quarterly meeting is scheduled for next Tuesday at 3pm.",
        "I don't like coming on Monday mornings.",
        # Mixed/Complex
        "Great benefits, but my manager said I'd advance faster if I was 'more friendly' with him.",
        "The training was excellent, but the VP kept asking about my relationship status.",
        "Love my job overall, but ever since I reported the expense issues, my workload tripled.",
        "Crazy day - server crashed, Jim called me 'sweetheart' again, pizza party was fun.",
        "Good velocity this sprint, but found weird access logs from an IP in Russia at 3am.",
    ]

    random.seed(42)
    samples = []
    while len(samples) < n_samples:
        samples.extend(calibration_texts)
    samples = samples[:n_samples]
    random.shuffle(samples)

    formatted_samples = []
    for text in samples:
        if zero_prompt:
            formatted = f"<s>[INST] {text} [/INST]"
        else:
            formatted = f"<s>[INST] You are a workplace alerts classifier. Analyze the input and return JSON with alerts detected.\n\nComment:\n{text} [/INST]"
        formatted_samples.append(formatted)

    return formatted_samples


def quantize_awq_native(config: dict):
    """Quantize model to native AWQ format using AutoAWQ."""
    print("=" * 60)
    print("AWQ QUANTIZATION (AutoAWQ - Native Format)")
    print("=" * 60)

    merged_path = config["merged_model_path"]
    output_path = config["awq_output_path"]

    print(f"\n📂 Input model: {merged_path}")
    print(f"📂 Output path: {output_path}")

    # Patch config
    print("\n🔧 Patching model config...")
    backup_path = patch_model_config(merged_path)

    try:
        # Load tokenizer
        print("\n🔧 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(merged_path)

        # Get calibration data
        print(f"📊 Preparing {config['n_samples']} calibration samples...")
        print(
            f"   Prompt format: {'Zero prompt' if config['zero_prompt'] else 'With system prompt'}"
        )
        calibration_data = get_calibration_data(tokenizer, config)

        # Load model for quantization
        print("\n📦 Loading model for quantization...")
        model = AutoAWQForCausalLM.from_pretrained(
            merged_path, device_map="auto", safetensors=True
        )

        # Quantization config
        quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

        print("\n⚡ Running AWQ quantization (this may take 10-30 minutes)...")
        print(f"   Config: {quant_config}")

        # Quantize
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calibration_data,
            max_calib_samples=config["n_samples"],
            max_calib_seq_len=config["max_seq_length"],
        )

        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Save quantized model
        print(f"\n💾 Saving quantized model to {output_path}...")
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        print("\n" + "=" * 60)
        print("✅ AWQ QUANTIZATION COMPLETE!")
        print("=" * 60)
        print(f"\nAWQ model saved to: {output_path}")

    finally:
        # Restore original config
        restore_model_config(merged_path, backup_path)


if __name__ == "__main__":
    quantize_awq_native(CONFIG)
