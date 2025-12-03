"""
AWQ Quantization using llm-compressor (vLLM's tool)
This script uses llm-compressor for quantization.
https://github.com/vllm-project/llm-compressor
Requirements:
    pip install llmcompressor
Usage:
    python quantize_awq.py
"""

from pathlib import Path
import random

from datasets import Dataset  # Import the Dataset class
from llmcompressor.modifiers.awq import AWQModifier

# from llmcompressor import oneshot
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# Configuration - UPDATE THESE PATHS
# =============================================================================
CONFIG = {
    # Input: Merged model (fp16) from merge_lora.py
    "merged_model_path": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-nemo-merged",
    # Output: AWQ quantized model
    "awq_output_path": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-nemo-awq",
    # Calibration
    "n_samples": 128,
    "max_seq_length": 512,
    # Set to True if you trained with zero prompt (no system prompt)
    "zero_prompt": True,
}


def get_calibration_data(tokenizer, config: dict):
    """Generate calibration data for AWQ quantization."""
    n_samples = config["n_samples"]
    zero_prompt = config["zero_prompt"]
    # Mix of alert and non-alert examples
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
    # # Repeat and shuffle
    # random.seed(42)
    # samples = []
    # while len(samples) < n_samples:
    #     samples.extend(calibration_texts)
    # samples = samples[:n_samples]
    # random.shuffle(samples)
    # # Format with correct prompt template
    # formatted_samples = []
    # for text in samples:
    #     if zero_prompt:
    #         formatted = f"<s>[INST] {text} [/INST]"
    #     else:
    #         formatted = f"<s>[INST] You are a workplace alerts classifier. Analyze the input and return JSON with alerts detected.\n\nComment:\n{text} [/INST]"
    #     formatted_samples.append(formatted)
    # return formatted_samples

    # Repeat and shuffle
    random.seed(42)
    samples = []
    while len(samples) < n_samples:
        samples.extend(calibration_texts)
    samples = samples[:n_samples]
    random.shuffle(samples)

    # Format with correct prompt template
    formatted_samples = []
    for text in samples:
        if zero_prompt:
            formatted = f"<s>[INST] {text} [/INST]"
        else:
            formatted = f"<s>[INST] You are a workplace alerts classifier. Analyze the input and return JSON with alerts detected.\n\nComment:\n{text} [/INST]"
        formatted_samples.append({"text": formatted})  # Change to a dict format

    # Create a Hugging Face Dataset from the list of dictionaries
    dataset = Dataset.from_dict({"text": [sample["text"] for sample in formatted_samples]})
    return dataset


def quantize_awq(config: dict):
    """Quantize model to AWQ format using llm-compressor."""
    print("=" * 60)
    print("AWQ QUANTIZATION (llm-compressor)")
    print("=" * 60)
    merged_path = config["merged_model_path"]
    output_path = config["awq_output_path"]
    print(f"\n📂 Input model: {merged_path}")
    print(f"📂 Output path: {output_path}")
    # Load tokenizer
    print("\n🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(merged_path)
    tokenizer.pad_token = tokenizer.eos_token
    # Get calibration data
    print(f"📊 Preparing {config['n_samples']} calibration samples...")
    print(
        f"   Prompt format: {'Zero prompt' if config['zero_prompt'] else 'With system prompt'}"
    )
    calibration_data = get_calibration_data(tokenizer, config)
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # AWQ quantization recipe
    recipe = AWQModifier(
        targets="Linear",
        scheme="W4A16_ASYM",  # 4-bit weights, 16-bit activations
        ignore=["lm_head"],  # Don't quantize the output layer
    )
    print("\n⚡ Running AWQ quantization (this may take 10-30 minutes)...")
    # Run quantization
    oneshot(
        model=merged_path,
        dataset=calibration_data,
        recipe=recipe,
        output_dir=output_path,
        max_seq_length=config["max_seq_length"],
        num_calibration_samples=config["n_samples"],
    )
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    print("\n" + "=" * 60)
    print("✅ AWQ QUANTIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nAWQ model saved to: {output_path}")
    print(f"""
To use with vLLM:
from vllm import LLM, SamplingParams
llm = LLM(
    model="{output_path}",
    quantization="awq",
    dtype="half",
)
# Zero-prompt format
prompt = "<s>[INST] He called me a slur [/INST]"
sampling_params = SamplingParams(temperature=0.1, max_tokens=512)
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
""")


if __name__ == "__main__":
    quantize_awq(CONFIG)
