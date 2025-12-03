"""
Standalone Inference Script for Fine-Tuned Alerts Classifier

Use this to test your fine-tuned model after training.
IMPORTANT: Match the prompt format used during training!
"""

import json
import re

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =============================================================================
# Configuration - UPDATE THESE PATHS
# =============================================================================
# For checkpoints (LoRA adapter), use CHECKPOINT_PATH
# For merged models, use MERGED_MODEL_PATH

CHECKPOINT_PATH = (
    "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-nemo-minimal-lora/checkpoint-2400"
)
MERGED_MODEL_PATH = None  # Set this if using a merged model instead

BASE_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"

# Set to True if you trained with zero prompt (no system prompt)
ZERO_PROMPT = True


# =============================================================================
# Load Model
# =============================================================================
def load_model():
    """Load the fine-tuned model."""

    # 4-bit quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    if CHECKPOINT_PATH and not MERGED_MODEL_PATH:
        # Load base model + LoRA adapter (for checkpoints)
        print(f"📂 Loading base model: {BASE_MODEL}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        print(f"📂 Loading LoRA adapter from: {CHECKPOINT_PATH}")
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)

    else:
        # Load merged model directly
        print(f"📂 Loading merged model from: {MERGED_MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    print("✅ Model loaded!")
    return model, tokenizer


# =============================================================================
# Inference
# =============================================================================
def classify_text(text: str, model, tokenizer) -> dict:
    """Classify a single text and return the parsed result."""

    # Build prompt - MUST MATCH TRAINING FORMAT
    if ZERO_PROMPT:
        # Zero-prompt format (what you trained with)
        prompt = f"<s>[INST] {text} [/INST]"
    else:
        # With system prompt
        system = "You are a workplace alerts classifier. Analyze the input and return JSON with alerts detected."
        prompt = f"<s>[INST] {system}\n\nComment:\n{text} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode WITHOUT skipping special tokens first to find [/INST]
    response_with_special = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract everything after [/INST]
    if "[/INST]" in response_with_special:
        json_part = response_with_special.split("[/INST]")[-1].strip()
        # Remove any trailing </s>
        json_part = json_part.replace("</s>", "").strip()
    else:
        # Fallback: try to find JSON in the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        json_part = response

    # Try to extract JSON from the response
    # Sometimes there might be extra text, so find the JSON object
    json_match = re.search(r"\{.*\}", json_part, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # If regex failed, try direct parse
    try:
        return json.loads(json_part)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_response": json_part[:500]}


def run_tests(model, tokenizer):
    """Run test cases."""
    test_cases = [
        "He called me a fucking chink in front of everyone",
        "The new software update is buggy and crashes frequently",
        "Ever since I filed that complaint, I've been excluded from all important meetings",
        "Great benefits, but my manager said I'd advance faster if I was 'more friendly' with him",
        "I don't see the point anymore. Everything feels hopeless.",
        "The coffee machine is broken again",
        "My supervisor keeps making comments about how women shouldn't be in engineering",
    ]

    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print(f"Prompt format: {'Zero prompt' if ZERO_PROMPT else 'With system prompt'}")
    print("=" * 60)

    for i, text in enumerate(test_cases, 1):
        print(f"\n{'─' * 50}")
        print(f"Test {i}")
        print(f"{'─' * 50}")
        print(f"Input: {text[:80]}...")

        result = classify_text(text, model, tokenizer)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            print(f"   Raw: {result.get('raw_response', '')[:300]}")
        else:
            has_alerts = result.get("has_alerts")
            print(f"✅ has_alerts: {has_alerts}")

            if has_alerts and result.get("alerts"):
                for alert in result["alerts"][:3]:
                    print(f"   └─ {alert.get('alert_type')} ({alert.get('severity')})")
                    excerpt = alert.get("excerpt", "")[:50]
                    print(f"      Excerpt: {excerpt}")
            elif not has_alerts:
                classification = result.get("non_alert_classification", "N/A")
                print(f"   └─ Classification: {classification}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    model, tokenizer = load_model()
    run_tests(model, tokenizer)

    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Type a comment to classify (or 'quit' to exit)")
    print("=" * 60)

    while True:
        print()
        text = input("Comment: ").strip()

        if text.lower() in ["quit", "exit", "q"]:
            break

        if not text:
            continue

        result = classify_text(text, model, tokenizer)
        print(json.dumps(result, indent=2))
