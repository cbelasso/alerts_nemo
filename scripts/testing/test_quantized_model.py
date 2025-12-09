from collections import Counter
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import List, Literal, Optional

from llm_parallelization.new_processor import NewProcessor
from llm_parallelization.parallelization import (
    LLAMA_33_70B_INSTRUCT,
    NEMO,
    QWEN_AWQ,
    FlexibleSchemaProcessor,
)
import pandas as pd
from pydantic import BaseModel


# -------------------------------
# Pydantic models
# -------------------------------
class AlertSpan(BaseModel):
    excerpt: str
    reasoning: str
    alert_type: Literal[
        "discrimination",
        "sexual_harassment",
        "severe_harassment",
        "bullying",
        "workplace_violence",
        "threat_of_violence",
        "coercive_threat",
        "safety_hazard",
        "retaliation",
        "substance_abuse_at_work",
        "data_breach",
        "security_incident",
        "fraud",
        "corruption",
        "quid_pro_quo",
        "ethics_violation",
        "mental_health_crisis",
        "pattern_of_unfair_treatment",
        "workload_burnout_risk",
        "management_concern",
        "interpersonal_conflict",
        "professional_misconduct",
        "inappropriate_language",
        "profanity",
        "suggestive_language",
        "mental_wellbeing_concern",
        "physical_safety_concern",
    ]
    severity: Literal["low", "moderate", "high", "critical"]


class AlertsOutput(BaseModel):
    has_alerts: bool
    alerts: List[AlertSpan]
    non_alert_classification: Optional[
        Literal[
            "performance_complaint",
            "quality_complaint",
            "workload_feedback",
            "process_improvement",
            "resource_request",
            "general_dissatisfaction",
            "constructive_feedback",
            "positive_feedback",
            "neutral_comment",
            "unclear",
        ]
    ] = None
    non_alert_reasoning: Optional[str] = None


def save_results_to_csv(
    texts: List[str], results: List[AlertsOutput], output_path: str = None
) -> None:
    """
    Save results to CSV in exploded format (one row per alert).
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"alerts_results_{timestamp}.csv"

    # Create directories if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, (text, result) in enumerate(zip(texts, results)):
        text_id = f"text_{idx + 1}"

        if result is None:
            rows.append(
                {
                    "text_id": text_id,
                    "original_text": text,
                    "has_alerts": None,
                    "alert_number": None,
                    "total_alerts": None,
                    "alert_type": None,
                    "severity": None,
                    "excerpt": None,
                    "reasoning": None,
                    "non_alert_classification": None,
                    "non_alert_reasoning": "PARSE_FAILED",
                }
            )
        elif result.has_alerts:
            for alert_num, alert in enumerate(result.alerts, 1):
                rows.append(
                    {
                        "text_id": text_id,
                        "original_text": text,
                        "has_alerts": True,
                        "alert_number": alert_num,
                        "total_alerts": len(result.alerts),
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "excerpt": alert.excerpt,
                        "reasoning": alert.reasoning,
                        "non_alert_classification": None,
                        "non_alert_reasoning": None,
                    }
                )
        else:
            rows.append(
                {
                    "text_id": text_id,
                    "original_text": text,
                    "has_alerts": False,
                    "alert_number": None,
                    "total_alerts": 0,
                    "alert_type": None,
                    "severity": None,
                    "excerpt": None,
                    "reasoning": None,
                    "non_alert_classification": result.non_alert_classification,
                    "non_alert_reasoning": result.non_alert_reasoning,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\n📁 Results saved to: {output_path}")
    print(f"   Total rows: {len(df)}")


def create_minimal_prompt(text: str) -> str:
    prompt = f"<s>[INST] {text} [/INST]"
    return prompt


def main():
    df = pd.read_csv(
        "/home/clyde/workspace/alerts_detection_llama/scripts/generation/datasets/claude_text_comments.csv"
    )

    model = "qwen_35ft"
    times = 10

    texts = df["comment"].tolist()

    texts = texts * times

    model_dict = {
        "nemo": NEMO,
        "llama70": LLAMA_33_70B_INSTRUCT,
        "qwen": QWEN_AWQ,
        "nemo_ft": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-nemo-awq-pure",
        "mistral": "mistralai/Ministral-3-14B-Instruct-2512",
        "mistral_big": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "ministral": "mistralai/Ministral-8B-Instruct-2410",
        "ministral_ft": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-ministral_8b-awq-native",
        "llama_ft": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-llama_3b-awq-native",
        "qwen_ft": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-qwen-awq",
        "qwen_35ft": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-qwen_32b-awq",
    }

    if model == "nemo_ft":
        batch_size = 250

        processor = NewProcessor(
            gpu_list=[4, 5, 6, 7],
            llm=model_dict[model],
            gpu_memory_utilization=0.95,
            tokenizer="mistralai/Mistral-Nemo-Instruct-2407",  # use with nemo_ft
            max_model_len=2048 * 5,
            multiplicity=1,
            enforce_eager=True,
        )

    elif model == "qwen_35ft":
        batch_size = 250

        processor = NewProcessor(
            gpu_list=[4, 5, 6, 7],
            llm=model_dict[model],
            gpu_memory_utilization=0.95,
            tokenizer="Qwen/Qwen2.5-32B-Instruct",  # use with nemo_ft
            max_model_len=2048,
            multiplicity=1,
            enforce_eager=True,
        )

    elif model == "llama_ft":
        batch_size = 250

        processor = NewProcessor(
            gpu_list=[4, 5, 6, 7],
            llm=model_dict[model],
            gpu_memory_utilization=0.95,
            tokenizer="meta-llama/Llama-3.2-3B-Instruct",  # use with nemo_ft
            max_model_len=2048,
            multiplicity=1,
            enforce_eager=True,
        )

    elif model == "ministral_ft":
        batch_size = 250

        processor = NewProcessor(
            gpu_list=[4, 5, 6, 7],
            llm=model_dict[model],
            gpu_memory_utilization=0.95,
            tokenizer="mistralai/Ministral-8B-Instruct-2410",  # use with nemo_ft
            max_model_len=2048,
            multiplicity=1,
            enforce_eager=True,
        )

    elif model == "mistral_big":
        batch_size = 25
        processor = NewProcessor(
            gpu_list=[4, 5, 6, 7],
            llm=model_dict[model],
            gpu_memory_utilization=0.95,
            # tokenizer="mistralai/Mistral-Nemo-Instruct-2407", #use with nemo_ft
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            skip_tokenizer_init=True,  # Don't try to load tokenizer externally
            max_model_len=2048 * 5,
            multiplicity=1,
            enforce_eager=True,
            tensor_parallel_size=2,
        )

    else:
        batch_size = 25

        processor = NewProcessor(
            gpu_list=[4, 5, 6, 7],
            llm=model_dict[model],
            gpu_memory_utilization=0.95,
            # tokenizer="mistralai/Mistral-Nemo-Instruct-2407", #use with nemo_ft
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            skip_tokenizer_init=True,  # Don't try to load tokenizer externally
            max_model_len=2048 * 5,
            multiplicity=1,
            enforce_eager=True,
        )

    try:
        prompts = [create_minimal_prompt(text=text) for text in texts]

        start_time = time.time()
        processor.process_with_schema(
            prompts=prompts, schema=AlertsOutput, batch_size=batch_size
        )
        results: List[AlertsOutput] = processor.parse_results_with_schema(schema=AlertsOutput)
        end_time = time.time()

        print(f"\n{'=' * 80}")
        print(f"{model} MODEL CLASSIFICATION RESULTS")
        print(f"{'=' * 80}")
        print(f"Processed {len(results)} responses")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"{'=' * 80}\n")

        successful_parses = [(i, r) for i, r in enumerate(results) if r is not None]
        failed_parses = len(results) - len(successful_parses)

        print(f"✓ Successful: {len(successful_parses)} | ✗ Failed: {failed_parses}\n")

        for idx, response in successful_parses:
            original_text = texts[idx]

            print(f"{'=' * 80}")
            print(f"ID: text_{idx + 1}")
            print(f"{'=' * 80}")
            print("📄 ORIGINAL TEXT:")
            print(f"{'-' * 80}")
            print(f"{original_text}")
            print(f"{'-' * 80}")

            print(f"\n🚨 ALERTS DETECTED: {response.has_alerts}")

            if response.has_alerts:
                print(f"   Number of alerts: {len(response.alerts)}")
                for j, alert in enumerate(response.alerts, 1):
                    print(f"\n  Alert {j}:")
                    print(f"    Type: {alert.alert_type}")
                    print(f"    Severity: {alert.severity}")
                    print(f"    Excerpt: {alert.excerpt}")
                    print(f"    Reasoning: {alert.reasoning}")
            else:
                print("   Number of alerts: 0")
                if response.non_alert_classification:
                    print(f"\n  📋 Content Type: {response.non_alert_classification}")
                    print(f"  💬 Description: {response.non_alert_reasoning}")

            print(f"\n{'=' * 80}\n")

        # Summary statistics
        if successful_parses:
            print("\n" + "=" * 80)
            print("SUMMARY STATISTICS")
            print("=" * 80)

            total_alerts = sum(len(response.alerts) for _, response in successful_parses)
            texts_with_alerts = sum(
                1 for _, response in successful_parses if response.has_alerts
            )

            print(f"Total texts analyzed: {len(successful_parses)}")
            print(f"Texts with alerts: {texts_with_alerts}")
            print(f"Texts without alerts: {len(successful_parses) - texts_with_alerts}")
            print(f"Total alerts detected: {total_alerts}")

            all_alert_types = []
            all_severities = []
            non_alert_types = []

            for _, response in successful_parses:
                if response.has_alerts:
                    for alert in response.alerts:
                        all_alert_types.append(alert.alert_type)
                        all_severities.append(alert.severity)
                else:
                    if response.non_alert_classification:
                        non_alert_types.append(response.non_alert_classification)

            if all_alert_types:
                alert_type_counts = Counter(all_alert_types)
                severity_counts = Counter(all_severities)

                print("\nAlert types distribution:")
                for alert_type, count in alert_type_counts.most_common():
                    print(f"  {alert_type}: {count}")

                print("\nSeverity distribution:")
                for severity, count in severity_counts.most_common():
                    print(f"  {severity}: {count}")

            if non_alert_types:
                non_alert_counts = Counter(non_alert_types)
                print("\nNon-alert content distribution:")
                for content_type, count in non_alert_counts.most_common():
                    print(f"  {content_type}: {count}")

        # Save results to CSV
        save_results_to_csv(
            texts,
            results,
            output_path=f"/home/clyde/workspace/llama_alerts/{model}/quantized_model_results.csv",
        )

        print(f"\n{'=' * 80}")
        print(f"{model} MODEL CLASSIFICATION RESULTS")
        print(f"{'=' * 80}")
        print(f"Processed {len(results)} responses")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"{'=' * 80}\n")

    finally:
        processor.terminate()


if __name__ == "__main__":
    main()
