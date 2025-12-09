"""
Alerts Classification & Fine-Tuning Data Preparation Script

This script:
1. Loads the combined dataframe (original + rewritten comments)
2. Classifies all texts using Mistral Nemo
3. Converts results to training pairs format for QLoRA fine-tuning
4. Saves training data in multiple formats for QA and training
"""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time
from typing import List, Literal, Optional, Tuple

from llm_parallelization.new_processor import NewProcessor
from llm_parallelization.parallelization import (
    NEMO,
    FlexibleSchemaProcessor,
)
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Input/Output paths
    "input_dataframe_path": "/home/clyde/workspace/alerts_detection_llama/scripts/generation/datasets/final_dataframe_for_alerts_classification.csv",
    "output_dir": "/home/clyde/workspace/alerts_detection_llama/scripts/finetuning/training_data",
    # Column configuration
    "text_columns": ["comment"],  # Columns to classify
    # GPU configuration
    "gpu_list": [4, 5, 6, 7],
    "gpu_memory_utilization": 0.95,
    "max_model_len": 2048 * 5,
    "multiplicity": 1,
}


# =============================================================================
# Pydantic Models (same as alerts_refined.py)
# =============================================================================
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


# =============================================================================
# Prompt (same as alerts_refined.py)
# =============================================================================
def alert_detection_prompt(text: str) -> str:
    """Generate optimized prompt for 8B model with clear category definitions."""
    return f"""You are an expert workplace safety and compliance analyzer. Analyze this employee comment for alerts requiring HR, management, or compliance attention.

COMMENT TO ANALYZE:
{text}

---

ALERT CATEGORIES WITH DEFINITIONS:

**HARASSMENT & DISCRIMINATION:**
- discrimination: Unfair treatment based on race, gender, age, religion, disability, ethnicity, nationality. Includes slurs like "chink", "nigger", "paki", "retard", "towelhead", comments about protected characteristics.
- sexual_harassment: Unwanted sexual advances, inappropriate touching, sexual comments about body/appearance, requests for sexual favors.
- severe_harassment: Sustained pattern of hostile, intimidating behavior creating toxic environment.
- bullying: Repeated verbal abuse, public humiliation, deliberate undermining, mocking.

**VIOLENCE & THREATS:**
- workplace_violence: PHYSICAL acts - hitting, punching, pushing, grabbing that causes harm, assault, physical altercations.
- threat_of_violence: Verbal/written threats of physical harm - "I'll hurt you", "watch your back", "waiting in parking lot".
- coercive_threat: Using power to force compliance - "do X or I'll fire you", "do X or you'll fail", conditional threats.

**FINANCIAL & ETHICAL MISCONDUCT:**
- fraud: Fake expense reports, embezzlement, stealing money, falsified financial records, fake invoices.
- corruption: Kickbacks, steering contracts to family/friends, bribery, conflicts of interest in procurement/hiring.
- ethics_violation: Told to lie to customers, falsify reports, cover up problems, misrepresent products, deceptive practices.

**SAFETY & SECURITY:**
- safety_hazard: Blocked exits, faulty equipment, fire risks, dangerous conditions, OSHA violations.
- physical_safety_concern: Personal injury at work, hurt back, fell from ladder, unsafe working conditions causing harm.
- data_breach: Customer data exposed, passwords leaked, hacking incidents, unauthorized data access.
- security_incident: Suspicious USB drives, malware, unauthorized system access, potential cyber threats.

**QUID PRO QUO & RETALIATION:**
- quid_pro_quo: Exchanging favors for advancement - "do X for promotion", "meet after hours for grade", implying benefits for personal favors.
- retaliation: Punishment for reporting concerns - excluded after complaint, demoted after HR report, "you'll regret going to HR".

**SUBSTANCE ABUSE:**
- substance_abuse_at_work: Drunk, high, intoxicated at work, using drugs on premises, impaired while working.

**MENTAL HEALTH:**
- mental_health_crisis: CRITICAL - suicidal thoughts, wanting to die, "end it all", self-harm ideation. ALWAYS severity: critical.
- mental_wellbeing_concern: Depression, anxiety, overwhelming stress, can't cope, burnout symptoms.

**WORKPLACE ISSUES:**
- pattern_of_unfair_treatment: Being singled out for different rules - "only I have to do X", "everyone else gets Y but not me". NOT about third parties or contracts.
- workload_burnout_risk: Extreme hours, constant overtime, unsustainable workload, denied help.
- management_concern: Poor leadership, arbitrary decisions, lack of transparency.
- interpersonal_conflict: Arguments with colleagues affecting work.
- professional_misconduct: Misusing company resources, running personal business at work.

**LANGUAGE:**
- profanity: Explicit swear words - fuck, shit, damn, bitch, ass, bastard, crap. Must contain ACTUAL profanity.
- inappropriate_language: Crude jokes, offensive non-sexual comments.
- suggestive_language: Sexual innuendo - "that's what she said", winking, double entendres.

---

CRITICAL CLASSIFICATION RULES:

1. **fraud** = MONEY/FINANCIAL deception (fake expenses, embezzlement, stealing)
2. **corruption** = CONFLICTS OF INTEREST (kickbacks, contracts to family, bribery)
3. **ethics_violation** = TOLD TO BE DISHONEST (lie to customers, falsify reports, cover-ups)
4. **workplace_violence** = PHYSICAL ACTS ONLY (hitting, assault). NOT lying or unethical behavior.
5. **physical_safety_concern** = BODILY INJURY/UNSAFE CONDITIONS. NOT financial issues.
6. **pattern_of_unfair_treatment** = PERSONAL treatment ("I am singled out"). NOT about third parties.
7. **profanity** = Must contain ACTUAL swear words. "buggy software" is NOT profanity.
8. **mental_health_crisis** = ALWAYS severity: critical

---

EXAMPLES:

Comment: "My manager has been submitting fake expense reports for trips he never took"
{{"has_alerts": true, "alerts": [{{"excerpt": "submitting fake expense reports for trips he never took", "reasoning": "Financial fraud through falsified expense claims", "alert_type": "fraud", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "The procurement guy is steering all contracts to his brother's company"
{{"has_alerts": true, "alerts": [{{"excerpt": "steering all contracts to his brother's company", "reasoning": "Corruption through conflict of interest in procurement", "alert_type": "corruption", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "We're being told to lie to customers about the product capabilities to close sales"
{{"has_alerts": true, "alerts": [{{"excerpt": "told to lie to customers about the product capabilities", "reasoning": "Ethics violation - instructed to deceive customers", "alert_type": "ethics_violation", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "He called me a fucking chink in front of everyone"
{{"has_alerts": true, "alerts": [{{"excerpt": "called me a fucking chink", "reasoning": "Racial slur and ethnic discrimination", "alert_type": "discrimination", "severity": "high"}}, {{"excerpt": "fucking", "reasoning": "Contains profanity", "alert_type": "profanity", "severity": "low"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I don't see the point anymore, I want to end it all"
{{"has_alerts": true, "alerts": [{{"excerpt": "don't see the point anymore, I want to end it all", "reasoning": "Suicidal ideation requiring immediate intervention", "alert_type": "mental_health_crisis", "severity": "critical"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "The new software update is buggy and crashes frequently"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "quality_complaint", "non_alert_reasoning": "Technical feedback about software quality, no profanity or serious concerns"}}

Comment: "The project timeline seems aggressive given our current resources"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "workload_feedback", "non_alert_reasoning": "Feedback about project timeline and resourcing"}}

Comment: "The training was fantastic! The instructor really knew their stuff"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "positive_feedback", "non_alert_reasoning": "Positive feedback about training quality"}}

---

SEVERITY GUIDE:
- critical: Immediate danger - violence, suicide risk, ongoing assault
- high: Serious violations - discrimination, harassment, fraud, threats, data breach, corruption
- moderate: Concerning - unfair treatment, substance abuse, mental wellbeing, coercion
- low: Minor - profanity, suggestive language, interpersonal conflicts

Analyze the comment and return ONLY valid JSON."""


# =============================================================================
# Training Data Conversion
# =============================================================================
def result_to_training_output(result: AlertsOutput) -> dict:
    """Convert a single AlertsOutput to the training output format."""
    return {
        "has_alerts": result.has_alerts,
        "alerts": [
            {
                "excerpt": alert.excerpt,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "reasoning": alert.reasoning,
            }
            for alert in result.alerts
        ],
        "non_alert_classification": result.non_alert_classification,
        "non_alert_reasoning": result.non_alert_reasoning,
    }


def create_training_pairs(
    texts: List[str],
    results: List[AlertsOutput],
    source_column: str = None,
    metadata: List[dict] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Convert classification results to fine-tuning training pairs.

    Returns:
        Tuple of (training_pairs, failed_pairs)
    """
    training_pairs = []
    failed_pairs = []

    for idx, (text, result) in enumerate(zip(texts, results)):
        text = str(text).strip()

        if not text or text.lower() == "nan":
            continue

        base_record = {
            "id": idx,
            "input": text,
            "source_column": source_column,
        }

        # Add metadata if provided
        if metadata and idx < len(metadata):
            base_record["metadata"] = metadata[idx]

        if result is None:
            failed_pairs.append(
                {
                    **base_record,
                    "error": "PARSE_FAILED",
                }
            )
        else:
            output = result_to_training_output(result)
            training_pairs.append(
                {
                    **base_record,
                    "output": json.dumps(output, ensure_ascii=False),
                    "output_parsed": output,  # Keep parsed version for QA
                }
            )

    return training_pairs, failed_pairs


# =============================================================================
# Data Loading
# =============================================================================
def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load dataframe from various formats."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif ext in [".pkl", ".pickle"]:
        return pd.read_pickle(file_path)
    elif ext == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# =============================================================================
# Saving Functions
# =============================================================================
def save_training_data(
    training_pairs: List[dict],
    failed_pairs: List[dict],
    output_dir: str,
    prefix: str = "alerts_training",
):
    """
    Save training data in multiple formats:
    1. JSONL for training (input/output pairs only)
    2. JSON for QA review (full records with metadata)
    3. CSV for easy spreadsheet review
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. JSONL for training (minimal format)
    jsonl_path = output_path / f"{prefix}_{timestamp}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for pair in training_pairs:
            training_record = {
                "input": pair["input"],
                "output": pair["output"],
            }
            f.write(json.dumps(training_record, ensure_ascii=False) + "\n")
    print(f"✅ Training JSONL saved: {jsonl_path}")

    # 2. Full JSON for QA (with metadata and parsed output)
    qa_json_path = output_path / f"{prefix}_qa_{timestamp}.json"
    with open(qa_json_path, "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
    print(f"✅ QA JSON saved: {qa_json_path}")

    # 3. CSV for spreadsheet review
    csv_rows = []
    for pair in training_pairs:
        output_parsed = pair.get("output_parsed", {})

        # Flatten alerts for CSV
        if output_parsed.get("has_alerts") and output_parsed.get("alerts"):
            alert_types = ", ".join([a["alert_type"] for a in output_parsed["alerts"]])
            severities = ", ".join([a["severity"] for a in output_parsed["alerts"]])
            excerpts = " | ".join([a["excerpt"] for a in output_parsed["alerts"]])
            num_alerts = len(output_parsed["alerts"])
        else:
            alert_types = ""
            severities = ""
            excerpts = ""
            num_alerts = 0

        csv_rows.append(
            {
                "id": pair["id"],
                "input": pair["input"][:500],  # Truncate for readability
                "has_alerts": output_parsed.get("has_alerts"),
                "num_alerts": num_alerts,
                "alert_types": alert_types,
                "severities": severities,
                "excerpts": (excerpts or "")[:300],
                "non_alert_classification": output_parsed.get("non_alert_classification"),
                "non_alert_reasoning": (output_parsed.get("non_alert_reasoning") or "")[:200],
                "source_column": pair.get("source_column"),
                "qa_status": "",  # Empty column for manual QA
                "qa_notes": "",  # Empty column for QA notes
            }
        )

    csv_path = output_path / f"{prefix}_qa_{timestamp}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"✅ QA CSV saved: {csv_path}")

    # 4. Save failed pairs for review
    if failed_pairs:
        failed_path = output_path / f"{prefix}_failed_{timestamp}.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_pairs, f, indent=2, ensure_ascii=False)
        print(f"⚠️  Failed pairs saved: {failed_path}")

    return {
        "jsonl": str(jsonl_path),
        "qa_json": str(qa_json_path),
        "qa_csv": str(csv_path),
        "failed": str(output_path / f"{prefix}_failed_{timestamp}.json")
        if failed_pairs
        else None,
    }


def print_summary(training_pairs: List[dict], failed_pairs: List[dict]):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)

    total = len(training_pairs) + len(failed_pairs)
    print(f"Total texts processed: {total}")
    print(f"✅ Successful: {len(training_pairs)} ({len(training_pairs) / total * 100:.1f}%)")
    print(f"❌ Failed: {len(failed_pairs)} ({len(failed_pairs) / total * 100:.1f}%)")

    if not training_pairs:
        return

    # Count alerts vs non-alerts
    alerts_count = sum(1 for p in training_pairs if p["output_parsed"]["has_alerts"])
    non_alerts_count = len(training_pairs) - alerts_count

    print("\n📊 Distribution:")
    print(f"   Alerts: {alerts_count} ({alerts_count / len(training_pairs) * 100:.1f}%)")
    print(
        f"   Non-alerts: {non_alerts_count} ({non_alerts_count / len(training_pairs) * 100:.1f}%)"
    )

    # Alert type distribution
    alert_types = Counter()
    severities = Counter()
    non_alert_types = Counter()

    for pair in training_pairs:
        output = pair["output_parsed"]
        if output["has_alerts"]:
            for alert in output["alerts"]:
                alert_types[alert["alert_type"]] += 1
                severities[alert["severity"]] += 1
        elif output.get("non_alert_classification"):
            non_alert_types[output["non_alert_classification"]] += 1

    if alert_types:
        print("\n🚨 Alert Types (top 10):")
        for atype, count in alert_types.most_common(10):
            print(f"   {atype}: {count}")

        print("\n⚡ Severity Distribution:")
        for sev, count in severities.most_common():
            print(f"   {sev}: {count}")

    if non_alert_types:
        print("\n📋 Non-Alert Types:")
        for ntype, count in non_alert_types.most_common():
            print(f"   {ntype}: {count}")

    # Source column distribution
    source_counts = Counter(p.get("source_column") for p in training_pairs)
    if len(source_counts) > 1:
        print("\n📁 Source Columns:")
        for source, count in source_counts.most_common():
            print(f"   {source}: {count}")


# =============================================================================
# Main Pipeline
# =============================================================================
def run_classification_pipeline(
    input_path: str,
    text_columns: List[str],
    output_dir: str,
    gpu_list: List[int],
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 8192,
    multiplicity: int = 1,
):
    """
    Main pipeline to classify texts and prepare training data.
    """
    print("=" * 60)
    print("ALERTS CLASSIFICATION & TRAINING DATA PREPARATION")
    print("=" * 60)

    # Load data
    print(f"\n📂 Loading data from: {input_path}")
    df = load_dataframe(input_path)
    print(f"   Loaded {len(df)} rows")

    # Validate columns
    available_columns = [col for col in text_columns if col in df.columns]
    if not available_columns:
        raise ValueError(
            f"None of the specified columns {text_columns} found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"   Text columns to process: {available_columns}")

    # Collect all texts to classify
    all_texts = []
    text_sources = []  # Track which column each text came from
    text_indices = []  # Track original row index

    for col in available_columns:
        for idx, text in enumerate(df[col].tolist()):
            text_str = str(text).strip()
            if text_str and text_str.lower() != "nan":
                all_texts.append(text_str)
                text_sources.append(col)
                text_indices.append(idx)

    print(f"\n📝 Total texts to classify: {len(all_texts)}")

    # Initialize processor
    print(f"\n🔧 Initializing processor with {len(gpu_list)} GPUs...")
    processor = NewProcessor(
        gpu_list=gpu_list,
        llm="Qwen/Qwen2.5-32B-Instruct-AWQ",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        multiplicity=multiplicity,
    )

    try:
        # Generate prompts
        print("📝 Generating prompts...")
        prompts = [
            alert_detection_prompt(text) for text in tqdm(all_texts, desc="Building prompts")
        ]

        # Run classification
        print(f"\n🚀 Running classification on {len(prompts)} texts...")
        start_time = time.time()

        processor.process_with_schema(prompts=prompts, schema=AlertsOutput)
        results: List[AlertsOutput] = processor.parse_results_with_schema(schema=AlertsOutput)

        elapsed = time.time() - start_time
        print(
            f"⏱️  Classification completed in {elapsed:.2f}s ({len(prompts) / elapsed:.1f} texts/sec)"
        )

        # Convert to training pairs
        print("\n🔄 Converting to training format...")
        training_pairs, failed_pairs = create_training_pairs(
            texts=all_texts,
            results=results,
            source_column=None,  # Will add per-record
        )

        # Add source column info
        for i, pair in enumerate(training_pairs):
            if i < len(text_sources):
                pair["source_column"] = text_sources[i]
                pair["original_row_idx"] = text_indices[i]

        for i, pair in enumerate(failed_pairs):
            if i < len(text_sources):
                pair["source_column"] = text_sources[i]
                pair["original_row_idx"] = text_indices[i]

        # Print summary
        print_summary(training_pairs, failed_pairs)

        # Save all outputs
        print("\n💾 Saving training data...")
        saved_paths = save_training_data(
            training_pairs=training_pairs,
            failed_pairs=failed_pairs,
            output_dir=output_dir,
            prefix="alerts_training",
        )

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print(f"1. Review the QA CSV: {saved_paths['qa_csv']}")
        print("2. Mark incorrect classifications in 'qa_status' column")
        print(f"3. After QA, use the JSONL for fine-tuning: {saved_paths['jsonl']}")

        return training_pairs, failed_pairs, saved_paths

    finally:
        processor.terminate()


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    training_pairs, failed_pairs, saved_paths = run_classification_pipeline(
        input_path=CONFIG["input_dataframe_path"],
        text_columns=CONFIG["text_columns"],
        output_dir=CONFIG["output_dir"],
        gpu_list=CONFIG["gpu_list"],
        gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
        max_model_len=CONFIG["max_model_len"],
        multiplicity=CONFIG["multiplicity"],
    )
