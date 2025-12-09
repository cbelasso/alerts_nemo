"""
Example Usage: FlexibleSchemaProcessor with Mistral Models

This file demonstrates how to use the improved FlexibleSchemaProcessor
with both standard models and Mistral models.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
import time
from typing import List, Literal, Optional

from llm_parallelization.new_processor import NewProcessor
import pandas as pd
from pydantic import BaseModel


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


def alert_detection_prompt(text: str) -> str:
    """Generate optimized prompt for 8B model with clear category definitions."""
    return f"""[INST] You are an expert workplace safety and compliance analyzer. Analyze this employee comment for alerts requiring HR, management, or compliance attention.

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

Comment: "The CFO is taking kickbacks from vendors"
{{"has_alerts": true, "alerts": [{{"excerpt": "taking kickbacks from vendors", "reasoning": "Corruption through accepting bribes from vendors", "alert_type": "corruption", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "They're falsifying the safety inspection reports"
{{"has_alerts": true, "alerts": [{{"excerpt": "falsifying the safety inspection reports", "reasoning": "Ethics violation - falsifying official safety documents", "alert_type": "ethics_violation", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "He punched me in the face during the meeting"
{{"has_alerts": true, "alerts": [{{"excerpt": "punched me in the face", "reasoning": "Physical violence - assault in workplace", "alert_type": "workplace_violence", "severity": "critical"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I hurt my back lifting heavy boxes without proper equipment"
{{"has_alerts": true, "alerts": [{{"excerpt": "hurt my back lifting heavy boxes without proper equipment", "reasoning": "Physical injury from unsafe working conditions", "alert_type": "physical_safety_concern", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I'm the only one who has to get approval for expenses under $50"
{{"has_alerts": true, "alerts": [{{"excerpt": "I'm the only one who has to get approval for expenses under $50", "reasoning": "Employee singled out for different rules than colleagues", "alert_type": "pattern_of_unfair_treatment", "severity": "moderate"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "He called me a fucking chink in front of everyone"
{{"has_alerts": true, "alerts": [{{"excerpt": "called me a fucking chink", "reasoning": "Racial slur and ethnic discrimination", "alert_type": "discrimination", "severity": "high"}}, {{"excerpt": "fucking", "reasoning": "Contains profanity", "alert_type": "profanity", "severity": "low"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "That's what she said! You know what I mean ;) wink wink"
{{"has_alerts": true, "alerts": [{{"excerpt": "That's what she said! You know what I mean ;) wink wink", "reasoning": "Sexual innuendo and suggestive language", "alert_type": "suggestive_language", "severity": "low"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "This fucking project is a nightmare"
{{"has_alerts": true, "alerts": [{{"excerpt": "fucking project", "reasoning": "Contains explicit profanity", "alert_type": "profanity", "severity": "low"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I don't see the point anymore, I want to end it all"
{{"has_alerts": true, "alerts": [{{"excerpt": "don't see the point anymore, I want to end it all", "reasoning": "Suicidal ideation requiring immediate intervention", "alert_type": "mental_health_crisis", "severity": "critical"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "She implied a promotion might happen faster if we spent private time together"
{{"has_alerts": true, "alerts": [{{"excerpt": "promotion might happen faster if we spent private time together", "reasoning": "Quid pro quo - implying advancement in exchange for personal relationship", "alert_type": "quid_pro_quo", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "You'll regret going to HR about this. Watch your back."
{{"has_alerts": true, "alerts": [{{"excerpt": "You'll regret going to HR about this. Watch your back", "reasoning": "Threatening retaliation for reporting to HR", "alert_type": "retaliation", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "The new software update is buggy and crashes frequently"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "quality_complaint", "non_alert_reasoning": "Technical feedback about software quality, no profanity or serious concerns"}}

Comment: "The project timeline seems aggressive given our current resources"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "workload_feedback", "non_alert_reasoning": "Feedback about project timeline and resourcing"}}

Comment: "It would be more efficient if we had a shared calendar for meeting room bookings"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "process_improvement", "non_alert_reasoning": "Constructive suggestion for process improvement"}}

Comment: "We really need a second printer on this floor"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "resource_request", "non_alert_reasoning": "Request for additional office equipment"}}

Comment: "The training was fantastic! The instructor really knew their stuff"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "positive_feedback", "non_alert_reasoning": "Positive feedback about training quality"}}

Comment: "My team lead never meets deadlines and it delays everyone else's work"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "performance_complaint", "non_alert_reasoning": "Complaint about colleague's performance, not a serious violation"}}

---

SEVERITY GUIDE:
- critical: Immediate danger - violence, suicide risk, ongoing assault
- high: Serious violations - discrimination, harassment, fraud, threats, data breach, corruption
- moderate: Concerning - unfair treatment, substance abuse, mental wellbeing, coercion
- low: Minor - profanity, suggestive language, interpersonal conflicts

Analyze the comment and return ONLY valid JSON. [/INST]"""


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


def main():
    df = pd.read_csv(
        "/home/clyde/workspace/alerts_detection_llama/scripts/generation/datasets/claude_text_comments.csv"
    )

    texts = df["comment"].to_list()

    # model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    # model = "mistralai/Ministral-3-14B-Instruct-2512"
    # model = "ministral/Ministral-3b-instruct"
    model = "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-nemo-awq-pure"
    model = "warshanks/Ministral-8B-Instruct-2410-AWQ"
    model = "casperhansen/mistral-nemo-instruct-2407-awq"
    model = "mistralai/Ministral-8B-Instruct-2410"
    model = "Qwen/Qwen2.5-7B-Instruct"

    processor = NewProcessor(
        gpu_list=[4, 5, 6, 7],  # Using GPUs 1 and 2
        llm=model,
        # tokenizer="mistralai/Mistral-Nemo-Instruct-2407",
        # tokenizer_mode="mistral",
        # config_format="mistral",
        # load_format="mistral",
        # skip_tokenizer_init=True,  # Don't try to load tokenizer externally
        tensor_parallel_size=1,  # Use 2 GPUs for tensor parallelism
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        # Batching
        # max_num_batched_tokens=16384,  # Limit tokens per batch
        # max_num_seqs=128,  # Limit sequences per batch
        multiplicity=1,
        enable_chunked_prefill=True,
    )

    # Important: For Mistral, format your prompts manually!
    prompts = [alert_detection_prompt(text) for text in texts]

    try:
        start_time = time.time()
        processor.process_with_schema(prompts=prompts, schema=AlertsOutput)

        results: List[AlertsOutput] = processor.parse_results_with_schema(schema=AlertsOutput)

        end_time = time.time()

        # Parse results
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
            output_path=f"/home/clyde/workspace/llama_alerts/{model}/golden_dataset_alerts_output.csv",
        )

        print(f"\n{'=' * 80}")
        print(f"{model} MODEL CLASSIFICATION RESULTS")
        print(f"{'=' * 80}")
        print(f"Processed {len(results)} responses")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"{'=' * 80}\n")

        print(f"✓ Successful: {len(successful_parses)} | ✗ Failed: {failed_parses}\n")

    finally:
        processor.terminate()


if __name__ == "__main__":
    main()
