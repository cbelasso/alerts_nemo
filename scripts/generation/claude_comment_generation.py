from itertools import product
import json
from pathlib import Path
import random
import time

import anthropic

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var


# ==================== DIVERSITY DIMENSIONS ====================

ALERT_TYPES = [
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
    "mental_wellbeing_concern",
    "physical_safety_concern",
    "profanity",
    "suggestive_language",
    "inappropriate_language",
]

NON_ALERT_TYPES = [
    "performance_complaint",
    "quality_complaint",
    "workload_feedback",
    "process_improvement",
    "resource_request",
    "general_dissatisfaction",
    "constructive_feedback",
    "positive_feedback",
    "neutral_comment",
]

INDUSTRIES = [
    "tech startup",
    "corporate finance",
    "healthcare hospital",
    "manufacturing factory",
    "retail store",
    "restaurant kitchen",
    "law firm",
    "construction site",
    "university",
    "government agency",
    "nonprofit organization",
    "call center",
    "warehouse logistics",
    "advertising agency",
    "pharmaceutical company",
]

ROLES = [
    "junior employee",
    "senior manager",
    "intern",
    "contractor",
    "executive",
    "HR representative",
    "team lead",
    "remote worker",
    "part-time employee",
    "new hire (first week)",
    "long-tenured employee (10+ years)",
]

TONES = [
    "angry venting",
    "calm and factual",
    "passive-aggressive",
    "scared and uncertain",
    "dismissive/minimizing",
    "casual/joking",
    "formal complaint",
    "desperate plea",
    "matter-of-fact",
    "sarcastic",
]

WRITING_STYLES = [
    "formal email",
    "casual slack message",
    "survey response",
    "text message style with typos",
    "run-on sentences",
    "bullet points",
    "very brief (under 20 words)",
    "detailed narrative (100+ words)",
    "ESL speaker patterns",
    "heavy slang/colloquialisms",
]

DIFFICULTY_LEVELS = [
    "obvious - clear and explicit",
    "subtle - requires reading between lines",
    "ambiguous - could go either way",
    "buried - hidden in normal content",
    "minimized - speaker downplays severity",
    "coded language - uses euphemisms",
]

DEMOGRAPHICS = [
    "young woman",
    "older man",
    "person of color",
    "LGBTQ+ individual",
    "person with disability",
    "immigrant worker",
    "pregnant employee",
    "religious minority",
    "veteran",
    "unspecified/neutral",
]


# ==================== GENERATION PROMPTS ====================


def build_generation_prompt(
    example_type: str,
    alert_types: list[str] | None,
    non_alert_type: str | None,
    industry: str,
    role: str,
    tone: str,
    writing_style: str,
    difficulty: str,
    demographic: str,
    include_mix: bool = False,
) -> str:
    """Build a prompt to generate a specific type of example."""

    if example_type == "alert":
        type_instruction = f"""Generate a workplace comment that contains the following alert type(s): {", ".join(alert_types)}
        
The comment should clearly (or subtly, depending on difficulty) demonstrate these issues."""

    elif example_type == "non_alert":
        type_instruction = f"""Generate a workplace comment that is NOT an alert - it should be classified as: {non_alert_type}

This should be a false-positive trap - it might sound concerning but is actually just normal workplace feedback/discussion.
Do NOT include any actual harassment, discrimination, threats, safety issues, or other alert-worthy content."""

    elif example_type == "mixed":
        type_instruction = f"""Generate a workplace comment that contains BOTH:
- Alert(s): {", ".join(alert_types)}  
- Non-alert content: regular workplace discussion

The alert content should be buried within or mixed with normal workplace commentary."""

    elif example_type == "hard_non_alert":
        type_instruction = f"""Generate a workplace comment that is a FALSE POSITIVE TRAP.

It should sound like it could be an alert but is actually benign. Examples:
- Technical jargon that sounds violent ("kill the process", "terminate the job")
- Metaphorical language ("this project is killing me")
- Discussing policy/training about harassment (not experiencing it)
- Past/resolved issues
- Dark humor about workload
- Quoting movies/shows
- Hypothetical scenarios from training
- Frustration without actual profanity

The comment should be classified as: {non_alert_type}"""

    elif example_type == "hard_alert":
        type_instruction = f"""Generate a workplace comment that is a FALSE NEGATIVE TRAP.

It contains a real alert ({", ".join(alert_types)}) but is easy to miss because:
- Speaker minimizes/dismisses severity ("probably nothing", "just joking", "I'm sure they didn't mean it")
- Uses passive voice or vague language
- Buries the issue in casual conversation
- Uses coded language or euphemisms
- Normalizes problematic behavior ("that's just how they are")
- Gaslighting language mixed in

The alert should be subtle but REAL - not imagined."""

    return f"""You are generating synthetic training data for a workplace alert detection system.

{type_instruction}

PARAMETERS:
- Industry/Setting: {industry}
- Speaker's Role: {role}
- Speaker's Tone: {tone}
- Writing Style: {writing_style}
- Difficulty Level: {difficulty}
- Speaker Demographic Context: {demographic}

RULES:
1. Write ONLY the comment text - no labels, no explanations, no quotes around it
2. Make it realistic - this should sound like something a real employee would write
3. Vary sentence structure and vocabulary
4. Include realistic details for the industry
5. Match the specified tone and writing style exactly
6. If demographic is specified, the comment may reference experiences related to that identity (if relevant to the alert type)
7. Do NOT be preachy or educational - this is raw employee feedback
8. Include typos/informal language if the writing style calls for it

Generate the comment now:"""


def generate_single_example(
    example_type: str,
    alert_types: list[str] | None = None,
    non_alert_type: str | None = None,
    industry: str = None,
    role: str = None,
    tone: str = None,
    writing_style: str = None,
    difficulty: str = None,
    demographic: str = None,
) -> dict:
    """Generate a single example with metadata."""

    # Randomize any unspecified parameters
    industry = industry or random.choice(INDUSTRIES)
    role = role or random.choice(ROLES)
    tone = tone or random.choice(TONES)
    writing_style = writing_style or random.choice(WRITING_STYLES)
    difficulty = difficulty or random.choice(DIFFICULTY_LEVELS)
    demographic = demographic or random.choice(DEMOGRAPHICS)

    if example_type in ["alert", "mixed", "hard_alert"] and not alert_types:
        # Random 1-3 alert types for mixed, 1 for single
        num_alerts = random.randint(1, 3) if example_type == "mixed" else 1
        alert_types = random.sample(ALERT_TYPES, num_alerts)

    if example_type in ["non_alert", "hard_non_alert", "mixed"] and not non_alert_type:
        non_alert_type = random.choice(NON_ALERT_TYPES)

    prompt = build_generation_prompt(
        example_type=example_type,
        alert_types=alert_types,
        non_alert_type=non_alert_type,
        industry=industry,
        role=role,
        tone=tone,
        writing_style=writing_style,
        difficulty=difficulty,
        demographic=demographic,
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    generated_text = response.content[0].text.strip()

    return {
        "text": generated_text,
        "metadata": {
            "example_type": example_type,
            "expected_alert_types": alert_types,
            "expected_non_alert_type": non_alert_type,
            "industry": industry,
            "role": role,
            "tone": tone,
            "writing_style": writing_style,
            "difficulty": difficulty,
            "demographic": demographic,
        },
    }


def generate_diverse_dataset(
    n_examples: int,
    output_path: str = "generated_examples.jsonl",
    distribution: dict = None,
) -> list[dict]:
    """
    Generate a diverse dataset with specified distribution.

    Args:
        n_examples: Total number of examples to generate
        output_path: Path to save JSONL output
        distribution: Dict specifying proportion of each type, e.g.:
            {
                "alert": 0.35,
                "non_alert": 0.25,
                "mixed": 0.15,
                "hard_non_alert": 0.15,
                "hard_alert": 0.10,
            }
    """
    if distribution is None:
        distribution = {
            "alert": 0.35,
            "non_alert": 0.25,
            "mixed": 0.15,
            "hard_non_alert": 0.15,
            "hard_alert": 0.10,
        }

    # Calculate counts per type
    type_counts = {t: int(n_examples * p) for t, p in distribution.items()}

    # Adjust for rounding
    total = sum(type_counts.values())
    if total < n_examples:
        type_counts["alert"] += n_examples - total

    # Build generation queue with systematic coverage
    generation_queue = []

    for example_type, count in type_counts.items():
        for i in range(count):
            # Systematically vary parameters to ensure coverage
            params = {
                "example_type": example_type,
                "industry": INDUSTRIES[i % len(INDUSTRIES)],
                "role": ROLES[i % len(ROLES)],
                "tone": TONES[i % len(TONES)],
                "writing_style": WRITING_STYLES[i % len(WRITING_STYLES)],
                "difficulty": DIFFICULTY_LEVELS[i % len(DIFFICULTY_LEVELS)],
                "demographic": DEMOGRAPHICS[i % len(DEMOGRAPHICS)],
            }

            # For alerts, cycle through alert types
            if example_type in ["alert", "hard_alert"]:
                params["alert_types"] = [ALERT_TYPES[i % len(ALERT_TYPES)]]
            elif example_type == "mixed":
                # 1-3 random alert types for mixed
                num_alerts = (i % 3) + 1
                start_idx = i % len(ALERT_TYPES)
                params["alert_types"] = [
                    ALERT_TYPES[(start_idx + j) % len(ALERT_TYPES)] for j in range(num_alerts)
                ]

            # For non-alerts, cycle through non-alert types
            if example_type in ["non_alert", "hard_non_alert"]:
                params["non_alert_type"] = NON_ALERT_TYPES[i % len(NON_ALERT_TYPES)]
            elif example_type == "mixed":
                params["non_alert_type"] = NON_ALERT_TYPES[i % len(NON_ALERT_TYPES)]

            generation_queue.append(params)

    # Shuffle for randomness
    random.shuffle(generation_queue)

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Generate examples
    examples = []

    with open(output_path, "w") as f:
        for i, params in enumerate(generation_queue):
            try:
                example = generate_single_example(**params)
                examples.append(example)

                # Write immediately (streaming)
                f.write(json.dumps(example) + "\n")
                f.flush()

                print(
                    f"[{i + 1}/{n_examples}] Generated {params['example_type']}: {example['text'][:60]}..."
                )

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"[{i + 1}/{n_examples}] ERROR: {e}")
                continue

    print(f"\n✅ Generated {len(examples)} examples")
    print(f"📁 Saved to: {output_path}")

    return examples


def generate_targeted_examples(
    alert_type: str = None,
    non_alert_type: str = None,
    n_examples: int = 10,
    difficulty: str = None,
    output_path: str = None,
) -> list[dict]:
    """Generate examples targeting a specific alert or non-alert type."""

    if alert_type:
        example_type = "hard_alert" if difficulty == "subtle" else "alert"
        params = {"alert_types": [alert_type]}
    else:
        example_type = "hard_non_alert" if difficulty == "subtle" else "non_alert"
        params = {"non_alert_type": non_alert_type}

    if output_path is None:
        target = alert_type or non_alert_type
        output_path = f"generated_{target}_{n_examples}.jsonl"

    examples = []

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i in range(n_examples):
            try:
                example = generate_single_example(
                    example_type=example_type,
                    difficulty=difficulty,
                    **params,
                )
                examples.append(example)
                f.write(json.dumps(example) + "\n")
                f.flush()

                print(f"[{i + 1}/{n_examples}] {example['text'][:80]}...")
                time.sleep(0.5)

            except Exception as e:
                print(f"[{i + 1}/{n_examples}] ERROR: {e}")

    return examples


# ==================== COVERAGE REPORT ====================


def check_coverage(examples: list[dict]) -> dict:
    """Analyze coverage across all diversity dimensions."""
    from collections import Counter

    stats = {
        "total": len(examples),
        "example_types": Counter(),
        "alert_types": Counter(),
        "non_alert_types": Counter(),
        "industries": Counter(),
        "roles": Counter(),
        "tones": Counter(),
        "writing_styles": Counter(),
        "difficulties": Counter(),
        "demographics": Counter(),
    }

    for ex in examples:
        meta = ex["metadata"]
        stats["example_types"][meta["example_type"]] += 1
        stats["industries"][meta["industry"]] += 1
        stats["roles"][meta["role"]] += 1
        stats["tones"][meta["tone"]] += 1
        stats["writing_styles"][meta["writing_style"]] += 1
        stats["difficulties"][meta["difficulty"]] += 1
        stats["demographics"][meta["demographic"]] += 1

        if meta["expected_alert_types"]:
            for at in meta["expected_alert_types"]:
                stats["alert_types"][at] += 1
        if meta["expected_non_alert_type"]:
            stats["non_alert_types"][meta["expected_non_alert_type"]] += 1

    # Find gaps
    stats["missing_alert_types"] = set(ALERT_TYPES) - set(stats["alert_types"].keys())
    stats["missing_non_alert_types"] = set(NON_ALERT_TYPES) - set(
        stats["non_alert_types"].keys()
    )

    return stats


def print_coverage_report(stats: dict):
    """Print a formatted coverage report."""
    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)

    print(f"\nTotal examples: {stats['total']}")

    print("\n📊 Example Types:")
    for t, c in stats["example_types"].most_common():
        print(f"  {t}: {c} ({c / stats['total'] * 100:.1f}%)")

    print("\n🚨 Alert Types:")
    for t, c in stats["alert_types"].most_common():
        print(f"  {t}: {c}")
    if stats["missing_alert_types"]:
        print(f"  ⚠️ Missing: {stats['missing_alert_types']}")

    print("\n📋 Non-Alert Types:")
    for t, c in stats["non_alert_types"].most_common():
        print(f"  {t}: {c}")
    if stats["missing_non_alert_types"]:
        print(f"  ⚠️ Missing: {stats['missing_non_alert_types']}")

    print("\n🏭 Industries:")
    for t, c in stats["industries"].most_common(5):
        print(f"  {t}: {c}")

    print("\n🎭 Tones:")
    for t, c in stats["tones"].most_common(5):
        print(f"  {t}: {c}")


# ==================== MAIN ====================

if __name__ == "__main__":
    # Generate diverse dataset
    examples = generate_diverse_dataset(
        n_examples=10,  # Adjust as needed
        output_path="data/generated_examples.jsonl",
    )

    # Check coverage
    stats = check_coverage(examples)
    print_coverage_report(stats)

    # Fill gaps if needed
    for missing_type in stats["missing_alert_types"]:
        print(f"\n🔧 Generating examples for missing alert type: {missing_type}")
        generate_targeted_examples(
            alert_type=missing_type,
            n_examples=5,
            output_path=f"data/gap_fill_{missing_type}.jsonl",
        )
