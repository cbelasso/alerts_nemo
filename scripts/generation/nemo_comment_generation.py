from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
import time
from typing import List, Literal, Optional

from llm_parallelization.parallelization import (
    NEMO,
    FlexibleSchemaProcessor,
)
import pandas as pd
from pydantic import BaseModel


# -------------------------------
# Pydantic models for generation
# -------------------------------
class GeneratedExample(BaseModel):
    text: str


# -------------------------------
# Diversity dimensions
# -------------------------------
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


# -------------------------------
# Example texts for few-shot prompting
# -------------------------------
ALERT_EXAMPLES = {
    "discrimination": [
        "My supervisor keeps making comments about how women shouldn't be in engineering roles",
        "They said I wasn't the right 'culture fit' for the leadership team, funny how no one who looks like me ever is",
        "The whole team calls him 'the retard' behind his back, mimics his stutter in meetings, and management thinks it's hilarious",
    ],
    "sexual_harassment": [
        "He keeps staring at my chest during meetings and made a comment about my legs yesterday",
        "He grabs my waist every time he walks by, calls me his 'little slut' when no one's around",
        "He just brushed against me a few times, probably an accident, though it keeps happening when we're alone",
    ],
    "severe_harassment": [
        "They created a group chat specifically to mock everything I do and say at work",
        "My manager publicly humiliates me daily, and when I tried to document it, someone deleted all my emails",
    ],
    "bullying": [
        "Every single day she finds something to criticize me about in front of the whole team",
        "Everyone keeps telling me I'm too sensitive and imagining things but I know he treats me differently than everyone else",
    ],
    "workplace_violence": [
        "He slammed me against the wall in the break room when no one was looking",
        "Dave came in high as a kite again, then that asshole threw a chair at me when I told him to go home",
        "He grabbed my arm pretty hard during the argument but I'm sure he didn't mean anything by it",
    ],
    "threat_of_violence": [
        "He said if I report him he'll be waiting for me in the parking lot",
        "He smiled and said he hopes nothing 'unfortunate' happens to my car while I'm thinking about whether to report him",
        "He told me if I go to HR about the n-word comments, he knows where I park and things could get ugly for me",
    ],
    "coercive_threat": [
        "She told me if I don't work this weekend unpaid, she'll give me a bad performance review",
        "Haha so my manager jokingly said he'd tank my review if I don't come to his cabin this weekend, classic him right",
        "I'm the only one required to work holidays, and when I asked why, my boss said questioning him would affect my bonus",
    ],
    "safety_hazard": [
        "The emergency exits have been blocked with boxes for weeks and nobody cares",
        "Half the warehouse crew is on meth, operating forklifts while tweaking out. This shithole is a disaster waiting to happen",
        "They're falsifying the safety inspection reports and selling equipment they know is defective - someone is going to get killed",
    ],
    "retaliation": [
        "Ever since I filed that complaint, I've been excluded from all important meetings",
        "You'll regret going to HR about this. Watch your back.",
        "Ever since I asked about the pay gap, I've noticed I'm no longer CC'd on important emails, probably just an oversight though",
    ],
    "substance_abuse_at_work": [
        "My coworker keeps a flask in his desk and is clearly drunk by lunch every day",
        "Oh that's just how Dave is, everyone knows he has a few drinks at lunch, it's not a big deal",
        "They only hire white guys for management, the expense reports are completely made up, and the CEO does lines in his office with the door open",
    ],
    "data_breach": [
        "Someone left all the customer credit card info on an unsecured shared drive",
        "Found some weird access logs showing someone downloaded the entire customer database last Tuesday at 3am from an IP in Russia",
        "The CFO is taking kickbacks from vendors and I caught him downloading the entire customer database to a personal drive last week",
    ],
    "security_incident": [
        "I found a USB drive in the parking lot and plugged it into my work computer, now it's acting weird",
        "After I reported the security breach where client passwords were exposed, my boss said he'd make my life hell",
    ],
    "fraud": [
        "My manager has been submitting fake expense reports for trips he never took",
        "They only hire white guys for management, the expense reports are completely made up",
    ],
    "corruption": [
        "The procurement guy is steering all contracts to his brother's company",
        "The CFO is taking kickbacks from vendors",
    ],
    "quid_pro_quo": [
        "She implied that a promotion might happen faster if we spent some private time together",
        "She keeps mentioning how she 'takes care of people who take care of her' whenever we discuss my promotion",
        "The director told me Asian women are more 'submissive' and that if I was nicer to him after work, he could fast-track my visa sponsorship",
    ],
    "ethics_violation": [
        "We're being told to lie to customers about the product capabilities to close sales",
        "They're falsifying the safety inspection reports",
        "My manager publicly humiliates me daily, and when I tried to document it, someone deleted all my emails. HR said to drop it or else",
    ],
    "mental_health_crisis": [
        "I don't see the point anymore. Everything feels hopeless and I've been thinking about not waking up",
        "They jumped him in the warehouse because he's gay, beat him pretty bad. He texted me saying he can't take it anymore and wants to end things",
        "I've been giving away my things at work, tying up loose ends, making sure my projects are documented in case I'm not around",
    ],
    "pattern_of_unfair_treatment": [
        "I'm the only one who has to get approval for expenses under $50 while everyone else just submits them",
        "He grabs my waist every time he walks by, calls me his 'little slut' when no one's around, and I'm the only one who never gets approved for PTO",
    ],
    "workload_burnout_risk": [
        "I've worked 80 hour weeks for three months straight and my requests for help keep getting denied",
        "I'm fucking exhausted, haven't seen my kids in weeks because of these deadlines, and honestly I just don't care about anything anymore",
    ],
    "mental_wellbeing_concern": [
        "I haven't been sleeping well and I feel overwhelmed all the time lately",
        "I've been really struggling with anxiety since the reorg, but I think having clearer role definitions would help everyone, not just me",
    ],
    "physical_safety_concern": [
        "The ladder they gave me is broken and I almost fell from the second floor yesterday",
        "I reported my injury from the faulty equipment and now they're threatening to fire me if I file workers comp",
        "That psycho bitch stabbed me with scissors, there's blood everywhere, and I can't feel my fucking hand",
    ],
    "profanity": [
        "This goddamn system crashes every five minutes and I'm sick of this bullshit",
        "Holy shit this new system is fucking amazing, best upgrade we've ever had, the IT team killed it",
    ],
    "suggestive_language": [
        "He always comments on how I look and says things like 'I bet you're fun outside of work'",
        "That's what she said! You know what I mean ;) wink wink",
    ],
    "inappropriate_language": [
        "He keeps telling really crude jokes that make everyone uncomfortable",
    ],
}

NON_ALERT_EXAMPLES = {
    "performance_complaint": [
        "My team lead never meets deadlines and it delays everyone else's work",
        "The professor doesn't seem to care about his job, he just marks the way he feels like",
    ],
    "quality_complaint": [
        "The new software update is buggy and crashes frequently",
        "I don't think this course was for me. It was really boring and repetitive. The instructor could've been more engaging",
    ],
    "workload_feedback": [
        "The project timeline seems aggressive given our current resources",
    ],
    "process_improvement": [
        "It would be more efficient if we had a shared calendar for meeting room bookings",
        "The onboarding process could be improved by adding a mentor system for new hires",
        "I think we should add more hands-on exercises to the workshop. It would help people retain the information better.",
    ],
    "resource_request": [
        "We really need a second printer on this floor to reduce wait times",
    ],
    "general_dissatisfaction": [
        "Things just aren't the same since the reorganization",
        "It is what it is I guess",
        "I don't like coming on Monday mornings",
    ],
    "constructive_feedback": [
        "The onboarding process could be improved by adding a mentor system for new hires",
    ],
    "positive_feedback": [
        "The new flexible work policy has really improved my work-life balance",
        "The training was fantastic! The instructor really knew their stuff and made complex topics easy to understand.",
        "Holy shit this new system is fucking amazing, best upgrade we've ever had, the IT team killed it",
    ],
    "neutral_comment": [
        "The quarterly meeting is scheduled for next Tuesday at 3pm",
    ],
}

HARD_NON_ALERT_EXAMPLES = [
    "We need to kill the zombie processes, terminate the hanging threads, and abort the failed jobs before we can restart the server",
    "This project is killing me, the deadline is murder, and I'm dying to get it finished",
    "Last year we had a manager who was really inappropriate but he was fired and HR handled it well",
    "My coworker keeps saying 'that's what she said' like Michael Scott from The Office, it's pretty funny actually",
    "The new harassment training explained that quid pro quo means exchanging favors for promotions, which is obviously wrong",
    "In the ethics workshop, they asked what we'd do if a manager offered a promotion in exchange for a date",
    "This freaking system is driving me nuts, I'm so done with this dang thing crashing",
    "Our sales team is going to crush the competition this quarter, we're absolutely destroying our targets",
    "I'm such an idiot, I deleted my own presentation five minutes before the meeting",
    "I've been seeing a therapist for my anxiety and it's really helping with work stress",
    "In the conflict resolution training, I played the angry customer who yells and threatens to sue",
    "Did you hear about that data breach at the other company? Glad our security team is on top of things",
    "Sarah mentioned she felt uncomfortable last month but talked to her manager and it's all sorted now",
    "Deploy on Friday? Sure, I too like to live dangerously. What could possibly go wrong?",
    "The client is absolutely insane, their demands are ridiculous and they yell at us on every call",
    "She's a real gun!",
    "It's like drinking from a fire hose",
]


# -------------------------------
# Prompt builders
# -------------------------------
def build_alert_generation_prompt(
    alert_types: List[str],
    industry: str,
    role: str,
    tone: str,
    writing_style: str,
    difficulty: str,
    demographic: str,
) -> str:
    examples = []
    for at in alert_types:
        if at in ALERT_EXAMPLES:
            examples.extend(random.sample(ALERT_EXAMPLES[at], min(2, len(ALERT_EXAMPLES[at]))))

    examples_str = "\n".join([f'- "{ex}"' for ex in examples[:5]])

    return f"""You are generating synthetic training data for a workplace alert detection system.

Generate a realistic workplace comment that contains the following alert type(s): {", ".join(alert_types)}

PARAMETERS:
- Industry/Setting: {industry}
- Speaker's Role: {role}
- Speaker's Tone: {tone}
- Writing Style: {writing_style}
- Difficulty Level: {difficulty}
- Speaker Demographic Context: {demographic}

SIMILAR EXAMPLES FOR REFERENCE:
{examples_str}

ALERT TYPE DEFINITIONS:
- discrimination: Unfair treatment based on race, gender, age, religion, disability, ethnicity, nationality. Includes slurs.
- sexual_harassment: Unwanted sexual advances, inappropriate touching, sexual comments about body/appearance.
- severe_harassment: Sustained pattern of hostile, intimidating behavior creating toxic environment.
- bullying: Repeated verbal abuse, public humiliation, deliberate undermining, mocking.
- workplace_violence: PHYSICAL acts - hitting, punching, pushing, grabbing that causes harm, assault.
- threat_of_violence: Verbal/written threats of physical harm.
- coercive_threat: Using power to force compliance - "do X or I'll fire you".
- safety_hazard: Blocked exits, faulty equipment, fire risks, dangerous conditions.
- retaliation: Punishment for reporting concerns - excluded after complaint, demoted after HR report.
- substance_abuse_at_work: Drunk, high, intoxicated at work, using drugs on premises.
- data_breach: Customer data exposed, passwords leaked, hacking incidents.
- security_incident: Suspicious USB drives, malware, unauthorized system access.
- fraud: Fake expense reports, embezzlement, stealing money, falsified financial records.
- corruption: Kickbacks, steering contracts to family/friends, bribery.
- quid_pro_quo: Exchanging favors for advancement.
- ethics_violation: Told to lie to customers, falsify reports, cover up problems.
- mental_health_crisis: Suicidal thoughts, wanting to die, self-harm ideation.
- pattern_of_unfair_treatment: Being singled out for different rules.
- workload_burnout_risk: Extreme hours, constant overtime, unsustainable workload.
- mental_wellbeing_concern: Depression, anxiety, overwhelming stress, burnout symptoms.
- physical_safety_concern: Personal injury at work, unsafe working conditions.
- profanity: Explicit swear words - fuck, shit, damn, bitch, ass.
- suggestive_language: Sexual innuendo, winking, double entendres.
- inappropriate_language: Crude jokes, offensive non-sexual comments.

RULES:
1. Write ONLY the comment text - no labels, no explanations, no quotes
2. Make it realistic - sound like something a real employee would write
3. Match the specified tone and writing style exactly
4. If demographic is specified, reference experiences related to that identity if relevant
5. Do NOT be preachy or educational - this is raw employee feedback
6. Include typos/informal language if the writing style calls for it
7. The comment should clearly demonstrate the alert type(s)

Generate the workplace comment now:"""


def build_non_alert_generation_prompt(
    non_alert_type: str,
    industry: str,
    role: str,
    tone: str,
    writing_style: str,
    demographic: str,
) -> str:
    examples = NON_ALERT_EXAMPLES.get(non_alert_type, [])
    examples_str = "\n".join([f'- "{ex}"' for ex in examples[:3]])

    return f"""You are generating synthetic training data for a workplace alert detection system.

Generate a realistic workplace comment that is NOT an alert - it should be classified as: {non_alert_type}

This should be normal workplace feedback/discussion with NO harassment, discrimination, threats, safety issues, or other alert-worthy content.

PARAMETERS:
- Industry/Setting: {industry}
- Speaker's Role: {role}
- Speaker's Tone: {tone}
- Writing Style: {writing_style}
- Speaker Demographic Context: {demographic}

SIMILAR EXAMPLES FOR REFERENCE:
{examples_str}

NON-ALERT TYPE DEFINITIONS:
- performance_complaint: Complaint about colleague's job performance (not harassment)
- quality_complaint: Feedback about product/service quality issues
- workload_feedback: Comments about workload, deadlines, resources
- process_improvement: Suggestions for better processes/workflows
- resource_request: Requests for equipment, tools, training
- general_dissatisfaction: Vague complaints without specific serious issues
- constructive_feedback: Helpful suggestions for improvement
- positive_feedback: Praise, appreciation, positive comments
- neutral_comment: Factual statements, scheduling, logistics

RULES:
1. Write ONLY the comment text - no labels, no explanations, no quotes
2. Make it realistic - sound like something a real employee would write
3. Match the specified tone and writing style exactly
4. Do NOT include any actual harassment, discrimination, threats, or safety issues
5. This should be mundane workplace feedback that does NOT require HR attention

Generate the workplace comment now:"""


def build_hard_non_alert_generation_prompt(
    non_alert_type: str,
    industry: str,
    role: str,
    tone: str,
    writing_style: str,
    demographic: str,
) -> str:
    examples_str = "\n".join([f'- "{ex}"' for ex in random.sample(HARD_NON_ALERT_EXAMPLES, 5)])

    return f"""You are generating synthetic training data for a workplace alert detection system.

Generate a workplace comment that is a FALSE POSITIVE TRAP - it should SOUND concerning but is actually benign.

The comment should be classified as: {non_alert_type}

WAYS TO MAKE IT TRICKY (pick one or more):
- Technical jargon that sounds violent ("kill the process", "terminate the job", "abort the failed task")
- Metaphorical language ("this project is killing me", "deadline is murder")
- Past tense / resolved issues ("we HAD a problem but it's fixed now")
- Quoting movies/shows/training scenarios
- Discussing policy about harassment (not experiencing it)
- Hypothetical scenarios from workshops
- Frustration without actual profanity ("freaking", "dang", "heck")
- Gaming/competition language ("crush the competition", "destroy our targets")
- Self-deprecating humor ("I'm such an idiot")
- Medical/health discussion that's not a crisis
- Role-playing exercise descriptions
- External news discussion (not internal issues)
- Resolved complaints from others

PARAMETERS:
- Industry/Setting: {industry}
- Speaker's Role: {role}
- Speaker's Tone: {tone}
- Writing Style: {writing_style}
- Speaker Demographic Context: {demographic}

SIMILAR FALSE POSITIVE EXAMPLES:
{examples_str}

RULES:
1. Write ONLY the comment text - no labels, no explanations, no quotes
2. Make it sound potentially concerning at first glance
3. But it should actually be benign/normal workplace content
4. Do NOT include any ACTUAL harassment, discrimination, threats, or safety issues
5. The trick is in the language/context, not actual problems

Generate the false-positive trap workplace comment now:"""


def build_hard_alert_generation_prompt(
    alert_types: List[str],
    industry: str,
    role: str,
    tone: str,
    writing_style: str,
    difficulty: str,
    demographic: str,
) -> str:
    examples = []
    for at in alert_types:
        if at in ALERT_EXAMPLES:
            examples.extend(ALERT_EXAMPLES[at])

    subtle_examples = [
        ex
        for ex in examples
        if any(
            word in ex.lower()
            for word in [
                "probably",
                "maybe",
                "just",
                "sure",
                "accident",
                "joking",
                "classic",
                "oversight",
                "coincidence",
            ]
        )
    ]

    examples_str = "\n".join([f'- "{ex}"' for ex in subtle_examples[:5]])

    return f"""You are generating synthetic training data for a workplace alert detection system.

Generate a workplace comment that contains a REAL alert ({", ".join(alert_types)}) but is EASY TO MISS.

This is a FALSE NEGATIVE TRAP - the alert is real but subtle/hidden.

WAYS TO HIDE THE ALERT:
- Speaker minimizes/dismisses severity ("probably nothing", "just joking", "I'm sure they didn't mean it")
- Uses passive voice or vague language ("mistakes were made", "comments may have been directed")
- Buries the issue in casual conversation with other topics
- Uses coded language or euphemisms
- Normalizes problematic behavior ("that's just how they are", "everyone knows")
- Gaslighting language mixed in ("I'm probably too sensitive")
- Frames it positively ("Great news - I got demoted after reporting!")
- Adds "probably coincidence" or "might be nothing"

PARAMETERS:
- Industry/Setting: {industry}
- Speaker's Role: {role}
- Speaker's Tone: {tone} (should lean toward dismissive/minimizing)
- Writing Style: {writing_style}
- Difficulty Level: {difficulty}
- Speaker Demographic Context: {demographic}

SUBTLE ALERT EXAMPLES FOR REFERENCE:
{examples_str}

RULES:
1. Write ONLY the comment text - no labels, no explanations, no quotes
2. The alert must be REAL - not imagined by the speaker
3. But the speaker should downplay, minimize, or bury it
4. Mix in normal content to hide the concerning parts
5. Make it require careful reading to identify the issue

Generate the subtle/hidden alert workplace comment now:"""


def build_mixed_generation_prompt(
    alert_types: List[str],
    non_alert_type: str,
    industry: str,
    role: str,
    tone: str,
    writing_style: str,
    difficulty: str,
    demographic: str,
) -> str:
    alert_examples = []
    for at in alert_types:
        if at in ALERT_EXAMPLES:
            alert_examples.extend(
                random.sample(ALERT_EXAMPLES[at], min(1, len(ALERT_EXAMPLES[at])))
            )

    non_alert_examples = NON_ALERT_EXAMPLES.get(non_alert_type, [])

    return f"""You are generating synthetic training data for a workplace alert detection system.

Generate a LONGER workplace comment (3-6 sentences) that contains BOTH:
- Alert content: {", ".join(alert_types)}
- Non-alert content: normal {non_alert_type} type discussion

The alert should be MIXED IN with normal workplace commentary.

PARAMETERS:
- Industry/Setting: {industry}
- Speaker's Role: {role}
- Speaker's Tone: {tone}
- Writing Style: {writing_style}
- Difficulty Level: {difficulty}
- Speaker Demographic Context: {demographic}

EXAMPLE MIXED TEXTS:
- "The training was actually pretty good, but afterwards the instructor asked me to stay behind and said we should 'get to know each other better' if I want the certification"
- "Team meeting notes: Discussed Q3 roadmap (on track), budget approved. Side note - Mark made another joke about the 'diversity hire' comment. Action items assigned."
- "I love my job overall! Great benefits, smart colleagues. My only concern is that ever since I reported the expense irregularities, my workload has tripled. Probably coincidence!"

RULES:
1. Write ONLY the comment text - no labels, no explanations, no quotes
2. Make it 3-6 sentences long
3. Include BOTH alert-worthy content AND normal workplace discussion
4. The alert should be buried within or mixed with normal content
5. Make it realistic - like a real employee rambling about multiple things

Generate the mixed workplace comment now:"""


# -------------------------------
# Generation functions
# -------------------------------
def build_generation_queue(
    n_examples: int,
    distribution: dict = None,
) -> List[dict]:
    """Build a queue of generation tasks with diverse parameters."""

    if distribution is None:
        distribution = {
            "alert": 0.30,
            "non_alert": 0.25,
            "mixed": 0.15,
            "hard_non_alert": 0.15,
            "hard_alert": 0.15,
        }

    type_counts = {t: int(n_examples * p) for t, p in distribution.items()}
    total = sum(type_counts.values())
    if total < n_examples:
        type_counts["alert"] += n_examples - total

    queue = []

    for example_type, count in type_counts.items():
        for i in range(count):
            params = {
                "example_type": example_type,
                "industry": INDUSTRIES[i % len(INDUSTRIES)],
                "role": ROLES[i % len(ROLES)],
                "tone": TONES[i % len(TONES)],
                "writing_style": WRITING_STYLES[i % len(WRITING_STYLES)],
                "difficulty": DIFFICULTY_LEVELS[i % len(DIFFICULTY_LEVELS)],
                "demographic": DEMOGRAPHICS[i % len(DEMOGRAPHICS)],
            }

            if example_type in ["alert", "hard_alert"]:
                params["alert_types"] = [ALERT_TYPES[i % len(ALERT_TYPES)]]
            elif example_type == "mixed":
                num_alerts = (i % 3) + 1
                start_idx = i % len(ALERT_TYPES)
                params["alert_types"] = [
                    ALERT_TYPES[(start_idx + j) % len(ALERT_TYPES)] for j in range(num_alerts)
                ]

            if example_type in ["non_alert", "hard_non_alert"]:
                params["non_alert_type"] = NON_ALERT_TYPES[i % len(NON_ALERT_TYPES)]
            elif example_type == "mixed":
                params["non_alert_type"] = NON_ALERT_TYPES[i % len(NON_ALERT_TYPES)]

            queue.append(params)

    random.shuffle(queue)
    return queue


def build_prompt_for_task(task: dict) -> str:
    """Build the appropriate prompt for a generation task."""
    example_type = task["example_type"]

    if example_type == "alert":
        return build_alert_generation_prompt(
            alert_types=task["alert_types"],
            industry=task["industry"],
            role=task["role"],
            tone=task["tone"],
            writing_style=task["writing_style"],
            difficulty=task["difficulty"],
            demographic=task["demographic"],
        )
    elif example_type == "non_alert":
        return build_non_alert_generation_prompt(
            non_alert_type=task["non_alert_type"],
            industry=task["industry"],
            role=task["role"],
            tone=task["tone"],
            writing_style=task["writing_style"],
            demographic=task["demographic"],
        )
    elif example_type == "hard_non_alert":
        return build_hard_non_alert_generation_prompt(
            non_alert_type=task["non_alert_type"],
            industry=task["industry"],
            role=task["role"],
            tone=task["tone"],
            writing_style=task["writing_style"],
            demographic=task["demographic"],
        )
    elif example_type == "hard_alert":
        return build_hard_alert_generation_prompt(
            alert_types=task["alert_types"],
            industry=task["industry"],
            role=task["role"],
            tone=task["tone"],
            writing_style=task["writing_style"],
            difficulty=task["difficulty"],
            demographic=task["demographic"],
        )
    elif example_type == "mixed":
        return build_mixed_generation_prompt(
            alert_types=task["alert_types"],
            non_alert_type=task["non_alert_type"],
            industry=task["industry"],
            role=task["role"],
            tone=task["tone"],
            writing_style=task["writing_style"],
            difficulty=task["difficulty"],
            demographic=task["demographic"],
        )
    else:
        raise ValueError(f"Unknown example type: {example_type}")


def generate_examples(
    n_examples: int,
    processor: FlexibleSchemaProcessor,
    output_path: str = None,
    distribution: dict = None,
) -> List[dict]:
    """Generate diverse training examples using the provided processor."""

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"generated_examples_{timestamp}.jsonl"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    queue = build_generation_queue(n_examples, distribution)
    prompts = [build_prompt_for_task(task) for task in queue]

    print(f"🚀 Generating {n_examples} examples...")
    start_time = time.time()

    processor.process_with_schema(prompts=prompts, schema=GeneratedExample)
    results: List[GeneratedExample] = processor.parse_results_with_schema(
        schema=GeneratedExample
    )

    end_time = time.time()
    print(f"⏱️ Generation took {end_time - start_time:.2f} seconds")

    examples = []
    with open(output_path, "w") as f:
        for task, result in zip(queue, results):
            if result is None:
                continue

            example = {
                "text": result.text.strip().strip('"').strip("'"),
                "metadata": {
                    "example_type": task["example_type"],
                    "expected_alert_types": task.get("alert_types"),
                    "expected_non_alert_type": task.get("non_alert_type"),
                    "industry": task["industry"],
                    "role": task["role"],
                    "tone": task["tone"],
                    "writing_style": task["writing_style"],
                    "difficulty": task.get("difficulty"),
                    "demographic": task["demographic"],
                },
            }
            examples.append(example)
            f.write(json.dumps(example) + "\n")

    print(f"✅ Generated {len(examples)} examples")
    print(f"📁 Saved to: {output_path}")

    return examples


def generate_targeted_examples(
    processor: FlexibleSchemaProcessor,
    alert_type: str = None,
    non_alert_type: str = None,
    n_examples: int = 10,
    difficulty: str = None,
    output_path: str = None,
) -> List[dict]:
    """Generate examples targeting a specific alert or non-alert type."""

    if alert_type:
        example_type = "hard_alert" if difficulty == "subtle" else "alert"
    else:
        example_type = "hard_non_alert" if difficulty == "subtle" else "non_alert"

    if output_path is None:
        target = alert_type or non_alert_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"generated_{target}_{timestamp}.jsonl"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i in range(n_examples):
        task = {
            "example_type": example_type,
            "industry": INDUSTRIES[i % len(INDUSTRIES)],
            "role": ROLES[i % len(ROLES)],
            "tone": TONES[i % len(TONES)],
            "writing_style": WRITING_STYLES[i % len(WRITING_STYLES)],
            "difficulty": difficulty or DIFFICULTY_LEVELS[i % len(DIFFICULTY_LEVELS)],
            "demographic": DEMOGRAPHICS[i % len(DEMOGRAPHICS)],
        }

        if alert_type:
            task["alert_types"] = [alert_type]
        if non_alert_type:
            task["non_alert_type"] = non_alert_type

        tasks.append(task)

    prompts = [build_prompt_for_task(task) for task in tasks]

    print(f"🎯 Generating {n_examples} targeted examples for: {alert_type or non_alert_type}")
    start_time = time.time()

    processor.process_with_schema(prompts=prompts, schema=GeneratedExample)
    results: List[GeneratedExample] = processor.parse_results_with_schema(
        schema=GeneratedExample
    )

    end_time = time.time()
    print(f"⏱️ Generation took {end_time - start_time:.2f} seconds")

    examples = []
    with open(output_path, "w") as f:
        for task, result in zip(tasks, results):
            if result is None:
                continue

            example = {
                "text": result.text.strip().strip('"').strip("'"),
                "metadata": {
                    "example_type": task["example_type"],
                    "expected_alert_types": task.get("alert_types"),
                    "expected_non_alert_type": task.get("expected_non_alert_type"),
                    "industry": task["industry"],
                    "role": task["role"],
                    "tone": task["tone"],
                    "writing_style": task["writing_style"],
                    "difficulty": task.get("difficulty"),
                    "demographic": task["demographic"],
                },
            }
            examples.append(example)
            f.write(json.dumps(example) + "\n")

    print(f"✅ Generated {len(examples)} examples")
    print(f"📁 Saved to: {output_path}")

    return examples


# -------------------------------
# Coverage analysis
# -------------------------------
def load_examples(path: str) -> List[dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def check_coverage(examples: List[dict]) -> dict:
    """Analyze coverage across all diversity dimensions."""

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
        if meta.get("difficulty"):
            stats["difficulties"][meta["difficulty"]] += 1
        stats["demographics"][meta["demographic"]] += 1

        if meta.get("expected_alert_types"):
            for at in meta["expected_alert_types"]:
                stats["alert_types"][at] += 1
        if meta.get("expected_non_alert_type"):
            stats["non_alert_types"][meta["expected_non_alert_type"]] += 1

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
    print(f"  ... and {len(stats['industries']) - 5} more")

    print("\n🎭 Tones:")
    for t, c in stats["tones"].most_common(5):
        print(f"  {t}: {c}")

    print("\n✍️ Writing Styles:")
    for t, c in stats["writing_styles"].most_common(5):
        print(f"  {t}: {c}")

    print("\n👤 Demographics:")
    for t, c in stats["demographics"].most_common(5):
        print(f"  {t}: {c}")


def save_examples_to_csv(examples: List[dict], output_path: str = None) -> str:
    """Save examples to CSV for easy review."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"generated_examples_{timestamp}.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, ex in enumerate(examples):
        meta = ex["metadata"]
        rows.append(
            {
                "id": i + 1,
                "text": ex["text"],
                "example_type": meta["example_type"],
                "expected_alert_types": ", ".join(meta.get("expected_alert_types") or []),
                "expected_non_alert_type": meta.get("expected_non_alert_type") or "",
                "industry": meta["industry"],
                "role": meta["role"],
                "tone": meta["tone"],
                "writing_style": meta["writing_style"],
                "difficulty": meta.get("difficulty") or "",
                "demographic": meta["demographic"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"📁 CSV saved to: {output_path}")
    return output_path


# -------------------------------
# Main
# -------------------------------
def main():
    processor = FlexibleSchemaProcessor(
        gpu_list=[0, 1, 2, 3, 4, 5, 6, 7],
        llm=NEMO,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        multiplicity=1,
    )

    try:
        # Generate diverse dataset
        examples = generate_examples(
            n_examples=10000,
            processor=processor,
            output_path="generated_data/generated_examples.jsonl",
            distribution={
                "alert": 0.30,
                "non_alert": 0.25,
                "mixed": 0.15,
                "hard_non_alert": 0.15,
                "hard_alert": 0.15,
            },
        )

        # Check coverage
        stats = check_coverage(examples)
        print_coverage_report(stats)

        # Fill gaps for missing alert types
        for missing_type in stats["missing_alert_types"]:
            print(f"\n🔧 Generating examples for missing alert type: {missing_type}")
            gap_examples = generate_targeted_examples(
                processor=processor,
                alert_type=missing_type,
                n_examples=10,
                output_path=f"generated_data/gap_fill_{missing_type}.jsonl",
            )
            examples.extend(gap_examples)

        # Fill gaps for missing non-alert types
        for missing_type in stats["missing_non_alert_types"]:
            print(f"\n🔧 Generating examples for missing non-alert type: {missing_type}")
            gap_examples = generate_targeted_examples(
                processor=processor,
                non_alert_type=missing_type,
                n_examples=10,
                output_path=f"data/gap_fill_{missing_type}.jsonl",
            )
            examples.extend(gap_examples)

        # Save all to CSV for review
        save_examples_to_csv(
            examples,
            "/home/clyde/workspace/alerts_detection_llama/scripts/generation/datasets/nemo_generated_examples.csv",
        )

    finally:
        processor.terminate()


if __name__ == "__main__":
    main()
