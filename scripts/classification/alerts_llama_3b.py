from collections import Counter
import time
from typing import List, Literal, Optional

from llm_parallelization.parallelization import (
    GEMMA3_4B_INSTRUCT,
    LLAMA32_3B_INSTRUCT_AWQ,
    LLAMA_33_8B_INSTRUCT,
    MISTRAL_MODEL,
    FlexibleSchemaProcessor,
)
from pydantic import BaseModel

a = (
    LLAMA32_3B_INSTRUCT_AWQ,
    LLAMA_33_8B_INSTRUCT,
    GEMMA3_4B_INSTRUCT,
    MISTRAL_MODEL,
)


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


# -------------------------------
# Prompt generation
# -------------------------------
def alert_detection_prompt(text: str) -> str:
    """Generate optimized prompt for alert detection."""
    return f"""You are an expert workplace safety analyzer. Analyze this employee comment for alerts.

COMMENT: {text}

---

ALERT = serious concern needing HR/management attention
NOT ALERT = normal feedback, complaints about quality/teaching

---

ALERT TYPES:
- discrimination: racial slurs, ethnic targeting
- sexual_harassment: unwanted advances, sexual comments
- suggestive_language: sexual innuendo, "that's what she said", wink wink
- profanity: fuck, shit, damn, bitch, ass
- threat_of_violence: "hurt you", "get back at", physical threats
- workplace_violence: hitting, grabbing, physical harm
- retaliation: "watch your back", "you'll regret", punishment for reporting
- coercive_threat: "I'll fail you if", conditional threats
- quid_pro_quo: "do X for promotion", "office after hours for grade"
- data_breach: hacked, leaked data, security breach
- substance_abuse_at_work: drunk, high, coked up at work
- mental_wellbeing_concern: depressed, anxious, can't cope, breakdown
- mental_health_crisis: suicidal, want to die, end it all
- physical_safety_concern: injury, hurt back, unsafe conditions
- pattern_of_unfair_treatment: "only happens to me", singled out
- bullying: constant criticism, public humiliation
- safety_hazard: fire risk, untested products, dangerous conditions

SEVERITY:
- critical: immediate danger, violence, suicide risk
- high: serious violations, threats, discrimination, data breach
- moderate: concerning but not urgent
- low: minor issues, mild profanity

---

EXAMPLES:

Comment: "That's what she said! You know what I mean ;) wink wink"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "That's what she said! You know what I mean ;) wink wink", "reasoning": "Sexual innuendo and suggestive language", "alert_type": "suggestive_language", "severity": "low"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "This fucking presentation was a waste of my damn time"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "This fucking presentation was a waste of my damn time", "reasoning": "Contains profanity", "alert_type": "profanity", "severity": "low"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "Come to my office after hours and we can discuss your grade improvement"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "Come to my office after hours and we can discuss your grade improvement", "reasoning": "Suggests quid pro quo arrangement", "alert_type": "quid_pro_quo", "severity": "moderate"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I am constantly getting bossed around, and it's not something that's being done to all of my colleagues. Just me."
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "not something that's being done to all of my colleagues. Just me", "reasoning": "Reports being singled out for unfair treatment", "alert_type": "pattern_of_unfair_treatment", "severity": "moderate"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "He grabbed me by the shoulder and really hurt me"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "grabbed me by the shoulder and really hurt me", "reasoning": "Physical violence causing injury", "alert_type": "workplace_violence", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "shiiiiiiit they let the VPN down and we got hacked"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "we got hacked", "reasoning": "Reports security breach", "alert_type": "data_breach", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "He called the Chinese guy a fucking chink"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "called the Chinese guy a fucking chink", "reasoning": "Racial slur and discrimination", "alert_type": "discrimination", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I am really going to get back at that bitch"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "going to get back at that bitch", "reasoning": "Threat of retaliation", "alert_type": "retaliation", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "You'll regret going to HR about this. Watch your back."
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "You'll regret going to HR about this. Watch your back", "reasoning": "Threatening retaliation for reporting", "alert_type": "retaliation", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "If you keep questioning my methods, I'll make sure you fail this course"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "I'll make sure you fail this course", "reasoning": "Coercive threat using power", "alert_type": "coercive_threat", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "these guys come in coked up and management expects us to turn the other cheek"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "come in coked up", "reasoning": "Substance abuse at workplace", "alert_type": "substance_abuse_at_work", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I really can't take it anymore. I'm about to breakdown in this fucking place"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "can't take it anymore. I'm about to breakdown", "reasoning": "Mental health distress and burnout", "alert_type": "mental_wellbeing_concern", "severity": "moderate"}}, {{"excerpt": "this fucking place", "reasoning": "Contains profanity", "alert_type": "profanity", "severity": "low"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I hurt my back lifting those boxes and now I can barely walk. No one even provided safety equipment"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "I hurt my back lifting those boxes and now I can barely walk. No one even provided safety equipment", "reasoning": "Physical injury from unsafe conditions", "alert_type": "physical_safety_concern", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I'm feeling really anxious and stressed about everything. I can barely sleep"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "feeling really anxious and stressed about everything. I can barely sleep", "reasoning": "Mental health concern", "alert_type": "mental_wellbeing_concern", "severity": "moderate"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "he told me to get on my knees to blow him and maybe that would get me promoted"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "get on my knees to blow him and maybe that would get me promoted", "reasoning": "Sexual harassment with quid pro quo", "alert_type": "sexual_harassment", "severity": "critical"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "The product was not even tested and poses a fire risk"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "not even tested and poses a fire risk", "reasoning": "Safety hazard with fire risk", "alert_type": "safety_hazard", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "The guy snorted heroin, punched the manager, called the chinese guy a chink, and leaked data to competitors"
Output: {{"has_alerts": true, "alerts": [{{"excerpt": "snorted heroin", "reasoning": "Drug use at work", "alert_type": "substance_abuse_at_work", "severity": "high"}}, {{"excerpt": "punched the manager", "reasoning": "Physical violence", "alert_type": "workplace_violence", "severity": "high"}}, {{"excerpt": "called the chinese guy a chink", "reasoning": "Racial slur", "alert_type": "discrimination", "severity": "high"}}, {{"excerpt": "leaked data to competitors", "reasoning": "Data breach", "alert_type": "data_breach", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "The training was boring and the instructor could've been more engaging"
Output: {{"has_alerts": false, "alerts": [], "non_alert_classification": "quality_complaint", "non_alert_reasoning": "General feedback about training quality"}}

Comment: "The professor doesn't seem to care about his job, he just marks the way he feels like"
Output: {{"has_alerts": false, "alerts": [], "non_alert_classification": "performance_complaint", "non_alert_reasoning": "Complaint about job performance"}}

Comment: "The training was fantastic! The instructor really knew their stuff"
Output: {{"has_alerts": false, "alerts": [], "non_alert_classification": "positive_feedback", "non_alert_reasoning": "Positive feedback about training"}}

Comment: "I think we should add more hands-on exercises to the workshop"
Output: {{"has_alerts": false, "alerts": [], "non_alert_classification": "constructive_feedback", "non_alert_reasoning": "Suggestion for improvement"}}

---

RULES:
1. Sexual innuendo like "that's what she said", wink wink = suggestive_language ALERT
2. Any profanity (fuck, shit, damn, bitch) = profanity ALERT  
3. "After hours" meetings about grades/promotions = quid_pro_quo ALERT
4. Being singled out / "just me" = pattern_of_unfair_treatment ALERT
5. Hacking/breach = data_breach ALERT
6. Physical contact that hurts = workplace_violence ALERT
7. Threats about reporting to HR = retaliation ALERT
8. Multiple issues in one comment = multiple alerts
9. If no alerts, provide non_alert_classification and non_alert_reasoning

Analyze the comment and return the JSON output."""


# -------------------------------
# Main
# -------------------------------
def main():
    # Test texts covering all alert types and non-alert types
    # These are different from the prompt examples
    texts = [
        # discrimination
        "My supervisor keeps making comments about how women shouldn't be in engineering roles",
        # sexual_harassment
        "He keeps staring at my chest during meetings and made a comment about my legs yesterday",
        # severe_harassment
        "They created a group chat specifically to mock everything I do and say at work",
        # bullying
        "Every single day she finds something to criticize me about in front of the whole team",
        # workplace_violence
        "He slammed me against the wall in the break room when no one was looking",
        # threat_of_violence
        "He said if I report him he'll be waiting for me in the parking lot",
        # coercive_threat
        "She told me if I don't work this weekend unpaid, she'll give me a bad performance review",
        # safety_hazard
        "The emergency exits have been blocked with boxes for weeks and nobody cares",
        # retaliation
        "Ever since I filed that complaint, I've been excluded from all important meetings",
        # substance_abuse_at_work
        "My coworker keeps a flask in his desk and is clearly drunk by lunch every day",
        # data_breach
        "Someone left all the customer credit card info on an unsecured shared drive",
        # security_incident
        "I found a USB drive in the parking lot and plugged it into my work computer, now it's acting weird",
        # fraud
        "My manager has been submitting fake expense reports for trips he never took",
        # corruption
        "The procurement guy is steering all contracts to his brother's company",
        # quid_pro_quo
        "She implied that a promotion might happen faster if we spent some private time together",
        # ethics_violation
        "We're being told to lie to customers about the product capabilities to close sales",
        # mental_health_crisis
        "I don't see the point anymore. Everything feels hopeless and I've been thinking about not waking up",
        # pattern_of_unfair_treatment
        "I'm the only one who has to get approval for expenses under $50 while everyone else just submits them",
        # workload_burnout_risk
        "I've worked 80 hour weeks for three months straight and my requests for help keep getting denied",
        # management_concern
        "Our director makes all decisions without consulting anyone and refuses to explain his reasoning",
        # interpersonal_conflict
        "Me and John from accounting have been arguing constantly and it's affecting our work",
        # professional_misconduct
        "She's been using company resources to run her personal Etsy business during work hours",
        # inappropriate_language
        "He keeps telling really crude jokes that make everyone uncomfortable",
        # profanity
        "This goddamn system crashes every five minutes and I'm sick of this bullshit",
        # suggestive_language
        "He always comments on how I look and says things like 'I bet you're fun outside of work'",
        # mental_wellbeing_concern
        "I haven't been sleeping well and I feel overwhelmed all the time lately",
        # physical_safety_concern
        "The ladder they gave me is broken and I almost fell from the second floor yesterday",
        # performance_complaint (non-alert)
        "My team lead never meets deadlines and it delays everyone else's work",
        # quality_complaint (non-alert)
        "The new software update is buggy and crashes frequently",
        # workload_feedback (non-alert)
        "The project timeline seems aggressive given our current resources",
        # process_improvement (non-alert)
        "It would be more efficient if we had a shared calendar for meeting room bookings",
        # resource_request (non-alert)
        "We really need a second printer on this floor to reduce wait times",
        # general_dissatisfaction (non-alert)
        "Things just aren't the same since the reorganization",
        # constructive_feedback (non-alert)
        "The onboarding process could be improved by adding a mentor system for new hires",
        # positive_feedback (non-alert)
        "The new flexible work policy has really improved my work-life balance",
        # neutral_comment (non-alert)
        "The quarterly meeting is scheduled for next Tuesday at 3pm",
        # unclear (non-alert)
        "It is what it is I guess",
        # discrimination + profanity + sexual_harassment
        "That fucking new guy is useless - what do you expect from a towelhead? And did you see him staring at Sarah's ass all day?",
        # substance_abuse + workplace_violence + profanity
        "Dave came in high as a kite again, then that asshole threw a chair at me when I told him to go home",
        # data_breach + retaliation + threat_of_violence
        "After I reported the security breach where client passwords were exposed, my boss said he'd make my life hell and that I should watch myself",
        # quid_pro_quo + discrimination + suggestive_language
        "The director told me Asian women are more 'submissive' and that if I was nicer to him after work, he could fast-track my visa sponsorship",
        # mental_wellbeing_concern + workload_burnout_risk + profanity
        "I'm fucking exhausted, haven't seen my kids in weeks because of these deadlines, and honestly I just don't care about anything anymore",
        # safety_hazard + fraud + ethics_violation
        "They're falsifying the safety inspection reports and selling equipment they know is defective - someone is going to get killed",
        # bullying + discrimination + hostile_environment
        "The whole team calls him 'the retard' behind his back, mimics his stutter in meetings, and management thinks it's hilarious",
        # physical_safety_concern + retaliation + coercive_threat
        "I reported my injury from the faulty equipment and now they're threatening to fire me if I file workers comp",
        # sexual_harassment + profanity + pattern_of_unfair_treatment
        "He grabs my waist every time he walks by, calls me his 'little slut' when no one's around, and I'm the only one who never gets approved for PTO",
        # corruption + data_breach + professional_misconduct
        "The CFO is taking kickbacks from vendors and I caught him downloading the entire customer database to a personal drive last week",
        # discrimination + workplace_violence + mental_health_crisis
        "They jumped him in the warehouse because he's gay, beat him pretty bad. He texted me saying he can't take it anymore and wants to end things",
        # substance_abuse + safety_hazard + profanity
        "Half the warehouse crew is on meth, operating forklifts while tweaking out. This shithole is a disaster waiting to happen",
        # threat_of_violence + retaliation + discrimination
        "He told me if I go to HR about the n-word comments, he knows where I park and things could get ugly for me",
        # suggestive_language + quid_pro_quo + non-alert mixed
        "The training was actually pretty good, but afterwards the instructor asked me to stay behind and said we should 'get to know each other better' if I want the certification",
        # profanity + positive feedback mixed (edge case)
        "Holy shit this new system is fucking amazing, best upgrade we've ever had, the IT team killed it",
        # mental_wellbeing_concern + constructive_feedback mixed
        "I've been really struggling with anxiety since the reorg, but I think having clearer role definitions would help everyone, not just me",
        # bullying + security_incident + ethics_violation
        "My manager publicly humiliates me daily, and when I tried to document it, someone deleted all my emails. HR said to drop it or else",
        # discrimination + fraud + substance_abuse
        "They only hire white guys for management, the expense reports are completely made up, and the CEO does lines in his office with the door open",
        # workplace_violence + profanity + physical_safety_concern (severe)
        "That psycho bitch stabbed me with scissors, there's blood everywhere, and I can't feel my fucking hand",
        # pattern_of_unfair_treatment + coercive_threat + non-alert mixed
        "The coffee machine is broken again. Also, I'm the only one required to work holidays, and when I asked why, my boss said questioning him would affect my bonus",
    ]

    processor = FlexibleSchemaProcessor(
        gpu_list=[0, 1, 2, 3, 4, 5, 6, 7],
        # llm = LLAMA32_3B_INSTRUCT_AWQ,
        llm=LLAMA_33_8B_INSTRUCT,
        # llm = GEMMA3_4B_INSTRUCT,
        # llm = MISTRAL_MODEL,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        multiplicity=1,
    )

    try:
        prompts = [alert_detection_prompt(text) for text in texts]

        start_time = time.time()
        processor.process_with_schema(prompts=prompts, schema=AlertsOutput)
        results: List[AlertsOutput] = processor.parse_results_with_schema(schema=AlertsOutput)
        end_time = time.time()

        print(f"\n{'=' * 80}")
        print("FLEXIBLE SCHEMA PROCESSOR CLASSIFICATION RESULTS")
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

    finally:
        processor.terminate()


if __name__ == "__main__":
    main()
