from collections import Counter
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import List, Literal, Optional

from llm_parallelization.parallelization import (
    LLAMA_33_70B_INSTRUCT,
    NEMO,
    QWEN_AWQ,
    FlexibleSchemaProcessor,
)
import pandas as pd
from pydantic import BaseModel

model = "nemo_ft"

model_dict = {
    "nemo": NEMO,
    "llama70": LLAMA_33_70B_INSTRUCT,
    "qwen": QWEN_AWQ,
    "nemo_ft": "/data-fast/data3/clyde/fine_tuning/alert_models/alerts-nemo-awq-pure",
}


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


# -------------------------------
# Prompt generation
# -------------------------------
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

Analyze the comment and return ONLY valid JSON."""


def main():
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
        # Mixed: discrimination + profanity + sexual_harassment
        "That fucking new guy is useless - what do you expect from a towelhead? And did you see him staring at Sarah's ass all day?",
        # Mixed: substance_abuse + workplace_violence + profanity
        "Dave came in high as a kite again, then that asshole threw a chair at me when I told him to go home",
        # Mixed: data_breach + retaliation + threat_of_violence
        "After I reported the security breach where client passwords were exposed, my boss said he'd make my life hell and that I should watch myself",
        # Mixed: quid_pro_quo + discrimination + suggestive_language
        "The director told me Asian women are more 'submissive' and that if I was nicer to him after work, he could fast-track my visa sponsorship",
        # Mixed: mental_wellbeing_concern + workload_burnout_risk + profanity
        "I'm fucking exhausted, haven't seen my kids in weeks because of these deadlines, and honestly I just don't care about anything anymore",
        # Mixed: safety_hazard + fraud + ethics_violation
        "They're falsifying the safety inspection reports and selling equipment they know is defective - someone is going to get killed",
        # Mixed: bullying + discrimination + hostile_environment
        "The whole team calls him 'the retard' behind his back, mimics his stutter in meetings, and management thinks it's hilarious",
        # Mixed: physical_safety_concern + retaliation + coercive_threat
        "I reported my injury from the faulty equipment and now they're threatening to fire me if I file workers comp",
        # Mixed: sexual_harassment + profanity + pattern_of_unfair_treatment
        "He grabs my waist every time he walks by, calls me his 'little slut' when no one's around, and I'm the only one who never gets approved for PTO",
        # Mixed: corruption + data_breach + professional_misconduct
        "The CFO is taking kickbacks from vendors and I caught him downloading the entire customer database to a personal drive last week",
        # Mixed: discrimination + workplace_violence + mental_health_crisis
        "They jumped him in the warehouse because he's gay, beat him pretty bad. He texted me saying he can't take it anymore and wants to end things",
        # Mixed: substance_abuse + safety_hazard + profanity
        "Half the warehouse crew is on meth, operating forklifts while tweaking out. This shithole is a disaster waiting to happen",
        # Mixed: threat_of_violence + retaliation + discrimination
        "He told me if I go to HR about the n-word comments, he knows where I park and things could get ugly for me",
        # Mixed: suggestive_language + quid_pro_quo + non-alert
        "The training was actually pretty good, but afterwards the instructor asked me to stay behind and said we should 'get to know each other better' if I want the certification",
        # Mixed: profanity + positive feedback (edge case)
        "Holy shit this new system is fucking amazing, best upgrade we've ever had, the IT team killed it",
        # Mixed: mental_wellbeing_concern + constructive_feedback
        "I've been really struggling with anxiety since the reorg, but I think having clearer role definitions would help everyone, not just me",
        # Mixed: bullying + security_incident + ethics_violation
        "My manager publicly humiliates me daily, and when I tried to document it, someone deleted all my emails. HR said to drop it or else",
        # Mixed: discrimination + fraud + substance_abuse
        "They only hire white guys for management, the expense reports are completely made up, and the CEO does lines in his office with the door open",
        # Mixed: workplace_violence + profanity + physical_safety_concern (severe)
        "That psycho bitch stabbed me with scissors, there's blood everywhere, and I can't feel my fucking hand",
        # Mixed: pattern_of_unfair_treatment + coercive_threat + non-alert
        "The coffee machine is broken again. Also, I'm the only one required to work holidays, and when I asked why, my boss said questioning him would affect my bonus",
        "She's a real gun!",
        "It's like drinking from a fire hose",
        # Technical jargon that sounds violent
        "We need to kill the zombie processes, terminate the hanging threads, and abort the failed jobs before we can restart the server",
        # Metaphorical language
        "This project is killing me, the deadline is murder, and I'm dying to get it finished",
        # Past tense / resolved issues
        "Last year we had a manager who was really inappropriate but he was fired and HR handled it well",
        # Quoting a movie/show
        "My coworker keeps saying 'that's what she said' like Michael Scott from The Office, it's pretty funny actually",
        # Discussing policy, not violating it
        "The new harassment training explained that quid pro quo means exchanging favors for promotions, which is obviously wrong",
        # Hypothetical / training scenario
        "In the ethics workshop, they asked what we'd do if a manager offered a promotion in exchange for a date",
        # Frustration without actual profanity
        "This freaking system is driving me nuts, I'm so done with this dang thing crashing",
        # Gaming/competition language
        "Our sales team is going to crush the competition this quarter, we're absolutely destroying our targets",
        # Self-deprecating humor
        "I'm such an idiot, I deleted my own presentation five minutes before the meeting",
        # Medical/health discussion (not crisis)
        "I've been seeing a therapist for my anxiety and it's really helping with work stress",
        # Role-playing exercise description
        "In the conflict resolution training, I played the angry customer who yells and threatens to sue",
        # News/external event discussion
        "Did you hear about that data breach at the other company? Glad our security team is on top of things",
        # Reporting someone else's resolved complaint
        "Sarah mentioned she felt uncomfortable last month but talked to her manager and it's all sorted now",
        # Dark humor about workload (common in tech)
        "Deploy on Friday? Sure, I too like to live dangerously. What could possibly go wrong?",
        # Complaining about difficult client (not internal)
        "The client is absolutely insane, their demands are ridiculous and they yell at us on every call",
        # ==================== HARD ALERTS (False Negative Traps) ====================
        # Minimized serious harassment
        "He just brushed against me a few times, probably an accident, though it keeps happening when we're alone",
        # Subtle discrimination ("culture fit")
        "They said I wasn't the right 'culture fit' for the leadership team, funny how no one who looks like me ever is",
        # Coded threat
        "He smiled and said he hopes nothing 'unfortunate' happens to my car while I'm thinking about whether to report him",
        # Gaslighting description
        "Everyone keeps telling me I'm too sensitive and imagining things but I know he treats me differently than everyone else",
        # Buried in casual language
        "Haha so my manager jokingly said he'd tank my review if I don't come to his cabin this weekend, classic him right",
        # Passive voice hiding severity
        "Mistakes were made and some inappropriate comments may have been directed at certain team members based on their background",
        # Normalized substance abuse
        "Oh that's just how Dave is, everyone knows he has a few drinks at lunch, it's not a big deal",
        # Implied quid pro quo
        "She keeps mentioning how she 'takes care of people who take care of her' whenever we discuss my promotion",
        # Subtle retaliation
        "Ever since I asked about the pay gap, I've noticed I'm no longer CC'd on important emails, probably just an oversight though",
        # Minimized violence
        "He grabbed my arm pretty hard during the argument but I'm sure he didn't mean anything by it",
        # Religious/cultural discrimination disguised
        "They keep scheduling mandatory team events on Saturdays even though they know I observe Shabbat and can't attend",
        # Ageism hidden in feedback
        "My manager said I need to be more 'digitally native' and have 'fresh ideas' like the younger team members",
        # Pregnancy discrimination
        "After I announced my pregnancy, suddenly my projects got reassigned because they 'didn't want to stress me out'",
        # Disability discrimination
        "They moved my desk far from the bathroom even though they know about my condition, said it was for 'team reorganization'",
        # Subtle mental health crisis
        "I've been giving away my things at work, tying up loose ends, making sure my projects are documented in case I'm not around",
        # ==================== LONG MIXED TEXTS ====================
        # Performance review with buried harassment
        """My quarterly review was mostly positive - exceeded targets, good client feedback, strong technical skills. 
    However, my manager did mention that I'd advance faster if I was 'more friendly' with leadership and 
    'dressed nicer' for him. He also said the team loves working with me and my documentation is excellent. 
    Overall he rated me as 'meets expectations' despite exceeding all my metrics.""",
        # Meeting recap with concerning elements
        """Team meeting notes: Discussed Q3 roadmap (on track), budget approved for new hires, holiday party planning 
    going well. Side note - Mark made another joke about the 'diversity hire' comment, third time this month. 
    Also reviewed the new security protocols which everyone seemed to understand. Action items assigned, 
    next meeting in two weeks.""",
        # Email-style mixed complaint
        """I wanted to share some feedback about the new office layout. The standing desks are great and the 
    natural lighting really helps productivity. However, I've noticed that all the women on the team were 
    moved to the back corner while the men got the window seats - not sure if intentional but feels off. 
    Also love the new coffee machine! The break room renovations are fantastic. Could we also look into 
    the temperature issues? It's freezing in here.""",
        # Casual conversation hiding multiple issues
        """Crazy day yesterday! The server crashed (IT fixed it fast though, kudos to them), then Jim from 
    accounting called me 'sweetheart' again during the presentation, pizza party was fun but someone 
    definitely spiked the punch and a few people drove home which seemed sketchy. Oh and great news - 
    the Thompson project got approved! Bad news - they want it done in half the time with no extra resources. 
    Same old same old lol.""",
        # Positive review with subtle retaliation
        """I love my job overall! Great benefits, smart colleagues, interesting work. My only concern is that 
    ever since I reported the expense irregularities I noticed, my workload has tripled and I've been 
    excluded from the strategy meetings I used to attend. Probably coincidence! The new project I'm on 
    is really challenging and I'm learning a lot. Just wish I had more support from management like before.""",
        # Training feedback with harassment buried
        """The leadership workshop was excellent - learned a lot about communication styles and conflict resolution. 
    The facilitator was engaging and knowledgeable. Only awkward moment was during the networking lunch 
    when one of the VPs cornered me and kept asking about my relationship status and whether I'd like to 
    'continue the conversation over dinner.' The afternoon sessions on strategic planning were very valuable 
    though, especially the SWOT analysis framework. Would recommend this training to others.""",
        # Venting that escalates
        """Look, I get that work is stressful and we all have bad days. The tight deadlines, the demanding clients, 
    the endless meetings - it's a lot. But I'm really struggling lately. I haven't slept properly in weeks, 
    I've stopped eating, nothing feels enjoyable anymore. My doctor increased my meds but they're not helping. 
    I keep thinking everyone would be better off without me dragging down the team. Anyway, sorry for the rant, 
    back to those TPS reports I guess.""",
        # Whistleblower retaliation with positive framing
        """Great news - I finally got clarity on my situation! After I reported the financial discrepancies to the 
    ethics hotline, my role has been 'restructured.' They say it's not retaliation, just organizational changes. 
    My new position is two levels below my old one, in a different department, with a 30% pay cut, but at least 
    I still have a job! My new manager seems nice, though he mentioned he was told I'm a 'troublemaker.' 
    Fresh start I guess!""",
        # Technical discussion with security breach buried
        """Sprint retrospective: Good velocity this sprint, knocked out 45 story points. The new CI/CD pipeline is 
    working smoothly. One thing we should probably mention to someone - found some weird access logs showing 
    someone downloaded the entire customer database last Tuesday at 3am from an IP in Russia. Might be nothing? 
    Jenkins integration needs work. Shoutout to Amy for fixing that nasty memory leak! Planning poker for next 
    sprint is Thursday.""",
        # HR complaint mixing valid and invalid
        """I need to document several concerns: 1) The coffee machine has been broken for two weeks, 2) My manager 
    called me 'too emotional' when I disagreed with him in a meeting, which feels gendered, 3) The parking lot 
    needs better lighting, 4) I overheard my manager telling someone I only got hired because of 'diversity 
    quotas,' 5) The lunch options in the cafeteria are limited, 6) I've been passed over for three promotions 
    that went to less qualified men, 7) Could we get more plants in the office? Thanks.""",
        "I don't like coming on Monday mornings",
    ] * 10

    if model == "nemo_ft":
        tokenizer = "mistralai/Mistral-Nemo-Instruct-2407"
    else:
        tokenizer = None

    processor = FlexibleSchemaProcessor(
        gpu_list=[1, 2, 3, 4],
        llm=model_dict[model],
        gpu_memory_utilization=0.95,
        tokenizer=tokenizer,
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

    finally:
        processor.terminate()


if __name__ == "__main__":
    main()
