"""scripts/build_ukps_items.py: author data/ukps/items.jsonl per protocol.

Distinction-level integrity is enforced by construction, not trusted:

  - every PASSAGE is sliced VERBATIM from the source document by
    locating start/end marker phrases, so no passage can drift from the
    source;
  - every QA ANSWER is located inside its passage and the EXACT matched
    substring is stored, so the span-based rule (Chapter 3) holds even
    across curly apostrophes and pound signs;
  - marker/answer search is apostrophe- and whitespace-insensitive
    (the stored text remains the original), so authoring is robust;
  - QA passages are asserted <= 6000 chars; summary passages 1500-6000;
    reference summaries 40-120 words (protocol band).

The script fails loudly on any violation, so a committed items.jsonl is
provably faithful. BLIND RULE: all reference summaries are authored
here, before any model is run on the corpus (see LABLOG.md).

Run from the repository root:
  python scripts/build_ukps_items.py
"""

from __future__ import annotations

import json
import os
import re

DOCS = os.path.join("data", "ukps", "documents")
OUT = os.path.join("data", "ukps", "items.jsonl")

_cache: dict[str, str] = {}


def doc(doc_id: str) -> str:
    if doc_id not in _cache:
        with open(os.path.join(DOCS, f"{doc_id}.txt")) as f:
            _cache[doc_id] = f.read()
    return doc_id and _cache[doc_id]


def _rx(probe: str) -> re.Pattern:
    """Regex that matches `probe` verbatim-ish: apostrophes and dashes
    are class-equivalent and whitespace runs are flexible, so ASCII
    authoring locates curly-quoted source text."""
    out = []
    for ch in probe:
        if ch in "'’":
            out.append("['’]")
        elif ch in "-–—":
            out.append("[-–—]")
        elif ch.isspace():
            out.append(r"\s+")
        else:
            out.append(re.escape(ch))
    return re.compile("".join(out))


def locate(text: str, probe: str, start: int = 0) -> re.Match:
    m = _rx(probe).search(text, start)
    if not m:
        raise SystemExit(f"[items] probe not found: {probe!r}")
    return m


def passage(doc_id: str, start_probe: str, end_probe: str) -> str:
    text = doc(doc_id)
    s = locate(text, start_probe)
    e = locate(text, end_probe, s.start())
    return text[s.start():e.end()].strip()


def passage_min(doc_id: str, start_probe: str, target: int = 1700) -> str:
    """Verbatim slice from the start marker, extended to the first
    paragraph boundary at or after `target` chars (capped 6000). Keeps
    summary passages inside the 1500-6000 protocol band without
    hand-tuning an end marker for every item."""
    text = doc(doc_id)
    s = locate(text, start_probe).start()
    tail = text[s:]
    cut = tail.find("\n\n", target)
    if cut == -1 or cut > 6000:
        cut = min(len(tail), 6000)
    return tail[:cut].strip()


def qa_passage(doc_id: str, start_probe: str, end_probe: str,
               floor: int = 450) -> str:
    """QA passage: the start..end verbatim slice, extended forward to a
    paragraph boundary past `floor` chars so the model reads a real
    self-contained excerpt rather than a bare answer sentence. Only
    grows the slice, so the answer (inside start..end) stays contained.
    Capped at 6000."""
    text = doc(doc_id)
    s = locate(text, start_probe).start()
    e = locate(text, end_probe, s).end()
    if e - s < floor:
        cut = text.find("\n\n", s + floor)
        if cut != -1 and cut - s <= 6000:
            e = max(e, cut)
        else:
            e = max(e, min(len(text), s + 6000))
    return text[s:e].strip()


def answers(passage_text: str, probes: list[str]) -> list[str]:
    """Return the EXACT substrings of the passage matching each probe."""
    return [locate(passage_text, p).group(0) for p in probes]


# =====================================================================
# QA SPEC: (doc, start_probe, end_probe, question, [answer_probes])
# =====================================================================
QA = [
    # ---- ukps01 Statutory Sick Pay ----
    ("ukps01", "You can get up to", "Agency workers may be entitled to Statutory Sick Pay.",
     "How much Statutory Sick Pay can you get per week?", ["£123.25"]),
    ("ukps01", "You can get up to", "Agency workers may be entitled to Statutory Sick Pay.",
     "For up to how many weeks is Statutory Sick Pay paid?", ["28 weeks"]),
    ("ukps01", "You can get up to", "Agency workers may be entitled to Statutory Sick Pay.",
     "How long must you have been ill to qualify for Statutory Sick Pay?",
     ["at least one full working day"]),
    # ---- ukps02 Jury service ----
    ("ukps02", "If you get a jury summons in the post", "Check your summons letter for the exact time.",
     "Within how many days must you respond to a jury summons?", ["within 7 days"]),
    ("ukps02", "If you get a jury summons in the post", "Check your summons letter for the exact time.",
     "How many people make up a jury?", ["12 people"]),
    ("ukps02", "If you get a jury summons in the post", "Check your summons letter for the exact time.",
     "How long does jury service usually last?", ["up to 10 working days"]),
    # ---- ukps03 Universal Credit ----
    ("ukps03", "Universal Credit is a payment to help", "check what benefits you could get.",
     "How often is Universal Credit usually paid?", ["monthly"]),
    ("ukps03", "Universal Credit is a payment to help", "check what benefits you could get.",
     "What is the age requirement to claim Universal Credit?", ["18 or over"]),
    ("ukps03", "Universal Credit is a payment to help", "check what benefits you could get.",
     "What is the maximum in savings and investments allowed to claim Universal Credit?",
     ["£16,000 or less"]),
    # ---- ukps04 Self Assessment ----
    ("ukps04", "Self Assessment is a system", "foreign income",
     "By which date must you tell HMRC you need to complete a tax return?", ["5 October"]),
    ("ukps04", "Self Assessment is a system", "foreign income",
     "By when must you pay your Self Assessment bill?", ["31 January"]),
    ("ukps04", "Self Assessment is a system", "foreign income",
     "Above what amount of trading income must a sole trader send a tax return?", ["£1,000"]),
    # ---- ukps05 MOT ----
    ("ukps05", "The MOT test checks that your vehicle", "keep the same renewal date.",
     "By when must a new vehicle first get an MOT?", ["the third anniversary of its registration"]),
    ("ukps05", "The MOT test checks that your vehicle", "keep the same renewal date.",
     "How much can you be fined for driving without a valid MOT?", ["£1,000"]),
    ("ukps05", "The MOT test checks that your vehicle", "keep the same renewal date.",
     "How long does an MOT last?", ["a year"]),
    # ---- ukps06 Lasting power of attorney ----
    ("ukps06", "There are 2 types of LPA", "selling your home",
     "How much does it cost to apply to register a lasting power of attorney?", ["£92"]),
    ("ukps06", "There are 2 types of LPA", "selling your home",
     "What minimum age must you be to make a lasting power of attorney?", ["18 or over"]),
    ("ukps06", "There are 2 types of LPA", "selling your home",
     "Which body do you register a lasting power of attorney with?",
     ["the Office of the Public Guardian"]),
    # ---- ukps07 Renting out your property ----
    ("ukps07", "You're a landlord if you rent out your property.", "if it's in England",
     "What must a landlord provide showing the property's energy efficiency?",
     ["Energy Performance Certificate"]),
    ("ukps07", "The most common type of tenancy", "National Code of practice",
     "What is the most common type of tenancy?", ["assured periodic tenancy"]),
    ("ukps07", "You're a landlord if you rent out your property.", "if it's in England",
     "Where must a landlord protect a tenant's deposit?",
     ["a government-approved scheme"]),
    # ---- ukps08 Holiday entitlement ----
    ("ukps08", "Almost all people classed as workers", "5.6 weeks of holiday.",
     "How many weeks of paid holiday are most workers legally entitled to each year?",
     ["5.6 weeks"]),
    ("ukps08", "Almost all people classed as workers", "5.6 weeks of holiday.",
     "How many days of paid annual leave must most workers on a 5-day week receive?",
     ["at least 28 days"]),
    ("ukps08", "Limits on statutory leave", "28 days' paid holiday.",
     "What is the upper limit on statutory paid holiday entitlement?", ["28 days"]),
    # ---- ukps09 Register a birth ----
    ("ukps09", "All births in England, Wales and Northern Ireland", "leaves.",
     "Within how many days must a birth be registered in England, Wales and Northern Ireland?",
     ["within 42 days"]),
    # Multi-valid-answer items (second review pass, 24 Jul): the passage
    # lists several equally correct options, so EVERY option is recorded
    # as an acceptable reference span. The scorers take the max over
    # references, so a model naming any listed item scores correctly.
    ("ukps09", "When registering the birth, you should know", "mother's maiden surname",
     "What is one detail you should know when registering a birth?",
     ["place and date of the birth", "name, surname and sex of the baby",
      "parents' names, surnames and address",
      "places and dates of parents' birth",
      "date of parents' marriage or civil partnership",
      "parents' jobs", "mother's maiden surname"]),
    ("ukps09", "Once you've registered the birth", "Child Tax Credit",
     "What benefit may you be able to claim once a birth is registered?", ["Child Benefit"]),
    # ---- ukps10 Flu vaccine ----
    ("ukps10", "The flu vaccine is recommended for people at higher risk", "carer's allowance",
     "From what age can adults get the free NHS flu vaccine on age grounds alone?",
     ["aged 65 or over"]),
    ("ukps10", "The flu vaccine helps protect against flu", "flu jab in pregnancy.",
     "In which seasons is the flu vaccine offered on the NHS?", ["autumn or winter"]),
    ("ukps10", "The flu vaccine is recommended for people with certain long-term",
     "a body mass index (BMI) of 40 or above",
     "At what body mass index is someone eligible for the flu vaccine for being very overweight?",
     ["a body mass index (BMI) of 40 or above"]),
    # ---- ukps11 High blood pressure ----
    ("ukps11", "High blood pressure (also called hypertension)", "medicines can help you stay healthy.",
     "What is another name for high blood pressure?", ["hypertension"]),
    ("ukps11", "Non-urgent advice: Get your blood pressure checked", "have not had your blood pressure checked for more than 5 years",
     "Adults over what age should get their blood pressure checked if it has not been done for over 5 years?",
     ["aged 40 or over"]),
    ("ukps11", "High blood pressure does not usually cause any symptoms.", "get your blood pressure checked.",
     "How is high blood pressure usually detected, given it rarely causes symptoms?",
     ["get your blood pressure checked"]),
    # ---- ukps13 Chickenpox ----
    ("ukps13", "Chickenpox is a common infection", "can be serious in some people.",
     "Within how long does chickenpox usually get better on its own?", ["1 to 2 weeks"]),
    ("ukps13", "The main symptom of chickenpox", "loss of appetite",
     "What is the main symptom of chickenpox?", ["an itchy, spotty rash"]),
    ("ukps13", "Chickenpox happens in 3 stages", "forming a scab.",
     "In how many stages does chickenpox happen?", ["3 stages"]),
    # ---- ukps15 Sepsis ----
    ("ukps15", "Sepsis is a serious reaction to an infection", "urgent treatment in hospital.",
     "Where does sepsis need to be treated?", ["in hospital"]),
    ("ukps15", "Common symptoms of sepsis in adults", "peeing very little in the past 18 hours",
     "What is one common symptom of sepsis in adults?",
     ["confusion or slurred speech", "uncontrollable shivering",
      "muscle pain", "difficulty breathing",
      "blue, pale, grey or blotchy skin, lips or tongue",
      "a high or low temperature",
      "not peeing all day or peeing very little in the past 18 hours"]),
    ("ukps15", "Sepsis usually develops quickly", "which you can die from.",
     "What can sepsis turn into if it is not treated?", ["septic shock"]),
    # ---- ukps17 Online Safety Act ----
    ("ukps17", "The Online Safety Act 2023", "when it does appear.",
     "Which Act is the Online Safety Act explainer about?", ["The Online Safety Act 2023"]),
    ("ukps17", "The Online Safety Act 2023", "when it does appear.",
     "For whom have the strongest protections in the Act been designed?", ["children"]),
    ("ukps17", "The Online Safety Act 2023", "on their platforms.",
     "On which types of company does the Act put new duties?",
     ["social media companies and search services"]),
    # ---- ukps19 Cyber security breaches survey ----
    ("ukps19", "The Cyber Security Breaches Survey is a research study", "respond.",
     "With which strategy does the Cyber Security Breaches Survey align?",
     ["the National Cyber Strategy"]),
    ("ukps19", "For this latest release, the quantitative survey", "early 2024.",
     "When was the quantitative survey for the latest release carried out?", ["winter 2023/24"]),
    ("ukps19", "The Cyber Security Breaches Survey is a research study", "respond.",
     "Which types of organisation does the Cyber Security Breaches Survey study?",
     ["businesses, charities and educational institutions"]),
    # ---- ukps21 State Pension ----
    ("ukps21", "A National Insurance qualifying year is one", "voluntary National Insurance contributions",
     "What is one way to earn a National Insurance qualifying year?",
     ["worked and paid National Insurance",
      "got National Insurance Credits",
      "paid voluntary National Insurance contributions"]),
    ("ukps21", "You also need to be either a", "the new State Pension instead.",
     "A man must have been born before which date to get the basic State Pension?",
     ["6 April 1951"]),
    ("ukps21", "You also need to be either a", "the new State Pension instead.",
     "What will you claim if you were born on or after the basic State Pension birth dates?",
     ["the new State Pension"]),
    # ---- ukps22 Redundancy ----
    ("ukps22", "Redundancy is a form of dismissal", "reduce their workforce.",
     "What is redundancy a form of?", ["dismissal"]),
    ("ukps22", "You cannot be selected because of", "unfair dismissal.",
     "On what grounds can you not be selected for redundancy?",
     ["age, gender, or if you're disabled or pregnant", "age, gender"]),
    ("ukps22", "Commonly used methods are", "experience",
     "What is one commonly used method of selecting employees for redundancy?",
     ["last in, first out", "asking for volunteers", "disciplinary records",
      "staff appraisal markings, skills, qualifications and experience"]),
    # ---- ukps23 Maternity pay and leave ----
    ("ukps23", "Statutory Maternity Leave is 52 weeks.", "last 26 weeks",
     "How many weeks is Statutory Maternity Leave?", ["52 weeks"]),
    ("ukps23", "You do not have to take 52 weeks", "work in a factory).",
     "How many weeks of leave must you take after your baby is born?", ["2 weeks"]),
    ("ukps23", "Usually, the earliest you can start your leave", "expected week of childbirth.",
     "How many weeks before the expected week of childbirth can maternity leave usually start?",
     ["11 weeks"]),
    # ---- ukps24 Child Benefit ----
    ("ukps24", "You get Child Benefit if you're responsible", "approved education or training",
     "Up to what age can you claim Child Benefit for a child in approved education or training?",
     ["under 20"]),
    ("ukps24", "You'll get National Insurance credits automatically", "under 12.",
     "Up to what age of the child do you automatically get National Insurance credits from Child Benefit?",
     ["under 12"]),
    ("ukps24", "You get Child Benefit if you're responsible", "claim for.",
     "How many people can get Child Benefit for a single child?", ["Only one person"]),
    # ---- ukps26 Food poisoning ----
    ("ukps26", "Food poisoning is rarely serious", "at home.",
     "Within what time does food poisoning usually get better?", ["within a week"]),
    ("ukps26", "The most important thing is to have lots of fluids", "avoid dehydration.",
     "What is the most important thing to do when treating food poisoning at home?",
     ["have lots of fluids"]),
    ("ukps26", "Symptoms of food poisoning include", "feeling generally unwell",
     "What is one symptom of food poisoning?",
     ["feeling sick or being sick", "diarrhoea", "tummy pain",
      "a high temperature", "feeling generally unwell"]),
]

# =====================================================================
# SUMMARY SPEC: (doc, start_probe, end_probe, reference_summary)
# reference_summary authored to 40-120 words, faithful to the passage.
# =====================================================================
SUM = [
    ("ukps01", "You can get up to",
     "Statutory Sick Pay is money your employer pays you if you are too ill to "
     "work. Eligible employees receive £123.25 a week, or 80% of their normal "
     "weekly earnings if that is lower, for up to 28 weeks. It is paid for the "
     "full days you would normally have worked, in the same way as your usual "
     "wages, and tax and National Insurance are deducted. Employers use average "
     "weekly earnings over an eight-week period to calculate the amount, and you "
     "can still qualify if you have not yet received eight weeks of pay."),
    ("ukps02", "You will not be paid for doing jury service",
     "Jurors are not paid for jury service, but they can claim money back if "
     "their earnings are affected. For each day at court an employee can usually "
     "claim up to £64.95 towards loss of earnings and the cost of care or "
     "childcare, £5.71 for food and drink, and the cost of travel to and from "
     "court. The court explains how to claim expenses once jury service has "
     "ended. Employers must allow time off, though they can ask for the service "
     "to be delayed if the absence would seriously affect their business."),
    ("ukps03", "Universal Credit is replacing Housing Benefit",
     "Universal Credit is replacing several older benefits, including Housing "
     "Benefit and income-related Employment and Support Allowance. People already "
     "receiving these do not need to act unless their circumstances change or "
     "they get a Migration Notice, which sets a deadline to move across to keep "
     "their support. Claiming Universal Credit stops these legacy benefits, and "
     "it can also stop a partner's Pension Credit. Other benefits such as "
     "Personal Independence Payment or Carer's Allowance continue, though some "
     "benefits received alongside Universal Credit reduce the amount paid."),
    ("ukps04", "You must send a tax return if",
     "A Self Assessment tax return must be sent by anyone who, in the last tax "
     "year, was self-employed as a sole trader earning more than £1,000, was a "
     "partner in a business partnership, had to pay Capital Gains Tax, faced the "
     "High Income Child Benefit Charge outside PAYE, or was an off-payroll worker "
     "repaying a student or postgraduate loan. A return may also be needed for "
     "other untaxed income, such as money from renting out a property, tips and "
     "commission, income from savings, investments and dividends, or foreign "
     "income."),
    ("ukps05", "The MOT test checks that your vehicle",
     "The MOT test checks that a vehicle meets road safety and environmental "
     "standards. A vehicle must have its first MOT by the third anniversary of "
     "its registration, or by the anniversary of its last MOT once it is over "
     "three years old, though some vehicles are tested at one year old. If the "
     "MOT has run out, the owner should register the vehicle as off the road, "
     "book a test and tax it once it passes. A vehicle with an expired MOT "
     "cannot be driven or parked on the road, except to a repair or a "
     "pre-arranged test, and driving without one risks prosecution."),
    ("ukps06", "A lasting power of attorney (LPA) is a legal document",
     "A lasting power of attorney is a legal document that lets a donor appoint "
     "attorneys to help make decisions, or make decisions for them, if they lose "
     "mental capacity. There are two types: health and welfare, and property and "
     "financial affairs, and a person can make one or both. To make an LPA you "
     "must be over 18 and have mental capacity, and you register it with the "
     "Office of the Public Guardian for a fee of £92. The health and welfare "
     "type can only be used once you cannot make your own decisions, while the "
     "property and financial affairs type can be used as soon as it is "
     "registered."),
    ("ukps07", "The rules about what can happen",
     "The rules governing a tenancy depend on its type, which also affects how "
     "it can be ended. The most common type is an assured periodic tenancy, "
     "which applies where the landlord does not live in the property, it is the "
     "tenant's main home, and the tenant has their own room. A tenancy is not an "
     "assured periodic tenancy in cases such as purpose-built student "
     "accommodation, very high or very low rents, business tenancies or holiday "
     "lets. Excluded tenancies arise where the landlord shares rooms with the "
     "tenant, offering less protection from eviction, while tenancies from "
     "before 15 January 1989 may be regulated with stronger protection."),
    ("ukps08", "Almost all people classed as workers",
     "Almost all workers are legally entitled to 5.6 weeks of paid holiday a "
     "year, known as statutory annual leave, and this includes agency workers "
     "and those with irregular hours. Most people working a five-day week must "
     "receive at least 28 days of paid annual leave, which is the equivalent of "
     "5.6 weeks. Part-time workers with regular hours get the same 5.6 weeks, "
     "but this amounts to fewer than 28 days; someone working three days a week, "
     "for example, is entitled to at least 16.8 days. An employer can include "
     "bank holidays within this statutory entitlement."),
    ("ukps10", "The flu vaccine is recommended for people at higher risk",
     "The free NHS flu vaccine is offered each autumn or winter to people at "
     "higher risk of becoming seriously ill from flu. Those eligible include "
     "people aged 65 or over, pregnant women, people living in care homes, main "
     "carers, and those living with someone who has a weakened immune system. It "
     "is also recommended for people with certain long-term conditions, such as "
     "asthma needing steroid treatment, chronic obstructive pulmonary disease, "
     "heart or kidney disease, diabetes, a weakened immune system, or a body "
     "mass index of 40 or above. Frontline health and social care workers can "
     "get it through their employer."),
    ("ukps12", "Measles usually starts with cold-like symptoms",
     "Measles usually begins with cold-like symptoms, followed a few days later "
     "by a rash, and some people also get small white spots inside the mouth. "
     "The first symptoms include a high temperature, a runny or blocked nose, "
     "sneezing, a cough, and red, sore, watery eyes. Small white spots may then "
     "appear inside the cheeks and on the back of the lips, usually lasting a "
     "few days. The rash typically appears a few days after the cold-like "
     "symptoms, starting on the face and behind the ears before spreading to the "
     "rest of the body, and it can be harder to see on brown and black skin."),
    ("ukps13", "The main symptom of chickenpox",
     "The main symptom of chickenpox is an itchy, spotty rash that can appear "
     "anywhere on the body. Before or after the rash, a person may also have a "
     "high temperature, aches and pains, a general feeling of being unwell, and "
     "loss of appetite. Chickenpox happens in three stages, although new spots "
     "can appear while others are already becoming blisters or forming scabs. In "
     "the first stage small spots appear, which may be anywhere on the body, "
     "including inside the mouth and around the genitals, and can either spread "
     "or stay within a small area."),
    ("ukps14", "The main symptoms of norovirus",
     "Norovirus, sometimes called the winter vomiting bug, is a stomach bug "
     "whose main symptoms usually start suddenly and include feeling sick, being "
     "sick and diarrhoea. Some people also have a high temperature, a headache, "
     "tummy pain, and body aches. Despite its nickname, norovirus can occur at "
     "any time of year. It can usually be treated at home, and most people start "
     "to feel better within two to three days. The most important thing is to "
     "drink plenty of fluids to avoid dehydration, taking small sips when "
     "feeling sick, resting at home, and continuing to feed babies as normal."),
    ("ukps15", "Symptoms of sepsis in babies and children",
     "Sepsis in babies and children can show through several warning signs. "
     "These include difficulty breathing or breathing very fast, having a "
     "seizure, and a high or low temperature so that the child feels very hot or "
     "cold to the touch. The skin, lips or tongue may look blue, grey, pale or "
     "blotchy, which on darker skin can be easier to see on the palms or soles. "
     "Other signs are a rash that does not fade under pressure, being sleepier "
     "than normal or hard to wake, and not passing urine for many hours. Young "
     "children may also lose interest in feeding or keep vomiting."),
    ("ukps16", "A pro-innovation approach to AI regulation",
     "A pro-innovation approach to AI regulation is a UK government white paper "
     "produced by the Department for Science, Innovation and Technology and the "
     "Office for Artificial Intelligence. It was presented to Parliament by the "
     "Secretary of State for Science, Innovation and Technology by command of "
     "His Majesty in March 2023 and updated in August 2023. Published as Command "
     "Paper 815 under Crown copyright, it is licensed under the Open Government "
     "Licence except where otherwise stated. The document sets out the "
     "government's proposed framework for regulating artificial intelligence and "
     "includes consultation questions on transparency, contestability and how "
     "organisations manage AI-related risk."),
    ("ukps17", "The Online Safety Act 2023 (the Act) is a new set of laws",
     "The Online Safety Act 2023 is a set of laws designed to protect children "
     "and adults online. It places new duties on social media companies and "
     "search services, making them more responsible for their users' safety. "
     "Providers must put in place systems and processes to reduce the risk of "
     "their services being used for illegal activity, and must take down illegal "
     "content when it appears. The strongest protections are designed for "
     "children, requiring platforms to prevent them from accessing harmful and "
     "age-inappropriate content and to give parents and children clear ways to "
     "report problems."),
    ("ukps18", "In the dark days of 2020 and 2021",
     "The foreword to the UK Biological Security Strategy reflects on the "
     "COVID-19 pandemic as the context for renewed attention to biological "
     "threats. It recalls the devastating impact of a novel infectious disease "
     "spreading across the world in 2020 and 2021, noting that the pandemic had "
     "killed over 200,000 people in the UK and close to seven million globally. "
     "It describes how the outbreak overwhelmed health systems, damaged "
     "economies and harmed livelihoods, and frames the pandemic as the biggest "
     "crisis the UK had faced in generations and the greatest peacetime "
     "challenge in a century."),
    ("ukps19", "The Cyber Security Breaches Survey is a research study",
     "The Cyber Security Breaches Survey is a research study that supports UK "
     "cyber resilience and aligns with the National Cyber Strategy. It is used "
     "mainly to inform government policy on cyber security and to help make UK "
     "cyberspace a secure place to do business. The study examines the policies, "
     "processes and overall approach to cyber security among businesses, "
     "charities and educational institutions. It also considers the different "
     "cyber attacks and cyber crimes these organisations face, as well as how "
     "they are affected by such incidents and how they respond to them."),
    ("ukps20", "Over the next ten years, the impact of AI",
     "The National AI Strategy sets out a ten-year plan to make Britain a global "
     "AI superpower. It argues that over the next decade the impact of "
     "artificial intelligence on businesses across the UK and the wider world "
     "will be profound, and notes that UK universities and startups are already "
     "leading in building the tools of the new economy. New discoveries and "
     "methods for harnessing machine learning emerge continually from "
     "universities and businesses. The strategy frames AI as an opportunity to "
     "grow and transform businesses of all sizes and to capture the benefits of "
     "innovation right across the UK."),
    ("ukps21", "A National Insurance qualifying year is one",
     "Entitlement to the basic State Pension depends on having enough National "
     "Insurance qualifying years. A qualifying year is one in which a person did "
     "at least one of a number of things: worked and paid National Insurance, "
     "received National Insurance credits, or paid voluntary National Insurance "
     "contributions. Credits can be given, for example, to people who were "
     "unemployed, sick, or acting as a parent or carer. The number of qualifying "
     "years needed to receive any basic State Pension depends on the person's "
     "circumstances, including whether they are a man or a woman and their year "
     "of birth."),
    ("ukps22", "Your employer should use a fair",
     "When selecting employees for redundancy, an employer should use a fair and "
     "objective method. Commonly used approaches include last in, first out, "
     "where those with the shortest service are chosen first, asking for "
     "volunteers, looking at disciplinary records, and considering staff "
     "appraisal markings, skills, qualifications and experience. In some "
     "situations an employer can make someone redundant without a selection "
     "process, for example where the job no longer exists because a whole "
     "operation is closing down and all the employees working in it are being "
     "made redundant."),
    ("ukps23", "When you take time off to have a baby",
     "When taking time off to have a baby, a person may be eligible for a range "
     "of support, including Statutory Maternity Leave, Statutory Maternity Pay, "
     "paid time off for antenatal care, and extra help from the government. "
     "There are rules on when and how to claim paid leave and on changing the "
     "chosen dates, and a maternity planner can help work out entitlements. "
     "Some leave may be taken as Shared Parental Leave and Pay. Employment "
     "rights are protected during Statutory Maternity Leave, including the right "
     "to pay rises, to build up holiday, and to return to work."),
    ("ukps24", "You get Child Benefit if you're responsible",
     "Child Benefit is paid to someone responsible for bringing up a child who "
     "is under 16, or under 20 if they stay in approved education or training. "
     "Only one person can claim it for a given child, but there is no limit on "
     "how many children a person can claim for. Claiming brings several "
     "advantages: an allowance usually paid every four weeks for each child, "
     "National Insurance credits that count towards the claimant's State "
     "Pension, and a National Insurance number issued to the child without them "
     "having to apply. Even those who opt out of payments should still claim to "
     "gain the other advantages."),
    ("ukps25", "The main symptoms of asthma",
     "Asthma is a common condition affecting breathing that cannot currently be "
     "cured, though good treatment should keep symptoms under control. Its main "
     "symptoms are breathing problems such as wheezing, coughing, shortness of "
     "breath, and a tight chest. These symptoms can be mild or severe, tend to "
     "come and go, and are often worse at night and early in the morning. A "
     "severe episode is called an asthma attack and can be life-threatening. "
     "Symptoms can be triggered by different things, including exercise, high "
     "levels of air pollution, cold air, or contact with an allergen such as "
     "pollen, dust, mould or animals."),
    ("ukps11", "High blood pressure is very common",
     "High blood pressure, or hypertension, is very common, especially in older "
     "adults, and usually causes no symptoms, so many people do not realise they "
     "have it. Several things raise the risk, including increasing age, having "
     "close relatives with high blood pressure, and an ethnic background that is "
     "Black African, Black Caribbean or South Asian. Lifestyle factors also "
     "contribute, such as an unhealthy diet high in salt, being overweight, "
     "smoking, drinking too much alcohol, and prolonged stress. Because there are "
     "usually no symptoms, the guidance advises people who may be at risk, and "
     "those aged 40 or over who have not been checked for more than five years, "
     "to have their blood pressure measured."),
    ("ukps26", "If you or your child have food poisoning",
     "Food poisoning is rarely serious and usually gets better within a week, so "
     "most people can treat themselves or their child at home. The symptoms, "
     "which include feeling or being sick, diarrhoea, tummy pain and a high "
     "temperature, generally improve within about a week. The most important "
     "step is to drink plenty of fluids to avoid dehydration, taking small sips "
     "if feeling sick, and getting plenty of rest at home. Babies should carry "
     "on being breast or bottle fed as normal, with smaller, more frequent "
     "feeds offered if they are being sick."),
]


def main() -> None:
    items = []
    per_doc: dict[str, int] = {}
    for i, (d, s, e, q, probes) in enumerate(QA, 1):
        p = qa_passage(d, s, e)
        if len(p) > 6000:
            raise SystemExit(f"[items] QA {i}: passage {len(p)} chars > 6000 ({d})")
        ans = answers(p, probes)
        items.append({"id": f"ukps-qa-{i:03d}", "type": "qa",
                      "source_doc": d, "passage": p, "question": q,
                      "answers": ans})
        per_doc[d] = per_doc.get(d, 0) + 1
    for i, (d, s, summary) in enumerate(SUM, 1):
        p = passage_min(d, s)
        if not 1500 <= len(p) <= 6000:
            raise SystemExit(f"[items] SUM {i}: passage {len(p)} chars "
                             f"outside 1500-6000 ({d})")
        words = len(summary.split())
        if not 40 <= words <= 120:
            raise SystemExit(f"[items] SUM {i}: summary {words} words "
                             f"outside 40-120 ({d})")
        items.append({"id": f"ukps-sum-{i:03d}", "type": "summary",
                      "source_doc": d, "passage": p,
                      "reference_summary": summary})

    # protocol checks: >= 15 distinct QA docs, <= 5 QA items per doc
    if len(per_doc) < 15:
        raise SystemExit(f"[items] QA covers {len(per_doc)} docs, need >= 15")
    over = {d: n for d, n in per_doc.items() if n > 5}
    if over:
        raise SystemExit(f"[items] more than 5 QA items in: {over}")

    with open(OUT, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    n_qa = sum(1 for it in items if it["type"] == "qa")
    n_sum = sum(1 for it in items if it["type"] == "summary")
    print(f"[items] {n_qa} QA + {n_sum} summaries -> {OUT}")
    print(f"[items] QA spans {len(per_doc)} documents, "
          f"max {max(per_doc.values())} per document")


if __name__ == "__main__":
    main()
