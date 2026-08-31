"""scripts/build_battery.py: build data/battery.jsonl per protocol.

Implements data/BATTERY_PROTOCOL.md: 30 prompts in three bands, all
text drawn from the OGL corpus documents. Short prompts are authored
direct questions with one or two sentences of context; medium and long
prompts are paragraph-aligned VERBATIM excerpts from the documents plus
a generic realistic instruction (summarise / list key points), so no
instruction can contradict its excerpt. Band compliance is validated
before writing; the script fails loudly if any prompt is out of band.

Once battery.jsonl is committed at the pilot freeze, changes require a
logged protocol deviation. Run from the repository root:
  python scripts/build_battery.py
"""

from __future__ import annotations

import json
import os

DOCS = os.path.join("data", "ukps", "documents")
OUT = os.path.join("data", "battery.jsonl")

BANDS = {"short": (200, 300, 64),
         "medium": (1200, 1500, 128),
         "long": (4000, 4500, 128)}

MED_SUM = ("Summarise the following extract from official guidance in "
           "2 to 3 sentences.\n\n")
MED_LIST = ("List the main requirements or steps described in the "
            "following extract, as short bullet points.\n\n")
LONG_SUM = ("Summarise the following section of an official document "
            "in 4 to 5 sentences covering its key points.\n\n")
LONG_LIST = ("Extract the main recommendations or requirements stated "
             "in the following section, as a short list.\n\n")

# --- Short band: authored questions grounded in corpus topics ----------
SHORT: list[tuple[str, str, str]] = [
    ("S01", "ukps01",
     "An employee has been off sick from work for six days and earns 150 "
     "pounds a week. Statutory Sick Pay is paid by employers for up to "
     "28 weeks to eligible employees. Explain in two sentences whether "
     "this employee is likely to qualify and what they should do next."),
    ("S02", "ukps02",
     "You have received a jury summons for a trial that starts during a "
     "holiday you booked months ago. Jury service normally lasts up to "
     "two weeks. Briefly explain what you should do about the summons "
     "and whether it can be moved to a later date."),
    ("S03", "ukps03",
     "Universal Credit is a monthly payment to help with living costs "
     "for people on a low income or out of work. A claimant has just "
     "started a part-time job on the minimum wage. In two sentences, "
     "explain how starting work generally affects their award."),
    ("S04", "ukps04",
     "Self Assessment is the system HMRC uses to collect Income Tax "
     "that is not taken automatically. A freelancer missed the 31 "
     "January online filing deadline by one week. Briefly state what "
     "penalty applies and what they should do now."),
    ("S05", "ukps05",
     "An MOT checks that a vehicle meets road safety and environmental "
     "standards. A driver's MOT certificate expired yesterday and the "
     "car is parked on a public road. In one or two sentences, explain "
     "whether the car can legally be driven and to where."),
    ("S06", "ukps06",
     "A lasting power of attorney lets someone you trust make decisions "
     "for you if you lose mental capacity. Briefly explain the "
     "difference between the two types of lasting power of attorney "
     "and when each type can be used by the attorney."),
    ("S07", "ukps07",
     "A landlord is preparing to let a flat to new tenants and has "
     "taken a five week deposit. Deposits for assured shorthold "
     "tenancies must be protected. In two sentences, state what the "
     "landlord must do with the deposit and by when."),
    ("S08", "ukps08",
     "Almost all workers are legally entitled to paid holiday each "
     "year. An employee works a regular five day week. State their "
     "minimum annual statutory paid holiday entitlement in days and "
     "whether bank holidays must be given on top. Answer briefly."),
    ("S09", "ukps10",
     "The flu vaccine is offered free on the NHS every year to people "
     "at higher risk from flu. A healthy 66 year old asks whether they "
     "qualify for a free NHS flu vaccine this autumn. Answer in one or "
     "two sentences using standard NHS eligibility rules."),
    ("S10", "ukps11",
     "High blood pressure rarely has noticeable symptoms but raises "
     "the risk of heart attack and stroke. A patient aged 42 asks how "
     "often adults should have their blood pressure checked and where "
     "this can be done. Reply in one or two sentences."),
]

# --- Medium and long bands: (id, doc, start_char, instruction) ---------
MEDIUM = [
    ("M01", "ukps01", 0, MED_SUM),
    ("M02", "ukps02", 1500, MED_LIST),
    ("M03", "ukps03", 0, MED_SUM),
    ("M04", "ukps04", 2000, MED_LIST),
    ("M05", "ukps05", 0, MED_SUM),
    ("M06", "ukps06", 3000, MED_LIST),
    ("M07", "ukps07", 1000, MED_SUM),
    ("M08", "ukps08", 0, MED_LIST),
    ("M09", "ukps13", 500, MED_SUM),
    ("M10", "ukps17", 2000, MED_SUM),
]
LONG = [
    ("L01", "ukps02", 2000, LONG_SUM),
    ("L02", "ukps03", 5000, LONG_LIST),
    ("L03", "ukps04", 3000, LONG_SUM),
    ("L04", "ukps06", 2000, LONG_LIST),
    ("L05", "ukps07", 4000, LONG_SUM),
    ("L06", "ukps16", 6000, LONG_SUM),
    ("L07", "ukps17", 3000, LONG_LIST),
    ("L08", "ukps18", 9000, LONG_SUM),
    ("L09", "ukps19", 12000, LONG_LIST),
    ("L10", "ukps20", 3000, LONG_SUM),
]


def doc_text(doc_id: str) -> str:
    with open(os.path.join(DOCS, f"{doc_id}.txt")) as f:
        return f.read()


def excerpt(doc_id: str, start_char: int, instruction: str,
            band: str) -> str:
    """Paragraph-aligned verbatim excerpt sized so the whole prompt
    (instruction + excerpt) lands inside the band."""
    lo, hi, _ = BANDS[band]
    paragraphs = doc_text(doc_id).split("\n\n")
    # advance to the paragraph containing start_char
    offset = 0
    start_index = 0
    for i, para in enumerate(paragraphs):
        if offset >= start_char:
            start_index = i
            break
        offset += len(para) + 2
    chosen: list[str] = []
    length = len(instruction)
    for para in paragraphs[start_index:]:
        added = len(para) + (2 if chosen else 0)
        if length + added > hi:
            if length < lo:
                # still under band: take the head of this paragraph up
                # to the last clean break that fits (text stays
                # verbatim, just ends early); prefer sentence ends,
                # then bullet/clause breaks, then a word boundary
                budget = hi - length - (2 if chosen else 0)
                head = para[:budget]
                for sep in (". ", "! ", "? ", "• ", "; ", " "):
                    cut = head.rfind(sep)
                    if cut > 0:
                        chosen.append(head[:cut + (1 if sep != " " else 0)]
                                      .rstrip())
                        length = len(instruction) + sum(
                            len(c) for c in chosen) + 2 * (len(chosen) - 1)
                        break
            break
        chosen.append(para)
        length += added
    prompt = instruction + "\n\n".join(chosen)
    if not lo <= len(prompt) <= hi:
        raise SystemExit(
            f"[battery] {doc_id} start={start_char} band={band}: prompt "
            f"is {len(prompt)} chars, outside [{lo}, {hi}]; adjust the "
            f"start offset in the spec")
    return prompt


def main() -> None:
    rows = []
    for pid, doc, text in SHORT:
        lo, hi, max_tokens = BANDS["short"]
        if not lo <= len(text) <= hi:
            raise SystemExit(f"[battery] {pid}: {len(text)} chars, "
                             f"outside short band [{lo}, {hi}]")
        rows.append({"id": pid, "band": "short", "prompt": text,
                     "max_tokens": max_tokens, "source": doc})
    for spec, band in ((MEDIUM, "medium"), (LONG, "long")):
        for pid, doc, start, instruction in spec:
            prompt = excerpt(doc, start, instruction, band)
            rows.append({"id": pid, "band": band, "prompt": prompt,
                         "max_tokens": BANDS[band][2], "source": doc})
    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[battery] {len(rows)} prompts written to {OUT}")
    for row in rows:
        print(f"  {row['id']} {row['band']:<6} {len(row['prompt']):>5} "
              f"chars  (source {row['source']})")


if __name__ == "__main__":
    main()
