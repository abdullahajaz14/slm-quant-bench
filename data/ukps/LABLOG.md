# UKPS corpus lab log

Chronological record required by data/ukps/PROTOCOL.md (authoring
dates for the blind rule) and by the pilot-freeze checklist.

## 19 July 2026: document collection (first batch, 20 documents)

- Fetched via `scripts/corpus_fetch.py` (urllib + html.parser main-
  content extraction; deterministic, no model paraphrase anywhere).
- Register spread: guidance ukps01-ukps09 (GOV.UK guides, /print
  aggregate pages used for text, base pages used for the licence
  footer); public-health information ukps10-ukps15 (NHS conditions and
  vaccination pages); departmental reports and policy papers
  ukps16-ukps20 (AI white paper, Online Safety Act explainer, UK
  Biological Security Strategy, Cyber Security Breaches Survey 2024,
  National AI Strategy).
- Licence verification: GOV.UK pages carry the Open Government Licence
  v3.0 statement in the page footer (checked per page by the fetch
  script). NHS pages do not carry a per-page statement; the licence
  basis is the NHS website terms and conditions ("Copyright and
  database rights in NHS Website Content are released free-of-charge
  under the current version of the Open Government Licence"), verified
  on 19 July 2026 and checked per fetch against that page. No page
  containing personal data was selected.
- One rejected candidate recorded: the Data (Use and Access) Act 2025
  factsheets URL is a landing page (no substantive HTML content) and
  was removed. A withdrawn publication (Generative AI Framework for
  HMG) was fetched, identified as withdrawn from its title, and
  replaced by the UK Biological Security Strategy.
- Count status: 20 of the protocol's 25-40 target. Top-up to at least
  25 planned during the second review pass; QA coverage rules (at
  least 15 distinct documents, at most 5 items per document) are
  already satisfiable.

## 19 July 2026: content cleaning pass

- `scripts/corpus_clean.py` removed 21 non-content blocks across the
  stored texts: stock-photo credit lines with agency URLs on the NHS
  condition pages (ukps12: 6, ukps13: 10; these image credits are also
  outside the NHS OGL grant, which excepts images), and the
  boilerplate "This publication is available at ..." line on the five
  GOV.UK publication documents. Remaining URL-bearing blocks were
  reviewed and are legitimate prose. The clean pass is idempotent and
  is rerun after any future fetch.

## 19 July 2026: battery prompt set authored

- `data/battery.jsonl` built by `scripts/build_battery.py` from the
  corpus documents only: short prompts are authored questions with
  brief context; medium and long prompts are paragraph-aligned
  verbatim excerpts from the documents plus a generic realistic
  instruction (summarise / list key points), per
  data/BATTERY_PROTOCOL.md. Band compliance validated by the script.
- No model has been run on any corpus text as of this date.

## 19 July 2026: corpus top-up and item authoring

- Corpus extended from 20 to **26 documents** (added State Pension,
  Redundancy, Maternity pay, Child Benefit guidance; Asthma and Food
  poisoning NHS pages). Register spread preserved; all OGL-verified and
  cleaned.
- **items.jsonl authored: 60 QA items + 25 reference summaries**, built
  by `scripts/build_ukps_items.py`. Integrity is enforced by
  construction: every passage is sliced verbatim from its source
  document by marker phrases, and every QA answer is located inside its
  passage and stored as the exact matched substring. The independent
  `UkpsAdapter` load then re-checks the span rule and accepted all 85
  items with zero drops.
- **BLIND RULE honoured: all 25 reference summaries were authored on
  this date, 19 July 2026, before any model was run on the corpus.** No
  model output has touched any corpus text.
- Bands confirmed: QA passages 451-2446 chars (<= 6000); summary
  passages 1701-1956 chars (1500-6000); reference summaries 84-113
  words (40-120). QA spans 20 distinct documents, max 3 items per
  document (protocol: >= 15 docs, <= 5 per doc).
- A first content-review pass (reading items against sources) was done
  during authoring.

## 24 July 2026: second review pass (protocol requirement)

Conducted five days after authoring, as the protocol requires (a later
day than authoring, fresh eyes). Every item was re-checked against its
source document. Findings and actions:

1. **Scoring-validity defect found and corrected in 5 QA items.** Items
   phrased "what is one X" (registering a birth, sepsis symptoms,
   National Insurance qualifying years, redundancy selection methods,
   food poisoning symptoms) recorded only a single reference answer,
   although their passages list several equally correct options. A model
   naming any other listed option would have scored zero on exact match
   and poorly on token F1, injecting systematic noise into the very
   measurements the study reports. All valid options are now recorded as
   multiple reference spans (26 spans added), which the protocol
   explicitly permits and which the scorers already support, since EM
   and F1 both take the maximum over references.
2. **One item broadened** (redundancy selection grounds, ukps-qa-050):
   the passage states "age, gender, or if you're disabled or pregnant";
   the fuller span is now accepted alongside the shorter one.
3. **Summary faithfulness verified.** Every numeric claim in all 25
   reference summaries was checked against its passage; three apparent
   mismatches were confirmed to be punctuation artefacts of the check
   (£1,000, £92, under 16 all appear verbatim in their passages). No
   unsupported factual claims found.
4. **Re-validation after the edits:** the builder's verbatim-span
   enforcement passed, and the independent `UkpsAdapter` load accepted
   all 85 items with zero drops.

No source document text was altered at any point, per protocol.

## 8 August 2026: PROTOCOL FREEZE

The pilot is complete on both sides (600 device records, 450 reference
environment records) and the protocol is frozen per Chapter 3. From this
point, changes to any frozen artefact are logged protocol deviations.

Frozen artefacts and their SHA-256 sums:

| Artefact | Contents | SHA-256 |
|---|---|---|
| `data/ukps/items.jsonl` | 60 QA items, 25 reference summaries, from 26 documents | `73f888a01f4152dfbdb729951a1a3a632ba5c0d8225856719e1b31ee2fe1befc` |
| `data/battery.jsonl` | 30 prompts, 10 per band | `502fa818ebac9aa032646c021e48516038fe6b19e30db623fd95fdb7104fb9c7` |

Also frozen at this point: the sample sizes of Chapter 3's Table 3.2 at
their planned values, the decoding settings (greedy, temperature zero,
per-task output limits), the prompt templates and chat wrappers, and the
composition of the battery.

Freeze checklist:

- [x] Sample sizes confirmed at planned values; two projected interval
      widths flagged and reported rather than adjusted
- [x] Decoding settings confirmed: greedy, temperature zero, per-task
      maximum output tokens as configured
- [x] `battery.jsonl` committed and frozen (checksum above)
- [x] `data/ukps/items.jsonl` committed and frozen (checksum above)
- [x] Divergences from Chapter 3's planned values recorded in
      `results/pilot_findings.md` and carried into the chapter
