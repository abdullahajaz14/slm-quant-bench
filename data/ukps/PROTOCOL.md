# Curated UK public-sector corpus: construction protocol

This file is the written protocol Chapter 3 (3.4) promises; it goes into
the dissertation appendices verbatim, so keep it current. The corpus is
a realism check, deliberately modest (Chapter 3's words), not a headline
benchmark.

## 1. Document selection (25-40 documents)
1. Sources: GOV.UK and NHS pages published under the Open Government
   Licence v3.0 only. Verify the licence statement on each page.
2. Spread across three registers: guidance (e.g. how-to pages),
   public-health information, departmental reports or policy papers.
3. For each document record in `documents/INDEX.csv`:
   `doc_id,title,url,publisher,licence,access_date`.
4. Save the extracted text as `documents/<doc_id>.txt` (plain text,
   no navigation chrome). No personal data anywhere; if a page contains
   any, choose another page.

## 2. QA items (target 60), file: items.jsonl, type "qa"
Schema per line:
`{"id":"ukps-qa-001","type":"qa","source_doc":"doc03",
  "passage":"...","question":"...","answers":["..."]}`
Rules:
1. Passage: a self-contained excerpt of at most ~6,000 characters.
2. Span-based: every answer string must appear VERBATIM in the passage
   (the adapter enforces this and fails the load otherwise).
3. Answerable from the passage alone; no outside knowledge required.
4. Cover at least 15 distinct documents; at most 5 items per document.
5. Multiple acceptable spans go in as multiple answers.

## 3. Summary items (target 25), type "summary"
Schema per line:
`{"id":"ukps-sum-001","type":"summary","source_doc":"doc07",
  "passage":"...","reference_summary":"..."}`
Rules:
1. Passage: a self-contained section of 1,500-6,000 characters.
2. Reference: 3-5 sentences, 40-120 words, covering the key points a
   competent reader would extract.
3. BLIND RULE: write every reference summary BEFORE running any model
   on the corpus. References written after seeing model output are
   contaminated; note the authoring date in your lab log.

## 4. Review pass
Every item is re-checked against its source document in a second pass
on a later day; fix or delete, never patch the source text.

## 5. Freeze
items.jsonl freezes with the pilot (Chapter 3 rule). Later changes are
logged protocol deviations.

## 6. Two template strings to add to prompts.py TASK_TEMPLATES
(paste verbatim; protocol data, same status as the existing strings)

    "grounded_qa": (
        "Read the passage and answer the question using only the "
        "passage. Reply with the exact answer text from the passage, "
        "no explanation.\n\nPassage:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    "document_summarise": (
        "Summarise the following public-sector document section in 3 "
        "to 5 sentences covering the key points. Reply with the "
        "summary only.\n\nDocument section:\n{context}\n\nSummary:"
    ),
