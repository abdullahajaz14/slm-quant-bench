# Battery prompt set: construction protocol (data/battery.jsonl)

Fixed at the pilot freeze; never edited afterwards. You author the 30
prompts from the OGL public-sector documents already gathered for the
curated corpus, so the battery measures the register of text the study
is about.

## Format: one JSON object per line
{"id": "S01", "band": "short", "prompt": "...", "max_tokens": 64}

## Bands (10 prompts each)
- short:  prompt of roughly 200-300 characters (~50 tokens), max_tokens 64.
  Style: a direct question with one or two sentences of context.
- medium: roughly 1,200-1,500 characters (~300 tokens), max_tokens 128.
  Style: an excerpt plus an instruction (answer from the excerpt,
  or summarise the excerpt).
- long:   roughly 4,000-4,500 characters (~1,000 tokens), max_tokens 128.
  Style: a full notice or guidance section plus an instruction.

## Rules
1. Ids S01-S10, M01-M10, L01-L10; band field must match the prefix.
2. Realistic tasks only (summarise, answer, extract); no trivia, no
   creative prompts; this battery represents enterprise inference.
3. Text drawn only from the OGL documents in data/ukps/documents/;
   note the source document id in an optional "source" field.
4. Prompt text is the raw user message; the harness applies each
   model's chat wrapper, so do not include any chat tokens.
5. Once battery.jsonl is committed at the freeze, changes require a
   logged protocol deviation (Chapter 3 rule), so proofread it once.
