# On-device RAG prototype: transcript

Deliverable O5, demonstrative only and not part of the controlled
experiment. Retrieval is TF-IDF cosine over paragraph chunks of the
curated corpus; generation runs through the same quantised artefact
and backend the study measured. Run on the target device
(MacBook Air M2, 8 GB) with Gemma 2 2B at Q4_K_M.

Decode rates below are single short generations and are NOT
comparable with the efficiency battery in Chapter 4, whose figures
are medians over 90 generations spanning three context bands.

Run: 2026-08-10 16:14 BST

## Q: How much Statutory Sick Pay can you get per week?
```
[rag] retrieved 3 chunk(s):
  ukps01  (cosine 0.519)
  ukps22  (cosine 0.340)
  ukps23  (cosine 0.308)
[rag] model loaded in 0.6s; generating...

Answer:
£123.25

(9 tokens at 31.8 tok/s decode; documents never left this machine)
```

## Q: How many weeks of paid holiday are most workers legally entitled to each year?
```
[rag] retrieved 3 chunk(s):
  ukps08  (cosine 0.440)
  ukps08  (cosine 0.294)
  ukps08  (cosine 0.251)
[rag] model loaded in 0.4s; generating...

Answer:
5.6

(5 tokens at 26.9 tok/s decode; documents never left this machine)
```

## Q: By when must you pay your Self Assessment bill?
```
[rag] retrieved 3 chunk(s):
  ukps04  (cosine 0.477)
  ukps04  (cosine 0.376)
  ukps04  (cosine 0.337)
[rag] model loaded in 0.4s; generating...

Answer:
31 July

(5 tokens at 26.7 tok/s decode; documents never left this machine)
```

