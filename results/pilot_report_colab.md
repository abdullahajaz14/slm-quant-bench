# Pilot report (protocol-freeze record)

## Pilot findings and protocol decisions

Written 5-6 August 2026 from the device-side pilot
(`results/pilot-device.jsonl`, 600 records across four configurations)
and two controlled follow-up experiments. This document is part of the
protocol-freeze record; `scripts/pilot.py` folds it into
`results/pilot_report.md`.

## 1. Benchmark identifiers required correction

Two of the four planned dataset identifiers no longer resolve, and a
third ships a loading script that the `datasets` library will no longer
execute. The working identifiers, confirmed by loading each one, are
recorded in the repository README. This is an infrastructure
correction, not a design change: the datasets, configurations, splits
and sample sizes are those the methodology specifies, and the row
counts match, including exactly 817 TruthfulQA questions.

## 2. A feasibility limit on the 8 GB endpoint

Phi-3 Mini at Q8\_0 failed every item of the pilot with
`llama_decode returned -3`. The failure was diagnosed rather than
recorded at face value:

| Configuration | Outcome |
|---|---|
| Full offload (`n_gpu_layers=-1`) | fails |
| 28 layers offloaded | fails |
| 24 layers offloaded | loads; ~0.05 tokens/s |
| 20 layers offloaded | loads; ~0.04 tokens/s |
| CPU only (0 layers) | loads; ~0.23 tokens/s |

The same behaviour appears when the artefact is read from internal
storage, so it is not a storage effect. It is memory exhaustion: 3.8 GB
of weights, plus the KV cache, plus the operating system, exceeds
8 GB, and the machine swaps. Gemma 2 2B at Q8\_0 (2.6 GB) runs
normally at 25.6 tokens/s on the same machine, so the practical
boundary for this device class lies between roughly 2.6 GB and 3.8 GB
of quantised weights.

**Protocol decision.** Full offload remains the study's standard
configuration. Where it fails, the run records that failure verbatim as
the feasibility result, and then measures the configuration again at
reduced offload, recorded as a supplementary row carrying `offload` and
`n_gpu_layers`. This preserves the distinction between a configuration
that cannot run at all and one that runs far too slowly to deploy,
which is a distinction a deploying organisation needs. Chapter 3
already admits partial offload as a technique at reference precision,
so this extends an established device rather than introducing one.

## 3. Storage placement affects load time, not throughput

The same 2.6 GB artefact was measured from internal storage and from
the external USB volume, with all else held constant:

| Metric | External USB | Internal | Ratio |
|---|---|---|---|
| Cold-start load | 70.5 s | 3.1 s | 23x |
| Prefill rate | 461.5 tok/s | 466.2 tok/s | 1.01x |
| Decode rate | 25.58 tok/s | 25.62 tok/s | 1.00x |

Cold-start load time is a reported metric and is affected by more than
an order of magnitude; throughput is not affected at all once the model
is resident. **Protocol decision.** The efficiency battery measures
artefacts staged on internal storage, one at a time, removed after
measurement. Model building and archival storage may remain external.

## 4. Planned sample sizes are supported

Projected confidence-interval widths at the planned sample sizes are
within the 0.10 planning threshold for the public benchmarks:
CNN/DailyMail 0.02, CUAD 0.07 to 0.10, HotpotQA token F1 0.08,
TruthfulQA 0.06 to 0.07. Two are flagged and both are reportable rather
than remediable. HotpotQA exact match projects to 0.104, marginally
over threshold, while its token F1, the primary metric for that task,
is comfortable at 0.078; exact match is binary and therefore noisier by
construction. The curated corpus projects to 0.17 to 0.21 at its
authored size of 60 items, which follows from the size of a corpus
Chapter 3 describes as deliberately modest and positions as a realism
check rather than a headline benchmark. **Protocol decision.** Sample
sizes are frozen at the planned values; the two flagged widths are
reported with the results.

## 5. Quality behaviour is coherent

Across the three configurations that completed, Q8\_0 outscored
Q4\_K\_M on most tasks (HotpotQA token F1 0.795 against 0.709; CUAD
0.413 against 0.336; TruthfulQA MC1 0.462 against 0.385), with one
reversal on the curated corpus. Task-dependent, non-uniform degradation
of this kind is what Chapter 2 leads the study to expect, and it is
already visible at pilot sample sizes.

## 6. Measured runtime, and a corrected projection

The Colab-side pilot ran Phi-3 Mini at all three precisions across all
six tasks, 25 items each: 450 records, no errors. Projected to the
planned sample sizes:

| Precision | Hours per model |
|---|---|
| Q4\_K\_M | 0.60 |
| Q8\_0 | 0.68 |
| FP16 | 0.81 |
| **All three** | **2.09** |

which gives roughly **8.4 hours for the full four-model quality sweep**
on the reference environment. Mistral 7B, being about twice the size of
the pilot model and requiring partial offload at reference precision on
a 16 GB accelerator, should be expected to exceed this per-model figure.

Two consequences. First, the free tier of the reference environment is
sufficient: a model completes inside a single session, and results are
committed after each precision, so an interrupted session resumes
rather than repeats. No paid tier or alternative hardware is required.

Second, this corrects a planning estimate made before measurement. That
estimate put the sweep at 25 to 40 hours on the assumption that
TruthfulQA's option-scoring would dominate, since it contributes some
13,000 log-likelihood evaluations. Measured, each evaluation takes 0.25
to 0.43 seconds, so the whole task costs 4 to 6 minutes per precision.
Chapter 3's assertion that choice scoring is computationally cheap is
therefore confirmed empirically rather than assumed, and the planning
estimate was pessimistic by a factor of three to five. The measured
figures supersede it.


---

## 1. Runtime projection

| model | precision | task | backend | n_pilot | median_s | planned_n | projected_min |
|---|---|---|---|---|---|---|---|
| phi3-mini | fp16 | cnndm | cuda | 25 | 5.66 | 150 | 14.1 |
| phi3-mini | fp16 | cuad | cuda | 25 | 4.27 | 200 | 14.2 |
| phi3-mini | fp16 | hotpotqa | cuda | 25 | 1.48 | 300 | 7.4 |
| phi3-mini | fp16 | truthfulqa_mc1 | cuda | 13 | 0.29 | 817 | 3.9 |
| phi3-mini | fp16 | truthfulqa_mc2 | cuda | 12 | 0.43 | 817 | 5.9 |
| phi3-mini | fp16 | ukps_qa | cuda | 25 | 0.71 | 60 | 0.7 |
| phi3-mini | fp16 | ukps_sum | cuda | 25 | 5.97 | 25 | 2.5 |
| phi3-mini | q4_k_m | cnndm | cuda | 25 | 3.38 | 150 | 8.4 |
| phi3-mini | q4_k_m | cuad | cuda | 25 | 3.11 | 200 | 10.4 |
| phi3-mini | q4_k_m | hotpotqa | cuda | 25 | 1.40 | 300 | 7.0 |
| phi3-mini | q4_k_m | truthfulqa_mc1 | cuda | 13 | 0.25 | 817 | 3.4 |
| phi3-mini | q4_k_m | truthfulqa_mc2 | cuda | 12 | 0.36 | 817 | 4.9 |
| phi3-mini | q4_k_m | ukps_qa | cuda | 25 | 0.48 | 60 | 0.5 |
| phi3-mini | q4_k_m | ukps_sum | cuda | 25 | 3.58 | 25 | 1.5 |
| phi3-mini | q8_0 | cnndm | cuda | 25 | 4.14 | 150 | 10.4 |
| phi3-mini | q8_0 | cuad | cuda | 25 | 3.58 | 200 | 11.9 |
| phi3-mini | q8_0 | hotpotqa | cuda | 25 | 1.47 | 300 | 7.3 |
| phi3-mini | q8_0 | truthfulqa_mc1 | cuda | 13 | 0.26 | 817 | 3.5 |
| phi3-mini | q8_0 | truthfulqa_mc2 | cuda | 12 | 0.38 | 817 | 5.1 |
| phi3-mini | q8_0 | ukps_qa | cuda | 25 | 0.51 | 60 | 0.5 |
| phi3-mini | q8_0 | ukps_sum | cuda | 25 | 4.06 | 25 | 1.7 |

Projected total on cuda: 2.1 hours

## 2. Confidence-interval projection

Planning approximation: CI width scales with 1/sqrt(n) (standard for means).

| task | metric | precision | n_pilot | width_pilot | n_planned | width_projected | verdict |
|---|---|---|---|---|---|---|---|
| cnndm | rouge1 | fp16 | 25 | 0.077 | 150 | 0.031 | OK |
| cnndm | rouge2 | fp16 | 25 | 0.054 | 150 | 0.022 | OK |
| cnndm | rougeL | fp16 | 25 | 0.060 | 150 | 0.024 | OK |
| cuad | em | fp16 | 25 | 0.000 | 200 | 0.000 | OK |
| cuad | f1 | fp16 | 25 | 0.242 | 200 | 0.085 | OK |
| hotpotqa | em | fp16 | 25 | 0.360 | 300 | 0.104 | FLAG |
| hotpotqa | f1 | fp16 | 25 | 0.324 | 300 | 0.094 | OK |
| truthfulqa_mc1 | mc1 | fp16 | 13 | 0.462 | 817 | 0.058 | OK |
| truthfulqa_mc2 | mc2 | fp16 | 12 | 0.492 | 817 | 0.060 | OK |
| ukps_qa | em | fp16 | 25 | 0.000 | 60 | 0.000 | OK |
| ukps_qa | f1 | fp16 | 25 | 0.175 | 60 | 0.113 | FLAG |
| ukps_sum | rouge1 | fp16 | 25 | 0.066 | 25 | 0.066 | OK |
| ukps_sum | rouge2 | fp16 | 25 | 0.072 | 25 | 0.072 | OK |
| ukps_sum | rougeL | fp16 | 25 | 0.071 | 25 | 0.071 | OK |

## 3. Template rendering check

### phi3-mini: extractive_qa
```
<|user|>
Read the contract excerpt and answer the question by quoting the exact text from the excerpt. If the answer is a span, reply with the span only, no explanation.

Contract excerpt:
EXAMPLE CONTRACT EXCERPT.

Question: EXAMPLE QUESTION?

Answer:<|end|>
<|assistant|>

```
### phi3-mini: mc_question
```
<|user|>
Q: EXAMPLE QUESTION?
A:<|end|>
<|assistant|>

```
### gemma2-2b: extractive_qa
```
<start_of_turn>user
Read the contract excerpt and answer the question by quoting the exact text from the excerpt. If the answer is a span, reply with the span only, no explanation.

Contract excerpt:
EXAMPLE CONTRACT EXCERPT.

Question: EXAMPLE QUESTION?

Answer:<end_of_turn>
<start_of_turn>model

```
### gemma2-2b: mc_question
```
<start_of_turn>user
Q: EXAMPLE QUESTION?
A:<end_of_turn>
<start_of_turn>model

```
### llama32-3b: extractive_qa
```
<|start_header_id|>user<|end_header_id|>

Read the contract excerpt and answer the question by quoting the exact text from the excerpt. If the answer is a span, reply with the span only, no explanation.

Contract excerpt:
EXAMPLE CONTRACT EXCERPT.

Question: EXAMPLE QUESTION?

Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```
### llama32-3b: mc_question
```
<|start_header_id|>user<|end_header_id|>

Q: EXAMPLE QUESTION?
A:<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```
### mistral7b: extractive_qa
```
[INST] Read the contract excerpt and answer the question by quoting the exact text from the excerpt. If the answer is a span, reply with the span only, no explanation.

Contract excerpt:
EXAMPLE CONTRACT EXCERPT.

Question: EXAMPLE QUESTION?

Answer: [/INST]
```
### mistral7b: mc_question
```
[INST] Q: EXAMPLE QUESTION?
A: [/INST]
```

## Freeze checklist

- [ ] sample sizes confirmed or adjusted (say which, and why)
- [ ] decoding settings confirmed (greedy, max tokens per task)
- [ ] battery.jsonl committed and frozen
- [ ] data/ukps/items.jsonl committed and frozen
- [ ] any deviation from Chapter 3 planned values, listed for the mechanical update
