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


---

## 1. Runtime projection

| model | precision | task | backend | n_pilot | median_s | planned_n | projected_min |
|---|---|---|---|---|---|---|---|
| gemma2-2b | q4_k_m | cnndm | metal | 25 | 10.04 | 150 | 25.1 |
| gemma2-2b | q4_k_m | cuad | metal | 25 | 5.85 | 200 | 19.5 |
| gemma2-2b | q4_k_m | hotpotqa | metal | 25 | 6.26 | 300 | 31.3 |
| gemma2-2b | q4_k_m | truthfulqa_mc1 | metal | 13 | 1.76 | 817 | 24.0 |
| gemma2-2b | q4_k_m | truthfulqa_mc2 | metal | 12 | 2.39 | 817 | 32.5 |
| gemma2-2b | q4_k_m | ukps_qa | metal | 25 | 0.84 | 60 | 0.8 |
| gemma2-2b | q4_k_m | ukps_sum | metal | 25 | 8.31 | 25 | 3.5 |
| gemma2-2b | q8_0 | cnndm | metal | 25 | 12.36 | 150 | 30.9 |
| gemma2-2b | q8_0 | cuad | metal | 25 | 9.56 | 200 | 31.9 |
| gemma2-2b | q8_0 | hotpotqa | metal | 25 | 6.06 | 300 | 30.3 |
| gemma2-2b | q8_0 | truthfulqa_mc1 | metal | 13 | 1.73 | 817 | 23.5 |
| gemma2-2b | q8_0 | truthfulqa_mc2 | metal | 12 | 2.21 | 817 | 30.2 |
| gemma2-2b | q8_0 | ukps_qa | metal | 25 | 1.01 | 60 | 1.0 |
| gemma2-2b | q8_0 | ukps_sum | metal | 25 | 9.90 | 25 | 4.1 |
| phi3-mini | q4_k_m | cnndm | metal | 25 | 17.17 | 150 | 42.9 |
| phi3-mini | q4_k_m | cuad | metal | 25 | 17.07 | 200 | 56.9 |
| phi3-mini | q4_k_m | hotpotqa | metal | 25 | 12.53 | 300 | 62.6 |
| phi3-mini | q4_k_m | truthfulqa_mc1 | metal | 13 | 2.13 | 817 | 29.0 |
| phi3-mini | q4_k_m | truthfulqa_mc2 | metal | 12 | 3.18 | 817 | 43.3 |
| phi3-mini | q4_k_m | ukps_qa | metal | 25 | 2.46 | 60 | 2.5 |
| phi3-mini | q4_k_m | ukps_sum | metal | 25 | 16.46 | 25 | 6.9 |

Projected total on metal: 8.9 hours

## 2. Confidence-interval projection

Planning approximation: CI width scales with 1/sqrt(n) (standard for means).

| task | metric | precision | n_pilot | width_pilot | n_planned | width_projected | verdict |
|---|---|---|---|---|---|---|---|
| cnndm | rouge1 | q8_0 | 25 | 0.053 | 150 | 0.022 | OK |
| cnndm | rouge2 | q8_0 | 25 | 0.047 | 150 | 0.019 | OK |
| cnndm | rougeL | q8_0 | 25 | 0.052 | 150 | 0.021 | OK |
| cuad | em | q8_0 | 25 | 0.200 | 200 | 0.071 | OK |
| cuad | f1 | q8_0 | 25 | 0.269 | 200 | 0.095 | OK |
| hotpotqa | em | q8_0 | 25 | 0.360 | 300 | 0.104 | FLAG |
| hotpotqa | f1 | q8_0 | 25 | 0.270 | 300 | 0.078 | OK |
| truthfulqa_mc1 | mc1 | q8_0 | 13 | 0.538 | 817 | 0.068 | OK |
| truthfulqa_mc2 | mc2 | q8_0 | 12 | 0.489 | 817 | 0.059 | OK |
| ukps_qa | em | q8_0 | 25 | 0.320 | 60 | 0.207 | FLAG |
| ukps_qa | f1 | q8_0 | 25 | 0.257 | 60 | 0.166 | FLAG |
| ukps_sum | rouge1 | q8_0 | 25 | 0.066 | 25 | 0.066 | OK |
| ukps_sum | rouge2 | q8_0 | 25 | 0.061 | 25 | 0.061 | OK |
| ukps_sum | rougeL | q8_0 | 25 | 0.074 | 25 | 0.074 | OK |

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
