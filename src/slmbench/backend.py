"""slmbench.backend: llama.cpp model backend.

One LlamaCppBackend instance wraps one GGUF file at one precision.
The runner constructs and closes these sequentially (one resident model
at a time; the target machine has 8 GB).

Two operating modes, chosen at construction:
  - generation mode (default): used for QA, summarisation, UKPS tasks.
  - choice mode (logits_all=True): used only for TruthfulQA MC scoring,
    because keeping all logits costs memory and time.

Timing method (fixed by protocol): stream the completion.
  t0 = clock before the call
  t_first = clock at first streamed chunk   -> prefill time = t_first - t0
  t_end = clock at final chunk              -> decode time  = t_end - t_first
  prefill_tps = n_prompt_tokens / (t_first - t0)
  decode_tps  = (n_gen_tokens - 1) / (t_end - t_first)   # first token
                                                         # belongs to prefill
Rationale: time-to-first-token isolates the prompt-processing phase
without a second perturbing call.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass


@dataclass
class GenResult:
    text: str
    n_prompt_tokens: int
    n_gen_tokens: int
    prefill_tps: float
    decode_tps: float
    total_s: float


class LlamaCppBackend:
    def __init__(
        self,
        gguf_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        seed: int = 42,
        choice_mode: bool = False,
    ) -> None:
        """Load the model and record load time in self.load_s (seconds)."""
        # imported here so the module stays importable without the
        # inference stack (tests of scorers/adapters need no llama_cpp)
        from llama_cpp import Llama

        self.gguf_path = gguf_path
        self.choice_mode = choice_mode
        t0 = time.perf_counter()
        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
            logits_all=choice_mode,
        )
        self.load_s: float = time.perf_counter() - t0

    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_tokens: int,
                 stop: list[str] | None = None) -> GenResult:
        """Greedy, deterministic completion with phase timings.

        Contract (fixed): temperature=0.0; stream=True with t0/t_first/
        t_end per the module docstring; prompt tokens counted with
        tokenize() on the prompt bytes; generated tokens counted as
        streamed chunks that carry text; n_gen_tokens <= 1 ->
        decode_tps = 0.0.
        """
        prompt_ids = self._llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        n_prompt = len(prompt_ids)
        t0 = time.perf_counter()
        t_first: float | None = None
        t_end = t0
        pieces: list[str] = []
        n_gen = 0
        for chunk in self._llm(prompt=prompt, max_tokens=max_tokens,
                               temperature=0.0, stream=True,
                               stop=stop or []):
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_end = now
            text = chunk["choices"][0].get("text", "")
            if text:
                pieces.append(text)
                n_gen += 1
        if t_first is None:          # nothing streamed at all
            t_first = t_end = time.perf_counter()
        prefill_s = t_first - t0
        decode_s = t_end - t_first
        prefill_tps = n_prompt / prefill_s if prefill_s > 0 else 0.0
        decode_tps = ((n_gen - 1) / decode_s
                      if n_gen > 1 and decode_s > 0 else 0.0)
        return GenResult(
            text="".join(pieces),
            n_prompt_tokens=n_prompt,
            n_gen_tokens=n_gen,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            total_s=t_end - t0,
        )

    # ------------------------------------------------------------------
    def choice_loglik(self, prompt: str, option: str) -> float:
        """Sum of log-probabilities of `option` tokens given `prompt`.

        Contract (fixed, TruthfulQA convention): RAW sum over option
        tokens, no length normalisation. Tokenisation is consistent:
        tokenize(prompt) and tokenize(prompt + option) with the same
        BOS setting; the option ids are the suffix of the combined
        sequence beyond len(prompt_ids). The option is never
        re-tokenised alone (leading-space tokens differ).

        Route chosen (defended in the write-up): llm.reset() + llm.eval()
        over the full sequence with logits_all=True, then read
        llm.scores; row i-1 predicts token i. Log-softmax is computed
        stably by subtracting the row max.
        """
        if not self.choice_mode:
            raise RuntimeError("choice_loglik requires choice_mode=True")
        import numpy as np  # ships with llama-cpp-python

        full_ids = self._llm.tokenize(
            (prompt + option).encode("utf-8"), add_bos=True)
        prompt_ids = self._llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        k = len(prompt_ids)
        self._llm.reset()
        self._llm.eval(full_ids)
        scores = self._llm.scores  # (n_ctx, n_vocab); rows filled to n_tokens
        logprob = 0.0
        for i in range(k, len(full_ids)):
            row = np.asarray(scores[i - 1], dtype=np.float64)
            m = row.max()
            log_z = m + math.log(np.exp(row - m).sum())
            logprob += float(row[full_ids[i]]) - log_z
        return logprob

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the model so the next precision can load on 8 GB."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        gc.collect()


if __name__ == "__main__":
    # Day 1 smoke test: point at a local GGUF and eyeball all six fields.
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "models/phi3-mini-q4_k_m.gguf"
    b = LlamaCppBackend(path)
    print(f"load_s: {b.load_s:.2f}")
    r = b.generate("Reply with exactly: hello", max_tokens=8)
    print(r)
    b.close()
