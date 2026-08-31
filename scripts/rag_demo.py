"""scripts/rag_demo.py: demonstrative on-device RAG prototype.

Dissertation deliverable O5 (demonstrative, NOT part of the controlled
experiment; Chapter 3, 3.9). Shows the evaluated stack operating end to
end on the target device: retrieval over the curated UK public-sector
corpus, then grounded generation through a quantised model.

Retrieval is a deliberately simple, dependency-free TF-IDF cosine over
paragraph chunks of data/ukps/documents/*.txt: the prototype
demonstrates the deployment architecture, not retrieval research.

Run from the repository root, venv active, with a model present:
  python scripts/rag_demo.py --gguf models/phi3-mini-q4_k_m.gguf \
      --model-key phi3-mini --question "Who is eligible for ...?"
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import Counter

from slmbench import prompts
from slmbench.adapters.base import Item
from slmbench.backend import LlamaCppBackend

DOCS_GLOB = os.path.join("data", "ukps", "documents", "*.txt")
CHUNK_CHARS = 1200


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_documents() -> list[tuple[str, str]]:
    """[(doc_id, chunk_text)] over paragraph-aligned chunks."""
    chunks = []
    for path in sorted(glob.glob(DOCS_GLOB)):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            text = f.read()
        buffer = ""
        for para in text.split("\n\n"):
            if len(buffer) + len(para) > CHUNK_CHARS and buffer:
                chunks.append((doc_id, buffer.strip()))
                buffer = ""
            buffer += para + "\n\n"
        if buffer.strip():
            chunks.append((doc_id, buffer.strip()))
    return chunks


class TfIdfIndex:
    def __init__(self, chunks: list[tuple[str, str]]) -> None:
        self.chunks = chunks
        self.term_freqs = [Counter(tokenize(text)) for _, text in chunks]
        doc_freq: Counter = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log(n / df) for t, df in doc_freq.items()}

    def _vector(self, tf: Counter) -> dict[str, float]:
        return {t: count * self.idf.get(t, 0.0) for t, count in tf.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[str, str, float]]:
        q_vec = self._vector(Counter(tokenize(query)))
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        scored = []
        for (doc_id, text), tf in zip(self.chunks, self.term_freqs):
            c_vec = self._vector(tf)
            dot = sum(q_vec.get(t, 0.0) * v for t, v in c_vec.items())
            c_norm = math.sqrt(sum(v * v for v in c_vec.values())) or 1.0
            scored.append((doc_id, text, dot / (q_norm * c_norm)))
        return sorted(scored, key=lambda x: -x[2])[:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--model-key", required=True,
                        choices=sorted(prompts.CHAT_FORMATS))
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()

    chunks = chunk_documents()
    if not chunks:
        raise SystemExit(f"no corpus documents found at {DOCS_GLOB}; "
                         "author the corpus first (data/ukps/PROTOCOL.md)")
    index = TfIdfIndex(chunks)
    hits = index.search(args.question, args.top_k)
    print(f"[rag] retrieved {len(hits)} chunk(s):")
    for doc_id, _text, score in hits:
        print(f"  {doc_id}  (cosine {score:.3f})")

    context = "\n\n".join(text for _, text, _ in hits)
    item = Item(task="rag_demo", item_id="rag", context=context,
                question=args.question, references=[])
    prompt = prompts.render(item, "grounded_qa", args.model_key)

    bk = LlamaCppBackend(args.gguf)
    print(f"[rag] model loaded in {bk.load_s:.1f}s; generating...")
    r = bk.generate(prompt, args.max_tokens,
                    stop=prompts.STOP[args.model_key])
    bk.close()
    print("\nAnswer:")
    print(r.text.strip())
    print(f"\n({r.n_gen_tokens} tokens at {r.decode_tps:.1f} tok/s "
          f"decode; documents never left this machine)")


if __name__ == "__main__":
    main()
