"""scripts/corpus_clean.py: strip non-content blocks from corpus texts.

Protocol rule (data/ukps/PROTOCOL.md): stored text is plain text with
no navigation chrome. NHS condition pages embed stock-photo credit
blocks (e.g. "... / Alamy Stock Photo https://www.alamy.com/...")
which are also outside the NHS OGL grant (images are excepted from the
licence), so they must not enter the corpus. This pass removes:

  - blocks mentioning stock-photo agencies or photo credits,
  - blocks that are mostly URL debris.

Removals are printed per document for the lab log. Idempotent; run
from the repository root after any fetch:
  python scripts/corpus_clean.py
"""

from __future__ import annotations

import glob
import os
import re

DOCS_GLOB = os.path.join("data", "ukps", "documents", "*.txt")

_CREDIT = re.compile(
    r"Alamy|Stock Photo|Getty Images|Science Photo Library|"
    r"Shutterstock|iStock|^Credit:", re.IGNORECASE)
_URL = re.compile(r"https?://\S+")


def is_junk(block: str) -> bool:
    if _CREDIT.search(block):
        return True
    urls = _URL.findall(block)
    if urls:
        url_chars = sum(len(u) for u in urls)
        if url_chars / len(block) > 0.4:   # block is mostly URL debris
            return True
    return False


def main() -> None:
    for path in sorted(glob.glob(DOCS_GLOB)):
        with open(path) as f:
            blocks = f.read().split("\n\n")
        kept, removed = [], []
        for block in blocks:
            (removed if is_junk(block) else kept).append(block)
        if removed:
            with open(path, "w") as f:
                f.write("\n\n".join(kept).rstrip() + "\n")
            doc = os.path.basename(path)
            print(f"[clean] {doc}: removed {len(removed)} block(s):")
            for block in removed:
                head = " ".join(block.split())[:90]
                print(f"    - {head}")


if __name__ == "__main__":
    main()
