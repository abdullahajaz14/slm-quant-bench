"""scripts/corpus_fetch.py: fetch one OGL document for the curated corpus.

Implements the document-selection step of data/ukps/PROTOCOL.md:
fetches a GOV.UK or NHS page, extracts the main content as plain text
(no navigation chrome), verifies the Open Government Licence statement
on the page, saves data/ukps/documents/<doc_id>.txt and appends the
INDEX.csv row (doc_id,title,url,publisher,licence,access_date).

Deliberately dependency-free (urllib + html.parser) so the extraction
is deterministic and auditable: the stored text is the page's own
words, never a model's paraphrase.

Usage (from the repository root):
  python scripts/corpus_fetch.py <doc_id> <url> <publisher> \
      [--licence-url URL]   # e.g. the non-print page for gov.uk guides
"""

from __future__ import annotations

import argparse
import csv
import datetime
import html
import os
import re
import sys
import urllib.request
from html.parser import HTMLParser

DOCS_DIR = os.path.join("data", "ukps", "documents")
INDEX = os.path.join(DOCS_DIR, "INDEX.csv")
UA = {"User-Agent": "Mozilla/5.0 (dissertation corpus builder; "
                    "OGL document collection)"}

_TEXT_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "caption", "th", "td"}
_SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "form",
              "button", "svg"}


class MainTextExtractor(HTMLParser):
    """Collects text of paragraph-level elements inside <main>...</main>
    (falls back to <body> if the page has no main landmark)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_main = 0
        self.skip = 0
        self.blocks: list[str] = []
        self._current: list[str] | None = None
        self._current_tag = ""
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag == "main":
            self.in_main += 1
        if tag in _SKIP_TAGS:
            self.skip += 1
        if (self.in_main and not self.skip and tag in _TEXT_TAGS
                and self._current is None):
            self._current = []
            self._current_tag = tag

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag == "main":
            self.in_main = max(0, self.in_main - 1)
        if tag in _SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        if self._current is not None and tag == self._current_tag:
            text = " ".join("".join(self._current).split())
            if text:
                if tag.startswith("h"):
                    self.blocks.append(f"\n{text}\n")
                elif tag == "li":
                    self.blocks.append(f"- {text}")
                else:
                    self.blocks.append(text)
            self._current = None

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._current is not None:
            self._current.append(data)


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def extract(html_text: str) -> tuple[str, str]:
    parser = MainTextExtractor()
    parser.feed(html_text)
    if not parser.blocks:  # no <main>: retry treating <body> as main
        parser = MainTextExtractor()
        parser.in_main = 1
        parser.feed(html_text)
    text = "\n\n".join(parser.blocks)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    title = html.unescape(parser.title)
    title = re.sub(r"\s*[-|]\s*(GOV\.UK|NHS).*$", "", title).strip()
    return title, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_id")
    parser.add_argument("url")
    parser.add_argument("publisher")
    parser.add_argument("--licence-url", default=None,
                        help="page carrying the OGL statement, if not "
                             "the content URL (gov.uk /print pages)")
    args = parser.parse_args()

    os.makedirs(DOCS_DIR, exist_ok=True)
    page = fetch(args.url)
    title, text = extract(page)
    if len(text) < 500:
        print(f"[corpus] REJECT {args.doc_id}: extracted only "
              f"{len(text)} chars from {args.url}", file=sys.stderr)
        sys.exit(2)

    licence_page = page if args.licence_url is None \
        else fetch(args.licence_url)
    if "Open Government Licence" not in licence_page:
        print(f"[corpus] REJECT {args.doc_id}: no Open Government "
              f"Licence statement found for {args.url}", file=sys.stderr)
        sys.exit(3)

    out_path = os.path.join(DOCS_DIR, f"{args.doc_id}.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")

    exists = os.path.exists(INDEX)
    with open(INDEX, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["doc_id", "title", "url", "publisher",
                             "licence", "access_date"])
        writer.writerow([args.doc_id, title, args.url, args.publisher,
                         "Open Government Licence v3.0",
                         datetime.date.today().isoformat()])
    print(f"[corpus] {args.doc_id}: '{title}' -> {len(text)} chars, "
          f"OGL verified")


if __name__ == "__main__":
    main()
