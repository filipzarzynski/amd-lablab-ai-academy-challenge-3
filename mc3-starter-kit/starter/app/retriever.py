"""Lightweight, pure-Python BM25 Retriever with 2-Hop Entity Linking.

Zero external C/Rust/GPU dependencies. Guarantees safety under ROCm.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re
from typing import NamedTuple

from parsers import parse_file


class Document(NamedTuple):
    rel_path: str
    abs_path: Path
    text: str
    is_image: bool
    tokens: list[str]


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, including compound and subword parts."""
    raw_tokens = re.findall(r'[a-zA-Z0-9]+(?:[-_.#][a-zA-Z0-9]+)*', text.lower())
    tokens = []
    for t in raw_tokens:
        tokens.append(t)
        # Also include constituent parts for hyphenated or underscore words
        parts = re.split(r'[-_.#]', t)
        if len(parts) > 1:
            for p in parts:
                if len(p) > 1 or p.isdigit():
                    tokens.append(p)
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[Document] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}

    def fit(self, docs: list[Document]) -> None:
        self.docs = docs
        self.doc_lens = [len(doc.tokens) for doc in docs]
        total_len = sum(self.doc_lens)
        self.avgdl = (total_len / len(docs)) if docs else 0.0

        self.doc_freqs = Counter()
        self.term_freqs = []
        for doc in docs:
            tf = Counter(doc.tokens)
            self.term_freqs.append(tf)
            for token in tf.keys():
                self.doc_freqs[token] += 1

        n_docs = len(docs)
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: list[str]) -> list[tuple[float, Document]]:
        scores: list[float] = [0.0] * len(self.docs)
        for q_tok in query_tokens:
            if q_tok not in self.idf:
                continue
            idf_val = self.idf[q_tok]
            for i, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(q_tok, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[i]
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[i] += idf_val * (num / den)

        # Image boost: images have short filename-based text, boost their BM25 score if matched
        scored = []
        for s, doc in zip(scores, self.docs):
            if s > 0 and doc.is_image:
                s *= 2.5
            scored.append((s, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored


class CorpusIndex:
    def __init__(self, corpus_root: Path):
        self.corpus_root = corpus_root
        self.docs: list[Document] = []
        self.bm25 = BM25Index()
        self.image_captions: dict[str, str] = {}

    def set_image_caption(self, rel_path: str, caption: str) -> None:
        self.image_captions[rel_path] = caption

    def build(self) -> None:
        self.docs = []
        for path in sorted(self.corpus_root.rglob("*")):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(self.corpus_root).as_posix()
                text, is_image = parse_file(path)
                
                # Combine path metadata and image caption
                combined_text = f"File: {rel}\n"
                if is_image and rel in self.image_captions:
                    combined_text += f"Image Content: {self.image_captions[rel]}\n"
                elif is_image:
                    combined_text += f"Image File: {rel.replace('/', ' ').replace('_', ' ')}\n"
                else:
                    combined_text += text

                if is_image or text.strip():
                    tokens = tokenize(combined_text)
                    self.docs.append(Document(
                        rel_path=rel,
                        abs_path=path,
                        text=text,
                        is_image=is_image,
                        tokens=tokens
                    ))
            except Exception:
                continue

        self.bm25.fit(self.docs)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve relevant documents with 2-hop entity chaining."""
        q_tokens = tokenize(query)
        ranked = self.bm25.score(q_tokens)
        
        candidates: list[Document] = [doc for score, doc in ranked[:top_k] if score > 0]
        if not candidates and self.docs:
            candidates = [doc for _, doc in ranked[:2]]

        # Hop 2: Scan retrieved documents for entity identifiers
        entity_patterns = [
            r'\b[A-Z]{2,}-\d+\b',          # Tickets like ORR-1847
            r'\bE\d{3,}\b',                 # Error codes like E7731
            r'\b[A-Z]{2,}-[A-Z0-9]+-\d+-[A-Z0-9]+\b', # Part numbers
        ]

        found_entities = set()
        for doc in candidates:
            search_text = doc.text or self.image_captions.get(doc.rel_path, "")
            for pat in entity_patterns:
                for match in re.findall(pat, search_text):
                    found_entities.add(match)

        # Remove entities already in query
        for q_tok in q_tokens:
            found_entities.discard(q_tok.upper())

        # If entities found, retrieve linked documents mentioning them
        if found_entities:
            existing_paths = {doc.rel_path for doc in candidates}
            for entity in found_entities:
                e_tokens = tokenize(entity)
                e_ranked = self.bm25.score(e_tokens)
                for score, doc in e_ranked:
                    if score > 0 and doc.rel_path not in existing_paths:
                        candidates.append(doc)
                        existing_paths.add(doc.rel_path)
                        break

        return candidates
