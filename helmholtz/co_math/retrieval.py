from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}|[0-9]+")
STOP_WORDS = {
    "and",
    "are",
    "can",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "let",
    "not",
    "that",
    "the",
    "then",
    "this",
    "with",
    "what",
    "when",
    "where",
    "which",
}

SUPPORTED_SUFFIXES = {
    ".bib",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def tokens(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text) if match.group(0).lower() not in STOP_WORDS]


@dataclass(frozen=True)
class DocumentChunk:
    path: Path
    start_line: int
    end_line: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}"

    @property
    def compact_text(self) -> str:
        return " ".join(self.text.split())


class LocalContextIndex:
    def __init__(self, chunks: Sequence[DocumentChunk]):
        self.chunks = list(chunks)
        self._chunk_tokens = [tokens(chunk.text) for chunk in self.chunks]
        self._idf = self._compute_idf(self._chunk_tokens)

    @classmethod
    def from_paths(cls, paths: Iterable[Path], chunk_lines: int = 32, overlap: int = 6) -> "LocalContextIndex":
        chunks: List[DocumentChunk] = []
        for path in paths:
            if not path.exists() or path.is_dir():
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            step = max(1, chunk_lines - overlap)
            for start in range(0, max(1, len(lines)), step):
                window = lines[start : start + chunk_lines]
                if not any(line.strip() for line in window):
                    continue
                chunks.append(
                    DocumentChunk(
                        path=path.resolve(),
                        start_line=start + 1,
                        end_line=min(len(lines), start + len(window)),
                        text="\n".join(window),
                    )
                )
        return cls(chunks)

    def search(self, query: str, k: int = 6) -> List[Tuple[DocumentChunk, float]]:
        query_tokens = tokens(query)
        if not query_tokens:
            return []
        query_counts = self._counts(query_tokens)
        scored: List[Tuple[DocumentChunk, float]] = []
        for chunk, chunk_tokens in zip(self.chunks, self._chunk_tokens):
            chunk_counts = self._counts(chunk_tokens)
            score = 0.0
            for term, query_count in query_counts.items():
                if term in chunk_counts:
                    score += query_count * chunk_counts[term] * self._idf.get(term, 1.0)
            if score > 0:
                title_boost = sum(1 for term in query_counts if term in chunk.path.name.lower())
                scored.append((chunk, score + 0.25 * title_boost))
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[:k]

    def overview(self) -> List[str]:
        seen = []
        for chunk in self.chunks:
            if chunk.path not in seen:
                seen.append(chunk.path)
        return [str(path) for path in seen]

    @staticmethod
    def _counts(values: Iterable[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return counts

    @staticmethod
    def _compute_idf(rows: Sequence[Sequence[str]]) -> Dict[str, float]:
        doc_freq: Dict[str, int] = {}
        for row in rows:
            for term in set(row):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        total = max(1, len(rows))
        return {term: math.log((1 + total) / (1 + freq)) + 1.0 for term, freq in doc_freq.items()}
