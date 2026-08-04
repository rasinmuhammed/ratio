"""Chunking: Document -> Chunk Stream, built for Indian court judgements dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

from rag.ingest import Document

TARGET_SIZE = 1200
OVERLAP = 150
MIN_CHUNK_SIZE = 100
MIN_NUMBERED_PARAS = 5

_NUMBERED_PARA = re.compile(r'^[ \t]*(\d{1,4})\.[ \t]+(?=[A-Z"(\'])', re.M)
_BLANK_LINE = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.?!])\s+")
_CASE_NO = re.compile(r"\b(?:No\.?\s?\d+\s+of\s+\d{4})", re.I)

@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    index: int
    metadata: dict[str, Any]

def _case_number(text: str) -> str:
    m = _CASE_NO.search(text[:3000])
    return m.group(0).strip() if m else ""

def context_prefix(doc: Document) -> str:
    """Identifiers that appear only in the header, attached to every chunk."""
    parts = [doc.metadata.get("court") or "", doc.metadata.get("title") or ""]
    case_no = _case_number(doc.text)
    if case_no:
        parts.append(case_no)
    return " | ".join( p for p in parts if p)

def _split_numbered(text: str) -> list[str] | None:
    """Split on numbered paragraphs."""
    matches = list(_NUMBERED_PARA.finditer(text))
    if len(matches) < MIN_NUMBERED_PARAS:
        return None
    numbers = [int(m.group(1)) for m in matches]
    if numbers != sorted(numbers):
        return None

    starts = [m.start() for m in matches]
    pieces: list[str] = []
    if starts[0] > 0:
        pieces.append(text[:starts[0]])
    for a, b in zip(starts, starts[1:] + [len(text)]):
        pieces.append(text[a:b])
    return [p for p in pieces if p.strip()]

def _split_units(text: str, size: int, length: Callable[[str], int]) -> list[str]:
    """Break text into units no larger than 'size', strongest boundary first."""
    if length(text) <= size:
        return [text]

    numbered = _split_numbered(text)
    if numbered and len(numbered) > 1:
        out: list[str] = []
        for piece in numbered:
            out.extend(_split_units(piece, size, length))
        return out

    for pattern in (_BLANK_LINE, _SENTENCE):
        parts = [p for p in pattern.split(text) if p.strip()]
        if len(parts) > 1:
            out = []
            for part in parts:
                out.extend(_split_units(part, size, length))
            return out

    return [text[i : i + size] for i in range(0, len(text), size)]

JOIN = "\n\n"


def _overlap_tail(body: str, overlap: int) -> str:
    """Last `overlap` chars, snapped forward to a word boundary.

    Slicing blindly lands mid-word ('ct empowers...'). Cutting at the first
    whitespace inside the window costs a few characters and keeps the tail
    readable. Falls back to the raw slice if the window has no whitespace.
    """
    if overlap <= 0:
        return ""
    tail = body[-overlap:]
    space = tail.find(" ")
    if space == -1:
        return tail
    snapped = tail[space + 1 :]
    return snapped if snapped.strip() else tail


def _pack(units: Iterable[str], size: int, overlap: int,
          length: Callable[[str], int]) -> Iterator[str]:
    """Greedily combine units into chunks of at most `size`.

    `current` counts the JOIN separators as well as the units, because the
    emitted chunk is JOIN.join(buffer). Ignoring them let chunks drift over
    the target by 2 chars per unit.
    """
    buffer: list[str] = []
    current = 0
    join_len = length(JOIN)

    for unit in units:
        unit_len = length(unit)
        added = unit_len + (join_len if buffer else 0)

        if buffer and current + added > size:
            body = JOIN.join(buffer)
            yield body
            tail = _overlap_tail(body, overlap)
            # The carried tail plus the unit that just overflowed must still
            # fit, otherwise the very next chunk starts already over budget.
            if tail and length(tail) + join_len + unit_len > size:
                tail = ""
            if tail:
                buffer = [tail, unit]
                current = length(tail) + join_len + unit_len
            else:
                buffer = [unit]
                current = unit_len
        else:
            buffer.append(unit)
            current += added

    if buffer:
        body = JOIN.join(buffer)
        if body.strip():
            yield body

def chunk_document(
        doc: Document,
        size: int = TARGET_SIZE,
        overlap: int = OVERLAP,
        min_size: int = MIN_CHUNK_SIZE,
        length: Callable[[str], int] = len,
        add_context: bool = True,
) -> Iterator[Chunk]:
    if overlap >= size:
        raise ValueError(f"Overlap ({overlap}) must be less than size ({size})")

    prefix = context_prefix(doc) if add_context else ""
    budget = size - (length(prefix) + 2 if prefix else 0)
    if budget < min_size:
        raise ValueError(f"Context prefix too long for size={size}")

    units = _split_units(doc.text, budget, length)

    index = 0
    for body in _pack(units, budget, overlap, length):
        body = body.strip()
        if length(body) < min_size:
            continue
        text = f"{prefix}\n\n{body}" if prefix else body
        yield Chunk(
            id=f"{doc.id}#{index}",
            doc_id=doc.id,
            text=text,
            index=index,
            metadata=doc.metadata,
        )
        index += 1

def chunk_documents(docs: Iterable[Document], **kw: Any) -> Iterator[Chunk]:
    for doc in docs:
        yield from chunk_document(doc, **kw)