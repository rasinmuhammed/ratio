""" 
Ingestion: raw Huggingface Corpus -> Document Stream.
"""

from __future__ import annotations

import logging
import re 

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterator

from datasets import load_from_disk

logger = logging.getLogger(__name__)

RAW_PATH = "data/raw"
SPLIT = "train"

@dataclass(frozen=True, slots=True)
class Document:
    """A single judgement document. Immutable."""

    id: str
    text: str
    metadata: dict[str, Any]

# ------------------------------------------------------------------------------
# Pure Functions.
# ------------------------------------------------------------------------------

def normalize_text(raw: str) -> str:
    """Strip representation artifacts from raw text."""
    # collapse runs of spaces/tabs as they are page-centering artifacts,
    # not information. newlines are preserved because they carry the documents
    # paragraph structure, which chunking will need.
    text = re.sub(r"[ \t]+", " ", raw)

    # strip trailing spaces left at line ends by the collapse above
    text = re.sub(r" *\n", "\n", text)

    # 3+ blank line carries no meaning than 1
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def exclusion_reason(row) -> str | None:
    """Returns a named reason if the row should be excluded."""
    if not row.get("Doc_url"):
        return "missing_url"
    if not row.get("Text") or not row["Text"].strip():
        return "empty_text"
    return None

def _percentile(sorted_values: list[int], p:float) -> int:
    if not sorted_values:
        return 0
    k = min(int(len(sorted_values)*p), len(sorted_values)-1)
    return sorted_values[k]

# ------------------------------------------------------------------------------
# I/0
# ------------------------------------------------------------------------------

def _build_label_index(ds) -> dict[str, list[str]]:
    """Map each document URL to all of its case types, deduplicated and sorted."""
    labels: defaultdict[str, set[str]] = defaultdict(set)
    for url, case_type in zip(ds["Doc_url"], ds["Case_Type"]):
        if case_type:
            labels[url].add(case_type)
    return {url: sorted(types) for url, types in labels.items()}

def load_documents(path: str = RAW_PATH, split: str = SPLIT) -> Iterator[Document]:
    """Yield one Document per distinct judgement, lazily, in source order."""
    ds = load_from_disk(path)[split]
    labels = _build_label_index(ds)
    seen: set[str] = set()

    for row in ds:
        if exclusion_reason(row) is not None:
            continue

        url = row["Doc_url"]
        if url in seen:
            continue
        seen.add(url)

        yield Document(
            id=url,
            text=normalize_text(row["Text"]),
            metadata={
                "title": row["Titles"],
                "court": row["Court_Name_Normalized"],
                "court_type": row["Court_Type"],
                "case_types": labels[url],
                "cites": row["Cites"],
                "cited_by": row["Cited_by"],
                "url": url
                },
        )

def profile(path: str = RAW_PATH, split: str = SPLIT) -> dict[str, Any]:
    """Account for every source row: documents emitted, duplicates collapsed."""
    ds = load_from_disk(path)[split]
    source_rows = len(ds)

    excluded: Counter[str] = Counter()
    missing_metadata: Counter[str] = Counter()
    seen: set[str] = set()
    duplicates = 0
    lengths: list[int] = []

    meta_fields = ["Titles", "Court_Name_Normalized", "Court_Type", "Cites", "Cited_by"]

    for row in ds:
        reason = exclusion_reason(row)
        if reason is not None:
            excluded[reason] += 1
            continue

        url = row["Doc_url"]
        if url in seen:
            duplicates += 1
            continue
        seen.add(url)

        lengths.append(len(normalize_text(row["Text"].strip())))
        for field in meta_fields:
            if row[field] is None or row[field] == "":
                missing_metadata[field] += 1

    lengths.sort()
    emitted = len(lengths)

    accounted = emitted + duplicates + sum(excluded.values())
    if accounted != source_rows:
        raise AssertionError(f"accounted={accounted} != source_rows={source_rows}")

    return {
        "source_rows": source_rows,
        "documents_emitted": emitted,
        "duplicates_collapsed": duplicates,
        "excluded": dict(excluded),
        "missing_metadata": dict(missing_metadata),
        "length_min": lengths[0] if lengths else 0,
        "length_p50": _percentile(lengths, 0.5),
        "length_p90": _percentile(lengths, 0.9),
        "length_p99": _percentile(lengths, 0.99),
        "length_max": lengths[-1] if lengths else 0,
    }

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s % (message)s")
    print(json.dumps(profile(), indent=2))