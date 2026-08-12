"""Hand-annotate the pooled candidates.

Progress is written after every judgment, so this can be stopped and resumed.
Several hundred judgments is not a single sitting.

    uv run python scripts/annotate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.evaluate import LabelledQuery, save_labels

POOL = Path("data/pool.json")
PROGRESS = Path("data/annotations.json")
LABELS = Path("data/labels-conceptual.json")

PREVIEW = 900
RULE = "=" * 78


def write_labels(pool: list[dict], progress: dict, path: Path) -> int:
    queries = []
    for entry in pool:
        judged = progress.get(entry["query"], {})
        relevant = sorted(cid for cid, ok in judged.items() if ok)
        if not relevant:
            # A query where nothing was judged relevant says nothing about the
            # retriever, it says the query was bad. Keeping it would drag every
            # recall score down for a reason unrelated to retrieval.
            continue
        queries.append(LabelledQuery(
            query=entry["query"],
            relevant_chunk_ids=relevant,
            kind="conceptual",
            note=f"pooled from dense, bm25 and hybrid; "
                 f"{len(judged)} of {len(entry['candidates'])} candidates judged",
        ))
    save_labels(queries, path)
    return len(queries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--progress", type=Path, default=PROGRESS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text())
    progress = (
        json.loads(args.progress.read_text()) if args.progress.exists() else {}
    )

    total = sum(len(e["candidates"]) for e in pool)
    done = sum(len(v) for v in progress.values())

    for entry in pool:
        query = entry["query"]
        judged = progress.setdefault(query, {})
        pending = [c for c in entry["candidates"] if c["id"] not in judged]

        for candidate in pending:
            text = " ".join(candidate["text"].split())
            print(f"\n{RULE}")
            print(f"QUERY   {query}")
            print(f"        {done}/{total} judged overall, "
                  f"{len(judged)}/{len(entry['candidates'])} on this query")
            print(f"        {candidate['id']}")
            print("-" * 78)
            print(text[:PREVIEW] + ("..." if len(text) > PREVIEW else ""))
            print("-" * 78)

            answer = ""
            while answer not in {"y", "n", "s", "q"}:
                answer = input("relevant?  [y]es  [n]o  [s]kip  [q]uit > ").strip().lower()

            if answer == "q":
                args.progress.write_text(json.dumps(progress, indent=2))
                kept = write_labels(pool, progress, args.labels)
                print(f"\nstopped at {done}/{total}. "
                      f"{kept} queries have at least one relevant chunk.")
                print(f"progress -> {args.progress}")
                print(f"labels   -> {args.labels}")
                return

            if answer == "s":
                # Skipped, not recorded. It returns on the next run, which is
                # the point: an undecided judgment is not a negative one.
                continue

            judged[candidate["id"]] = answer == "y"
            done += 1
            args.progress.write_text(json.dumps(progress, indent=2))

    kept = write_labels(pool, progress, args.labels)
    print(f"\nall {total} judged. {kept} queries kept -> {args.labels}")


if __name__ == "__main__":
    main()
