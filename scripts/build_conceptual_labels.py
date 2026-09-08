"""Build conceptual labels the honest way: reverse-generate the query.

build_labels.py gets exact-match labels for free, if six chunks contain the
literal string "AIR 1974", the six chunks are the answer, no judgment
required. Conceptual relevance has no equivalent free lunch: "does this
passage answer this question" needs a legal reader, which is exactly why
this gap has stood open since the project started.

The way out is to generate the query from the answer instead of judging an
answer against a query someone else wrote. Start from a chunk already known
to state a legal conclusion (rag.stance classifies it HOLDING or BOTH, the
same classifier already used to reorder generation sources), ask Mercury 2
to write one conceptual question that chunk answers, in the way a person
would actually ask it, without quoting the chunk's own case-specific
details. The chunk is then relevant to that question *by construction*: no
judgment call is needed, because the question was built to fit it.

The honest limitation, worth keeping in mind whenever these labels get
used: the label says "this chunk answers this question", not "this is the
only chunk in 414,122 that does". Some other judgment likely states the same
principle. A retriever that surfaces a different, equally correct holding
scores as a miss here. Recall on these labels is a lower bound on real
recall, not a ceiling, treat it as evidence of a floor, not a final number.

    uv run python scripts/build_conceptual_labels.py --count 25
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

from rag.evaluate import LabelledQuery, load_labels, save_labels
from rag.generate import MercuryLLM
from rag.stance import BOTH, HOLDING, classify

INDEX_DIR = Path("data/index")
LABELS_PATH = Path("data/labels.json")
SEED = 20260908  # today's date: reproducible, not cherry-picked after seeing results

SYSTEM_PROMPT = """You write one conceptual legal question that a given \
passage from an Indian court judgment answers.

Rules:
- Ask the way a person actually asks a legal question in general terms, \
for example "what must be shown to establish adverse possession", not a \
lookup for this specific case.
- Do not name the parties, the court, the case number, or quote a distinctive \
phrase from the passage. The question must stand on its own with no way to \
tell which judgment it came from.
- One question. No preamble, no quotation marks, no "Question:" prefix, \
just the question itself.
"""

# A crude leak check, not a semantic one: if the model's question shares a
# long run of words with the source passage, it copied rather than
# generalised, and the label would secretly be an exact-match query wearing
# a conceptual label. Reject and retry rather than silently keep it.
LEAK_RUN = 6


def _shares_a_long_run(question: str, passage: str, run: int = LEAK_RUN) -> bool:
    q_words = question.lower().split()
    p_words = passage.lower().split()
    p_ngrams = {tuple(p_words[i:i + run]) for i in range(len(p_words) - run + 1)}
    return any(
        tuple(q_words[i:i + run]) in p_ngrams
        for i in range(len(q_words) - run + 1)
    )


def generate_question(llm: MercuryLLM, passage: str, attempts: int = 3) -> str | None:
    for _ in range(attempts):
        question = llm.complete(SYSTEM_PROMPT, passage).strip().strip('"')
        if question and not _shares_a_long_run(question, passage):
            return question
    return None  # every attempt leaked or came back empty; skip this chunk rather than keep a bad label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--count", type=int, default=25,
                        help="how many new conceptual labels to add")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    payloads = [json.loads(line) for line in (args.index / "payloads.jsonl").open()]
    candidates = [p for p in payloads if classify(p["text"]) in (HOLDING, BOTH)]
    print(f"{len(candidates)} holding-quality chunks of {len(payloads)} total",
          file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    existing = load_labels(args.labels) if args.labels.exists() else []
    existing_queries = {l.query for l in existing}
    # By chunk, not just by exact question text. A fixed seed reshuffles
    # candidates identically across separate runs, so without this, a chunk
    # already labelled gets sampled again, and at temperature 0.5 Mercury
    # rarely repeats a question verbatim, it rewords it just enough to slip
    # past a text-only dedup ("public body" vs "public authority", same
    # chunk, two labels). Found this exact case in the first real batch.
    already_labelled = {cid for l in existing for cid in l.relevant_chunk_ids}
    new_labels: list[LabelledQuery] = []

    llm = MercuryLLM()
    started = time.time()

    for chunk in candidates:
        if len(new_labels) >= args.count:
            break
        if chunk["id"] in already_labelled:
            continue
        question = generate_question(llm, chunk["text"])
        if question is None:
            print(f"  skip (leaked or empty): {chunk['id']}", file=sys.stderr)
            continue
        if question in existing_queries:
            continue  # a repeat, from an earlier run at the same seed
        new_labels.append(LabelledQuery(
            query=question,
            relevant_chunk_ids=[chunk["id"]],
            kind="conceptual",
            note="reverse-generated from a HOLDING/BOTH chunk via Mercury 2, "
                 "see scripts/build_conceptual_labels.py",
        ))
        print(f"  [{len(new_labels)}/{args.count}] {question}", file=sys.stderr)

    save_labels(existing + new_labels, args.labels)
    print(f"\n{len(new_labels)} new conceptual labels added "
          f"({time.time() - started:.0f}s) -> {args.labels}",
          file=sys.stderr)
    print(f"total labels in file: {len(existing) + len(new_labels)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
