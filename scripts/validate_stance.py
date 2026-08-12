"""How wrong is the stance classifier?

`stance.classify` is a handful of regexes and it leaves about 70% of the corpus
as `neither`. Some of that is honest, since recitals of fact and statutory
quotation carry no stance. Some is the patterns missing phrasings. Without
knowing the split, the label is a component the system depends on and nobody
has checked.

The sample is stratified rather than random, and the `neither` stratum is the
point: a passage the regex declined to label but which is plainly a submission
is a false negative, and those are the ones that reach a model unmarked.

What this establishes and what it does not. Agreement with a language model
bounds the classifier, it does not prove it correct, and both could be wrong in
the same direction. It is a cheap first pass and the disagreements are worth
reading by hand afterwards, which is what `--show` prints.

    export GROQ_API_KEY=...
    uv run python scripts/validate_stance.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from rag.generate import GroqLLM
from rag.stance import ARGUMENT, BOTH, HOLDING, NEITHER, classify

INDEX_DIR = Path("data/index-full")
LABELS = (ARGUMENT, HOLDING, BOTH, NEITHER)

# The chunk is truncated because a 450 token passage costs more than the
# judgment needs. Reporting formulas cluster near the start of a sentence and
# a model that cannot decide from 1,200 characters will not do better with
# 2,000, it will only cost more.
EXCERPT = 1200

SYSTEM = """You classify passages from Indian court judgments by whose voice \
they are in.

Answer with exactly one word:

argument  - the passage only reports what a party or their counsel submitted, \
contended or argued, without the court's own conclusion
holding   - the passage states the court's own reasoning, finding or \
conclusion
both      - the passage contains a party's submission and the court's response \
to it
neither   - narrative, facts, procedural history or statutory text, with no \
position taken by anyone

One word. No punctuation, no explanation."""


def sample(index_dir: Path, per_label: int, seed: int) -> list[tuple[str, str]]:
    """Reservoir sample per regex label, so a rare class is still represented.

    `both` is 2.1% of the corpus, so a uniform sample of 100 would contain two
    of them and say nothing about that class.
    """
    rng = random.Random(seed)
    buckets: defaultdict[str, list[str]] = defaultdict(list)
    seen: Counter[str] = Counter()

    with (index_dir / "payloads.jsonl").open() as handle:
        for line in handle:
            text = json.loads(line)["text"]
            label = classify(text)
            seen[label] += 1
            bucket = buckets[label]
            if len(bucket) < per_label:
                bucket.append(text)
            else:
                j = rng.randrange(seen[label])
                if j < per_label:
                    bucket[j] = text

    return [(label, text) for label in LABELS for text in buckets[label]]


def ask(llm: GroqLLM, text: str, attempts: int = 4) -> str | None:
    """One label, or None if the model would not produce one."""
    for attempt in range(attempts):
        try:
            reply = llm.complete(SYSTEM, text[:EXCERPT]).strip().lower()
        except RuntimeError:
            # Rate limited. Backing off is the only correct response, and
            # giving up would silently shrink the sample.
            time.sleep(4 * (attempt + 1))
            continue
        word = reply.split()[0].strip(".,:;") if reply.split() else ""
        if word in LABELS:
            return word
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--limit", type=int, default=100,
                        help="total chunks, split evenly across the four labels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=6,
                        help="disagreements to print for reading by hand")
    args = parser.parse_args()

    pairs = sample(args.index, per_label=max(1, args.limit // 4), seed=args.seed)
    print(f"{len(pairs)} chunks sampled, {args.limit // 4} per regex label\n")

    llm = GroqLLM()
    matrix: Counter[tuple[str, str]] = Counter()
    disagreements: list[tuple[str, str, str]] = []
    skipped = 0

    for i, (regex_label, text) in enumerate(pairs, start=1):
        model_label = ask(llm, text)
        if model_label is None:
            skipped += 1
            continue
        matrix[(regex_label, model_label)] += 1
        if model_label != regex_label:
            disagreements.append((regex_label, model_label, text))
        if i % 10 == 0:
            print(f"  {i}/{len(pairs)}")

    judged = sum(matrix.values())
    agreed = sum(n for (a, b), n in matrix.items() if a == b)
    print(f"\nagreement {agreed}/{judged} = {100 * agreed / max(1, judged):.1f}%"
          f"   ({skipped} unparseable)\n")

    header = f"{'regex':<10}" + "".join(f"{label:>10}" for label in LABELS)
    print(header + "   <- model")
    print("-" * len(header))
    for regex_label in LABELS:
        row = sum(matrix[(regex_label, m)] for m in LABELS)
        cells = "".join(f"{matrix[(regex_label, m)]:>10}" for m in LABELS)
        rate = matrix[(regex_label, regex_label)] / row * 100 if row else 0.0
        print(f"{regex_label:<10}{cells}   {rate:>5.0f}% agree")

    missed = sum(matrix[(NEITHER, m)] for m in (ARGUMENT, HOLDING, BOTH))
    total_neither = sum(matrix[(NEITHER, m)] for m in LABELS)
    if total_neither:
        print(f"\n{missed}/{total_neither} chunks the regex declined to label "
              f"({100 * missed / total_neither:.0f}%) carry a stance according "
              f"to the model. These reach the prompt unmarked.")

    for regex_label, model_label, text in disagreements[:args.show]:
        print(f"\n--- regex={regex_label}  model={model_label}")
        print(" ".join(text.split())[:300])


if __name__ == "__main__":
    main()
