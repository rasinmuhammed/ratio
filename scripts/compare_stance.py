"""Does labelling a source's voice change the answer?

The stance labels were added to the prompt on the argument that a model shown
"learned counsel submitted that X" as an undifferentiated source will report X
as the law. That is a claim about model behaviour, and claims about behaviour
have to be run rather than reasoned about.

Same query, same retrieval, same model. The only difference is whether each
source carries a note about whose voice it is in. Retrieval is deterministic,
so anything that differs between the two answers came from the labels.

    uv run python scripts/compare_stance.py
    uv run python scripts/compare_stance.py "grounds for granting an injunction"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from rag.generate import GroqLLM, answer
from rag.hybrid import HybridRetriever
from rag.route import RoutedRetriever
from rag.stance import ARGUMENT, classify

INDEX_DIR = Path("data/index-full")
RULE = "=" * 78

# Queries whose retrieved passages are known to include a party's submission,
# since a comparison on sources that are all in the court's voice would show
# nothing either way.
DEFAULT = [
    "when can a writ petition be dismissed for delay",
    "what must be proved to establish adverse possession",
    "conditions for granting anticipatory bail",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="*", default=None)
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    queries = args.queries or DEFAULT

    print("loading index...", file=sys.stderr)
    started = time.time()
    hybrid = HybridRetriever(args.index)
    retriever = RoutedRetriever(hybrid, hybrid.dense.payloads)
    llm = GroqLLM()
    print(f"ready in {time.time() - started:.0f}s", file=sys.stderr)

    for query in queries:
        plain = answer(query, retriever, llm, k=args.k, stance_notes=False)
        labelled = answer(query, retriever, llm, k=args.k, stance_notes=True)

        stances = [classify(s.text) for s in labelled.sources]
        n_argument = stances.count(ARGUMENT)

        print(f"\n{RULE}\n{query}")
        print(f"{len(labelled.sources)} sources, {n_argument} of them a party's "
              f"submission: {', '.join(stances)}")

        if n_argument == 0:
            print("(no argument passage retrieved, so nothing to distinguish)")

        print(f"\n--- without stance labels\n{plain.text}")
        print(f"\n--- with stance labels\n{labelled.text}")


if __name__ == "__main__":
    main()
