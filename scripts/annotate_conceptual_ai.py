"""Use K2-Horizon as an AI judge to evaluate chunk relevance for conceptual queries.

This script replaces human annotation. It takes a JSON file of pooled chunks
for a set of conceptual queries, and asks the LLM to score each chunk as:
  y: highly relevant (directly answers the query)
  p: partially relevant (related context but incomplete)
  n: not relevant

Usage:
    uv run python scripts/annotate_conceptual_ai.py --pool data/pool.json --out data/conceptual_labels.json
"""

import argparse
import json
import logging
from pathlib import Path

from rag.generate import get_llm
from rag.evaluate import LABELS_PATH

PROMPT = """You are an expert Indian legal judge.
Evaluate whether the following excerpt from a court judgment answers the user's conceptual query.

Query: "{query}"

Excerpt:
{chunk_text}

Rate the relevance of this excerpt to the query using EXACTLY ONE of the following letters:
y - Highly relevant (The excerpt directly addresses the query and provides the legal principle, test, or rule).
p - Partially relevant (The excerpt provides related context or discusses the topic, but does not provide the definitive answer or rule).
n - Not relevant (The excerpt does not address the query).

Output ONLY the single letter (y, p, or n) and nothing else. Do not explain your reasoning.
"""

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("data/pool.json"),
                        help="JSON file containing queries and their pooled chunks")
    parser.add_argument("--out", type=Path, default=Path("data/conceptual_labels.json"),
                        help="Output path for the generated labels")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of queries to annotate for quick tests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("annotate")

    llm = get_llm()
    log.info(f"Using LLM: {llm.model}")

    if not args.pool.exists():
        log.error(f"Pool file not found: {args.pool}")
        return

    with open(args.pool) as f:
        pool_data = json.load(f)

    if args.limit:
        pool_data = pool_data[:args.limit]

    # Output format matches evaluate.py's LabelledQuery format
    results = []

    for item in pool_data:
        query_text = item["query"]
        log.info(f"\nScoring query: {query_text}")
        
        judgments = {}
        chunks = item.get("chunks", [])
        
        for chunk in chunks:
            chunk_id = chunk["id"]
            chunk_text = chunk["text"]
            
            prompt = PROMPT.format(query=query_text, chunk_text=chunk_text)
            
            try:
                # Ask the LLM to score the chunk
                reply = llm.complete(system="", user=prompt)
                score = reply.strip().lower()
                
                # Sanitize the output
                if score not in ("y", "p", "n"):
                    if "y" in score: score = "y"
                    elif "p" in score: score = "p"
                    else: score = "n"
                
                judgments[chunk_id] = score
                log.info(f"  {chunk_id}: {score}")
                
            except Exception as e:
                log.error(f"  Failed to score {chunk_id}: {e}")
                judgments[chunk_id] = "n" # Default to not relevant on failure
        
        results.append({
            "query": query_text,
            "kind": "conceptual",
            "matches": judgments
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nSaved {len(results)} queries to {args.out}")

if __name__ == "__main__":
    main()
