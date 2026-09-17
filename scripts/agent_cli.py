"""CLI for interacting with the multi-turn LegalAgent."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from rag.agent import LegalAgent
from rag.audit import init_db, load_session, save_session
from rag.generate import get_llm
from rag.hybrid import HybridRetriever
from rag.rerank import CrossEncoderReranker, RerankedRetriever
from rag.route import RoutedRetriever

INDEX_DIR = Path("data/index-full")
RULE = "=" * 78


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="omit for interactive mode")
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--session", type=str, help="Session ID to persist state across runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print("loading index...", file=sys.stderr)
    started = time.time()
    hybrid = HybridRetriever(args.index)
    reranked = RerankedRetriever(hybrid, CrossEncoderReranker())
    retriever = RoutedRetriever(reranked, hybrid.dense.payloads)

    llm = get_llm()
    print(f"ready in {time.time() - started:.0f}s "
          f"({len(hybrid.dense.payloads):,} chunks, "
          f"{len(retriever.exact):,} identifiers)", file=sys.stderr)

    if args.session:
        init_db()

    # Create a single agent instance if we are holding state
    agent = LegalAgent(llm, retriever, k=args.k)
    
    if args.session:
        past_messages = load_session(args.session)
        if past_messages:
            agent.messages = past_messages
            print(f"[Loaded session {args.session!r} with {len(past_messages)} messages]")

    if args.query:
        print(f"\n{RULE}\n{args.query}\n")
        answer = agent.run(args.query)
        print(f"\n[Final Answer]:\n{answer}\n")
        if args.session:
            save_session(args.session, agent.messages)
        return

    while True:
        try:
            query = input("\nagent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            continue
        if query in {"quit", "exit"}:
            return
            
        try:
            if not args.session:
                # If no session, don't hold chat memory across queries
                agent = LegalAgent(llm, retriever, k=args.k)
                
            started = time.time()
            answer = agent.run(query)
            print(f"\n[Final Answer]:\n{answer}\n")
            print(f"Finished in {time.time() - started:.1f}s")
            
            if args.session:
                save_session(args.session, agent.messages)
                
        except Exception as exc:
            print(f"failed: {exc}")


if __name__ == "__main__":
    main()
