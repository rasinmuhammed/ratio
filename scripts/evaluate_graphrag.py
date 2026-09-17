#!/usr/bin/env python3
"""Evaluate GraphRAG implementation."""

import logging
import sys
import time
from pathlib import Path

from rag.agent import LegalAgent
from rag.generate import get_llm
from rag.hybrid import HybridRetriever
from rag.rerank import CrossEncoderReranker, RerankedRetriever
from rag.route import RoutedRetriever

INDEX_DIR = Path("data/index-sweep-450")

def main():
    logging.basicConfig(level=logging.WARNING)
    print("Loading index...", file=sys.stderr)
    started = time.time()
    
    hybrid = HybridRetriever(INDEX_DIR)
    reranked = RerankedRetriever(hybrid, CrossEncoderReranker())
    retriever = RoutedRetriever(reranked, hybrid.dense.payloads)
    llm = get_llm()
    
    print(f"Ready in {time.time() - started:.0f}s", file=sys.stderr)
    
    agent = LegalAgent(llm, retriever, k=4)
    
    query = "Does promissory estoppel apply against the State? Can you give examples from Patna High Court?"
    print(f"\n==================================================")
    print(f"Query: {query}")
    print(f"==================================================\n")
    
    for event in agent.stream(query):
        if event["type"] == "scratchpad":
            print(f"\n[Agent Memory/Scratchpad]:\n{event['content']}")
        elif event["type"] == "search":
            print(f"\n[Tool Use]: {event['content']}")
        elif event["type"] == "final_answer":
            print(f"\n==================================================")
            print(f"FINAL ANSWER:")
            print(f"==================================================\n")
            print(event["content"])
        elif event["type"] == "error":
            print(f"\n[ERROR]: {event['content']}")

if __name__ == "__main__":
    main()
