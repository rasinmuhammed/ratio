"""Multi-turn agentic reasoning for complex legal queries.

This script defines the LegalAgent which manages a conversation loop with an LLM, 
giving it access to a `search_index` tool to autonomously gather context before 
generating a final answer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rag.generate import LLM, SYSTEM_PROMPT, CITE_REMINDER
from rag.retrieve import Result
from rag.stance import classify as stance_of
from rag.graph import GraphDB

log = logging.getLogger(__name__)

# System prompt is identical to normal RAG, but with added instructions for tool use.
AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT + """
You have access to tools to search the legal database and submit your final answer.
If the user's question is complex, you must use the `search_index` tool multiple times to gather all necessary context.
You also have the `explore_network` tool. Use it when you find a highly relevant case (using its chunk_id) and want to traverse the GraphRAG citation network to find related precedents, dissenting opinions, or connected rules.
Once you have enough context, or if you cannot find any more relevant cases after 3 searches, you MUST use the `submit_final_answer` tool.

CRITICAL INSTRUCTION:
Every time you use a tool, you must write a `scratchpad`. This is your ONLY memory. The raw search results will be deleted to save space, so you MUST extract and write down any key quotes, citations, and rules you want to remember in your `scratchpad`.
"""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_index",
        "description": "Searches the legal database. Use this to gather context.",
        "parameters": {
            "type": "object",
            "properties": {
                "scratchpad": {
                    "type": "string",
                    "description": "Your memory. Summarize what you learned from the PREVIOUS search, extracting quotes and citations you want to keep."
                },
                "query": {
                    "type": "string",
                    "description": "The legal query to search for."
                }
            },
            "required": ["scratchpad", "query"],
        },
    },
}

EXPLORE_TOOL = {
    "type": "function",
    "function": {
        "name": "explore_network",
        "description": "Traverses the legal GraphRAG network. Given a chunk_id from a previous search, finds the top 5 most highly connected cases (citations and semantic neighbors).",
        "parameters": {
            "type": "object",
            "properties": {
                "scratchpad": {
                    "type": "string",
                    "description": "Your memory. Extract key quotes and citations before querying."
                },
                "chunk_id": {
                    "type": "string",
                    "description": "The exact chunk_id (e.g. 'https://indiankanoon.org/doc/12345/#0') of the case you want to explore around."
                }
            },
            "required": ["scratchpad", "chunk_id"],
        },
    },
}

ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_final_answer",
        "description": "Submit your final answer to the user once you have enough context.",
        "parameters": {
            "type": "object",
            "properties": {
                "scratchpad": {
                    "type": "string",
                    "description": "Final synthesis of what you learned."
                },
                "final_answer": {
                    "type": "string",
                    "description": "Your comprehensive final answer, adhering to all citation rules."
                }
            },
            "required": ["scratchpad", "final_answer"],
        },
    },
}

class LegalAgent:
    def __init__(self, llm: LLM, retriever: Any, k: int = 6):
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.messages: list[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT + CITE_REMINDER}
        ]
        self.all_sources: dict[str, Any] = {}
        # data/ is entirely gitignored, so graph.db (built by
        # scripts/build_graph.py, never committed) exists only on whatever
        # machine happened to run that script. Without this guard, cloning
        # this repo and constructing a LegalAgent raised FileNotFoundError
        # before a single query could run, on every machine except the one
        # that built it. explore_network degrades to "not available" at
        # call time instead of taking down the whole agent at construction.
        try:
            self.graph: GraphDB | None = GraphDB()
        except FileNotFoundError as exc:
            log.warning("GraphRAG unavailable, explore_network will report "
                        "no results: %s", exc)
            self.graph = None

    def format_search_results(self, results: list[Result]) -> str:
        """Format retrieved chunks exactly as generate.answer does.

        Numbers every source by its position in self.all_sources, not by
        position within this one call's result list. Multiple search_index
        calls each used to restart numbering at [1], so a citation the model
        wrote in its final answer could not be mapped back to a real source
        once more than one search had happened, since [1] meant a different
        chunk depending on which call it came from. self.all_sources
        preserves insertion order, so the index assigned here is the same
        one the citation parser and the API's sources event use later.
        """
        if not results:
            return "No results found."

        context_parts = []
        for r in results:
            index = list(self.all_sources.keys()).index(r.chunk_id) + 1
            stance = stance_of(r.text)
            meta = r.metadata
            court = meta.get("court", "Unknown Court")
            title = meta.get("title", "Unknown Title")
            context_parts.append(
                f"--- Source [{index}] ({stance}) ---\n"
                f"Court: {court} | Title: {title} | Cited by: {meta.get('cited_by', 0)}\n"
                f"{r.text}\n"
            )
        return "\n".join(context_parts)

    def stream(self, user_query: str):
        """Executes the autonomous reasoning loop and yields events."""
        self.messages.append({"role": "user", "content": user_query})
        
        # Guard against infinite loops
        max_turns = 5
        turn = 0
        
        while turn < max_turns:
            turn += 1
            log.info(f"Agent turn {turn}...")
            
            # If we are approaching the limit, force the agent to submit
            if turn == max_turns - 1:
                self.messages.append({
                    "role": "system",
                    "content": "CRITICAL: You are running out of time. You MUST use the `submit_final_answer` tool in this turn. Do not search anymore. Use the context you have."
                })
            
            # Send current memory to LLM
            response_msg = self.llm.chat(self.messages, tools=[SEARCH_TOOL, EXPLORE_TOOL, ANSWER_TOOL])
            
            # Append LLM's raw response to memory (required for function calling)
            self.messages.append(response_msg)
            
            # Memory Pruning: Once the LLM has successfully responded
            for msg in self.messages[:-2]:
                if msg.get("role") == "tool" and len(msg.get("content", "")) > 500:
                    msg["content"] = "[Raw search results excised for memory pruning. Key information was preserved by the agent in its scratchpad argument.]"

            if "tool_calls" in response_msg and response_msg["tool_calls"]:
                for tool_call in response_msg["tool_calls"]:
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        scratchpad = args.get("scratchpad", "")
                        if scratchpad:
                            yield {"type": "scratchpad", "content": scratchpad}
                    except json.JSONDecodeError:
                        args = {}

                    if tool_call["function"]["name"] == "search_index":
                        search_query = args.get("query", "")
                        yield {"type": "search", "content": search_query}
                        
                        if search_query:
                            results = self.retriever.search(search_query, k=self.k)
                            for r in results:
                                if r.chunk_id not in self.all_sources:
                                    self.all_sources[r.chunk_id] = r
                            formatted_context = self.format_search_results(results)
                        else:
                            formatted_context = "Error: Invalid query provided."
                            
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": formatted_context,
                        })
                        
                    elif tool_call["function"]["name"] == "explore_network":
                        chunk_id = args.get("chunk_id", "")
                        yield {"type": "search", "content": f"[GraphRAG] Traversing network for {chunk_id}"}
                        
                        if chunk_id and self.graph is None:
                            formatted_context = (
                                "GraphRAG network unavailable on this "
                                "deployment (graph.db not built).")
                        elif chunk_id:
                            neighbors = self.graph.get_neighbors(chunk_id)
                            if not neighbors:
                                formatted_context = f"No connected cases found in the GraphRAG network for {chunk_id}."
                            else:
                                context_parts = [f"GraphRAG Network Results for {chunk_id}:"]
                                for n in neighbors:
                                    context_parts.append(f"- Related Case [{n.chunk_id}]: {n.court} | {n.title} (Relevance Score: {n.score:.2f})")
                                formatted_context = "\n".join(context_parts)
                        else:
                            formatted_context = "Error: Invalid chunk_id provided."
                            
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": formatted_context,
                        })
                        
                    elif tool_call["function"]["name"] == "submit_final_answer":
                        final_answer = args.get("final_answer", "")
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": "Final answer submitted successfully.",
                        })
                        yield {"type": "final_answer", "content": final_answer}
                        return
                        
                # Loop continues so LLM can read the tool result and act again
                continue
                
            # If no tool was called, force it to use the submit tool
            log.warning("Agent failed to use a tool. Forcing retry.")
            self.messages.append({
                "role": "system",
                "content": "Error: You returned plain text. You MUST invoke either 'search_index' or 'submit_final_answer'."
            })
        yield {"type": "error", "content": "Agent stopped: Reached maximum turn limit without returning an answer."}

    def run(self, user_query: str) -> str:
        """Synchronous wrapper for stream() for CLI usage."""
        for event in self.stream(user_query):
            if event["type"] == "scratchpad":
                print(f"\n[Agent Scratchpad]: {event['content']}")
            elif event["type"] == "search":
                print(f"[Turn] Agent searching index for: {event['content']!r}")
            elif event["type"] == "final_answer":
                return event["content"]
            elif event["type"] == "error":
                return event["content"]
        return "Unknown error"
