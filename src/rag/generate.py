"""Generation: Retrieved chunk -> a grounded answer with citations."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Protocol

import httpx

from rag.retrieve import Result

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Measured in whatever unit the injected `length` function returns. Pass
# token_length(model) and this means tokens; the default len() means
# characters. The unit is the caller's choice, so it is never implicit.
CONTEXT_BUDGET = 6000
MAX_ANSWER_TOKENS = 800

# A sentinel rather than free prose, so refusal is detectable in code without
# parsing natural language.
REFUSAL = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You answer questions about Indian court judgments using \
only the numbered sources provided.

Rules:
- Use ONLY the sources below. Do not use outside knowledge.
- Cite every factual claim with the source number in square brackets, e.g. [2].
- If the sources do not contain enough information to answer, reply with \
exactly this and nothing else: {REFUSAL}
- Quote the judgment's language where precision matters.
- You are summarising what these judgments say. You are not giving legal advice.
"""

class LLM(Protocol):
    """Anything that turns a prompt into a text.
    Protocol so generation logic never depends on a provider."""

    def complete(self, system: str, user: str) -> str: ...


class Searcher(Protocol):
    """Anything that returns ranked chunks. Retriever and HybridRetriever
    both satisfy this, and so does a fake in tests."""

    def search(self, query: str, k: int) -> list[Result]: ...

@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    refused: bool
    sources: list[Result]        # chunks actually sent to the model
    cited: list[int]             # 1-based source numbers the model cited
    invalid_citations: list[int] # cited numbers with no matching source
    score_gap: float             # top score minus median, logged not enforced

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

def _load_env_file(path: Path = Path(".env")) -> None:
    """Minimal .env reader.

    os.environ does not see a .env file on its own. A dependency could do
    this, but it is six lines and avoids one. Existing environment variables
    win, so an explicit export always overrides the file.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class GroqLLM:
    def __init__(
            self,
            model: str = GROQ_MODEL,
            api_key: str | None = None,
            timeout: float = 60.0,
            max_tokens: int = MAX_ANSWER_TOKENS,
            ) -> None:
        if api_key is None:
            _load_env_file()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set (env or .env file)")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = httpx.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],

                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
        )

        if response.status_code == 429:
            raise RuntimeError(f"Groq rate limit hit: {response.text[:200]}")
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------

def build_prompt(
        query: str,
        chunks: list[Result],
        budget: int = CONTEXT_BUDGET,
        length: Callable[[str], int] = len,
    ) -> tuple[str, list[Result]]:
    """Assemble the user prompt, returning it and the chunks that fit.

    Greedy in rank order, stopping at the first chunk that would overflow
    rather than skipping it. Skipping would let a lower-ranked chunk displace
    a higher-ranked one, which is not obviously an improvement.
    """
    used: list[Result] = []
    blocks: list[str] = []
    spent = length(query)

    for chunk in chunks:
        block = f"[{len(used) + 1}] {chunk.text}"
        cost = length(block)
        if used and spent + cost > budget:
            break
        used.append(chunk)
        blocks.append(block)
        spent += cost

    sources = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    prompt = f"Sources:\n\n{sources}\n\nQuestion: {query}\n\nAnswer:"

    return prompt, used

_CITATION = re.compile(r"\[(\d+)\]")

def parse_answer(text: str, n_sources: int) -> tuple[bool, list[int], list[int]]:
    """Return (refused, cited, invalid_citations)."""
    refused = text.strip().startswith(REFUSAL)

    seen: list[int] = []
    for match in _CITATION.finditer(text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)

    cited = [n for n in seen if 1 <= n <= n_sources]
    invalid = [n for n in seen if n < 1 or n > n_sources]
    return refused, cited, invalid

# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def answer(
    query: str,
    retriever: Searcher,
    llm: LLM,
    k: int = 5,
    budget: int = CONTEXT_BUDGET,
    length: Callable[[str], int] = len,
) -> Answer:
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    chunks = retriever.search(query, k=k)

    # Logged, not enforced. Measured separation between a relevant query and
    # nonsense was 0.203 vs 0.120: directionally right, too noisy to gate on.
    scores = [c.score for c in chunks]
    gap = (max(scores) - median(scores)) if scores else 0.0

    prompt, used = build_prompt(query, chunks, budget=budget, length=length)
    text = llm.complete(SYSTEM_PROMPT, prompt)
    refused, cited, invalid = parse_answer(text, len(used))

    if invalid:
        logger.warning("model cited non-existent sources %s (had %d)",
                       invalid, len(used))

    return Answer(
        text=text,
        refused=refused,
        sources=used,
        cited=cited,
        invalid_citations=invalid,
        score_gap=gap,
    )


