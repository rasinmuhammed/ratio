"""Generation: Retrieved chunk -> a grounded answer with citations."""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Protocol

import httpx

from rag.retrieve import Result
from rag.stance import describe

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Overridable, because this is the second time a provider has retired the model
# underneath the code. llama-3.3-70b-versatile worked on 12 August and returned
# 404 on 23 August, with every Llama model gone from the catalogue. A hardcoded
# id turns a provider's roadmap into an outage here.
#
#     GROQ_MODEL=openai/gpt-oss-20b uv run python scripts/ask.py "..."
#
# Current models: https://api.groq.com/openai/v1/models
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Measured in whatever unit the injected `length` function returns. Pass
# token_length(model) and this means tokens; the default len() means
# characters. The unit is the caller's choice, so it is never implicit.
CONTEXT_BUDGET = 6000

# Reasoning models bill reasoning against the same budget as the answer. On
# openai/gpt-oss the `reasoning` field routinely runs longer than `content`, so
# 800, which was ample for a non-reasoning model, truncated real answers before
# they reached their citations, and the audit then scored the missing citations
# as the model failing to cite.
MAX_ANSWER_TOKENS = 2000

# A sentinel rather than free prose, so refusal is detectable in code without
# parsing natural language.
REFUSAL = "INSUFFICIENT_CONTEXT"

# The prompt as it stood before stance labelling existed. Kept so the change
# can be measured against it: with only the per-source notes toggled, both arms
# still carried the stance rules below and the comparison showed nothing.
SYSTEM_PROMPT_PLAIN = f"""You answer questions about Indian court judgments using \
only the numbered sources provided.

Rules:
- Use ONLY the sources below. Do not use outside knowledge.
- Cite every factual claim with the source number in square brackets, e.g. [2].
- If the sources do not contain enough information to answer, reply with \
exactly this and nothing else: {REFUSAL}
- Quote the judgment's language where precision matters.
- You are summarising what these judgments say. You are not giving legal advice.
"""

SYSTEM_PROMPT = f"""You answer questions about Indian court judgments using \
only the numbered sources provided.

Each source carries a note about whose voice it is in. A judgment records what \
counsel argued as well as what the court decided, and the two frequently \
disagree, so a submission is evidence of what was claimed and not of what the \
law is.

Rules:
- Use ONLY the sources below. Do not use outside knowledge.
- Cite every factual claim with the source number in square brackets, e.g. [2].
- If the sources do not contain enough information to answer, reply with \
exactly this and nothing else: {REFUSAL}
- Never state a party's submission as the legal position. If the only support \
for a point is a submission, say so explicitly, for example "the petitioner \
argued that ..., though the extract does not record the court's conclusion".
- Prefer the court's own reasoning over a submission wherever both are present.
- Quote the judgment's language where precision matters.
- You are summarising what these judgments say. You are not giving legal advice.
"""

# Repeated at the end of the user message, not only in the system prompt.
# openai/gpt-oss ignores the system-prompt citation rule outright on roughly one
# run in three, producing a fluent answer with no brackets at all, and it does
# so non-deterministically at temperature 0. Instructions in the last position
# get followed more reliably. Measured over three trials each, plain had one
# total failure and this had none, which is suggestive rather than conclusive.
CITE_REMINDER = (
    "\nEvery factual sentence must end with its source number in square "
    "brackets, for example [2]. An answer with no bracketed citations is not "
    "acceptable."
)


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
    truncated: bool = False      # answer hit max_tokens, so citations may be lost

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
            retries: int = 6,
            sleep=time.sleep,
            ) -> None:
        if api_key is None:
            _load_env_file()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set (env or .env file)")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retries = retries
        self.sleep = sleep
        self.last_finish_reason: str | None = None

    def complete(self, system: str, user: str) -> str:
        """Retries on 429 rather than raising it.

        The free tier allows 8,000 tokens per minute, and one audited answer
        costs more than that, so a rate limit is the expected steady state of a
        long run and not an error. Retrying belongs here at the provider
        boundary: the alternative was every caller growing its own backoff
        loop, which is how attribute._ask and validate_stance each ended up
        with a slightly different one.

        Groq sends `retry-after` when it knows how long to wait. Honouring it
        beats guessing, and guessing short is worse than waiting.
        """
        for attempt in range(self.retries):
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

            if response.status_code != 429:
                response.raise_for_status()
                choice = response.json()["choices"][0]
                # Recorded rather than ignored. A truncated answer has fewer
                # citations than the model intended, and scoring it as though
                # it were complete turns a budget setting into a finding about
                # the model.
                self.last_finish_reason = choice.get("finish_reason")
                if self.last_finish_reason == "length":
                    logger.warning(
                        "answer truncated at max_tokens=%d, citations may be "
                        "missing", self.max_tokens,
                    )
                return choice["message"]["content"].strip()

            if attempt == self.retries - 1:
                break
            wait = float(response.headers.get("retry-after", 0)) or 2 ** attempt * 5
            logger.info("rate limited, waiting %.0fs", wait)
            self.sleep(min(wait + 1, 90))

        raise RuntimeError(f"Groq rate limit hit: {response.text[:200]}")

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------

def build_prompt(
        query: str,
        chunks: list[Result],
        budget: int = CONTEXT_BUDGET,
        length: Callable[[str], int] = len,
        stance_notes: bool = True,
    ) -> tuple[str, list[Result]]:
    """Assemble the user prompt, returning it and the chunks that fit.

    Greedy in rank order, stopping at the first chunk that would overflow
    rather than skipping it. Skipping would let a lower-ranked chunk displace
    a higher-ranked one, which is not obviously an improvement.

    `stance_notes` exists so the labelled and unlabelled prompts can be run
    against the same sources and compared. Without it the old behaviour is
    unreachable and the claim that labelling changes anything is untestable.
    """
    used: list[Result] = []
    blocks: list[str] = []
    spent = length(query)

    for chunk in chunks:
        # The stance note is counted against the budget rather than added
        # afterwards, otherwise every source silently costs more than the
        # accounting says and the context overflows near the limit.
        note = f"({describe(chunk.text)}) " if stance_notes else ""
        block = f"[{len(used) + 1}] {note}{chunk.text}"
        cost = length(block)
        if used and spent + cost > budget:
            break
        used.append(chunk)
        blocks.append(block)
        spent += cost

    sources = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    prompt = (f"Sources:\n\n{sources}\n\nQuestion: {query}\n{CITE_REMINDER}"
              f"\n\nAnswer:")

    return prompt, used

# A run of one or more bracketed citations, so "[1]", "[1, 3]" and "[1][2]"
# are all recognised. The earlier pattern was \[(\d+)\], which matches "[1]"
# and silently ignores "[1, 3]" entirely: the model writes that form regularly,
# so cited sources were being dropped and every count built on them was low.
#
# Numbers are capped at three digits to keep law reports out. Indian citations
# are written "[1998] 2 SCC 341", and the model quotes judgment language, so an
# unbounded \d+ reads 1998 as a source number and then reports it as a citation
# to a source that does not exist. A hallucinated source number is small, near
# the range actually offered, so nothing real is lost by the bound.
_CITATION = re.compile(r"(?:\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\]\s*)+")
_NUMBER = re.compile(r"\d+")


def parse_answer(text: str, n_sources: int) -> tuple[bool, list[int], list[int]]:
    """Return (refused, cited, invalid_citations)."""
    refused = text.strip().startswith(REFUSAL)

    seen: list[int] = []
    for run in _CITATION.finditer(text):
        for number in _NUMBER.findall(run.group(0)):
            n = int(number)
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
    stance_notes: bool = True,
) -> Answer:
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    chunks = retriever.search(query, k=k)

    # Logged, not enforced. Measured separation between a relevant query and
    # nonsense was 0.203 vs 0.120: directionally right, too noisy to gate on.
    scores = [c.score for c in chunks]
    gap = (max(scores) - median(scores)) if scores else 0.0

    prompt, used = build_prompt(query, chunks, budget=budget, length=length,
                                stance_notes=stance_notes)
    # The system prompt moves with the flag. Toggling only the per-source notes
    # leaves the stance rules in place, so both arms behave the same way and
    # the comparison measures nothing.
    system = SYSTEM_PROMPT if stance_notes else SYSTEM_PROMPT_PLAIN
    text = llm.complete(system, prompt)
    refused, cited, invalid = parse_answer(text, len(used))

    if invalid:
        logger.warning("model cited non-existent sources %s (had %d)",
                       invalid, len(used))

    return Answer(
        truncated=getattr(llm, "last_finish_reason", None) == "length",
        text=text,
        refused=refused,
        sources=used,
        cited=cited,
        invalid_citations=invalid,
        score_gap=gap,
    )


