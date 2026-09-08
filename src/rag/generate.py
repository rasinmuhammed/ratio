"""Generation: Retrieved chunk -> a grounded answer with citations."""

from __future__ import annotations

import json
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

# 429 was the only status ever retried, until a TokenRouter run crashed on
# query 8 of 15 with a bare 503, real progress lost to an error class that
# exists specifically to mean "transient, try again". 429 says the caller
# asked too fast; 5xx says the server is having a bad moment, both are the
# same kind of not-actually-broken failure from a retry loop's point of
# view, and only one of them was getting one.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

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

# Mercury 2, from Inception Labs, is a diffusion LLM rather than autoregressive,
# but it exposes an OpenAI-compatible chat completions endpoint, so it satisfies
# the same LLM Protocol as GroqLLM with no changes to answer() or build_prompt().
# https://docs.inceptionlabs.ai/capabilities/chat-completions
MERCURY_URL = "https://api.inceptionlabs.ai/v1/chat/completions"
MERCURY_MODEL = os.environ.get("MERCURY_MODEL", "mercury-2")

# TokenRouter, OpenAI-compatible, one free model tried so far: z-ai/glm-5.3-free.
# Nothing here is measured yet the way Groq and Mercury have been, so
# TokenRouterLLM copies GroqLLM's retry shape defensively rather than
# assuming it behaves the same way under load, the pattern is proven, this
# provider's actual rate limits and failure modes are not.
TOKEN_ROUTER_URL = "https://api.tokenrouter.com/v1/chat/completions"
TOKEN_ROUTER_MODEL = os.environ.get("TOKEN_ROUTER_MODEL", "z-ai/glm-5.3-free")

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

# Added after Mercury 2's audit showed the opposite failure from gpt-oss-120b:
# zero uncited claims across 8 queries, but only 66.7% citation precision,
# where gpt-oss-120b under-cites and Mercury over-commits. CITE_REMINDER only
# says a citation must be present, never that it must be correct, so a model
# that complies eagerly has no reason not to attach a plausible-looking number
# it hasn't actually verified. This makes a wrong number cost more than a
# missing one. Off by default so it never silently changes an existing
# benchmark; pass precision_reminder=True to answer() to test it.
PRECISION_REMINDER = (
    "\nA citation is only correct if the exact source you numbered actually "
    "states the claim it is attached to. Check this before writing the "
    "number, not after. A wrong citation is a worse failure than a missing "
    "one, so when you are not certain which source supports a claim, leave "
    "the claim uncited rather than guess.\n"
    "Cite one source per claim: the single strongest match, not every source "
    "that seems related. Only attach a second source number if the claim "
    "genuinely states two separate facts that come from two different "
    "sources. Citing three sources for one sentence to be safe makes each "
    "individual citation more likely to be wrong, not less, because the "
    "check is applied to every number you write, not just the best one."
)


# Structured citation output. Every prompt-engineering fix so far,
# CITE_REMINDER, PRECISION_REMINDER, has been a stronger request, not a
# guarantee, and the measured ceiling on both models stayed around 10-25%
# wrong or missing regardless. A schema-constrained response changes the
# category of guarantee: source_ids is a required field on every claim
# object, so the decoder cannot produce a claim with no citation slot, the
# same way it cannot produce a string where the schema says integer. This
# does not make a citation correct, that is still what citation_precision
# measures, it makes an *uncited* claim structurally impossible instead of
# merely discouraged.
#
# https://docs.inceptionlabs.ai/capabilities/structured-outputs
GROUNDED_ANSWER_SCHEMA = {
    "name": "GroundedAnswer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "refused": {
                "type": "boolean",
                "description": "true if the sources do not contain enough "
                               "information to answer the question",
            },
            "claims": {
                "type": "array",
                "description": "empty if refused is true",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "one factual claim, quoting the "
                                           "source's language where "
                                           "precision matters",
                        },
                        "source_ids": {
                            "type": "array",
                            "description": "1-based numbers of every source "
                                           "that supports this exact claim, "
                                           "the strongest match rather than "
                                           "every source that seems related",
                            "items": {"type": "integer"},
                            "minItems": 1,
                        },
                    },
                    "required": ["text", "source_ids"],
                },
            },
        },
        "required": ["refused", "claims"],
    },
}


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
            # 6 (with a 90s wait cap) was fine standalone, but attribute._ask
            # wraps every call to this in its own retry loop, and the two
            # multiply: dropped to 4 with a 30s cap on 2026-09-07 alongside
            # _ask's own attempts drop, see that docstring for the full
            # arithmetic. Still real resilience, no longer an hours-long tail.
            retries: int = 4,
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

        A 15-query judge run crashed on query 12 with
        httpcore.ConnectError: Connection reset by peer, a transport failure
        below the HTTP layer that the 429 check never sees, because
        httpx.post() raised before a response object existed at all. One
        dropped packet was throwing away every prior query's work. Retried
        the same as a 429, since a reset connection is exactly as transient.
        """
        for attempt in range(self.retries):
            try:
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
            except httpx.TransportError as exc:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Groq connection failed: {exc}") from exc
                wait = 2 ** attempt * 5
                logger.info("connection error (%s), retrying in %ds", exc, wait)
                self.sleep(wait)
                continue

            if response.status_code not in RETRYABLE_STATUS:
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
                # Same guard as MercuryLLM.complete: a reasoning model (the
                # judge is gpt-oss-20b) can spend the whole budget on
                # reasoning tokens and never reach content, same failure,
                # same fix, applied here second because the judge's one-word
                # answers made it look impossible until it happened anyway.
                content = choice["message"].get("content")
                return (content or "").strip()

            if attempt == self.retries - 1:
                break
            wait = float(response.headers.get("retry-after", 0)) or 2 ** attempt * 5
            logger.info("status %d, waiting %.0fs", response.status_code, wait)
            self.sleep(min(wait + 1, 30))

        raise RuntimeError(f"Groq gave status {response.status_code} after all retries: {response.text[:200]}")


class MercuryLLM:
    def __init__(
            self,
            model: str = MERCURY_MODEL,
            api_key: str | None = None,
            timeout: float = 60.0,
            # gpt-oss needed MAX_ANSWER_TOKENS raised from 800 to 2000 because a
            # reasoning model bills its reasoning against the same budget as the
            # answer. reasoning_effort="high" hits the identical failure much
            # harder: at 4000 (double) the live 15-query run still truncated 4
            # times before a single answer completed. Doubled tokens did not
            # buy double headroom, because reasoning at "high" apparently scales
            # with the budget offered rather than the query's real difficulty.
            # None here means "pick from reasoning_effort", not "unset".
            max_tokens: int | None = None,
            retries: int = 6,
            sleep=time.sleep,
            # None omits the field so the original baseline (no override, and
            # the run 8-query numbers were measured against) stays reachable.
            # "high" trades speed for verification; see complete()'s docstring.
            reasoning_effort: str | None = None,
            ) -> None:
        if api_key is None:
            _load_env_file()
        self.api_key = api_key or os.environ.get("INCEPTION_API_KEY")
        if not self.api_key:
            raise ValueError("INCEPTION_API_KEY is not set (env or .env file)")
        self.model = model
        self.timeout = timeout
        # 8000 for "high": 4000 still truncated 4 of the first handful of
        # queries on this benchmark, so this is a measured retry, not a guess
        # padded for safety. 4000 for everything else, matching what already
        # ran clean at default effort.
        if max_tokens is not None:
            self.max_tokens = max_tokens
        elif reasoning_effort == "high":
            self.max_tokens = MAX_ANSWER_TOKENS * 4
        else:
            self.max_tokens = MAX_ANSWER_TOKENS * 2
        self.retries = retries
        self.sleep = sleep
        # Mercury exposes tunable reasoning effort (instant/low/medium/high) as
        # a real request parameter, not just a prompt-level nudge. The default
        # citation-audit run used whatever Mercury's own default is, which is
        # tuned for speed, and speed is exactly the axis this project doesn't
        # need from generation. "high" trades some of that latency back for
        # the verification a citation actually requires.
        self.reasoning_effort = reasoning_effort
        self.last_finish_reason: str | None = None

    def complete(self, system: str, user: str) -> str:
        """Same shape as GroqLLM.complete: Bearer auth, retry on 429, same
        choices[0].message.content response, because the endpoint is
        OpenAI-compatible by design.

        Not 0.0. That was the original guess, flagged here as untested
        against a diffusion model. It turned out to be actively wrong: a
        direct test call got back
        "Requested temperature 0.0 is not within [0.5, 1]. Temperature has
        been set to 0.75", which the API returns in a `warning` field this
        code was never reading. Every prior Mercury run was silently
        sampling at 0.75, not the 0.0 Groq runs at, an unannounced
        confound behind every run-to-run swing measured so far. 0.5 is now
        sent explicitly, Mercury's floor, closest available to deterministic.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.5,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort

        for attempt in range(self.retries):
            # Same transport-error retry as GroqLLM.complete, see that
            # docstring: a raw connection reset raises before there is a
            # response to check the status code of.
            try:
                response = httpx.post(
                    MERCURY_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    # An explicit per-phase Timeout, not a bare float, after a
                    # request sat in an ESTABLISHED TCP connection for 30+
                    # minutes with no response and no exception, well past
                    # what a 60.0 float timeout should have allowed on any
                    # phase. Whether that was a real httpx/timeout edge case
                    # or something server-side, an explicit read timeout is
                    # the direct fix either way, and this makes the applied
                    # value visible instead of trusting the float's coercion.
                    timeout=httpx.Timeout(
                        connect=10.0, read=self.timeout, write=10.0, pool=10.0,
                    ),
                )
            except httpx.TransportError as exc:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Mercury connection failed: {exc}") from exc
                wait = 2 ** attempt * 5
                logger.info("connection error (%s), retrying in %ds", exc, wait)
                self.sleep(wait)
                continue

            if response.status_code not in RETRYABLE_STATUS:
                response.raise_for_status()
                body = response.json()
                # Mercury returns request-level corrections in a top-level
                # `warning` field rather than failing the request, exactly how
                # temperature 0.0 silently became 0.75 for every run before
                # this one. Surfaced now instead of discarded.
                if body.get("warning"):
                    logger.warning("Mercury: %s", body["warning"])
                choice = body["choices"][0]
                self.last_finish_reason = choice.get("finish_reason")
                if self.last_finish_reason == "length":
                    logger.warning(
                        "answer truncated at max_tokens=%d, citations may be "
                        "missing", self.max_tokens,
                    )
                # content came back None once at reasoning_effort="high" on a
                # query that spent the entire budget on reasoning tokens and
                # never reached an answer. That is what last_finish_reason ==
                # "length" already means, and answer() already excludes a
                # truncated answer from scoring rather than crashing on it.
                content = choice["message"].get("content")
                return (content or "").strip()

            if attempt == self.retries - 1:
                break
            wait = float(response.headers.get("retry-after", 0)) or 2 ** attempt * 5
            logger.info("status %d, waiting %.0fs", response.status_code, wait)
            self.sleep(min(wait + 1, 90))

        raise RuntimeError(f"Mercury gave status {response.status_code} after all retries: {response.text[:200]}")

    def complete_structured(self, system: str, user: str, schema: dict) -> dict:
        """Same request/retry shape as complete(), constrained to `schema`
        via response_format, and returns the parsed object instead of a
        string: the API still returns the completion as a JSON *string* in
        message.content (confirmed against the docs example, not assumed),
        this method is what parses it, so a caller never touches json.loads.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.5,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": schema},
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort

        for attempt in range(self.retries):
            try:
                response = httpx.post(
                    MERCURY_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=httpx.Timeout(
                        connect=10.0, read=self.timeout, write=10.0, pool=10.0,
                    ),
                )
            except httpx.TransportError as exc:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Mercury connection failed: {exc}") from exc
                wait = 2 ** attempt * 5
                logger.info("connection error (%s), retrying in %ds", exc, wait)
                self.sleep(wait)
                continue

            if response.status_code not in RETRYABLE_STATUS:
                response.raise_for_status()
                body = response.json()
                if body.get("warning"):
                    logger.warning("Mercury: %s", body["warning"])
                choice = body["choices"][0]
                self.last_finish_reason = choice.get("finish_reason")
                content = choice["message"].get("content")
                if not content:
                    # Truncated before any JSON closed, or content=None from
                    # the reasoning-eats-the-budget failure documented in
                    # complete(). An empty dict here reads downstream as
                    # zero claims, not a crash, the same "excluded rather
                    # than scored" treatment truncated free-text answers get.
                    if self.last_finish_reason == "length":
                        logger.warning(
                            "structured answer truncated at max_tokens=%d",
                            self.max_tokens,
                        )
                    return {}
                try:
                    return json.loads(content)
                except json.JSONDecodeError as exc:
                    # strict:True is supposed to make this unreachable. If it
                    # happens anyway, that is itself a finding worth seeing
                    # in the log, not a silent empty answer indistinguishable
                    # from a truncation.
                    logger.warning("Mercury returned invalid JSON despite "
                                   "strict schema: %s", exc)
                    return {}

            if attempt == self.retries - 1:
                break
            wait = float(response.headers.get("retry-after", 0)) or 2 ** attempt * 5
            logger.info("status %d, waiting %.0fs", response.status_code, wait)
            self.sleep(min(wait + 1, 30))

        raise RuntimeError(f"Mercury gave status {response.status_code} after all retries: {response.text[:200]}")


class TokenRouterLLM:
    def __init__(
            self,
            model: str = TOKEN_ROUTER_MODEL,
            api_key: str | None = None,
            # Measured, not guessed: a real prompt (retrieved sources, not
            # the trivial "say hello" the integration was first checked
            # with) took 185s end to end on the free tier, needing one
            # retry past a 60s attempt to get there. 200s gives one attempt
            # real room to finish before falling back to a retry at all.
            timeout: float = 200.0,
            # 2000 (MAX_ANSWER_TOKENS) truncated 6 of 6 real answers in a
            # row in the first live audit, the exact reasoning-eats-the-
            # budget pattern already found and fixed for gpt-oss and
            # Mercury: GLM appears to spend real budget on something before
            # content, the same way those two do. Doubling to 4000 was the
            # first fix tried here and is what Mercury needed; set straight
            # to 8000, the level Mercury only needed at reasoning_effort=
            # "high", since GLM's failure rate at 2000 was total (6/6) and
            # there's no cost pressure to stay lower on a free-tier model,
            # so there is no reason to re-discover this ceiling twice.
            max_tokens: int = MAX_ANSWER_TOKENS * 4,
            retries: int = 4,
            sleep=time.sleep,
            ) -> None:
        if api_key is None:
            _load_env_file()
        self.api_key = api_key or os.environ.get("TOKEN_ROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("TOKEN_ROUTER_API_KEY is not set (env or .env file)")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retries = retries
        self.sleep = sleep
        self.last_finish_reason: str | None = None

    def complete(self, system: str, user: str) -> str:
        """Non-streaming, same shape as GroqLLM.complete: the reference
        snippet streamed and reassembled chunks, but the LLM Protocol here
        is complete() -> str, one string, so this asks for the whole
        response directly rather than reimplementing stream reassembly for
        no benefit at this call site.
        """
        for attempt in range(self.retries):
            try:
                response = httpx.post(
                    TOKEN_ROUTER_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.0,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=httpx.Timeout(
                        connect=10.0, read=self.timeout, write=10.0, pool=10.0,
                    ),
                )
            except httpx.TransportError as exc:
                if attempt == self.retries - 1:
                    raise RuntimeError(
                        f"TokenRouter connection failed: {exc}") from exc
                wait = 2 ** attempt * 5
                logger.info("connection error (%s), retrying in %ds", exc, wait)
                self.sleep(wait)
                continue

            if response.status_code not in RETRYABLE_STATUS:
                response.raise_for_status()
                choice = response.json()["choices"][0]
                self.last_finish_reason = choice.get("finish_reason")
                if self.last_finish_reason == "length":
                    logger.warning(
                        "answer truncated at max_tokens=%d, citations may be "
                        "missing", self.max_tokens,
                    )
                content = choice["message"].get("content")
                return (content or "").strip()

            if attempt == self.retries - 1:
                break
            wait = float(response.headers.get("retry-after", 0)) or 2 ** attempt * 5
            logger.info("status %d, waiting %.0fs", response.status_code, wait)
            self.sleep(min(wait + 1, 30))

        raise RuntimeError(f"TokenRouter gave status {response.status_code} after all retries: {response.text[:200]}")


# LLM_PROVIDER switches the whole system between providers with no code
# changes at the call site, the same shape GROQ_MODEL already uses to survive
# a model swap within Groq. This is what makes the Mercury-vs-gpt-oss
# comparison an env var instead of a temporary edit someone forgets to revert.
#
#     LLM_PROVIDER=mercury uv run python scripts/ask.py "..."
def get_llm() -> LLM:
    provider = os.environ.get("LLM_PROVIDER", "groq")
    if provider == "mercury":
        return MercuryLLM(reasoning_effort=os.environ.get("MERCURY_REASONING_EFFORT"))
    if provider == "tokenrouter":
        return TokenRouterLLM()
    return GroqLLM()

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------

def build_prompt(
        query: str,
        chunks: list[Result],
        budget: int = CONTEXT_BUDGET,
        length: Callable[[str], int] = len,
        stance_notes: bool = True,
        precision_reminder: bool = False,
        cite_reminder: bool = True,
    ) -> tuple[str, list[Result]]:
    """Assemble the user prompt, returning it and the chunks that fit.

    Greedy in rank order, stopping at the first chunk that would overflow
    rather than skipping it. Skipping would let a lower-ranked chunk displace
    a higher-ranked one, which is not obviously an improvement.

    `stance_notes` exists so the labelled and unlabelled prompts can be run
    against the same sources and compared. Without it the old behaviour is
    unreachable and the claim that labelling changes anything is untestable.

    `precision_reminder` is the same idea applied to PRECISION_REMINDER: off
    by default, on only for the run being compared against the baseline.

    `cite_reminder` exists for answer_structured(): telling a model to "cite
    in square brackets" while also constraining it to a JSON schema with a
    source_ids array is not a stronger instruction, it is a contradictory
    one. Structured mode gets the schema to enforce citation instead of a
    prompt asking for it, so the free-text reminder is not just redundant
    there, it is actively wrong guidance and needs to be off, not just
    unused.
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
    reminder = ""
    if cite_reminder:
        reminder = CITE_REMINDER + (PRECISION_REMINDER if precision_reminder else "")
    prompt = (f"Sources:\n\n{sources}\n\nQuestion: {query}\n{reminder}"
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
    precision_reminder: bool = False,
) -> Answer:
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    chunks = retriever.search(query, k=k)

    # Logged, not enforced. Measured separation between a relevant query and
    # nonsense was 0.203 vs 0.120: directionally right, too noisy to gate on.
    scores = [c.score for c in chunks]
    gap = (max(scores) - median(scores)) if scores else 0.0

    prompt, used = build_prompt(query, chunks, budget=budget, length=length,
                                stance_notes=stance_notes,
                                precision_reminder=precision_reminder)
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


class StructuredLLM(Protocol):
    def complete_structured(self, system: str, user: str, schema: dict) -> dict: ...


def answer_structured(
    query: str,
    retriever: Searcher,
    llm: StructuredLLM,
    k: int = 5,
    budget: int = CONTEXT_BUDGET,
    length: Callable[[str], int] = len,
    stance_notes: bool = True,
) -> Answer:
    """Same contract as answer(), an Answer with the same fields, built by a
    different route: GROUNDED_ANSWER_SCHEMA constrains the model to return
    claims with a required source_ids array, so this function reconstructs
    `.text` from that structure instead of asking parse_answer's regex to
    find brackets the model was merely asked, not required, to write.

    Everything downstream, attribute.split_claims, ask.py's display,
    audit_answers.py, reads `.text` and expects "claim text [n]" or
    "claim text [n, m]". Reconstructing exactly that format means none of it
    needs to know this Answer came from a schema rather than free text.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    chunks = retriever.search(query, k=k)
    scores = [c.score for c in chunks]
    gap = (max(scores) - median(scores)) if scores else 0.0

    # cite_reminder=False: see build_prompt's docstring. The schema is the
    # instruction now, a free-text bracket reminder would contradict it.
    prompt, used = build_prompt(query, chunks, budget=budget, length=length,
                                stance_notes=stance_notes, cite_reminder=False)
    system = SYSTEM_PROMPT if stance_notes else SYSTEM_PROMPT_PLAIN
    result = llm.complete_structured(system, prompt, GROUNDED_ANSWER_SCHEMA)

    refused = bool(result.get("refused"))
    claims = result.get("claims") or []
    n_sources = len(used)

    seen: list[int] = []
    blocks: list[str] = []
    for claim in claims:
        claim_text = str(claim.get("text", "")).strip()
        ids = [i for i in claim.get("source_ids", []) if isinstance(i, int)]
        if not claim_text or not ids:
            continue  # malformed despite the schema; drop rather than guess
        for i in ids:
            if i not in seen:
                seen.append(i)
        blocks.append(f"{claim_text} [{', '.join(str(i) for i in ids)}]")

    cited = [i for i in seen if 1 <= i <= n_sources]
    invalid = [i for i in seen if i < 1 or i > n_sources]
    if invalid:
        logger.warning("model cited non-existent sources %s (had %d)",
                       invalid, n_sources)

    # The audit found this exact gap live: refused=false with an empty
    # claims array, on 3 of 15 real queries, a state GROUNDED_ANSWER_SCHEMA
    # never ruled out (JSON Schema's strict mode here does not reliably
    # support "claims non-empty when refused is false" as a cross-field
    # constraint) and free-text answers structurally cannot produce, so it
    # was inflating precision/support by quietly removing hard cases from
    # the denominator instead of counting them as the non-answers they are.
    # Collapsed here into the same refusal semantics parse_answer already
    # has, so "no answer" always means one thing, not two.
    if not refused and not blocks:
        refused = True

    text = REFUSAL if refused else "\n\n".join(blocks)

    return Answer(
        truncated=getattr(llm, "last_finish_reason", None) == "length",
        text=text,
        refused=refused,
        sources=used,
        cited=cited,
        invalid_citations=invalid,
        score_gap=gap,
    )


