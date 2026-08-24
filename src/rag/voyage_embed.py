"""Voyage AI embeddings, behind the same shape Retriever already expects.

voyage-law-2 was trained on roughly a trillion legal tokens spanning US,
Chinese, German and Indian law. bge-small-en-v1.5 has seen none of that, and
this is the largest unpulled lever in the retrieval pipeline: dense recall on
exact-match queries sits at 0.7% of achievable, and no amount of chunk-size or
fusion-weight tuning changes what the model was trained on.

Unlike sentence-transformers, encoding is an API call, so every batch can fail
on a network error or a rate limit, and every call costs tokens against the
free tier. Both are handled here rather than at each call site, the same
reason retry logic lives inside GroqLLM.complete rather than in every caller.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "voyage-law-2"
DIM = 1024

# The API's own per-request budget is 120K tokens for voyage-law-2, but an
# unverified account (no card on file) is throttled far below that: 3
# requests/minute, 10,000 tokens/minute. A legal chunk runs close to 450
# tokens, so 200 texts is ~90K tokens, nine times over the unverified ceiling
# in one call. That request cannot succeed no matter how long a retry waits,
# since the limit is fixed and the request is simply too large.
#
# 15 texts stays around 6,750 tokens, safe margin under 10,000 even for
# longer-than-average chunks. Raise this once a card is added and standard
# limits apply.
BATCH_SIZE = 15

# Rolling-window limits are per-minute totals across every request, not a
# per-request check, so two compliant-sized batches sent seconds apart can
# still exceed 10K TPM together. A flat sleep between requests is simpler and
# safer than tracking a rolling token count, at the cost of some idle time.
# 65s (not 60) leaves margin for clock skew between this machine and Voyage's.
MIN_INTERVAL_SECONDS = 65.0


def _load_env_file(path: Path = Path(".env")) -> None:
    """Same six-line reader as generate.py's. Duplicated rather than
    imported, so this module has no dependency on the Groq provider."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class VoyageEmbedder:
    """Embeds text with Voyage's API.

    Query and document embeddings are asymmetric by design in Voyage's
    models: `input_type="query"` and `input_type="document"` prepend
    different retrieval-focused prompts internally. That replaces the manual
    QUERY_PREFIX string bge-small needed, so there is deliberately no prefix
    constant here, unlike retrieve.py.
    """

    def __init__(
        self,
        model: str = MODEL_NAME,
        api_key: str | None = None,
        batch_size: int = BATCH_SIZE,
        retries: int = 5,
        min_interval: float = MIN_INTERVAL_SECONDS,
    ) -> None:
        import voyageai

        if api_key is None:
            _load_env_file()
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise ValueError("VOYAGE_API_KEY is not set (env or .env file)")

        self.client = voyageai.Client(api_key=key)
        self.model = model
        self.batch_size = batch_size
        self.retries = retries
        # min_interval=0 once a card is on file and standard limits apply.
        self.min_interval = min_interval
        self._last_request: float | None = None

    def get_embedding_dimension(self) -> int:
        return DIM

    def _pace(self) -> None:
        """Block until min_interval has passed since the last request.

        This is what actually keeps an unverified account under 10K TPM: the
        limit is a rolling window over all requests, so retry backoff on a
        429 does not help when every batch is already correctly sized. The
        pacing has to happen before the request, not after a failure."""
        if self._last_request is not None:
            elapsed = time.monotonic() - self._last_request
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        out: list[list[float]] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for batch_no, start in enumerate(range(0, len(texts), self.batch_size), 1):
            batch = texts[start:start + self.batch_size]
            self._pace()
            if n_batches > 1:
                logger.info("voyage batch %d/%d", batch_no, n_batches)
            for attempt in range(self.retries):
                try:
                    result = self.client.embed(
                        batch, model=self.model, input_type=input_type,
                    )
                    out.extend(result.embeddings)
                    break
                except Exception as exc:  # noqa: BLE001 - the SDK's own
                    # exception hierarchy isn't documented here, so this
                    # catches broadly and relies on retries running out
                    # rather than pattern-matching a message string.
                    if attempt == self.retries - 1:
                        raise
                    wait = 2 ** attempt * 5
                    logger.warning("voyage embed failed (%s), retrying in %ds",
                                  exc, wait)
                    time.sleep(wait)

        vecs = np.array(out, dtype=np.float32)
        # Normalised here rather than trusting the API, so downstream code's
        # assumption that cosine similarity reduces to a dot product (the same
        # assumption retrieve.py's rank() makes) holds regardless of what the
        # API returns by default.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], input_type="query")[0]
