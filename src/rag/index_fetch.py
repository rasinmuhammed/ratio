"""Fetch a built vector index from a Hugging Face Hub dataset repo if it is
not already present on disk.

data/ is gitignored (see .gitignore), on purpose: the point was to keep an
unfinished 630MB+ index out of the public portfolio repo, not to make the
index unreachable. But that decision means a deploy target (a fresh
Hugging Face Space, a fresh Cloud Run container, a fresh Oracle VM) starts
with no index at all unless something puts one there. Baking a 630MB index
into a Docker image is the wrong fix, since it makes every rebuild of the
image slower and the image itself bloated; pulling it once at container
startup from a dataset repo built for exactly this is the same shape of
decision embed.py already made about sharding the vectors themselves.

    uv run python scripts/upload_index_to_hf.py --index data/index --repo <you>/ratio-index

Deploy-time, this module is called once during lifespan before
HybridRetriever ever touches INDEX_DIR:

    ensure_index(INDEX_DIR, repo=os.environ.get("RATIO_INDEX_REPO"))

If RATIO_INDEX_REPO is unset (the default, matching local development),
this is a no-op: nothing about running locally changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def index_is_complete(index_dir: Path) -> bool:
    """The same three files HybridRetriever/Retriever actually read.
    Checking for the directory alone was the earlier bug (data/index-full
    exists as a directory with vectors in it and still can't load, because
    payloads.jsonl is missing) - existence of the directory is not
    existence of a usable index.
    """
    index_dir = Path(index_dir)
    if not index_dir.exists():
        return False
    required = ["ids.json", "meta.json", "payloads.jsonl"]
    if not all((index_dir / name).exists() for name in required):
        return False
    return any(index_dir.glob("vectors-*.npy"))


def ensure_index(index_dir: Path, repo: str | None) -> None:
    """Download `repo` into `index_dir` if the index isn't already usable
    there. Safe to call on every startup: a no-op once the index exists,
    and a no-op entirely when `repo` is None.
    """
    index_dir = Path(index_dir)
    if index_is_complete(index_dir):
        logger.info("index already present at %s, skipping download", index_dir)
        return

    if not repo:
        # Not an error: this is the local-development path, where the
        # index is expected to already exist from build_index.py. Raising
        # here would make every local run that hasn't set the env var
        # fail with a misleading "download failed" instead of the real,
        # existing FileNotFoundError HybridRetriever already gives.
        logger.info(
            "no usable index at %s and RATIO_INDEX_REPO is unset, "
            "leaving HybridRetriever to raise its own error", index_dir,
        )
        return

    logger.info("downloading index from %s into %s...", repo, index_dir)
    from huggingface_hub import snapshot_download

    index_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(index_dir),
    )

    if not index_is_complete(index_dir):
        # Downloaded something, but not the three files retrieval actually
        # needs, most likely the wrong repo or a partial upload. Failing
        # loudly here beats a HybridRetriever FileNotFoundError three
        # frames deeper with no mention that a download even happened.
        raise RuntimeError(
            f"downloaded {repo!r} into {index_dir}, but it is still "
            f"missing one of ids.json / meta.json / payloads.jsonl / "
            f"vectors-*.npy. Check the repo contents."
        )
    logger.info("index download complete")
