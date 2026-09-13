"""One-time: push a built index to a Hugging Face Hub dataset repo, so a
deploy target can pull it down at startup instead of needing it baked
into an image or rebuilt from scratch.

    uv run python scripts/upload_index_to_hf.py \
        --index data/index-full --repo yourname/ratio-index

Needs a Hugging Face account and a write-scoped token (huggingface-cli
login, or HF_TOKEN in the environment). Creates the dataset repo if it
does not already exist. Run this whenever the index changes; it does not
run automatically, matching the fact that rebuilding the index itself is
already a deliberate, manual step (see build_index.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.index_fetch import index_is_complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True,
                         help="local index directory, e.g. data/index")
    parser.add_argument("--repo", type=str, required=True,
                         help="HF Hub dataset repo id, e.g. yourname/ratio-index")
    args = parser.parse_args()

    if not index_is_complete(args.index):
        raise SystemExit(
            f"{args.index} does not look like a complete index (missing "
            f"ids.json, meta.json, payloads.jsonl, or vectors-*.npy). "
            f"Refusing to upload a partial index."
        )

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)
    print(f"uploading {args.index} to {args.repo} (dataset repo)...")
    api.upload_folder(
        folder_path=str(args.index),
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"done. Set RATIO_INDEX_REPO={args.repo} on the deploy target.")


if __name__ == "__main__":
    main()
