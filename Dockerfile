# Backend only. The frontend deploys separately (Vercel); this image serves
# the FastAPI app that frontend calls. Defaults to port 7860 (Hugging Face
# Spaces' Docker SDK convention) but reads $PORT at container start, since
# Cloud Run injects its own (typically 8080) and overriding the ENV below
# rather than baking in one platform's port keeps this image portable
# between the two without a second Dockerfile.
FROM python:3.12-slim

# Layer-cache the dependency install separately from the source, so an edit
# to src/ doesn't force reinstalling every package on the next build.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
# --no-install-project: only third-party dependencies get installed here,
# never this project's own package. Building that package needs hatchling
# to read pyproject.toml's readme = "README.md", and this Dockerfile is
# meant to be pushed to a Hugging Face Space whose OWN README.md (required
# at the space repo's root, with sdk/app_port frontmatter) is a different
# file from this project's real one, see SPACE_README.md. Skipping the
# local-package build sidesteps that clash entirely: the project already
# runs everywhere else (tests, scripts) via PYTHONPATH=src rather than an
# editable install, so the container does the same thing, not something new.
# --no-dev: pytest and ruff have no reason to ship in a serving image.
# --frozen: the lockfile is the source of truth, matching CI (uv sync --frozen
# in .github/workflows/test.yml), not whatever `uv` would resolve today.
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/

# Spaces run containers as a non-root UID. Everything the process writes at
# runtime, the pulled index, the HF model cache, the in-process sqlite files
# ensure_index() and rag.audit create on first use, needs an owner that
# actually is that UID, not root's leftover permissions from the COPY above.
RUN useradd --create-home --uid 1000 ratio && \
    mkdir -p /app/data && \
    chown -R ratio:ratio /app
USER ratio

ENV PYTHONPATH=/app/src \
    HOME=/home/ratio \
    HF_HOME=/home/ratio/.cache/huggingface \
    PORT=7860

EXPOSE 7860

# --workers 1: the retriever, reranker and cache all live in one process's
# memory (state.retriever et al. in api.py's lifespan); a second worker would
# load its own full copy of every model and the index rather than sharing
# any of it, the wrong trade on a free CPU tier's RAM budget.
#
# Shell form (not exec-form JSON) deliberately: $PORT only expands through a
# shell, and Cloud Run sets that env var at container start, after this
# image was built, so it cannot be baked in as a literal at build time.
#
# --no-sync: without it, `uv run` re-checks the project against
# pyproject.toml on every container start, not just at build time, and that
# re-check tries to build this project's own package, which needs
# hatchling to read README.md, deliberately absent from this image (see the
# --no-install-project comment above). That crashed every single container
# start in production with "Readme file does not exist", the sync at build
# time succeeded and was silently redone, and failed, at the one moment it
# actually mattered. --no-sync trusts the environment uv sync already built
# in the layer above and never touches it again.
CMD uv run --no-sync uvicorn rag.api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1
