import json

import numpy as np
import pytest

from rag.chunk import Chunk
from rag.embed import build_index, load_index, load_model, token_length


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split()  # word count stands in for token count


class FakeModel:
    """Enough of SentenceTransformer's surface for build_index() to run
    without loading real weights, so the checkpoint/resume mechanics (the
    part actually under test here) run in the normal fast suite rather than
    behind -m model."""

    def __init__(self, dim: int = 3, max_seq_length: int = 1000):
        self.max_seq_length = max_seq_length
        self.tokenizer = FakeTokenizer()
        self._dim = dim

    def get_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, batch_size=64, normalize_embeddings=True,
               show_progress_bar=False):
        return np.zeros((len(texts), self._dim), dtype=np.float32)


def _write_index(path, ids, vectors, dim=3):
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors-0000.npy", np.array(vectors, dtype=np.float32))
    (path / "ids.json").write_text(json.dumps(ids))
    (path / "meta.json").write_text(json.dumps({"model": "fake", "dim": dim}))


def test_load_index_rejects_id_vector_mismatch(tmp_path):
    """Row order is an invisible contract. Load time is the cheapest place
    to catch a violation, so a mismatch must raise rather than silently
    misalign every chunk against the wrong vector."""
    idx = tmp_path / "index"
    _write_index(idx, ids=["a", "b", "c"], vectors=[[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="id/vector mismatch"):
        load_index(idx)


def test_load_index_roundtrip(tmp_path):
    idx = tmp_path / "index"
    _write_index(idx, ids=["a", "b"], vectors=[[1, 0, 0], [0, 1, 0]])
    vectors, ids, meta = load_index(idx)
    assert ids == ["a", "b"]
    assert vectors.shape == (2, 3)
    assert meta["model"] == "fake"


def test_load_index_concatenates_shards_in_order(tmp_path):
    """Shards are written for crash-resumability. They must reassemble in
    filename order, otherwise ids no longer line up with vectors."""
    idx = tmp_path / "index"
    idx.mkdir(parents=True)
    np.save(idx / "vectors-0000.npy", np.array([[1, 0]], dtype=np.float32))
    np.save(idx / "vectors-0001.npy", np.array([[0, 1]], dtype=np.float32))
    (idx / "ids.json").write_text(json.dumps(["first", "second"]))
    (idx / "meta.json").write_text(json.dumps({"model": "fake", "dim": 2}))

    vectors, ids, _ = load_index(idx)
    assert ids == ["first", "second"]
    assert vectors[0].tolist() == [1.0, 0.0]
    assert vectors[1].tolist() == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Model-backed. Slow and needs the weights, so opt in with:
#     uv run pytest -m model
# ---------------------------------------------------------------------------


@pytest.mark.model
def test_token_length_counts_tokens_not_characters():
    f = token_length(load_model())
    text = "the petition is accordingly dismissed"
    assert f(text) < len(text)
    assert f(text) > 0


@pytest.mark.model
def test_build_index_writes_normalized_vectors(tmp_path):
    """Vectors are normalized at write time so retrieval is a single dot
    product rather than a dot product plus two norms on every query."""
    chunks = [
        Chunk(id=f"d#{i}", doc_id="d", text=f"paragraph {i} of the judgment",
              index=i, metadata={})
        for i in range(5)
    ]
    meta = build_index(chunks, out_dir=tmp_path / "index")

    assert meta["count"] == 5
    assert meta["normalized"] is True

    vectors, ids, _ = load_index(tmp_path / "index")
    assert ids == [c.id for c in chunks]
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.model
def test_build_index_counts_truncation(tmp_path):
    """Silent truncation at the model limit is the failure this whole
    pipeline is most exposed to. It must be counted, not discovered later."""
    model = load_model()
    long_text = "word " * (model.max_seq_length * 2)
    chunks = [Chunk(id="d#0", doc_id="d", text=long_text, index=0, metadata={})]
    meta = build_index(chunks, out_dir=tmp_path / "index")
    assert meta["truncated"] == 1


# ---------------------------------------------------------------------------
# Resumability: a crash partway through a corpus-wide build must not force
# re-embedding chunks whose vectors are already safely on disk.
# ---------------------------------------------------------------------------


def _chunks(n):
    return [Chunk(id=f"d#{i}", doc_id="d", text=f"chunk number {i}",
                  index=i, metadata={}) for i in range(n)]


class Boom(Exception):
    pass


def _crashing_stream(items, crash_at):
    for i, item in enumerate(items):
        if i == crash_at:
            raise Boom()
        yield item


def test_resumed_run_never_reembeds_a_completed_shard():
    """The whole point: chunks already backed by a flushed shard must be
    skipped, not passed to encode() again, on the resumed call."""

    class CountingModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.encoded_texts: list[str] = []

        def encode(self, texts, **kw):
            self.encoded_texts.extend(texts)
            return super().encode(texts, **kw)

    def run(tmp_path):
        idx = tmp_path / "idx"
        chunks = _chunks(10)
        model = CountingModel()
        with pytest.raises(Boom):
            build_index(_crashing_stream(chunks, crash_at=7), out_dir=idx,
                        model=model, shard_size=3)
        # Shards 0 and 1 (chunks 0-2, 3-5) flush and checkpoint before the
        # crash at index 7; chunk 6 was consumed into the third, never
        # flushed shard buffer and must not count as done.
        checkpoint = json.loads((idx / ".checkpoint.json").read_text())
        assert checkpoint["chunks_written"] == 6

        model2 = CountingModel()
        meta = build_index(iter(chunks), out_dir=idx, model=model2, shard_size=3)
        assert meta["count"] == 10
        # Resumed run only ever encodes chunk 6 onward; chunks 0-5 stay
        # embedded from the first attempt and are never handed to encode()
        # a second time.
        assert "chunk number 0" not in model2.encoded_texts
        assert "chunk number 5" not in model2.encoded_texts
        assert "chunk number 6" in model2.encoded_texts
        assert "chunk number 9" in model2.encoded_texts

        vectors, ids, _ = load_index(idx)
        assert ids == [c.id for c in chunks]
        assert vectors.shape[0] == 10

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        run(Path(d))


def test_payloads_jsonl_orphaned_tail_is_discarded_on_resume(tmp_path):
    """payloads.jsonl is written per-chunk, vectors per-shard, so a crash can
    leave payload lines with no backing vectors. Those must be dropped, not
    trusted, or the resumed run's ids.json would list chunks that were never
    actually embedded into any shard."""
    idx = tmp_path / "idx"
    chunks = _chunks(10)
    model = FakeModel()
    with pytest.raises(Boom):
        build_index(_crashing_stream(chunks, crash_at=7), out_dir=idx,
                    model=model, shard_size=3)

    lines_after_crash = (idx / "payloads.jsonl").read_text().splitlines()
    assert len(lines_after_crash) == 7  # chunks 0-6 written, only 0-5 flushed

    build_index(iter(chunks), out_dir=idx, model=FakeModel(), shard_size=3)
    ids = json.loads((idx / "ids.json").read_text())
    assert ids == [c.id for c in chunks]
    assert len(set(ids)) == 10  # no chunk duplicated across the two runs


def test_a_completed_run_leaves_no_checkpoint_behind(tmp_path):
    idx = tmp_path / "idx"
    build_index(iter(_chunks(5)), out_dir=idx, model=FakeModel(), shard_size=3)
    assert not (idx / ".checkpoint.json").exists()


def test_fresh_run_with_no_checkpoint_ignores_stale_payloads(tmp_path):
    """A directory with no checkpoint is treated as a fresh start even if it
    happens to hold old files (a previous unrelated build, say): the writer
    opens in 'w' mode, not 'a', exactly as before this feature existed."""
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "payloads.jsonl").write_text('{"id": "stale#0"}\n')

    meta = build_index(iter(_chunks(3)), out_dir=idx, model=FakeModel())
    assert meta["count"] == 3
    ids = json.loads((idx / "ids.json").read_text())
    assert ids == ["d#0", "d#1", "d#2"]
