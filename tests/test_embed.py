import json

import numpy as np
import pytest

from rag.embed import build_index, load_index, load_model, token_length
from rag.chunk import Chunk


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
