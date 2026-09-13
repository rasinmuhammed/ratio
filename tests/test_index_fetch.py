from pathlib import Path

import pytest

from rag.index_fetch import ensure_index, index_is_complete


def _make_complete_index(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "ids.json").write_text("[]")
    (path / "meta.json").write_text("{}")
    (path / "payloads.jsonl").write_text("")
    (path / "vectors-0000.npy").write_bytes(b"")


def test_missing_directory_is_incomplete(tmp_path):
    assert not index_is_complete(tmp_path / "nope")


def test_directory_missing_payloads_is_incomplete(tmp_path):
    """The exact real bug this check exists to catch: data/index-full has
    vectors but no payloads.jsonl and still cannot load, so a directory
    merely existing must not read as a usable index."""
    index_dir = tmp_path / "index-full"
    index_dir.mkdir()
    (index_dir / "ids.json").write_text("[]")
    (index_dir / "meta.json").write_text("{}")
    (index_dir / "vectors-0000.npy").write_bytes(b"")
    assert not index_is_complete(index_dir)


def test_directory_with_all_required_files_is_complete(tmp_path):
    index_dir = tmp_path / "index"
    _make_complete_index(index_dir)
    assert index_is_complete(index_dir)


def test_ensure_index_is_a_noop_when_already_complete(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    _make_complete_index(index_dir)

    def exploding_download(*a, **k):
        raise AssertionError("should not download when already complete")

    monkeypatch.setattr("huggingface_hub.snapshot_download", exploding_download)
    ensure_index(index_dir, repo="someone/ratio-index")  # must not raise


def test_ensure_index_is_a_noop_when_no_repo_given(tmp_path):
    index_dir = tmp_path / "missing-index"
    ensure_index(index_dir, repo=None)  # must not raise, must not create anything
    assert not index_dir.exists()


def test_ensure_index_downloads_when_incomplete_and_repo_given(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    calls = []

    def fake_download(repo_id, repo_type, local_dir):
        calls.append((repo_id, repo_type, local_dir))
        _make_complete_index(Path(local_dir))

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    ensure_index(index_dir, repo="someone/ratio-index")

    assert calls == [("someone/ratio-index", "dataset", str(index_dir))]
    assert index_is_complete(index_dir)


def test_ensure_index_raises_if_download_is_still_incomplete(tmp_path, monkeypatch):
    """A download that succeeds but leaves a partial index (wrong repo,
    partial upload) must fail loudly here, not three frames deeper inside
    HybridRetriever with no mention a download even happened."""
    index_dir = tmp_path / "index"

    def fake_download(repo_id, repo_type, local_dir):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "ids.json").write_text("[]")  # missing the rest

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)

    with pytest.raises(RuntimeError, match="still missing"):
        ensure_index(index_dir, repo="someone/ratio-index")
