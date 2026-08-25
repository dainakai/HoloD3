from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import zstandard

from holod3.datasets import _extract_tar_zst, fetch_dataset_bundles


def make_archive(path: Path, members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    compressed = zstandard.ZstdCompressor(level=1).compress(buffer.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return compressed


def write_manifest(root: Path, archive: bytes, *, digest: str | None = None) -> Path:
    manifest = root / "data/remote_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "huggingface",
                "repo_id": "owner/private-data",
                "repo_type": "dataset",
                "revision": "pinned-commit",
                "private": True,
                "bundles": [
                    {
                        "id": "tiny",
                        "archive": "bundles/tiny.tar.zst",
                        "bytes": len(archive),
                        "sha256": digest or hashlib.sha256(archive).hexdigest(),
                        "scopes": ["training", "all"],
                        "required_paths": ["manifest.csv", "images"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_fetch_verifies_extracts_atomically_and_reuses_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source/tiny.tar.zst"
    archive = make_archive(
        source,
        {
            "tiny/manifest.csv": b"id,value\n1,2\n",
            "tiny/images/sample.png": b"fixture",
        },
    )
    write_manifest(tmp_path, archive)
    calls: list[str] = []

    def fake_download(**kwargs: object) -> Path:
        destination = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive)
        calls.append(str(kwargs["filename"]))
        return destination

    monkeypatch.setattr("holod3.datasets._download_hugging_face_file", fake_download)
    first = fetch_dataset_bundles(root=tmp_path, scope="training")
    assert calls == ["bundles/tiny.tar.zst"]
    assert first[0].downloaded and first[0].extracted and first[0].ok
    destination = tmp_path / "data/downloaded/tiny"
    assert (destination / "manifest.csv").is_file()
    marker = json.loads((destination / ".holod3_bundle.json").read_text(encoding="utf-8"))
    assert marker["revision"] == "pinned-commit"

    calls.clear()
    second = fetch_dataset_bundles(root=tmp_path, scope="training")
    assert calls == []
    assert not second[0].downloaded and not second[0].extracted and second[0].ok

    replaced = fetch_dataset_bundles(root=tmp_path, scope="training", force=True)
    assert replaced[0].downloaded and replaced[0].extracted and (destination / "manifest.csv").is_file()
    assert not list(destination.parent.glob(".tiny-previous-*"))


def test_dataset_fetch_refuses_size_only_installation(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "source/tiny.tar.zst", {"tiny/manifest.csv": b"x", "tiny/images/x": b"x"})
    write_manifest(tmp_path, archive)
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_dataset_bundles(root=tmp_path, check_hash=False)


def test_archive_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.zst"
    make_archive(archive, {"tiny/manifest.csv": b"ok", "../escape.txt": b"no"})
    with pytest.raises(ValueError, match="Unsafe or unexpected"):
        _extract_tar_zst(archive, tmp_path / "staging", "tiny")
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "staging").exists()


def test_checksum_mismatch_is_rejected_before_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source/tiny.tar.zst"
    archive = make_archive(source, {"tiny/manifest.csv": b"id\n1\n", "tiny/images/x": b"x"})
    write_manifest(tmp_path, archive, digest="0" * 64)

    def fake_download(**kwargs: object) -> Path:
        destination = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive)
        return destination

    monkeypatch.setattr("holod3.datasets._download_hugging_face_file", fake_download)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        fetch_dataset_bundles(root=tmp_path, scope="training")
    assert not (tmp_path / "data/downloaded/tiny").exists()


def test_unknown_bundle_id_is_reported_without_download(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "source/tiny.tar.zst", {"tiny/manifest.csv": b"x", "tiny/images/x": b"x"})
    write_manifest(tmp_path, archive)
    with pytest.raises(KeyError, match="Unknown dataset bundle ids"):
        fetch_dataset_bundles(root=tmp_path, bundle_ids=["missing"])
