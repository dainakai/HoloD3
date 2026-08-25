"""Download and safely extract versioned HoloD3 dataset bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import zstandard

from holod3.config import repository_root

DataScope = Literal["training", "evaluation", "all"]


@dataclass(frozen=True)
class DatasetBundleStatus:
    bundle_id: str
    destination: Path
    downloaded: bool
    extracted: bool
    ok: bool
    archive_sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_manifest(
    path: str | Path | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    asset_root = (root or repository_root()).resolve()
    resolved = Path(path or "data/remote_manifest.json").expanduser()
    if not resolved.is_absolute():
        resolved = asset_root / resolved
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("provider") != "huggingface"
        or not isinstance(payload.get("bundles"), list)
    ):
        raise ValueError(f"Unsupported dataset manifest: {resolved}")
    return payload


def _download_hugging_face_file(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    local_dir: Path,
    force_download: bool,
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=local_dir,
            force_download=force_download,
        )
    )


def _validate_member(member: tarfile.TarInfo, expected_root: str) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != expected_root:
        raise ValueError(f"Unsafe or unexpected archive member: {member.name}")
    if member.issym() or member.islnk() or member.isdev():
        raise ValueError(f"Dataset bundles may not contain links or device files: {member.name}")


def _extract_tar_zst(archive: Path, staging_root: Path, expected_root: str) -> Path:
    staging_root.mkdir(parents=True, exist_ok=False)
    try:
        with archive.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        _validate_member(member, expected_root)
                        tar.extract(member, path=staging_root, filter="data")
        extracted = staging_root / expected_root
        if not extracted.is_dir():
            raise ValueError(f"Archive did not create its declared root directory: {expected_root}")
        return extracted
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _marker_matches(destination: Path, *, revision: str, archive_sha256: str) -> bool:
    marker = destination / ".holod3_bundle.json"
    if not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("revision") == revision and value.get("archive_sha256") == archive_sha256


def _replace_installed_directory(new_directory: Path, destination: Path) -> None:
    """Install a staged directory while preserving the prior install on failure."""

    backup = destination.parent / f".{destination.name}-previous-{uuid.uuid4().hex}"
    had_destination = destination.exists()
    if had_destination:
        destination.replace(backup)
    try:
        new_directory.replace(destination)
    except Exception:
        if had_destination and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def fetch_dataset_bundles(
    *,
    root: Path | None = None,
    manifest_path: str | Path | None = None,
    scope: DataScope = "training",
    bundle_ids: list[str] | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    force: bool = False,
    check_hash: bool = True,
) -> list[DatasetBundleStatus]:
    """Download selected archives, verify them, and atomically install each bundle."""

    if scope not in {"training", "evaluation", "all"}:
        raise ValueError(f"Unsupported data scope: {scope}")
    if not check_hash:
        raise ValueError("Dataset installation always requires SHA-256 verification; size-only extraction is unsafe.")
    asset_root = (root or repository_root()).resolve()
    manifest = load_dataset_manifest(manifest_path, root=asset_root)
    effective_repo_id = repo_id or str(manifest["repo_id"])
    effective_revision = revision or str(manifest["revision"])
    selected_ids = set(bundle_ids or [])
    known_ids = {str(bundle["id"]) for bundle in manifest["bundles"]}
    unknown = sorted(selected_ids - known_ids)
    if unknown:
        raise KeyError(f"Unknown dataset bundle ids: {unknown}")

    download_root = asset_root / "data" / ".downloads"
    install_root = asset_root / "data" / "downloaded"
    staging_parent = asset_root / "data" / ".staging"
    download_root.mkdir(parents=True, exist_ok=True)
    install_root.mkdir(parents=True, exist_ok=True)
    staging_parent.mkdir(parents=True, exist_ok=True)
    statuses: list[DatasetBundleStatus] = []

    for bundle in manifest["bundles"]:
        bundle_id = str(bundle["id"])
        scopes = {str(value) for value in bundle.get("scopes", [])}
        if selected_ids:
            if bundle_id not in selected_ids:
                continue
        elif scope != "all" and scope not in scopes:
            continue

        archive_name = str(bundle["archive"])
        archive = (download_root / Path(archive_name)).resolve()
        if not archive.is_relative_to(download_root.resolve()):
            raise ValueError(f"Archive destination escapes data/.downloads: {archive_name}")
        expected_size = int(bundle["bytes"])
        expected_hash = str(bundle["sha256"])
        archive_ok = archive.is_file() and archive.stat().st_size == expected_size
        if archive_ok and check_hash:
            archive_ok = sha256_file(archive) == expected_hash
        downloaded = False
        if force or not archive_ok:
            try:
                downloaded_path = _download_hugging_face_file(
                    repo_id=effective_repo_id,
                    revision=effective_revision,
                    filename=archive_name,
                    local_dir=download_root,
                    force_download=force or archive.exists(),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download dataset bundle {bundle_id!r} from the private Hugging Face repository "
                    f"{effective_repo_id}@{effective_revision}. Request access and run `uv run hf auth login`, "
                    "or set HF_TOKEN in the environment."
                ) from exc
            archive = downloaded_path.resolve()
            downloaded = True
            if not archive.is_relative_to(download_root.resolve()):
                raise RuntimeError(f"Hugging Face client wrote outside data/.downloads: {archive}")
        if not archive.is_file() or archive.stat().st_size != expected_size:
            raise RuntimeError(f"Dataset archive size mismatch: {bundle_id}")
        actual_hash = sha256_file(archive)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Dataset archive checksum mismatch: {bundle_id}")

        destination = install_root / bundle_id
        extracted = False
        if force or not _marker_matches(destination, revision=effective_revision, archive_sha256=expected_hash):
            staging_root = staging_parent / f"{bundle_id}-{uuid.uuid4().hex}"
            extracted_root = _extract_tar_zst(archive, staging_root, bundle_id)
            required = [str(value) for value in bundle.get("required_paths", [])]
            missing = [value for value in required if not (extracted_root / value).exists()]
            if missing:
                shutil.rmtree(staging_root, ignore_errors=True)
                raise RuntimeError(f"Dataset bundle {bundle_id} is incomplete: {missing}")
            marker = {
                "schema_version": 1,
                "bundle_id": bundle_id,
                "repo_id": effective_repo_id,
                "revision": effective_revision,
                "archive_sha256": expected_hash,
            }
            (extracted_root / ".holod3_bundle.json").write_text(
                json.dumps(marker, indent=2) + "\n",
                encoding="utf-8",
            )
            _replace_installed_directory(extracted_root, destination)
            shutil.rmtree(staging_root, ignore_errors=True)
            extracted = True
        required = [str(value) for value in bundle.get("required_paths", [])]
        ok = _marker_matches(destination, revision=effective_revision, archive_sha256=expected_hash) and all(
            (destination / value).exists() for value in required
        )
        if not ok:
            raise RuntimeError(f"Installed dataset bundle failed verification: {bundle_id}")
        statuses.append(
            DatasetBundleStatus(
                bundle_id=bundle_id,
                destination=destination,
                downloaded=downloaded,
                extracted=extracted,
                ok=ok,
                archive_sha256=expected_hash,
            )
        )
    if not statuses:
        raise ValueError("No dataset bundles matched the requested scope or ids.")
    return statuses
