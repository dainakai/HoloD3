"""Model and training-data integrity checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from holod3.config import repository_root, resolve_asset_path


@dataclass(frozen=True)
class ArtifactStatus:
    artifact_id: str
    path: Path
    exists: bool
    size_matches: bool
    hash_matches: bool | None
    expected_sha256: str
    actual_sha256: str | None

    @property
    def ok(self) -> bool:
        return self.exists and self.size_matches and self.hash_matches is not False


ModelScope = Literal["production", "reproduction", "all"]


@dataclass(frozen=True)
class RemoteArtifactStatus:
    path: Path
    downloaded: bool
    size_matches: bool
    hash_matches: bool | None
    expected_sha256: str
    actual_sha256: str | None

    @property
    def ok(self) -> bool:
        return self.path.is_file() and self.size_matches and self.hash_matches is not False


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_production_manifest(path: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_asset_path(path or "models/production/manifest.json")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("artifacts"), list):
        raise ValueError(f"Unsupported production manifest: {resolved}")
    return payload


def load_remote_manifest(
    path: str | Path | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    asset_root = (root or repository_root()).resolve()
    resolved = Path(path or "models/remote_manifest.json").expanduser()
    if not resolved.is_absolute():
        resolved = asset_root / resolved
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("provider") != "huggingface"
        or not isinstance(payload.get("artifacts"), list)
    ):
        raise ValueError(f"Unsupported remote model manifest: {resolved}")
    return payload


def _download_hugging_face_file(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    filename: str,
    local_dir: Path,
    force_download: bool,
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            filename=filename,
            local_dir=local_dir,
            force_download=force_download,
        )
    )


def fetch_model_artifacts(
    *,
    root: Path | None = None,
    manifest_path: str | Path | None = None,
    scope: ModelScope = "production",
    repo_id: str | None = None,
    revision: str | None = None,
    force: bool = False,
    check_hash: bool = True,
) -> list[RemoteArtifactStatus]:
    """Download a model scope from the configured private Hugging Face repository.

    Valid local files are never downloaded again unless ``force`` is true. Every
    selected destination is constrained to the repository root and checked by
    byte size and, by default, SHA-256 after download.
    """

    if scope not in {"production", "reproduction", "all"}:
        raise ValueError(f"Unsupported model scope: {scope}")
    if not check_hash:
        raise ValueError("Model downloads always require SHA-256 verification; omit the size-only request.")
    asset_root = (root or repository_root()).resolve()
    manifest = load_remote_manifest(manifest_path, root=asset_root)
    effective_repo_id = repo_id or str(manifest["repo_id"])
    effective_revision = revision or str(manifest["revision"])
    repo_type = str(manifest.get("repo_type", "model"))
    statuses: list[RemoteArtifactStatus] = []

    for artifact in manifest["artifacts"]:
        scopes = {str(value) for value in artifact.get("scopes", [])}
        if scope != "all" and scope not in scopes:
            continue
        relative = Path(str(artifact["path"]))
        destination = (asset_root / relative).resolve()
        if relative.is_absolute() or not destination.is_relative_to(asset_root):
            raise ValueError(f"Remote artifact path escapes the repository root: {relative}")

        expected_size = int(artifact["bytes"])
        expected_hash = str(artifact["sha256"])
        exists = destination.is_file()
        size_matches = exists and destination.stat().st_size == expected_size
        actual_hash = sha256_file(destination) if exists and size_matches and check_hash else None
        hash_matches = bool(exists and size_matches and actual_hash == expected_hash)
        downloaded = False

        if force or not hash_matches:
            try:
                downloaded_path = _download_hugging_face_file(
                    repo_id=effective_repo_id,
                    repo_type=repo_type,
                    revision=effective_revision,
                    filename=relative.as_posix(),
                    local_dir=asset_root,
                    force_download=force or exists,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download {relative} from the private Hugging Face repository "
                    f"{effective_repo_id}@{effective_revision}. Request access and run `uv run hf auth login`, "
                    "or set HF_TOKEN in the environment."
                ) from exc
            downloaded = True
            destination = downloaded_path.resolve()
            if not destination.is_relative_to(asset_root):
                raise RuntimeError(f"Hugging Face client wrote outside the repository root: {destination}")
            exists = destination.is_file()
            size_matches = exists and destination.stat().st_size == expected_size
            actual_hash = sha256_file(destination) if exists and check_hash else None
            hash_matches = bool(exists and size_matches and actual_hash == expected_hash)

        status = RemoteArtifactStatus(
            path=destination,
            downloaded=downloaded,
            size_matches=size_matches,
            hash_matches=hash_matches,
            expected_sha256=expected_hash,
            actual_sha256=actual_hash,
        )
        if not status.ok:
            raise RuntimeError(
                f"Downloaded artifact failed integrity verification: {relative} "
                f"(size_matches={size_matches}, hash_matches={hash_matches})"
            )
        statuses.append(status)
    if not statuses:
        raise ValueError(f"No artifacts are assigned to model scope: {scope}")
    return statuses


def verify_production_artifacts(
    *,
    root: Path | None = None,
    artifact_ids: Iterable[str] | None = None,
    check_hash: bool = True,
) -> list[ArtifactStatus]:
    asset_root = (root or repository_root()).resolve()
    manifest = load_production_manifest(asset_root / "models" / "production" / "manifest.json")
    selected = set(artifact_ids) if artifact_ids is not None else None
    statuses: list[ArtifactStatus] = []
    for artifact in manifest["artifacts"]:
        artifact_id = str(artifact["id"])
        if selected is not None and artifact_id not in selected:
            continue
        path = resolve_asset_path(str(artifact["path"]), asset_root)
        exists = path.is_file()
        size_matches = exists and path.stat().st_size == int(artifact["bytes"])
        actual = sha256_file(path) if exists and check_hash else None
        hash_matches = bool(exists and actual == artifact["sha256"]) if check_hash else None
        statuses.append(
            ArtifactStatus(
                artifact_id=artifact_id,
                path=path,
                exists=exists,
                size_matches=size_matches,
                hash_matches=hash_matches,
                expected_sha256=str(artifact["sha256"]),
                actual_sha256=actual,
            )
        )
    if selected is not None:
        found = {status.artifact_id for status in statuses}
        unknown = sorted(selected - found)
        if unknown:
            raise KeyError(f"Unknown production artifact ids: {unknown}")
    return statuses
