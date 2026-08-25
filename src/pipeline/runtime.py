"""Small, shared runtime helpers for the executable inference pipeline."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from holod3.artifacts import sha256_file


PIPELINE_DIR = Path(__file__).resolve().parent
SRC_DIR = PIPELINE_DIR.parent
DETECTION_DIR = SRC_DIR / "detection"


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())


def repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (REPO_ROOT / expanded).resolve()


def portable_path(path: str | Path, *, run_dir: Path | None = None, acquisition_dir: Path | None = None) -> str:
    """Represent paths without embedding workstation roots in shareable output."""

    resolved = Path(path).expanduser().resolve()
    for prefix, root in (("run", run_dir), ("acquisition", acquisition_dir), ("repository", REPO_ROOT)):
        if root is None:
            continue
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
            return f"{prefix}:{relative or '.'}"
        except ValueError:
            pass
    return f"external:{resolved.name}"


def artifact_fingerprint(path: Path, *, run_dir: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "id": portable_path(resolved, run_dir=run_dir),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _portable_value(value: Any, *, run_dir: Path, acquisition_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _portable_value(item, run_dir=run_dir, acquisition_dir=acquisition_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item, run_dir=run_dir, acquisition_dir=acquisition_dir) for item in value]
    if isinstance(value, str) and Path(value).expanduser().is_absolute():
        return portable_path(value, run_dir=run_dir, acquisition_dir=acquisition_dir)
    return value


def sanitize_json_paths(path: Path, *, run_dir: Path, acquisition_dir: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sanitized = _portable_value(payload, run_dir=run_dir, acquisition_dir=acquisition_dir)
    path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def display_command(command: list[str]) -> list[str]:
    displayed: list[str] = []
    for token in command:
        if token == command[0]:
            displayed.append("python")
        elif Path(token).expanduser().is_absolute():
            displayed.append(portable_path(token))
        else:
            displayed.append(token)
    return displayed


def run_step(name: str, command: list[str], steps: list[dict[str, object]]) -> None:
    started = time.perf_counter()
    shown_command = display_command(command)
    print(json.dumps({"event": "step_start", "step": name, "command": shown_command}), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    elapsed = time.perf_counter() - started
    steps.append({"step": name, "elapsed_sec": elapsed, "command": shown_command})
    print(json.dumps({"event": "step_done", "step": name, "elapsed_sec": round(elapsed, 3)}), flush=True)
