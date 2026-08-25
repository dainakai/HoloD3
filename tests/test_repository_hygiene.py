from __future__ import annotations

import re
import subprocess
from pathlib import Path

from holod3.config import repository_root

ROOT = repository_root()
TEXT_SUFFIXES = {
    "",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_TEXT = (
    "/media/" + "dai-server/",
    "/home/" + "dai",
    "daiPC" + "atKIT",
    "M2Disk" + "2TB",
    "6TB" + "HDD",
    "HDD" + "16TB",
    "\u30dc\u30ea\u30e5\u30fc\u30e0",
    "260" + "712",
    "260" + "710_mltrain",
    "current" + "_data",
    "add" + "img",
    "wide500" + "_v2",
    "edgepad" + "_v2",
    "real4" + "_schedule",
)
JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def tracked_paths() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [ROOT / line for line in output.splitlines() if line]


def test_tracked_text_is_english_and_free_of_workstation_context() -> None:
    failures: list[str] = []
    for path in tracked_paths():
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for value in FORBIDDEN_TEXT:
            if value.lower() in text.lower() or value.lower() in relative.lower():
                failures.append(f"{relative}: forbidden context {value!r}")
        if JAPANESE.search(text):
            failures.append(f"{relative}: non-English Japanese text")
    assert not failures, "\n".join(failures[:100])


def test_git_contains_only_the_small_demo_data() -> None:
    data_paths = [path for path in tracked_paths() if path.is_relative_to(ROOT / "data") and path.is_file()]
    assert len(data_paths) <= 30
    assert sum(path.stat().st_size for path in data_paths) < 20 * 1024 * 1024
    assert all("data/downloaded/" not in path.relative_to(ROOT).as_posix() for path in data_paths)
    assert not any(path.suffix == ".pt" for path in tracked_paths())


def test_session_memory_is_not_tracked_inside_repository() -> None:
    assert all(not path.is_relative_to(ROOT / "session") for path in tracked_paths())
    assert "session/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_local_markdown_links_resolve() -> None:
    missing: list[str] = []
    for path in tracked_paths():
        if path.suffix.lower() != ".md" or not path.is_file():
            continue
        for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            clean = target.strip("<>").split("#", 1)[0]
            if not clean or clean.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / clean).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not missing, "\n".join(missing)
