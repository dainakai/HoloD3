#!/usr/bin/env python3
"""Run the repository-local training and benchmark reproduction plan.

This module intentionally contains no model logic.  It validates and executes
the existing Python entry points described by
``configs/training/reproduce-production.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/training/reproduce-production.json")
STAGE_ORDER = (
    "check",
    "yolo-baseline",
    "yolo-production",
    "depth-fallback",
    "depth-primary",
    "diameter",
    "evaluate",
)
STAGE_CHOICES = (*STAGE_ORDER, "all")
YOLO_BASELINE_SOURCES = ("packaged", "reproduced")
FORBIDDEN_ARGUMENTS = {"--overwrite"}


class ReproductionError(RuntimeError):
    """Raised for a configuration or preflight safety error."""


@dataclass(frozen=True)
class Step:
    stage: str
    name: str
    script: Path
    args: tuple[str, ...]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]

    @property
    def command(self) -> list[str]:
        return [sys.executable, str(self.script), *self.args]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _load_repo_path(raw: Any, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ReproductionError(f"{field} must be a non-empty repository-relative path")
    relative = Path(raw)
    if raw.startswith("~") or relative.is_absolute() or ".." in relative.parts:
        raise ReproductionError(f"{field} must be repository-relative and cannot use '~' or contain '..': {raw!r}")
    resolved = (REPO_ROOT / relative).resolve()
    if not _inside(resolved, REPO_ROOT.resolve()):
        raise ReproductionError(f"{field} escapes the repository: {raw!r}")
    return resolved


def _load_string_list(raw: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ReproductionError(f"{field} must be a list of strings")
    return tuple(raw)


def _load_path_list(raw: Any, *, field: str) -> tuple[Path, ...]:
    if not isinstance(raw, list):
        raise ReproductionError(f"{field} must be a list")
    return tuple(_load_repo_path(value, field=f"{field}[{index}]") for index, value in enumerate(raw))


def _validate_argument_safety(args: tuple[str, ...], *, step_name: str) -> None:
    forbidden = sorted(FORBIDDEN_ARGUMENTS.intersection(args))
    if forbidden:
        raise ReproductionError(f"Step {step_name!r} contains destructive arguments: {forbidden}")
    for value in args:
        if value.startswith("-"):
            continue
        if "://" in value:
            raise ReproductionError(f"Step {step_name!r} contains a URL/network argument: {value!r}")
        candidate = Path(value)
        if value.startswith("~") or candidate.is_absolute() or ".." in candidate.parts:
            raise ReproductionError(
                f"Step {step_name!r} arguments must not use absolute, home-relative, or parent-relative paths: "
                f"{value!r}"
            )


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    resolved = resolved.resolve()
    if not _inside(resolved, REPO_ROOT.resolve()):
        raise ReproductionError(f"Configuration must be inside this repository: {resolved}")
    if not resolved.is_file():
        raise ReproductionError(f"Configuration does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ReproductionError(f"Could not read configuration {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReproductionError("Configuration root must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ReproductionError("Unsupported reproduction config schema_version; expected 1")
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        raise ReproductionError("Configuration needs a stages object")
    missing = sorted(set(STAGE_ORDER) - set(stages))
    unknown = sorted(set(stages) - set(STAGE_ORDER))
    if missing or unknown:
        raise ReproductionError(f"Configuration stage mismatch: missing={missing}, unknown={unknown}")
    return payload


def _select_yolo_baseline_source(plan: list[Step], source: str) -> list[Step]:
    if source not in YOLO_BASELINE_SOURCES:
        raise ReproductionError(f"Unknown YOLO baseline source: {source}")
    if source == "packaged":
        return plan

    packaged_arg = "models/initializers/detector-baseline.pt"
    reproduced_arg = "runs/reproduction/production/train/detector_experimental_base/weights/best.pt"
    packaged_path = (REPO_ROOT / packaged_arg).resolve()
    reproduced_path = (REPO_ROOT / reproduced_arg).resolve()
    selected: list[Step] = []
    for step in plan:
        if step.name != "fine-tune-production-yolo":
            selected.append(step)
            continue
        args = list(step.args)
        try:
            weight_index = args.index("--weights") + 1
        except ValueError as exc:
            raise ReproductionError("Production YOLO step has no --weights argument") from exc
        if args[weight_index] != packaged_arg or packaged_path not in step.inputs:
            raise ReproductionError("Production YOLO baseline input does not match the canonical packaged path")
        args[weight_index] = reproduced_arg
        inputs = tuple(reproduced_path if path == packaged_path else path for path in step.inputs)
        selected.append(replace(step, args=tuple(args), inputs=inputs))
    return selected


def build_plan(
    config: dict[str, Any],
    selected_stage: str,
    *,
    yolo_baseline_source: str = "packaged",
) -> list[Step]:
    if selected_stage not in STAGE_CHOICES:
        raise ReproductionError(f"Unknown stage: {selected_stage}")
    selected = STAGE_ORDER if selected_stage == "all" else (selected_stage,)
    stages = config["stages"]
    plan: list[Step] = []
    seen_names: set[str] = set()
    runs_root = (REPO_ROOT / "runs").resolve()
    for stage in selected:
        raw_steps = stages[stage]
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ReproductionError(f"stages.{stage} must be a non-empty list")
        for index, raw_step in enumerate(raw_steps):
            field = f"stages.{stage}[{index}]"
            if not isinstance(raw_step, dict):
                raise ReproductionError(f"{field} must be an object")
            expected = {"name", "script", "args", "inputs", "outputs"}
            missing = sorted(expected - set(raw_step))
            unknown = sorted(set(raw_step) - expected)
            if missing or unknown:
                raise ReproductionError(f"{field} keys mismatch: missing={missing}, unknown={unknown}")
            name = raw_step["name"]
            if not isinstance(name, str) or not name.strip():
                raise ReproductionError(f"{field}.name must be a non-empty string")
            if name in seen_names:
                raise ReproductionError(f"Duplicate step name in selected plan: {name}")
            seen_names.add(name)
            script = _load_repo_path(raw_step["script"], field=f"{field}.script")
            if not script.is_file() or script.suffix != ".py":
                raise ReproductionError(f"Configured Python entry point does not exist: {script}")
            args = _load_string_list(raw_step["args"], field=f"{field}.args")
            _validate_argument_safety(args, step_name=name)
            inputs = _load_path_list(raw_step["inputs"], field=f"{field}.inputs")
            outputs = _load_path_list(raw_step["outputs"], field=f"{field}.outputs")
            for output in outputs:
                if output == runs_root or not _inside(output, runs_root):
                    raise ReproductionError(f"Step {name!r} may write only below runs/: {output}")
            plan.append(Step(stage, name, script, args, inputs, outputs))
    return _select_yolo_baseline_source(plan, yolo_baseline_source)


def _is_within_any(path: Path, roots: list[Path]) -> bool:
    return any(path == root or _inside(path, root) for root in roots)


def reserved_output_roots(plan: list[Step]) -> list[Path]:
    """Return minimal output roots, collapsing nested evaluator directories."""

    roots: list[Path] = []
    for output in sorted({path for step in plan for path in step.outputs}, key=lambda path: len(path.parts)):
        if not _is_within_any(output, roots):
            roots.append(output)
    return roots


def inspect_runtime(plan: list[Step]) -> list[str]:
    """Return non-mutating diagnostics used by dry-run and actual preflight."""

    issues: list[str] = []
    prior_outputs: list[Path] = []
    for step in plan:
        for path in step.inputs:
            if not path.exists() and not _is_within_any(path, prior_outputs):
                issues.append(f"missing input for {step.name}: {path.relative_to(REPO_ROOT)}")
        prior_outputs.extend(step.outputs)
    for output in reserved_output_roots(plan):
        if output.exists():
            issues.append(f"reserved output already exists: {output.relative_to(REPO_ROOT)}")
    return issues


def preflight(plan: list[Step]) -> None:
    issues = inspect_runtime(plan)
    if issues:
        joined = "\n- ".join(issues)
        raise ReproductionError(
            "Preflight refused to run. Existing outputs are never reused or overwritten, and all standalone inputs "
            f"must exist:\n- {joined}"
        )


def print_plan(plan: list[Step]) -> None:
    print(f"repository: {REPO_ROOT}")
    for index, step in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {step.stage}: {step.name}")
        print(shlex.join(step.command))


def execute(plan: list[Step]) -> int:
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("WANDB_DISABLED", "true")
    for index, step in enumerate(plan, start=1):
        print(f"\n==> [{index}/{len(plan)}] {step.name}", flush=True)
        completed = subprocess.run(step.command, cwd=REPO_ROOT, env=environment, check=False)
        if completed.returncode != 0:
            print(f"step failed ({completed.returncode}): {step.name}", file=sys.stderr)
            return completed.returncode
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the self-contained three-model training/benchmark reproduction plan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=STAGE_CHOICES, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--yolo-baseline-source",
        choices=YOLO_BASELINE_SOURCES,
        default="packaged",
        help=(
            "Use the exact packaged baseline for production parity, or feed the baseline trained earlier in the "
            "same reproduction plan into production fine-tuning"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands and safety diagnostics without executing."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        plan = build_plan(config, args.stage, yolo_baseline_source=args.yolo_baseline_source)
        print_plan(plan)
        if args.dry_run:
            issues = inspect_runtime(plan)
            if issues:
                print("\ndry-run diagnostics (execution would be refused):", file=sys.stderr)
                for issue in issues:
                    print(f"- {issue}", file=sys.stderr)
            print("\ndry-run: no commands executed")
            return 0
        preflight(plan)
        return execute(plan)
    except ReproductionError as exc:
        print(f"reproduction error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
