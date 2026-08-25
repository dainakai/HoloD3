from __future__ import annotations

from holod3.config import repository_root
from scripts.run_reproduction import STAGE_ORDER, build_plan, inspect_runtime, load_config

ROOT = repository_root()
CONFIG = ROOT / "configs/training/reproduce-production.json"


def test_reproduction_plan_is_complete_semantic_and_safe() -> None:
    plan = build_plan(load_config(CONFIG), "all")
    assert STAGE_ORDER == (
        "check",
        "yolo-baseline",
        "yolo-production",
        "depth-fallback",
        "depth-primary",
        "diameter",
        "evaluate",
    )
    assert len(plan) == 11
    assert all(step.script.is_file() for step in plan)
    assert all("--overwrite" not in step.args for step in plan)
    assert all(output.is_relative_to(ROOT / "runs") and output != ROOT / "runs" for step in plan for output in step.outputs)
    serialized = "\n".join(" ".join(step.args) for step in plan)
    assert "data/downloaded/" in serialized
    assert "runs/reproduction/production" in serialized
    assert "260" + "712" not in serialized
    assert "current" + "_data" not in serialized


def test_detector_fine_tuning_and_selection_are_explicit() -> None:
    training, selection = build_plan(load_config(CONFIG), "yolo-production")
    assert "models/initializers/detector-baseline.pt" in training.args
    assert training.args[training.args.index("--epochs") + 1] == "20"
    assert training.args[training.args.index("--lr0") + 1] == "0.00002"
    assert "--freeze-bn-stats" in training.args
    assert selection.args[selection.args.index("--source") + 1].endswith("weights/epoch19.pt")
    assert selection.args[selection.args.index("--output") + 1].endswith("selected/detector.pt")


def test_full_scratch_path_feeds_reproduced_detector_base() -> None:
    plan = build_plan(load_config(CONFIG), "all", yolo_baseline_source="reproduced")
    training = next(step for step in plan if step.name == "fine-tune-production-yolo")
    expected = "runs/reproduction/production/train/detector_experimental_base/weights/best.pt"
    assert training.args[training.args.index("--weights") + 1] == expected
    assert ROOT / expected in training.inputs


def test_training_values_and_bundle_paths_are_frozen() -> None:
    config = load_config(CONFIG)
    fallback = build_plan(config, "depth-fallback")[0]
    primary = build_plan(config, "depth-primary")[0]
    diameter = build_plan(config, "diameter")[0]
    assert fallback.script.name == "train_depth_fallback.py"
    assert fallback.args[fallback.args.index("--manifest") + 1] == "data/downloaded/depth-fallback/manifest.csv"
    assert fallback.args[fallback.args.index("--pairs-per-epoch") + 1] == "8000"
    assert primary.args[primary.args.index("--manifest") + 1] == "data/downloaded/depth-primary/manifest.csv"
    assert primary.args[primary.args.index("--diameter-sampling") + 1] == "bin-balanced"
    assert diameter.args[diameter.args.index("--crops-csv") + 1] == "data/downloaded/diameter-combined/crops.csv"
    assert diameter.args[diameter.args.index("--epochs") + 1] == "70"


def test_asset_check_declares_every_required_bundle_and_checkpoint() -> None:
    check = build_plan(load_config(CONFIG), "check")[0]
    declared = {path.relative_to(ROOT).as_posix() for path in check.inputs}
    assert {
        "models/production/detector.pt",
        "models/production/depth-primary.pt",
        "models/production/depth-fallback.pt",
        "models/production/diameter.pt",
        "models/initializers/detector-architecture.pt",
        "models/initializers/detector-baseline.pt",
        "data/downloaded/detector-experimental/data.yaml",
        "data/downloaded/detector-mixed/data.yaml",
    }.issubset(declared)


def test_dry_run_diagnostics_report_missing_downloads_without_mutating() -> None:
    plan = build_plan(load_config(CONFIG), "check")
    issues = inspect_runtime(plan)
    if not (ROOT / "data/downloaded").exists():
        assert any("data/downloaded" in issue for issue in issues)
