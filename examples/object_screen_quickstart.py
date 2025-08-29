# examples/object_screen_quickstart.py
from __future__ import annotations

import os, sys, pathlib, argparse, logging, uuid, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# .env first (silence TF before heavy imports)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings(
    "ignore",
    message=r"Protobuf gencode version .* is exactly one major version older",
    category=UserWarning,
    module=r"google\.protobuf\.runtime_version",
)

from fait.core.logging_config import setup_logging
from fait.core.config_io import load_yaml, merge_into_dataclass
from fait.vision.pipelines.object_screen import ScreenConfig, run_object_screen


def resolve_gallery(p: str | None) -> str:
    if not p:
        return str((ROOT / "datasets" / "images" / "objects" / "raw").resolve())
    pp = pathlib.Path(p)
    return str((ROOT / p).resolve() if not pp.is_absolute() else pp.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description="Object screening (YAML-driven)")
    ap.add_argument("--config", default="configs/vision/object_screen.yaml")
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger("fait.vision.pipelines.object_screen")
    run_id = str(uuid.uuid4())
    log.info("object_screen:start", extra={"run_id": run_id})

    # ---- Load YAML preset ----
    yml = load_yaml(args.config)

    # ---- Build default cfg then merge YAML ----
    cfg = ScreenConfig(
        prompts=[],
        gallery_dir=resolve_gallery(yml.get("gallery_dir")),
        output_dir=yml.get("output_dir"),     # None -> .fait/outputs/object
        save_crops=bool(yml.get("save_crops", False)),
    )
    merge_into_dataclass(cfg, yml)  # fills strategy, gdino/yolo/detr/fusion, etc.

    # ---- Log resolved config (high level) ----
    log.info("object_screen:config", extra={
        "strategy": getattr(cfg, "strategy", "two_stage"),
        "verifier": cfg.verifier,
        "gallery_dir": cfg.gallery_dir,
        "prompts": cfg.prompts,
        "gdino": {
            "model_id": cfg.gdino.model_id,
            "box_threshold": cfg.gdino.box_threshold,
            "text_threshold": cfg.gdino.text_threshold,
            "nms_iou": cfg.gdino.nms_iou,
            "long_side": cfg.gdino.long_side,
            "box_expand": cfg.gdino.box_expand,
            "per_prompt": getattr(cfg.gdino, "per_prompt", False),
        },
        "fusion": {
            "rule": cfg.fusion.rule,
            "alpha": cfg.fusion.alpha,
            "tau_star": cfg.fusion.tau_star,
            "iou_gate": cfg.fusion.iou_gate,
            "gdino_only_default_tau": cfg.fusion.gdino_only_default_tau,
            "borderline_window": cfg.fusion.borderline_window,
            "class_thresholds": cfg.fusion.class_thresholds,
        },
        "detector_only": getattr(cfg, "detector_only", None).__dict__ if hasattr(cfg, "detector_only") else {},
    })

    # ---- Run ----
    summary = run_object_screen(cfg)

    print("\n=== OBJECT SCREEN SUMMARY ===")
    print(f"Processed     : {summary['processed']}")
    print(f"Found images  : {summary['found']}")
    print(f"Review queue  : {summary['review']}")
    print(f"Run directory : {summary['run_dir']}")
    print(f"Log (JSONL)   : {summary['log_path']}")
    print(f"Run ID        : {run_id}")


if __name__ == "__main__":
    main()
