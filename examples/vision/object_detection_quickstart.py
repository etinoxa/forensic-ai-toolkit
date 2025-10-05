# examples/vision/object_detection_quickstart.py
"""
Object Detection Quickstart - Uses config.yaml for all settings.
Run-specific overrides can be passed as CLI arguments.
"""

import os
import sys
import pathlib
import argparse
import logging
import uuid
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Silence warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings(
    "ignore",
    message=r"Protobuf gencode version .* is exactly one major version older",
    category=UserWarning,
    module=r"google\.protobuf\.runtime_version",
)

from fait.core.logging_config import setup_logging
from fait.core.app_config import get_app_config
from fait.vision.pipelines.object_detection_pipeline import (
    ScreenConfig,
    FusionConfig,
    DetectorOnlyConfig,
    run_object_detection
)
from fait.vision.object_detection.models.grounding_dino import GDINOConfig
from fait.vision.object_detection.models.yolo import YoloConfig
from fait.vision.object_detection.models.deformable_detr import DefDETRConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Object Detection Quickstart")
    parser.add_argument("--gallery", help="Gallery directory (overrides default)")
    parser.add_argument("--output", help="Output directory (overrides default)")
    parser.add_argument("--prompts", nargs="+", help="Detection prompts (e.g., 'knife' 'gun')")
    parser.add_argument("--strategy", choices=["gdino_only", "detector_only", "two_stage"],
                       help="Override strategy")
    parser.add_argument("--verifier", choices=["yolo", "deformable_detr", "none"],
                       help="Override verifier")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped detections")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("fait.vision.pipelines.object_screen")
    run_id = str(uuid.uuid4())
    log.info("object_screen:start", extra={"run_id": run_id})

    # Load application config
    app_config = get_app_config()
    obj_config = app_config.vision.object_detection

    # Resolve paths
    gallery_dir = args.gallery or str(ROOT / "datasets" / "images" / "objects" / "raw")
    output_dir = args.output

    # Config with CLI overrides
    prompts = args.prompts or ["person with weapon", "gun", "knife"]
    strategy = args.strategy or obj_config.strategy
    verifier = args.verifier or obj_config.verifier
    save_crops = args.save_crops

    # Build model configs from app_config
    cfg = ScreenConfig(
        prompts=prompts,
        gallery_dir=gallery_dir,
        output_dir=output_dir,
        save_crops=save_crops,
        strategy=strategy,
        verifier=verifier,
        gdino=GDINOConfig(
            model_id=obj_config.gdino_model_id,
            box_threshold=obj_config.gdino_box_threshold,
            text_threshold=obj_config.gdino_text_threshold,
            nms_iou=obj_config.gdino_nms_iou,
            long_side=obj_config.gdino_long_side,
            box_expand=obj_config.gdino_box_expand,
        ),
        yolo=YoloConfig(
            model_id=obj_config.yolo_model_id,
            score_threshold=obj_config.yolo_score_threshold,
            nms_iou=obj_config.yolo_nms_iou,
            imgsz=obj_config.yolo_imgsz,
        ),
        detr=DefDETRConfig(
            model_id=obj_config.detr_model_id,
            score_threshold=obj_config.detr_score_threshold,
            nms_iou=obj_config.detr_nms_iou,
        ),
        fusion=FusionConfig(
            rule=obj_config.fusion_rule,
            alpha=obj_config.fusion_alpha,
            tau_star=obj_config.fusion_tau_star,
            iou_gate=obj_config.fusion_iou_gate,
            borderline_window=obj_config.fusion_borderline_window,
            gdino_only_default_tau=obj_config.gdino_only_default_tau,
        ),
        detector_only=DetectorOnlyConfig(
            default_tau=obj_config.detector_only_default_tau,
        ),
    )

    print("=== Object Detection Configuration ===")
    print(f"Strategy      : {cfg.strategy}")
    print(f"Verifier      : {cfg.verifier}")
    print(f"Prompts       : {cfg.prompts}")
    print(f"Gallery       : {cfg.gallery_dir}")
    print(f"Output        : {cfg.output_dir or '(auto)'}")
    print(f"Save crops    : {cfg.save_crops}")
    print()

    log.info("object_screen:config", extra={
        "strategy": cfg.strategy,
        "verifier": cfg.verifier,
        "gallery_dir": cfg.gallery_dir,
        "prompts": cfg.prompts,
    })

    # Run
    summary = run_object_detection(cfg)

    print("\n=== OBJECT SCREEN SUMMARY ===")
    print(f"Processed     : {summary['processed']}")
    print(f"Found images  : {summary['found']}")
    print(f"Review queue  : {summary['review']}")
    print(f"Run directory : {summary['run_dir']}")
    print(f"Log (JSONL)   : {summary.get('log_path')}")
    print(f"Report        : {summary.get('report_path')}")
    print(f"Run ID        : {run_id}")


if __name__ == "__main__":
    main()