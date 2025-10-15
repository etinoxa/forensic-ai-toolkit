# examples/vision/ocr_quickstart.py
"""
OCR Quickstart - Uses config.yaml for all settings.
Run-specific overrides can be passed as CLI arguments.
"""

import sys
import pathlib
import argparse

script_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parents[1] / "src"))

from fait.core.app_config import get_app_config
from fait.core.logging_config import setup_logging
from fait.vision.ocr.models.config import OcrConfig, FusionCfg, EngineCfg
from fait.vision.pipelines.ocr_pipeline import run_ocr


def build_engine_configs(app_config):
    """Build engine configs dict from app_config."""
    engines = {}
    ocr_cfg = app_config.vision.ocr

    for engine_name in ocr_cfg.engine_order:
        if engine_name == "paddle":
            engines["paddle"] = EngineCfg(
                enabled=True,
                lang=ocr_cfg.paddle_lang,
                det_db_thresh=ocr_cfg.paddle_det_db_thresh,
                det_db_box_thresh=ocr_cfg.paddle_det_db_box_thresh,
                det_db_unclip_ratio=ocr_cfg.paddle_det_db_unclip_ratio,
            )
        elif engine_name == "trocr":
            engines["trocr"] = EngineCfg(
                enabled=True,
                model_id=ocr_cfg.trocr_model_id,
            )
        elif engine_name == "donut":
            engines["donut"] = EngineCfg(
                enabled=True,
                model_id=ocr_cfg.donut_model_id,
            )
        elif engine_name == "tesseract":
            engines["tesseract"] = EngineCfg(
                enabled=True,
                lang=ocr_cfg.tesseract_lang,
                tesseract_cmd=ocr_cfg.tesseract_cmd,
            )
        elif engine_name == "doctr":
            engines["doctr"] = EngineCfg(
                enabled=True,
                det_arch=ocr_cfg.doctr_det_arch,
                reco_arch=ocr_cfg.doctr_reco_arch,
            )
        elif engine_name == "olmocr":
            engines["olmocr"] = EngineCfg(
                enabled=True,
                model_id=ocr_cfg.olmocr_model_id,
            )

    return engines


def main():
    parser = argparse.ArgumentParser(description="OCR Quickstart")
    parser.add_argument("--gallery", help="Gallery directory (overrides default)")
    parser.add_argument("--output", help="Output directory (overrides default)")
    parser.add_argument("--strategy",
                        choices=["first_nonempty", "best_of", "consensus", "two_stage", "detector_only"],
                        help="Override strategy")
    parser.add_argument("--engines", help="Comma-separated engine list (e.g., 'olmocr,trocr,tesseract')")
    args = parser.parse_args()

    setup_logging()

    # Load application config
    app_config = get_app_config()
    ocr_cfg = app_config.vision.ocr

    # Resolve paths
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    gallery_dir = args.gallery or str(repo_root / "datasets" / "images" / "text_ocr")
    output_dir = args.output

    # Override engine order if provided
    if args.engines:
        engine_order = [e.strip() for e in args.engines.split(",")]
    else:
        engine_order = list(ocr_cfg.engine_order)

    # Build OcrConfig
    cfg = OcrConfig(
        gallery_dir=gallery_dir,
        output_dir=output_dir,
        min_file_kb=ocr_cfg.min_file_kb,
        min_dim_px=ocr_cfg.min_dim_px,
        rotations=list(ocr_cfg.rotations),
        engines=build_engine_configs(app_config),
        engine_order=engine_order,
        fusion=FusionCfg(
            strategy=args.strategy or ocr_cfg.strategy,
            verifier=ocr_cfg.verifier,
        ),
    )

    print("=== OCR Configuration ===")
    print(f"Strategy      : {cfg.fusion.strategy}")
    print(f"Verifier      : {cfg.fusion.verifier}")
    print(f"Engine order  : {cfg.engine_order}")
    print(f"Gallery       : {cfg.gallery_dir}")
    print(f"Output        : {cfg.output_dir or '(auto)'}")
    print(f"Rotations     : {cfg.rotations}")
    print()

    # Run OCR
    out = run_ocr(cfg)

    print("\n=== OCR SUMMARY ===")
    print(f"Processed : {out['processed']}")
    print(f"Found     : {out['found']}")
    print(f"Failed    : {out['failures']}")
    print(f"Run dir   : {out['out_dir']}")
    print(f"Found CSV : {out['found_csv']}")
    print(f"Failures  : {out['failures_csv']}")


if __name__ == "__main__":
    main()