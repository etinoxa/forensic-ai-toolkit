# examples/vision/facial_recognition_quickstart.py
"""
Face Recognition Quickstart - Uses config.yaml for all settings.
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
from fait.core.registry import get_embedder
from fait.vision.pipelines.facial_recognition_pipeline import run_facial_recognition

# Trigger registration
import fait.vision.facial_recognition.models.arcface as _arcface
import fait.vision.facial_recognition.models.clip as _clip


def main() -> None:
    parser = argparse.ArgumentParser(description="Face Recognition Quickstart")
    parser.add_argument("--reference", help="Reference directory (overrides default)")
    parser.add_argument("--gallery", help="Gallery directory (overrides default)")
    parser.add_argument("--recognizer", choices=["arcface", "clip"], help="Override recognizer")
    parser.add_argument("--metric", choices=["euclidean", "cosine", "auto"], help="Override metric")
    parser.add_argument("--thresholds", help="Comma-separated thresholds (e.g., '0.8,0.9')")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("fait.vision.pipelines.face_match")
    run_id = str(uuid.uuid4())
    log.info("face_match:start", extra={"run_id": run_id})

    # Load application config
    app_config = get_app_config()
    face_config = app_config.vision.face_recognition

    # Resolve paths
    ref_dir = pathlib.Path(args.reference) if args.reference else (
        ROOT / "datasets" / "images" / "face" / "reference_images"
    )
    gal_dir = pathlib.Path(args.gallery) if args.gallery else (
        ROOT / "datasets" / "images" / "face" / "gallery"
    )

    # Config with CLI overrides
    model = args.recognizer or face_config.recognizer
    metric = args.metric or face_config.metric
    if metric == "auto":
        metric = "euclidean" if model == "arcface" else "cosine"

    if args.thresholds:
        thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    else:
        thresholds = face_config.thresholds

    plot_results = not args.no_plot and face_config.plot_results

    # Sanity checks
    for p in (ref_dir, gal_dir):
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")

    log.info("face_match:config", extra={
        "recognizer": model,
        "metric": metric,
        "thresholds": thresholds,
        "reference_dir": str(ref_dir),
        "gallery_dir": str(gal_dir),
        "plot_results": plot_results,
    })

    print("=== Face Recognition Configuration ===")
    print(f"Recognizer    : {model}")
    print(f"Metric        : {metric}")
    print(f"Thresholds    : {thresholds}")
    print(f"Reference dir : {ref_dir}")
    print(f"Gallery dir   : {gal_dir}")
    print(f"Plot results  : {plot_results}")
    print()

    # Run
    embedder = get_embedder(model)
    res = run_facial_recognition(
        embedder=embedder,
        reference_dir=str(ref_dir),
        gallery_dir=str(gal_dir),
        thresholds=thresholds,
        metric=metric,
        plot_results=plot_results,
    )

    print("\n=== FACE MATCH SUMMARY ===")
    print(f"Model                 : {res['model']}")
    print(f"Metric                : {res['metric']}")
    print(f"Processed             : {res['processed']}")
    print(f"Matches per threshold : {res['matches_per_threshold']}")
    if res.get('closest'):
        print(f"\nTop 10 closest:")
        for i, (name, dist) in enumerate(res['closest'][:10], 1):
            print(f"  {i:2d}. {name:<30} ({metric}: {dist:.4f})")
    print(f"\nReport    : {res.get('report_path')}")
    print(f"Plot      : {res.get('plot_path')}")
    print(f"Output dir: {res.get('output_dir')}")
    print(f"Run ID    : {run_id}")


if __name__ == "__main__":
    main()