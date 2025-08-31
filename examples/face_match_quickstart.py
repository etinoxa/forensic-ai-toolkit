# examples/face_match_quickstart.py
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
from fait.core.config_io import load_yaml
from fait.core.registry import get_embedder
from fait.vision.pipelines.face_match import run_face_match


def resolve_path(p: str | None, fallback: pathlib.Path) -> pathlib.Path:
    if not p:
        return fallback
    pp = pathlib.Path(p)
    return (ROOT / p).resolve() if not pp.is_absolute() else pp.resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Face match (YAML-driven)")
    ap.add_argument("--config", default="configs/vision/face_match.yaml")
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger("fait.vision.pipelines.face_match")
    run_id = str(uuid.uuid4())
    log.info("face_match:start", extra={"run_id": run_id})

    # ---- Load YAML ----
    cfg = load_yaml(args.config)

    # ---- Resolve paths ----
    ref_dir = resolve_path(
        cfg.get("reference_dir"),
        ROOT / "datasets" / "images" / "face" / "reference_images",
    )
    gal_dir = resolve_path(
        cfg.get("gallery_dir"),
        ROOT / "datasets" / "images" / "face" / "gallery",
    )

    # ---- Recognizer & metric ----
    model = str(cfg.get("recognizer", "arcface")).lower()  # arcface | clip
    metric = cfg.get("metric", "auto").lower()             # euclidean | cosine | auto
    if metric == "auto":
        metric = "euclidean" if model == "arcface" else "cosine"

    thresholds = cfg.get("thresholds", [0.8, 0.9])  # floats
    plot_results = bool(cfg.get("plot_results", True))

    # ---- Basic sanity ----
    for p in (ref_dir, gal_dir):
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
    if not thresholds or not isinstance(thresholds, (list, tuple)):
        raise ValueError("Invalid thresholds in YAML (expected list of floats).")

    log.info("face_match:config", extra={
        "recognizer": model,
        "metric": metric,
        "thresholds": thresholds,
        "reference_dir": str(ref_dir),
        "gallery_dir": str(gal_dir),
        "plot_results": plot_results,
    })

    # ---- Run ----
    embedder = get_embedder(model)
    res = run_face_match(
        embedder=embedder,
        reference_dir=str(ref_dir),
        gallery_dir=str(gal_dir),
        thresholds=[float(t) for t in thresholds],
        metric=metric,
        plot_results=plot_results,
    )

    print("\n=== FACE MATCH SUMMARY ===")
    print(f"Matches per threshold : {res['matches_per_threshold']}")
    print(f"Closest (top 10)      : {res.get('top10', [])}")
    print(f"Report                : {res.get('report_path')}")
    print(f"Plot                  : {res.get('plot_path')}")
    print(f"Run ID                : {run_id}")


if __name__ == "__main__":
    main()


'''
Sample usage:
match faces with arcface; reference_dir=datasets/images/face/reference_images; gallery_dir=datasets/images/face/gallery; thresholds=[0.8,0.9]; plot_results=true

match faces with clip; reference_dir=datasets/images/face/reference_images; gallery_dir=datasets/images/face/gallery; thresholds=[0.8,0.9]; plot_results=true
'''

