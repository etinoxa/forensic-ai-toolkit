# examples/object_screen_quickstart.py
"""
Quickstart: Open-vocabulary → Closed-set object screening.

Flow:
  GroundingDINO (prompts) -> proposals -> crops -> Deformable DETR (verification)
  -> IoU gate + fusion -> copy FOUND images (and optional crops) to .fait/outputs/object/<run>/...
  -> JSONL log per candidate

Adjust the PROMPTS list and (optionally) fusion/threshold knobs below.
"""

# --- load .env from repo root & silence TF/Protobuf BEFORE any heavy imports ---
import os, warnings, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root (parent of examples/)
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hide TF INFO/WARN banners
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # hide oneDNN banner

warnings.filterwarnings(
    "ignore",
    message=r"Protobuf gencode version .* is exactly one major version older",
    category=UserWarning,
    module="google\.protobuf\.runtime_version",
)

# --- standard imports after env is set ---
import logging
import uuid

from fait.core.logging_config import setup_logging
from fait.vision.pipelines.object_screen import ScreenConfig, run_object_screen


def _count_images(dir_path: pathlib.Path) -> int:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    try:
        return sum(1 for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in exts)
    except FileNotFoundError:
        return 0


def main() -> None:
    setup_logging()

    log = logging.getLogger("fait.vision.pipelines.object_screen")
    run_id = str(uuid.uuid4())
    log.info("object_screen:start", extra={"run_id": run_id})

    # --- Paths (relative to repo root) ---
    gallery_dir = (ROOT / "datasets" / "images" / "objects" / "raw").resolve()

    # --- Prompt pack (tune to your needs) ---
    PROMPTS = [
        "gun","handgun","pistol","revolver","firearm",
        "gun barrel","pistol grip",
        "knife","kitchen knife","chef knife","blade","dagger"
    ]

    # --- Basic validation ---
    if not gallery_dir.exists():
        print(f"Error: Gallery directory does not exist: {gallery_dir}")
        print("Create it and add images, or update the path in this script.")
        return

    n_gallery = _count_images(gallery_dir)
    if n_gallery == 0:
        print(f"Error: No image files found in gallery directory: {gallery_dir}")
        print("Please add image files (.jpg, .png, etc.) to this directory.")
        return

    print(f"Gallery images: {n_gallery}")
    print(f"Gallery dir   : {gallery_dir}")
    print(f"Prompts       : {PROMPTS}")

    # --- Configure the pipeline ---
    cfg = ScreenConfig(
        prompts=PROMPTS,
        gallery_dir=str(gallery_dir),
        output_dir=None,      # None -> <repo>/.fait/outputs/object
        save_crops=True,      # store ROI crops for triage
    )

    # GroundingDINO (open-vocab) knobs
    cfg.gdino.box_threshold = 0.35
    cfg.gdino.text_threshold = 0.45
    cfg.gdino.box_expand = 0.20     # 15% padding around proposals
    cfg.gdino.nms_iou = 0.5

    # Deformable DETR (closed-set) knobs
    # Whitelist COCO classes that are most relevant to your prompts
    cfg.detr.class_whitelist = [
        "knife", "scissors", "laptop", "cell phone", "remote", "backpack",
        "handbag", "suitcase", "mouse", "keyboard",
    ]
    cfg.detr.score_threshold = 0.25
    cfg.detr.nms_iou = 0.5

    # YoloV8 knobs
    # cfg.verifier = "yolo"
    # Optionally, pick relevant COCO labels YOLO knows about:
    cfg.yolo.class_whitelist = ["knife", "laptop", "cell phone", "backpack", "handbag", "suitcase"]
    cfg.yolo.model_id = "yolov8n.pt"  # or "yolo11n.pt" if you have it
    cfg.yolo.score_threshold = 0.25
    cfg.yolo.imgsz = 640

    # Fusion / decision policy
    cfg.fusion.iou_gate = 0.50
    cfg.fusion.rule = "weighted"     # "and" or "weighted"
    cfg.fusion.alpha = 0.5           # S = alpha*gdino + (1-alpha)*detector
    cfg.fusion.tau_star = 0.60       # accept if S >= tau*
    cfg.fusion.borderline_window = 0.05
    # per-class overrides (optional)
    cfg.fusion.class_thresholds.update({
        "handgun": 0.60,
        "knife": 0.55,
        "micro sd": 0.50,
        "usb": 0.55,
        "laptop": 0.55,
        "phone": 0.55,
    })



    try:
        summary = run_object_screen(cfg)
        print("\n=== OBJECT SCREEN SUMMARY ===")
        print(f"Processed     : {summary['processed']}")
        print(f"Found images  : {summary['found']}")
        print(f"Review queue  : {summary['review']}")
        print(f"Run directory : {summary['run_dir']}")
        print(f"Log (JSONL)   : {summary['log']}")
    except Exception as e:
        print(f"Error during object screening: {e}")
        print("Please check your images, prompts, and environment, then try again.")


if __name__ == "__main__":
    main()
