from __future__ import annotations
import os, json, time, shutil, logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime
from pathlib import Path
from PIL import Image

from fait.core.paths import get_paths
from fait.core.utils import ensure_folder, is_image_file, file_md5, to_safe_filename

from fait.vision.detectors.grounding_dino import GroundingDINO, GDINOConfig
from fait.vision.detectors.deformable_detr import DeformableDETR, DefDETRConfig
from fait.vision.detectors.yolo import YOLODetector, YoloConfig

log = logging.getLogger("fait.vision.pipelines.object_screen")

@dataclass
class FusionConfig:
    iou_gate: float = 0.50
    rule: str = "and"            # "and" or "weighted"
    alpha: float = 0.5           # for weighted: S = a*gdino + (1-a)*det
    tau_star: float = 0.6        # acceptance threshold for weighted score
    class_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "handgun": 0.60, "knife": 0.55, "micro sd card": 0.50, "micro sd": 0.50,
        "sim card": 0.50, "laptop": 0.55, "usb": 0.55
    })
    borderline_window: float = 0.05  # ± window for review queue
    gdino_only_default_tau: float = 0.55 # used when verifier == "none"

def _iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def _norm_label(name: str) -> str:
    n = name.lower().strip()
    aliases = {
        "cell phone": "phone", "mobile phone": "phone", "cellphone": "phone",
        "micro-sd": "micro sd", "microsd": "micro sd", "micro sd card": "micro sd",
        "usb stick": "usb", "thumb drive": "usb",
        "hand gun": "handgun"
    }
    return aliases.get(n, n)

def _pick_class_threshold(cfg: FusionConfig, label: str) -> float:
    l = _norm_label(label)
    return cfg.class_thresholds.get(l, 0.55)

@dataclass
class ScreenConfig:
    prompts: List[str]
    gallery_dir: str
    output_dir: Optional[str] = None
    save_crops: bool = False

    verifier: Literal["none", "deformable_detr", "yolo", "auto"] = "deformable_detr"
    run_name: Optional[str] = None
    gdino: GDINOConfig = GDINOConfig()
    detr: DefDETRConfig = DefDETRConfig(class_whitelist=[
        # choose classes you care about from COCO label set
        "knife", "scissors", "laptop", "cell phone", "remote", "backpack",
        "handbag", "suitcase", "mouse", "keyboard"
    ])
    yolo: YoloConfig = YoloConfig()
    fusion: FusionConfig = FusionConfig()

def _crop(image: Image.Image, box: List[float]) -> Image.Image:
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, image.width - 1))
    y1 = max(0, min(y1, image.height - 1))
    x2 = max(1, min(x2, image.width))
    y2 = max(1, min(y2, image.height))
    return image.crop((x1, y1, x2, y2))

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_object_screen(cfg: ScreenConfig) -> Dict:
    t0 = time.time()
    paths = get_paths()

    # Resolve base output dir
    base = Path(cfg.output_dir) if cfg.output_dir else (paths.outputs / "object")
    ensure_folder(base)

    # Run name (custom or timestamp) and ensure uniqueness
    name = cfg.run_name.strip() if cfg.run_name else f"screen_{_now_tag()}"
    name = to_safe_filename(name) or f"screen_{_now_tag()}"
    run_dir = base / name
    if run_dir.exists():
        run_dir = base / f"{name}_{_now_tag()}"
    ensure_folder(run_dir)

    # Subfolders (created lazily)
    crops_dir = run_dir / "crops"
    found_dir = run_dir / "found_images"
    review_dir = run_dir / "review_queue"
    _made_crops = _made_found = _made_review = False

    log_path = run_dir / "log.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")

    # --- pick verifier from env if provided ---
    env_v = os.getenv("FAIT_OBJECT_VERIFIER", "").strip().lower()
    if env_v in {"yolo", "deformable_detr", "none"}:
        cfg.verifier = env_v
    elif env_v:
        logging.getLogger("fait.vision.pipelines.object_screen").warning(
            "object_screen: invalid FAIT_OBJECT_VERIFIER=%r (using %s)",
            env_v, cfg.verifier
        )

    # Resolve verifier: CLI > env > default
    if cfg.verifier == "auto":
        env_v = os.getenv("FAIT_OBJECT_VERIFIER", "").strip().lower()
        if env_v in {"yolo", "deformable_detr", "none"}:
            cfg.verifier = env_v
        else:
            cfg.verifier = "deformable_detr"  # fallback default


    # Models
    gd = GroundingDINO(cfg.gdino, cache_dir=str(paths.models_cache))

    cd = None
    if cfg.verifier == "deformable_detr":
        cd = DeformableDETR(cfg.detr, cache_dir=str(paths.models_cache))
    elif cfg.verifier == "yolo":
        cd = YOLODetector(cfg.yolo, cache_dir=str(paths.models_cache))
    elif cfg.verifier == "none":
        cd = None
    else:
        raise ValueError(f"Unknown verifier: {cfg.verifier}")

    log.info("object_screen:verifier", extra={"verifier": cfg.verifier})

    gallery_files = [f for f in sorted(os.listdir(cfg.gallery_dir)) if is_image_file(os.path.join(cfg.gallery_dir, f))]
    log_path = run_dir / "log.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")

    found_images = []
    reviewed = []
    processed = 0

    for fname in gallery_files:
        fpath = os.path.join(cfg.gallery_dir, fname)
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            continue

        proposals = gd.propose(img, cfg.prompts)
        if not proposals:
            processed += 1
            continue

        accepted = False
        borderline_hit = False
        best_entry = None

        for prop in proposals:
            if cfg.verifier == "none":
                # ------- GDINO-ONLY MODE -------
                g = float(prop["score"])
                label = prop["prompt"]  # we treat the prompt as the "class" label here
                tau = _pick_class_threshold(cfg.fusion, label) or cfg.fusion.gdino_only_default_tau

                entry = {
                    "image": fpath,
                    "hash": file_md5(fpath),
                    "prompt": prop["prompt"],
                    "gdino_score": g,
                    "detector_label": None,
                    "detector_score": None,
                    "iou": 1.0,  # not applicable; set to 1 for traceability
                    "fused": g,
                    "threshold": tau,
                    "gdino_box": prop["box"],
                    "detector_mapped_box": None,
                    "model_versions": {
                        "grounding_dino": gd.cfg.model_id,
                        "verifier": "none",
                    },
                    "time": datetime.now().isoformat(timespec="seconds"),
                }

                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

                if g >= tau:
                    accepted = True
                    best_entry = entry
                    break
                elif abs(g - tau) <= cfg.fusion.borderline_window:
                    borderline_hit = True
                    best_entry = best_entry or entry

            else:
                # ------- TWO-STAGE MODE (current implementation) -------
                roi = _crop(img, prop["box"])
                dets = cd.detect(roi)

                px1, py1, px2, py2 = prop["box"]

                for det in dets:
                    dx1, dy1, dx2, dy2 = det["box"]
                    mapped = [px1 + dx1, py1 + dy1, px1 + dx2, py1 + dy2]
                    iou = _iou_xyxy(prop["box"], mapped)

                    if iou < cfg.fusion.iou_gate:
                        continue

                    label = det["label"]
                    g, c = float(prop["score"]), float(det["score"])

                    if cfg.fusion.rule == "and":
                        tau = _pick_class_threshold(cfg.fusion, label)
                        ok = (g >= tau) and (c >= tau)
                        fused = min(g, c)
                    else:
                        fused = cfg.fusion.alpha * g + (1 - cfg.fusion.alpha) * c
                        tau = cfg.fusion.tau_star
                        ok = fused >= tau

                    entry = {
                        "image": fpath,
                        "hash": file_md5(fpath),
                        "prompt": prop["prompt"],
                        "gdino_score": g,
                        "detector_label": label,
                        "detector_score": c,
                        "iou": iou,
                        "fused": fused,
                        "threshold": tau,
                        "gdino_box": prop["box"],
                        "detector_mapped_box": mapped,
                        "model_versions": {
                            "grounding_dino": gd.cfg.model_id,
                            "deformable_detr": cd.cfg.model_id,
                        },
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }

                    log_file.write(json.dumps(entry) + "\n")
                    log_file.flush()

                    if ok:
                        accepted = True
                        best_entry = entry
                        break
                    elif abs(fused - tau) <= cfg.fusion.borderline_window:
                        borderline_hit = True
                        best_entry = best_entry or entry

                if accepted:
                    break

        processed += 1

        if accepted and best_entry:
            found_images.append(fpath)
            if not _made_found:
                ensure_folder(found_dir);
                _made_found = True
            shutil.copy2(fpath, found_dir / os.path.basename(fpath))

            if cfg.save_crops:
                if not _made_crops:
                    ensure_folder(crops_dir);
                    _made_crops = True
                # for GDINO-only, detector_mapped_box may be None; fall back to gdino_box
                crop_box = best_entry.get("detector_mapped_box") or best_entry.get("gdino_box")
                if crop_box:
                    _crop(img, crop_box).save(crops_dir / f"{Path(fname).stem}_crop.jpg")

        elif borderline_hit and best_entry:
            reviewed.append(fpath)
            if not _made_review:
                ensure_folder(review_dir);
                _made_review = True
            shutil.copy2(fpath, review_dir / os.path.basename(fpath))

    log_file.close()

    elapsed = time.time() - t0
    summary = {
        "processed": processed,
        "found": len(found_images),
        "review": len(reviewed),
        "run_dir": str(run_dir),
        "log": str(log_path),
    }
    log.info("object_screen:done", extra=summary)
    return summary
