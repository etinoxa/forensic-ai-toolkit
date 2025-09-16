# src/fait/vision/pipelines/ocr_pipeline.py
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from fait.core.config_io import load_yaml
from fait.core.paths import get_paths
from fait.core.utils import (
    ensure_folder,
    is_image_file,
    human_size,
    ProgressMeter,
)
# ⬇️ Import the ONE TRUE config (no local duplicates here)
from fait.vision.ocr.config import OcrConfig, FusionCfg, load_ocr_config, EngineCfg
# ⬇️ Engine registry (expected to exist in src/fait/vision/ocr/engines/__init__.py)
from fait.vision.ocr.engines import get_engine

import logging
log = logging.getLogger("fait.vision.pipelines.ocr")


# ---------- helpers ----------

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_engine_order(cfg: OcrConfig) -> List[str]:
    # Keep only enabled engines, preserve the given order
    enabled = {name for name, e in cfg.engines.items() if e.get("enabled", True)}
    return [name for name in cfg.engine_order if name in enabled]


def _resolve_strategy_verifier(fu: FusionCfg) -> Tuple[str, str]:
    """ENV wins only if YAML says 'auto'."""
    strategy = (fu.strategy or "first_nonempty").lower()
    verifier = (fu.verifier or "none").lower()

    env_s = os.getenv("OCR_STRATEGY", "").strip().lower()
    env_v = os.getenv("OCR_VERIFIER", "").strip().lower()

    if strategy == "auto" and env_s in {"first_nonempty", "best_of", "consensus", "two_stage"}:
        strategy = env_s
    if verifier == "auto" and env_v in {"trocr", "donut", "tesseract", "paddleocr", "doctr", "none"}:
        verifier = env_v

    # clamp invalid combos
    if strategy != "two_stage":
        verifier = "none"

    return strategy, verifier


def _avg_conf(conf) -> Optional[float]:
    if conf is None:
        return None
    if isinstance(conf, (int, float)):
        return float(conf)
    try:
        vals = [float(x) for x in conf if x is not None]
        return float(sum(vals) / max(1, len(vals))) if vals else None
    except Exception:
        return None


def _normalize_engine_result(res) -> Tuple[str, Optional[float]]:
    """
    Accepts dicts like {'text':..., 'confidence':...} or tuples (text, conf) or just text.
    Returns (text:str, confidence: Optional[float]).
    """
    if res is None:
        return "", None
    if isinstance(res, tuple) and len(res) == 2:
        text, conf = res
        return str(text or "").strip(), _avg_conf(conf)
    if isinstance(res, dict):
        text = str(res.get("text", "") or "").strip()
        conf = _avg_conf(res.get("confidence"))
        return text, conf
    return str(res or "").strip(), None


def _should_skip(path: Path, min_kb: int, min_dim: int) -> Tuple[bool, str]:
    try:
        if path.stat().st_size < min_kb * 1024:
            return True, f"file <{min_kb}KB ({human_size(path.stat().st_size)})"
    except Exception:
        return True, "stat_failed"

    try:
        with Image.open(path) as im:
            w, h = im.size
        if w < min_dim or h < min_dim:
            return True, f"too_small ({w}x{h} < {min_dim}px)"
    except Exception:
        return True, "open_failed"

    return False, ""

def _hydrate_cfg(raw: dict) -> OcrConfig:
    """
    Accepts either the whole YAML dict or the nested 'ocr' block,
    returns a fully-typed OCRConfig (no bare dicts for nested fields).
    """
    data = raw.get("ocr", raw) or {}

    # engines -> Dict[str, OCREngineCfg]
    engines_in = data.get("engines") or {}
    engines = {name: EngineCfg(**(cfg or {})) for name, cfg in engines_in.items()}

    # fusion -> OCRFusionCfg
    fusion = FusionCfg(**(data.get("fusion") or {}))

    # rotations as tuple[int, ...]
    rotations = tuple(int(x) for x in (data.get("rotations") or [0, 90, 180, 270]))

    return OcrConfig(

        gallery_dir=data.get("gallery_dir"),
        output_dir=data.get("output_dir"),
        min_file_kb=int(data.get("min_file_kb", 10)),
        min_dim_px=int(data.get("min_dim_px", 100)),
        rotations=rotations,
        engines=engines,
        engine_order=list(data.get("engine_order") or ["trocr", "donut", "tesseract", "paddleocr", "doctr"]),
        fusion=fusion,
        # found_csv / failures_csv / log_file keep their dataclass defaults
    )


def run_ocr_from_yaml(yaml_path: str | Path, gallery_dir: str | Path | None = None) -> Dict:
    cfg = load_ocr_config(str(yaml_path))
    if gallery_dir:
        cfg.gallery_dir = str(gallery_dir)
    return run_ocr(cfg)


# ---------- main pipeline ----------

def run_ocr(cfg: OcrConfig) -> Dict:
    """
    Runs OCR with rotations and fusion. Writes:
      - found.csv
      - failures.csv
      - log.jsonl
    Returns a summary dict with paths and counts.
    """
    t0 = time.time()
    paths = get_paths()

    # Ensure Paddle-related caches point to the project cache BEFORE engines are created
    try:
        ocr_cache = paths.models_cache / "ocr"
        ensure_folder(ocr_cache)
        os.environ["PADDLEX_HOME"] = str(ocr_cache)
        os.environ["PPOCR_HOME"] = str(ocr_cache)
        os.environ.setdefault("PADDLEHUB_HOME", str(ocr_cache))
    except Exception:
        # Non-fatal: if anything goes wrong here, Paddle will fall back to its defaults
        pass

    # Normalize nested config in case callers passed raw dicts or dataclass instances
    if isinstance(getattr(cfg, "fusion", None), dict):
        cfg.fusion = FusionCfg(**cfg.fusion)  # type: ignore[assignment]

    # Ensure engines are dict-like for .get() access throughout
    def _eng_as_dict(e) -> dict:
        if isinstance(e, dict):
            return e
        # Fallback to attribute access if it's a dataclass/object
        return {
            "enabled": getattr(e, "enabled", True),
            "lang": getattr(e, "lang", "auto"),
            # pass through optional known keys if present
            "model_id": getattr(e, "model_id", None),
            "tesseract_cmd": getattr(e, "tesseract_cmd", None),
            "psm": getattr(e, "psm", None),
            "oem": getattr(e, "oem", None),
            "det_arch": getattr(e, "det_arch", None),
            "reco_arch": getattr(e, "reco_arch", None),
        }

    if isinstance(getattr(cfg, "engines", None), dict):
        cfg.engines = {name: _eng_as_dict(val) for name, val in cfg.engines.items()}  # type: ignore[assignment]

    # Resolve dirs
    gallery_dir = Path(cfg.gallery_dir) if cfg.gallery_dir else Path(
        os.getenv("OCR_GALLERY_DIR", paths.repo_root / "datasets/images/text_ocr")
    )
    if not gallery_dir.exists():
        raise FileNotFoundError(f"Gallery not found: {gallery_dir}")

    base_out = Path(cfg.output_dir) if cfg.output_dir else (paths.outputs / "ocr" / _now_tag())
    run_dir = base_out
    ensure_folder(run_dir)

    # Fixed filenames here (so we don’t need cfg.found_csv / cfg.failures_csv in YAML)
    found_csv = run_dir / "found.csv"
    failures_csv = run_dir / "failures.csv"
    jsonl_log = run_dir / "log.jsonl"

    # Strategy/verifier resolution (ENV wins only when YAML has 'auto')
    strategy, verifier = _resolve_strategy_verifier(cfg.fusion)
    log.info("ocr:start", extra={
        "gallery": str(gallery_dir),
        "run_dir": str(run_dir),
        "strategy": strategy,
        "verifier": verifier,
        "min_file_kb": cfg.min_file_kb,
        "min_dim_px": cfg.min_dim_px,
        "rotations": cfg.rotations,
        "engine_order": cfg.engine_order,
    })

    # Build engines in the order we’ll try
    order = _sanitize_engine_order(cfg)
    engines = {}
    for name in order:
        eng = get_engine(name, cfg.engines.get(name))  # pass per-engine cfg only
        if eng is not None:
            engines[name] = eng
    if not engines:
        raise RuntimeError("No OCR engines available/enabled after config filtering.")

    # IO setup
    found_f = found_csv.open("w", newline="", encoding="utf-8")
    fail_f = failures_csv.open("w", newline="", encoding="utf-8")
    log_f = jsonl_log.open("w", encoding="utf-8")
    found_w = csv.writer(found_f)
    fail_w = csv.writer(fail_f)
    found_w.writerow(["file", "text", "lang", "confidence", "notes"])
    fail_w.writerow(["file", "reason"])

    # Files to process
    files = [p for p in sorted(gallery_dir.iterdir()) if p.is_file() and is_image_file(p)]
    total = len(files)
    processed = 0
    found = 0
    failed = 0

    pm = ProgressMeter(
        total=total,
        label="ocr:progress",
        logger=log,
        log_path=jsonl_log,
        emit_every_n=10,
        emit_every_sec=2.0,
    )

    def _jsonl(entry: dict) -> None:
        log_f.write(json.dumps(entry) + "\n"); log_f.flush()

    try:
        for fp in files:
            processed += 1
            # Skip rules
            skip, reason = _should_skip(fp, cfg.min_file_kb, cfg.min_dim_px)
            if skip:
                fail_w.writerow([fp.name, reason])
                failed += 1
                pm.set_counts(processed, found=found, review=0)
                _jsonl({"file": fp.name, "action": "skipped", "reason": reason})
                continue

            # Try rotations in order; stop at first non-empty text given fusion policy
            best_text, best_conf, best_notes, best_lang = "", None, "", None

            with Image.open(fp) as base_img:
                for rot in (cfg.rotations or [0]):
                    try:
                        img = base_img.rotate(rot, expand=True) if rot else base_img
                    except Exception:
                        continue

                    # --- Strategy execution ---
                    # 1) first_nonempty: try each engine in order until text
                    # 2) best_of: run all, pick highest confidence
                    # 3) consensus: run all; if multiple non-empty match closely, prefer the higher confidence;
                    #               otherwise fallback to first_nonempty
                    # 4) two_stage: primary = engine_order[0], verifier = <resolved>; if primary empty/low conf,
                    #               try verifier and prefer it if more confident.

                    results: List[Tuple[str, Optional[float], str, str]] = []  # (text, conf, engine, lang_used)

                    if strategy == "first_nonempty":
                        for name in order:
                            lang = cfg.engines[name].get("lang", "auto")
                            text, conf = _normalize_engine_result(engines[name].ocr(img, lang=lang))
                            if text:
                                best_text, best_conf, best_notes, best_lang = text, conf, f"{name};rot={rot}", lang
                                break

                    elif strategy == "best_of":
                        for name in order:
                            lang = cfg.engines[name].get("lang", "auto")
                            text, conf = _normalize_engine_result(engines[name].ocr(img, lang=lang))
                            if text:
                                results.append((text, conf, name, lang))
                        if results:
                            # pick max confidence; if all conf None, pick the first
                            pick = max(results, key=lambda r: (r[1] is not None, r[1] or 0.0))
                            best_text, best_conf, best_notes, best_lang = pick[0], pick[1], f"{pick[2]};rot={rot}", pick[3]

                    elif strategy == "consensus":
                        for name in order:
                            lang = cfg.engines[name].get("lang", "auto")
                            text, conf = _normalize_engine_result(engines[name].ocr(img, lang=lang))
                            if text:
                                results.append((text, conf, name, lang))
                        if results:
                            # naive consensus: if two or more non-empty share a long common prefix, prefer higher conf
                            results_sorted = sorted(results, key=lambda r: (r[1] is not None, r[1] or 0.0), reverse=True)
                            best_text, best_conf, best_notes, best_lang = results_sorted[0][0], results_sorted[0][1], f"{results_sorted[0][2]};rot={rot}", results_sorted[0][3]

                    elif strategy == "two_stage":
                        primary_name = order[0]
                        p_lang = cfg.engines[primary_name].get("lang", "auto")
                        p_text, p_conf = _normalize_engine_result(engines[primary_name].ocr(img, lang=p_lang))

                        if p_text and (p_conf is None or p_conf >= 0.5):
                            best_text, best_conf, best_notes, best_lang = p_text, p_conf, f"{primary_name};rot={rot}", p_lang
                        else:
                            if verifier != "none" and verifier in engines:
                                v_lang = cfg.engines[verifier].get("lang", "auto")
                                v_text, v_conf = _normalize_engine_result(engines[verifier].ocr(img, lang=v_lang))
                                # prefer verifier if it has non-empty text & >= primary confidence
                                if v_text and ((v_conf or 0.0) >= (p_conf or 0.0)):
                                    best_text, best_conf, best_notes, best_lang = v_text, v_conf, f"{verifier};rot={rot}", v_lang
                                else:
                                    best_text, best_conf, best_notes, best_lang = p_text, p_conf, f"{primary_name};rot={rot}", p_lang
                            else:
                                best_text, best_conf, best_notes, best_lang = p_text, p_conf, f"{primary_name};rot={rot}", p_lang

                    else:
                        # fallback to first_nonempty
                        for name in order:
                            lang = cfg.engines[name].get("lang", "auto")
                            text, conf = _normalize_engine_result(engines[name].ocr(img, lang=lang))
                            if text:
                                best_text, best_conf, best_notes, best_lang = text, conf, f"{name};rot={rot}", lang
                                break

                    if best_text:
                        break  # stop trying more rotations

            if best_text:
                found_w.writerow([fp.name, best_text, best_lang or "", f"{best_conf:.3f}" if best_conf is not None else "", best_notes])
                found += 1
                _jsonl({"file": fp.name, "engine": best_notes.split(";")[0], "rotation": best_notes.split("rot=")[-1], "len_text": len(best_text), "conf": best_conf})
            else:
                fail_w.writerow([fp.name, "empty text after rotations"])
                failed += 1
                _jsonl({"file": fp.name, "action": "failed", "reason": "empty_after_rotations"})

            pm.set_counts(processed, found=found, review=0)

    finally:
        try:
            found_f.close()
        except Exception:
            pass
        try:
            fail_f.close()
        except Exception:
            pass
        try:
            log_f.close()
        except Exception:
            pass

    dt = time.time() - t0
    log.info("ocr:done", extra={"processed": processed, "found": found, "failed": failed, "secs": round(dt, 2), "run_dir": str(run_dir)})

    return {
        "processed": processed,
        "found": found,
        "failed": failed,
        "run_dir": str(run_dir),
        "found_csv": str(found_csv),
        "failures_csv": str(failures_csv),
        "log_jsonl": str(jsonl_log),
        "strategy": strategy,
        "verifier": verifier,
        "engine_order": order,
    }
