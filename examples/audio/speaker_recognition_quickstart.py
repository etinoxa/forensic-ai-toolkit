# examples/speaker_recognition_quickstart.py
from __future__ import annotations
import os, sys, pathlib, warnings, yaml
from dataclasses import fields, is_dataclass

# Add repo /src to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from fait.core.logging_config import setup_logging
setup_logging()

# Ensure the audio embedders are imported (so registry decorators run)
import fait.audio.speaker_recognition.models.speechbrain_embedder  # noqa: F401

from fait.audio.pipelines.speaker_recognition_pipeline import (
    SpeakerMatchConfig,
    run_speaker_match,
)

# SpeechBrain/HF hub: avoid symlink strategies on Windows
os.environ.setdefault("SPEECHBRAIN_LOCAL_FILES_STRATEGY", "copy")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated.*", module="webrtcvad")


def _from_dict(cls, data):
    """Recursively map a dict into a dataclass (ignores unknown keys)."""
    if not is_dataclass(cls):
        return data
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            val = data[f.name]
            if hasattr(f.type, "__dataclass_fields__") and isinstance(val, dict):
                kwargs[f.name] = _from_dict(f.type, val)
            else:
                kwargs[f.name] = val
    return cls(**kwargs)


def main():
    # Resolve repo root and default folders
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    reference_dir = repo_root / "datasets" / "audio" / "reference"
    gallery_dir   = repo_root / "datasets" / "audio" / "gallery"

    # Load YAML config (strategy/detector/tertiary/progress/etc.)
    yaml_path = repo_root / "config" / "audio" / "speaker_match.yaml"
    cfg_dict = {}
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}

    # Make sure YAML cannot override our paths (we set them here)
    cfg_dict.pop("reference_dir", None)
    cfg_dict.pop("gallery_dir", None)

    # Build config from YAML, then inject paths from quickstart
    cfg_dict["reference_dir"] = str(reference_dir.resolve())
    cfg_dict["gallery_dir"] = str(gallery_dir.resolve())
    cfg = _from_dict(SpeakerMatchConfig, cfg_dict)

    # IMPORTANT: we do NOT set output_dir/strategy/detector/tertiary here.
    # The pipeline should apply ".env only if all auto" internally.

    print("Reference dir:", cfg.reference_dir)
    print("Gallery dir  :", cfg.gallery_dir)

    out = run_speaker_match(cfg)

    print("\n=== SPEAKER MATCH SUMMARY ===")
    print("Processed   :", out.get("processed"))
    print("Found       :", out.get("found"))
    print("Run dir     :", out.get("run_dir"))
    if out.get("report_path"):
        print("Report      :", out["report_path"])
    if out.get("matches_per_threshold") is not None:
        print("Matches     :", out["matches_per_threshold"])
    if out.get("top") is not None:
        print("Top         :", out["top"])


if __name__ == "__main__":
    main()
