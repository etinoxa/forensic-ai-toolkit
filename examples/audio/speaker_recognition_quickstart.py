# examples/audio/speaker_recognition_quickstart.py
"""
Speaker Recognition Quickstart - Uses config.yaml for all settings.
Run-specific overrides can be passed as CLI arguments.
"""

import os
import sys
import pathlib
import argparse
import warnings

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from dotenv import load_dotenv
load_dotenv()

from fait.core.logging_config import setup_logging
from fait.core.app_config import get_app_config

# Trigger embedder registration
import fait.audio.speaker_recognition.models.speechbrain_embedder  # noqa: F401
import fait.audio.speaker_recognition.models.titanet_embedder  # noqa: F401
import fait.audio.speaker_recognition.models.wavlm_embedder  # noqa: F401

from fait.audio.pipelines.speaker_recognition_pipeline import (
    SpeakerMatchConfig,
    SingleStageCfg,
    TwoStageCfg,
    ThreeStageCfg,
    SpeakerModelsCfg,
    run_speaker_match,
)

# Silence warnings
os.environ.setdefault("SPEECHBRAIN_LOCAL_FILES_STRATEGY", "copy")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated.*", module="webrtcvad")


def main():
    parser = argparse.ArgumentParser(description="Speaker Recognition Quickstart")
    parser.add_argument("--reference", help="Reference directory (overrides default)")
    parser.add_argument("--gallery", help="Gallery directory (overrides default)")
    parser.add_argument("--strategy",
                       choices=["speechbrain_only", "titanet_only", "two_stage", "three_stage", "detector_only"],
                       help="Override strategy")
    parser.add_argument("--detector", choices=["titanet", "wavlm", "none"], help="Override detector")
    parser.add_argument("--tertiary", choices=["titanet", "wavlm", "none"], help="Override tertiary")
    parser.add_argument("--output", help="Output directory (overrides default)")
    args = parser.parse_args()

    setup_logging()

    # Load application config
    app_config = get_app_config()
    audio_config = app_config.audio.speaker_recognition

    # Resolve paths
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    reference_dir = args.reference or str(repo_root / "datasets" / "audio" / "reference")
    gallery_dir = args.gallery or str(repo_root / "datasets" / "audio" / "gallery")
    output_dir = args.output

    # Config with CLI overrides
    strategy = args.strategy or audio_config.strategy
    detector = args.detector or audio_config.detector
    tertiary = args.tertiary or audio_config.tertiary

    cfg = SpeakerMatchConfig(
        reference_dir=reference_dir,
        gallery_dir=gallery_dir,
        output_dir=output_dir,
        strategy=strategy,
        detector=detector,
        tertiary=tertiary,
        single=SingleStageCfg(tau=0.70),
        two_stage=TwoStageCfg(
            method=audio_config.fusion_method,
            alpha=audio_config.alpha,
            tau_star=audio_config.tau_star,
        ),
        three_stage=ThreeStageCfg(
            method=audio_config.fusion_method,
            alpha12=audio_config.alpha,
            alpha123=audio_config.alpha,
            tau_star=audio_config.tau_star,
        ),
        models=SpeakerModelsCfg(
            speechbrain_id=audio_config.speechbrain_id,
            titanet_id=audio_config.titanet_id,
            wavlm_id=audio_config.wavlm_id,
        ),
        progress="log",
        use_cache=True,
    )

    print("=== Speaker Recognition Configuration ===")
    print(f"Strategy      : {cfg.strategy}")
    print(f"Detector      : {cfg.detector}")
    print(f"Tertiary      : {cfg.tertiary}")
    print(f"Reference dir : {cfg.reference_dir}")
    print(f"Gallery dir   : {cfg.gallery_dir}")
    print(f"Output dir    : {cfg.output_dir or '(auto)'}")
    print()

    # Run
    out = run_speaker_match(cfg)

    print("\n=== SPEAKER MATCH SUMMARY ===")
    print(f"Processed   : {out.get('processed')}")
    print(f"Found       : {out.get('found')}")
    print(f"Run dir     : {out.get('run_dir')}")
    if out.get("log_jsonl"):
        print(f"Log (JSONL) : {out['log_jsonl']}")


if __name__ == "__main__":
    main()