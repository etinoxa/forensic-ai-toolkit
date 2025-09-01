# examples/speaker_match_quickstart.py
import os, pathlib, sys, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from fait.core.logging_config import setup_logging
setup_logging()

# Ensure the audio embedders are imported (so registry decorators run)
import fait.audio.embeddings.speechbrain_embedder  # noqa: F401
import fait.audio.embeddings.resemblyzer_embedder  # noqa: F401

from fait.audio.pipelines.speaker_match import SpeakerMatchConfig, run_speaker_match

os.environ.setdefault("SPEECHBRAIN_LOCAL_FILES_STRATEGY", "copy")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated.*", module="webrtcvad")


def main():
    script_dir = pathlib.Path(__file__).resolve().parent
    reference_dir = script_dir / "datasets/audio/reference"   # adjust to your data
    gallery_dir   = script_dir / "datasets/audio/gallery"     # adjust to your data

    cfg = SpeakerMatchConfig(
        reference_dir=str(reference_dir.resolve()),
        gallery_dir=str(gallery_dir.resolve()),
        thresholds=[0.70, 0.75, 0.80],        # higher = stricter
        embedder=os.getenv("FAIT_AUDIO_EMBEDDER", "speechbrain").lower()
    )
    out = run_speaker_match(cfg)

    print("\n=== SPEAKER MATCH SUMMARY ===")
    print("Processed   :", out["processed"])
    print("Run dir     :", out["run_dir"])
    print("Report      :", out["report_path"])
    print("Matches     :", out["matches_per_threshold"])
    print("Top         :", out["top"])

if __name__ == "__main__":
    main()
