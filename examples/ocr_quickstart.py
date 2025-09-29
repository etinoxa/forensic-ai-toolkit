# examples/ocr_quickstart.py
import os, pathlib, sys
from dataclasses import asdict

# Compute repo_root and set Paddle caches BEFORE importing any fait modules
_repo_root = pathlib.Path(__file__).resolve().parents[1]
_ocr_cache = _repo_root / ".fait" / "cache" / "models" / "ocr"
os.makedirs(_ocr_cache, exist_ok=True)
os.environ["PADDLEX_HOME"] = str(_ocr_cache)
os.environ["PPOCR_HOME"] = str(_ocr_cache)
os.environ.setdefault("PADDLEHUB_HOME", str(_ocr_cache))

# Add src to path after envs are set
sys.path.insert(0, str(_repo_root / "src"))

from dotenv import load_dotenv
load_dotenv()

from fait.core.logging_config import setup_logging
setup_logging()

# from fait.vision.ocr.config import load_ocr_config

from fait.vision.pipelines.ocr_pipeline import run_ocr_from_yaml


def main():
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent
    gallery_dir = (repo_root / "datasets/images/text_ocr").resolve()
    yaml_path   = (repo_root / "configs/vision/ocr.yaml").resolve()


    # cfg = load_ocr_config(str(yaml_path))
    # if gallery_dir.exists():
    #     cfg.gallery_dir = str(gallery_dir)

    # yaml_cfg = load_ocr_config(str(yaml_path))
    # … then coerce to pipeline OCRConfig so defaults like found.csv apply
    # cfg = OCRConfig(**asdict(yaml_cfg))
    out = run_ocr_from_yaml(str(yaml_path), gallery_dir=str(gallery_dir))

    print("\n=== OCR SUMMARY ===")
    print("Processed :", out["processed"])
    print("Found     :", out["found"])
    print("Failed    :", out["failures"])
    print("Run dir   :", out["out_dir"])
    print("found.csv :", out["found_csv"])
    print("failures  :", out["failures_csv"])

if __name__ == "__main__":
    main()