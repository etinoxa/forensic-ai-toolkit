from dotenv import load_dotenv
load_dotenv()

# If you've done: `pip install -e .` you can delete the next 2 lines.
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import os
from fait.core.registry import get_embedder
from fait.vision.pipelines.face_match import run_face_match
from fait.core.logging_config import setup_logging
setup_logging()

import logging, uuid
log = logging.getLogger("fait.vision.pipelines.face_match")
run_id = str(uuid.uuid4())
log.info("face_match: starting", extra={"run_id": run_id, "component": "arcface"})


def main():
    # Define paths relative to the script location
    script_dir = pathlib.Path(__file__).resolve().parent
    reference_dir = script_dir / "/datasets/images/face/reference_images"
    gallery_dir = script_dir / "datasets/images/face/gallery"


    # Convert to absolute paths and check if they exist
    reference_dir = reference_dir.resolve()
    gallery_dir = gallery_dir.resolve()


    # Validate that required directories exist
    if not reference_dir.exists():
        print(f"Error: Reference directory does not exist: {reference_dir}")
        print("Please create this directory and add reference images, or update the path.")
        return

    if not gallery_dir.exists():
        print(f"Error: Gallery directory does not exist: {gallery_dir}")
        print("Please create this directory and add gallery images, or update the path.")
        return

    # Check if directories contain image files
    ref_images = [f for f in os.listdir(reference_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))]
    gallery_images = [f for f in os.listdir(gallery_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))]

    if not ref_images:
        print(f"Error: No image files found in reference directory: {reference_dir}")
        print("Please add image files (.jpg, .png, etc.) to this directory.")
        return

    if not gallery_images:
        print(f"Error: No image files found in gallery directory: {gallery_dir}")
        print("Please add image files (.jpg, .png, etc.) to this directory.")
        return

    print(f"Found {len(ref_images)} reference image(s) and {len(gallery_images)} gallery image(s)")
    print(f"Reference directory: {reference_dir}")
    print(f"Gallery directory: {gallery_dir}")


    try:
        embedder = get_embedder("arcface")  # or "clip"
        res = run_face_match(
            embedder=embedder,
            reference_dir=str(reference_dir),
            gallery_dir=str(gallery_dir),
            thresholds=[0.80, 0.90],  # ArcFace (euclidean): lower = stricter
            metric="euclidean",  # use "cosine" for CLIP (1 - similarity)
            plot_results=True,
        )
        print("Matches:", res["matches_per_threshold"])
        print("Report :", res["report_path"])
        print("Plot   :", res["plot_path"])
    except Exception as e:
        print(f"Error during face matching: {e}")
        print("Please check your image files and try again.")


if __name__ == "__main__":
    main()


