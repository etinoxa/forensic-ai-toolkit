# Suspect Facial Matcher
Offline image recognition pipeline using OpenAI CLIP ViT and ArcFace CNN.

## Features
- ArcFace or CLIP embeddings
- Supports multiple similarity thresholds
- Uses weighted similarity for better matching
- Optional face cropping for ViT models
- Optional face-only cropping for ViT
- Caches embeddings for faster re-runs
- Sorts matched images by threshold folders

## Installation
```bash
pip install -r requirements.txt
```

## Usage
**- ArcFace**
```bash
python facial_recognition_arcface.py \
    --reference-dir ./path/to/reference_images \
    --gallery-dir ./path/to/suspect_gallery \
    --output-dir ./path/to/output_dir  \
    --use-cache \
    --thresholds 0.8 0.9 1.0 \
    --distance-metric euclidean \
    --plot-results 
```

**- OpenAI CLIP ViT**
```bash
python facial_recognition_openai_clip.py \
    --reference-dir ./path/to/reference_images \
    --gallery-dir ./path/to/suspect_gallery \
    --output-dir ./path/to/output_dir \
    --thresholds 0.8 0.9 1.0 \
    --use-cache \
    --use-weighted \
    --crop-face-vits \
    --use-individual-refs \
    --plot-results
```

## Notes
- Embeddings saved in `embedding_cache/`
- Adjust similarity weights as needed
