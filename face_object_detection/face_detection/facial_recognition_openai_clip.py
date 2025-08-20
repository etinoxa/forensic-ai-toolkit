## Facial Recognition with OpenAI ViT
"""
Description:
This script matches faces in a gallery against a set of reference images using ViT embeddings.
"""

'''
Example usage:
python facial_recognition_openai_clip.py \
    --reference-dir ./path/to/reference_images --gallery-dir ./path/to/suspect_gallery --output-dir ./path/to/output_dir \
    --thresholds 0.8 0.9 1.0 --use-individual-refs --plot-results --use-cache
'''


import os, torch, shutil, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# ======================= ARGUMENT PARSING =======================
parser = argparse.ArgumentParser(description="CLIP-only Face Recognition")
parser.add_argument("--reference-dir", required=True, help="Directory containing reference images")
parser.add_argument("--gallery-dir", required=True, help="Directory containing gallery images to search")
parser.add_argument("--output-dir", required=True, help="Output directory for matches")
parser.add_argument("--thresholds", nargs="+", type=float, required=True,
                    help="Similarity thresholds (0.0-1.0, higher = more restrictive)")
parser.add_argument("--use-cache", action="store_true", help="Use embedding cache")
parser.add_argument("--plot-results", action="store_true", help="Generate similarity plots")
parser.add_argument("--use-individual-refs", action="store_true",
                    help="Use individual references like notebook (recommended)")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./model_cache"
EMBED_CACHE_DIR = "./embedding_cache"


# ======================= MODEL INITIALIZATION =======================
def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


ensure_folder(CACHE_DIR)
ensure_folder(EMBED_CACHE_DIR)

# Initialize CLIP
print("Loading CLIP model...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=CACHE_DIR)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=CACHE_DIR).to(DEVICE)
clip_model.eval()

print(f"✅ CLIP model initialized on {DEVICE}")


# ======================= HELPER FUNCTIONS =======================
def cache_embedding_path(image_path):
    import hashlib
    fname = os.path.basename(image_path)
    key = hashlib.md5(f"{image_path}_clip".encode()).hexdigest()
    return os.path.join(EMBED_CACHE_DIR, f"{key}_{fname}.emb.pkl")


def is_image_file(path):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    return os.path.isfile(path) and path.lower().endswith(valid_exts)


def get_clip_embedding(image_path):
    """Extract CLIP embedding from image (like the notebook)"""
    try:
        # Use PIL directly for better CLIP compatibility
        image = Image.open(image_path).convert('RGB')
        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # Use get_image_features for optimal CLIP performance
            embedding = clip_model.get_image_features(**inputs)
            # Normalize embedding
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.cpu().numpy().flatten()

        return embedding

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def get_embedding(image_path):
    """Get embedding with caching support"""
    cache_file = cache_embedding_path(image_path)
    if args.use_cache and os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))

    embedding = get_clip_embedding(image_path)
    if embedding is not None and args.use_cache:
        pickle.dump(embedding, open(cache_file, "wb"))

    return embedding


def load_reference_embeddings(folder_path):
    """Load reference embeddings (individual or mean based on setting)"""
    embeddings = []
    processed_files = []

    print(f"[*] Processing reference images from {folder_path}...")
    for fname in tqdm(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if not is_image_file(fpath):
            continue
        emb = get_embedding(fpath)
        if emb is not None:
            embeddings.append(emb)
            processed_files.append(fname)
        else:
            print(f"[!] Skipped {fname} (processing error)")

    if not embeddings:
        raise ValueError("No valid reference embeddings found.")

    if args.use_individual_refs:
        # Keep individual embeddings like the notebook
        reference_embeddings = np.vstack(embeddings)
        print(f"✅ Loaded {len(embeddings)} individual reference embeddings:")
    else:
        # Compute mean embedding
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        reference_embeddings = mean_emb
        print(f"✅ Mean embedding computed from {len(embeddings)} reference images:")

    for fname in processed_files:
        print(f"   - {fname}")

    return reference_embeddings


def compute_similarity(reference_embeddings, query_embedding):
    """Compute similarity like the notebook"""
    if args.use_individual_refs:
        # Multiple references - compute similarities and take max (like notebook)
        if len(reference_embeddings.shape) == 1:
            # Single reference
            similarities = cosine_similarity([query_embedding], [reference_embeddings])[0]
        else:
            # Multiple references
            similarities = cosine_similarity([query_embedding], reference_embeddings)[0]
        max_similarity = np.max(similarities)
    else:
        # Single mean reference
        max_similarity = cosine_similarity([reference_embeddings], [query_embedding])[0][0]

    return max_similarity

def save_final_report_to_file(args, processed_count, elapsed_time, match_counts, similarities):
    """Save the final report to a text file"""
    report_path = os.path.join(args.output_dir, "matching_report.txt")
    with open(report_path, 'w') as f:
        f.write("=== FINAL REPORT ===\n")
        f.write(f"Model type: CLIP\n")
        f.write(f"Similarity metric: Cosine Similarity\n")
        f.write(f"Total gallery images processed: {processed_count}\n")
        f.write(f"Processing time: {elapsed_time:.2f} seconds\n")
        f.write(f"\nMatches per threshold:\n")

        for threshold in sorted(args.thresholds):
            f.write(f"  Threshold {threshold}: {match_counts[threshold]} matches\n")

        # Show closest matches (highest similarities for CLIP)
        if similarities:
            f.write(f"\nBest matches (top 10 - highest similarities):\n")
            sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            for i, (fname, sim) in enumerate(sorted_similarities[:10]):
                f.write(f"  {i + 1:2d}. {fname:<30} (similarity: {sim:.4f})\n")

        f.write("=" * 50 + "\n")

    print(f"Final report saved to: {report_path}")

def plot_similarities(similarities, thresholds, output_dir):
    """Generate similarity visualization plots"""
    if not args.plot_results or not similarities:
        return

    # Sort by similarity (highest first - best matches)
    similarities_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in similarities_sorted[:20]]
    sims = [x[1] for x in similarities_sorted[:20]]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(labels)), sims, color='lightgreen')

    # Add threshold lines
    for threshold in thresholds:
        plt.axvline(x=threshold, linestyle='--', alpha=0.7,
                    label=f'Threshold {threshold}')

    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Cosine Similarity')
    plt.title(f'CLIP Similarity Analysis\n(Showing top 20 matches)')
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'clip_similarity_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Similarity plot saved to {plot_path}")


# ======================= MAIN PROCESSING =======================
def main():
    start_time = time.time()
    print(f"\n🚀 CLIP face recognition started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Reference mode: {'Individual' if args.use_individual_refs else 'Mean'} (notebook uses individual)")
    print(f"Thresholds: {args.thresholds} (higher = more restrictive)")

    # ======================= LOAD REFERENCE EMBEDDINGS =======================
    print("\n[*] Loading reference embeddings...")
    reference_embeddings = load_reference_embeddings(args.reference_dir)

    # ======================= PROCESS GALLERY =======================
    print(f"\n[*] Processing gallery images from {args.gallery_dir}...")
    ensure_folder(args.output_dir)

    # Create threshold directories
    for threshold in args.thresholds:
        threshold_dir = os.path.join(args.output_dir, f"threshold_{threshold}")
        ensure_folder(threshold_dir)

    similarities = []
    processed_count = 0
    match_counts = {threshold: 0 for threshold in args.thresholds}

    gallery_files = [f for f in os.listdir(args.gallery_dir) if is_image_file(os.path.join(args.gallery_dir, f))]

    for fname in tqdm(gallery_files, desc="Matching images"):
        fpath = os.path.join(args.gallery_dir, fname)
        emb = get_embedding(fpath)
        if emb is None:
            continue

        similarity = compute_similarity(reference_embeddings, emb)
        similarities.append((fname, similarity))
        processed_count += 1

        # Process each threshold (similarity: higher threshold = more restrictive)
        for threshold in args.thresholds:
            if similarity >= threshold:
                out_path = os.path.join(args.output_dir, f"threshold_{threshold}")
                shutil.copy(fpath, os.path.join(out_path, fname))
                match_counts[threshold] += 1

    # ======================= RESULTS AND VISUALIZATION =======================
    elapsed_time = time.time() - start_time

    # Generate similarity plot
    plot_similarities(similarities, args.thresholds, args.output_dir)

    # Print detailed results
    print(f"\n=== CLIP RESULTS ===")
    print(f"Reference mode: {'Individual references' if args.use_individual_refs else 'Mean reference'}")
    print(f"Total gallery images processed: {processed_count}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"\nMatches per threshold (higher threshold = fewer matches):")
    for threshold in sorted(args.thresholds, reverse=True):
        print(f"  Threshold {threshold}: {match_counts[threshold]} matches")

    # Show best matches (highest similarities)
    if similarities:
        print(f"\nBest matches (top 10 - highest similarities):")
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        for i, (fname, sim) in enumerate(sorted_similarities[:10]):
            print(f"  {i + 1:2d}. {fname:<30} (similarity: {sim:.4f})")

    print("=" * 50)
    print("[✅] CLIP recognition complete!")

    # Save the final report to file
    save_final_report_to_file(args, processed_count, elapsed_time, match_counts, similarities)


if __name__ == "__main__":
    main()