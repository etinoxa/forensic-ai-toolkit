## Face Recognition with ArcFace
"""
Description:
This script matches faces in a gallery against a set of reference images using ArcFace embeddings.
"""

'''
Example usage:
# ArcFace only
python facial_recognition_arcface.py \
    --reference-dir ./path/to/reference_images --gallery-dir ./path/to/suspect_gallery --output-dir ./path/to/output_dir  \
    --thresholds 0.8 0.9 1.0 --distance-metric euclidean --plot-results --use-cache
'''


import os, cv2, torch, shutil, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ======================= ARGUMENT PARSING =======================
parser = argparse.ArgumentParser(description="ArcFace-only Face Recognition")
parser.add_argument("--reference-dir", required=True, help="Directory containing reference images")
parser.add_argument("--gallery-dir", required=True, help="Directory containing gallery images to search")
parser.add_argument("--output-dir", required=True, help="Output directory for matches")
parser.add_argument("--thresholds", nargs="+", type=float, required=True,
                    help="Distance thresholds (lower = more restrictive)")
parser.add_argument("--use-cache", action="store_true", help="Use embedding cache")
parser.add_argument("--distance-metric", choices=["euclidean", "cosine"], default="euclidean", help="Distance metric")
parser.add_argument("--plot-results", action="store_true", help="Generate distance plots")
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

# Initialize ArcFace
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == "cuda" else ['CPUExecutionProvider']
ctx_id = 0 if DEVICE == "cuda" else -1
face_app = FaceAnalysis(name="buffalo_l", root=CACHE_DIR, providers=providers)
face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

print(f"✅ ArcFace model initialized on {DEVICE}")


# ======================= HELPER FUNCTIONS =======================
def cache_embedding_path(image_path):
    import hashlib
    fname = os.path.basename(image_path)
    key = hashlib.md5(f"{image_path}_arcface".encode()).hexdigest()
    return os.path.join(EMBED_CACHE_DIR, f"{key}_{fname}.emb.pkl")


def is_image_file(path):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    return os.path.isfile(path) and path.lower().endswith(valid_exts)


def get_arcface_embedding(image_path):
    """Extract and normalize ArcFace embedding from image"""
    if not is_image_file(image_path):
        return None

    image = cv2.imread(image_path)
    if image is None:
        return None

    faces = face_app.get(image)
    if faces:
        embedding = faces[0].embedding
        # Normalize the embedding (crucial for consistent comparisons)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    return None


def get_embedding(image_path):
    """Get embedding with caching support"""
    cache_file = cache_embedding_path(image_path)
    if args.use_cache and os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))

    embedding = get_arcface_embedding(image_path)
    if embedding is not None and args.use_cache:
        pickle.dump(embedding, open(cache_file, "wb"))

    return embedding


def get_mean_embedding_from_folder(folder_path):
    """Compute mean embedding from multiple reference images"""
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
            print(f"[!] Skipped {fname} (no face detected)")

    if not embeddings:
        raise ValueError("No valid reference embeddings found.")

    # Compute mean embedding and normalize
    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)

    print(f"✅ Mean embedding computed from {len(embeddings)} reference images:")
    for fname in processed_files:
        print(f"   - {fname}")

    return mean_emb


def compute_distance(reference_embedding, query_embedding, metric="euclidean"):
    """Compute distance between normalized embeddings"""
    if metric == "euclidean":
        return np.linalg.norm(reference_embedding - query_embedding)
    elif metric == "cosine":
        similarity = cosine_similarity([reference_embedding], [query_embedding])[0][0]
        return 1.0 - similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")

def save_final_report_to_file(args, processed_count, elapsed_time, match_counts, distances):
    """Save the final report to a text file"""
    report_path = os.path.join(args.output_dir, "matching_report.txt")
    with open(report_path, 'w') as f:
        f.write("=== FINAL REPORT ===\n")
        f.write(f"Model type: ArcFace\n")
        f.write(f"Distance metric: {args.distance_metric}\n")
        f.write(f"Total gallery images processed: {processed_count}\n")
        f.write(f"Processing time: {elapsed_time:.2f} seconds\n")
        f.write(f"\nMatches per threshold:\n")

        for threshold in sorted(args.thresholds):
            f.write(f"  Threshold {threshold}: {match_counts[threshold]} matches\n")

        # Show closest matches
        if distances:
            f.write(f"\nClosest matches (top 10):\n")
            sorted_distances = sorted(distances, key=lambda x: x[1])
            for i, (fname, dist) in enumerate(sorted_distances[:10]):
                f.write(f"  {i + 1:2d}. {fname:<30} (distance: {dist:.4f})\n")

        f.write("=" * 50 + "\n")

    print(f"Final report saved to: {report_path}")

def plot_distances(distances, thresholds, output_dir):
    """Generate distance visualization plots"""
    if not args.plot_results or not distances:
        return

    distances_sorted = sorted(distances, key=lambda x: x[1])
    labels = [x[0] for x in distances_sorted[:20]]
    dists = [x[1] for x in distances_sorted[:20]]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(labels)), dists, color='lightcoral')

    # Add threshold lines
    for threshold in thresholds:
        plt.axvline(x=threshold, linestyle='--', alpha=0.7,
                    label=f'Threshold {threshold}')

    plt.yticks(range(len(labels)), labels)
    plt.xlabel(f'{args.distance_metric.title()} Distance')
    plt.title(f'ArcFace Distance Analysis\nMetric: {args.distance_metric}\n(Showing closest 20 matches)')
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'arcface_distance_plot_{args.distance_metric}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Distance plot saved to {plot_path}")


# ======================= MAIN PROCESSING =======================
def main():
    start_time = time.time()
    print(f"\n🚀 ArcFace face recognition started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Distance metric: {args.distance_metric}")
    print(f"Thresholds: {args.thresholds} (lower = more restrictive)")

    # ======================= LOAD REFERENCE EMBEDDING =======================
    print("\n[*] Computing reference mean embedding...")
    reference_embedding = get_mean_embedding_from_folder(args.reference_dir)

    # ======================= PROCESS GALLERY =======================
    print(f"\n[*] Processing gallery images from {args.gallery_dir}...")
    ensure_folder(args.output_dir)

    # Create threshold directories
    for threshold in args.thresholds:
        threshold_dir = os.path.join(args.output_dir, f"threshold_{threshold}")
        ensure_folder(threshold_dir)

    distances = []
    processed_count = 0
    match_counts = {threshold: 0 for threshold in args.thresholds}

    gallery_files = [f for f in os.listdir(args.gallery_dir) if is_image_file(os.path.join(args.gallery_dir, f))]

    for fname in tqdm(gallery_files, desc="Matching faces"):
        fpath = os.path.join(args.gallery_dir, fname)
        emb = get_embedding(fpath)
        if emb is None:
            continue

        distance = compute_distance(reference_embedding, emb, args.distance_metric)
        distances.append((fname, distance))
        processed_count += 1

        # Process each threshold (distance: lower threshold = more restrictive)
        for threshold in args.thresholds:
            if distance <= threshold:
                out_path = os.path.join(args.output_dir, f"threshold_{threshold}")
                shutil.copy(fpath, os.path.join(out_path, fname))
                match_counts[threshold] += 1

    # ======================= RESULTS AND VISUALIZATION =======================
    elapsed_time = time.time() - start_time

    # Generate distance plot
    plot_distances(distances, args.thresholds, args.output_dir)

    # Print detailed results
    print(f"\n=== ARCFACE RESULTS ===")
    print(f"Distance metric: {args.distance_metric}")
    print(f"Total gallery images processed: {processed_count}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"\nMatches per threshold (lower threshold = fewer matches):")
    for threshold in sorted(args.thresholds):
        print(f"  Threshold {threshold}: {match_counts[threshold]} matches")

    # Show closest matches
    if distances:
        print(f"\nClosest matches (top 10):")
        sorted_distances = sorted(distances, key=lambda x: x[1])
        for i, (fname, dist) in enumerate(sorted_distances[:10]):
            print(f"  {i + 1:2d}. {fname:<30} (distance: {dist:.4f})")

    print("=" * 50)
    print("[✅] ArcFace recognition complete!")

    # Save the final report to file
    save_final_report_to_file(args, processed_count, elapsed_time, match_counts, distances)

if __name__ == "__main__":
    main()