import os, cv2, torch, shutil, argparse, pickle
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ======================= ARGUMENT PARSING =======================
parser = argparse.ArgumentParser()
parser.add_argument("--reference-dir", required=True)
parser.add_argument("--gallery-dir", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--thresholds", nargs="+", type=float, required=True)
parser.add_argument("--use-cache", action="store_true")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./model_cache"
EMBED_CACHE_DIR = "./embedding_cache"

# ======================= MODEL INITIALIZATION =======================
def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_folder(CACHE_DIR)
# Use correct ctx_id for CPU (-1). If CUDA is available, allow GPU with ctx_id=0.
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == "cuda" else ['CPUExecutionProvider']
ctx_id = 0 if DEVICE == "cuda" else -1
face_app = FaceAnalysis(name="buffalo_l", root=CACHE_DIR, providers=providers)
face_app.prepare(ctx_id=ctx_id)

# ======================= HELPER FUNCTIONS =======================
def cache_embedding_path(image_path):
    import hashlib
    ensure_folder(EMBED_CACHE_DIR)
    fname = os.path.basename(image_path)
    key = hashlib.md5(image_path.encode()).hexdigest()
    return os.path.join(EMBED_CACHE_DIR, f"{key}_{fname}.emb.pkl")

def is_image_file(path):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    return os.path.isfile(path) and path.lower().endswith(valid_exts)

def get_arcface_embedding(image_path):
    # Avoid attempting to read non-image files
    if not is_image_file(image_path):
        return None
    image = cv2.imread(image_path)
    if image is None:
        # Unreadable or corrupted image
        return None
    faces = face_app.get(image)
    if faces:
        return faces[0].embedding
    return None

def get_embedding(image_path):
    cache_file = cache_embedding_path(image_path)
    if args.use_cache and os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))
    # ... existing code ...
    arc_emb = get_arcface_embedding(image_path)
    if arc_emb is None:
        return None
    # ... existing code ...
    if args.use_cache:
        pickle.dump(arc_emb, open(cache_file, "wb"))
    return arc_emb

def compute_similarity(reference_embeddings, query_embedding):
    return max([cosine_similarity([query_embedding], [ref])[0][0] for ref in reference_embeddings])

# ======================= LOAD REFERENCE EMBEDDINGS =======================
print("[*] Extracting reference embeddings...")
ref_embeddings = []
for fname in os.listdir(args.reference_dir):
    fpath = os.path.join(args.reference_dir, fname)
    if not is_image_file(fpath):
        continue
    emb = get_embedding(fpath)
    if emb is not None:
        ref_embeddings.append(emb)

if not ref_embeddings:
    raise ValueError("No valid reference embeddings found.")

# ======================= MATCHING =======================
print("[*] Comparing gallery images...")
ensure_folder(args.output_dir)

for fname in tqdm(os.listdir(args.gallery_dir)):
    fpath = os.path.join(args.gallery_dir, fname)
    if not is_image_file(fpath):
        continue
    emb = get_embedding(fpath)
    if emb is None:
        continue
    similarity = compute_similarity(ref_embeddings, emb)
    for threshold in sorted(args.thresholds, reverse=True):
        if similarity >= threshold:
            out_path = os.path.join(args.output_dir, f"threshold_{threshold}")
            ensure_folder(out_path)
            shutil.copy(fpath, os.path.join(out_path, fname))
            break

print("[✓] Matching complete.")