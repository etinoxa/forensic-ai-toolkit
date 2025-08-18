"""Face Matching Script
This script matches faces in a gallery against a set of reference images using ViT and ArcFace embeddings.
It supports caching, weighted embeddings, and face cropping for ViT.
It requires the `transformers`, `insightface`, and `scikit-learn` libraries.
"""
'''
Example usage:
python face_match.py \
    --reference-dir path/to/reference_images \
    --gallery-dir path/to/reference_images \
    --output-dir path/to/output_dir \
    --thresholds 0.7 0.75 0.8 \
    --use-cache \
    --use-weighted \
    --crop-face-vits
'''

'''
Example usage:
python face_match.py \
    --reference-dir datasets/images/face/reference_images \
    --gallery-dir datasets/images/face/gallery \
    --output-dir datasets/images/face/matched_images\
    --thresholds 0.7 0.75 0.8 \
    --use-cache \
    --use-weighted \
    --crop-face-vits
'''

'''
python face_match.py --reference-dir ../../datasets/images/face/reference_images --gallery-dir ../../datasets/images/face/gallery --output-dir ../../datasets/images/face/matched_images --thresholds 0.8 0.9 1.0 --use-cache --use-weighted --crop-face-vits


python arcface.py --reference-dir ../../datasets/images/face/reference_images --gallery-dir ../../datasets/images/face/gallery --output-dir ../../datasets/images/face/matched_images --thresholds 0.8 0.9 1.0 --use-cache
'''

import os, cv2, torch, shutil, argparse, pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ======================= ARGUMENT PARSING =======================
parser = argparse.ArgumentParser()
parser.add_argument("--reference-dir", required=True)
parser.add_argument("--gallery-dir", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--thresholds", nargs="+", type=float, required=True)
parser.add_argument("--use-cache", action="store_true")
parser.add_argument("--use-weighted", action="store_true")
parser.add_argument("--crop-face-vits", action="store_true")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./model_cache"
EMBED_CACHE_DIR = "./embedding_cache"

# ======================= MODEL INITIALIZATION =======================
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=CACHE_DIR)
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=CACHE_DIR).to(DEVICE)
vit_model.eval()

face_app = FaceAnalysis(name="buffalo_l", root=CACHE_DIR, providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# ======================= HELPER FUNCTIONS =======================
def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def cache_embedding_path(image_path):
    import hashlib
    ensure_folder(EMBED_CACHE_DIR)
    fname = os.path.basename(image_path)
    key = hashlib.md5(image_path.encode()).hexdigest()
    return os.path.join(EMBED_CACHE_DIR, f"{key}_{fname}.emb.pkl")

def is_image_file(path):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    return os.path.isfile(path) and path.lower().endswith(valid_exts)

def get_vit_embedding(image_path, crop_face=False):
    image = cv2.imread(image_path)
    if image is None:
        # Unreadable or non-image file
        return None
    if crop_face:
        faces = face_app.get(image)
        if not faces:
            return None
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        image = image[y1:y2, x1:x2]
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((224, 224))
    inputs = vit_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = vit_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

def get_arcface_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    faces = face_app.get(image)
    if faces:
        return faces[0].embedding
    return None

def get_combined_embedding(image_path):
    cache_file = cache_embedding_path(image_path)
    if args.use_cache and os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))

    vit_emb = get_vit_embedding(image_path, crop_face=args.crop_face_vits)
    arc_emb = get_arcface_embedding(image_path)

    if vit_emb is None or arc_emb is None:
        return None

    if args.use_weighted:
        result = {"vit": vit_emb, "arc": arc_emb}
    else:
        result = np.concatenate((vit_emb, arc_emb), axis=0)

    if args.use_cache:
        pickle.dump(result, open(cache_file, "wb"))

    return result

def compute_similarity(reference_embeddings, query_embedding):
    if args.use_weighted:
        vit_scores = [cosine_similarity([query_embedding["vit"]], [ref["vit"]])[0][0] for ref in reference_embeddings]
        arc_scores = [cosine_similarity([query_embedding["arc"]], [ref["arc"]])[0][0] for ref in reference_embeddings]
        return max([0.4 * v + 0.6 * a for v, a in zip(vit_scores, arc_scores)])
    else:
        return max([cosine_similarity([query_embedding], [ref])[0][0] for ref in reference_embeddings])

# ======================= LOAD REFERENCE EMBEDDINGS =======================
print("[*] Extracting reference embeddings...")
ref_embeddings = []
for fname in os.listdir(args.reference_dir):
    fpath = os.path.join(args.reference_dir, fname)
    if not is_image_file(fpath):
        continue
    emb = get_combined_embedding(fpath)
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
    emb = get_combined_embedding(fpath)
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