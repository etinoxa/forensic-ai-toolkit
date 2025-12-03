# fait/ingestion/archive_ingestion.py
"""
Chip-off archive → extract → classify by content.

Pipeline:
1. Take a chip-off image archive (zip/7z/rar/tar/…).
2. Extract to EXTRACT_DIR.
3. Classify all extracted files by *content* using:
   - python-magic (libmagic)
   - Hachoir
   - Extension fallback
4. Sort into: SORTED_DIR/<category>/<extension>/*.ext
   e.g.  SORTED_DIR/documents/pdf/*.pdf
         SORTED_DIR/images/jpg/*.jpg

Dependencies (install as needed):

    pip install python-magic-bin hachoir py7zr rarfile   # Windows
    # or on Linux/macOS:
    pip install python-magic hachoir py7zr rarfile
"""

import logging
import shutil
import zipfile
import tarfile
from pathlib import Path
import mimetypes

# ---- third-party imports (optional but recommended) ----
try:
    import magic  # python-magic
    HAS_MAGIC = True
except ImportError:
    magic = None
    HAS_MAGIC = False

try:
    from hachoir.parser import createParser
    from hachoir.metadata import extractMetadata
    HAS_HACHOIR = True
except ImportError:
    createParser = None
    extractMetadata = None
    HAS_HACHOIR = False

try:
    import py7zr
    HAS_PY7ZR = True
except ImportError:
    py7zr = None
    HAS_PY7ZR = False

try:
    import rarfile
    HAS_RARFILE = True
except ImportError:
    rarfile = None
    HAS_RARFILE = False


log = logging.getLogger("chipoff_classifier")

# ===================== CONFIG (HARD-CODED) =====================

# Path to the chip-off archive (zip, 7z, rar, tar, etc.)
ARCHIVE_PATH = Path(r"D:\Datasets\Android_Dataset\Google Pixel 3.tar")

# Where to extract the archive
EXTRACT_DIR = Path(r"D:\Datasets\Android_Dataset\extracted")

# Where to put the sorted output
SORTED_DIR = Path(r"D:\Datasets\Android_Dataset\sorted")

# Forensic-friendly: copy instead of move
MOVE_FILES = False

# ===============================================================

# ---------- Archive extraction ----------

def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """
    Extract `archive_path` into `extract_dir` based on its extension.
    Supports: zip, 7z, rar, tar, tar.gz, tar.bz2, etc.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    ext = archive_path.suffix.lower()

    log.info("Extracting archive: %s -> %s", archive_path, extract_dir)

    if ext == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        return

    if ext in {".tar", ".gz", ".bz2", ".xz", ".tgz", ".tbz", ".txz"} or archive_path.name.endswith(
        (".tar.gz", ".tar.bz2", ".tar.xz")
    ):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
        return

    if ext == ".7z":
        if not HAS_PY7ZR:
            raise RuntimeError("py7zr not installed; cannot extract .7z archives.")
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extractall(path=extract_dir)
        return

    if ext == ".rar":
        if not HAS_RARFILE:
            raise RuntimeError("rarfile not installed; cannot extract .rar archives.")
        rf = rarfile.RarFile(archive_path)
        rf.extractall(extract_dir)
        rf.close()
        return

    raise ValueError(f"Unsupported archive type: {archive_path}")


# ---------- Detection helpers ----------

def detect_with_magic(path: Path) -> tuple[str | None, str | None]:
    """
    Return (mime, description) from python-magic, or (None, None) if unavailable/fails.
    """
    if not HAS_MAGIC:
        return None, None

    try:
        mime = magic.from_file(str(path), mime=True)   # e.g. "image/jpeg"
        desc = magic.from_file(str(path))              # e.g. "JPEG image data..."
        return mime, desc
    except Exception as e:
        log.debug("magic_failed: %s (%s)", path, e)
        return None, None


def detect_with_hachoir(path: Path) -> tuple[str | None, str | None]:
    """
    Use Hachoir to get (format, description) or (None, None).
    More tolerant of partially damaged files than magic.
    """
    if not HAS_HACHOIR:
        return None, None

    try:
        parser = createParser(str(path))
        if not parser:
            return None, None

        metadata = extractMetadata(parser)
        if metadata:
            fmt = metadata.format  # e.g. "JPEG picture"
            txt = str(metadata)    # full metadata text
            return fmt, txt
        else:
            # Fallback: parser class name gives some hint
            return parser.__class__.__name__, None
    except Exception as e:
        log.debug("hachoir_failed: %s (%s)", path, e)
        return None, None


def guess_category(
    mime: str | None,
    desc: str | None,
    hachoir_fmt: str | None,
    hachoir_desc: str | None,
) -> str:
    """
    Decide high-level category:
      images, videos, audio, documents, archives, other, unknown.
    Priority: MIME -> Hachoir strings -> "other"/"unknown".
    """

    # 1) MIME-based classification
    if mime:
        if mime.startswith("image/"):
            return "images"
        if mime.startswith("video/"):
            return "videos"
        if mime.startswith("audio/"):
            return "audio"
        if mime in {
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "text/plain",
            "text/html",
            "text/rtf",
        } or mime.startswith("text/"):
            return "documents"
        if mime in {
            "application/zip",
            "application/x-rar-compressed",
            "application/x-7z-compressed",
            "application/x-tar",
            "application/gzip",
            "application/x-bzip2",
        }:
            return "archives"

    # 2) Hachoir / magic description classification
    for s in (hachoir_fmt, hachoir_desc, desc):
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in ("jpeg", "jpg", "png", "gif", "bitmap", "tiff", "picture", "image")):
            return "images"
        if any(k in low for k in ("mpeg-4", "mp4", "video", "avi", "matroska", "mkv", "quicktime")):
            return "videos"
        if any(k in low for k in ("audio", "mp3", "flac", "wav", "aac", "ogg", "opus")):
            return "audio"
        if any(k in low for k in ("pdf", "document", "word", "excel", "powerpoint", "rtf", "text")):
            return "documents"
        if any(k in low for k in ("zip archive", "rar archive", "7-zip", "tar archive", "compressed data")):
            return "archives"

    # 3) Fallback: other vs unknown
    return "other"


def guess_extension(path: Path, mime: str | None) -> str:
    """
    Determine the extension bucket (no leading dot).
    Order:
      1) Use existing extension if present.
      2) Use mimetypes.guess_extension(mime).
      3) "noext" if nothing else.
    """
    ext = path.suffix.lower()
    if ext:
        return ext.lstrip(".")

    if mime:
        guessed = mimetypes.guess_extension(mime)
        if guessed:
            return guessed.lstrip(".")

    return "noext"


def classify_file(path: Path) -> tuple[str, str, dict]:
    """
    Fully classify a single file.
    Returns: (category, extension_bucket, debug_info_dict)
    """
    mime, desc = detect_with_magic(path)
    hfmt, hdesc = detect_with_hachoir(path)

    category = guess_category(mime, desc, hfmt, hdesc)
    ext_bucket = guess_extension(path, mime)

    debug_info = {
        "mime": mime,
        "magic_desc": desc,
        "hachoir_format": hfmt,
        "hachoir_desc": hdesc,
        "category": category,
        "ext_bucket": ext_bucket,
    }
    return category, ext_bucket, debug_info


# ---------- Sorting logic ----------

def process_tree(src: Path, dest: Path, move: bool = False):
    """
    Walk `src`, classify each file, and copy/move into:
        dest/<category>/<ext_bucket>/<filename>
    Example:
        dest/documents/pdf/*.pdf
        dest/images/jpg/*.jpg
    """
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")

    dest.mkdir(parents=True, exist_ok=True)

    action = "MOVE" if move else "COPY"
    log.info("Starting classification: %s -> %s (%s)", src, dest, action)

    total = 0
    by_category: dict[str, int] = {}

    for f in src.rglob("*"):
        if not f.is_file():
            continue

        total += 1
        category, ext_bucket, info = classify_file(f)
        by_category[category] = by_category.get(category, 0) + 1

        target_dir = dest / category / ext_bucket
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f.name

        # Handle name collisions: append counter
        if target_path.exists():
            base = target_path.stem
            ext = target_path.suffix
            i = 1
            while True:
                candidate = target_dir / f"{base}__{i}{ext}"
                if not candidate.exists():
                    target_path = candidate
                    break
                i += 1

        log.debug(
            "File: %s -> %s (cat=%s, ext_bucket=%s, mime=%s)",
            f, target_path, category, ext_bucket, info["mime"],
        )

        if move:
            shutil.move(str(f), str(target_path))
        else:
            shutil.copy2(str(f), str(target_path))

    # Summary
    log.info("Done. Processed %d files.", total)
    for cat, count in sorted(by_category.items(), key=lambda x: x[0]):
        log.info("  %s: %d", cat, count)


# ---------- Main orchestration ----------

def run_pipeline():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not HAS_MAGIC:
        log.warning("python-magic not available; classification may be less accurate.")
    if not HAS_HACHOIR:
        log.warning("hachoir not available; damaged files may be harder to classify.")
    if ARCHIVE_PATH and not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Archive not found: {ARCHIVE_PATH}")

    # 1) Extract chip-off archive
    extract_archive(ARCHIVE_PATH, EXTRACT_DIR)

    # 2) Classify all extracted files into SORTED_DIR
    process_tree(EXTRACT_DIR, SORTED_DIR, move=MOVE_FILES)


if __name__ == "__main__":
    run_pipeline()
