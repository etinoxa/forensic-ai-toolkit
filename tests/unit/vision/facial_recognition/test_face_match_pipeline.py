# tests/unit/vision/test_face_match_pipeline.py
import os, shutil
from pathlib import Path
from PIL import Image
import numpy as np
import fait.vision.pipelines.facial_recognition_pipeline as fm

def _mk_png(path):
    Image.new("RGB", (8, 8), (200, 0, 0)).save(path)

class FakeEmbedder:
    def name(self): return "arcface(fake)"

    def embed_image(self, image_path: str, use_cache: bool = True):
        from pathlib import Path
        p = str(image_path).replace("\\", "/")
        if "/ref/" in p or p.endswith("/ref") or p.endswith("/ref/r1.png"):
            v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif Path(image_path).name.startswith("hit"):
            v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return v / np.linalg.norm(v)

    # NOTE: accept use_cache to mirror real embedder signature
    def mean_embedding(self, folder: str, use_cache: bool = True) -> np.ndarray:
        from pathlib import Path
        embs = []
        for f in Path(folder).iterdir():
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
                e = self.embed_image(str(f), use_cache=use_cache)
                if e is not None:
                    embs.append(e)
        m = np.mean(np.stack(embs, axis=0), axis=0)
        return m / np.linalg.norm(m)

def test_face_match_smoke(monkeypatch, tmp_paths, tmp_path):
    monkeypatch.setattr(fm, "get_paths", lambda: tmp_paths)

    ref = tmp_path / "ref"; ref.mkdir()
    gal = tmp_path / "gal"; gal.mkdir()
    _mk_png(ref / "r1.png")
    for name in ["hit1.png","miss1.png","hit2.png"]:
        _mk_png(gal / name)

    # reference mean embedding (we’ll pretend is [1,0,0])
    # achieve that by patching the function fm.get_embedding or computing directly:

    res = fm.run_facial_recognition(
        embedder=FakeEmbedder(),
        reference_dir=str(ref),
        gallery_dir=str(gal),
        thresholds=[0.1, 0.5],      # euclidean: smaller = closer
        metric="euclidean",
        plot_results=False,
    )

    assert res["processed"] == 3
    # hits have distance 0 to reference → should be <= 0.1
    assert res["matches_per_threshold"][0.1] >= 2
    assert Path(res["report_path"]).exists()
