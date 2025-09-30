# tests/unit/vision/test_arcface_cache_path.py
import fait.vision.facial_recognition.models.arcface as arc

class DummyFS:
    def embed(self, *_a, **_k): return None

def test_arcface_cache_dir(tmp_paths, monkeypatch):
    # make arcface module use our tmp paths
    monkeypatch.setattr(arc, "get_paths", lambda: tmp_paths)
    emb = arc.ArcFaceEmbedder(embed_cache_dir=None, face_service=DummyFS())
    base = emb._cache_base("some/dir/image.jpg")
    assert str(tmp_paths.embeddings_cache) in base
