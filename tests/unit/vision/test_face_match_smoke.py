import numpy as np
import fait.vision.pipelines.face_match as fm

class FakeEmbedder:
    def name(self): return "fake"
    def embed_image(self, p, use_cache=True):
        return np.ones(4, dtype=np.float32)
    def mean_embedding(self, d, use_cache=True):
        return np.ones(4, dtype=np.float32)

def test_face_match_smoke(tmp_paths, tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "get_paths", lambda: tmp_paths)
    ref = tmp_path / "ref"; gal = tmp_path / "gal"
    ref.mkdir(); gal.mkdir()
    (ref / "a.jpg").write_bytes(b"x")
    (gal / "b.jpg").write_bytes(b"x")
    out = fm.run_face_match(FakeEmbedder(), str(ref), str(gal), thresholds=[0.5], metric="euclidean", plot_results=False)
    assert out["processed"] == 1
    assert 0.5 in out["matches_per_threshold"]
