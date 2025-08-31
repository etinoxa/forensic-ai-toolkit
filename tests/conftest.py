# tests/conftest.py
import sys, pathlib
# make src importable for tests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import types, pytest
from fait.core import paths as paths_mod  # now safe

@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    p = types.SimpleNamespace(
        repo_root=tmp_path,
        fait_root=tmp_path / ".fait",
        cache_root=tmp_path / ".fait" / "cache",
        models_cache=tmp_path / ".fait" / "cache" / "models",
        models_face_match=tmp_path / ".fait" / "cache" / "models" / "face_match",
        models_object_screen=tmp_path / ".fait" / "cache" / "models" / "object_screen",
        embeddings_cache=tmp_path / ".fait" / "cache" / "embeddings",
        outputs=tmp_path / ".fait" / "outputs",
        logs=tmp_path / ".fait" / "logs",
    )
    monkeypatch.setattr(paths_mod, "get_paths", lambda: p)
    return p
