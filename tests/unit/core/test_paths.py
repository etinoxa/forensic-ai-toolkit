import os
from pathlib import Path
from fait.core.paths import get_paths, ensure_on_first_write

def test_paths_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FAIT_OUTPUTS_DIR", str(tmp_path / ".custom_out"))
    p = get_paths()
    assert p.outputs.as_posix().endswith(".custom_out")

def test_ensure_on_first_write(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()
    ensure_on_first_write(target)
    assert target.exists()
