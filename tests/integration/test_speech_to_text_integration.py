"""
Integration tests for FAIT Speech-to-Text CLI and service.
These tests simulate the CLI execution with a fake model to verify
that transcription and output saving work correctly.
"""

import sys
import json
import pytest
import types
from pathlib import Path

# ==========================================================
# 🧩 Ensure correct import paths
# ==========================================================
ROOT = Path(__file__).resolve().parents[2]
APPS_DIR = ROOT / "apps"

# Add both repo root and apps directory to sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

# ==========================================================
# 🧪 Fake Whisper Model for isolation
# ==========================================================
class _FakeWhisperModel:
    """Mock Faster-Whisper model to simulate transcriptions."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.transcribe_calls = []

    def transcribe(self, audio_path, beam_size=5, vad_filter=True, task="transcribe", **extra_kwargs):
        """
        Simulate transcribe/translate for Faster-Whisper.
        Accepts the same args as the real model to avoid breaking.
        """
        # Track the call for debug
        self.transcribe_calls.append({
            "audio_path": audio_path,
            "task": task,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "extra": extra_kwargs,
        })

        # Mock metadata object
        class Info:
            duration = 5.7
            language = "ja" if task == "transcribe" else "en"
            language_probability = 1.0

        # Mock segment result
        class Segment:
            def __init__(self, text):
                self.text = text

        if task == "translate":
            segments = [Segment("Translation: It’s not impossible to recover.")]
        else:
            segments = [Segment("長まで続いているとするとなおりっこないんですか そんなことはありません")]

        return segments, Info()


# ==========================================================
# 🧪 Fixtures
# ==========================================================
@pytest.fixture
def tmp_repo_with_config(tmp_path):
    """Simulate a FAIT repo root with a minimal config.yaml and .fait dirs."""
    repo_root = tmp_path / "repo_root_stt_int"
    (repo_root / ".fait" / "logs").mkdir(parents=True, exist_ok=True)
    (repo_root / ".fait" / "models" / "audio" / "speech_to_text").mkdir(parents=True, exist_ok=True)

    config_file = repo_root / "config.yaml"
    config_file.write_text("""
audio:
  speech_to_text:
    output_dir: .fait/outputs/audio/speech_to_text
""")

    from fait.core import paths

    def fake_get_paths():
        return types.SimpleNamespace(repo_root=repo_root)

    return types.SimpleNamespace(repo_root=repo_root, fake_get_paths=fake_get_paths)


@pytest.fixture
def tiny_audio_dataset(tmp_path):
    """Create a minimal fake dataset with one WAV file."""
    data_dir = tmp_path / "speech_to_text_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Fake audio file
    (data_dir / "meian_0000.wav").write_bytes(b"fake_audio_bytes")

    return data_dir


# ==========================================================
# 🧪 Integration Tests
# ==========================================================
class TestSpeechToTextCLIIntegration:
    """Integration tests for Speech-to-Text CLI behavior."""

    @pytest.mark.integration
    def test_cli_end_to_end_saves_output(self, monkeypatch, tmp_repo_with_config, tiny_audio_dataset, tmp_path):
        """
        Run the CLI 'apps/speech_to_text_cli.py' in-process:
          - Patch Faster-Whisper model to fake one
          - Point --audio to tiny dataset
          - Provide --output-dir so it saves summary
        Then assert the transcript file is created and contains expected markers.
        """
        from fait.core import paths
        monkeypatch.setattr(paths, "get_paths", tmp_repo_with_config.fake_get_paths)
        monkeypatch.setattr("faster_whisper.WhisperModel", _FakeWhisperModel)

        sys.argv = [
            "speech_to_text_cli.py",
            "--audio", str(tiny_audio_dataset),
            "--engine", "whisper-faster",
            "--output-dir", str(tmp_path / "out_dir"),
        ]

        from speech_to_text_cli import main as cli_main
        cli_main()

        out_dir = tmp_path / "out_dir"
        txt_files = list(out_dir.glob("*.txt"))
        assert txt_files, "❌ Expected at least one transcription output file."

        content = txt_files[0].read_text(encoding="utf-8")
        assert "TRANSCRIPTION SUMMARY" in content
        assert "Detected Language" in content
        assert "長まで続いているとする" in content


    @pytest.mark.integration
    def test_cli_translation_disabled_by_default(self, monkeypatch, tmp_repo_with_config, tiny_audio_dataset, tmp_path):
        """Verify that translation is skipped when --translate is not provided."""
        from fait.core import paths
        monkeypatch.setattr(paths, "get_paths", tmp_repo_with_config.fake_get_paths)
        monkeypatch.setattr("faster_whisper.WhisperModel", _FakeWhisperModel)

        sys.argv = [
            "speech_to_text_cli.py",
            "--audio", str(tiny_audio_dataset),
            "--engine", "whisper-faster",
            "--output-dir", str(tmp_path / "out_dir"),
        ]

        from speech_to_text_cli import main as cli_main
        cli_main()

        out_file = next((tmp_path / "out_dir").glob("*.txt"))
        content = out_file.read_text(encoding="utf-8")
        assert "🌐 Translation:" not in content


    @pytest.mark.integration
    def test_cli_with_translate_flag(self, monkeypatch, tmp_repo_with_config, tiny_audio_dataset, tmp_path):
        """Verify translation flag runs successfully with fake model."""
        from fait.core import paths
        monkeypatch.setattr(paths, "get_paths", tmp_repo_with_config.fake_get_paths)
        monkeypatch.setattr("faster_whisper.WhisperModel", _FakeWhisperModel)

        sys.argv = [
            "speech_to_text_cli.py",
            "--audio", str(tiny_audio_dataset),
            "--engine", "whisper-faster",
            "--translate",
            "--target-lang", "en",
            "--output-dir", str(tmp_path / "out_dir"),
        ]

        from speech_to_text_cli import main as cli_main
        cli_main()

        out_file = next((tmp_path / "out_dir").glob("*.txt"))
        content = out_file.read_text(encoding="utf-8")
        assert "TRANSCRIPTION SUMMARY" in content
        assert "Translation:" in content or "🌐 Translation:" in content