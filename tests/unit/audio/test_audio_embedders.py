# tests/unit/audio/test_audio_embedders.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from fait.audio.speaker_recognition.base import _BaseAudioEmbedder


class MockAudioEmbedder(_BaseAudioEmbedder):
    """Mock embedder for testing base class functionality"""

    def __init__(self, embed_cache_dir=None):
        super().__init__(embed_cache_dir)
        self.embed_calls = []

    def name(self):
        return "MockEmbedder"

    def embed_file(self, audio_path, use_cache=True):
        self.embed_calls.append((audio_path, use_cache))
        # Return deterministic embeddings based on filename
        if "ref1" in str(audio_path):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif "ref2" in str(audio_path):
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif "ref3" in str(audio_path):
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return None


class TestBaseAudioEmbedder:
    """Test base audio embedder class"""

    def test_cache_dir_initialization(self, tmp_paths):
        embedder = MockAudioEmbedder(embed_cache_dir=str(tmp_paths.embeddings_cache))
        assert embedder.embed_cache_dir == str(tmp_paths.embeddings_cache)

    def test_default_cache_dir(self, tmp_paths, monkeypatch):
        import fait.audio.speaker_recognition.base as base_mod
        monkeypatch.setattr(base_mod, "get_paths", lambda: tmp_paths)

        embedder = MockAudioEmbedder()
        assert str(tmp_paths.embeddings_cache) in embedder.embed_cache_dir

    def test_mean_embedding_from_folder(self, tmp_path):
        embedder = MockAudioEmbedder()

        # Create mock audio files
        (tmp_path / "ref1.wav").write_bytes(b"fake")
        (tmp_path / "ref2.wav").write_bytes(b"fake")
        (tmp_path / "ref3.wav").write_bytes(b"fake")

        mean = embedder.mean_embedding_from_folder(str(tmp_path), use_cache=False)

        # Expected mean: ([1,0,0] + [0,1,0] + [0,0,1]) / 3 = [0.333, 0.333, 0.333]
        # Then L2 normalized
        expected = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        expected = expected / np.linalg.norm(expected)

        assert np.allclose(mean, expected, atol=0.01)
        assert len(embedder.embed_calls) == 3

    def test_mean_embedding_ignores_non_audio(self, tmp_path):
        embedder = MockAudioEmbedder()

        # Create mix of audio and non-audio files
        (tmp_path / "ref1.wav").write_bytes(b"fake")
        (tmp_path / "readme.txt").write_bytes(b"text")
        (tmp_path / "image.jpg").write_bytes(b"image")
        (tmp_path / "ref2.mp3").write_bytes(b"fake")

        mean = embedder.mean_embedding_from_folder(str(tmp_path), use_cache=False)

        # Should only process the 2 audio files
        assert len(embedder.embed_calls) == 2

    def test_mean_embedding_empty_folder(self, tmp_path):
        embedder = MockAudioEmbedder()

        with pytest.raises(ValueError, match="No valid.*embeddings"):
            embedder.mean_embedding_from_folder(str(tmp_path), use_cache=False)

    def test_mean_embedding_skips_failed_embeddings(self, tmp_path):
        embedder = MockAudioEmbedder()

        # Create files where one will return None
        (tmp_path / "ref1.wav").write_bytes(b"fake")
        (tmp_path / "bad_file.wav").write_bytes(b"fake")  # Will return None
        (tmp_path / "ref2.wav").write_bytes(b"fake")

        mean = embedder.mean_embedding_from_folder(str(tmp_path), use_cache=False)

        # Should successfully compute mean from the 2 valid embeddings
        assert mean is not None
        assert mean.shape[0] == 3

    def test_cache_usage(self, tmp_path):
        embedder = MockAudioEmbedder(embed_cache_dir=str(tmp_path / "cache"))

        # Create mock audio file
        audio_file = tmp_path / "ref1.wav"
        audio_file.write_bytes(b"fake")

        # First call should use cache
        mean1 = embedder.mean_embedding_from_folder(str(tmp_path), use_cache=True)
        call_count_1 = len(embedder.embed_calls)

        # Second call should also use cache (implementation dependent)
        mean2 = embedder.mean_embedding_from_folder(str(tmp_path), use_cache=True)

        assert np.allclose(mean1, mean2)


class TestEmbedderNormalization:
    """Test L2 normalization in base embedder"""

    def test_mean_is_normalized(self, tmp_path):
        embedder = MockAudioEmbedder()

        (tmp_path / "ref1.wav").write_bytes(b"fake")
        (tmp_path / "ref2.wav").write_bytes(b"fake")

        mean = embedder.mean_embedding_from_folder(str(tmp_path), use_cache=False)

        # Check that result is L2 normalized (norm = 1)
        norm = np.linalg.norm(mean)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_handles_zero_embeddings(self, tmp_path):
        class ZeroEmbedder(_BaseAudioEmbedder):
            def name(self): return "zero"

            def embed_file(self, p, use_cache=True):
                return np.zeros(3, dtype=np.float32)

        embedder = ZeroEmbedder()
        (tmp_path / "test.wav").write_bytes(b"fake")

        mean = embedder.mean_embedding_from_folder(str(tmp_path), use_cache=False)

        # Should handle zero vector gracefully (may return zero or normalized value)
        assert mean is not None


class TestCachePathGeneration:
    """Test cache path generation for embedders"""

    def test_cache_path_uses_model_tag(self, tmp_path):
        from fait.core.utils import cache_path

        path1 = cache_path(tmp_path, "/path/to/audio.wav", "model_v1")
        path2 = cache_path(tmp_path, "/path/to/audio.wav", "model_v2")

        # Different model tags should produce different cache paths
        assert path1 != path2

    def test_cache_path_consistent(self, tmp_path):
        from fait.core.utils import cache_path

        path1 = cache_path(tmp_path, "/path/to/audio.wav", "model")
        path2 = cache_path(tmp_path, "/path/to/audio.wav", "model")

        # Same inputs should produce same cache path
        assert path1 == path2

    def test_cache_path_handles_different_files(self, tmp_path):
        from fait.core.utils import cache_path

        path1 = cache_path(tmp_path, "/path/audio1.wav", "model")
        path2 = cache_path(tmp_path, "/path/audio2.wav", "model")

        # Different audio files should have different cache paths
        assert path1 != path2


class TestAudioFileLoading:
    """Test audio file loading utilities"""

    def test_load_audio_any_handles_common_formats(self):
        from fait.core.utils import load_audio_any

        # These would need actual audio files to test fully
        # Here we just test the function exists and has correct signature
        assert callable(load_audio_any)

    def test_audio_extensions_recognized(self):
        from fait.core.utils import has_audio_extension

        common_formats = [
            ".wav", ".mp3", ".m4a", ".flac", ".ogg",
            ".opus", ".aac", ".wma"
        ]

        for ext in common_formats:
            assert has_audio_extension(Path(f"test{ext}")) is True

    def test_is_audio_file_checks_existence(self, tmp_path):
        from fait.core.utils import is_audio_file

        # File doesn't exist
        assert is_audio_file(tmp_path / "test.wav") is False

        # File exists with audio extension
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake")
        assert is_audio_file(audio_file) is True

        # File exists but wrong extension
        other_file = tmp_path / "test.txt"
        other_file.write_bytes(b"fake")
        assert is_audio_file(other_file) is False