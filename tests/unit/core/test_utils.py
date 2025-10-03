# tests/unit/core/test_utils.py
import numpy as np
import pytest
from pathlib import Path
from fait.core.utils import (
    compute_distance,
    cosine_similarity,
    l2_normalize,
    sort_pairs,
    topk_pairs,
    fuse_scores,
    human_size,
    to_safe_filename,
    is_image_file,
    is_audio_file,
    is_video_file,
)


class TestDistanceMetrics:
    """Test distance and similarity calculations"""

    def test_l2_normalize(self):
        v = np.array([3.0, 4.0])
        normalized = l2_normalize(v)
        assert np.allclose(np.linalg.norm(normalized), 1.0)
        assert normalized[0] == pytest.approx(0.6)
        assert normalized[1] == pytest.approx(0.8)

    def test_l2_normalize_zero_vector(self):
        v = np.array([0.0, 0.0])
        normalized = l2_normalize(v)
        # Should return original for zero vector
        assert np.allclose(normalized, v)

    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(-1.0)

    def test_compute_distance_euclidean(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        dist = compute_distance(a, b, metric="euclidean")
        assert dist == pytest.approx(5.0)

    def test_compute_distance_cosine(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        dist = compute_distance(a, b, metric="cosine")
        # cosine distance = 1 - similarity = 1 - 0 = 1
        assert dist == pytest.approx(1.0)

    def test_compute_distance_invalid_metric(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_distance(a, b, metric="invalid")


class TestPairSorting:
    """Test pair sorting and selection functions"""

    def test_sort_pairs_ascending(self):
        pairs = [("a", 0.5), ("b", 0.3), ("c", 0.8)]
        sorted_pairs = sort_pairs(pairs, higher_is_better=False)
        assert sorted_pairs == [("b", 0.3), ("a", 0.5), ("c", 0.8)]

    def test_sort_pairs_descending(self):
        pairs = [("a", 0.5), ("b", 0.3), ("c", 0.8)]
        sorted_pairs = sort_pairs(pairs, higher_is_better=True)
        assert sorted_pairs == [("c", 0.8), ("a", 0.5), ("b", 0.3)]

    def test_topk_pairs(self):
        pairs = [("a", 0.5), ("b", 0.3), ("c", 0.8), ("d", 0.9)]
        top2 = topk_pairs(pairs, k=2, higher_is_better=True)
        assert len(top2) == 2
        assert top2[0][0] == "d"
        assert top2[1][0] == "c"

    def test_topk_pairs_k_larger_than_list(self):
        pairs = [("a", 0.5), ("b", 0.3)]
        top10 = topk_pairs(pairs, k=10, higher_is_better=True)
        assert len(top10) == 2


class TestScoreFusion:
    """Test score fusion logic"""

    def test_fuse_scores_and_method(self):
        # Mock fusion config
        class FusionCfg:
            method = "and"
            class_thresholds = {"knife": 0.5}
            gdino_only_default_tau = 0.5

        fcfg = FusionCfg()
        fused, tau, ok = fuse_scores(0.7, 0.6, "knife", fcfg)

        # "and" method: min of both scores
        assert fused == 0.6
        assert tau == 0.5
        assert ok is True  # both >= 0.5

    def test_fuse_scores_and_method_fails(self):
        class FusionCfg:
            method = "and"
            class_thresholds = {"knife": 0.5}
            gdino_only_default_tau = 0.5

        fcfg = FusionCfg()
        fused, tau, ok = fuse_scores(0.7, 0.3, "knife", fcfg)

        assert fused == 0.3
        assert ok is False  # one score < 0.5

    def test_fuse_scores_weighted_method(self):
        class FusionCfg:
            method = "weighted"
            alpha = 0.6
            tau_star = 0.5
            class_thresholds = {}
            gdino_only_default_tau = 0.5

        fcfg = FusionCfg()
        fused, tau, ok = fuse_scores(0.8, 0.4, "knife", fcfg)

        # weighted: 0.6 * 0.8 + 0.4 * 0.4 = 0.48 + 0.16 = 0.64
        assert fused == pytest.approx(0.64)
        assert tau == 0.5
        assert ok is True

    def test_fuse_scores_max_method(self):
        class FusionCfg:
            method = "max"
            tau_star = 0.5
            class_thresholds = {}
            gdino_only_default_tau = 0.5

        fcfg = FusionCfg()
        fused, tau, ok = fuse_scores(0.7, 0.9, "knife", fcfg)

        assert fused == 0.9
        assert ok is True


class TestFileUtils:
    """Test file utility functions"""

    def test_to_safe_filename(self):
        assert to_safe_filename("hello world.txt") == "hello_world.txt"
        assert to_safe_filename("file/with\\slashes") == "file_with_slashes"
        assert to_safe_filename("special!@#$chars") == "special_chars"

    def test_to_safe_filename_long(self):
        long_name = "a" * 200
        safe = to_safe_filename(long_name, maxlen=80)
        assert len(safe) <= 80

    def test_is_image_file(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")
        assert is_image_file(img) is True

        not_img = tmp_path / "test.txt"
        not_img.write_bytes(b"text")
        assert is_image_file(not_img) is False

    def test_is_audio_file(self, tmp_path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake")
        assert is_audio_file(audio) is True

        not_audio = tmp_path / "test.txt"
        not_audio.write_bytes(b"text")
        assert is_audio_file(not_audio) is False

    def test_is_video_file(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        assert is_video_file(video) is True

        not_video = tmp_path / "test.txt"
        not_video.write_bytes(b"text")
        assert is_video_file(not_video) is False

    def test_human_size(self):
        assert human_size(0) == "0.0 B"
        assert human_size(1024) == "1.0 KiB"
        assert human_size(1024 * 1024) == "1.0 MiB"
        assert human_size(1024 * 1024 * 1024) == "1.0 GiB"
        assert human_size(1536) == "1.5 KiB"

    def test_human_size_si(self):
        assert human_size(1000, si=True) == "1.0 kB"
        assert human_size(1000 * 1000, si=True) == "1.0 MB"