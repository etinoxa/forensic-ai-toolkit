# tests/unit/core/test_utils_report.py
from pathlib import Path
from fait.core.utils import write_report

def test_write_report(tmp_path):
    out = tmp_path / "out"
    pairs = [("a.jpg", 0.9), ("b.jpg", 0.7), ("c.jpg", 0.95)]
    match_counts = {0.8: 2, 0.9: 1}
    p = write_report(
        output_dir=out,
        model_type="CLIP",
        metric_name="cosine_similarity",
        processed=3,
        elapsed_seconds=1.23,
        match_counts=match_counts,
        pairs=pairs,
        top_k=2,
        higher_is_better=True,
        filename="report.txt",
    )
    text = Path(p).read_text()
    assert "FINAL REPORT" in text
    assert "Model type: CLIP" in text
    assert "Threshold 0.8: 2" in text
    assert "cosine_similarity" in text
