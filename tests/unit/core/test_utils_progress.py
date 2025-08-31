# tests/unit/core/test_utils_progress.py
import logging
from fait.core.utils import ProgressMeter, append_jsonl, write_report

def test_progress_and_jsonl(tmp_path, caplog):
    log = logging.getLogger("t")
    caplog.set_level(logging.INFO, logger="t")   # <- capture this logger

    log_path = tmp_path / "run" / "log.jsonl"
    pm = ProgressMeter(total=3, label="face_match:progress", logger=log, log_path=log_path)
    for _ in range(3):
        pm.step()
    pm.close()

    # JSONL written
    assert log_path.exists() and log_path.read_text().strip()

    # CLI line captured
    assert any("face_match:progress" in rec.message for rec in caplog.records)
