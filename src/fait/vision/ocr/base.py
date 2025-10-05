from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol
from PIL.Image import Image

@dataclass
class OcrResult:
    text: str
    lang: Optional[str] = None
    confidence: Optional[float] = None   # 0..1 (if available)
    notes: str = ""                      # e.g. rotation, normalization notes
    engine: str = ""                     # "tesseract" | "paddle" | "trocr" | "doctr" | "donut"

class OcrEngine(Protocol):
    name: str
    lang: Optional[str]
    def recognize(self, img: Image) -> OcrResult | None: ...
