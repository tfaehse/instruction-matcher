from __future__ import annotations

import re
import warnings
from functools import lru_cache
from typing import Optional

import cv2
import easyocr


warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*not supported on MPS.*",
    category=UserWarning,
)


@lru_cache(maxsize=1)
def _easyocr_reader() -> easyocr.Reader:
    return easyocr.Reader(["en"], gpu=True, verbose=False)


def _easyocr_readtext(img_bgr: np.ndarray, allowlist: str) -> list[tuple[str, float, list]]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    reader = _easyocr_reader()
    results = reader.readtext(img_rgb, detail=1, allowlist=allowlist, text_threshold=0.5, low_text=0.3, contrast_ths=0.1, decoder="greedy")
    out = []
    for bbox, text, conf in results:
        if text:
            out.append((text, float(conf), bbox))
    return out


def _ocr_qty_from_crop(crop_bgr: np.ndarray) -> Optional[int]:
    """OCR a quantity like '1x' from a small crop (EasyOCR only)."""
    results = _easyocr_readtext(crop_bgr, allowlist="0123456789xX")
    if not results:
        return None

    best_qty = None
    best_conf = -1.0
    for text, conf, bbox in results:
        m = re.search(r"(\d{1,3})\s*[xX]?", text)
        if not m:
            continue
        try:
            qty = int(m.group(1))
        except ValueError:
            continue
        if conf > best_conf:
            best_conf = conf
            best_qty = qty
    return best_qty


def _ocr_regex_hits(img_bgr: np.ndarray, pattern: str, allowlist: str) -> int:
    hits = 0
    for text, _, _ in _easyocr_readtext(img_bgr, allowlist=allowlist):
        hits += len(re.findall(pattern, text))
    return hits
