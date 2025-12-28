from __future__ import annotations

import numpy as np
import cv2

from .callout import find_callout_box, _split_parts_from_foreground
from .ocr import _ocr_qty_from_crop, _ocr_regex_hits
from .utils import rotate_90


def _callout_qty_hits(callout_bgr: np.ndarray) -> int:
    h, w = callout_bgr.shape[:2]
    band_top = int(h * 0.6)
    band = callout_bgr[band_top:h, 0:w]
    if band.size == 0:
        return 0
    return _ocr_regex_hits(band, r"\b\d+\s*[xX]\b", allowlist="0123456789xX")


def _callout_text_blob_score(callout_bgr: np.ndarray) -> int:
    """Score likely text blobs in the bottom-left of the callout."""
    h, w = callout_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0
    hsv = cv2.cvtColor(callout_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 10, 180], dtype=np.uint8)
    upper = np.array([130, 120, 255], dtype=np.uint8)
    blue = cv2.inRange(hsv, lower, upper)
    non_blue = cv2.bitwise_not(blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    non_blue = cv2.morphologyEx(non_blue, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(non_blue, connectivity=8)
    x1 = 0
    y1 = int(h * 0.6)
    x2 = int(w * 0.5)
    y2 = h

    score = 0
    for i in range(1, num_labels):
        bx, by, bw, bh, area = stats[i]
        if area < 20 or area > 800:
            continue
        if bw > 80 or bh > 80:
            continue
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            score += 1
    return score


def _callout_part_qty_hits(callout_bgr: np.ndarray) -> int:
    """Count how many parts yield a qty when reading the lower-left text area."""
    h, w = callout_bgr.shape[:2]
    part_boxes = _split_parts_from_foreground(callout_bgr)
    hits = 0
    for (bx, by, bw2, bh2) in part_boxes:
        pad = 6
        left = max(0, bx - pad)
        right = min(w, bx + bw2 + pad)
        top = max(0, by - pad)
        bottom = min(h, by + bh2 + pad)

        extend = int(max(16, bh2 * 0.35))
        ext_bottom = min(h, bottom + extend)
        ext_crop = callout_bgr[top:ext_bottom, left:right]
        if ext_crop.size == 0:
            continue

        eh, ew = ext_crop.shape[:2]
        text_h = max(10, int(eh * 0.35))
        text_w = max(10, int(ew * 0.4))
        text_crop = ext_crop[eh - text_h : eh, 0:text_w]
        qty = _ocr_qty_from_crop(text_crop)
        if qty is not None:
            hits += 1
    return hits


def ocr_text_score(img_bgr: np.ndarray) -> int:
    """Estimate whether text is readable in this orientation."""
    score = 0

    box = find_callout_box(img_bgr)
    if box is not None:
        x, y, bw, bh = box
        callout = img_bgr[y : y + bh, x : x + bw]
        score += 1000 * _callout_part_qty_hits(callout)
        score += 100 * _callout_qty_hits(callout)
        score += 20 * _callout_text_blob_score(callout)

    score += _ocr_regex_hits(img_bgr, r"\b\d+\b", allowlist="0123456789")

    return int(score)


def normalize_page_orientation(img_bgr: np.ndarray) -> np.ndarray:
    """Rotate by 0/90/180/270 to maximize text readability."""
    candidates = []
    for rot in (0, 90, 180, 270):
        rotated = rotate_90(img_bgr, rot)
        score = ocr_text_score(rotated)
        candidates.append((score, rotated))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
