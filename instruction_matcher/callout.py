from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .ocr import _ocr_qty_from_crop


def find_callout_box(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Locate the light-blue parts callout box."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([95, 30, 180], dtype=np.uint8)
    upper = np.array([110, 50, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = 0.0
    h, w = img_bgr.shape[:2]

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.004 * (w * h):
            continue
        if cw < 100 or ch < 80:
            continue

        contour_area = cv2.contourArea(c)
        rectness = float(contour_area) / float(area + 1e-6)
        top_bias = 1.0 - (y / max(1, h))
        score = rectness * area * (0.3 + 0.7 * top_bias)

        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    return best


def callout_qty_hits(callout_bgr: np.ndarray) -> int:
    h, w = callout_bgr.shape[:2]
    band_top = int(h * 0.6)
    band = callout_bgr[band_top:h, 0:w]
    if band.size == 0:
        return 0
    # Reuse text crop logic by taking a conservative bottom-left region.
    eh, ew = band.shape[:2]
    text_h = max(50, int(eh * 0.5))
    text_w = max(20, int(ew * 0.6))
    text_crop = band[eh - text_h : eh, 0:text_w]
    qty = _ocr_qty_from_crop(text_crop)
    return 1 if qty is not None else 0


def callout_text_blob_score(callout_bgr: np.ndarray) -> int:
    """Score likely text blobs in the bottom-left of the callout."""
    h, w = callout_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0
    hsv = cv2.cvtColor(callout_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([95, 30, 180], dtype=np.uint8)
    upper = np.array([110, 50, 255], dtype=np.uint8)
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


def _callout_foreground_boxes(callout_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return bounding boxes of non-blue foreground parts in the callout."""
    hsv = cv2.cvtColor(callout_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([95, 30, 180], dtype=np.uint8)
    upper = np.array([110, 50, 255], dtype=np.uint8)
    blue = cv2.inRange(hsv, lower, upper)
    fg = cv2.bitwise_not(blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 400:
            continue
        if w < 18 or h < 18:
            continue
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect > 10.0:
            continue
        boxes.append((x, y, w, h))
    filtered: List[Tuple[int, int, int, int]] = []
    for i, (x, y, w, h) in enumerate(boxes):
        x2 = x + w
        y2 = y + h
        contains_other = False
        for j, (ox, oy, ow, oh) in enumerate(boxes):
            if i == j:
                continue
            ox2 = ox + ow
            oy2 = oy + oh
            if ox >= x and oy >= y and ox2 <= x2 and oy2 <= y2:
                contains_other = True
                break
        if not contains_other:
            filtered.append((x, y, w, h))
    return filtered


def _split_parts_from_foreground(callout_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return part boxes based on foreground blobs split into horizontal bins."""
    h, w = callout_bgr.shape[:2]
    fg_boxes = _callout_foreground_boxes(callout_bgr)
    if not fg_boxes:
        return []

    centers = sorted([x + bw / 2.0 for x, y, bw, bh in fg_boxes])
    boundaries = [0.0]
    for i in range(1, len(centers)):
        boundaries.append((centers[i - 1] + centers[i]) / 2.0)
    boundaries.append(float(w))

    part_boxes: List[Tuple[int, int, int, int]] = []
    for x1, x2 in zip(boundaries[:-1], boundaries[1:]):
        bin_left = int(x1)
        bin_right = int(x2)
        candidates = []
        for bx, by, bw2, bh2 in fg_boxes:
            cx = bx + bw2 / 2.0
            if cx < bin_left or cx > bin_right:
                continue
            candidates.append((bx, by, bw2, bh2))
        if not candidates:
            continue
        candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
        part_boxes.append(candidates[0])

    return part_boxes


def extract_parts_from_callout(
    callout_bgr: np.ndarray,
    page_index: int,
) -> List[Tuple[np.ndarray, Optional[int], np.ndarray, np.ndarray]]:
    """Extract (part_image, qty, ext_crop, text_crop) candidates from a callout box."""
    h, w = callout_bgr.shape[:2]
    part_boxes = _split_parts_from_foreground(callout_bgr)
    if not part_boxes:
        return []

    out: List[Tuple[np.ndarray, Optional[int], np.ndarray, np.ndarray]] = []
    for idx, (bx, by, bw2, bh2) in enumerate(part_boxes):
        pad = 6
        left = max(0, bx - pad)
        right = min(w, bx + bw2 + pad)
        top = max(0, by - pad)
        bottom = min(h, by + bh2 + pad)

        part_crop = callout_bgr[top:bottom, left:right]
        if part_crop.size == 0:
            continue

        extend = int(max(48, bh2 * 0.35))
        ext_bottom = min(h, bottom + extend)
        ext_crop = callout_bgr[top:ext_bottom, left:right]
        if ext_crop.size == 0:
            continue

        eh, ew = ext_crop.shape[:2]
        text_h = max(50, int(eh * 0.5))
        text_w = max(48, int(ew * 0.6))
        text_crop = ext_crop[eh - text_h : eh, 0:text_w]
        qty = _ocr_qty_from_crop(text_crop)

        out.append((part_crop, qty, ext_crop, text_crop))

    return out
