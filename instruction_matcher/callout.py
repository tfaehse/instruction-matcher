from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .ocr import _ocr_qty_from_crop


def find_callout_box(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Locate the light-blue parts callout box."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 10, 180], dtype=np.uint8)
    upper = np.array([130, 120, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = 0.0
    h, w = img_bgr.shape[:2]

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.01 * (w * h):
            continue
        if cw < 150 or ch < 80:
            continue

        contour_area = cv2.contourArea(c)
        rectness = float(contour_area) / float(area + 1e-6)
        top_bias = 1.0 - (y / max(1, h))
        score = rectness * area * (0.3 + 0.7 * top_bias)

        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    return best


def _callout_foreground_boxes(callout_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return bounding boxes of non-blue foreground parts in the callout."""
    hsv = cv2.cvtColor(callout_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 10, 180], dtype=np.uint8)
    upper = np.array([130, 120, 255], dtype=np.uint8)
    blue = cv2.inRange(hsv, lower, upper)
    fg = cv2.bitwise_not(blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 300:
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
    callout_bgr: np.ndarray, debug_dir: Path, page_index: int
) -> List[Tuple[np.ndarray, Optional[int]]]:
    """Extract (part_image, qty) candidates from a callout box."""
    h, w = callout_bgr.shape[:2]
    part_boxes = _split_parts_from_foreground(callout_bgr)
    if not part_boxes:
        return []

    out: List[Tuple[np.ndarray, Optional[int]]] = []
    for idx, (bx, by, bw2, bh2) in enumerate(part_boxes):
        pad = 6
        left = max(0, bx - pad)
        right = min(w, bx + bw2 + pad)
        top = max(0, by - pad)
        bottom = min(h, by + bh2 + pad)

        part_crop = callout_bgr[top:bottom, left:right]
        if part_crop.size == 0:
            continue

        extend = int(max(16, bh2 * 0.35))
        ext_bottom = min(h, bottom + extend)
        ext_crop = callout_bgr[top:ext_bottom, left:right]
        if ext_crop.size == 0:
            continue

        eh, ew = ext_crop.shape[:2]
        text_h = max(10, int(eh * 0.35))
        text_w = max(20, int(ew * 0.6))
        text_crop = ext_crop[eh - text_h : eh, 0:text_w]
        qty = _ocr_qty_from_crop(text_crop)

        cv2.imwrite(str(debug_dir / f"p{page_index:03d}_item{idx:02d}_part.png"), part_crop)
        cv2.imwrite(str(debug_dir / f"p{page_index:03d}_item{idx:02d}_text.png"), text_crop)
        out.append((part_crop, qty))

    return out
