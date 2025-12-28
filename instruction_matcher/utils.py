from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rotate_90(img: np.ndarray, angle_deg: int) -> np.ndarray:
    """Rotate by a multiple of 90 degrees."""
    angle_deg = angle_deg % 360
    if angle_deg == 0:
        return img
    if angle_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"angle_deg must be multiple of 90, got {angle_deg}")
