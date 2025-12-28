from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _normalize_size(img_bgr: np.ndarray, size: int = 160) -> np.ndarray:
    """Resize with padding to a square for scale-invariant features."""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return img_bgr
    scale = size / float(max(h, w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


def normalize_to_512(img_bgr: np.ndarray, size: int = 512) -> np.ndarray:
    """Resize to longest side == size, pad right/bottom with white."""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return img_bgr
    scale = size / float(max(h, w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_right = size - new_w
    pad_bottom = size - new_h
    return cv2.copyMakeBorder(
        resized, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


def compute_color_hist(img_bgr: np.ndarray, h_bins: int = 24, s_bins: int = 24) -> np.ndarray:
    """Compute normalized HS histogram (robust to shading changes)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def compute_hog_desc(img_bgr: np.ndarray) -> np.ndarray:
    """Compute HOG descriptor for shape matching."""
    norm = _normalize_size(img_bgr, size=128)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9,
    )
    desc = hog.compute(gray)
    if desc is None:
        return np.zeros((1,), dtype=np.float32)
    return desc.flatten()


def compute_phash(img_bgr: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Perceptual hash (pHash) to cluster identical part images across scale."""
    norm = _normalize_size(img_bgr, size=64)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    resized = np.float32(resized)

    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]

    med = np.median(dct_low[1:, 1:])
    bits = dct_low > med
    flat = bits.flatten().astype(np.uint8)
    bit_string = "".join("1" if b else "0" for b in flat)
    return hex(int(bit_string, 2))[2:].rjust((hash_size * hash_size) // 4, "0")


def hamming_hex(a: str, b: str) -> int:
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))


def _hog_match_score(desc_a: np.ndarray, desc_b: np.ndarray) -> float:
    denom = (np.linalg.norm(desc_a) * np.linalg.norm(desc_b)) + 1e-6
    return float(np.dot(desc_a, desc_b) / denom)


def _phash_score(phash_a: str, phash_b: str) -> float:
    dist = hamming_hex(phash_a, phash_b)
    return float(max(0.0, (64 - dist) / 64.0))


def assign_cluster(
    phash: str,
    color_hist: np.ndarray,
    hog_desc: np.ndarray,
    clusters: Dict[int, Dict],
    min_hist: float = 0.998,
    min_hog: float = 0.5,
    min_phash: float = 0.9,
) -> Tuple[int, Dict[str, float]]:
    """Assign to an existing cluster or create a new one."""
    best_id = None
    best_scores = {"hist": 0.0, "sift": 0.0, "phash": 0.0, "final": 0.0}

    for cid, c in clusters.items():
        hist_score = cv2.compareHist(color_hist, c["rep_hist"], cv2.HISTCMP_CORREL)
        if hist_score < min_hist:
            continue

        hog_score = _hog_match_score(hog_desc, c["rep_hog"])

        ph_score = _phash_score(phash, c["rep_phash"])
        if hog_score < min_hog:
            continue
        if ph_score < min_phash:
            continue

        best_id = cid
        best_scores = {
            "hist": float(hist_score),
            "sift": float(hog_score),
            "phash": float(ph_score),
            "final": 0.0,
        }
        break

    if best_id is not None:
        return int(best_id), best_scores

    new_id = int(max(clusters.keys(), default=-1) + 1)
    clusters[new_id] = {
        "rep_phash": phash,
        "rep_hist": color_hist,
        "rep_hog": hog_desc,
        "count": 0,
        "examples": [],
    }
    return new_id, {"hist": 0.0, "sift": 0.0, "phash": 0.0, "final": 0.0}
