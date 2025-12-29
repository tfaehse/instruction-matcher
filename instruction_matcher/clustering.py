from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


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



def _score_pair(
    a: dict,
    b: dict,
    min_hist: float,
    min_hog: float,
    min_phash: float,
) -> tuple[float, float, float, float]:
    hist_score = float(cv2.compareHist(a["hist"], b["hist"], cv2.HISTCMP_CORREL))
    hog_score = float(_hog_match_score(a["hog"], b["hog"]))
    ph_score = float(_phash_score(a["phash"], b["phash"]))
    if hist_score < min_hist or hog_score < min_hog or ph_score < min_phash:
        return hist_score, hog_score, ph_score, 0.0
    dist = (1.0 - max(0.0, hist_score)) * 0.4 + (1.0 - hog_score) * 0.4 + (1.0 - ph_score) * 0.2
    final = 1.0 - dist
    return hist_score, hog_score, ph_score, final


def offline_cluster(
    items: list[dict],
    eps: float = 0.1,
    min_samples: int = 1,
    min_hist: float = 0.995,
    min_hog: float = 0.975,
    min_phash: float = 0.925,
) -> tuple[list[dict], list[dict], np.ndarray, Dict[str, np.ndarray]]:
    """Cluster items offline with DBSCAN and return updated items + cluster summaries + scores."""
    if not items:
        empty = np.zeros((0, 0), dtype=np.float64)
        return items, [], empty, {"hist": empty, "hog": empty, "phash": empty, "final": empty}

    n = len(items)
    dist = np.zeros((n, n), dtype=np.float64)
    hist_m = np.zeros((n, n), dtype=np.float64)
    hog_m = np.zeros((n, n), dtype=np.float64)
    phash_m = np.zeros((n, n), dtype=np.float64)
    final_m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            hist_score, hog_score, ph_score, final = _score_pair(items[i], items[j], min_hist, min_hog, min_phash)
            if final <= 0.0:
                d = 1.0
            else:
                d = 1.0 - final
                if d < 0.0:
                    d = 0.0
            dist[i, j] = d
            dist[j, i] = d
            hist_m[i, j] = hist_m[j, i] = hist_score
            hog_m[i, j] = hog_m[j, i] = hog_score
            phash_m[i, j] = phash_m[j, i] = ph_score
            final_m[i, j] = final_m[j, i] = final
    for i in range(n):
        hist_m[i, i] = 1.0
        hog_m[i, i] = 1.0
        phash_m[i, i] = 1.0
        final_m[i, i] = 1.0

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clusterer.fit_predict(dist)

    next_cluster = 0
    label_map: Dict[int, int] = {}
    for lab in labels:
        if lab == -1:
            continue
        if lab not in label_map:
            label_map[lab] = next_cluster
            next_cluster += 1

    clusters: Dict[int, Dict] = {}
    for idx, lab in enumerate(labels):
        if lab == -1:
            cid = next_cluster
            next_cluster += 1
        else:
            cid = label_map[lab]

        items[idx]["cluster_id"] = cid
        clusters.setdefault(cid, {"count": 0, "examples": [], "rep_index": idx})
        clusters[cid]["count"] += int(items[idx]["qty"])
        if len(clusters[cid]["examples"]) < 10:
            clusters[cid]["examples"].append(str(items[idx]["crop_path"]))

    for cid in clusters.keys():
        members = [i for i, it in enumerate(items) if it["cluster_id"] == cid]
        if len(members) == 1:
            clusters[cid]["rep_index"] = members[0]
            continue
        best_idx = members[0]
        best_score = 1e9
        for i in members:
            avg_dist = float(np.mean([dist[i, j] for j in members]))
            if avg_dist < best_score:
                best_score = avg_dist
                best_idx = i
        clusters[cid]["rep_index"] = best_idx

    for idx, item in enumerate(items):
        rep = items[clusters[item["cluster_id"]]["rep_index"]]
        hist_score, hog_score, ph_score, final = _score_pair(item, rep, min_hist, min_hog, min_phash)
        item["hist_score"] = hist_score
        item["hog_score"] = hog_score
        item["phash_score"] = ph_score
        item["cluster_score"] = final

    out_clusters: list[dict] = []
    for cid, c in sorted(clusters.items(), key=lambda kv: kv[0]):
        rep = items[c["rep_index"]]
        out_clusters.append(
            {
                "cluster_id": int(cid),
                "rep_phash": str(rep["phash"]),
                "count": int(c["count"]),
                "examples": list(c["examples"]),
            }
        )

    thresholds = {"hist": min_hist, "hog": min_hog, "phash": min_phash}
    scores = {"hist": hist_m, "hog": hog_m, "phash": phash_m, "final": final_m, "thresholds": thresholds}
    return items, out_clusters, dist, scores
