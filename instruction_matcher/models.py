from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class DetectedPart:
    page_index: int
    qty: int
    qty_confident: bool
    crop_path: str
    phash: str
    cluster_id: int = -1
    cluster_score: float = 0.0
    hist_score: float = 0.0
    sift_score: float = 0.0
    phash_score: float = 0.0


@dataclass
class Cluster:
    cluster_id: int
    rep_phash: str
    count: int
    examples: List[str]
