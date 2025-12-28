from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm

from .callout import extract_parts_from_callout, find_callout_box
from .clustering import compute_color_hist, compute_hog_desc, compute_phash, normalize_to_512, offline_cluster
from .models import Cluster, DetectedPart
from .orientation import normalize_page_orientation
from .utils import mkdirp


def render_pdf_page(doc, page_index: int, dpi: int = 200) -> np.ndarray:
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def process(
    pdf_path: Path, start_page: int, end_page: int, dpi: int = 200, out_dir: Path = Path("out")
) -> Tuple[List[DetectedPart], List[Cluster]]:
    mkdirp(out_dir)
    debug_dir = out_dir / "debug"
    mkdirp(debug_dir)

    doc = fitz.open(str(pdf_path))

    detected: List[DetectedPart] = []
    items: List[Dict] = []

    start_idx = max(0, start_page - 1)
    end_idx = min(end_page - 1, doc.page_count - 1)
    if end_idx < start_idx:
        return [], []

    for page_index in tqdm(range(start_idx, end_idx + 1), desc="pages", unit="page"):
        img = render_pdf_page(doc, page_index, dpi=dpi)
        cv2.imwrite(str(debug_dir / f"p{page_index:03d}_render.png"), img)

        img = normalize_page_orientation(img)
        cv2.imwrite(str(debug_dir / f"p{page_index:03d}_oriented.png"), img)

        box = find_callout_box(img)
        if box is None:
            print(f"[warn] page {page_index+1}: callout box not found")
            continue

        x, y, w, h = box
        callout = img[y : y + h, x : x + w]
        cv2.imwrite(str(debug_dir / f"p{page_index:03d}_callout.png"), callout)

        parts = extract_parts_from_callout(callout, debug_dir, page_index)
        if not parts:
            print(f"[warn] page {page_index+1}: no parts extracted from callout")
            continue

        for i, (part_img, qty) in enumerate(parts):
            if qty is None:
                qty_int = 1
                qty_confident = False
            else:
                qty_int = int(qty)
                qty_confident = True

            norm_part = normalize_to_512(part_img)
            ph = compute_phash(norm_part)
            hist = compute_color_hist(norm_part)
            hog_desc = compute_hog_desc(norm_part)

            crop_path = debug_dir / f"p{page_index:03d}_partcrop_{i:02d}.png"
            cv2.imwrite(str(crop_path), norm_part)

            detected.append(
                DetectedPart(
                    page_index=page_index,
                    qty=qty_int,
                    qty_confident=qty_confident,
                    crop_path=str(crop_path),
                    phash=ph,
                )
            )
            items.append(
                {
                    "index": len(detected) - 1,
                    "phash": ph,
                    "hist": hist,
                    "hog": hog_desc,
                    "qty": qty_int,
                    "crop_path": str(crop_path),
                }
            )

    items, out_clusters_raw, dist, scores = offline_cluster(items)
    if dist.size:
        np.save(out_dir / "dbscan_distances.npy", dist)
        np.savez_compressed(
            out_dir / "dbscan_scores.npz",
            hist=scores["hist"],
            hog=scores["hog"],
            phash=scores["phash"],
            final=scores["final"],
            thresholds=np.array([scores["thresholds"]["hist"], scores["thresholds"]["hog"], scores["thresholds"]["phash"]]),
        )
        (out_dir / "dbscan_items.json").write_text(
            json.dumps(
                [{"index": i, "crop_path": d.crop_path, "page_index": d.page_index} for i, d in enumerate(detected)],
                indent=2,
            ),
            encoding="utf-8",
        )
    for item in items:
        idx = item["index"]
        detected[idx].cluster_id = int(item["cluster_id"])
        detected[idx].cluster_score = float(item["cluster_score"])
        detected[idx].hist_score = float(item["hist_score"])
        detected[idx].hog_score = float(item["hog_score"])
        detected[idx].phash_score = float(item["phash_score"])

    out_clusters: List[Cluster] = []
    for c in out_clusters_raw:
        out_clusters.append(
            Cluster(
                cluster_id=int(c["cluster_id"]),
                rep_phash=str(c["rep_phash"]),
                count=int(c["count"]),
                examples=list(c["examples"]),
            )
        )

    return detected, out_clusters


def build_results(
    pdf_path: Path,
    detected: List[DetectedPart],
    clusters: List[Cluster],
    start_page: int,
    end_page: int,
) -> Dict:
    total_parts_confident = int(sum(d.qty for d in detected if d.qty_confident))
    total_parts_including_uncertain = int(sum(d.qty for d in detected))
    return {
        "pdf": str(pdf_path),
        "pages_processed": int(len(set(d.page_index for d in detected))),
        "page_range": {"start": int(start_page), "end": int(end_page)},
        "detected_parts": [asdict(d) for d in detected],
        "clusters": [asdict(c) for c in clusters],
        "total_parts_confident": total_parts_confident,
        "total_parts_including_uncertain": total_parts_including_uncertain,
        "total_clusters": int(len(clusters)),
    }


def write_results(out_dir: Path, results: Dict) -> Path:
    mkdirp(out_dir)
    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out_json
