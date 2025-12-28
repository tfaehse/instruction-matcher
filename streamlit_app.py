"""Streamlit dashboard for instruction matcher results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st


def _load_results(out_dir: Path) -> Dict:
    results_path = out_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}. Run main.py first.")
    return json.loads(results_path.read_text(encoding="utf-8"))


def _page_image_path(out_dir: Path, page_index: int) -> Path:
    return out_dir / "debug" / f"p{page_index:03d}_oriented.png"


def _parts_for_page(results: Dict, page_index: int) -> List[Dict]:
    return [p for p in results.get("detected_parts", []) if p.get("page_index") == page_index]


def _cluster_to_pages(results: Dict) -> Dict[int, List[int]]:
    pages = defaultdict(set)
    for p in results.get("detected_parts", []):
        cid = int(p.get("cluster_id", -1))
        pages[cid].add(int(p.get("page_index", 0)))
    return {cid: sorted(list(pages[cid])) for cid in pages}


def _max_page_index(results: Dict) -> int:
    if not results.get("detected_parts"):
        return 0
    return max(int(p.get("page_index", 0)) for p in results["detected_parts"])


def _set_page_param(page_index: int) -> None:
    st.query_params.page = page_index


def _get_page_param(default_index: int) -> int:
    params = st.query_params
    if "page" in params:
        try:
            return int(params["page"][0])
        except Exception:
            return default_index
    return default_index


def render_overview(results: Dict, out_dir: Path) -> None:
    st.header("Overview")
    clusters = results.get("clusters", [])
    cluster_pages = _cluster_to_pages(results)
    cluster_parts = defaultdict(list)
    for part in results.get("detected_parts", []):
        cid = int(part.get("cluster_id", -1))
        cluster_parts[cid].append(part)

    if not clusters:
        st.info("No clusters found.")
        return

    for cluster in clusters:
        cid = int(cluster.get("cluster_id", -1))
        count = int(cluster.get("count", 0))
        examples = cluster.get("examples", [])
        pages = cluster_pages.get(cid, [])
        parts = cluster_parts.get(cid, [])

        with st.container():
            st.subheader(f"Cluster {cid:03d} · Count {count}")
            cols = st.columns([1, 3])
            with cols[0]:
                if examples:
                    img_path = Path(examples[0])
                    if img_path.exists():
                        st.image(str(img_path), width=180)
            with cols[1]:
                if pages:
                    st.write("Pages:", ", ".join(str(p + 1) for p in pages))
                    jump_cols = st.columns(min(6, len(pages)))
                    for i, p in enumerate(pages[:6]):
                        if jump_cols[i].button(f"Go to {p + 1}", key=f"go_{cid}_{p}"):
                            _set_page_param(p)
                else:
                    st.write("Pages: none")

            with st.expander("Show all patches"):
                if not parts:
                    st.write("No patches recorded for this cluster.")
                else:
                    patch_cols = st.columns(4)
                    for i, part in enumerate(parts):
                        crop_path = Path(part["crop_path"])
                        with patch_cols[i % 4]:
                            st.caption(
                                "Page {} · Qty {} · {} · dist {:.3f}".format(
                                    int(part.get("page_index", 0)) + 1,
                                    part["qty"],
                                    "confident" if part.get("qty_confident") else "uncertain",
                                    1.0 - float(part.get("cluster_score", 0.0)),
                                )
                            )
                            st.caption(
                                "hist {:.3f} · hog {:.3f} · phash {:.3f}".format(
                                    float(part.get("hist_score", 0.0)),
                                    float(part.get("hog_score", 0.0)),
                                    float(part.get("phash_score", 0.0)),
                                )
                            )
                            if crop_path.exists():
                                st.image(str(crop_path), width=160)
                            else:
                                st.write(f"Missing: {crop_path}")

        st.divider()


def render_page_view(results: Dict, out_dir: Path) -> None:
    st.header("Page Viewer")
    max_page = _max_page_index(results)
    default_page = _get_page_param(0)
    page_index = st.slider("Page", min_value=0, max_value=max_page, value=default_page, step=1)

    left_col, right_col = st.columns([2, 1], gap="large")
    with left_col:
        page_path = _page_image_path(out_dir, page_index)
        if page_path.exists():
            st.image(str(page_path), caption=f"Page {page_index + 1}", width=700)
        else:
            st.warning(f"Missing page image: {page_path}")

    with right_col:
        st.subheader("Parts")
        parts = _parts_for_page(results, page_index)
        if not parts:
            st.info("No parts detected for this page.")
            return

        for part in parts:
            crop_path = Path(part["crop_path"])
            if part.get("qty_confident"):
                st.caption(f"Qty: {part['qty']} · confident")
            else:
                st.markdown(
                    f"<span style='color:#d11a2a;'>Qty: {part['qty']} · uncertain</span>",
                    unsafe_allow_html=True,
                )
            if crop_path.exists():
                st.image(str(crop_path), width=220)
            else:
                st.write(f"Missing crop: {crop_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Streamlit dashboard for instruction matcher results")
    parser.add_argument("--out", type=str, default="out", help="Output directory from main.py")
    args, _ = parser.parse_known_args()

    out_dir = Path(args.out).expanduser().resolve()
    results = _load_results(out_dir)

    st.set_page_config(page_title="Instruction Matcher", layout="wide")
    st.title("Instruction Matcher")

    view = st.sidebar.radio("View", ["Overview", "Page Viewer", "Distance Matrix"], index=0)
    if view == "Overview":
        render_overview(results, out_dir)
    elif view == "Page Viewer":
        render_page_view(results, out_dir)
    else:
        render_distance_matrix(out_dir)


def render_distance_matrix(out_dir: Path) -> None:
    st.header("Distance Matrix")
    items_path = out_dir / "dbscan_items.json"
    scores_path = out_dir / "dbscan_scores.npz"
    if not items_path.exists() or not scores_path.exists():
        st.info("Run the extractor to generate dbscan_scores.npz and dbscan_items.json.")
        return

    items = json.loads(items_path.read_text(encoding="utf-8"))
    scores = np.load(scores_path)
    thresholds = scores.get("thresholds")
    if thresholds is None:
        thresholds = np.array([0.0, 0.0, 0.0])
    hist_th, hog_th, phash_th = float(thresholds[0]), float(thresholds[1]), float(thresholds[2])

    st.subheader("Patches")
    cols = st.columns(6)
    for i, item in enumerate(items):
        with cols[i % 6]:
            st.caption(f"{i}: Page {int(item['page_index']) + 1}")
            p = Path(item["crop_path"])
            if p.exists():
                st.image(str(p), width=100)
            else:
                st.caption("Missing crop")

    def show_matrix(name: str, threshold: float):
        mat = scores[name]
        mat = np.clip(mat, 0.0, 1.0)
        ok = (mat >= threshold).astype("uint8")
        img = np.zeros((mat.shape[0], mat.shape[1], 3), dtype="uint8")
        img[ok == 1] = (0, 200, 0)
        img[ok == 0] = (200, 0, 0)
        st.subheader(f"{name.upper()} Matrix")
        st.image(img, caption=f"{name} scores (green >= {threshold:.3f}, red < {threshold:.3f})")
        df = pd.DataFrame(np.round(mat, 3))
        def _style(val: float) -> str:
            if val >= threshold:
                return "background-color: #1b7f3b; color: white;"
            return "background-color: #b51d1a; color: white;"
        st.dataframe(df.style.map(_style))

    show_matrix("hist", hist_th)
    show_matrix("hog", hog_th)
    show_matrix("phash", phash_th)
    show_matrix("final", 0.0)


if __name__ == "__main__":
    main()
