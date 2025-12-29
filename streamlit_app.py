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
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


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


def _apply_overrides(results: Dict, overrides: Dict[int, int]) -> Dict:
    if not overrides:
        return results
    parts = []
    for p in results.get("detected_parts", []):
        p = dict(p)
        if p.get("index") in overrides:
            p["cluster_id"] = overrides[p["index"]]
        parts.append(p)
    results = dict(results)
    results["detected_parts"] = parts
    return results


def _rebuild_clusters(results: Dict) -> List[Dict]:
    clusters: Dict[int, Dict] = {}
    for p in results.get("detected_parts", []):
        cid = int(p.get("cluster_id", -1))
        clusters.setdefault(cid, {"cluster_id": cid, "count": 0, "examples": []})
        clusters[cid]["count"] += int(p.get("qty", 0))
        if len(clusters[cid]["examples"]) < 10:
            clusters[cid]["examples"].append(p.get("crop_path", ""))
    return [clusters[cid] for cid in sorted(clusters.keys())]


def _save_overrides(out_dir: Path, overrides: Dict[int, int]) -> None:
    (out_dir / "cluster_overrides.json").write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def _load_overrides(out_dir: Path) -> Dict[int, int]:
    path = out_dir / "cluster_overrides.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in data.items()}


def _export_pdf(out_dir: Path, results: Dict) -> Path:
    out_path = out_dir / "clusters.pdf"
    c = canvas.Canvas(str(out_path), pagesize=letter)
    w, h = letter
    margin = 36
    thumb = 180
    gap = 16

    clusters = _rebuild_clusters(results)
    cluster_parts = defaultdict(list)
    for p in results.get("detected_parts", []):
        cid = int(p.get("cluster_id", -1))
        cluster_parts[cid].append(p)

    per_page = 6
    for cluster in clusters:
        cid = int(cluster.get("cluster_id", -1))
        parts = cluster_parts.get(cid, [])
        rep_path = (cluster.get("examples") or [None])[0]

        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, h - margin, f"Cluster {cid:03d} · Count {cluster.get('count', 0)}")
        c.setFont("Helvetica", 10)

        items = []
        if rep_path and Path(rep_path).exists():
            items.append((rep_path, f"Count {cluster.get('count', 0)}"))
        for p in parts:
            path = p.get("crop_path")
            if not path or not Path(path).exists():
                continue
            items.append((path, f"Qty {p.get('qty', '')}"))

        x0 = margin
        y0 = h - margin - 30
        col_w = thumb + gap
        row_h = thumb + 28
        col = 0
        row = 0

        for i, (path, label) in enumerate(items):
            if i > 0 and i % per_page == 0:
                c.showPage()
                c.setFont("Helvetica-Bold", 16)
                c.drawString(margin, h - margin, f"Cluster {cid:03d} · Count {cluster.get('count', 0)}")
                c.setFont("Helvetica", 10)
                col = 0
                row = 0

            x = x0 + col * col_w
            y = y0 - row * row_h
            img = ImageReader(path)
            c.drawImage(img, x, y - thumb, width=thumb, height=thumb, preserveAspectRatio=True, mask="auto")
            c.drawString(x, y - thumb - 12, label)

            col += 1
            if col >= 2:
                col = 0
                row += 1

        c.showPage()
    c.save()
    return out_path


def render_overview(results: Dict, out_dir: Path) -> None:
    st.header("Overview")
    if st.button("Export PDF"):
        path = _export_pdf(out_dir, results)
        st.success(f"Wrote {path}")
    clusters = results.get("clusters", [])
    cluster_pages = _cluster_to_pages(results)
    cluster_parts = defaultdict(list)
    for part in results.get("detected_parts", []):
        cid = int(part.get("cluster_id", -1))
        cluster_parts[cid].append(part)

    if "overrides" not in st.session_state:
        st.session_state["overrides"] = _load_overrides(out_dir)

    max_cluster = max([int(c.get("cluster_id", -1)) for c in clusters] or [0])

    sort_mode = st.selectbox("Sort clusters by", ["Cluster ID", "Size", "Hist similarity to largest", "Greedy similarity"], index=0)
    if sort_mode == "Size":
        clusters = sorted(clusters, key=lambda c: c.get("count", 0), reverse=True)
    elif sort_mode == "Hist similarity to largest" and clusters:
        largest = max(clusters, key=lambda c: c.get("count", 0))
        rep_path = (largest.get("examples") or [None])[0]
        rep_hist = None
        if rep_path and Path(rep_path).exists():
            img = cv2.imread(rep_path)
            if img is not None:
                rep_hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1], None, [24, 24], [0, 180, 0, 256])
                cv2.normalize(rep_hist, rep_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        if rep_hist is not None:
            def _hist_sim(c):
                ex = (c.get("examples") or [None])[0]
                if not ex or not Path(ex).exists():
                    return -1.0
                img = cv2.imread(ex)
                if img is None:
                    return -1.0
                h = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1], None, [24, 24], [0, 180, 0, 256])
                cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                return float(cv2.compareHist(rep_hist, h, cv2.HISTCMP_CORREL))
            clusters = sorted(clusters, key=_hist_sim, reverse=True)
    elif sort_mode == "Greedy similarity" and clusters:
        reps = {}
        for c in clusters:
            ex = (c.get("examples") or [None])[0]
            if not ex or not Path(ex).exists():
                continue
            img = cv2.imread(ex)
            if img is None:
                continue
            h = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1], None, [24, 24], [0, 180, 0, 256])
            cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            reps[int(c.get("cluster_id", -1))] = h

        remaining = {int(c.get("cluster_id", -1)) for c in clusters}
        ordered = []
        if remaining:
            current = int(max(clusters, key=lambda c: c.get("count", 0)).get("cluster_id", -1))
            ordered.append(current)
            remaining.discard(current)
            while remaining:
                best_id = None
                best_score = -1.0
                for cid in list(remaining):
                    if current not in reps or cid not in reps:
                        continue
                    score = float(cv2.compareHist(reps[current], reps[cid], cv2.HISTCMP_CORREL))
                    if score > best_score:
                        best_score = score
                        best_id = cid
                if best_id is None:
                    ordered.extend(sorted(list(remaining)))
                    break
                ordered.append(best_id)
                remaining.discard(best_id)
                current = best_id

        order_map = {cid: i for i, cid in enumerate(ordered)}
        clusters = sorted(clusters, key=lambda c: order_map.get(int(c.get("cluster_id", -1)), 1e9))

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

                merge_cols = st.columns([2, 1, 1])
                with merge_cols[0]:
                    merge_target = st.selectbox(
                        "Merge into",
                        [str(c) for c in sorted(cluster_parts.keys()) if c != cid],
                        key=f"merge_target_{cid}",
                    )
                with merge_cols[1]:
                    if st.button("Merge", key=f"merge_btn_{cid}") and merge_target:
                        target = int(merge_target)
                        for p in parts:
                            idx = int(p.get("index", -1))
                            st.session_state["overrides"][idx] = target
                        _save_overrides(out_dir, st.session_state["overrides"])
                        st.rerun()
                with merge_cols[2]:
                    if st.button("Merge Up", key=f"merge_up_{cid}"):
                        prev_idx = max(0, clusters.index(cluster) - 1)
                        if prev_idx != clusters.index(cluster):
                            target = int(clusters[prev_idx].get("cluster_id", -1))
                            for p in parts:
                                idx = int(p.get("index", -1))
                                st.session_state["overrides"][idx] = target
                            _save_overrides(out_dir, st.session_state["overrides"])
                            st.rerun()

                if st.button("Break cluster", key=f"break_{cid}"):
                    next_cluster = max_cluster + 1
                    for p in parts:
                        idx = int(p.get("index", -1))
                        st.session_state["overrides"][idx] = next_cluster
                        next_cluster += 1
                    _save_overrides(out_dir, st.session_state["overrides"])
                    st.rerun()

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
                            if st.button("Remove", key=f"rm_{cid}_{i}"):
                                idx = int(part.get("index", -1))
                                next_cluster = max_cluster + 1
                                st.session_state["overrides"][idx] = next_cluster
                                _save_overrides(out_dir, st.session_state["overrides"])
                                st.rerun()

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
    overrides = _load_overrides(out_dir)
    results = _apply_overrides(results, overrides)
    results["clusters"] = _rebuild_clusters(results)

    st.set_page_config(page_title="Instruction Matcher", layout="wide")
    st.title("Instruction Matcher")

    view = st.sidebar.radio("View", ["Overview", "Page Viewer", "Distance Matrix", "Uncertain Qty"], index=0)
    if view == "Overview":
        render_overview(results, out_dir)
    elif view == "Page Viewer":
        render_page_view(results, out_dir)
    elif view == "Distance Matrix":
        render_distance_matrix(out_dir)
    else:
        render_uncertain(out_dir)


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


def render_uncertain(out_dir: Path) -> None:
    st.header("Uncertain Quantities")
    uncertain_path = out_dir / "uncertain.json"
    if not uncertain_path.exists():
        st.info("No uncertain.json found. Re-run the extractor.")
        return

    entries = json.loads(uncertain_path.read_text(encoding="utf-8"))
    if not entries:
        st.info("No uncertain quantities found.")
        return

    for entry in entries:
        st.subheader(f"Page {int(entry['page_index']) + 1} · Part {int(entry['part_index']) + 1}")
        cols = st.columns(4)
        for col, key, label in zip(
            cols,
            ["callout_path", "part_path", "extended_path", "text_path"],
            ["Callout", "Part", "Extended", "Text"],
        ):
            with col:
                st.caption(label)
                p = Path(entry[key])
                if p.exists():
                    st.image(str(p), width=220)
                else:
                    st.caption("Missing")


if __name__ == "__main__":
    main()
