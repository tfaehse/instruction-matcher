"""Dash dashboard to browse instruction pages and detected parts."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, List

from dash import Dash, Input, Output, dcc, html


def _img_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ext = path.suffix.lower().lstrip(".") or "png"
    return f"data:image/{ext};base64,{b64}"


def _load_results(out_dir: Path) -> Dict:
    results_path = out_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}. Run main.py first.")
    return json.loads(results_path.read_text(encoding="utf-8"))


def _page_image_path(out_dir: Path, page_index: int) -> Path:
    return out_dir / "debug" / f"p{page_index:03d}_oriented.png"


def _parts_for_page(results: Dict, page_index: int) -> List[Dict]:
    return [p for p in results.get("detected_parts", []) if p.get("page_index") == page_index]


def build_app(results: Dict, out_dir: Path) -> Dash:
    app = Dash(__name__)

    max_page = 0
    if results.get("detected_parts"):
        max_page = max(int(p.get("page_index", 0)) for p in results["detected_parts"])

    app.layout = html.Div(
        style={"fontFamily": "Helvetica, Arial, sans-serif", "padding": "16px"},
        children=[
            html.H2("Instruction Matcher"),
            dcc.Slider(
                id="page-slider",
                min=0,
                max=max_page,
                value=0,
                step=1,
                marks={i: str(i + 1) for i in range(max_page + 1)},
            ),
            html.Div(id="page-view", style={"marginTop": "16px"}),
            html.H3("Parts", style={"marginTop": "24px"}),
            html.Div(id="parts-view", style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}),
        ],
    )

    @app.callback(
        Output("page-view", "children"),
        Output("parts-view", "children"),
        Input("page-slider", "value"),
    )
    def update_page(page_index: int):
        page_path = _page_image_path(out_dir, page_index)
        if page_path.exists():
            page_img = html.Img(
                src=_img_to_data_uri(page_path),
                style={"maxWidth": "100%", "border": "1px solid #ccc"},
            )
        else:
            page_img = html.Div(f"Missing page image: {page_path}")

        parts = _parts_for_page(results, page_index)
        part_cards = []
        for part in parts:
            crop_path = Path(part["crop_path"])
            if crop_path.exists():
                img = html.Img(src=_img_to_data_uri(crop_path), style={"width": "220px"})
            else:
                img = html.Div(f"Missing crop: {crop_path}")
            part_cards.append(
                html.Div(
                    style={"border": "1px solid #ddd", "padding": "8px", "width": "240px"},
                    children=[
                        html.Div(f"Qty: {part['qty']}"),
                        html.Div("Confident" if part.get("qty_confident") else "Uncertain"),
                        img,
                    ],
                )
            )

        return page_img, part_cards

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Dash dashboard for instruction matcher results")
    parser.add_argument("--out", type=str, default="out", help="Output directory from main.py")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    results = _load_results(out_dir)
    app = build_app(results, out_dir)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
