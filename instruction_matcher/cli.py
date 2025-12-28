from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import build_results, process, write_results
from .utils import mkdirp


def main() -> None:
    parser = argparse.ArgumentParser(description="Count LEGO parts from instruction PDF callouts")
    parser.add_argument("pdf", type=str, help="Path to instruction PDF")
    parser.add_argument("--start", type=int, required=True, help="Start page (1-based, inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End page (1-based, inclusive)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    detected, clusters = process(pdf_path, start_page=args.start, end_page=args.end, dpi=args.dpi, out_dir=out_dir)
    results = build_results(pdf_path, detected, clusters, args.start, args.end)

    mkdirp(out_dir)
    out_json = write_results(out_dir, results)

    print(f"Wrote: {out_json}")
    print(f"Clusters: {results['total_clusters']}")
    print(f"Total parts (confident): {results['total_parts_confident']}")
    print(f"Total parts (including uncertain): {results['total_parts_including_uncertain']}")
    if any(not d["qty_confident"] for d in results["detected_parts"]):
        print("[warn] Some quantities were uncertain and defaulted to 1.")


if __name__ == "__main__":
    main()
