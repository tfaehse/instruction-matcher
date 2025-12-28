# instruction_matcher

Vibecoded prototype to count LEGO parts from instruction PDFs by reading the callout boxes per step.

## Requirements

- Python 3.12+
- `uv` (recommended)
- OCR uses EasyOCR (downloads models on first run)

Install deps:

```bash
uv sync
```

## Run the extractor

Process a page range (1-based, inclusive):

```bash
uv run python ./main.py --start 1 --end 20 --out out ./merged.pdf
```

Outputs:
- `out/results.json` (detections, clusters, counts)
- `out/debug/` (rendered pages, callouts, part crops)

## Dashboard (Streamlit)

```bash
uv run streamlit run streamlit_app.py -- --out out
```

## Project layout

- `instruction_matcher/cli.py` entrypoint
- `instruction_matcher/pipeline.py` main pipeline
- `instruction_matcher/callout.py` callout detection + part extraction
- `instruction_matcher/ocr.py` fast qty OCR (template-based)
- `instruction_matcher/orientation.py` orientation scoring
- `instruction_matcher/clustering.py` global clustering
- `streamlit_app.py` dashboard

## Notes

- Orientation is chosen by scoring readability of callout quantities.
- Clustering uses color hist + ORB + pHash on 512×512 normalized crops.
- Part crops saved in `out/debug` are already normalized.

## Profiling

cProfile:

```bash
uv run python -m cProfile -o out/profile.prof ./main.py --start 1 --end 20 --out out ./merged.pdf
```

Memory profile (py-spy):

```bash
py-spy record --mem --format flamegraph -o out/profile_mem.svg -- uv run python ./main.py --start 1 --end 20 --out out ./merged.pdf
```
