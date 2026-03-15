# RevenueSegmentationLLM

LLM-based pipeline to extract and classify revenue segments of listed Indian businesses using publicly available filings (Annual Reports, Quarterly PPTs, Concall Transcripts).

## Pipeline

1. **`llm_segmentation.py`** — Downloads PDFs from URLs, uploads to Gemini, extracts structured revenue segment data per company. Saves JSONs to GCP bucket.
2. **`download_segments.py`** — Downloads all extracted JSONs from GCP locally.
3. **`preprocess_segments.py`** — Converts JSONs into a pandas DataFrame. Computes `effective_revenue`, `revenue_pct`, and YoY changes. Outputs `.parquet` and `.csv`.
4. **`norm_segments.py`** — Classifies each revenue segment into GICS sub-industries using FAISS semantic search + Gemini. Saves progress checkpoints every 50 rows.
5. **`clean_parquet.py`** — Detects and removes misattributed segments (LLM cross-contamination) using word-overlap matching.
6. **`prepare_for_discovery.ipynb`** — Final post-processing. Filters low-confidence GICS predictions (<0.45), consolidates predictions into list columns, generates screener links.

## Outputs

| File | Description |
|------|-------------|
| `dis_baselist_v1.csv` | Base list of stocks >400cr mcap with metadata (mcap, industry, screener link) |
| `dis_rev_segment_clean.parquet` | Cleaned revenue segments with GICS classifications |

## Configuration

Key settings at the top of `llm_segmentation.py`:
- `demo_run` — set `True` to test on a single random stock
- `local_run` — set `True` to save output locally instead of GCP
- `MAX_COUNT` — max stocks to process per run (default 200)

Required env vars (`.env`):
```
GEMINI_PAID_KEY1=
GEMINI_FREE_KEY=
LLM_MODEL_NAME=
GCP_BUCKET=
GCP_CREDENTIALS=
```
