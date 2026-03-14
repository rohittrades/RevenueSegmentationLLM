import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import GICSAutomator, upload_file, download_file

PARQUET_PATH = Path('data/output/revenue_segments.parquet')
CHECKPOINT_EVERY = 50  # upload to GCS after every N rows

# GCS sync — enabled automatically when GCP_BUCKET env var is set (GitHub Actions)
GCP_BUCKET = os.getenv('GCP_BUCKET')
GCS_PARQUET_BLOB = 'data/revenue_segments.parquet'

# Optional cap on rows processed per run (set MAX_ROWS env var; 0 = no limit)
MAX_ROWS = int(os.getenv('MAX_ROWS', '0'))

# Flat column names written per row
GICS_COLS = (
    ['gics_pred_1', 'gics_conf_1',
     'gics_pred_2', 'gics_conf_2',
     'gics_pred_3', 'gics_conf_3',
     'gics_reasoning']
)


def flatten_result(result) -> dict:
    """Converts a GICSResponse (up to 3 predictions) into a flat dict.

    If the LLM call failed (result is an error dict), all values are None.
    Reasoning from all predictions is combined into a single pipe-separated string.
    """
    out = {col: None for col in GICS_COLS}
    if isinstance(result, dict):   # error path from GICSAutomator.llm()
        return out

    reasoning_parts = []
    for i, pred in enumerate(result.predictions[:3], 1):
        out[f'gics_pred_{i}'] = pred.sub_industry_name
        out[f'gics_conf_{i}'] = round(pred.confidence_score, 4)
        reasoning_parts.append(f'[{pred.sub_industry_name}] {pred.reasoning}')

    out['gics_reasoning'] = ' | '.join(reasoning_parts)
    return out


def sync_to_gcs(label: str = ''):
    """Uploads the local parquet to GCS if GCP_BUCKET is configured."""
    if GCP_BUCKET:
        upload_file(GCP_BUCKET, GCS_PARQUET_BLOB, str(PARQUET_PATH))
        if label:
            print(f'  GCS synced — {label}')


def main():
    # Pull the latest parquet from GCS so this run resumes from where the last left off.
    # Falls back to a local file if GCS is not configured (local dev).
    if GCP_BUCKET:
        print(f'Pulling latest parquet from GCS (gs://{GCP_BUCKET}/{GCS_PARQUET_BLOB})...')
        found = download_file(GCP_BUCKET, GCS_PARQUET_BLOB, str(PARQUET_PATH))
        if not found:
            raise FileNotFoundError(
                f'Parquet not found in GCS at {GCS_PARQUET_BLOB}. '
                'Run preprocess_segments.py locally first and upload the output.'
            )

    df = pd.read_parquet(PARQUET_PATH)

    # Initialize GICS columns if not present (first run), leave existing values
    # intact so a resumed run picks up where it left off.
    for col in GICS_COLS:
        if col not in df.columns:
            df[col] = None

    gics_automator = GICSAutomator(
        csv_path='data/gics_map_2023.csv',
        index_path='data/gics_index.faiss'
    )

    # Resume: skip rows already classified or with no business_segment
    pending_mask = df['gics_pred_1'].isna() & df['business_segment'].notna()
    pending_idx = df.index[pending_mask].tolist()

    if MAX_ROWS > 0:
        pending_idx = pending_idx[:MAX_ROWS]
        print(f'MAX_ROWS={MAX_ROWS} — capping this run.')

    done = len(df) - df['gics_pred_1'].isna().sum()
    print(f'Total: {len(df)} rows | Already done: {done} | Pending this run: {len(pending_idx)}')

    for count, idx in enumerate(tqdm(pending_idx, desc='Classifying segments'), 1):
        row = df.loc[idx]
        result = gics_automator.categorize_project(
            row['company_name'],
            row['business_segment'],
            row['segment_summary'],
            row['products_or_services']
        )
        for col, val in flatten_result(result).items():
            df.at[idx, col] = val

        if count % CHECKPOINT_EVERY == 0:
            df.to_parquet(PARQUET_PATH, index=False)
            sync_to_gcs(f'checkpoint {count}/{len(pending_idx)}')

    df.to_parquet(PARQUET_PATH, index=False)
    sync_to_gcs('final')
    print(f'Done. Saved to {PARQUET_PATH}')


if __name__ == '__main__':
    main()
