import json
import pandas as pd
from pathlib import Path
from utils import GICSAutomator

JSONS_DIR = Path('data/output/segments/v1')
OUTPUT_DIR = Path('data/output')


def load_df(jsons_dir: Path) -> pd.DataFrame:
    rows = []

    for filepath in jsons_dir.glob('*.json'):
        stock_id = filepath.stem
        parts = filepath.stem.rsplit('_', 1)
        nse_id = parts[0]
        bse_id = parts[1] if len(parts) == 2 else None

        with open(filepath) as f:
            data = json.load(f)
        if not data:
            print(f'Warning: No data in {filepath}')
            continue
        company_name = data.get('company_name')
        for vertical in data.get('verticals', []):
            rows.append({
                'stock_id': stock_id,
                'nse_id': nse_id,
                'bse_id': bse_id,
                'company_name': company_name,
                'business_segment': vertical.get('business_segment'),
                'segment_summary': vertical.get('segment_summary'),
                'status': vertical.get('status'),
                'commission_year': vertical.get('commission_year'),
                'has_domestic_presence': vertical.get('has_domestic_presence'),
                'has_international_presence': vertical.get('has_international_presence'),
                'capex_status': vertical.get('capex_status'),
                'capex_value_inr_crores': vertical.get('capex_value_inr_crores'),
                'capex_details': vertical.get('capex_details'),
                'is_segment_revenue_disclosed': vertical.get('is_segment_revenue_disclosed'),
                'segment_revenue': vertical.get('segment_revenue'),
                'previous_year_revenue': vertical.get('previous_year_revenue'),
                'client_concentration': vertical.get('client_concentration'),
                'products_or_services': vertical.get('products_or_services', []),
                'operational_locations': vertical.get('operational_locations', []),
                'domestic_reach_states': vertical.get('domestic_reach_states', []),
                'international_markets': vertical.get('international_markets', []),
                'top_clients': vertical.get('top_clients', []),
            })

    df = pd.DataFrame(rows)
    df['commission_year'] = pd.array(df['commission_year'], dtype=pd.Int64Dtype())
    df['segment_revenue'] = pd.to_numeric(df['segment_revenue'], errors='coerce')
    df['previous_year_revenue'] = pd.to_numeric(df['previous_year_revenue'], errors='coerce')
    df['capex_value_inr_crores'] = pd.to_numeric(df['capex_value_inr_crores'], errors='coerce')
    return df


def _add_revenue_pct(group: pd.DataFrame) -> pd.DataFrame:
    """Applied per-company group to compute revenue % across verticals.

    Uses segment_revenue if all verticals have it, else falls back to
    previous_year_revenue for all. Missing values yield None.
    """
    effective_revenue = (
        group['segment_revenue'] if group['segment_revenue'].notna().all()
        else group['previous_year_revenue']
    )

    total = effective_revenue.sum()
    if total > 0:
        revenue_pct = (effective_revenue / total * 100).round(2)
        revenue_pct = revenue_pct.where(effective_revenue.notna(), other=None)
    else:
        revenue_pct = pd.Series([None] * len(group), index=group.index, dtype=object)

    group = group.copy()
    group['effective_revenue'] = effective_revenue
    group['revenue_pct'] = revenue_pct
    return group


def add_revenue_pct(df: pd.DataFrame) -> pd.DataFrame:
    # Group by stock_id (unique per file) not ticker — multiple companies share
    # ticker='missing' when the NSE symbol was unavailable at generation time.
    # Pass .to_numpy() instead of column name string — pandas 3.0 drops the
    # groupby key column from results when passed as a string.
    return df.groupby(df['stock_id'].to_numpy(), group_keys=False).apply(_add_revenue_pct)


def add_yoy_revenue_change(df: pd.DataFrame) -> pd.DataFrame:
    """Computes YoY % change per segment row where both revenue values are available.

    yoy_revenue_change = (segment_revenue - previous_year_revenue) / previous_year_revenue * 100
    Yields None when either value is missing or previous_year_revenue is zero.
    """
    curr = df['segment_revenue']
    prev = df['previous_year_revenue']

    valid = curr.notna() & prev.notna() & (prev != 0)
    df = df.copy()
    df['yoy_revenue_change'] = None
    df.loc[valid, 'yoy_revenue_change'] = ((curr[valid] - prev[valid]) / prev[valid] * 100).round(2)
    return df


def save(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure stock_id is always the first column
    cols = ['stock_id'] + [c for c in df.columns if c != 'stock_id']
    df = df[cols]

    list_cols = ['products_or_services', 'operational_locations', 'domestic_reach_states',
                 'international_markets', 'top_clients']
    pydantic_cols = ['normalized_business_segment']

    df_parquet = df.copy()
    for col in pydantic_cols:
        if col in df_parquet.columns:
            df_parquet[col] = df_parquet[col].apply(
                lambda v: v.model_dump_json() if v is not None else None
            )
    df_parquet.to_parquet(output_dir / 'revenue_segments.parquet', index=False)

    df_csv = df.copy()
    for col in list_cols:
        df_csv[col] = df_csv[col].apply(json.dumps)
    for col in pydantic_cols:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].apply(
                lambda v: v.model_dump_json() if v is not None else None
            )
    df_csv.to_csv(output_dir / 'revenue_segments.csv', index=False)

    print(f'Saved to {output_dir}/')


def main() -> None:
    df = load_df(JSONS_DIR)
    print(f'Loaded {df.shape[0]} rows, {df["stock_id"].nunique()} companies')
    print('Sample data:')
    print(df.head())

    # add revenue_pct and effective_revenue columns per company
    df = add_revenue_pct(df)

    # add yoy segment revenue change where both current and previous year revenue are available
    df = add_yoy_revenue_change(df)

    save(df, OUTPUT_DIR)


if __name__ == '__main__':
    main()
