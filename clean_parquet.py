import pandas as pd
import re

INPUT     = "dis_rev_segment_v1.parquet"
BASELIST  = "dis_baselist_v1.csv"
OUTPUT    = "dis_rev_segment_clean.parquet"

df   = pd.read_parquet(INPUT)
base = pd.read_csv(BASELIST)[["stock_id", "name"]]

# Find company names that appear across multiple stock_ids (contamination signal)
company_stock_counts = df.groupby("company_name")["stock_id"].nunique()
contaminating_names  = company_stock_counts[company_stock_counts > 1].index

STOP = {"limited", "ltd", "private", "pvt", "the", "and", "of", "india",
        "co", "company", "industries", "enterprises", "group", "holdings",
        "services", "solutions", "technologies", "technology"}

def sig_words(s):
    if pd.isna(s):
        return set()
    tokens = re.sub(r"[^a-z ]", " ", str(s).lower()).split()
    return {t for t in tokens if t not in STOP and len(t) >= 3}

# For each contaminating company, identify imposter stock_ids:
# a stock_id is an imposter if the baselist name for that stock_id shares
# NO significant words with the contaminating company_name it carries.
imposter_stock_ids = set()

for company_name in contaminating_names:
    group = df[df["company_name"] == company_name][["stock_id"]].drop_duplicates()
    company_words = sig_words(company_name)

    for sid in group["stock_id"]:
        base_row = base[base["stock_id"] == sid]
        if base_row.empty:
            # Not in baselist at all — definitely imposter
            imposter_stock_ids.add(sid)
            continue
        expected_name = base_row.iloc[0]["name"]
        expected_words = sig_words(expected_name)
        if not (company_words & expected_words):
            # No word overlap between parquet company_name and baselist expected name
            imposter_stock_ids.add(sid)

before = len(df)
clean  = df[~df["stock_id"].isin(imposter_stock_ids)].copy()
after  = len(clean)

print(f"Dropped {before - after} rows from {len(imposter_stock_ids)} imposter stock_ids")
print(f"Kept {after} rows across {clean['stock_id'].nunique()} stocks")
print(f"\nImposter stock_ids removed ({len(imposter_stock_ids)}):")
for sid in sorted(imposter_stock_ids):
    base_row = base[base["stock_id"] == sid]
    expected = base_row.iloc[0]["name"] if not base_row.empty else "NOT IN BASELIST"
    parquet_name = df[df["stock_id"] == sid]["company_name"].iloc[0]
    print(f"  {sid:30s}  baselist={expected!r:40s}  parquet={parquet_name!r}")

clean.to_parquet(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
