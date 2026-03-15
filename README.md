# RevenueSegmentationLLM
1. Using LLM to segment listed businesses using publicly available documents (llm_segmentation.py)
2. Download segments from GCP (download_segments.py)
3. Pre-process the segments data into a pandas df and add other required columns (outputs a csv and parquet output)
4. normalise the revenue segments using workflow (norm_segments.py)
5. post-process to get final dataframe


output from prepare_for_discovery.ipynb
No 1
-> data/output/dis_baselist_v1.csv 
-> base list of stocks that can be searched on tool 
-> contains all stock > 400 cr mcap with meta data (mcap , industry, screener link, etc)

No 2
-> some post processign of norm segments into a single list

output of clean_parquet.py

-> coreects mistakes of llm segmentation
