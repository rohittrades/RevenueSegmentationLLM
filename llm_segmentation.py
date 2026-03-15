from google import genai
from google.api_core import exceptions
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import pandas as pd
import numpy as np
import json
from datetime import datetime
import random
import time
import PyPDF2
import requests
from utils import upload_json, download_json, list_files_in_folder, get_prompt
from models import CompanyAnalysis
from pathlib import Path
import shutil
import os
from dotenv import load_dotenv
load_dotenv()


"""   CONFIGURATION   """

demo_run = False
local_run = False


""" Gemini configuration """

MAX_COUNT = 200 #Free gemini limit is 250

free_tier = False
if free_tier:
    api_key = os.getenv("GEMINI_FREE_KEY")
    use_model = os.getenv("LLM_MODEL_NAME")
else:
    api_key = os.getenv("GEMINI_PAID_KEY1")
    use_model = os.getenv("LLM_MODEL_NAME")

client = genai.Client(api_key=api_key)


""" GCP configuration """

bucket_name = os.getenv("GCP_BUCKET")
gcp_folder = "gendata/v1/"


""" Load prompts """

SYSTEM_PROMPT = get_prompt('sys_prompt.txt')


"""  PDF Handling  """

# Download PDF
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/pdf"
}

def is_valid_pdf(path):
    """Checks if the file starts with the PDF magic number %PDF-."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 5:
            return False
        with open(path, "rb") as f:
            header = f.read(5)
            return header.startswith(b"%PDF-")
    except Exception:
        return False

# Persistent session for connection pooling
session = requests.Session()
session.headers.update(HEADERS)

def load_file(path):
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return "" # Return empty string so the loop continues

def download_pdf_fast(url, out_dir, fname):
    if pd.isna(url):
        return False, None
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path):
        os.remove(out_path)
    try:
        # 1. Use 'with' to ensure the stream closes correctly
        with session.get(url, timeout=15, stream=True) as r:
            # 2. Immediately stop if the status is not 200-299
            r.raise_for_status() 
            
            # 3. Double-check Content-Type header if available
            if "application/pdf" not in r.headers.get("Content-Type", "").lower():
                print(f"Warning: URL may not be a PDF. Type: {r.headers.get('Content-Type')}")

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=32768): # Slightly larger chunks
                    f.write(chunk)

        # 4. Validation
        if is_valid_pdf(out_path):
            text = load_file(out_path)
            if not text:
                return False, None
            return True, out_path
        
        print(f"Validation failed for {fname}")
        if os.path.exists(out_path): os.remove(out_path)
        
    except requests.exceptions.RequestException as e:
        print(f"Download failed for {fname}: {e}")
        if os.path.exists(out_path): os.remove(out_path)

    return False, None

# download_pdf_fast('https://www.bseindia.com/xml-data/corpfiling/AttachHis/b2d477f4-4489-449e-b3e5-6c9ffbfae75a.pdf', 'tmp', 'y.pdf')


# Util Funcs
def process_with_retry(client, model_name, contents, config):
    """Wrapper to handle 429 Resource Exhausted errors."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
        except exceptions.ResourceExhausted as e:
            # Exponential backoff: 2, 4, 8, 16... seconds + jitter
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"⚠️ Rate limit hit. Waiting {wait:.2f}s...")
            time.sleep(wait)
    raise Exception("Max retries exceeded.")

def wait_for_files_active(client, files):
    """
    Waits for all uploaded files to reach the 'ACTIVE' state.
    """
    print("⏳ Waiting for files to process...")
    for f in files:
        while True:
            # Refresh file metadata
            current_file = client.files.get(name=f.name)
            
            if current_file.state.name == "ACTIVE":
                break
            elif current_file.state.name == "FAILED":
                raise Exception(f"File {f.display_name} failed to process.")
            
            # Progress indicator
            print(f"  - {f.display_name} is {current_file.state.name}...")
            time.sleep(2)
    print("✅ All files ready!")

def downl_upl_files(doc_urls, dl_dir, client):

    files_to_send = []
    for doc_type, doc_url in doc_urls.items():
        if doc_type == 'ar_or_dhrp':
            ar_status, ar_path = download_pdf_fast(doc_url, dl_dir, 'annual_report.pdf')
            if ar_status:
                file1 = client.files.upload(file=ar_path, config={'display_name': 'Latest Annual Report'})
                files_to_send.append(file1)
        elif doc_type == 'ppt':
            ppt_status, ppt_path = download_pdf_fast(doc_url, dl_dir, 'quarter_ppt.pdf')
            if ppt_status:
                file2 = client.files.upload(file=ppt_path, config={'display_name': 'FY 2026 Latest Quarterly PPT'})
                files_to_send.append(file2)
        elif doc_type == 'concall':
            concall_status, concall_path = download_pdf_fast(doc_url, dl_dir, 'concall.pdf')
            if concall_status:
                file3 = client.files.upload(file=concall_path, config={'display_name': 'FY 2026 Latest Concall Transcript'})
                files_to_send.append(file3)

    return files_to_send

def load_processed_names_local(bucket, folder_path):

    json_filenames = [f.replace('.json', '') for f in os.listdir(folder_path) if f.endswith('.json')]
    return json_filenames

def load_processed_names(bucket, folder_path):

    done_ids = []
    company_files = list_files_in_folder(bucket, folder_path)
    for file_dir in company_files:
        filename = file_dir.split('/')[-1]
        stock_id = filename.split('.')[0]
        done_ids.append(stock_id)
    return done_ids


def main_task():

    # Load processed stocks
    processed_ids = load_processed_names(bucket_name, gcp_folder) 
    print("Total processed stocks : ", len(processed_ids))

    # Load stocks to process
    stocks_df = pd.read_csv('data/input/all_stocks_link_0226.csv') 
    stocks_df.sort_values(by='mcap', ascending=True, inplace=True)

    # get latest documents 
    stocks_df['ar_or_dhrp'] = stocks_df['ar_2025'].combine_first(stocks_df['ar_2024']).combine_first(stocks_df['dhrp'])
    stocks_df['ppt'] = stocks_df['q3_26_ppt'].combine_first(stocks_df['q2_26_ppt']).combine_first(stocks_df['q1_26_ppt'])
    stocks_df['concall'] = stocks_df['q3_26_concall'].combine_first(stocks_df['q2_26_concall']).combine_first(stocks_df['q1_26_concall'])

    if demo_run:
        stocks_df = stocks_df.sample(1)
        
    # stocks_df = stocks_df[stocks_df['nse_code'].isin(['SRF', 'FLUOROCHEM'])]
    # stocks_df = stocks_df[stocks_df['nse_code']=='AARTIIND']
    # stocks_df.head(3)

    """ MAIN RUN """
    counter = 0
    failed = []

    for idx, row in stocks_df.iterrows():

        if row['id'] in processed_ids:
            print('👍🏼 Already processed : ', row['company_name'])
            continue
        
        counter += 1
        if counter > MAX_COUNT:
            print(" - - - - - - - - - - - - - ")
            print("Maximum limit reached! ⛔️ ")
            print(" - - - - - - - - - - - - - ")
            break

        print("Processing stock : ", row['company_name'])
        #create directory
        dl_path = Path('data/tmp/documents/'+row['id'])
        dl_path.mkdir(parents=True, exist_ok=True)

        #prepare document urls
        url_cols = ['ar_or_dhrp', 'ppt', 'concall']
        files = {k: row[k] for k in url_cols if k in row and row[k] is not None}

        if not files:
            print("⛔️ No URLs available - counting as failure ⛔️")
            counter -= 1
            failed.append(row['company_name'])
            continue
        #download files - AR or DRHP / latest Q ppt / latest Q concall
        file_references = downl_upl_files(files, dl_path, client)

        if not file_references:
            print("⛔️ All downloads failed - no attachments to process ⛔️")
            counter -= 1
            failed.append(row['company_name'])
            if dl_path.exists() and dl_path.is_dir():
                shutil.rmtree(dl_path)
            continue

        # Wait for everything in the list
        wait_for_files_active(client, file_references)

        sys_prompt = SYSTEM_PROMPT.format(company_name=row['company_name'])
        file_references.append(sys_prompt)

        try:
            # Get the pre-flight token count
            token_count = client.models.count_tokens(
                model=use_model,
                contents=file_references
            )
            total = token_count.total_tokens
            print(f"📊 Estimated Total Tokens: {total:,}")

            # Safety Check for Free Tier (1 Million limit)
            if total > 300_000:
                print("⚠️ PROMPT TOO LARGE: Skipping or requires manual chunking.")
                counter -= 1
                if dl_path.exists() and dl_path.is_dir():
                    shutil.rmtree(dl_path)
                continue
            
            # Final API Call
            response = process_with_retry(
                client,
                use_model,
                file_references,
                {
                    "response_mime_type": "application/json",
                    "response_schema": CompanyAnalysis,
                }
            )
            
            # TODO clean llm memory 
            # NOT needed now as we aren't caching the inputs
            # We can use caching later on when extracting in stages

            # Dump ONLY the extracted data to JSON with beautiful indenting
            ar = response.parsed 
            if not local_run:
                upload_json(bucket_name, f"gendata/v1/{row['id']}.json", ar)
            else:
                data_to_save = ar.model_dump_json(indent=4)
                Path(f"data/output/gendata_v2/{row['id']}.json").write_text(data_to_save)
            print(f"✅ Clean JSON saved - {row['company_name']}")     

            if dl_path.exists() and dl_path.is_dir():
                shutil.rmtree(dl_path)
            # MANDATORY SPACER for TPM/RPM - free tier mode
            if free_tier:
                time.sleep(4) 

        except Exception as e:
            print(f"❌ Failed to process {row['company_name']}: {e}")
            failed.append(row['company_name'])
        print(" - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * ")


if __name__ == "__main__":
    main_task()