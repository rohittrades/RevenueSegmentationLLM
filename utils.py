import json
import os
from google.cloud import storage

import json
import os
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

def get_gcs_client():
    """Creates a GCS client using the environment variable JSON string."""
    creds_json_str = os.getenv("GCP_CREDENTIALS")
    
    if not creds_json_str:
        raise ValueError("GCP_CREDENTIALS environment variable is not set.")

    # Clean the string (removes potential hidden newlines from .env)
    clean_creds_str = creds_json_str.replace('\n', '').replace('\r', '')
    
    try:
        creds_dict = json.loads(clean_creds_str)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        return storage.Client(credentials=credentials, project=creds_dict.get('project_id'))
    except json.JSONDecodeError as e:
        print(f"Failed to parse GCP_CREDENTIALS: {e}")
        raise

# 1. Point to your credentials

def upload_json(bucket_name, destination_blob_name, data):

    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    payload = data.model_dump_json(indent=4) if hasattr(data, 'model_dump_json') else json.dumps(data, indent=4)

    blob.upload_from_string(payload, content_type='application/json')
    print(f"File {destination_blob_name} uploaded to {bucket_name}.")

# Example Usage:
# my_data = {"id": 123, "content": "Hello from Gemini"}
# upload_json("super_trader_bucket1", "data_001.json", my_data)

def upload_file(bucket_name, destination_blob_name, local_path):
    """Uploads any local file to GCS (parquet, faiss index, etc.)."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} → gs://{bucket_name}/{destination_blob_name}")

def download_file(bucket_name, source_blob_name, local_path):
    """Downloads a file from GCS to a local path. Returns True if found, False if not found."""
    import pathlib
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    if not blob.exists():
        print(f"gs://{bucket_name}/{source_blob_name} not found — skipping download.")
        return False

    pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded gs://{bucket_name}/{source_blob_name} → {local_path}")
    return True

def download_json(bucket_name, source_blob_name):
    """Downloads a JSON blob from GCS and returns it as a dict."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    raw_text = blob.download_as_text()
    return json.loads(raw_text)

# Usage:
# data = download_json("super_trader_bucket1", "portfolios/user_42.json")
# print(data["user_id"])

def list_files_in_folder(bucket_name, folder_prefix):
    """Lists files that start with a specific prefix (folder)."""
    client = get_gcs_client()
    blobs = client.list_blobs(bucket_name, prefix=folder_prefix)
    return [blob.name for blob in blobs]

# Usage
# company_files = list_files_in_folder("super_trader_bucket1", "analysis/Apple/")

#Download jsons from google bucket
def download_all_jsons(bucket_name, destination_folder):
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    blobs = bucket.list_blobs()
    count = 0
    for blob in blobs:
        if blob.name.startswith("gendata/v1/") and blob.name.endswith(".json"):
            local_path = os.path.join(destination_folder, blob.name.split('/')[-1])
            blob.download_to_filename(local_path)
            count += 1
    print(f"Finished! Downloaded {count} files.")


# # Usage:
# download_all_jsons("super_trader_bucket1", "data/output/gendata_v2_test")

def get_prompt(file_path):
    """
    Reads a prompt from a text file and returns it as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # .strip() removes leading/trailing whitespace/newlines
            prompt = file.read().strip()
            return prompt
    except FileNotFoundError:
        return "Error: The file was not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# TARGET Industry Classifier
import pandas as pd
import numpy as np
from typing import List, Optional, Literal
import faiss
from google import genai
from sentence_transformers import SentenceTransformer
from pydantic import ValidationError, create_model, BaseModel, Field
from models import IndustryPrediction, GICSResponse
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_PAID_KEY1")
use_model = os.getenv("LLM_MODEL_NAME")

class GICSAutomator:
    def __init__(self, csv_path, index_path):
        self.csv_path = csv_path
        self.index_path = index_path
        self.model_name = 'all-MiniLM-L6-v2'
        
        # 1. Lazy-load the embedder only when needed
        self._embedder = None 
        
        # 2. Load Taxonomy Metadata
        self.df = pd.read_csv(csv_path)

        # 3. Load or Build Index
        self.index = self._load_or_build_index()
        
        # 4. Initialize Gemini
        self.client = genai.Client(api_key=api_key)

    @property
    def embedder(self):
        """Getter that loads the model into memory only on first use."""
        if self._embedder is None:
            print(f"Loading {self.model_name} into memory...")
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _load_or_build_index(self):
        """Checks if a pre-computed FAISS index exists; otherwise builds it."""
        if os.path.exists(self.index_path):
            print(f"Loading existing index from {self.index_path}")
            return faiss.read_index(self.index_path)
        
        print("Index not found. Building new vector store...")
        # Create the rich search string
        search_texts = (
            self.df['Sector'] + " > " + 
            self.df['IndustryGroup'] + " > " + 
            self.df['Industry'] + " > " + 
            self.df['SubIndustry'] + ": " + 
            self.df['SubIndustryDescription']
        ).tolist()
        
        embeddings = self.embedder.encode(search_texts, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save for next time
        faiss.write_index(index, self.index_path)
        return index

    def get_prompt(self, file_path):
        """
        Reads a prompt from a text file and returns it as a string.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # .strip() removes leading/trailing whitespace/newlines
                prompt = file.read().strip()
                return prompt
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading prompt file {file_path}: {e}") from e

    def llm(self, prompt, dynamic_schema, retries: int = 3):
        """Calls Gemini with structured JSON output and smart retry logic.

        Error handling strategy:
        - ValidationError       → return error dict immediately (schema mismatch, retrying won't help)
        - Rate limit (429)      → retry with long backoff: 60s, 120s, 180s
        - Quota / billing fatal → raise immediately so the caller can abort the whole run
        - Other transient error → retry with short backoff: 5s, 10s, 15s
        """
        import time

        # Signals that the API key / project is out of credits — retrying is pointless
        _FATAL_SIGNALS = (
            'billing', 'payment', 'out of credit', 'quota exceeded',
            'monthly limit', 'daily limit', 'project quota',
        )
        # Signals for per-minute / per-day rate limits — retry after a long wait
        _RATE_LIMIT_SIGNALS = (
            '429', 'resource_exhausted', 'rate limit', 'too many requests',
            'quota_exceeded',
        )

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=use_model or "gemini-2.5-flash",
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": dynamic_schema,
                    }
                )
                return dynamic_schema.model_validate_json(response.text)

            except ValidationError as e:
                # Schema mismatch — retrying won't help
                return {"error": "Validation failed", "details": e.errors()}

            except Exception as e:
                last_error = e
                err_lower = str(e).lower()

                # Hard stop — quota / billing exhausted; bubble up to abort the run
                if any(sig in err_lower for sig in _FATAL_SIGNALS):
                    print(f"\n[FATAL] Quota/billing error detected: {e}")
                    raise

                if attempt < retries:
                    if any(sig in err_lower for sig in _RATE_LIMIT_SIGNALS):
                        wait = 60 * attempt   # 60s, 120s, 180s for rate limits
                        print(f"  Rate limit hit — waiting {wait}s before retry {attempt+1}/{retries}")
                    else:
                        wait = 5 * attempt    # 5s, 10s, 15s for other transient errors
                    time.sleep(wait)

        return {"error": str(last_error)}


    def create_dynamic_sector_model(self, valid_sub_inds):

        # Override sub_industry_name with a Literal to constrain the LLM to
        # only the FAISS candidates. Using the same field name as the base class
        # keeps the schema consistent — no duplicate / conflicting fields.
        DynamicSubIndPrediction = create_model(
            "DynamicSubIndPrediction",
            sub_industry_name=(Literal[tuple(valid_sub_inds)], ...),
            __base__=IndustryPrediction
        )
        DynamicResponse = create_model(
            "DynamicGICSResponse",
            predictions=(list[DynamicSubIndPrediction], ...),
            __base__=GICSResponse
        )
        return DynamicResponse


    def categorize_project(self, company_name, segment_name, segment_description, products_services):
        query = segment_name + " " + segment_description + " products/services: " + ", ".join(products_services)
        query_vec = self.embedder.encode([query]).astype('float32')
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, k=20)
        candidates = self.df.iloc[indices[0]].copy()
        candidates['similarity'] = scores[0]

        # Extract unique Sub-Industry names from your search results
        valid_names = candidates['SubIndustry'].unique().tolist()
        valid_names.append("Other / Unclassified") # The escape hatch

        # Create a Dynamic Pydantic Model on the fly
        # This forces the LLM to choose ONLY from the FAISS shortlist
        ResponseModel = self.create_dynamic_sector_model(valid_names)

        # Format context for prompt
        candidate_context = ""
        for _, row in candidates.iterrows():
            candidate_context += (
                f"- {row['SubIndustry']} (Similarity: {row['similarity']:.2f}): "
                f"{row['SubIndustryDescription']}\n"
            )

        prompt = self.get_prompt('prompts/industryclassifier.txt').format(
            company_name=company_name,
            segment_name=segment_name,
            segment_description=segment_description,
            products_services=products_services,
            candidate_context=candidate_context
        )
        return self.llm(prompt, ResponseModel)