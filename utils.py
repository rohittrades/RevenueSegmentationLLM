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