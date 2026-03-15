import os
from dotenv import load_dotenv
from utils import download_all_jsons

load_dotenv()

bucket_name = os.getenv("GCP_BUCKET")
destination_folder = "data/output/llm_segments_jsons/v1"

if __name__ == "__main__":

    if True: #jsons download status
        print(f"Downloading JSONs from bucket '{bucket_name}' to '{destination_folder}'...")
        download_all_jsons(bucket_name, destination_folder)
    

