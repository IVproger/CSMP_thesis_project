import requests
import os
from pathlib import Path

def download_db():
    url = "https://huggingface.co/datasets/Tingxie/CSU-MS2-DB/resolve/main/csu_ms2_db.db"
    filename = "../csu_db_storage/csu_ms2_db.db"
    
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    
    print(f"\nDownload complete! File saved as {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    download_db()