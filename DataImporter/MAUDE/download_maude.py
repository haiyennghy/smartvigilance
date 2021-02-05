import requests
import json
import sys
import zipfile
from pathlib import Path
import os

def download_url(url, save_path, file_name, chunk_size=128):
    r = requests.get(url, stream=True)
    file_path = Path(save_path) / file_name
    with open(file_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):#prevent loading the entire response into memory at once
            fd.write(chunk)
try:
    maude_save_path = Path(sys.argv[1])
except:
    print("Please provide a valid download path")
    exit()

reference_json_url = 'https://api.fda.gov/download.json'
reference_json_name = 'download.json'
download_url(reference_json_url,maude_save_path,reference_json_name)

ref_json_path = maude_save_path / reference_json_name
with open(ref_json_path, 'r') as f:
    reference_json = json.load(f)

for record in reference_json["results"]["device"]["event"]["partitions"]:
    url = record["file"]
    zip_name = url.split('/')[-2] + url.split('/')[-1]
    download_url(url,maude_save_path,zip_name)
    zip_path = maude_save_path / zip_name
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
         extract_path = maude_save_path / url.split('/')[-2]
         zip_ref.extractall(extract_path)
    os.remove(zip_path)
