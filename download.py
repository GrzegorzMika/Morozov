import os

from google.cloud import storage
from tqdm import tqdm


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


storage_client = storage.Client.from_service_account_json(find('secretgc_ip.json', '/home'))

blobs = storage_client.list_blobs('ip-free')
for blob in tqdm(blobs):
    blob.download_to_filename(blob.name)
