# Code taken form GLIDE
import os
from functools import lru_cache
from typing import Dict, Optional

import requests
import torch as th
from filelock import FileLock
from tqdm.auto import tqdm
import gdown
@lru_cache()
def default_cache_dir() -> str:
    return os.path.join(os.path.abspath(os.getcwd()), "weights")


MODEL_PATHS = {
    "FaRL-Base-Patch16-LAIONFace20M-ep64":"https://www.dropbox.com/s/eid6rnsii1oewgf/FaRL-Base-Patch16-LAIONFace20M-ep64.pth?dl=1",
    "model_latest": "https://www.dropbox.com/s/st88q851w8mkexk/model_latest.pt?dl=1",
    "model_sketch": "https://www.dropbox.com/s/5wzq7wvi20pcj5b/model_sketch.pt?dl=1",
    "base": "https://www.dropbox.com/s/i2ufli8n83y1sll/base.pt?dl=1",
    "upsample": "https://www.dropbox.com/s/5xndsn3y529dj32/upsample.pt?dl=1",
    "64x64_diffusion": "https://www.dropbox.com/s/tltbh4atedp4ree/64x64_diffusion.pt?dl=1",
}


LOCAL_PATHS ={
    "base": "./weights/base.pt",
    "upsample": "./weights/upsample.pt",
    "64x64_diffusion": "./weights/64x64_diffusion.pt",
    "FaRL-Base-Patch16-LAIONFace20M-ep64": "./weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
    "model_latest": "./weights/model_latest.pt",
    "model_sketch": "./weights/model_sketch.pt"  
}

if(os.path.exists('./weights')==False):
    os.mkdir('./weights')
#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)  

def fetch_file_cached(
    url: str, key: str ,progress: bool = True, cache_dir: Optional[str] = None, chunk_size: int = 4096
) -> str:
    """
    Download the file at the given URL into a local file and return the path.
    If cache_dir is specified, it will be used to download the files.
    Otherwise, default_cache_dir() is used.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = LOCAL_PATHS[key]
    print(local_path)
    if os.path.exists(local_path):
        return LOCAL_PATHS[key]
    response = requests.get(url,stream=True)
    size = int(response.headers.get("content-length", "0"))
    with FileLock(local_path + ".lock"):
        if progress:
            pbar = tqdm(total=size, unit="iB", unit_scale=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if progress:
                    pbar.update(len(chunk))
                f.write(chunk)
        os.rename(tmp_path, local_path)
        if progress:
            pbar.close()
        return local_path


# for _ in GLINK_PATHS:
#     print(_)
#     url=GLINK_PATHS[_]
#     output=LOCAL_PATHS[_]
#     download_file_from_google_drive(url, output)

def download_files():
    for _ in MODEL_PATHS:
        model= fetch_file_cached(MODEL_PATHS[_],_)

    

if __name__ == "__main__":
    download_files()