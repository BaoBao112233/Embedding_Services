# download_model.py
import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_model(model_id=None, hf_token=None, dest_dir=None):
    """Download a model from Hugging Face to dest_dir.

    Args:
        model_id (str): repo id on HF (e.g. 'google/embeddinggemma-300m').
        hf_token (str): huggingface token string.
        dest_dir (str): where to store the model files.
    """
    model_id = model_id or os.environ.get("MODEL_ID", "google/embeddinggemma-300m")
    hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
    dest_dir = dest_dir or os.environ.get("MODEL_DIR", "/models/embeddinggemma-300m")

    if hf_token is None:
        raise SystemExit("ERROR: Set environment variable HUGGINGFACE_TOKEN with your HF token (and accept model grant on HF).")

    print(f"Downloading {model_id} into {dest_dir} ...")
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # snapshot_download will download model files to cache; using cache_dir directs
    # the hub to place files under dest_dir for easier inspection in containers.
    try:
        snapshot_download(repo_id=model_id, cache_dir=dest_dir, use_auth_token=hf_token, repo_type="model")
    except Exception as e:
        msg = str(e)
        # Handle gated / permission errors with a helpful message
        if "gated" in msg or "awaiting a review" in msg or "Access to model" in msg or "401" in msg or "403" in msg:
            print("ERROR: Cannot access the requested model. The repository appears to be gated or requires approval.")
            print("Details:", msg)
            print("Fix: ensure your Hugging Face account has access to the model (visit the model page and accept any grant),")
            print("and use a token with read scope (export HUGGINGFACE_TOKEN).")
            raise SystemExit(2)
        # otherwise re-raise to surface unexpected errors
        raise

    print("Download complete. Files saved under", dest_dir)


def main():
    download_model()


if __name__ == "__main__":
    main()
