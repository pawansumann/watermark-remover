import os
from huggingface_hub import snapshot_download

# Token ko direct script ke andar set kar rahe hain taaki koi confusion na rahe
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("Downloading Stable Diffusion Inpainting model... Please wait.")

snapshot_download(
    repo_id="stabilityai/stable-diffusion-2-inpainting",
    local_dir_use_symlinks=False
)

print("Download complete successfully!")