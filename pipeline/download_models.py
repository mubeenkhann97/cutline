"""Fetch the open-weight models the pipeline uses into models/ (about 75 GB with FLUX).

The only stage that needs the internet; every other stage runs offline.
Name groups to fetch only those: python pipeline\download_models.py flux

FLUX.1-schnell (Apache 2.0) is gated on Hugging Face: accept its terms on the model page while
signed in, then run `hf auth login` once with a read-only token. The other models need no account.
"""
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "0"  # common.py makes the other stages offline
import common  # noqa: E402  (sets HF_HOME and fixes certificates)
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

groups = sys.argv[1:] or ["voice", "sdxl", "wan", "flux"]

if "voice" in groups:
    import whisper

    snapshot_download("hexgrad/Kokoro-82M")
    whisper.load_model("base", download_root=str(common.WHISPER_DIR))
    print("Voice and captions ready", flush=True)

if "sdxl" in groups:
    snapshot_download("stabilityai/stable-diffusion-xl-base-1.0",
                      allow_patterns=["model_index.json", "*/*.json", "*/*.txt", "*fp16.safetensors"],
                      ignore_patterns=["vae_1_0/*"])
    snapshot_download("madebyollin/sdxl-vae-fp16-fix", allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
    hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_lora.safetensors")
    print("SDXL ready", flush=True)

if "wan" in groups:
    snapshot_download("Wan-AI/Wan2.2-TI2V-5B-Diffusers", ignore_patterns=["assets/*", "examples/*"])
    print("Wan 2.2 ready", flush=True)

if "flux" in groups:
    from huggingface_hub.errors import GatedRepoError

    try:
        # Diffusers folders only (about 34 GB); the root single-file checkpoints duplicate them
        snapshot_download("black-forest-labs/FLUX.1-schnell",
                          ignore_patterns=["flux1-schnell.safetensors", "ae.safetensors", "*.jpg", "*.jpeg", "*.png"])
        print("FLUX.1-schnell ready", flush=True)
    except GatedRepoError:
        print("FLUX.1-schnell skipped: accept its terms at huggingface.co/black-forest-labs/FLUX.1-schnell "
              "and run `hf auth login`; keyframes use SDXL until then", flush=True)
