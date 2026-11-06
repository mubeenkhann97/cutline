"""Log in to Hugging Face once, for the gated FLUX.1-schnell download.

Use this instead of `hf auth login`: the CLI does not load the certificate fix in common.py and
crashes with a RecursionError behind TLS inspection. The token is typed at a hidden prompt and
saved in your user profile (not in the project folder).
"""
import os

os.environ["HF_HUB_OFFLINE"] = "0"  # common.py makes the pipeline offline
import common  # noqa: E402,F401  (fixes certificates, sets where the token is saved)
from huggingface_hub import login  # noqa: E402

login(add_to_git_credential=False)
