"""Shared paths and helpers for the Shorts pipeline."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"
FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
os.environ["PATH"] = f"{FFMPEG.parent}{os.pathsep}{os.environ['PATH']}"  # Whisper calls ffmpeg by name
STYLES = json.loads((ROOT / "pipeline" / "styles.json").read_text(encoding="utf-8"))

# pip_system_certs injects pip's vendored truststore at start-up; huggingface_hub's
# HTTP client brings its own copy, and the two recurse forever. Keep one copy only.
try:
    from pip._vendor import truststore as _pip_truststore
    import truststore

    _pip_truststore.extract_from_ssl()
    truststore.inject_into_ssl()
except ImportError:
    pass

# Keep model downloads inside the project folder rather than the user profile
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
# ...but read the login token from where `hf auth login` saves it, so it never sits in the project folder
os.environ.setdefault("HF_TOKEN_PATH", str(Path.home() / ".cache" / "huggingface" / "token"))
WHISPER_DIR = ROOT / "models" / "whisper"
# Work offline from the local copies: no update checks or telemetry. download_models.py turns this off.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Output sizes: generation happens small, the final render is full HD vertical
# 24 fps is Wan's own rate: at 30, a third of the frames were repeats (judder), and motion
# interpolation ghosted wherever Wan's motion jumps
FINAL_W, FINAL_H, FINAL_FPS = 1080, 1920, 24


def project_dir(slug: str) -> Path:
    d = PROJECTS / slug
    if not d.exists():
        raise SystemExit(f"No project folder: {d}")
    return d


def load_spec(slug: str, refined: bool = False) -> dict:
    name = "spec.refined.json" if refined else "spec.json"
    path = project_dir(slug) / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the previous stage first.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run(cmd: list, **kw) -> None:
    print(">", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())
