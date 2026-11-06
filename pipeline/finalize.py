"""Stage 5: final refine and quality check, ready to upload.

- Loudness normalised to -14 LUFS (YouTube's playback target), peaks under -1.5 dB
- H.264 High / AAC, 1080x1920, 24 fps, fast-start for upload
- Checks: resolution, length, black frames, frozen frames (common AI-clip faults)
- Writes Exports/<slug>.mp4, a review sheet <slug>_qc.png, <slug>_upload.txt and the prompt pack
  <slug>_prompts/ (see prompt_pack.py)
"""
import argparse
import re
import subprocess

from common import FFMPEG, FFPROBE, FINAL_FPS, FINAL_H, FINAL_W, ROOT, load_spec, probe_duration, project_dir, run
from prompt_pack import write_pack


def detect(path, vf: str, pattern: str) -> list:
    out = subprocess.run([str(FFMPEG), "-hide_banner", "-i", str(path), "-vf", vf, "-an", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return re.findall(pattern, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    args = ap.parse_args()

    pdir = project_dir(args.slug)
    spec = load_spec(args.slug, refined=True)
    draft = pdir / "out" / "draft.mov"
    exports = ROOT / "Exports"
    exports.mkdir(exist_ok=True)
    final = exports / f"{args.slug}.mp4"

    run([FFMPEG, "-loglevel", "error", "-y", "-i", draft,
         "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "48000",
         "-c:v", "libx264", "-profile:v", "high", "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p",
         "-r", str(FINAL_FPS), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", final])

    issues = []
    dur = probe_duration(final)
    size = subprocess.run([str(FFPROBE), "-v", "error", "-select_streams", "v:0",
                           "-show_entries", "stream=width,height", "-of", "csv=p=0", str(final)],
                          capture_output=True, text=True).stdout.split()[0].strip(",")
    if size != f"{FINAL_W},{FINAL_H}":
        issues.append(f"resolution {size}, expected {FINAL_W}x{FINAL_H}")
    if dur > 180:
        issues.append(f"{dur:.1f}s is longer than the 3-minute Shorts limit")
    for start, end in detect(final, "blackdetect=d=0.3:pix_th=0.08", r"black_start:([\d.]+) black_end:([\d.]+)"):
        issues.append(f"black frames {start}-{end}s")
    for start in detect(final, "freezedetect=n=0.002:d=1.2", r"freeze_start: ([\d.]+)"):
        issues.append(f"frozen picture from {start}s (AI clip may have stalled)")

    run([FFMPEG, "-loglevel", "error", "-y", "-i", final, "-vf",
         f"fps={8 / dur:.4f},scale=270:-1,tile=8x1", "-frames:v", "1", exports / f"{args.slug}_qc.png"])

    upload = [spec["title"], "", spec.get("description", ""), "", " ".join(spec.get("hashtags", []))]
    if spec.get("sources"):
        upload += ["", "Sources:"] + spec["sources"]
    (exports / f"{args.slug}_upload.txt").write_text("\n".join(upload), encoding="utf-8")
    pack = write_pack(spec, pdir)

    print(f"Final: {final}  ({dur:.1f}s, {size})")
    print(f"Prompt pack: {pack}")
    print("QC: " + ("passed" if not issues else "; ".join(issues)))


if __name__ == "__main__":
    main()
