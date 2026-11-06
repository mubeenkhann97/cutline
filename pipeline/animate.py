"""Stage 3: animate each picked keyframe with Wan 2.2 TI2V-5B (Apache 2.0).

The 5B model does not fit in 8 GB of VRAM as shipped (see load_pipe). Measured on
an 8 GB RTX 2000 Ada: 97 frames at 480x832 or 41 frames at 704x1280 take about 8 min
(30 steps); longer clips overflow into system RAM and take about an hour (see max_frames).
704x1280 is the default: details shimmer far less than at 480x832. A line longer than one
clip gets chained shots: each starts on the last frame of the one before, and assembly cuts
between them. --steps 4 gives a 3-minute preview.

    python animate.py <slug>                 # all scenes that have no clip yet
    python animate.py <slug> --only s2 --seed 7
"""
import argparse
import math
import random
import time

from common import FFMPEG, load_spec, project_dir, run, save_json

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_NEGATIVE = ("static, frozen, still image, text, subtitles, watermark, blurry, low quality, jpeg artefacts, "
                "deformed, disfigured, extra limbs, fused fingers, flicker, morphing face, "
                "subject leaving the frame, cropped subject, fast camera movement")


def load_pipe():
    """Fit Wan 2.2 into 8 GB without diffusers' group-offload hooks, which on Windows left whole
    models on the GPU and let the driver spill into system RAM (70 min per 3 s clip)."""
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, dtype=torch.bfloat16)
    # fp8 storage halves the 10 GB transformer so it fits on the GPU; maths still runs in bf16
    pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    # The 11 GB text encoder never fits: it runs on the CPU, once per clip (see encode_prompts)
    text_encoder = pipe.text_encoder
    pipe.text_encoder = None
    pipe.enable_model_cpu_offload()  # transformer and VAE take turns on the GPU
    pipe.vae.enable_tiling()

    # The offload sequence only evicts the VAE (2.8 GB, used first to encode the image) after the
    # transformer has finished, so both sit on the GPU while denoising and the driver spills to RAM
    # (218 s per step). Evict it ourselves; its hook brings it back for the final decode.
    def evict_vae(module, args):
        if next(pipe.vae.parameters()).device.type == "cuda":
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()

    pipe.transformer.register_forward_pre_hook(evict_vae)
    return pipe, text_encoder


def encode_prompts(pipe, text_encoder, prompt: str):
    import torch

    pipe.text_encoder = text_encoder
    try:
        pos, neg = pipe.encode_prompt(prompt=prompt, negative_prompt=WAN_NEGATIVE, device=torch.device("cpu"))
    finally:
        pipe.text_encoder = None
    return pos.to("cuda"), neg.to("cuda")


TOKEN_BUDGET = 39000  # video tokens that fit in 8 GB: 97 frames at 480x832; 121 spilled into RAM


def max_frames(w: int, h: int) -> int:
    """Longest clip that fits on the GPU. The VAE packs 16x16 pixels and 4 frames into one token,
    so 704x1280 (3520 tokens a frame) gets 41 frames where 480x832 (1560) gets 97."""
    return 4 * (TOKEN_BUDGET // ((w // 16) * (h // 16)) - 1) + 1


def plan_shots(seconds: float, limit: int) -> tuple[int, int]:
    """(shots, frames per shot) for a scene. Assembly may stretch clips by up to 1.35x: split only
    when one clip would need more than 1.3x, and aim each shot a little under its share (stretching
    1.15x is invisible). Frames follow Wan's 4k+1 rule; chained shots may be as short as 1 s."""
    shots = math.ceil(seconds * 24 / (1.3 * limit))
    per = math.ceil(seconds * 24 / 1.15 / shots)
    return shots, max(min(33 if shots == 1 else 25, limit), min(limit, 4 * math.ceil((per - 1) / 4) + 1))


def last_frame(frames):
    """The final frame as an image, to start the next chained shot from."""
    from PIL import Image

    f = frames[-1]
    return f if isinstance(f, Image.Image) else Image.fromarray((f * 255).round().clip(0, 255).astype("uint8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    ap.add_argument("--only", nargs="*", help="scene ids to (re)animate")
    ap.add_argument("--redo", action="store_true", help="overwrite existing clips")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--frames", type=int,
                    help="4k+1, one shot per scene (more than max_frames spills out of 8 GB). "
                         "Default: fitted to each scene, chaining shots for long lines")
    ap.add_argument("--size", default="704x1280", help="WxH, multiples of 32 (480x832 renders 2.4x faster)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--restart", action="store_true",
                    help="chained shots start again from the keyframe (new seed), for a clip whose last frame "
                         "degrades: a chained shot inherits and worsens the noise")
    ap.add_argument("--still", action="store_true",
                    help="no AI motion: slow camera move over the keyframe (seconds, not minutes; for drafts)")
    args = ap.parse_args()

    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    pdir = project_dir(args.slug)
    spec = load_spec(args.slug, refined=True)
    (pdir / "clips").mkdir(exist_ok=True)
    w, h = map(int, args.size.lower().split("x"))

    todo = [s for s in spec["scenes"]
            if (not args.only or s["id"] in args.only)
            and (args.redo or args.only or not (pdir / "clips" / f"{s['id']}.mp4").exists())]
    if not todo:
        print("All clips exist; use --redo or --only to re-render")
        return

    if spec.get("voice", {}).get("enabled"):
        # Scene lengths come from the spoken lines, so fit them now (seconds of TTS) rather than
        # rendering every clip at the maximum length
        import voice
        voice.synthesise(spec, pdir)
        save_json(pdir / "spec.refined.json", spec)

    if args.still:
        for i, s in enumerate(todo):
            frames = int(s["seconds"] * 24)
            zoom = "1+0.0015*on" if i % 2 == 0 else "1.18-0.0015*on"  # alternate push in / pull out
            run([FFMPEG, "-loglevel", "error", "-y", "-i", pdir / s["keyframe"], "-vf",
                 f"scale=1536:-1,zoompan=z='{zoom}':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d={frames}:s=768x1344:fps=24",
                 "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", pdir / "clips" / f"{s['id']}.mp4"])
        print(f"Made {len(todo)} still-motion clips")
        return

    pipe, text_encoder = load_pipe()
    for s in todo:
        if not s.get("keyframe"):
            raise SystemExit(f"{s['id']} has no keyframe; run the keyframes stage")
        seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
        image = Image.open(pdir / s["keyframe"]).convert("RGB").resize((w, h), Image.LANCZOS)
        shots, n = (1, args.frames) if args.frames else plan_shots(s["seconds"], max_frames(w, h))
        for old in (pdir / "clips").glob(f"{s['id']}-*.mp4"):  # chained shots from an earlier render
            old.unlink()
        pos, neg = encode_prompts(pipe, text_encoder, s["video_prompt"])
        for j in range(shots):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            frames = pipe(
                image=image, prompt_embeds=pos, negative_prompt_embeds=neg,
                width=w, height=h, num_frames=n,
                num_inference_steps=args.steps, guidance_scale=5.0,
                generator=torch.Generator("cpu").manual_seed(seed + j),
            ).frames[0]
            name = s["id"] if j == 0 else f"{s['id']}-{j + 1}"  # s6.mp4, s6-2.mp4, ...
            export_to_video(frames, str(pdir / "clips" / f"{name}.mp4"), fps=24)
            if not args.restart:
                image = last_frame(frames)
            print(f"{name}: {len(frames)} frames in {(time.time() - t0) / 60:.1f} min, "
                  f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB (seed {seed + j})")
        s["clip_seed"] = seed
        save_json(pdir / "spec.refined.json", spec)


if __name__ == "__main__":
    main()
