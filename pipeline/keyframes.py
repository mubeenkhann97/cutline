"""Stage 2: draw a keyframe per scene (fast preview before animating).

FLUX.1-schnell is used when it has been downloaded (download_models.py flux), otherwise
SDXL; choose with --model. FLUX follows long descriptions far better and keeps a richer,
more cinematic look; SDXL is faster and needs no Hugging Face login.

Each scene gets --n candidates and a contact sheet (keyframes/_sheet.png).
Pick the best with --pick s1=2 s3=0 (no generation), or re-roll one scene with
--only s2. Animation in the next stage starts from the picked keyframe, which
keeps the character consistent and stops slow video renders being wasted.

If one scene's character drifts from the others, redraw it from a good scene:
--only s1 --ref s4 starts from s4's picked keyframe (image-to-image), so the
character's shape and colours carry over. --strength sets how much changes
(0.5 keeps most of the reference, 0.85 keeps little more than the palette).
"""
import argparse
import random
import time

from common import ROOT, load_spec, project_dir, save_json

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"
FLUX = "black-forest-labs/FLUX.1-schnell"
# The 12B FLUX transformer is quantised to 4-bit (NF4) once and saved here: about 7 GB, so it stays on the GPU
FLUX_NF4 = ROOT / "models" / "flux-schnell-nf4"
SIZES = {"sdxl": (768, 1344),  # SDXL's native vertical bucket
         "flux": (720, 1280)}  # multiples of 16; larger runs out of the 8 GB card
W, H = SIZES["sdxl"]  # contact sheet tiles


def flux_available() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(FLUX, "model_index.json"), str)


def load_flux(img2img: bool = False):
    """FLUX.1-schnell on 8 GB: NF4 transformer and VAE on the GPU, both text encoders on the CPU.

    The T5 encoder alone is 9.5 GB, so prompts are encoded on the CPU first (encode_flux) and the
    encoders are then dropped; model CPU offload would move T5 onto the card and spill into shared memory.
    """
    import torch
    from diffusers import BitsAndBytesConfig, FluxImg2ImgPipeline, FluxPipeline, FluxTransformer2DModel

    if FLUX_NF4.exists():
        transformer = FluxTransformer2DModel.from_pretrained(FLUX_NF4, dtype=torch.bfloat16)
    else:
        print("Quantising the FLUX transformer to 4-bit (first run only, a few minutes)", flush=True)
        nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        # local_files_only: diffusers asks the Hub about sharded checkpoints even in offline mode
        transformer = FluxTransformer2DModel.from_pretrained(FLUX, subfolder="transformer", quantization_config=nf4,
                                                             dtype=torch.bfloat16, local_files_only=True)
        transformer.save_pretrained(FLUX_NF4)
    cls = FluxImg2ImgPipeline if img2img else FluxPipeline
    pipe = cls.from_pretrained(FLUX, transformer=transformer, dtype=torch.bfloat16, local_files_only=True)
    pipe.vae.to("cuda")  # the pipeline runs on the device of its first module, the VAE
    pipe.vae.enable_tiling()
    return pipe


def encode_flux(pipe, scenes: list) -> dict:
    """Encode every scene's prompt on the CPU, then free the text encoders: {scene id: embeddings}."""
    import torch

    embeds = {}
    with torch.no_grad():
        for s in scenes:
            # CLIP reads the 77-token prompt, T5 the full description
            pos, pooled, _ = pipe.encode_prompt(prompt=s["image_prompt"],
                                                prompt_2=s.get("image_prompt_long", s["image_prompt"]),
                                                device="cpu", max_sequence_length=256)
            embeds[s["id"]] = {"prompt_embeds": pos.to("cuda"), "pooled_prompt_embeds": pooled.to("cuda")}
    pipe.text_encoder = pipe.text_encoder_2 = None
    return embeds


def load_pipe(quality: str, img2img: bool = False):
    import torch
    from diffusers import (AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLImg2ImgPipeline,
                           StableDiffusionXLPipeline)
    from huggingface_hub import hf_hub_download

    vae = AutoencoderKL.from_pretrained(VAE_FIX, dtype=torch.float16)
    # Load the image-to-image class directly: converting with from_pipe() left weights on the GPU
    # alongside the offload hooks and stalled in shared memory
    cls = StableDiffusionXLImg2ImgPipeline if img2img else StableDiffusionXLPipeline
    pipe = cls.from_pretrained(BASE, vae=vae, dtype=torch.float16, variant="fp16")
    if quality == "fast":
        pipe.load_lora_weights(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_lora.safetensors"))
        pipe.fuse_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.enable_model_cpu_offload()  # fits the 8 GB card
    pipe.vae.enable_tiling()
    return pipe


def contact_sheet(paths_by_scene: dict, out):
    from PIL import Image, ImageDraw

    rows = list(paths_by_scene.items())
    cols = max(len(p) for _, p in rows)
    tw, th = W // 4, H // 4
    sheet = Image.new("RGB", (cols * tw, len(rows) * th), "black")
    draw = ImageDraw.Draw(sheet)
    for r, (sid, paths) in enumerate(rows):
        for c, p in enumerate(paths):
            sheet.paste(Image.open(p).resize((tw, th)), (c * tw, r * th))
            draw.text((c * tw + 6, r * th + 6), f"{sid}={c}", fill="yellow")
    sheet.save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    ap.add_argument("--n", type=int, default=2, help="candidates per scene")
    ap.add_argument("--only", nargs="*", help="scene ids to (re)generate")
    ap.add_argument("--pick", nargs="*", help="choose candidates, e.g. s1=1 s2=0")
    ap.add_argument("--model", choices=["flux", "sdxl"], help="default: flux if downloaded, else sdxl")
    ap.add_argument("--quality", choices=["fast", "full"], default="fast",
                    help="SDXL only: fast = SDXL-Lightning 8 steps; full = 30 steps with negative prompt")
    ap.add_argument("--ref", help="scene id whose picked keyframe to start from (s4), or an image path "
                                  "relative to the project (characters/lungs.png)")
    ap.add_argument("--strength", type=float, default=0.7, help="with --ref: how much of the reference to redraw")
    args = ap.parse_args()

    pdir = project_dir(args.slug)
    spec = load_spec(args.slug, refined=True)
    kdir = pdir / "keyframes"
    kdir.mkdir(exist_ok=True)

    if args.pick:
        picks = dict(p.split("=") for p in args.pick)
        for s in spec["scenes"]:
            if s["id"] in picks:
                s["keyframe"] = f"keyframes/{s['id']}_{picks[s['id']]}.png"
        save_json(pdir / "spec.refined.json", spec)
        print("Picked:", {s["id"]: s.get("keyframe") for s in spec["scenes"]})
        return

    import torch
    from PIL import Image

    model = args.model or ("flux" if flux_available() else "sdxl")
    w, h = SIZES[model]
    scenes = [s for s in spec["scenes"] if not args.only or s["id"] in args.only]
    ref = None
    if args.ref:
        # A scene id (use its picked keyframe) or an image file, e.g. a rough character drawing
        ref_scene = next((s for s in spec["scenes"] if s["id"] == args.ref), None)
        if ref_scene:
            if not ref_scene.get("keyframe"):
                raise SystemExit(f"--ref {args.ref}: that scene has no picked keyframe")
            ref_path = pdir / ref_scene["keyframe"]
        else:
            ref_path = pdir / args.ref
            if not ref_path.exists():
                raise SystemExit(f"--ref {args.ref}: not a scene id or a file in {pdir}")
        ref = Image.open(ref_path).convert("RGB").resize((w, h))
    if model == "flux":
        pipe = load_flux(img2img=ref is not None)
        embeds = encode_flux(pipe, scenes)
        # schnell is distilled for 4 steps without guidance (no negative prompt); image-to-image
        # only runs the last `strength` share of the steps, so it gets more
        steps, cfg = (4 if ref is None else 6), 0.0
    else:
        pipe = load_pipe(args.quality, img2img=ref is not None)
        steps, cfg = (8, 0.0) if args.quality == "fast" else (30, 6.0)
    print(f"Drawing with {model} at {w}x{h}", flush=True)

    for s in scenes:
        if model == "flux":
            prompt = {**embeds[s["id"]], "max_sequence_length": 256}
        else:
            prompt = {"prompt": s["image_prompt"], "negative_prompt": s["negative_prompt"] if cfg > 1 else None}
        for i in range(args.n):
            seed = random.randint(0, 2**31 - 1)
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            extra = {"image": ref, "strength": args.strength} if ref is not None else {}
            if ref is None or model == "flux":  # SDXL image-to-image takes its size from the image
                extra.update(width=w, height=h)
            img = pipe(
                **prompt,
                num_inference_steps=steps, guidance_scale=cfg,
                generator=torch.Generator("cpu").manual_seed(seed), **extra,
            ).images[0]
            img.save(kdir / f"{s['id']}_{i}.png")
            print(f"{s['id']} candidate {i} seed {seed}, {time.time() - t0:.0f} s, "
                  f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        s.setdefault("keyframe", f"keyframes/{s['id']}_0.png")

    save_json(pdir / "spec.refined.json", spec)
    by_scene = {s["id"]: sorted(kdir.glob(f"{s['id']}_*.png")) for s in spec["scenes"]}
    contact_sheet({k: v for k, v in by_scene.items() if v}, kdir / "_sheet.png")
    print(f"Review {kdir / '_sheet.png'}; change choices with --pick")


if __name__ == "__main__":
    main()
