"""Stage 1: turn a hand-written spec into model-ready prompts.

Combines each scene with the series style bible, keeps image prompts inside the
77-token CLIP window SDXL actually reads, strips words that make models draw
garbled text, and writes spec.refined.json plus a short report.

Optional: --llm ollama:<model> asks a local Ollama model to make the scene
descriptions more visual before the rules are applied.
"""
import argparse
import json
import re
import urllib.request

from common import STYLES, load_spec, project_dir, save_json

# Words that push image/video models into drawing (badly spelled) text.
# On-screen text is added later by the assembly stage, never by the model.
TEXT_TRIGGERS = re.compile(r"\b(text|caption|label|sign|title|words?|letters?|says|written|logo)\b", re.I)
CAMERA_TERMS = {
    "slow push in": "camera slowly pushes in",
    "slow pull out": "camera slowly pulls back",
    "slow orbit": "camera slowly orbits around the subject",
    "static": "locked-off static camera",
}
MAX_CLIP_TOKENS = 75  # 77 minus start/end tokens


def clip_token_count(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def fit_to_tokens(core: list, extras: list, tokenizer, warnings: list, scene_id: str) -> str:
    """Core parts always stay (the scene is trimmed word by word if needed); extras fill any room left."""
    core = list(core)
    while clip_token_count(", ".join(core), tokenizer) > MAX_CLIP_TOKENS:
        words = core[-1].split()
        if len(words) <= 6:
            break
        core[-1] = " ".join(words[:-1])
    joined = ", ".join(core)
    for extra in extras:
        if clip_token_count(f"{joined}, {extra}", tokenizer) <= MAX_CLIP_TOKENS:
            joined = f"{joined}, {extra}"
    if clip_token_count(joined, tokenizer) > MAX_CLIP_TOKENS:
        warnings.append(f"{scene_id}: image prompt still over 77 tokens; shorten the character or scene")
    return joined


def ollama_rewrite(model: str, scene: dict, character: str) -> dict:
    prompt = (
        "Rewrite these two descriptions for an AI image and video generator. "
        "Make them concrete and visual (subject, action, setting, expression). "
        "Do not mention text, labels or words. Keep each under 40 words. "
        'Reply with JSON only: {"image": "...", "motion": "..."}\n\n'
        f"Character: {character}\nImage: {scene['image']}\nMotion: {scene['motion']}"
    )
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json"}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(json.loads(r.read())["response"])
    return {"image": out.get("image", scene["image"]), "motion": out.get("motion", scene["motion"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    ap.add_argument("--llm", help="ollama:<model>, e.g. ollama:qwen2.5:7b")
    args = ap.parse_args()

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")

    spec = load_spec(args.slug)
    style = STYLES[spec["series"]]
    characters = spec.get("characters", {})  # story format: {"lungs": {"look": ..., "voice": ...}}
    warnings = []

    for scene in spec["scenes"]:
        sid = scene["id"]
        # Who is in the shot: the scene's cast (story format) or the video's single mascot
        cast = scene.get("cast") or ([scene["speaker"]] if scene.get("speaker") else [])
        missing = [c for c in cast if c not in characters]
        if missing:
            raise SystemExit(f"{sid}: cast {missing} not defined under 'characters'")
        character = " and ".join(characters[c]["look"] for c in cast) if cast else spec.get("character", "")
        if args.llm and args.llm.startswith("ollama:"):
            scene.update(ollama_rewrite(args.llm.split(":", 1)[1], scene, character))

        for field in ("image", "motion"):
            if TEXT_TRIGGERS.search(scene[field]):
                warnings.append(f"{sid}: '{field}' mentions text-like words; models will try to draw text")
                scene[field] = TEXT_TRIGGERS.sub("", scene[field])
            scene[field] = re.sub(r"\s{2,}", " ", scene[field]).strip(" ,")

        # CLIP only reads 77 tokens: short style tag, character and scene must survive;
        # the long style description and setting are added only if they still fit
        core = [p for p in (style["look_short"], character, scene["image"]) if p]
        # A spec can move the story out of the series' usual place (e.g. a red heart needs a non-red room)
        setting = spec.get("setting", style["setting"])
        scene["image_prompt"] = fit_to_tokens(core, [setting, "vertical composition", style["look"]],
                                              tokenizer, warnings, sid)
        # FLUX's T5 encoder reads the whole description, so it gets everything
        scene["image_prompt_long"] = ", ".join(core + [setting, "vertical composition", style["look"]])
        scene["negative_prompt"] = style["negative"]

        camera = CAMERA_TERMS.get(scene.get("camera", "").split(",")[0].strip(), scene.get("camera", ""))
        if camera and "static" not in camera:
            camera += ", gentle and subtle"
        # The video model (UMT5 encoder) reads long prompts, so it gets full sentences.
        # Without the framing line Wan lets the camera wander until the character leaves the shot.
        # A speaking character gets an explicit talking cue: Wan then animates the mouth (not lip-synced,
        # but it reads as speech at Shorts speed)
        speaker = scene.get("speaker")
        talking = (f"The {characters[speaker].get('name', speaker)} is talking, mouth moving as it speaks "
                   f"with expressive eyes and eyebrows. ") if speaker and scene.get("narration") else ""
        framing = "The characters stay fully in frame" if len(cast) > 1 else "The character stays centred and fully in frame"
        scene["video_prompt"] = (
            f"{character}. {scene['motion']}. {talking}{scene['image']}. "
            f"{camera}. {framing} the whole time. "
            f"{style['motion_style']}. {style['look']}."
        )
        # Story videos cut fast (a new shot every 2-4 s); explainer scenes run longer
        if not 1.5 <= scene.get("seconds", 5) <= 8:
            warnings.append(f"{sid}: {scene.get('seconds')}s is outside the 1.5-8 s range that reads well in Shorts")

    total = sum(s.get("seconds", 5) for s in spec["scenes"])
    if total > 60:
        warnings.append(f"Total {total}s is over 60 s; YouTube may not treat it as a Short")
    if spec["series"] == "body-explained" and not spec.get("sources"):
        warnings.append("Health video has no sources listed; fact-check before publishing")

    # Keep keyframe picks and clip seeds from an earlier run of the later stages
    refined_path = project_dir(args.slug) / "spec.refined.json"
    if refined_path.exists():
        previous = {s["id"]: s for s in json.loads(refined_path.read_text(encoding="utf-8"))["scenes"]}
        for scene in spec["scenes"]:
            for key in ("keyframe", "clip_seed"):
                if key in previous.get(scene["id"], {}):
                    scene[key] = previous[scene["id"]][key]

    spec["refine_report"] = warnings
    save_json(refined_path, spec)
    print(f"Refined {len(spec['scenes'])} scenes, total {total}s")
    for w in warnings:
        print("  WARN", w)


if __name__ == "__main__":
    main()
