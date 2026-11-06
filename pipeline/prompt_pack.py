"""Prompt pack: everything needed to make a video's shots in another generator.

Exports/<slug>_prompts/ holds prompts.md (to read and paste from), prompts.json (the same, for
scripts), the picked keyframe of every scene (for image-to-video tools) and every spoken line
as its own WAV (for tools that lip-sync to audio). finalize.py writes it for every video.

    python prompt_pack.py <slug>
"""
import argparse
import json
import shutil

from common import ROOT, load_spec, project_dir


def write_pack(spec: dict, pdir) -> "Path":
    import soundfile as sf

    out = ROOT / "Exports" / f"{spec['slug']}_prompts"
    if out.exists():
        shutil.rmtree(out)  # only ever holds files this function wrote
    (out / "keyframes").mkdir(parents=True)
    voice = pdir / "audio" / "voice.wav"
    audio, sr = sf.read(str(voice)) if voice.exists() else (None, 0)
    if audio is not None:
        (out / "audio").mkdir()

    characters = spec.get("characters", {})
    scenes, cursor = [], 0.0
    for s in spec["scenes"]:
        scene = {
            "id": s["id"],
            "speaker": characters.get(s.get("speaker"), {}).get("name", s.get("speaker", "")),
            "seconds": s["seconds"],
            "narration": s.get("narration", ""),
            "image_prompt": s.get("image_prompt_long") or s.get("image_prompt", ""),
            "video_prompt": s.get("video_prompt", ""),
            "camera": s.get("camera", ""),
        }
        if s.get("keyframe") and (pdir / s["keyframe"]).exists():
            scene["keyframe"] = f"keyframes/{s['id']}.png"
            shutil.copy(pdir / s["keyframe"], out / scene["keyframe"])
        if audio is not None and s.get("narration"):
            # voice.py lays each line out from the scene's start, so the scene's slice is that line
            scene["audio"] = f"audio/{s['id']}.wav"
            sf.write(str(out / scene["audio"]), audio[int(cursor * sr): int((cursor + s["seconds"]) * sr)], sr)
        scenes.append(scene)
        cursor += s["seconds"]

    pack = {"title": spec["title"], "aspect_ratio": "9:16", "setting": spec.get("setting", ""),
            "characters": {k: {"name": c.get("name", k), "look": c["look"]} for k, c in characters.items()},
            "scenes": scenes}
    (out / "prompts.json").write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# {spec['title']}: prompt pack", "",
          "Make one vertical (9:16) clip per scene, then bring the clips back for voice, captions and music "
          "(put them in projects/<slug>/clips/ as s1.mp4, s2.mp4, ... and run the assemble stage).",
          "- Image-to-video tools: upload the scene's keyframe and paste its video prompt.",
          "- Text-to-video tools: paste the image prompt and the video prompt together.",
          "- Lip-sync tools: add the scene's audio file.",
          "- Keep each character's look text word for word in every scene so they stay consistent.", ""]
    if characters:
        md += ["## Characters", ""] + [f"- **{c['name']}**: {c['look']}" for c in pack["characters"].values()] + [""]
    for sc in scenes:
        md += [f"## {sc['id']}: {sc['speaker'] or 'no speaker'} ({sc['seconds']:.1f} s)", ""]
        if sc["narration"]:
            md += [f"Line: \"{sc['narration']}\"", ""]
        md += ["Image prompt:", "```", sc["image_prompt"], "```", "Video prompt:", "```", sc["video_prompt"], "```"]
        files = [sc[k] for k in ("keyframe", "audio") if k in sc]
        md += ([f"Files: {', '.join(files)}"] if files else []) + [""]
    (out / "prompts.md").write_text("\n".join(md), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    args = ap.parse_args()
    print("Prompt pack:", write_pack(load_spec(args.slug, refined=True), project_dir(args.slug)))


if __name__ == "__main__":
    main()
