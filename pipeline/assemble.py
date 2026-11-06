"""Stage 4: voice, captions, music, then the FFmpeg render of the full Short.

Each AI clip is conformed to its scene length at 1080x1920/24 fps with a
snap-in zoom on the cut; FFmpeg then joins the scenes, burns in the header,
labels and captions (ASS subtitles via libass) and mixes voice and music.
Everything here is FFmpeg and Python: no Remotion licence or browser needed.
"""
import argparse
import shutil

from common import (FFMPEG, FINAL_FPS, FINAL_H, FINAL_W, ROOT, STYLES, load_spec, probe_duration,
                    project_dir, run, save_json)
from overlay import write_ass

FONTS_DIR = r"C:\Windows\Fonts"  # Arial Black and Segoe UI ship with Windows


SNAP_FRAMES = 8  # cut-in zoom settles over this many frames


def conform_clip(src, dst, seconds: float, punch: bool = False) -> int:
    """Scale to full HD vertical and stretch (or loop back and forth) to the scene length.

    Adds the snap-in on the cut (zoom 1.12 -> 1.04, ease-out) and a slow drift to 1.08 so AI clips
    never feel static. zoompan works on a 2x frame so the zoom moves smoothly, not in whole pixels.
    punch frames 15% tighter and a little high (where the faces are), so a chained shot that
    continues the one before reads as a camera cut, not a jump. Returns the frame count.
    """
    src_len = probe_duration(src)
    frames = round(seconds * FINAL_FPS)
    ratio = seconds / src_len
    if ratio <= 1.35:
        timing = f"setpts=PTS*{ratio:.4f}"
    else:  # too long to slow down gracefully: play forwards then backwards
        timing = f"split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0,setpts=PTS*{seconds / (2 * src_len):.4f}"
    k = SNAP_FRAMES
    base = 0.15 if punch else 0
    zoom = (f"{base}+if(lt(on,{k}),1.12-0.08*(1-pow(1-on/{k},3)),1.04+0.04*(on-{k})/{max(frames - k, 1)})")
    y = "(ih-ih/zoom)*0.35" if punch else "ih/2-ih/zoom/2"
    # hqdn3d calms the frame-to-frame shimmer of AI texture (about a third less in still areas).
    # Clips stay at their own 24 fps, so only the stretch repeats frames (about 1 in 10 at 1.1x).
    # minterpolate was tried instead: it ghosted (double images) where Wan's motion jumps.
    smooth = f"hqdn3d=1:1:6:6,fps={FINAL_FPS}"
    vf = (f"{timing},{smooth},"
          f"scale={2 * FINAL_W}:{2 * FINAL_H}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={2 * FINAL_W}:{2 * FINAL_H},"
          f"zoompan=z='{zoom}':d=1:x='iw/2-iw/zoom/2':y='{y}':s={FINAL_W}x{FINAL_H}:fps={FINAL_FPS},"
          f"unsharp=5:5:0.25,format=yuv420p")  # light: stronger sharpening brought the shimmer back
    run([FFMPEG, "-loglevel", "error", "-y", "-i", src, "-filter_complex", vf, "-frames:v", str(frames),
         "-an", "-c:v", "libx264", "-crf", "14", "-preset", "fast", dst])
    return frames


def ffmpeg_path(path) -> str:
    """Path as a value inside an FFmpeg filter string (forward slashes, drive colon escaped)."""
    return str(path).replace("\\", "/").replace(":", "\\:")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slug")
    args = ap.parse_args()

    pdir = project_dir(args.slug)
    spec = load_spec(args.slug, refined=True)
    style = STYLES[spec["series"]]
    (pdir / "render").mkdir(exist_ok=True)
    (pdir / "audio").mkdir(exist_ok=True)
    (pdir / "out").mkdir(exist_ok=True)

    voice_on = spec.get("voice", {}).get("enabled")
    words = []
    if voice_on:
        import voice
        voice.synthesise(spec, pdir)  # may lengthen scenes to fit the narration
        words = voice.word_timings(pdir, " ".join(s.get("narration", "") for s in spec["scenes"]))
        save_json(pdir / "spec.refined.json", spec)

    scenes, parts, cursor = [], [], 0.0
    for s in spec["scenes"]:
        clip = pdir / "clips" / f"{s['id']}.mp4"
        if not clip.exists():
            raise SystemExit(f"Missing {clip}; run the animate stage")
        # A long line has chained shots (s6.mp4, s6-2.mp4, ...); they share its time by length
        chain = [clip] + sorted(pdir.glob(f"clips/{s['id']}-*.mp4"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        lengths = [probe_duration(c) for c in chain]
        frames = 0
        for j, (c, length) in enumerate(zip(chain, lengths)):
            parts.append(pdir / "render" / f"{c.stem}.mp4")
            frames += conform_clip(c, parts[-1], s["seconds"] * length / sum(lengths), punch=j % 2 == 1)
        seconds = frames / FINAL_FPS  # whole frames, so text and cuts stay in step
        scenes.append({"id": s["id"], "start": cursor, "end": cursor + seconds,
                       "label": s.get("label", ""), "sublabel": s.get("sublabel", "")})
        cursor += seconds
    total = cursor

    music = None
    if spec.get("music") == "auto":
        from music import make_music
        music = pdir / "audio" / "music.wav"
        make_music(total, style["music_mood"], music)
    elif spec.get("music"):
        src = ROOT / spec["music"]
        music = pdir / "audio" / f"music_in{src.suffix}"
        shutil.copy(src, music)

    ass = pdir / "out" / "overlay.ass"
    write_ass(ass, scenes, spec.get("pov") or spec.get("header", ""), style["header_style"], style["accent"],
              words, spec.get("disclaimer", ""), total, style.get("caption_mode", "group"))

    # Inputs: the scene clips, then voice and music if present
    cmd = [FFMPEG, "-loglevel", "error", "-y"]
    for part in parts:
        cmd += ["-i", part]
    n = len(parts)
    graph = ["".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[cat]",
             f"[cat]ass='{ffmpeg_path(ass)}':fontsdir='{ffmpeg_path(FONTS_DIR)}'[v]"]
    tracks = []
    if voice_on:
        cmd += ["-i", pdir / "audio" / "voice.wav"]
        graph.append(f"[{n + len(tracks)}:a]aresample=48000,atrim=0:{total:.3f}[voice]")
        tracks.append("[voice]")
    if music:
        cmd += ["-stream_loop", "-1", "-i", music]  # loops a custom track that is shorter than the video
        level = 0.18 if voice_on else 0.8  # music sits under the voice
        graph.append(f"[{n + len(tracks)}:a]aresample=48000,atrim=0:{total:.3f},volume={level},"
                     f"afade=t=in:d=0.33,afade=t=out:st={total - 0.67:.3f}:d=0.67[music]")
        tracks.append("[music]")
    if len(tracks) > 1:
        graph.append(f"{''.join(tracks)}amix=inputs={len(tracks)}:normalize=0:duration=longest[mix]")
        tracks = ["[mix]"]

    draft = pdir / "out" / "draft.mov"  # PCM audio inside; finalize does the delivery encode
    cmd += ["-filter_complex", ";".join(graph), "-map", "[v]"]
    if tracks:
        cmd += ["-map", tracks[0], "-c:a", "pcm_s16le"]
    cmd += ["-t", f"{total:.3f}", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-r", str(FINAL_FPS), draft]
    run(cmd)
    print("Draft:", draft)


if __name__ == "__main__":
    main()
