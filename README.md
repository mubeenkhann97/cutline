# AI Shorts Pipeline

Topic → refined prompts → AI keyframes → AI video clips → voice, captions and music → final upload-ready Short.
Everything runs locally with free, open-source tools and open-weight models.

## Tools

| Stage | Tool | Licence |
|---|---|---|
| Prompt refine | `pipeline/refine_prompt.py` (style bible, 77-token CLIP fitting, text-word removal); optional local LLM via [Ollama](https://ollama.com) | MIT |
| Keyframes | FLUX.1-schnell, 4-bit (diffusers + bitsandbytes); fallback SDXL 1.0 + SDXL-Lightning 8-step LoRA | Apache 2.0 / OpenRAIL++ |
| Animation | Wan 2.2 TI2V-5B (image + text → video) | Apache 2.0 |
| Voice | Kokoro-82M | Apache 2.0 |
| Captions | OpenAI Whisper (word timings), script-aligned | MIT |
| Music | `pipeline/music.py` – synthesised locally, no Content ID risk | – |
| Assembly | FFmpeg + libass: `pipeline/assemble.py` (cuts, snap-in zoom, audio mix) and `pipeline/overlay.py` (header, labels, word captions as ASS subtitles) | LGPL/GPL |
| Finish + QC | FFmpeg (loudness −14 LUFS, H.264, black/freeze detection) | LGPL/GPL |

The pipeline makes no network calls once the models are downloaded.

Remotion and HyperFrames are still installed in `node_modules` but the pipeline does not use them:

- **Remotion** is source-available, not open source. Organisations with more than 3 employees need a company licence.
- **HyperFrames** (Apache 2.0) sends usage telemetry to PostHog by default (opt-out: `HYPERFRAMES_NO_TELEMETRY` or `DO_NOT_TRACK`). It also loads GSAP from a CDN, and GSAP is free but not open source.

`remotion/Short.jsx` is the earlier layout that `overlay.py` reproduces.

Hardware used: RTX 2000 Ada laptop GPU (8 GB) + 64 GB RAM. FLUX.1-schnell (4-bit, 720x1280, 4 steps) takes about 30 s per keyframe. Wan runs with an fp8 transformer and a CPU text encoder at 704x1280. One clip is at most 41 frames (1.7 s; about 2.3 s once stretched) and takes about 8.5 minutes at 30 steps, so a 30-second story Short (about 18 clips) takes about 2.5 hours to animate. Lines longer than one clip get chained shots: each continues from the last frame of the one before, and assembly cuts between them with a punch-in. At 704x1280 small details shimmer about a third less than at 480x832 (`--size 480x832`: 4 s clips, 2.4x faster).

Videos are rendered at 24 fps, Wan's own frame rate. At 30 fps a third of the frames were repeats, which made characters judder. Motion interpolation (FFmpeg `minterpolate`) was tried, but it made double images whenever Wan's motion jumps. Assembly also lightly smooths the AI shimmer (`hqdn3d`).

## Setup
```powershell
python pipeline\download_models.py          # all models; or name groups: voice sdxl wan flux
```
FLUX.1-schnell is gated: accept its terms on its Hugging Face model page, then log in once with
`python pipeline\hf_login.py` (a read-only token, typed at a hidden prompt). `hf auth login` crashes
on this network's certificate inspection. Without FLUX the keyframe stage uses SDXL.

## Make a video

```powershell
. .\env.ps1
python make_video.py teeth-vs-sugar --stages refine keyframes
# open projects\teeth-vs-sugar\keyframes\_sheet.png, choose the best frames:
python pipeline\keyframes.py teeth-vs-sugar --pick s1=1 s3=0
python make_video.py teeth-vs-sugar --stages animate assemble finalize
```

Output in `Exports\`: `<slug>.mp4`, `<slug>_qc.png` (frame strip to review), `<slug>_upload.txt` (title, description, hashtags, sources), and `<slug>_prompts\`.

The prompt pack (`<slug>_prompts\`) lets you make the shots in another video generator:
- `prompts.md` gives each scene's line, image prompt and video prompt, ready to paste.
- `keyframes\` holds the picked image for each scene, for image-to-video tools.
- `audio\` holds each spoken line, for lip-sync tools.
- `prompts.json` has the same information for scripts.

To use outside clips, save them as `projects\<slug>\clips\s1.mp4`, `s2.mp4` and so on, then run the assemble and finalize stages. Those stages add the voice, captions and music. First delete any chained shots for those scenes (`s1-2.mp4` and so on), otherwise assembly cuts to them.

`python pipeline\prompt_pack.py <slug>` rebuilds the pack on its own. Check a service's terms and get approval before sending work content to it.

### New video
1. Copy `projects\lungs-after-quitting\spec.json` (story) or `projects\teeth-vs-sugar\spec.json` (narrated list) to `projects\<new-slug>\spec.json`.
2. Pick a topic from `content\topic-library.md` and a `series` from `pipeline\styles.json`.
3. Write 3–6 scenes: `image` (what we see), `motion` (what moves), `label`/`sublabel` (on-screen text), `narration` (optional voice).
4. Never ask the model for text; labels and captions are added in assembly.

### Story format (recommended)
The most-viewed reference Shorts are mini-stories, not narrated lists: characters talk to each other, with a twist and a one-line lesson at the end, a new shot every 2–5 s, and one big word on screen at a time. `projects\lungs-after-quitting\spec.json` is the template:

- `characters`: each one has a `look` (prompt text) and a Kokoro `voice` (e.g. `af_heart`, `bm_george`, `am_puck`).
- Each scene has a `speaker`, an optional `cast`, and a `narration` line. Scene lengths are fitted to the spoken lines.
- No `header` or `label`s. The series' `caption_mode: "word"` shows single words below the faces.
- An optional top-level `setting` replaces the series background. For example, the heart story is set in an office, because a red heart on the default red background can wash out to flat red when animated.
- Keep lines short, about 2 s each. A longer line works, but it is split into chained shots, and each extra shot adds about 8 minutes of animation.

Character tips:
- FLUX reads the whole scene description (`image_prompt_long`); SDXL only the first 77 tokens.
- FLUX draws organ mascots (lungs too) straight from text and keeps them consistent across shots when the `look` is specific. Spell out object orientation it might get wrong, e.g. "orange filter at the bottom as its feet, glowing ember tip on top".
- With SDXL only: it draws a heart or a cigarette from text, but not lungs, kidneys or a liver. For those, run `python pipeline\draw_mascot.py <slug> lungs --colour grey`, then `keyframes.py <slug> --only s1 --ref characters/lungs_grey.png --strength 0.82`.
- Avoid "human", "anthropomorphic", "anatomy" or "ribs" in prompts: SDXL then draws people and skeletons.

### Refining
- Bad keyframe: `python pipeline\keyframes.py <slug> --only s2 --n 4`, then `--pick`.
- Bad motion: `python pipeline\animate.py <slug> --only s2` (new seed) or edit that scene's `motion`, re-run `refine`, then animate.
- If a chained shot (`s1-2`) inherits noise or a distorted face from the end of the shot before, re-render that scene with `python pipeline\animate.py <slug> --only s1 --restart`. Each chained shot then starts from the keyframe again, and the punch-in makes the cut read as a second camera angle.
- Characters with a hole that looks like a mouth (such as a shoe opening) can change shape when animated. Pick keyframes where the face is clearly drawn, such as a shoe with its face on the tongue.
- Quick motion preview: `python pipeline\animate.py <slug> --steps 4` (about 3 minutes per clip).
- No AI motion at all (seconds): `python pipeline\animate.py <slug> --still`.
- On an 8 GB GPU, do not use more than 41 `--frames` at 704x1280, or more than 97 at 480x832. Memory then spills into system RAM, and a clip takes about an hour instead of 8 minutes.
- Voice off (music only, like many viral Shorts): set `"voice": {"enabled": false}`.
- Own music: `"music": "assets/music/track.mp3"` (use tracks you have the rights to, e.g. the YouTube Audio Library).

## Health content
Check every claim against a reputable source (NHS, WHO, national health bodies), list it in `sources`, keep the disclaimer, and avoid cure claims.
