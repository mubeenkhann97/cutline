"""Optional narration (Kokoro-82M, Apache 2.0) and word-timed captions (Whisper).

Writes audio/voice.wav with each scene's line placed at that scene's start, and
audio/words.json for the caption track. Scenes stretch to fit their narration.
"""
import json

import numpy as np
import soundfile as sf

SR = 24000  # Kokoro output rate


def synthesise(spec: dict, pdir) -> None:
    """Story videos give each speaker their own voice (spec["characters"][speaker]["voice"])."""
    from kokoro import KPipeline

    cfg = spec.get("voice", {})
    characters = spec.get("characters", {})
    story = bool(characters)
    pipelines = {}  # one per accent: Kokoro voice ids start with a (American) or b (British)
    adir = pdir / "audio"
    adir.mkdir(exist_ok=True)

    pieces, cursor = [], 0.0
    for s in spec["scenes"]:
        voice_id = characters.get(s.get("speaker"), {}).get("voice") or cfg.get("voice", "af_heart")
        if voice_id[0] not in pipelines:
            pipelines[voice_id[0]] = KPipeline(lang_code=voice_id[0])
        tts = pipelines[voice_id[0]]
        chunks = [a.numpy() if hasattr(a, "numpy") else a
                  for _, _, a in tts(s["narration"], voice=voice_id, speed=cfg.get("speed", 1.0))] \
            if s.get("narration") else []
        line = np.concatenate(chunks) if chunks else np.zeros(0)
        # Dialogue needs snappy turn-taking; narration gets a little more air around each line
        lead, tail = (0.1, 0.2) if story else (0.25, 0.35)
        s["seconds"] = max(s["seconds"], round(len(line) / SR + lead + tail, 2))
        start = int((cursor + lead) * SR)
        pieces.append((start, line))
        cursor += s["seconds"]

    track = np.zeros(int(cursor * SR) + SR)
    for start, line in pieces:
        track[start: start + len(line)] += line
    sf.write(str(adir / "voice.wav"), track.astype(np.float32), SR)


def align_to_script(heard: list, script_words: list) -> list:
    """Map script words onto Whisper's timed words; merged or split words share their timing."""
    from difflib import SequenceMatcher

    norm = lambda t: "".join(c for c in t.lower() if c.isalnum())
    ops = SequenceMatcher(None, [norm(w["text"]) for w in heard], [norm(t) for t in script_words],
                          autojunk=False).get_opcodes()
    out = []
    for tag, i1, i2, j1, j2 in ops:
        if j1 == j2:
            continue  # Whisper heard something the script does not say
        if i1 == i2:  # script word Whisper missed: borrow the neighbouring time
            t = heard[min(i1, len(heard) - 1)]["start"] if heard else 0.0
            out += [{"text": s, "start": t, "end": t} for s in script_words[j1:j2]]
            continue
        if i2 - i1 == j2 - j1:  # same words (or one-for-one swaps like "9" for "Nine"): keep each one's timing
            out += [{"text": s, "start": h["start"], "end": h["end"]}
                    for s, h in zip(script_words[j1:j2], heard[i1:i2])]
            continue
        start, end = heard[i1]["start"], heard[i2 - 1]["end"]
        n = j2 - j1
        for k, s in enumerate(script_words[j1:j2]):
            out.append({"text": s, "start": round(start + (end - start) * k / n, 2),
                        "end": round(start + (end - start) * (k + 1) / n, 2)})
    return out


def word_timings(pdir, script: str = "") -> list:
    import whisper

    from common import WHISPER_DIR
    model = whisper.load_model("base", download_root=str(WHISPER_DIR))
    result = model.transcribe(str(pdir / "audio" / "voice.wav"), language="en", word_timestamps=True, fp16=False)
    words = [
        {"text": w["word"].strip(), "start": round(float(w["start"]), 2), "end": round(float(w["end"]), 2)}
        for seg in result["segments"] for w in seg.get("words", [])
    ]
    # Whisper's timing is good but its spelling drifts ("sends" -> "sands", "clean-up" -> "clean" "-up"):
    # keep its timings and show the script's own words
    if script:
        words = align_to_script(words, script.split())
    (pdir / "audio" / "words.json").write_text(json.dumps(words, indent=1), encoding="utf-8")
    return words
