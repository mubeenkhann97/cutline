"""Original background music, synthesised locally, so there are no licence or Content ID issues.

A soft pad plus a plucked arpeggio over a 4-chord loop, sized to the video
length. Drop your own royalty-free track in the spec ("music": "path.mp3") to
use that instead.
"""
import numpy as np
import soundfile as sf

SR = 44100
# Moods: (bpm, chord roots in semitones from A3, chord quality per chord)
MOODS = {
    "playful": (112, [0, 5, 7, 5], ["maj", "maj", "maj", "min"]),
    "uplifting": (96, [0, 7, 9, 5], ["maj", "maj", "min", "maj"]),
    "warm": (78, [0, 9, 5, 7], ["maj", "min", "maj", "maj"]),
    "curious": (100, [0, 3, 8, 7], ["min", "maj", "maj", "maj"]),
}
INTERVALS = {"maj": [0, 4, 7, 12], "min": [0, 3, 7, 12]}


def hz(semitones_from_a3: float) -> float:
    return 220.0 * 2 ** (semitones_from_a3 / 12)


def tone(freq, dur, harmonics=(1, 0.35, 0.12), attack=0.01, decay=None):
    t = np.arange(int(dur * SR)) / SR
    wave = sum(a * np.sin(2 * np.pi * freq * (i + 1) * t) for i, a in enumerate(harmonics))
    env = np.minimum(1, t / attack)
    if decay:
        env *= np.exp(-t / decay)
    return wave * env


def make_music(seconds: float, mood: str, out_path) -> None:
    bpm, roots, quals = MOODS.get(mood, MOODS["playful"])
    beat = 60 / bpm
    bar = beat * 4
    n = int(seconds * SR) + SR
    mix = np.zeros((n, 2))

    def add(sig, start, pan=0.0, gain=1.0):
        s = int(start * SR)
        if s >= n:
            return
        e = min(n, s + len(sig))
        mix[s:e, 0] += sig[: e - s] * gain * (1 - pan) / 2
        mix[s:e, 1] += sig[: e - s] * gain * (1 + pan) / 2

    t, i = 0.0, 0
    while t < seconds:
        root, iv = roots[i % 4], INTERVALS[quals[i % 4]]
        for k, semi in enumerate(iv[:3]):  # pad
            add(tone(hz(root + semi - 12), bar + 0.5, (1, 0.2), attack=0.6), t, pan=(k - 1) * 0.5, gain=0.10)
        add(tone(hz(root - 24), bar, (1, 0.5), attack=0.05, decay=1.2), t, gain=0.22)  # bass
        for step in range(8):  # pluck arpeggio on 8th notes
            semi = iv[[0, 1, 2, 3, 2, 1, 2, 3][step]]
            add(tone(hz(root + semi + 12), beat, (1, 0.4, 0.2), decay=0.18), t + step * beat / 2,
                pan=0.4 if step % 2 else -0.4, gain=0.09)
        t += bar
        i += 1

    mix = mix[: int(seconds * SR)]
    fade = int(0.8 * SR)
    mix[:fade] *= np.linspace(0, 1, fade)[:, None]
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    mix /= max(1e-6, np.abs(mix).max()) / 0.8
    sf.write(str(out_path), mix.astype(np.float32), SR)
