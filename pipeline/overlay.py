"""On-screen text for the Short as an ASS subtitle file, burned in by FFmpeg (libass).

Same layout as the old Remotion composition. Everything stays inside the Shorts
safe zone: YouTube's UI covers roughly the bottom 20% and a strip on the right,
so text sits between 8% and 80% of the height (the disclaimer is lowest, at 78%,
clear of a two-line label).
"""
import math

from common import FINAL_H, FINAL_W

FONT = "Arial Black"
UI_FONT = "Segoe UI"
WORD_FONT = "Impact"  # tall and condensed: one word fills the width without covering the scene
WORD_SIZE = 190  # ASS size; capitals come out about 140 px tall on the 1080 px frame
WORD_CHAR_WIDTH = 0.5  # Impact capital width as a share of its size; long words shrink to fit
# libass sizes fonts by line height, CSS by em: measured against the Remotion render, ASS needs
# 1.33x the CSS pixel size for the same text width. Sizes below are written in CSS pixels.
CSS_TO_ASS = 1.33


def fs(css_px: int) -> int:
    return round(css_px * CSS_TO_ASS)


def y(fraction: float) -> int:
    return round(FINAL_H * fraction)


def ass_colour(hex_rgb: str, alpha: int = 0) -> str:
    """'#RRGGBB' -> ASS '&HAABBGGRR' (alpha 0 = opaque)."""
    r, g, b = hex_rgb.lstrip("#")[0:2], hex_rgb.lstrip("#")[2:4], hex_rgb.lstrip("#")[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ts(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def write_ass(path, scenes: list, header: str, header_style: str, accent: str, words: list,
              disclaimer: str, total: float, caption_mode: str = "group") -> None:
    """scenes: [{"start", "end", "label", "sublabel"}] in seconds.

    caption_mode "group": three words at a time near the top, the spoken one highlighted (explainers).
    caption_mode "word": one big word at a time below the characters' faces, each popping in (story videos,
    where the frame changes with every word and nothing else is on screen).
    """
    white, acc = "&H00FFFFFF", ass_colour(accent)
    # Outline + soft drop shadow, like the CSS text-shadow it replaces
    styles = [
        # name, font, size, colour, outline colour, back colour, bold, border style, outline, shadow, align, margin L/R, margin V
        ("Label", FONT, fs(74), acc, "&H00000000", "&H26000000", 0, 1, 4, 3, 8, 86, y(0.63)),
        ("Header", FONT, fs(64), white, "&H00000000", "&H26000000", 0, 1, 4, 3, 8, 54, y(0.08)),
        ("Caption", FONT, fs(60), white, "&H00000000", "&H26000000", 0, 1, 4, 3, 8, 76, y(0.16)),
        ("Word", WORD_FONT, WORD_SIZE, acc, "&H00000000", "&H59000000", 0, 1, 7, 5, 8, 60, y(0.71)),  # below centred faces
        ("Pov", UI_FONT, fs(50), "&H00111111", "&H00FFFFFF", "&H00FFFFFF", -1, 1, 0, 0, 8, 48, y(0.08) + 22),
        ("Disclaimer", UI_FONT, fs(30), "&H1AFFFFFF", "&H00000000", "&H26000000", 0, 1, 2, 1, 8, 60, y(0.78)),
        ("Box", UI_FONT, 10, white, white, white, 0, 1, 0, 0, 7, 0, 0),
    ]
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {FINAL_W}", f"PlayResY: {FINAL_H}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
    ]
    for name, font, size, col, out, back, bold, border, outline, shadow, align, mlr, mv in styles:
        lines.append(f"Style: {name},{font},{size},{col},{col},{out},{back},{bold},0,0,0,100,100,"
                     f"{2 if name in ('Header', 'Label') else 0},0,{border},{outline},{shadow},{align},{mlr},{mlr},{mv},1")
    lines += ["", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]

    def event(start, end, style, text, layer=1, margin_v=0):  # margin_v 0 = use the style's
        lines.append(f"Dialogue: {layer},{ts(start)},{ts(end)},{style},,0,0,{margin_v},,{text}")

    # Scene labels pop in (overshoot, then settle) shortly after each cut
    pop = r"{\fscx0\fscy0\t(130,300,\fscx112\fscy112)\t(300,450,\fscx100\fscy100)}"
    for s in scenes:
        if not s.get("label"):
            continue
        text = pop + esc(s["label"])
        if s.get("sublabel"):
            text += r"\N{\fs" + str(fs(46)) + r"\c" + white + "}" + esc(s["sublabel"])
        event(s["start"], s["end"], "Label", text)

    caption_v = 0
    if header and header_style == "pov":
        # Full-width white bar behind the POV line; height grows with wrapped lines
        n_lines = max(1, math.ceil(len(header) / 36))
        bar_h = 44 + 68 * n_lines
        event(0, total, "Box", rf"{{\an7\pos(0,{y(0.08)})\p1}}m 0 0 l {FINAL_W} 0 {FINAL_W} {bar_h} 0 {bar_h}", layer=0)
        event(0, total, "Pov", esc(header), layer=1)
        # A two-line bar reaches past the usual caption line, so captions start below it
        if y(0.08) + bar_h + 24 > y(0.16):
            caption_v = y(0.08) + bar_h + 24
    elif header:
        event(0, total, "Header", esc(header))

    if caption_mode == "word":
        # Each word stays up until the next one starts (short pauses do not blank the screen)
        word_pop = r"{\fscx70\fscy70\t(0,90,\fscx108\fscy108)\t(90,160,\fscx100\fscy100)}"
        for i, w in enumerate(words):
            end = w["end"] + 0.3
            if i + 1 < len(words):
                nxt = words[i + 1]["start"]
                end = nxt if nxt - w["end"] < 0.6 else min(end, nxt)
            if end > w["start"]:
                text = w["text"].upper()
                size = min(WORD_SIZE, int((FINAL_W - 140) / (WORD_CHAR_WIDTH * max(len(text), 1))))
                event(w["start"], end, "Word", "{\\fs" + str(size) + "}" + word_pop + esc(text))
        words = []  # skip the grouped captions below

    # Word-by-word captions in groups of three; the spoken word takes the accent colour
    for i, w in enumerate(words):
        start = w["start"]
        end = w["end"] + 0.15
        if i + 1 < len(words):
            end = min(end, max(words[i + 1]["start"], start))
        if end <= start:
            continue
        g0 = i - i % 3
        parts = []
        for j, gw in enumerate(words[g0:g0 + 3], g0):
            word = esc(gw["text"].upper())
            parts.append(r"{\c" + acc + "}" + word + r"{\c" + white + "}" if j == i else word)
        event(start, end, "Caption", " ".join(parts), margin_v=caption_v)

    if disclaimer:
        # Big single words sit where the disclaimer normally goes; story videos have no header, so use the top
        top = caption_mode == "word" and not header
        event(max(0, total - 2), total, "Disclaimer", esc(disclaimer), margin_v=y(0.08) if top else 0)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
