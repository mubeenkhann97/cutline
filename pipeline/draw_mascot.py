"""Rough flat drawings of organ mascots, used as the starting image for keyframes.

SDXL draws a cute heart or tooth from text alone, but not lungs, kidneys or a
liver: it falls back to a generic blob. A flat silhouette with a face, passed
to keyframes.py --ref at strength 0.6-0.7, gives it the shape to render in 3D.

    python draw_mascot.py <slug> lungs [--colour grey]
"""
import argparse

from PIL import Image, ImageDraw, ImageFilter

from common import project_dir

W, H = 768, 1344
COLOURS = {"pink": ((236, 120, 140), (200, 80, 105)), "grey": ((128, 120, 122), (92, 86, 88))}


def background(draw):
    for y in range(H):  # warm red-to-pink vertical gradient, like soft tissue
        t = y / H
        draw.line([(0, y), (W, y)], fill=(int(120 + 60 * t), int(30 + 30 * t), int(45 + 30 * t)))


def face(draw, cx, cy, scale=1.0):
    for dx in (-70, 70):
        x = cx + dx * scale
        draw.ellipse([x - 52 * scale, cy - 60 * scale, x + 52 * scale, cy + 60 * scale], fill="white")
        draw.ellipse([x - 26 * scale, cy - 22 * scale, x + 26 * scale, cy + 34 * scale], fill=(40, 30, 30))
        draw.ellipse([x - 10 * scale, cy - 14 * scale, x + 4 * scale, cy], fill="white")
    draw.chord([cx - 60 * scale, cy + 70 * scale, cx + 60 * scale, cy + 150 * scale], 0, 180, fill=(90, 20, 30))


def lungs(draw, colour):
    body, shade = COLOURS[colour]
    # Windpipe and the two branches into the lobes
    draw.rounded_rectangle([W / 2 - 34, 230, W / 2 + 34, 520], 30, fill=(235, 215, 210))
    draw.line([(W / 2, 500), (W / 2 - 120, 600)], fill=(235, 215, 210), width=50)
    draw.line([(W / 2, 500), (W / 2 + 120, 600)], fill=(235, 215, 210), width=50)
    # Two lobes, the left one slightly smaller as in real lungs
    draw.ellipse([90, 480, W / 2 + 10, 1120], fill=shade)
    draw.ellipse([105, 470, W / 2 - 5, 1100], fill=body)
    draw.ellipse([W / 2 - 10, 460, W - 70, 1130], fill=shade)
    draw.ellipse([W / 2 + 5, 450, W - 85, 1110], fill=body)
    # Tiny arms
    draw.line([(110, 800), (40, 700)], fill=shade, width=26)
    draw.line([(W - 90, 800), (W - 20, 700)], fill=shade, width=26)
    face(draw, W / 2, 700)


SHAPES = {"lungs": lungs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    ap.add_argument("shape", choices=sorted(SHAPES))
    ap.add_argument("--colour", choices=sorted(COLOURS), default="pink")
    args = ap.parse_args()

    out = project_dir(args.slug) / "characters" / f"{args.shape}_{args.colour}.png"
    out.parent.mkdir(exist_ok=True)
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    background(draw)
    SHAPES[args.shape](draw, args.colour)
    img.filter(ImageFilter.GaussianBlur(2)).save(out)  # soft edges read as a shape, not as flat graphics
    print("Wrote", out)


if __name__ == "__main__":
    main()
