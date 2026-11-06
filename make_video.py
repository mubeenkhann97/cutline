"""One command for the whole Short.

    python make_video.py <slug>                       # every stage
    python make_video.py <slug> --stages refine keyframes
    python make_video.py <slug> --stages animate assemble finalize

Stages: refine -> keyframes -> animate -> assemble -> finalize
Extra options for a stage go after "--", e.g.
    python make_video.py teeth-vs-sugar --stages keyframes -- --n 4
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    "refine": "refine_prompt.py",
    "keyframes": "keyframes.py",
    "animate": "animate.py",
    "assemble": "assemble.py",
    "finalize": "finalize.py",
}


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        extra = argv[argv.index("--") + 1:]
        argv = argv[: argv.index("--")]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    ap.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    args = ap.parse_args(argv)

    for stage in args.stages:
        print(f"\n=== {stage} ===")
        cmd = [sys.executable, str(ROOT / "pipeline" / STAGES[stage]), args.slug]
        if len(args.stages) == 1:
            cmd += extra
        subprocess.run(cmd, check=True, cwd=ROOT / "pipeline")


if __name__ == "__main__":
    main()
