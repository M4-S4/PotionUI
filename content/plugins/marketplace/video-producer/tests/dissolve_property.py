import json
import random
import sys
from pathlib import Path

from backend.chat import edits
from backend.dissolve import cuts, joins, shown
from backend.exports.common import js_round
from backend.formats import SLOW_SPEEDS

SEED = 116
COUNT = 60
CADENCES = (24, 25, 30, 48, 50, 60)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dissolve_property.json"
ASKS = (0, 0, 0.4, 0.6, 1, 1.2, 2)


def _clip(rng):
    dur = round(rng.choice([rng.uniform(0.08, 6), rng.uniform(0.08, 0.5), rng.choice([0.4, 1, 2, 2.5, 4])]), 2)
    clip = {"dur": dur}
    ask = rng.choice([*ASKS, round(rng.uniform(0.08, 2), 2)])
    if ask:
        clip["dissolve"] = ask
    if rng.random() < 0.15:
        clip["speed"] = rng.choice(SLOW_SPEEDS)
    if rng.random() < 0.3:
        clip["fadeIn"] = rng.choice([0.2, 0.4, 1])
    if rng.random() < 0.3:
        clip["fadeOut"] = rng.choice([0.2, 0.4, 1])
    return clip


def documents():
    rng = random.Random(SEED)
    found = []
    for _ in range(COUNT):
        clips = [_clip(rng) for _ in range(rng.randint(2, 14))]
        gaps = {index: rng.choice([0.04, 0.2, 1]) for index in range(1, len(clips)) if rng.random() < 0.1}
        bare = sorted(index for index in range(len(clips)) if rng.random() < 0.15)
        found.append({"clips": clips, "gaps": {str(index): gap for index, gap in sorted(gaps.items())}, "bare": bare})
    return found


def laid(case):
    doc = {"clips": [{"id": f"c{index}", "start": 99, **clip} for index, clip in enumerate(case["clips"])], "prompts": [], "frames": []}
    edits.relayout(doc)
    laid_out = [clip["start"] for clip in doc["clips"]]
    shift = 0.0
    for index, clip in enumerate(doc["clips"]):
        shift += case["gaps"].get(str(index), 0)
        if shift:
            clip["start"] = float(f"{clip['start'] + shift:.3f}")
    return laid_out, doc["clips"]


def expected(case):
    laid_out, clips = laid(case)
    return {
        **case,
        "laid": laid_out,
        "starts": [clip["start"] for clip in clips],
        "joins": joins(clips),
        "spans": [[span.start, span.end] for span in shown(clips)],
        "cuts": {str(fps): [None if cut is None else [cut.start, cut.end, cut.overlap, cut.head] for cut in cuts(clips, lambda seconds, rate=fps: js_round(seconds * rate))] for fps in CADENCES},
    }


def make():
    return {"seed": SEED, "fps": list(CADENCES), "docs": [expected(case) for case in documents()]}


def write(path=FIXTURE):
    path.write_text(json.dumps(make(), separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    write(Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE)
