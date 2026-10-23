from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

LISTING_DIR = Path(__file__).resolve().parent / "fixtures" / "headers"
SOURCE_DIR = Path(__file__).resolve().parents[4] / "ai"
SCALAR = "-"


def listing_text(header: Dict[str, dict]) -> str:
    lines = []
    for name in sorted(header):
        if name == "__metadata__":
            continue
        entry = header[name]
        shape = ",".join(str(int(dim)) for dim in entry["shape"]) or SCALAR
        lines.append(f"{name} {shape} {entry['dtype']}")
    return "\n".join(lines) + "\n"


def parse_listing(text: str) -> Dict[str, dict]:
    header: Dict[str, dict] = {}
    for line in text.splitlines():
        if not line:
            continue
        name, shape, dtype = line.split(" ")
        header[name] = {
            "dtype": dtype,
            "shape": [] if shape == SCALAR else [int(dim) for dim in shape.split(",")],
        }
    return header


def listing_path(name: str) -> Path:
    return LISTING_DIR / f"{name}.txt"


def source_path(name: str) -> Path:
    family, stem = name.split("/", 1)
    return SOURCE_DIR / family / f"{stem}_header.json"


def load_header(name: str) -> Dict[str, dict]:
    return parse_listing(listing_path(name).read_text(encoding="utf-8"))


def listing_names() -> List[str]:
    return sorted(path.relative_to(LISTING_DIR).with_suffix("").as_posix() for path in LISTING_DIR.rglob("*.txt"))


def derive_listing(name: str) -> str:
    with source_path(name).open(encoding="utf-8") as handle:
        return listing_text(json.load(handle))


def regenerate(names: Optional[Iterable[str]] = None) -> List[str]:
    written = []
    for name in names or listing_names():
        if not source_path(name).is_file():
            continue
        target = listing_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(derive_listing(name))
        written.append(name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    for name in regenerate(parser.parse_args().names or None):
        print(name)


if __name__ == "__main__":
    main()
