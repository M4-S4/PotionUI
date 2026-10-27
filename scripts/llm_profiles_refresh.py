from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import yaml

from src.features.llm.harness.profile_catalog import MODELS_DEV_URL, refresh_seed, seed_providers
from src.features.llm.harness.profile_seed import CONTENT_DIR, MARKETPLACE_ROOT
from src.platform.http.tls import client_ssl_context

FETCH_TIMEOUT_SECONDS = 120.0
PLUGINS_DIR = Path(__file__).resolve().parent.parent / "content" / "plugins" / "marketplace"


def load_models_dev(input_path: str | None, url: str) -> dict:
    if input_path:
        return json.loads(Path(input_path).read_text(encoding="utf-8"))
    response = httpx.get(url, timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True, verify=client_ssl_context())
    response.raise_for_status()
    return response.json()


def plugin_profile_dirs() -> List[Path]:
    dirs: List[Path] = []
    for manifest_path in sorted(PLUGINS_DIR.glob("*/manifest.yml")):
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        for entry in manifest.get("llm_profiles") or []:
            dirs.append(manifest_path.parent / entry["path"])
    return dirs


def default_seed_files() -> List[Path]:
    dirs = [CONTENT_DIR / MARKETPLACE_ROOT, *plugin_profile_dirs()]
    return [path for directory in dirs for path in sorted(directory.glob("*.yml"))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the catalog section of every shipped LLM profile file that declares models_dev providers."
    )
    parser.add_argument("--input", help="read this models.dev api.json instead of fetching it")
    parser.add_argument("--url", default=MODELS_DEV_URL, help="models.dev api.json location")
    parser.add_argument("--seed", action="append", default=[], help="profile file to rewrite; repeat for several (default: every shipped file)")
    parser.add_argument("--check", action="store_true", help="exit 1 instead of writing when a catalog is stale")
    args = parser.parse_args(argv)

    paths = [Path(item) for item in args.seed] or default_seed_files()
    models_dev = load_models_dev(args.input, args.url)
    stale: List[Path] = []
    for path in paths:
        current = path.read_text(encoding="utf-8")
        if not seed_providers(current):
            continue
        refreshed = refresh_seed(current, models_dev)
        if refreshed == current:
            continue
        stale.append(path)
        if not args.check:
            path.write_text(refreshed, encoding="utf-8", newline="\n")
    if not stale:
        print("catalog is up to date")
        return 0
    names = ", ".join(path.as_posix() for path in stale)
    if args.check:
        print(f"catalog is stale: {names}")
        return 1
    print(f"catalog rewritten: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
