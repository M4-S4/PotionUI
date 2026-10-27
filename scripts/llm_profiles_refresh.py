from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from src.features.llm.harness.profile_catalog import MODELS_DEV_URL, PROVIDER_ADAPTERS, refresh_seed
from src.features.llm.harness.profile_seed import SEED_PATH
from src.platform.http.tls import client_ssl_context

FETCH_TIMEOUT_SECONDS = 120.0


def load_models_dev(input_path: str | None, url: str) -> dict:
    if input_path:
        return json.loads(Path(input_path).read_text(encoding="utf-8"))
    response = httpx.get(url, timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True, verify=client_ssl_context())
    response.raise_for_status()
    return response.json()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the catalog section of src/std/llm/profiles.yml from models.dev.")
    parser.add_argument("--input", help="read this models.dev api.json instead of fetching it")
    parser.add_argument("--url", default=MODELS_DEV_URL, help="models.dev api.json location")
    parser.add_argument("--seed", default=str(SEED_PATH), help="seed table to rewrite")
    parser.add_argument("--check", action="store_true", help="exit 1 instead of writing when the catalog is stale")
    args = parser.parse_args(argv)

    seed_path = Path(args.seed)
    current = seed_path.read_text(encoding="utf-8")
    refreshed = refresh_seed(current, load_models_dev(args.input, args.url))
    if refreshed == current:
        print("catalog is up to date")
        return 0
    if args.check:
        print("catalog is stale")
        return 1
    seed_path.write_text(refreshed, encoding="utf-8", newline="\n")
    print(f"catalog rewritten for {', '.join(sorted(PROVIDER_ADAPTERS))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
