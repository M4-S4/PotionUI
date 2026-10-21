import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

from src.pipelines.pipes.model_tools.errors import ModelToolsRefused

EXTENSION = ".safetensors"
MAX_STEM_LENGTH = 150
LORA_JOINER = "+"
ILLEGAL_CHARS = frozenset('/\\:*?"<>|')
RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)
_UNSAFE_RUN = re.compile(r"[^\w.+()\- ]+", re.UNICODE)
_SPACE_RUN = re.compile(r"\s+")
_UNDERSCORE_RUN = re.compile(r"_{2,}")
_MODEL_EXTENSIONS = (".safetensors", ".sft")


def strip_extension(name: str) -> str:
    lowered = name.lower()
    for ext in _MODEL_EXTENSIONS:
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return name


def sanitize_fragment(text: str) -> str:
    cleaned = _UNSAFE_RUN.sub("_", strip_extension(text.strip()))
    cleaned = _SPACE_RUN.sub("_", cleaned)
    return _UNDERSCORE_RUN.sub("_", cleaned).strip("._ ")


def validate_stem(stem: str) -> str:
    if not stem:
        raise ModelToolsRefused("The output file name is empty")
    bad = sorted({c for c in stem if c in ILLEGAL_CHARS})
    if bad:
        raise ModelToolsRefused(
            "The output file name cannot contain path separators or reserved characters "
            f"({' '.join(bad)}); give a plain file name"
        )
    if any(ord(c) < 32 or ord(c) == 127 for c in stem):
        raise ModelToolsRefused("The output file name contains control characters")
    if ".." in stem or stem.startswith(".") or stem.endswith((".", " ")) or stem != stem.strip():
        raise ModelToolsRefused("The output file name cannot start with a dot, end with a dot or space, or contain '..'")
    if len(stem) > MAX_STEM_LENGTH:
        raise ModelToolsRefused(f"The output file name is longer than {MAX_STEM_LENGTH} characters")
    if stem.split(".")[0].lower() in RESERVED_STEMS:
        raise ModelToolsRefused(f"'{stem}' is a reserved file name on Windows")
    return stem


def _fit(stem: str, salt: str) -> str:
    if len(stem) <= MAX_STEM_LENGTH:
        return stem
    digest = hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8]
    return f"{stem[: MAX_STEM_LENGTH - 9].rstrip('._ ')}-{digest}"


def default_stem(mode: str, source: str | None, loras: Sequence[str], quant_suffix: str) -> str:
    lora_part = LORA_JOINER.join(filter(None, (sanitize_fragment(Path(p).name) for p in loras)))
    base = sanitize_fragment(Path(source).name) if source else ""
    if mode == "quantize":
        stem = f"{base}{quant_suffix}"
    elif mode == "merge_into_model":
        stem = f"{base}{LORA_JOINER}{lora_part}"
    elif mode == "merge_and_quantize":
        stem = f"{base}{LORA_JOINER}{lora_part}{quant_suffix}"
    elif mode == "merge_loras":
        stem = lora_part
    else:
        raise ModelToolsRefused(f"Unknown mode {mode!r}")
    stem = stem.strip("._ ") or "model-tools-output"
    return _fit(stem, "|".join([mode, source or "", *loras, quant_suffix]))


def resolve_stem(
    requested: str | None, mode: str, source: str | None, loras: Sequence[str], quant_suffix: str
) -> str:
    raw = (requested or "").strip()
    if not raw:
        return validate_stem(default_stem(mode, source, loras, quant_suffix))
    return validate_stem(strip_extension(raw))


def file_name(stem: str) -> str:
    return stem + EXTENSION
