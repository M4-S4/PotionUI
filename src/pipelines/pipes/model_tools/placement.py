import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.platform.util.path_resolution import is_within

from src.pipelines.pipes.model_tools.errors import ModelToolsRefused

LORA_TYPE = "lora"
MODEL_TYPES = frozenset({"diffusion_model", "unet", "checkpoint", "text_encoder"})
SAFETENSORS_SUFFIXES = (".safetensors", ".sft")
FREE_SPACE_MARGIN_BYTES = 64 * 1024 * 1024
GIB = 1024 ** 3


@dataclass(frozen=True)
class ModelFile:
    path: Path
    model_type: str
    type_dir: Path
    root_id: str


@dataclass(frozen=True)
class Destination:
    directory: Path
    path: Path
    model_type: str
    type_dir: Path


def format_size(num_bytes: int) -> str:
    if num_bytes >= GIB:
        return f"{num_bytes / GIB:.2f} GiB"
    return f"{num_bytes / (1024 ** 2):.1f} MiB"


def _root_for(roots: Any, root_id: str) -> Any:
    return next((r for r in roots.roots() if r.id == root_id), None)


def _require_writable(roots: Any, root_id: str, label: str) -> None:
    root = _root_for(roots, root_id)
    if root is None:
        raise ModelToolsRefused(f"{label} is in a model folder that is no longer configured")
    if root.read_only:
        raise ModelToolsRefused(f"{label} is in the read-only model folder '{root.label}'; output cannot be written next to it")


def locate_model_file(roots: Any, raw: Any, label: str, allowed_types: frozenset[str]) -> ModelFile:
    if not isinstance(raw, str) or not raw.strip():
        raise ModelToolsRefused(f"{label}: no file was selected")
    path = Path(os.path.abspath(os.path.expanduser(raw.strip())))
    if path.suffix.lower() not in SAFETENSORS_SUFFIXES:
        raise ModelToolsRefused(f"{label}: only safetensors files are supported ({path.name})")
    if not path.is_file():
        raise ModelToolsRefused(f"{label}: {path.name} was not found on this host")
    location = roots.to_logical(path)
    if location is None:
        raise ModelToolsRefused(f"{label}: {path.name} is not inside a configured model folder")
    if location.model_type not in allowed_types:
        wanted = ", ".join(sorted(allowed_types))
        raise ModelToolsRefused(f"{label}: {path.name} is a {location.model_type} file; expected one of: {wanted}")
    binding = roots.binding_dir(location.binding_id) if location.binding_id else None
    if binding is None:
        raise ModelToolsRefused(f"{label}: {path.name} is in a model folder that is not bound to {location.model_type}")
    return ModelFile(path=path, model_type=location.model_type, type_dir=Path(binding.path), root_id=location.root_id)


def contained_destination(directory: Path, type_dir: Path, name: str, model_type: str) -> Destination:
    target = directory / name
    if Path(name).name != name or target.parent != directory:
        raise ModelToolsRefused("The output file name must be a plain file name inside the model folder")
    if not is_within(type_dir, target):
        raise ModelToolsRefused("The output would land outside the model folder and was refused")
    if not is_within(os.path.realpath(type_dir), os.path.realpath(directory)):
        raise ModelToolsRefused("The output folder resolves outside the model folder (symbolic link) and was refused")
    return Destination(directory=directory, path=target, model_type=model_type, type_dir=type_dir)


def destination_beside(roots: Any, source: ModelFile, name: str) -> Destination:
    _require_writable(roots, source.root_id, source.path.name)
    return contained_destination(source.path.parent, source.type_dir, name, source.model_type)


def destination_in_lora_root(roots: Any, name: str) -> Destination:
    try:
        entry = roots.write_dir(LORA_TYPE)
    except Exception as exc:
        raise ModelToolsRefused(f"No writable LoRA folder is configured ({exc})") from exc
    type_dir = Path(entry.path)
    return contained_destination(type_dir, type_dir, name, LORA_TYPE)


def check_overwrite(destination: Destination, protected: list[Path], overwrite: bool) -> None:
    resolved = Path(os.path.realpath(destination.path))
    for source in protected:
        if Path(os.path.realpath(source)) == resolved:
            raise ModelToolsRefused(f"The output would overwrite its own input ({source.name}); pick another file name")
    if os.path.lexists(destination.path) and not overwrite:
        raise ModelToolsRefused(
            f"{destination.path.name} already exists. Tick Overwrite to replace it, or pick another file name"
        )


def check_free_space(directory: Path, required_bytes: int) -> None:
    free = shutil.disk_usage(directory).free
    needed = required_bytes + FREE_SPACE_MARGIN_BYTES
    if needed > free:
        raise ModelToolsRefused(
            f"Not enough free disk space in {directory}: the output needs about {format_size(needed)} "
            f"but only {format_size(free)} is available"
        )
