from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Optional

from src.platform.observability.logger import logger
from src.platform.runtime.native.ops.model_tools.quantized import is_unquantized_file


def keep_unquantized(
    models: Iterable[Any],
    path_for_model: Callable[[str], Optional[Path]],
) -> list[Any]:
    kept: list[Any] = []
    for model in models:
        try:
            path = path_for_model(model.id)
        except Exception as exc:
            logger.debug(f"[QUANT_FILTER] no readable path for model {getattr(model, 'id', None)}: {exc}")
            continue
        if path is not None and is_unquantized_file(path):
            kept.append(model)
    return kept
