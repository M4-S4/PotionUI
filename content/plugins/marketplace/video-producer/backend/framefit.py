import asyncio
import copy
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from PIL import Image

from . import core_bridge, formats, workdir
from .core_bridge import MediaRefError, User

logger = logging.getLogger(__name__)

TOLERANCE = float(formats.TABLE["startImages"]["tolerance"])

CROPS = "video_producer_frame_crops"

ROLES = ("first", "last")

SUFFIX = ".png"

Ratio = Tuple[int, int]

Box = Tuple[int, int, int, int]


class CropError(Exception):
    code = "frame_crop_failed"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class Fitted:
    source: str
    value: str
    item_id: Optional[str]
    width: int
    height: int


def ratio_of(value: Any) -> Optional[Ratio]:
    size = formats.parse_size(value)
    if size is not None:
        exact = Fraction(size[0], size[1])
    else:
        ratio = formats.ratio_of(value)
        if ratio is None:
            return None
        exact = Fraction(ratio).limit_denominator(1000)
    return exact.numerator, exact.denominator


def ratio_text(ratio: Ratio) -> str:
    return f"{ratio[0]}:{ratio[1]}"


def request_ratio(resolution: Any, request: Optional[Mapping[str, Any]] = None, aspect: Any = None) -> Optional[Ratio]:
    if formats.parse_size(resolution) is not None:
        return ratio_of(resolution)
    form = (request or {}).get("form_data") or {}
    for candidate in (form.get("aspect_ratio"), aspect):
        found = ratio_of(candidate) if candidate is not None else None
        if found is not None:
            return found
    return None


def plan_ratio(plan: Any) -> Optional[Ratio]:
    return request_ratio(plan.resolution, plan.request, (plan.cloud_params or {}).get("aspect_ratio"))


def lost(width: int, height: int, ratio: Ratio) -> float:
    return formats.loss(width / height, ratio[0] / ratio[1])


def _even(value: float, limit: int) -> int:
    rounded = max(2, 2 * round(value / 2))
    return rounded if rounded <= limit else limit - limit % 2


def crop_box(width: int, height: int, ratio: Ratio) -> Optional[Box]:
    if width < 2 or height < 2 or lost(width, height, ratio) <= TOLERANCE:
        return None
    wanted = ratio[0] / ratio[1]
    if width / height > wanted:
        kept_height = height - height % 2
        kept_width = _even(kept_height * wanted, width)
    else:
        kept_width = width - width % 2
        kept_height = _even(kept_width / wanted, height)
    if kept_width < 2 or kept_height < 2:
        return None
    return (width - kept_width) // 2, (height - kept_height) // 2, kept_width, kept_height


def picture_size(path: Optional[Path]) -> Optional[Tuple[int, int]]:
    if path is None:
        return None
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def bound(request: Mapping[str, Any]) -> List[Dict[str, Any]]:
    document = ((request or {}).get("form_data") or {}).get("video_director")
    if not isinstance(document, Mapping):
        return []
    return [
        entry
        for entry in document.get("media") or []
        if isinstance(entry, Mapping) and entry.get("role") in ROLES and (entry.get("media") or {}).get("type") == "image"
    ]


def values(request: Mapping[str, Any]) -> List[str]:
    found = [str((entry.get("media") or {}).get("path") or "") for entry in bound(request)]
    return list(dict.fromkeys(value for value in found if value))


def roles(request: Mapping[str, Any]) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for entry in bound(request):
        value = str((entry.get("media") or {}).get("path") or "")
        if value and entry["role"] not in found.setdefault(value, []):
            found[value].append(entry["role"])
    return found


def replace(request: Mapping[str, Any], old: str, new: str) -> Dict[str, Any]:
    changed = copy.deepcopy(dict(request))
    for entry in bound(changed):
        media = entry["media"]
        for key in ("path", "relative_path"):
            if media.get(key) == old:
                media[key] = new
    return changed


def _key(user_id: str, source: str, ratio: Ratio) -> str:
    return f"{user_id}|{ratio_text(ratio)}|{source}"


def recorded(user_id: str, source: str, ratio: Ratio) -> Optional[Fitted]:
    from src.plugin_api import db

    with db.get_cursor() as cursor:
        cursor.execute(f"SELECT value, item_id, width, height FROM {CROPS} WHERE key = ?", (_key(user_id, source, ratio),))
        row = cursor.fetchone()
    if row is None:
        return None
    return Fitted(source=source, value=row["value"], item_id=row["item_id"], width=int(row["width"]), height=int(row["height"]))


def record(user_id: str, ratio: Ratio, fitted: Fitted) -> None:
    from src.plugin_api import db

    now = datetime.now(timezone.utc).isoformat()
    with db.get_cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {CROPS} (key, user_id, source, ratio, value, item_id, width, height, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, item_id = excluded.item_id, width = excluded.width, height = excluded.height, created_at = excluded.created_at",
            (_key(user_id, fitted.source, ratio), user_id, fitted.source, ratio_text(ratio), fitted.value, fitted.item_id, fitted.width, fitted.height, now),
        )


def forget(user_id: str, source: str, ratio: Ratio) -> None:
    from src.plugin_api import db

    with db.get_cursor() as cursor:
        cursor.execute(f"DELETE FROM {CROPS} WHERE key = ?", (_key(user_id, source, ratio),))


def _still_there(owner: User, fitted: Fitted) -> bool:
    if not fitted.item_id:
        return False
    try:
        resolved = core_bridge.resolve_media(owner, {"kind": "library", "item_id": fitted.item_id})
    except MediaRefError:
        return False
    return str(resolved.get("value")) == fitted.value


def _known(owner: User, source: str, ratio: Ratio) -> Optional[Fitted]:
    found = recorded(owner.id, source, ratio)
    if found is None:
        return None
    if _still_there(owner, found):
        return found
    forget(owner.id, source, ratio)
    return None


def _picture(owner: User, ref: Mapping[str, Any]) -> Tuple[Optional[Path], Optional[Path]]:
    try:
        path = core_bridge.media_file_path(owner, ref)
    except Exception:
        path = None
    if path and Path(path).is_file():
        return Path(path), None
    try:
        folder = workdir.scratch(workdir.FRAMES, owner.id)
    except Exception:
        logger.warning("[VIDEO-PRODUCER] No work folder to read the picture %s in", ref, exc_info=True)
        return None, None
    try:
        copied = core_bridge.local_media_file(owner, ref, folder)
    except Exception:
        logger.warning("[VIDEO-PRODUCER] Could not read the picture %s to crop it", ref, exc_info=True)
        copied = None
    if copied is None:
        shutil.rmtree(folder, True)
        return None, None
    return copied, folder


def cropped_name(name: str, box: Box) -> str:
    stem = Path(str(name or "")).stem or "image"
    return f"{stem}-crop-{box[2]}x{box[3]}{SUFFIX}"


def crop_file(source: Path, box: Box, folder: Path, name: str) -> Path:
    left, top, width, height = box
    with Image.open(source) as image:
        kept = image.crop((left, top, left + width, top + height))
        if kept.mode not in ("1", "L", "LA", "I", "I;16", "P", "RGB", "RGBA"):
            kept = kept.convert("RGB")
        output = folder / cropped_name(name, box)
        kept.save(output, format="PNG")
    return output


def _made(owner: User, source: str, picture: Path, ratio: Ratio, name: str) -> Optional[Tuple[Path, Path, Box]]:
    size = picture_size(picture)
    box = crop_box(size[0], size[1], ratio) if size is not None else None
    if box is None:
        return None
    try:
        folder = workdir.scratch(workdir.FRAMES, owner.id)
    except workdir.WorkdirError as exc:
        raise CropError(exc.message) from None
    try:
        return crop_file(picture, box, folder, name), folder, box
    except Exception:
        shutil.rmtree(folder, True)
        logger.warning("[VIDEO-PRODUCER] Could not crop the picture %s", source, exc_info=True)
        raise CropError("The start or end image could not be cropped to the frame of the take.") from None


async def fit(owner: User, source: str, ref: Mapping[str, Any], ratio: Ratio, name: str) -> Optional[Fitted]:
    known = await asyncio.to_thread(_known, owner, source, ratio)
    if known is not None:
        return known
    picture, copied = await asyncio.to_thread(_picture, owner, ref)
    if picture is None:
        return None
    folder: Optional[Path] = None
    try:
        made = await asyncio.to_thread(_made, owner, source, picture, ratio, name)
        if made is None:
            return None
        output, folder, box = made
        try:
            stored = await core_bridge.save_user_media(owner, output)
        except Exception as exc:
            logger.warning("[VIDEO-PRODUCER] Could not store the cropped picture of %s", source, exc_info=True)
            raise CropError(str(exc) or "The cropped start or end image could not be stored.") from None
    finally:
        for leftover in (folder, copied):
            if leftover is not None:
                await asyncio.to_thread(shutil.rmtree, leftover, True)
    fitted = Fitted(source=source, value=str(stored["value"]), item_id=(stored.get("ref") or {}).get("item_id"), width=box[2], height=box[3])
    await asyncio.to_thread(record, owner.id, ratio, fitted)
    return fitted
