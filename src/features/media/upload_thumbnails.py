import logging
import threading
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Set

from PIL import Image

from src.features.generation.handlers.image_handler import generate_thumbnails
from src.features.generation.handlers.video_handler import generate_video_thumbnails
from src.features.generation.thumbnail_profile import SIZE_ORDER, ThumbnailProfile, profile_hash
from src.platform.filesystem.storage_driver import FileStorageDriver, local_copy

logger = logging.getLogger(__name__)

UPLOADS_BASE_KEY = "uploads"


def render_image_thumbnails(
    storage_driver: FileStorageDriver, profile: ThumbnailProfile, image: Image.Image, stem: str
) -> Dict[str, str]:
    return generate_thumbnails(image, storage_driver, UPLOADS_BASE_KEY, stem, profile)


def render_video_thumbnails(
    storage_driver: FileStorageDriver,
    upload_repo,
    profile: ThumbnailProfile,
    key: str,
    suffix: str,
    upload_id: str,
    stem: str,
) -> None:
    try:
        with local_copy(storage_driver, key, suffix) as local_path:
            thumbnail_paths = generate_video_thumbnails(
                str(local_path), storage_driver, UPLOADS_BASE_KEY, stem, profile
            )

        if thumbnail_paths:
            upload_repo.set_thumbnail_paths(
                upload_id,
                thumbnail_paths.get("small"),
                thumbnail_paths.get("medium"),
                thumbnail_paths.get("large"),
                profile_hash(profile),
            )
        else:
            logger.warning(f"Failed to generate thumbnails for uploaded video: {key}")
    except Exception as e:
        logger.error(f"Error generating thumbnails for uploaded video {key}: {e}")


def run_in_background(worker: Callable[[], None]) -> None:
    threading.Thread(target=worker, daemon=True).start()


def thumbnail_keys(record, is_video: bool) -> Set[str]:
    keys: Set[str] = set()
    for size in SIZE_ORDER:
        relative = getattr(record, f"thumbnail_{size}", None)
        if not relative:
            continue
        keys.add(f"{UPLOADS_BASE_KEY}/{relative}")
        if is_video:
            sibling = PurePosixPath(relative)
            keys.add(f"{UPLOADS_BASE_KEY}/{sibling.parent}/{sibling.stem}_animated.webp")
    return keys


def paths_to_keys(paths: Dict[str, str], is_video: bool) -> Set[str]:
    keys: Set[str] = set()
    for relative in paths.values():
        keys.add(f"{UPLOADS_BASE_KEY}/{relative}")
        if is_video:
            sibling = PurePosixPath(relative)
            keys.add(f"{UPLOADS_BASE_KEY}/{sibling.parent}/{sibling.stem}_animated.webp")
    return keys


def delete_keys(storage_driver: FileStorageDriver, keys: Set[str]) -> None:
    for key in sorted(keys):
        try:
            storage_driver.delete(key)
        except Exception as e:
            logger.warning("Could not delete stale thumbnail %s: %s", key, e)
