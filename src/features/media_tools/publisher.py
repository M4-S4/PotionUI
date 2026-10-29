import logging
import uuid
from pathlib import Path
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

from src.features.media.records import Upload
from src.features.media_tools.dto import MediaToolItem
from src.platform.database.rows import dt_iso
from src.platform.filesystem.storage_driver import StorageKeyError, uploads_key
from src.platform.http.errors import NotFoundError
from src.platform.media_tools import ToolMedia
from src.platform.plugins.limit_kinds import AdmissionRequest

if TYPE_CHECKING:
    from src.features.media.media_types import MediaTypeResolver
    from src.features.media.upload_repository import UploadRepository
    from src.features.plans.guard import LimitGuard
    from src.platform.filesystem.storage_driver import FileStorageDriver

logger = logging.getLogger(__name__)


def to_item(upload: Upload) -> MediaToolItem:
    return MediaToolItem(
        id=upload.id,
        filename=upload.filename,
        original_filename=upload.original_filename,
        media_type=upload.media_type,
        mime_type=upload.mime_type,
        url=f"/api/media/uploads/{upload.filename}",
        width=upload.width,
        height=upload.height,
        duration_seconds=upload.duration_seconds,
        fps=upload.fps,
        size=upload.file_size,
        created_at=dt_iso(upload.created_at),
    )


class MediaToolPublisher:
    def __init__(
        self,
        upload_repository: "UploadRepository",
        media_type_resolver: "MediaTypeResolver",
        storage_driver: "FileStorageDriver",
        limit_guard: Optional["LimitGuard"] = None,
    ):
        self.upload_repo = upload_repository
        self.media_types = media_type_resolver
        self.storage_driver = storage_driver
        self.limit_guard = limit_guard

    def precheck(self, user_id: str) -> None:
        if self.limit_guard is not None:
            self.limit_guard.check(AdmissionRequest(point="upload", user_id=user_id))

    def publish_new(
        self, user_id: str, outputs: Sequence[ToolMedia], names: Sequence[Optional[str]]
    ) -> list[Upload]:
        stored: list[str] = []
        created: list[Upload] = []
        try:
            for media in outputs:
                key = self._new_key(media.path.suffix)
                self.storage_driver.put_file(key, media.path)
                stored.append(key)
            sizes = tuple(self.storage_driver.size(key) or 0 for key in stored)
            self._admit(user_id, sum(sizes), sizes)
            for key, media, name, size in zip(stored, outputs, names, sizes):
                created.append(self.upload_repo.create(self._row(user_id, key, media, name, size)))
        except Exception:
            for row in created:
                self._forget_row(row, user_id)
            for key in stored:
                self._remove(key)
            raise
        return created

    def publish_replace(
        self, user_id: str, upload: Upload, source_key: str, output: ToolMedia
    ) -> Upload:
        dest_key = self._new_key(output.path.suffix)
        try:
            self.storage_driver.put_file(dest_key, output.path)
            written = self.storage_driver.size(dest_key) or 0
            self._admit(user_id, written - (upload.file_size or 0), ())
            updated = self.upload_repo.update_file(
                upload_id=upload.id,
                user_id=user_id,
                filename=Path(dest_key).name,
                mime_type=self.media_types.get_media_type(Path(dest_key).suffix),
                width=output.width,
                height=output.height,
                duration_seconds=output.duration,
                fps=output.fps,
                file_size=written,
            )
            if not updated:
                raise NotFoundError("not_found", "Media not found")
        except Exception:
            self._remove(dest_key)
            raise
        self._remove(source_key)
        logger.info("Replaced library resource %s with an edited file", upload.id)
        return updated

    def _admit(self, user_id: str, incoming: Optional[int], items: Tuple[int, ...]) -> None:
        if self.limit_guard is None:
            return
        self.limit_guard.admit(AdmissionRequest(
            point="upload", user_id=user_id, incoming_bytes=incoming, item_bytes=items,
        ))

    def _new_key(self, suffix: str) -> str:
        try:
            return uploads_key(f"{uuid.uuid4()}{suffix.lower()}")
        except StorageKeyError as exc:
            raise ValueError("Invalid output file name") from exc

    def _row(
        self, user_id: str, key: str, media: ToolMedia, name: Optional[str], size: int
    ) -> Upload:
        return Upload(
            user_id=user_id,
            filename=Path(key).name,
            original_filename=name,
            media_type=media.kind,
            mime_type=self.media_types.get_media_type(Path(key).suffix),
            width=media.width,
            height=media.height,
            duration_seconds=media.duration,
            fps=media.fps,
            file_size=size,
        )

    def _forget_row(self, row: Upload, user_id: str) -> None:
        try:
            self.upload_repo.delete(row.filename, user_id)
        except Exception as exc:
            logger.warning("Could not roll back library row %s: %s", row.id, exc)

    def _remove(self, key: str) -> None:
        try:
            self.storage_driver.delete(key)
        except Exception as exc:
            logger.warning("Could not remove %s after a media tool run: %s", key, exc)
