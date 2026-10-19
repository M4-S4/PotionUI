import logging
import os
from typing import Any, Dict, Optional

from src.pipelines.outputs import SplatGenerationOutput
from src.features.generation.handlers.base_handler import BaseGenerationOutputHandler
from src.features.generation.handlers.image_handler import generate_thumbnails
from src.features.generation.output_types import OutputTypeSpec, SerializeContext, output_type_registry
from src.features.generation.records import File
from src.features.generation.repository import generation_repo
from src.features.generation.temp_source_tracker import temp_source_tracker
from src.features.generation.thumbnail_profile import profile_hash
from src.platform.filesystem.splat_formats import InvalidSplatError, splat_format_registry

logger = logging.getLogger(__name__)

SPLAT_FILE_TYPE = 'SPLAT'


class SplatGenerationOutputHandler(BaseGenerationOutputHandler):

    def can_handle(self, output) -> bool:
        return isinstance(output, SplatGenerationOutput)

    def handle(self, output: SplatGenerationOutput) -> Dict[str, Any]:
        metadata = {
            'handler': 'SplatGenerationOutputHandler',
            'processed': True,
            'temporary': output.temporary,
            'saved_path': None,
            'file_record': None,
        }

        try:
            if not output.splat_path:
                return metadata

            try:
                output.splat_count = self._probe_source(output)
            except InvalidSplatError as e:
                logger.error(f"Refusing to store invalid splat: {e}")
                metadata['processed'] = False
                metadata['save_error'] = f"Invalid splat file: {e}"
                return metadata

            saved_path = self._save_splat_file(output)
            if saved_path:
                metadata['saved_path'] = saved_path
                output._saved_path = saved_path

                if not output.temporary:
                    thumbnail_paths = self._save_thumbnails(output, saved_path)
                    file_record = self._create_file_record(output, saved_path, thumbnail_paths)
                    if file_record:
                        metadata['file_record'] = {
                            'id': file_record.id,
                            'filename': os.path.basename(file_record.file_path),
                            'file_path': file_record.file_path,
                            'file_size': file_record.file_size,
                        }
            elif not output.temporary:
                logger.warning("Failed to save splat")
                metadata['processed'] = False
                metadata['save_error'] = "Failed to save splat"

        except Exception as e:
            logger.error(f"Error handling SplatGenerationOutput: {str(e)}")
            metadata['processed'] = False
            metadata['error'] = str(e)

        return metadata

    def _probe_source(self, output: SplatGenerationOutput) -> Optional[int]:
        extension = os.path.splitext(str(output.splat_path))[1]
        splat_format = splat_format_registry.get(extension)
        if splat_format is None:
            raise InvalidSplatError(
                f"no splat format registered for extension {extension!r}: {output.splat_path}"
            )
        if not splat_format.generated:
            raise InvalidSplatError(
                f"{extension} splats are accepted only as user imports, not as pipe output: {output.splat_path}"
            )
        probed_count = splat_format.probe(str(output.splat_path))
        return output.splat_count if output.splat_count is not None else probed_count

    def _save_splat_file(self, output: SplatGenerationOutput) -> Optional[str]:
        try:
            from src.platform.filesystem.file_store import FileStore

            file_service = FileStore(self._resolve_storage_dir(), storage_driver=self._resolve_storage_driver())
            extension = os.path.splitext(str(output.splat_path))[1].lstrip('.').lower()

            if output.temporary:
                storage_type = 'tmp'
                prefix = f"tmp_splat_{self.generation_id}" if self.generation_id else "tmp_splat"
            else:
                storage_type = 'generations'
                self.image_counter += 1
                prefix = str(self.image_counter)

            full_path, file_metadata = file_service.save_file_from_path(
                generation_id=self.generation_id if not output.temporary else None,
                source_path=str(output.splat_path),
                extension=extension,
                prefix=prefix,
                storage_type=storage_type,
                is_temporary=output.temporary,
            )

            if full_path and file_metadata:
                temp_source_tracker.register(self.generation_id, str(output.splat_path))
                return file_metadata['file_path']

            logger.error("Failed to save splat file")
            return None

        except Exception as e:
            logger.error(f"Error saving splat file: {str(e)}")
            return None

    def _save_thumbnails(self, output: SplatGenerationOutput, saved_path: str) -> Dict[str, str]:
        if output.preview is None:
            return {}
        try:
            base_key = saved_path.rsplit('/', 1)[0]
            return generate_thumbnails(
                output.preview, self._resolve_storage_driver(), base_key, self.image_counter, self.thumbnail_profile
            )
        except Exception as e:
            logger.error(f"Error writing splat thumbnails: {str(e)}")
            return {}

    def _create_file_record(
        self, output: SplatGenerationOutput, saved_path: str, thumbnail_paths: Dict[str, str]
    ) -> Optional[File]:
        try:
            file_size = self._resolve_storage_driver().size(saved_path) or 0
            file_record = File(
                file_path=saved_path,
                file_type=SPLAT_FILE_TYPE,
                user_id=self.user_id,
                file_size=file_size,
                pipe_name=getattr(output, 'pipe_name', None),
                is_final=not output.temporary,
                is_derived=bool(getattr(output, 'derived', False)),
                thumbnail_small=thumbnail_paths.get('small'),
                thumbnail_medium=thumbnail_paths.get('medium'),
                thumbnail_large=thumbnail_paths.get('large'),
                thumbnail_profile=profile_hash(self.thumbnail_profile) if thumbnail_paths else None,
            )
            return generation_repo.add_file(self.generation_id, file_record)

        except Exception as e:
            logger.error(f"Error creating splat file record: {str(e)}")
            return None


def splat_api_path(output: SplatGenerationOutput, generation_id: str) -> Optional[str]:
    saved_path = getattr(output, '_saved_path', None)
    if not saved_path:
        return None
    filename = os.path.basename(str(saved_path))
    if getattr(output, 'temporary', True):
        return f"/api/media/tmp/{filename}"
    return f"/api/media/generations/{generation_id}/{filename}"


def splat_format_of(output: SplatGenerationOutput) -> Optional[str]:
    source = getattr(output, '_saved_path', None) or getattr(output, 'splat_path', None)
    if not source:
        return None
    extension = os.path.splitext(str(source))[1]
    return extension.lstrip('.').lower() or None


def serialize_splat_output(output: SplatGenerationOutput, ctx: SerializeContext) -> Dict[str, Any]:
    result = {
        'file_type': 'splat',
        'splat_format': splat_format_of(output),
        'temporary': getattr(output, 'temporary', True),
        'derived': bool(getattr(output, 'derived', False)),
        'seed': getattr(output, 'seed', None),
        'splat_count': getattr(output, 'splat_count', None),
    }

    try:
        path = splat_api_path(output, ctx.generation_id)
        if path:
            result['path'] = path
            result['splat_name'] = os.path.basename(path)
    except Exception as e:
        logger.error(f"Failed to serialize splat path: {str(e)}")

    return result


output_type_registry.register(OutputTypeSpec(
    output_cls=SplatGenerationOutput,
    key='splat',
    message_type='workbench_update',
    serializer=serialize_splat_output,
    handler_cls=SplatGenerationOutputHandler,
))
