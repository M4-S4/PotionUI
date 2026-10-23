from typing import List

from .media_field import MediaField
from .specs import FieldConfigSpec, FieldExampleSpec


class Video(MediaField):
    """Video field for handling video uploads and processing"""
    KIND = 'video'
    SUPPORTED_FORMATS = frozenset({'.mp4', '.webm', '.avi', '.mov', '.mkv'})
    MIME_TYPES = {
        '.mp4': 'video/mp4', '.webm': 'video/webm', '.avi': 'video/avi',
        '.mov': 'video/quicktime', '.mkv': 'video/x-matroska',
    }
    DEFAULT_MAX_SIZE = 100 * 1024 * 1024
    DEFAULT_MIME = 'video/mp4'
    ACCEPT_FALLBACK = 'video/*'
    HAS_DIMENSIONS = True
    HAS_DURATION = True
    FORMATS_EXAMPLE = ['.mp4', '.webm', '.mov']
    VALIDATION_EXAMPLE = {"max_size": 104857600, "max_duration": 300, "max_width": 1920, "max_height": 1080}
    EXTRA_CONFIG = (
        FieldConfigSpec(
            name="max_resolution",
            param_type=int,
            default=None,
            description="Maximum width AND height, in pixels, for any uploaded video",
            example=2048
        ),
        FieldConfigSpec(
            name="max_video_duration_seconds",
            param_type=float,
            default=None,
            description="Maximum duration, in seconds, of a single uploaded video",
            example=5
        ),
        FieldConfigSpec(
            name="max_total_video_duration_seconds",
            param_type=float,
            default=None,
            description="Maximum combined duration, in seconds, of every video in a `multi` field",
            example=12
        ),
    )

    @classmethod
    def examples(cls) -> List[FieldExampleSpec]:
        """Return example configurations for this field"""
        return [
            FieldExampleSpec(
                title="Basic Video Upload",
                description="Single video upload field",
                yaml_config="""type: video
name: input_video
label: Upload Video
validation:
  max_size: 52428800  # 50MB
  formats: [".mp4", ".webm"]""",
                rendered_output={
                    "type": "video",
                    "name": "input_video",
                    "title": "Upload Video",
                    "accept": "video/mp4,video/webm",
                    "multiple": False,
                    "validation": {
                        "maxSize": 52428800,
                        "formats": [".mp4", ".webm"]
                    }
                },
                frontend_preview={
                    "type": "video",
                    "name": "preview_input_video",
                    "title": "Upload Video",
                    "accept": "video/mp4,video/webm",
                    "multiple": False
                }
            ),
            FieldExampleSpec(
                title="Multiple Video Upload",
                description="Allow uploading multiple videos",
                yaml_config="""type: video
name: reference_videos
label: Reference Videos
configuration:
  multi: true
  formats: [".mp4", ".webm", ".mov"]
validation:
  max_size: 104857600  # 100MB
  max_duration: 600  # 10 minutes
  max_width: 1920
  max_height: 1080""",
                rendered_output={
                    "type": "video",
                    "name": "reference_videos",
                    "title": "Reference Videos",
                    "accept": "video/mp4,video/webm,video/quicktime",
                    "multiple": True,
                    "validation": {
                        "maxSize": 104857600,
                        "maxDuration": 600,
                        "maxWidth": 1920,
                        "maxHeight": 1080,
                        "formats": [".mp4", ".webm", ".mov"]
                    }
                },
                frontend_preview={
                    "type": "video",
                    "name": "preview_reference_videos",
                    "title": "Reference Videos",
                    "accept": "video/mp4,video/webm,video/quicktime",
                    "multiple": True
                }
            ),
            FieldExampleSpec(
                title="Short Video Clip",
                description="Upload short video clips for processing",
                yaml_config="""type: video
name: video_clip
label: Video Clip
validation:
  max_size: 20971520  # 20MB
  max_duration: 30  # 30 seconds
  max_width: 1280
  max_height: 720
  formats: [".mp4"]""",
                rendered_output={
                    "type": "video",
                    "name": "video_clip",
                    "title": "Video Clip",
                    "accept": "video/mp4",
                    "multiple": False,
                    "validation": {
                        "maxSize": 20971520,
                        "maxDuration": 30,
                        "maxWidth": 1280,
                        "maxHeight": 720,
                        "formats": [".mp4"]
                    }
                },
                frontend_preview={
                    "type": "video",
                    "name": "preview_video_clip",
                    "title": "Video Clip",
                    "accept": "video/mp4",
                    "multiple": False
                }
            ),
        ]