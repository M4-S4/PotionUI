from typing import List

from .audio import Audio
from .image import Image
from .media_field import MediaField
from .specs import FieldConfigSpec, FieldExampleSpec
from .video import Video


class Media(MediaField):
    """Generic media field for a single upload slot that can hold more than
    one kind of item - "images or videos", "any of the three" - unlike
    `Image`/`Video`/`Audio`, which are each locked to one category by
    construction. Configured with `accepted_types` (which categories an item
    may be) plus the shared `max_items`/`max_resolution`/duration limits -
    see `media_input._check_media_constraints`."""
    KIND = 'media'
    SUPPORTED_FORMATS = Image.SUPPORTED_FORMATS | Video.SUPPORTED_FORMATS | Audio.SUPPORTED_FORMATS
    MIME_TYPES = {**Image.MIME_TYPES, **Video.MIME_TYPES, **Audio.MIME_TYPES}
    DEFAULT_MAX_SIZE = Video.DEFAULT_MAX_SIZE
    HAS_DIMENSIONS = True
    CHECK_LEGACY_LIMITS = False

    MAX_ITEMS_EXAMPLE = 3
    FORMATS_EXAMPLE = ['.jpg', '.png', '.mp4']
    VALIDATION_EXAMPLE = {"max_size": 104857600}
    EXTRA_CONFIG = (
        FieldConfigSpec(
            name="max_items_by_kind",
            param_type=dict,
            default=None,
            description="Per-category item caps when `multi` is enabled, as a map of "
                         "'image'/'video'/'audio' to a positive integer; a category left out has no "
                         "cap of its own, and `max_items` still caps the total",
            example={"image": 9, "video": 3, "audio": 3}
        ),
        FieldConfigSpec(
            name="accepted_types",
            param_type=list,
            default=None,
            description="Media categories this field accepts (any of 'image', 'video', 'audio'); "
                         "unset means no restriction",
            example=["image", "video"]
        ),
        FieldConfigSpec(
            name="max_resolution",
            param_type=int,
            default=None,
            description="Maximum width AND height, in pixels, for any image or video item",
            example=2048
        ),
        FieldConfigSpec(
            name="max_video_duration_seconds",
            param_type=float,
            default=None,
            description="Maximum duration, in seconds, of a single video item",
            example=5
        ),
        FieldConfigSpec(
            name="max_total_video_duration_seconds",
            param_type=float,
            default=None,
            description="Maximum combined duration, in seconds, of every video item in a `multi` field",
            example=12
        ),
        FieldConfigSpec(
            name="max_audio_duration_seconds",
            param_type=float,
            default=None,
            description="Maximum duration, in seconds, of a single audio item",
            example=30
        ),
        FieldConfigSpec(
            name="max_total_audio_duration_seconds",
            param_type=float,
            default=None,
            description="Maximum combined duration, in seconds, of every audio item in a `multi` field",
            example=60
        ),
    )

    @classmethod
    def examples(cls) -> List[FieldExampleSpec]:
        return [
            FieldExampleSpec(
                title="Up to 3 references, images or video",
                description="A multi-item field accepting either images or videos, capped in count, "
                             "resolution and total video duration",
                yaml_config="""type: media
name: reference_media
label: Reference Media
configuration:
  multi: true
  max_items: 3
  accepted_types: [image, video]
  max_resolution: 2048
  max_video_duration_seconds: 5
  max_total_video_duration_seconds: 12""",
                rendered_output={
                    "type": "media",
                    "name": "reference_media",
                    "title": "Reference Media",
                    "multiple": True,
                    "max_items": 3,
                    "accepted_types": ["image", "video"],
                    "max_resolution": 2048,
                    "max_video_duration_seconds": 5,
                    "max_total_video_duration_seconds": 12
                },
                frontend_preview={
                    "type": "media",
                    "name": "preview_reference_media",
                    "title": "Reference Media",
                    "multiple": True
                }
            ),
        ]
