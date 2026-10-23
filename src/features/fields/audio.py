from typing import List

from .media_field import MediaField
from .specs import FieldConfigSpec, FieldExampleSpec


class Audio(MediaField):
    """Audio field for handling audio uploads and processing"""
    KIND = 'audio'
    SUPPORTED_FORMATS = frozenset({'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'})
    MIME_TYPES = {
        '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac',
        '.ogg': 'audio/ogg', '.m4a': 'audio/mp4', '.aac': 'audio/aac',
    }
    DEFAULT_MAX_SIZE = 50 * 1024 * 1024
    DEFAULT_MIME = 'audio/mpeg'
    ACCEPT_FALLBACK = 'audio/*'
    HAS_DURATION = True
    FORMATS_EXAMPLE = ['.mp3', '.wav', '.flac']
    VALIDATION_EXAMPLE = {"max_size": 52428800, "max_duration": 300}
    EXTRA_CONFIG = (
        FieldConfigSpec(
            name="max_audio_duration_seconds",
            param_type=float,
            default=None,
            description="Maximum duration, in seconds, of a single uploaded audio file",
            example=30
        ),
        FieldConfigSpec(
            name="max_total_audio_duration_seconds",
            param_type=float,
            default=None,
            description="Maximum combined duration, in seconds, of every audio file in a `multi` field",
            example=60
        ),
    )

    @classmethod
    def examples(cls) -> List[FieldExampleSpec]:
        """Return example configurations for this field"""
        return [
            FieldExampleSpec(
                title="Basic Audio Upload",
                description="Single audio upload field",
                yaml_config="""type: audio
name: input_audio
label: Upload Audio
validation:
  max_size: 26214400  # 25MB
  formats: [".mp3", ".wav"]""",
                rendered_output={
                    "type": "audio",
                    "name": "input_audio",
                    "title": "Upload Audio",
                    "accept": "audio/mpeg,audio/wav",
                    "multiple": False,
                    "validation": {
                        "maxSize": 26214400,
                        "formats": [".mp3", ".wav"]
                    }
                },
                frontend_preview={
                    "type": "audio",
                    "name": "preview_input_audio",
                    "title": "Upload Audio",
                    "accept": "audio/mpeg,audio/wav",
                    "multiple": False
                }
            ),
            FieldExampleSpec(
                title="Multiple Audio Upload",
                description="Allow uploading multiple audio files",
                yaml_config="""type: audio
name: reference_audios
label: Reference Audios
configuration:
  multi: true
  formats: [".mp3", ".wav", ".flac"]
validation:
  max_size: 52428800  # 50MB
  max_duration: 600  # 10 minutes""",
                rendered_output={
                    "type": "audio",
                    "name": "reference_audios",
                    "title": "Reference Audios",
                    "accept": "audio/mpeg,audio/wav,audio/flac",
                    "multiple": True,
                    "validation": {
                        "maxSize": 52428800,
                        "maxDuration": 600,
                        "formats": [".mp3", ".wav", ".flac"]
                    }
                },
                frontend_preview={
                    "type": "audio",
                    "name": "preview_reference_audios",
                    "title": "Reference Audios",
                    "accept": "audio/mpeg,audio/wav,audio/flac",
                    "multiple": True
                }
            ),
            FieldExampleSpec(
                title="Short Audio Clip",
                description="Upload short audio clips for processing",
                yaml_config="""type: audio
name: audio_clip
label: Audio Clip
validation:
  max_size: 10485760  # 10MB
  max_duration: 30  # 30 seconds
  formats: [".mp3"]""",
                rendered_output={
                    "type": "audio",
                    "name": "audio_clip",
                    "title": "Audio Clip",
                    "accept": "audio/mpeg",
                    "multiple": False,
                    "validation": {
                        "maxSize": 10485760,
                        "maxDuration": 30,
                        "formats": [".mp3"]
                    }
                },
                frontend_preview={
                    "type": "audio",
                    "name": "preview_audio_clip",
                    "title": "Audio Clip",
                    "accept": "audio/mpeg",
                    "multiple": False
                }
            ),
        ]
