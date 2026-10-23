from typing import List

from .media_field import MediaField
from .specs import FieldConfigSpec, FieldExampleSpec


class Image(MediaField):
    """Image field for handling image uploads and processing"""
    KIND = 'image'
    SUPPORTED_FORMATS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'})
    MIME_TYPES = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
        '.webp': 'image/webp', '.gif': 'image/gif', '.bmp': 'image/bmp',
    }
    DEFAULT_MAX_SIZE = 10 * 1024 * 1024
    DEFAULT_MIME = 'image/png'
    ACCEPT_FALLBACK = 'image/*'
    HAS_DIMENSIONS = True
    FORMATS_EXAMPLE = ['.jpg', '.png', '.webp']
    VALIDATION_EXAMPLE = {"max_size": 10485760, "max_width": 1920, "max_height": 1080}
    EXTRA_CONFIG = (
        FieldConfigSpec(
            name="allow_inpaint",
            param_type=bool,
            default=False,
            description="Show the inpaint mask button on this image field",
            example=True
        ),
        FieldConfigSpec(
            name="max_resolution",
            param_type=int,
            default=None,
            description="Maximum width AND height, in pixels, for any uploaded image",
            example=2048
        ),
    )

    @classmethod
    def examples(cls) -> List[FieldExampleSpec]:
        """Return example configurations for this field"""
        return [
            FieldExampleSpec(
                title="Basic Image Upload",
                description="Single image upload field",
                yaml_config="""type: image
name: input_image
label: Upload Image
validation:
  max_size: 5242880  # 5MB
  formats: [".jpg", ".png"]""",
                rendered_output={
                    "type": "image",
                    "name": "input_image",
                    "title": "Upload Image",
                    "accept": "image/jpeg,image/png",
                    "multiple": False,
                    "validation": {
                        "maxSize": 5242880,
                        "formats": [".jpg", ".png"]
                    }
                },
                frontend_preview={
                    "type": "image",
                    "name": "preview_input_image",
                    "title": "Upload Image",
                    "accept": "image/jpeg,image/png",
                    "multiple": False
                }
            ),
            FieldExampleSpec(
                title="Multiple Image Upload",
                description="Allow uploading multiple images",
                yaml_config="""type: image
name: reference_images
label: Reference Images
configuration:
  multi: true
  formats: [".jpg", ".png", ".webp"]
validation:
  max_size: 10485760  # 10MB
  max_width: 2048
  max_height: 2048""",
                rendered_output={
                    "type": "image",
                    "name": "reference_images",
                    "title": "Reference Images",
                    "accept": "image/jpeg,image/png,image/webp",
                    "multiple": True,
                    "validation": {
                        "maxSize": 10485760,
                        "maxWidth": 2048,
                        "maxHeight": 2048,
                        "formats": [".jpg", ".png", ".webp"]
                    }
                },
                frontend_preview={
                    "type": "image",
                    "name": "preview_reference_images",
                    "title": "Reference Images",
                    "accept": "image/jpeg,image/png,image/webp",
                    "multiple": True
                }
            ),
            FieldExampleSpec(
                title="Profile Picture Upload",
                description="Small square image for profiles",
                yaml_config="""type: image
name: avatar
label: Profile Picture
validation:
  max_size: 1048576  # 1MB
  max_width: 512
  max_height: 512
  formats: [".jpg", ".png"]""",
                rendered_output={
                    "type": "image",
                    "name": "avatar",
                    "title": "Profile Picture",
                    "accept": "image/jpeg,image/png",
                    "multiple": False,
                    "validation": {
                        "maxSize": 1048576,
                        "maxWidth": 512,
                        "maxHeight": 512,
                        "formats": [".jpg", ".png"]
                    }
                },
                frontend_preview={
                    "type": "image",
                    "name": "preview_avatar",
                    "title": "Profile Picture",
                    "accept": "image/jpeg,image/png",
                    "multiple": False
                }
            ),
        ]