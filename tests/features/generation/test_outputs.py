import io

from PIL import Image
from dataclasses import fields, MISSING

from src.pipelines.outputs import ImageGenerationOutput


class TestImageGenerationOutput:
    def test_lazy_file_backed_image_is_eagerly_decoded(self):
        """
        A freshly Image.open()-ed image is lazy (only the header is read; .fp
        stays set until .load() runs). Pipes run on a worker thread while the
        output handlers that touch .image run on the event-loop thread, so a
        lazy image would be decoded by two threads at once. Construction must
        force the decode so no lazy state escapes the producing thread.
        """
        buf = io.BytesIO()
        Image.new('RGB', (64, 64), color='green').save(buf, format='PNG')
        buf.seek(0)
        lazy_image = Image.open(buf)
        assert lazy_image.fp is not None  # sanity: still lazy before construction

        output = ImageGenerationOutput(image=lazy_image)

        assert output.image.fp is None
        assert output.image.im is not None

    def test_in_memory_image_passes_through_untouched(self):
        """An already-decoded, non-file-backed image (Image.new) must not raise
        and must be left as-is (no fp to worry about)."""
        in_memory_image = Image.new('RGB', (32, 32), color='blue')

        output = ImageGenerationOutput(image=in_memory_image)

        assert output.image is in_memory_image
        assert getattr(output.image, 'fp', None) is None

    def test_none_image_does_not_raise(self):
        """image=None must be a no-op, not an error."""
        output = ImageGenerationOutput(image=None)

        assert output.image is None


class TestImageOutputFields:
    def test_image_is_a_required_field(self):
        image_fields = fields(ImageGenerationOutput)
        image_field_names = [f.name for f in image_fields]
        assert 'image' in image_field_names

        image_field = next(f for f in image_fields if f.name == 'image')
        assert image_field.default == MISSING
        assert image_field.default_factory == MISSING

