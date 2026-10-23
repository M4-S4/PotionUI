import io

import pytest
from PIL import Image


@pytest.fixture
def fake_image() -> Image.Image:
    img = Image.new("RGB", (512, 512))
    pixels = img.load()
    for y in range(512):
        for x in range(512):
            pixels[x, y] = (int((x / 512) * 255), int((y / 512) * 255), 128)
    return img


@pytest.fixture
def fake_image_bytes(fake_image) -> bytes:
    buffer = io.BytesIO()
    fake_image.save(buffer, format="PNG")
    return buffer.getvalue()
