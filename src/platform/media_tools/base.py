from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.platform.media_tools.types import FilterFragment, ToolContext, ToolMedia

if TYPE_CHECKING:
    from PIL import Image


class MediaTool:
    def run(self, inputs: list[ToolMedia], params: dict, ctx: ToolContext) -> list[ToolMedia]:
        from src.platform.media_tools.fastpath import run_single

        return [run_single(self, media, params, ctx) for media in inputs]

    def video_filter(self, media: ToolMedia, params: dict) -> Optional[FilterFragment]:
        return None

    def image(self, image: "Image.Image", params: dict) -> "Image.Image":
        raise NotImplementedError

    def check_params(self, params: dict) -> None:
        return None

    def has_image_path(self) -> bool:
        return type(self).image is not MediaTool.image
