from __future__ import annotations

import shutil
from typing import Optional

from PIL import Image

from src.platform.media_tools import (
    FilterFragment,
    InvalidToolParams,
    MediaTool,
    ToolMedia,
    ToolSpec,
)

BOX_PARAMS = [
    {"name": "x", "type": "integer", "default": 0, "min": 0},
    {"name": "y", "type": "integer", "default": 0, "min": 0},
    {"name": "width", "type": "integer", "min": 1},
    {"name": "height", "type": "integer", "min": 1},
]


def make_spec(tool_id: str, impl: Optional[MediaTool], **overrides) -> ToolSpec:
    values = dict(
        id=tool_id,
        label=tool_id.title(),
        accepts=("image", "video", "audio"),
        produces="same",
        cardinality="one_to_one",
        params=[],
        impl=impl,
    )
    values.update(overrides)
    return ToolSpec(**values)


class BoxCrop(MediaTool):
    def video_filter(self, media, params):
        self._check(media.width, media.height, params)
        return FilterFragment(
            vf=[f"crop={params['width']}:{params['height']}:{params['x']}:{params['y']}"],
            width=params["width"],
            height=params["height"],
        )

    def image(self, image, params):
        self._check(image.width, image.height, params)
        return image.crop((params["x"], params["y"], params["x"] + params["width"], params["y"] + params["height"]))

    @staticmethod
    def _check(width, height, params):
        if params["x"] + params["width"] > width:
            raise InvalidToolParams("width", f"The box does not fit inside {width}x{height}")
        if params["y"] + params["height"] > height:
            raise InvalidToolParams("height", f"The box does not fit inside {width}x{height}")


class Mirror(MediaTool):
    def video_filter(self, media, params):
        return FilterFragment(vf=["hflip"])

    def image(self, image, params):
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


class Window(MediaTool):
    def video_filter(self, media, params):
        return FilterFragment(start=params["start"], end=params["end"], duration=params["end"] - params["start"])


class Faster(MediaTool):
    def video_filter(self, media, params):
        factor = params["factor"]
        return FilterFragment(
            vf=[f"setpts=PTS/{factor}"],
            af=[f"atempo={factor}"],
            duration=(media.duration or 0) / factor,
        )


class Mute(MediaTool):
    def video_filter(self, media, params):
        return FilterFragment(drop_audio=True)


class Quality(MediaTool):
    def video_filter(self, media, params):
        return FilterFragment(crf=params["crf"])


class Tint(MediaTool):
    def image(self, image, params):
        rgba = image.convert("RGBA")
        rgba.putalpha(128)
        return rgba


class CopyOnly(MediaTool):
    def run(self, inputs, params, ctx):
        outputs = []
        for media in inputs:
            target = ctx.new_path(media.path.suffix, stem="copy")
            shutil.copyfile(media.path, target)
            outputs.append(ctx.probe(target, media.kind))
        return outputs


class FrameGrab(MediaTool):
    def run(self, inputs, params, ctx):
        target = ctx.new_path(".png", stem="frame")
        ctx.ffmpeg(["-i", str(inputs[0].path), "-frames:v", "1", str(target)])
        return [ctx.probe(target, "image")]

    def video_filter(self, media, params):
        return FilterFragment(vf=["hflip"])


class Stack(MediaTool):
    def run(self, inputs, params, ctx):
        target = ctx.new_path(inputs[0].path.suffix, stem="stack")
        shutil.copyfile(inputs[0].path, target)
        return [ctx.probe(target, inputs[0].kind)]


class Fan(MediaTool):
    def run(self, inputs, params, ctx):
        outputs = []
        for index in range(params["parts"]):
            target = ctx.new_path(inputs[0].path.suffix, stem=f"part{index}")
            shutil.copyfile(inputs[0].path, target)
            outputs.append(ctx.probe(target, inputs[0].kind))
        return outputs


class Broken(MediaTool):
    def run(self, inputs, params, ctx):
        return []


class Recording(MediaTool):
    def __init__(self):
        self.seen: list[ToolMedia] = []

    def video_filter(self, media, params):
        self.seen.append(media)
        return FilterFragment(vf=["hflip"])

    def image(self, image, params):
        return image
