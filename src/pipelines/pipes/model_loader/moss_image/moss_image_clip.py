from src.pipelines.pipes.model_loader.flux3.flux3_clip import Flux3ClipTextEncoder


class MossImageClipTextEncoder(Flux3ClipTextEncoder):
    cache_tag = "moss_image_te"
