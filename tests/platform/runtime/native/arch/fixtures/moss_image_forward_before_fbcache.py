from __future__ import annotations

import torch

from src.platform.runtime.native.arch.moss_image.layers import rope_table
from src.platform.runtime.native.arch.moss_image.model import _sinusoid


def legacy_forward(self, x, timestep, context, attention_mask=None, skip_layers=None):
    cfg = self.cfg
    skip = self._skip_set(skip_layers)
    batch, _, height, width = x.shape
    mask = attention_mask
    if mask is None:
        mask = torch.ones(context.shape[:2], dtype=torch.bool, device=x.device)
    dtype = x.dtype
    context = context.to(device=x.device, dtype=dtype)

    embedding = _sinusoid(timestep.to(x.device), batch, cfg.freq_dim)
    time = self.time_embedding(embedding.to(dtype)).float()
    mod = self.shared_block_modulation(time.to(dtype)).view(batch, 6, -1)
    head = self.video_head_modulation(time.to(dtype)).view(batch, 2, -1)

    outputs = []
    for i in range(batch):
        text_in = context[i : i + 1, mask[i].bool().to(x.device)]
        video = self.video_embedder(x[i : i + 1].flatten(2).transpose(1, 2).contiguous())
        rope = rope_table(height, width, text_in.shape[1], self._bands, x.device).to(video.dtype)
        text = self.text_embedder(text_in, rope[:, height * width :])
        stream = torch.cat((video, text), 1)
        for index, layer in enumerate(self.layers):
            if index not in skip:
                stream = layer(stream, rope, mod[i : i + 1])
        out = self.final_norm_video(stream[:, : height * width])
        out = out * (1 + head[i : i + 1, 1, None]) + head[i : i + 1, 0, None]
        out = self.final_linear_video(out)
        outputs.append(out.transpose(1, 2).reshape(1, -1, height, width))
    return torch.cat(outputs)
