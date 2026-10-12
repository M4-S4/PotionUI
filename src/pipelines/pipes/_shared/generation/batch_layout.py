from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class PromptBatchLayout:
    per_prompt: int
    count: int

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> Optional['PromptBatchLayout']:
        marker = (config or {}).get('prompt_batch')
        if not isinstance(marker, Mapping):
            return None
        try:
            per_prompt, count = int(marker['per_prompt']), int(marker['count'])
        except (KeyError, TypeError, ValueError):
            return None
        if per_prompt < 1 or count < 2:
            return None
        return cls(per_prompt=per_prompt, count=count)

    def status(self, image_index: int, total: int) -> str:
        prompt = min(image_index // self.per_prompt, self.count - 1) + 1
        return f"Generating image {image_index + 1} of {total} · prompt {prompt} of {self.count}"

    def reading(self) -> str:
        return f"Reading {self.count} prompts"
