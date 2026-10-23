from src.platform.plugins.hooks import hooks_registry

MEDIA_INDEX_HOOKS = hooks_registry.declare(
    "media_index", "backend",
    "after_tags_written",
    specs={
        "after_tags_written": {
            "description": (
                "Fired after Media Tagging has written the detected tags of a generation's final "
                "files, once per generation per drained batch. Auto-organize re-runs that "
                "generation's rules on it."
            ),
            "payload": {
                "generation_id": {"type": "str", "description": "Generation whose files were tagged"},
                "file_ids": {"type": "List[str]", "description": "Files of the generation tagged in this batch"},
            },
            "use_when": [
                "React to a generation's detected tags becoming available, e.g. file it by what the tagger saw",
            ],
        },
    },
)
