from src.features.chat.modes import ChatModeRegistry, build_generation_mode


def mode_registry() -> ChatModeRegistry:
    registry = ChatModeRegistry()
    registry.register(build_generation_mode())
    return registry
