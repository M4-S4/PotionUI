import asyncio
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import bible, core_bridge, director
from .core_bridge import User
from .director import DirectorError
from .responses import fail

logger = logging.getLogger("plugins.video-producer")

MAX_REQUEST_CHARS = 1000
MAX_CURRENT_CHARS = 1500
WRITE_TIMEOUT_S = 120.0

FRAMING = (
    "You write the prompt of one shot of a video made with Video Producer, for the engine of this preset. "
    "Video Producer adds the rest of the engine's prompt itself: the labelled sections or fields its guide describes, "
    "the [Shot 1] marker, the global direction of the film, the cast definitions from the project's bible with their "
    "<Subject N> and <Picture N> tags, the dialogue and the sound. Follow the guide for its vocabulary (camera moves, "
    "framing, motion, light), not for its layout: reply with one to three plain sentences that say what the camera "
    "sees in this shot, with no section label, no [Shot] marker and no <Subject N>, <Picture N> or <Video N> tag."
)

SECTION = re.compile(r"^[ \t]*(?:detailed_description|integrated_multimodal_description)[ \t]*:[ \t]*", re.MULTILINE)
NEXT_SECTION = re.compile(r"\n[ \t]*[a-z_]+[ \t]*:")
SHOT_MARKER = re.compile(r"\[Shot 1\]\s*")
TAGGED = re.compile(r"\s*\(\s*<(?:Subject|Picture|Video|Audio) \d+>\s*\)")
VREF_TAG = re.compile(r"@V\d+")

UNAVAILABLE = {
    "available": False,
    "vision": False,
    "name": None,
    "code": "writer_unavailable",
    "message": "The prompt writer of the application cannot be reached.",
}


def status(user: User) -> Dict[str, Any]:
    try:
        found = core_bridge.prompt_writer_status(user.id)
    except Exception:
        logger.warning("[VIDEO-PRODUCER] Could not read the prompt writer status", exc_info=True)
        return dict(UNAVAILABLE)
    available = found.get("available") is True
    return {
        "available": available,
        "vision": available and found.get("vision") is True,
        "name": found.get("name") if isinstance(found.get("name"), str) else None,
        "code": None if available else (found.get("code") if isinstance(found.get("code"), str) else "no_llm"),
        "message": None if available else (found.get("message") if isinstance(found.get("message"), str) else None),
    }


def _quoted(labels: Sequence[str]) -> str:
    return ", ".join(f'"{label}"' for label in labels)


def instruction(
    request: str,
    *,
    current: str,
    seconds: float,
    characters: Sequence[str],
    place: Optional[str],
    images: int,
) -> str:
    lines = [request.strip()[:MAX_REQUEST_CHARS], "", FRAMING]
    lines.append(f"Write in English. The shot lasts {seconds:.1f} seconds.")
    if characters:
        lines.append(
            "Name each character the shot casts by its label, written exactly as given, and do not describe their looks: "
            f"their card in the bible does. The characters: {_quoted(characters)}."
        )
    if place:
        lines.append(f'The shot takes place in "{place}"; name it by that label.')
    if images == 2:
        lines.append("Image 1 is the shot's first frame and image 2 its last frame: describe the motion from one to the other, not what the images already show.")
    elif images == 1:
        lines.append("The image is the shot's first frame: describe the motion forward from it, not what it already shows.")
    text = current.strip()[:MAX_CURRENT_CHARS]
    if VREF_TAG.search(text):
        lines.append("Keep every tag written like @V1 exactly as it is: it cites a reference video of the shot.")
    if text:
        lines.append(f'The shot\'s current prompt, to rewrite as the request says: "{text}"')
    return "\n".join(lines)


def cleaned(text: Any) -> str:
    written = text.strip() if isinstance(text, str) else ""
    section = SECTION.search(written)
    if section is not None:
        rest = written[section.end():]
        following = NEXT_SECTION.search(rest)
        written = rest[: following.start()] if following is not None else rest
    marker = SHOT_MARKER.search(written)
    if marker is not None:
        written = written[marker.end():]
    written = TAGGED.sub("", written)
    return "\n".join(" ".join(line.split()) for line in written.strip().splitlines()).strip()


def _plan(owner: User, doc: Mapping[str, Any], shot_id: str, preset_id: str, model_id: Optional[str]) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    try:
        plans = director.build_plans(owner, doc, [shot_id], preset_id, "draft", model_id, overrides=True)
    except DirectorError:
        return preset_id, None, None
    plan = next((item for item in plans.shots if item.shot_id == shot_id), None)
    if plan is None:
        return preset_id, None, None
    form_data = plan.request.get("form_data") if isinstance(plan.request, Mapping) else None
    return plan.preset_id or preset_id, plan.preset_mode or None, dict(form_data) if isinstance(form_data, Mapping) else None


def _seconds(clip: Mapping[str, Any]) -> float:
    value = clip.get("dur")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else 0.0


async def write(
    user: User,
    owner: User,
    doc: Mapping[str, Any],
    clip: Mapping[str, Any],
    request: str,
    preset_id: str,
    model_id: Optional[str],
    current: str,
) -> Dict[str, Any]:
    found = status(user)
    if not found["available"]:
        fail(409, found["code"] or "no_llm", found["message"] or "No language model is available to you.")
    shot_id = str(clip.get("id"))
    preset, mode, form_data = await asyncio.to_thread(_plan, owner, doc, shot_id, preset_id, model_id)
    frames: List[Dict[str, Any]] = await asyncio.to_thread(director.key_frame_media, owner, doc, shot_id)
    media = frames if found["vision"] else []
    characters, place = bible.cast_of(doc, clip)
    text = instruction(
        request,
        current=current,
        seconds=_seconds(clip),
        characters=[card.label for card in characters],
        place=place.label if place is not None else None,
        images=len(media),
    )
    try:
        reply = await asyncio.wait_for(
            core_bridge.write_prompt_for_preset(user.id, text, preset_id=preset, mode=mode, form_data=form_data, media=media),
            WRITE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        fail(504, "write_timeout", f"The language model did not answer within {WRITE_TIMEOUT_S:.0f} seconds.")
    except core_bridge.PromptWriteRefused as exc:
        fail(422, getattr(exc, "code", None) or "write_refused", str(exc))
    except Exception:
        logger.warning("[VIDEO-PRODUCER] The prompt writer failed for shot %s", shot_id, exc_info=True)
        fail(502, "write_failed", "The language model could not write the prompt.")
    written = cleaned(reply.get("text") if isinstance(reply, Mapping) else None)
    if not written:
        fail(502, "empty_reply", "The language model returned no prompt.")
    return {
        "text": written,
        "name": found["name"],
        "images": len(media),
        "images_skipped": len(frames) - len(media),
    }
