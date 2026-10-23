import copy

import pytest

from src.features.music_director.normalize import (
    MusicDirectorValidationError,
    apply_preset_mode_overlay,
    compile_sections_to_lyrics,
    normalize_music_director,
)

_LIMITS = {"default_duration": 120, "max_duration": 300, "sample_rate": 32000, "stereo": True}
_SETTINGS_OFF = {"bpm": False, "key": False, "time_signature": False}

# Music3-like: song + director, single-shot compile, no structured settings.
MUSIC3_CAPS = {
    "preset_modes": ["song"],
    "modes": {
        "t2m": {},
        "song": {},
        "director": {
            "max_sections": 12,
            "per_section_prompts": True,
            "section_duration_hints": True,
            "references": "whole",
            "compile": "single_shot",
        },
    },
    "settings": _SETTINGS_OFF,
    "limits": _LIMITS,
}

# ACE-like: adds style/repaint, per-section references, structured bpm.
ACE_CAPS = {
    "preset_modes": ["song"],
    "modes": {
        "t2m": {},
        "song": {},
        "style": {"max_reference_seconds": 30},
        "repaint": {},
        "director": {"max_sections": 12, "per_section_prompts": True, "references": "per_section", "compile": "single_shot"},
    },
    "settings": {"bpm": True, "key": False, "time_signature": False},
    "limits": _LIMITS,
}

# YuE-like: song + extend, no director.
YUE_CAPS = {
    "preset_modes": ["song"],
    "modes": {
        "t2m": {},
        "song": {},
        "extend": {},
    },
    "settings": _SETTINGS_OFF,
    "limits": _LIMITS,
}


@pytest.fixture
def storage_dir(tmp_path):
    (tmp_path / "ref.wav").write_bytes(b"fake-audio")
    (tmp_path / "track.wav").write_bytes(b"fake-track")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "clip.wav").write_bytes(b"fake-clip")
    return tmp_path


def _base_doc(mode, **overrides):
    doc = {
        "schema_version": 1,
        "mode": mode,
        "description": "warm lo-fi, vinyl crackle",
        "sections": None,
        "references": None,
        "extend_source": None,
        "repaint": None,
        "settings": {"duration": 120, "seed": -1},
    }
    doc.update(overrides)
    return doc


# --- t2m -----------------------------------------------------------------

def test_t2m_happy_path(storage_dir):
    doc = _base_doc("t2m")
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert out["mode"] == "t2m"
    assert out["sections"] == []
    assert out["references"] == []
    assert isinstance(out["settings"]["seed"], int)
    assert out["settings"]["seed"] >= 0


# --- description_required (real MiniMax-Music3 bug: an empty caption reached
# the generator pipe, which failed mid-generation with "'caption' cannot be
# empty -- there is nothing to generate music from") ------------------------

_DESCRIPTION_REQUIRED_CAPS = {
    "preset_modes": ["song"],
    "modes": {
        "t2m": {"description_required": True},
        "song": {"description_required": True},
        "director": {"description_required": True, "max_sections": 12, "compile": "single_shot"},
    },
    "settings": _SETTINGS_OFF,
    "limits": _LIMITS,
}


def test_description_required_rejects_missing_description(storage_dir):
    doc = _base_doc("song", sections={"id": "s-1", "lyrics": "la la"})
    doc.pop("description")
    with pytest.raises(MusicDirectorValidationError) as excinfo:
        normalize_music_director(doc, _DESCRIPTION_REQUIRED_CAPS, str(storage_dir))
    assert "non-empty description" in str(excinfo.value)


def test_description_required_accepts_non_empty_description(storage_dir):
    doc = _base_doc("t2m", description="warm 90s boom-bap, vinyl crackle")
    out = normalize_music_director(doc, _DESCRIPTION_REQUIRED_CAPS, str(storage_dir))
    assert out["description"] == "warm 90s boom-bap, vinyl crackle"


def test_description_optional_by_default_unchanged_behavior(storage_dir):
    """Without `description_required`, an empty description stays valid --
    the documented default (a `t2m` document with an empty description is a
    valid, if useless, request)."""
    doc = _base_doc("t2m", description="")
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert out["description"] == ""


# --- song ------------------------------------------------------------------

def test_song_happy_path_single_section_object(storage_dir):
    # "plain lyrics" shorthand: a bare section object, not a list.
    doc = _base_doc("song", sections={"id": "s-1", "kind": "verse", "lyrics": "hello world"})
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert len(out["sections"]) == 1
    assert out["sections"][0]["lyrics"] == "hello world"


def test_song_happy_path_section_list(storage_dir):
    doc = _base_doc("song", sections=[
        {"id": "s-1", "kind": "verse", "lyrics": "verse one"},
        {"id": "s-2", "kind": "chorus", "lyrics": "chorus one"},
    ])
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert [s["id"] for s in out["sections"]] == ["s-1", "s-2"]


def test_song_references_allowed_when_style_declared(storage_dir):
    doc = _base_doc(
        "song",
        sections={"id": "s-1", "lyrics": "la"},
        references=[{"id": "r-1", "media": {"relative_path": "ref.wav"}}],
    )
    out = normalize_music_director(doc, ACE_CAPS, str(storage_dir))
    assert len(out["references"]) == 1
    assert out["references"][0]["media"]["path"] == str(storage_dir / "ref.wav")


# --- style -------------------------------------------------------------

def test_style_happy_path(storage_dir):
    doc = _base_doc("style", references=[{"id": "r-1", "media": {"relative_path": "ref.wav"}}])
    out = normalize_music_director(doc, ACE_CAPS, str(storage_dir))
    assert len(out["references"]) == 1


def test_style_reference_without_duration_hint_passes(storage_dir):
    # No client-declared duration_seconds -- nothing to check against the cap.
    doc = _base_doc("style", references=[{"id": "r-1", "media": {"relative_path": "ref.wav"}}])
    out = normalize_music_director(doc, ACE_CAPS, str(storage_dir))
    assert len(out["references"]) == 1


# --- extend ----------------------------------------------------------------

def test_extend_happy_path(storage_dir):
    doc = _base_doc("extend", extend_source={"media": {"relative_path": "track.wav"}})
    out = normalize_music_director(doc, YUE_CAPS, str(storage_dir))
    assert out["extend_source"]["media"]["path"] == str(storage_dir / "track.wav")


# --- repaint -----------------------------------------------------------

def test_repaint_happy_path(storage_dir):
    doc = _base_doc(
        "repaint",
        repaint={"source": {"media": {"relative_path": "track.wav"}}, "start": 12.0, "end": 20.0},
    )
    out = normalize_music_director(doc, ACE_CAPS, str(storage_dir))
    assert out["repaint"]["start"] == 12.0
    assert out["repaint"]["end"] == 20.0
    assert out["repaint"]["source"]["media"]["path"] == str(storage_dir / "track.wav")


# --- director ----------------------------------------------------------

def test_director_happy_path(storage_dir):
    doc = _base_doc(
        "director",
        sections=[
            {"id": "s-1", "kind": "intro", "lyrics": ""},
            {"id": "s-2", "kind": "verse", "lyrics": "verse one", "style_hint": "gritty"},
            {"id": "s-3", "kind": "chorus", "lyrics": "chorus one"},
        ],
    )
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert [s["kind"] for s in out["sections"]] == ["intro", "verse", "chorus"]
    assert out["sections"][1]["style_hint"] == "gritty"


def test_director_max_sections_enforced(storage_dir):
    caps = copy.deepcopy(MUSIC3_CAPS)
    caps["modes"]["director"]["max_sections"] = 2
    doc = _base_doc("director", sections=[
        {"id": "s-1", "kind": "verse", "lyrics": "a"},
        {"id": "s-2", "kind": "chorus", "lyrics": "b"},
        {"id": "s-3", "kind": "outro", "lyrics": "c"},
    ])
    with pytest.raises(MusicDirectorValidationError) as excinfo:
        normalize_music_director(doc, caps, str(storage_dir))
    assert "at most 2 sections" in str(excinfo.value)


def test_director_style_hint_without_capability_rejected(storage_dir):
    caps = copy.deepcopy(MUSIC3_CAPS)
    caps["modes"]["director"]["per_section_prompts"] = False
    doc = _base_doc("director", sections=[{"id": "s-1", "kind": "verse", "lyrics": "a", "style_hint": "gritty"}])
    with pytest.raises(MusicDirectorValidationError) as excinfo:
        normalize_music_director(doc, caps, str(storage_dir))
    assert "style_hint" in str(excinfo.value)


def test_director_duration_hint_without_capability_rejected(storage_dir):
    caps = copy.deepcopy(MUSIC3_CAPS)
    caps["modes"]["director"]["section_duration_hints"] = False
    doc = _base_doc("director", sections=[{"id": "s-1", "kind": "verse", "lyrics": "a", "duration_hint": 16.0}])
    with pytest.raises(MusicDirectorValidationError) as excinfo:
        normalize_music_director(doc, caps, str(storage_dir))
    assert "duration_hint" in str(excinfo.value)


def test_director_duration_hint_capability_defaults_to_false_when_absent(storage_dir):
    caps = copy.deepcopy(MUSIC3_CAPS)
    del caps["modes"]["director"]["section_duration_hints"]
    doc = _base_doc("director", sections=[{"id": "s-1", "kind": "verse", "lyrics": "a", "duration_hint": 16.0}])
    with pytest.raises(MusicDirectorValidationError) as excinfo:
        normalize_music_director(doc, caps, str(storage_dir))
    assert "duration_hint" in str(excinfo.value)


def test_director_duration_hint_kept_when_capability_declared(storage_dir):
    doc = _base_doc("director", sections=[{"id": "s-1", "kind": "verse", "lyrics": "a", "duration_hint": 16.0}])
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert out["sections"][0]["duration_hint"] == 16.0


def test_director_per_section_references_happy_path(storage_dir):
    doc = _base_doc(
        "director",
        sections=[{"id": "s-1", "kind": "verse", "lyrics": "a", "references": ["r-1"]}],
        references=[{"id": "r-1", "media": {"relative_path": "ref.wav"}}],
    )
    out = normalize_music_director(doc, ACE_CAPS, str(storage_dir))
    assert out["sections"][0]["references"] == ["r-1"]


# --- settings ----------------------------------------------------------

def test_settings_duration_defaults_and_seed_rolled(storage_dir):
    doc = _base_doc("t2m", settings={})
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert out["settings"]["duration"] == 120
    assert isinstance(out["settings"]["seed"], int)
    assert out["settings"]["seed"] >= 0


def test_settings_explicit_seed_passes_through(storage_dir):
    doc = _base_doc("t2m", settings={"duration": 60, "seed": 12345})
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert out["settings"]["seed"] == 12345


def test_settings_bpm_accepted_when_declared(storage_dir):
    doc = _base_doc("t2m", settings={"duration": 60, "seed": -1, "bpm": 92})
    out = normalize_music_director(doc, ACE_CAPS, str(storage_dir))
    assert out["settings"]["bpm"] == 92


# --- schema / mode / error accumulation --------------------------------

def test_schema_version_missing_rejected(storage_dir):
    doc = _base_doc("t2m")
    del doc["schema_version"]
    with pytest.raises(MusicDirectorValidationError) as excinfo:
        normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert "missing schema_version" in str(excinfo.value)


def test_schema_version_2_rejected(storage_dir):
    doc = _base_doc("t2m")
    doc["schema_version"] = 2
    with pytest.raises(MusicDirectorValidationError) as excinfo:
        normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert "schema_version" in str(excinfo.value)


def test_all_errors_collected(storage_dir):
    doc = _base_doc(
        "director",
        sections=[{"id": "s-1", "kind": "breakdown", "lyrics": "a"}],
        settings={"duration": 9999, "seed": -1, "bpm": 92},
    )
    with pytest.raises(MusicDirectorValidationError) as excinfo:
        normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    errors = excinfo.value.errors
    assert len(errors) >= 3
    joined = str(excinfo.value)
    assert "kind" in joined
    assert "settings.bpm" in joined
    assert "duration" in joined


def test_upload_shaped_path_falls_back_to_relative_path(storage_dir):
    doc = _base_doc("extend", extend_source={
        "media": {"path": "storage/uploads/track.wav", "relative_path": "track.wav"},
    })
    out = normalize_music_director(doc, YUE_CAPS, str(storage_dir))
    assert out["extend_source"]["media"]["path"] == str(storage_dir / "track.wav")


def test_unknown_top_level_key_preserved(storage_dir):
    doc = _base_doc("t2m")
    doc["client_version"] = "1.2.3"
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert out["client_version"] == "1.2.3"


def test_document_segments_survives_normalize_round_trip(storage_dir):
    # `segments` is a frontend-only echo of the Song structure segment
    # editor (frontend/src/lib/utils/musicDirector.ts's
    # `MusicDirectorValue.segments`) -- the compiler and every other backend
    # reader only ever look at `sections[].lyrics`/`.kind` (derived from
    # these segments by `buildMusicDirectorSubmission` before the doc ever
    # reaches here), but the field itself has to survive normalize opaquely
    # so a history/session restore of this generation gets the user's
    # original chip/segment editor state back instead of always falling
    # back to a freshly re-derived segment.
    segments = [
        {"id": "seg-1", "type": "content", "content": "verse one", "chips": {}, "enabled": True, "name": "Verse"}
    ]
    doc = _base_doc(
        "song",
        sections={"id": "s-1", "kind": "verse", "lyrics": "verse one"},
        segments=segments,
    )
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert out["sections"][0]["lyrics"] == "verse one"
    assert out["segments"] == segments


def test_document_segments_absent_normalizes_to_none(storage_dir):
    doc = _base_doc("song", sections={"id": "s-1", "kind": "verse", "lyrics": "verse one"})
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert out["segments"] is None


def test_document_segments_wrong_type_dropped_not_validated(storage_dir):
    # "opaque list without validation beyond type" -- a non-list value is
    # simply dropped to None rather than raising, since this module has no
    # business understanding the frontend's Segment shape.
    doc = _base_doc("song", sections={"id": "s-1", "kind": "verse", "lyrics": "x"}, segments="not-a-list")
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    assert out["segments"] is None


def test_compiled_lyrics_still_come_from_sections_not_segments(storage_dir):
    # compile_sections_to_lyrics reads only `sections[].lyrics` -- a
    # doc-level `segments` carrying stale/different content must never leak
    # into the compiled output.
    doc = _base_doc(
        "director",
        sections=[{"id": "s-1", "kind": "verse", "lyrics": "the real lyrics"}],
        segments=[{"id": "x", "type": "content", "content": "stale segment text", "chips": {}, "enabled": True}],
    )
    out = normalize_music_director(doc, MUSIC3_CAPS, str(storage_dir))
    compiled = compile_sections_to_lyrics(out["sections"])
    assert "the real lyrics" in compiled
    assert "stale segment text" not in compiled


# --- overlay -------------------------------------------------------------

def test_overlay_applies_preset_mode_override():
    base = {
        "preset_modes": ["song", "cover"],
        "modes": {
            "t2m": {},
            "song": {},
            "director": {"max_sections": 12, "per_section_prompts": True, "references": "whole"},
        },
        "settings": _SETTINGS_OFF,
        "limits": _LIMITS,
    }
    caps = copy.deepcopy(base)
    caps["preset_mode_overrides"] = {
        "cover": {
            "modes": {
                "director": {"references": "per_section"},
            },
        },
    }
    merged = apply_preset_mode_overlay(caps, "cover")
    assert merged["modes"]["director"]["references"] == "per_section"
    # Untouched keys survive the per-composition-mode merge.
    assert merged["modes"]["director"]["max_sections"] == 12
    assert "preset_mode_overrides" not in merged

    unmerged = apply_preset_mode_overlay(caps, "song")
    assert unmerged["modes"]["director"]["references"] == "whole"


def test_overlay_missing_preset_mode_returns_base_unchanged():
    caps = {
        "preset_modes": ["song"],
        "modes": {"t2m": {}},
        "preset_mode_overrides": {"other": {"modes": {"t2m": {}}}},
    }
    merged = apply_preset_mode_overlay(caps, "song")
    assert merged["modes"] == {"t2m": {}}
    assert "preset_mode_overrides" not in merged


# --- compiler ------------------------------------------------------------

def test_compile_sections_to_lyrics_golden():
    sections = [
        {"id": "s-1", "kind": "intro", "lyrics": ""},
        {"id": "s-2", "kind": "verse", "lyrics": "walking down the empty street"},
        {"id": "s-3", "kind": "chorus", "lyrics": "we are the signal fade"},
    ]
    result = compile_sections_to_lyrics(sections)
    assert result == (
        "[Intro]\n\n\n"
        "[Verse]\nwalking down the empty street\n\n"
        "[Chorus]\nwe are the signal fade"
    )


def test_compile_sections_to_lyrics_canonicalizes_and_strips():
    sections = [
        {"id": "s-1", "kind": "CHORUS", "lyrics": "  padded lyrics  "},
        {"id": "s-2", "kind": "pre_chorus", "lyrics": "rise"},
        {"id": "s-3", "kind": "solo", "lyrics": ""},
    ]
    result = compile_sections_to_lyrics(sections)
    assert result == "[Chorus]\npadded lyrics\n\n[Pre-Chorus]\nrise\n\n[Solo]\n"


def test_compile_sections_to_lyrics_empty_list():
    assert compile_sections_to_lyrics([]) == ""


def test_compile_author_supplied_tags_pass_through_untagged():
    """A section whose lyrics already open with a bracket tag on its own line
    is the author's structure -- prepending the kind tag would double it
    (maintainer, 2026-08-18: pasting whole pre-tagged lyrics must just work)."""
    sections = [
        {"id": "s-1", "kind": "verse", "lyrics": "[Verse]\nwalking down the empty street"},
        {"id": "s-2", "kind": "chorus", "lyrics": "we are the signal fade"},
        {"id": "s-3", "kind": "verse", "lyrics": "[Intro]\nhum\n\n[Outro]\nfade out"},
    ]
    result = compile_sections_to_lyrics(sections)
    assert result == (
        "[Verse]\nwalking down the empty street\n\n"
        "[Chorus]\nwe are the signal fade\n\n"
        "[Intro]\nhum\n\n[Outro]\nfade out"
    )


def test_compile_mid_line_brackets_are_not_structure():
    """"[x2]"-style annotations mid-line, or a first line that only CONTAINS
    a bracket, still get the kind tag prepended -- only a whole-line leading
    tag suppresses it."""
    sections = [
        {"id": "s-1", "kind": "chorus", "lyrics": "we are the signal [x2]"},
        {"id": "s-2", "kind": "bridge", "lyrics": "quiet now [fades] and gone"},
    ]
    result = compile_sections_to_lyrics(sections)
    assert result == (
        "[Chorus]\nwe are the signal [x2]\n\n"
        "[Bridge]\nquiet now [fades] and gone"
    )


REJECTED_DOCUMENTS = [
    pytest.param(
        lambda: _base_doc("t2m", sections={"id": "s-1", "lyrics": "la la"}),
        MUSIC3_CAPS, ("sections",), None,
        id="t2m_rejects_sections",
    ),
    pytest.param(
        lambda: _base_doc("t2m", references=[{"id": "r-1", "media": {"relative_path": "ref.wav"}}]),
        MUSIC3_CAPS, ("references",), None,
        id="t2m_rejects_references",
    ),
    pytest.param(
        lambda: _base_doc("t2m", description=""),
        _DESCRIPTION_REQUIRED_CAPS, ("description", "non-empty description"), None,
        id="description_required_rejects_empty_description",
    ),
    pytest.param(
        lambda: _base_doc("t2m", description="   \n\t"),
        _DESCRIPTION_REQUIRED_CAPS, (), None,
        id="description_required_rejects_whitespace_only_description",
    ),
    pytest.param(
        lambda: _base_doc("song"),
        MUSIC3_CAPS, ("song mode requires lyrics",), None,
        id="song_requires_at_least_one_section",
    ),
    pytest.param(
        lambda: _base_doc(
            "song",
            sections={"id": "s-1", "lyrics": "la"},
            references=[{"id": "r-1", "media": {"relative_path": "ref.wav"}}],
        ),
        MUSIC3_CAPS, ("references",), None,
        id="song_references_rejected_without_style_mode",
    ),
    pytest.param(
        lambda: _base_doc("style"),
        ACE_CAPS, ("requires at least one reference",), None,
        id="style_requires_references",
    ),
    pytest.param(
        lambda: _base_doc("style", references=[]),
        ACE_CAPS, ("requires at least one reference",), None,
        id="style_empty_references_list_rejected",
    ),
    pytest.param(
        lambda: _base_doc("style", references=[
            {"id": "r-1", "media": {"relative_path": "ref.wav", "duration_seconds": 45}},
        ]),
        ACE_CAPS, ("max_reference_seconds",), None,
        id="style_max_reference_seconds_violation",
    ),
    pytest.param(
        lambda: _base_doc("style", references=[{"id": "r-1", "media": {"relative_path": "ref.wav"}}]),
        MUSIC3_CAPS, ("unsupported mode",), None,
        id="style_not_declared_rejected",
    ),
    pytest.param(
        lambda: _base_doc("extend"),
        YUE_CAPS, ("extend_source",), None,
        id="extend_requires_extend_source",
    ),
    pytest.param(
        lambda: _base_doc("t2m", extend_source={"media": {"relative_path": "track.wav"}}),
        MUSIC3_CAPS, ("extend_source",), None,
        id="extend_source_rejected_outside_extend_mode",
    ),
    pytest.param(
        lambda: _base_doc("repaint"),
        ACE_CAPS, ("repaint",), None,
        id="repaint_requires_block",
    ),
    pytest.param(
        lambda: _base_doc(
            "repaint",
            repaint={"source": {"media": {"relative_path": "track.wav"}}, "start": 20.0, "end": 12.0},
        ),
        ACE_CAPS, ("start must be >= 0 and < end",), None,
        id="repaint_start_after_end_rejected",
    ),
    pytest.param(
        lambda: _base_doc("director", sections=[]),
        MUSIC3_CAPS, ("director mode requires at least one section",), None,
        id="director_requires_at_least_one_section",
    ),
    pytest.param(
        lambda: _base_doc("director", sections=[{"id": "s-1", "kind": "breakdown", "lyrics": "a"}]),
        MUSIC3_CAPS, ("kind must be one of",), None,
        id="director_invalid_kind_rejected",
    ),
    pytest.param(
        lambda: _base_doc("director", sections=[{"id": "s-1", "kind": "verse", "lyrics": "a", "duration_hint": -1}]),
        MUSIC3_CAPS, ("duration_hint",), None,
        id="director_duration_hint_must_be_positive",
    ),
    pytest.param(
        lambda: _base_doc(
            "director",
            sections=[{"id": "s-1", "kind": "verse", "lyrics": "a", "references": ["nope"]}],
            references=[{"id": "r-1", "media": {"relative_path": "ref.wav"}}],
        ),
        ACE_CAPS, ("not part of this document's references pool",), None,
        id="director_per_section_references_unknown_id_rejected",
    ),
    pytest.param(
        lambda: _base_doc(
            "director",
            sections=[{"id": "s-1", "kind": "verse", "lyrics": "a", "references": ["r-1"]}],
            references=[{"id": "r-1", "media": {"relative_path": "ref.wav"}}],
        ),
        MUSIC3_CAPS, ("does not allow per-section selection",), None,
        id="director_section_references_rejected_when_capability_is_whole",
    ),
    pytest.param(
        lambda: _base_doc("t2m", settings={"duration": 999, "seed": -1}),
        MUSIC3_CAPS, ("exceeds the allowed maximum",), None,
        id="settings_duration_exceeds_max_rejected",
    ),
    pytest.param(
        lambda: _base_doc("t2m", settings={"duration": 60, "seed": -1, "bpm": 92}),
        MUSIC3_CAPS, ("settings.bpm is not supported", "description text"), None,
        id="settings_bpm_rejected_when_not_declared",
    ),
    pytest.param(
        lambda: _base_doc("t2m", settings={"duration": 60, "seed": -1, "bpm": -5}),
        ACE_CAPS, ("settings.bpm must be a positive number",), None,
        id="settings_bpm_must_be_positive_number",
    ),
    pytest.param(
        lambda: _base_doc("t2m", settings={"duration": 60, "seed": -1, "key": "C minor"}),
        ACE_CAPS, ("settings.key is not supported",), None,
        id="settings_key_rejected_when_not_declared_even_on_ace",
    ),
    pytest.param(
        lambda: _base_doc("repaint"),
        MUSIC3_CAPS, ("unsupported mode", "repaint"), None,
        id="mode_rejected_when_not_capable",
    ),
    pytest.param(
        lambda: _base_doc("extend", extend_source={"media": {"relative_path": "../outside.wav"}}),
        YUE_CAPS, ("escapes the storage directory",), None,
        id="media_path_traversal_rejected",
    ),
    pytest.param(
        lambda: _base_doc("extend", extend_source={"media": {"relative_path": "missing.wav"}}),
        YUE_CAPS, ("does not exist",), None,
        id="media_path_missing_rejected",
    ),
]


@pytest.mark.parametrize("build_doc,caps,fragments,match", REJECTED_DOCUMENTS)
def test_invalid_document_rejected(storage_dir, build_doc, caps, fragments, match):
    with pytest.raises(MusicDirectorValidationError, match=match) as excinfo:
        normalize_music_director(build_doc(), caps, str(storage_dir))
    for fragment in fragments:
        assert fragment in str(excinfo.value)
