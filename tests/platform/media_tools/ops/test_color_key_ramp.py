from __future__ import annotations

import numpy as np

from src.platform.media_tools.ops._imaging import MAX_CHROMA_DISTANCE, soft_alpha


def test_zero_softness_is_a_hard_binary_cut():
    alpha = soft_alpha(np.array([0.0, 50.0, 100.0]), threshold=60.0, ramp_half=0.0)

    assert list(alpha) == [0, 0, 255]


def test_the_ramp_is_monotonic_and_bounded():
    distance = np.linspace(0, MAX_CHROMA_DISTANCE, 50)

    alpha = [int(value) for value in soft_alpha(
        distance, threshold=MAX_CHROMA_DISTANCE / 2, ramp_half=MAX_CHROMA_DISTANCE / 4,
    )]

    assert alpha == sorted(alpha)
    assert alpha[0] == 0
    assert alpha[-1] == 255


def test_the_ramp_has_values_strictly_between_transparent_and_opaque():
    alpha = soft_alpha(np.linspace(40.0, 80.0, 20), threshold=60.0, ramp_half=20.0)

    assert any(0 < int(value) < 255 for value in alpha)


def test_the_ramp_is_centred_on_the_threshold():
    alpha = soft_alpha(np.array([60.0]), threshold=60.0, ramp_half=20.0)

    assert 120 <= int(alpha[0]) <= 135
