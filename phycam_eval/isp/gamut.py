"""Explicit pre-tone and post-tone gamut/range policies."""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..color import validate_rgb_array

__all__ = [
    "PostToneGamutPolicy",
    "PreToneGamutPolicy",
    "apply_post_tone_gamut",
    "apply_pre_tone_gamut",
]


class PreToneGamutPolicy(str, Enum):
    """Policy for signed output-linear RGB before fractional-power tone."""

    CLIP_NEGATIVE = "clip_negative"


class PostToneGamutPolicy(str, Enum):
    """Policy for producing the final bounded output-linear RGB gamut."""

    CLIP_UNIT_RANGE = "clip_unit_range"


def _pre_tone_policy(value: PreToneGamutPolicy | str) -> PreToneGamutPolicy:
    try:
        return PreToneGamutPolicy(value)
    except ValueError as exc:
        choices = ", ".join(policy.value for policy in PreToneGamutPolicy)
        raise ValueError(
            f"unknown pre-tone gamut policy {value!r}; expected one of: {choices}"
        ) from exc


def _post_tone_policy(value: PostToneGamutPolicy | str) -> PostToneGamutPolicy:
    try:
        return PostToneGamutPolicy(value)
    except ValueError as exc:
        choices = ", ".join(policy.value for policy in PostToneGamutPolicy)
        raise ValueError(
            f"unknown post-tone gamut policy {value!r}; expected one of: {choices}"
        ) from exc


def apply_pre_tone_gamut(
    signed_output_rgb: np.ndarray,
    *,
    policy: PreToneGamutPolicy | str = PreToneGamutPolicy.CLIP_NEGATIVE,
) -> np.ndarray:
    """Map signed color-transform output to tone's nonnegative domain.

    ``clip_negative`` is deliberately simple and explicit: each negative
    channel becomes zero and every nonnegative value is unchanged.  More
    sophisticated gamut maps can be added as new named policies without
    changing this stage's contract.
    """

    values = validate_rgb_array(signed_output_rgb, name="signed_output_rgb")
    selected = _pre_tone_policy(policy)
    if selected is PreToneGamutPolicy.CLIP_NEGATIVE:
        return np.maximum(values, np.asarray(0.0, dtype=values.dtype))
    raise AssertionError("unreachable pre-tone gamut policy")


def apply_post_tone_gamut(
    tone_mapped_rgb: np.ndarray,
    *,
    policy: PostToneGamutPolicy | str = PostToneGamutPolicy.CLIP_UNIT_RANGE,
) -> np.ndarray:
    """Map tone output to the declared output-linear range and gamut.

    ``clip_unit_range`` independently clips channels to ``[0, 1]``.  It may
    alter chromaticity, so tone invariants are evaluated before this stage.
    """

    values = validate_rgb_array(tone_mapped_rgb, name="tone_mapped_rgb")
    selected = _post_tone_policy(policy)
    if selected is PostToneGamutPolicy.CLIP_UNIT_RANGE:
        return np.clip(
            values,
            np.asarray(0.0, dtype=values.dtype),
            np.asarray(1.0, dtype=values.dtype),
        )
    raise AssertionError("unreachable post-tone gamut policy")
