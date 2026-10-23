"""Deterministic (non-ancestral) Euler with the CFG++ velocity/target split.

CFG++ (Chung et al., "CFG++: Manifold-Constrained Classifier Free Guidance
for Diffusion Models", arXiv:2406.08070) keeps the CFG-guided prediction as
the step's TARGET, but derives the step's DIRECTION from the plain
UNCONDITIONAL prediction instead of the guided one -- see
``euler_ancestral_cfg_pp.py``'s module docstring for the full re-derivation
of the eps/x0 formulation this reuses (identical algebra, ``eta`` fixed at
``0.0`` here instead of user-configurable).

Provenance: this is the ``eta=0.0`` case of THIS codebase's own
``euler_ancestral_cfg_pp.py`` (an in-repo re-derivation from the CFG++ paper,
not a port of any ComfyUI/GPL sampler -- see that module's docstring for the
citation and the ComfyUI-consultation disclosure). :func:`sample_euler_cfg_pp`
delegates to it with ``eta`` pinned to ``0.0``: no fresh-noise draw ever happens,
so there is no ancestral/stochastic component and no ``generator``/``eta`` knob
to configure. It stays a separately REGISTERED sampler (``euler_cfg_pp``) so
callers picking a sampler by name can distinguish "the deterministic recipe" from
"the ancestral recipe with eta forced to 0" (e.g. the LTX-2.3 distilled speed
profile, whose first-party recipe is deterministic -- see ``docs/models/ltx.md``).

Target/direction split, per step (``sigma`` -> ``sigma_next``):

* ``v_cfg`` (the guidance strategy's combined output) gives the TARGET:
  ``x0_est = x - sigma * v_cfg``.
* ``v_uncond`` (``guidance.last_uncond_v``, the strategy's raw uncond-branch
  prediction) gives the DIRECTION: ``eps = x + (1 - sigma) * v_uncond``.
* Next state: ``x_next = (1 - sigma_next) * x0_est + sigma_next * eps`` --
  the same DDIM-style anchor ``euler_ancestral_cfg_pp`` uses, which collapses
  EXACTLY to ``x0_est`` at the terminal step (``sigma_next == 0``) regardless
  of any ``v_cfg``/``v_uncond`` mismatch.

Guidance strategies with no uncond branch (``EmbeddedGuidance``, ``NoCFG``, or
``TrueCFG`` at ``scale == 1.0`` -- exactly the LTX-2.3 distilled recipe's own
``cfg=1.0``) expose no ``last_uncond_v`` (or expose ``None``); this sampler
then falls back to ``v_uncond = v_cfg``, at which point the CFG++ split is a
pure no-op and the per-step update reduces algebraically to plain Euler
(``x = x + (sigma_next - sigma) * v_cfg``) -- taken here as an exact shortcut,
not merely a numerically-close approximation, so this sampler is BIT-IDENTICAL
to :func:`~.euler.sample_euler` whenever the guidance strategy has no genuine
uncond branch (in particular, always at ``cfg=1.0``). This is the property the
LTX-2.3 distilled-speed-profile fix relies on: swapping the profile's sampler
from ``euler_ancestral_cfg_pp`` to this one changes nothing about the
per-step math at the recipe's own ``cfg=1.0`` (no ancestral noise was ever
correct there either -- see this file's sibling module for the same finding),
while giving CFG++'s target/direction split for free to any custom profile
that raises CFG above 1.0.
"""

from __future__ import annotations

import torch

from ..cfg import GuidanceStrategy
from .euler_ancestral_cfg_pp import sample_euler_ancestral_cfg_pp

Tensor = torch.Tensor


@torch.no_grad()
def sample_euler_cfg_pp(
    model_fn,
    x: Tensor,
    sigmas: Tensor,
    guidance: GuidanceStrategy,
    cond: dict,
    uncond: dict | None = None,
    hooks=(),
    is_cancelled=None,
    sampler_options: dict | None = None,
) -> Tensor:
    """Deterministic Euler, CFG++ target/direction decoupling. See module
    docstring for the formula and the ``cfg=1.0`` <-> plain-Euler equivalence.

    Same signature as :func:`~.euler.sample_euler`; ``sampler_options`` is
    part of the uniform sampler contract (:mod:`~..registry`); ``eta`` is
    pinned to ``0.0`` and nothing else is read.
    """
    return sample_euler_ancestral_cfg_pp(
        model_fn,
        x,
        sigmas,
        guidance,
        cond,
        uncond,
        hooks=hooks,
        is_cancelled=is_cancelled,
        sampler_options={**(sampler_options or {}), "eta": 0.0},
    )
