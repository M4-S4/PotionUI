"""Restart sampling (arXiv:2306.14878) for flow-matching models.

Restart sampling spends extra compute re-visiting high-noise territory: at a
chosen trigger level ``sigma_low`` reached during the descent, re-noise back up
to ``sigma_hi`` and descend again to that level, giving the model another chance
to correct compounding ODE-discretization error before the descent continues to
``0``. Restarts INTERLEAVE into the descent — they are NOT appended after it —
so the final latent is always the clean ``sigma == 0`` state.

This is a loop wrapper around plain Euler, not a new integrator: the main descent
over ``sigmas`` is byte-for-byte the same computation as
:func:`~.euler.sample_euler` (so zero restarts reduces to it exactly). A restart
fires the first time the descent reaches a sigma at-or-below its ``sigma_low``;
the current Euler step already produced a clean estimate ``x0_est = x - sigma*v``
at that point, so the re-noise reuses it (no extra model evaluation) via the flow
interpolation with FRESH noise: ``x = (1 - sigma_hi)*x0_est + sigma_hi*eps``. The
segment then descends ``n_steps`` of plain Euler from ``sigma_hi`` back to the
current level, and the main descent resumes toward ``0``. Multiple restarts fire
in list order as their trigger levels are crossed.
"""

from __future__ import annotations

import torch

from ..cfg import GuidanceStrategy
from ..hooks import StepLoop
from ..noise import fresh_noise

Tensor = torch.Tensor

# (sigma_hi, sigma_low, n_steps) per restart segment.
RestartSpec = tuple[float, float, int]


def _resolve_restarts(sampler_options: dict, total_main_steps: int) -> list[RestartSpec]:
    """Explicit ``restarts`` list wins; else a simple ``restart_count`` /
    ``restart_strength`` pair builds ``restart_count`` identical segments that
    re-noise from ``sigma_low=0.0`` (the end of the main descent) up to
    ``restart_strength`` of the way back to ``sigma_max=1.0``, each redescended
    over a share of the main step count. ``restart_count=0`` (default) yields no
    segments, i.e. plain Euler.
    """
    if "restarts" in sampler_options:
        return [
            (float(hi), float(lo), int(n))
            for hi, lo, n in sampler_options["restarts"]
        ]
    restart_count = int(sampler_options.get("restart_count", 0))
    if restart_count <= 0:
        return []
    restart_strength = float(sampler_options.get("restart_strength", 0.3))
    if not (0.0 < restart_strength <= 1.0):
        raise ValueError(f"restart_strength must be in (0, 1], got {restart_strength}")
    sigma_hi = restart_strength
    n_steps = max(1, total_main_steps // restart_count)
    return [(sigma_hi, 0.0, n_steps)] * restart_count


@torch.no_grad()
def sample_euler_restart(
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
    """Euler descent + restart segments. Same signature/semantics as
    :func:`sample_euler`.

    ``sampler_options``:

    * ``restarts`` — explicit ``[(sigma_hi, sigma_low, n_steps), ...]`` list,
      run in order after the main descent. Takes precedence over
      ``restart_count``/``restart_strength``.
    * ``restart_count`` (default ``0``) / ``restart_strength`` (default
      ``0.3``) — simple convenience config; see :func:`_resolve_restarts`.
      ``restart_count == 0`` (and no explicit ``restarts``) is exactly plain
      Euler.
    * ``generator`` (default ``None``) — ``torch.Generator`` for reproducible
      re-noise draws.
    """
    opts = sampler_options or {}
    generator = opts.get("generator")

    total_main_steps = len(sigmas) - 1
    restarts = _resolve_restarts(opts, total_main_steps)
    total_steps = total_main_steps + sum(n for _, _, n in restarts)

    with StepLoop(sigmas, hooks, is_cancelled, total_steps=total_steps) as loop:
        s_in = x.new_ones((x.shape[0],))
        step_counter = 0
        x0_est = x
        fired = [False] * len(restarts)
        for i in range(total_main_steps):
            loop.check_cancelled(step_counter)

            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            v = guidance(model_fn, x, sigma * s_in, cond, uncond, step_counter)
            x0_est = x - sigma * v
            x = x + (sigma_next - sigma) * v

            x = loop.finish_step(x, step_counter, sigma, sigma_next, x0_est)
            step_counter += 1

            # Fire every not-yet-fired restart whose trigger level we have now
            # reached, in order. x is at ``sigma_next`` with ``x0_est`` its clean
            # estimate; each segment re-noises to ``sigma_hi`` and descends back to
            # ``sigma_next`` so the main descent resumes consistently.
            reached = float(sigma_next)
            for ri, (sigma_hi, sigma_low, n_steps) in enumerate(restarts):
                if fired[ri] or reached > sigma_low + 1e-9:
                    continue
                fired[ri] = True
                eps_fresh = fresh_noise(x, generator)
                x = (1.0 - sigma_hi) * x0_est + sigma_hi * eps_fresh
                seg_sigmas = torch.linspace(
                    sigma_hi, reached, n_steps + 1, device=x.device, dtype=x.dtype
                )
                for j in range(n_steps):
                    loop.check_cancelled(step_counter)

                    seg_sigma = seg_sigmas[j]
                    seg_next = seg_sigmas[j + 1]

                    v = guidance(model_fn, x, seg_sigma * s_in, cond, uncond, step_counter)
                    x0_est = x - seg_sigma * v
                    x = x + (seg_next - seg_sigma) * v

                    x = loop.finish_step(x, step_counter, seg_sigma, seg_next, x0_est)
                    step_counter += 1

    return x
