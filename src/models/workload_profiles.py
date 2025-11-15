"""Workload profiles for thermal control experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np


WorkloadGenerator = Callable[[float, int], np.ndarray]
"""Callable signature for workload factories."""


def generate_edge_ai_workload(
    Ts: float, horizon: int, profile: str = "baseline", jitter_seed: int | None = 42
) -> np.ndarray:
    """Generate edge AI workloads tailored to different deployment scenarios.

    Parameters
    ----------
    Ts: float
        Sampling period [s].
    horizon: int
        Number of simulation steps.
    profile: str, optional
        Identifier of the workload storyline to generate. Supported options are
        ``"baseline"``, ``"burst_chain"``, ``"idle_recovery"``, ``"noisy_lab"``,
        ``"thermal_shock"``, ``"tracking_drill"``, ``"progressive_ramp"``,
        ``"sensor_edge"``, ``"cooling_loss"``, and ``"thermal_resilience"``.
    jitter_seed: int or None, optional
        Seed for the Gaussian jitter applied to workloads. ``None`` disables jitter.

    Returns
    -------
    np.ndarray
        Disturbance sequence ``d[k]`` normalized to ``[0, 1]``.
    """

    time = np.arange(horizon) * Ts
    workload = np.zeros_like(time)

    if profile == "baseline":
        workload[:] = _piecewise_levels(time, [0.022, 0.055, 0.034], [5.0, 12.0])
    elif profile == "burst_chain":
        workload[:] = _piecewise_levels(time, [0.03, 0.07, 0.05, 0.06], [3.0, 9.0, 15.0])
    elif profile == "idle_recovery":
        workload[:] = _piecewise_levels(time, [0.06, 0.018, 0.045], [4.0, 13.0])
    elif profile == "noisy_lab":
        base = _piecewise_levels(time, [0.028, 0.06, 0.032], [6.0, 11.0])
        workload[:] = base + 0.005 * np.sin(2 * np.pi * time / 1.6)
    elif profile == "thermal_shock":
        workload[:] = 0.03 + 0.018 * np.tanh((time - 6.5) / 0.6)
        workload += 0.012 * np.exp(-0.5 * ((time - 10.5) / 0.8) ** 2)
    elif profile == "tracking_drill":
        workload[:] = 0.035 + 0.015 * np.sin(2 * np.pi * time / 2.4)
    elif profile == "progressive_ramp":
        workload[:] = 0.02 + 0.002 * time
        workload[time > 12.0] += 0.01
    elif profile == "sensor_edge":
        workload[:] = _piecewise_levels(time, [0.018, 0.05, 0.026, 0.045], [2.5, 8.0, 16.0])
        workload += 0.006 * np.maximum(0.0, np.cos(2 * np.pi * time / 3.2))
    elif profile == "cooling_loss":
        workload[:] = _piecewise_levels(time, [0.03, 0.062, 0.048], [5.5, 13.5])
        workload += 0.01 * (time / time.max())
    elif profile == "thermal_resilience":
        workload[:] = _piecewise_levels(time, [0.025, 0.058, 0.036, 0.05], [4.0, 10.0, 17.0])
        workload += 0.004 * np.sin(2 * np.pi * time / 1.1)
    else:
        raise ValueError(f"Unknown workload profile '{profile}'")

    if jitter_seed is not None:
        rng = np.random.default_rng(seed=jitter_seed)
        workload += rng.normal(loc=0.0, scale=0.003, size=workload.shape)

    return np.clip(workload, 0.0, 0.12)


def _piecewise_levels(time: np.ndarray, levels: list[float], breakpoints: list[float]) -> np.ndarray:
    """Return a piecewise-constant workload with the provided breakpoints."""

    segments = np.zeros_like(time)
    idx_start = 0.0
    for level, bp in zip(levels, breakpoints):
        mask = (time >= idx_start) & (time < bp)
        segments[mask] = level
        idx_start = bp
    segments[time >= idx_start] = levels[len(breakpoints)]
    return segments


__all__ = ["generate_edge_ai_workload", "WorkloadGenerator"]
