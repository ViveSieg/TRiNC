"""Workload profiles for thermal control experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np

WorkloadGenerator = Callable[[float, int], np.ndarray]
"""Callable signature for workload factories."""


def _piecewise_levels(time: np.ndarray, levels: list[float], breakpoints: list[int]) -> np.ndarray:
    """Build piecewise-constant load over the given time indices."""

    segments = np.zeros_like(time, dtype=float)
    idx_start = 0
    for level, bp in zip(levels, breakpoints):
        mask = (time >= idx_start) & (time < bp)
        segments[mask] = level
        idx_start = bp
    segments[time >= idx_start] = levels[len(breakpoints)]
    return segments


def generate_edge_ai_workload(
    Ts: float,
    horizon: int,
    profile: str = "baseline",
    jitter_seed: int | None = 42,
) -> np.ndarray:
    """Generate edge-AI workload profiles with normalized heat input.

    The workloads mimic common edge inference patterns: bursts from camera
    pipelines, idle recoveries after batch jobs, noisy lab benches, and
    deliberate stress tests. They are *not* intended to exactly match any
    commercial TPU trace; rather, they provide controlled, repeatable
    stimuli for cross-controller comparisons.

    Parameters
    ----------
    Ts : float
        Sampling time [s].
    horizon : int
        Number of simulation steps.
    profile : str, optional
        Name of the workload storyline to generate. Supported options:
        ``"baseline"``, ``"burst_chain"``, ``"idle_recovery"``,
        ``"noisy_lab"``, ``"thermal_shock"``, ``"tracking_drill"``,
        ``"progressive_ramp"``, ``"sensor_edge"``, ``"cooling_loss"``,
        and ``"thermal_resilience"``.
    jitter_seed : int | None, optional
        Seed controlling small Gaussian jitter. ``None`` disables noise.

    Returns
    -------
    np.ndarray
        1D array ``w[k]`` of length ``horizon`` with values in ``[0, 1]``.
    """

    time = np.arange(horizon)
    workload = np.zeros_like(time, dtype=float)

    if profile == "baseline":
        workload = _piecewise_levels(time, [0.25, 0.55, 0.3], [int(0.3 * horizon), int(0.65 * horizon)])
    elif profile == "burst_chain":
        burst = _piecewise_levels(time, [0.25, 0.8, 0.4, 0.7, 0.3], [int(0.15 * horizon), int(0.3 * horizon), int(0.5 * horizon), int(0.65 * horizon)])
        workload = burst
    elif profile == "idle_recovery":
        workload = _piecewise_levels(time, [0.7, 0.1, 0.45], [int(0.2 * horizon), int(0.55 * horizon)])
    elif profile == "noisy_lab":
        base = _piecewise_levels(time, [0.35, 0.55, 0.4], [int(0.35 * horizon), int(0.7 * horizon)])
        sinusoid = 0.05 * np.sin(2 * np.pi * time * Ts / 2.5)
        workload = base + sinusoid
    elif profile == "thermal_shock":
        ramp = 0.15 + 0.65 * (np.tanh((time - 0.35 * horizon) / (0.08 * horizon)) + 1) / 2
        peak = 0.25 * np.exp(-0.5 * ((time - 0.55 * horizon) / (0.07 * horizon)) ** 2)
        workload = ramp + peak
    elif profile == "tracking_drill":
        workload = 0.45 + 0.2 * np.sin(2 * np.pi * time * Ts / 4.0)
    elif profile == "progressive_ramp":
        ramp = np.linspace(0.15, 0.75, horizon)
        step = np.zeros_like(ramp)
        step[int(0.6 * horizon) :] += 0.15
        workload = ramp + step
    elif profile == "sensor_edge":
        baseline = _piecewise_levels(time, [0.12, 0.45, 0.2, 0.4], [int(0.2 * horizon), int(0.55 * horizon), int(0.85 * horizon)])
        periodic = 0.15 * np.maximum(0.0, np.cos(2 * np.pi * time * Ts / 3.5))
        workload = baseline + periodic
    elif profile == "cooling_loss":
        rising = np.linspace(0.25, 0.8, horizon)
        wobble = 0.08 * np.sin(2 * np.pi * time * Ts / 5.0)
        workload = rising + wobble
    elif profile == "thermal_resilience":
        piecewise = _piecewise_levels(time, [0.2, 0.65, 0.35, 0.75, 0.25], [int(0.18 * horizon), int(0.35 * horizon), int(0.55 * horizon), int(0.72 * horizon)])
        bursts = 0.18 * np.exp(-0.5 * ((time - 0.42 * horizon) / (0.04 * horizon)) ** 2)
        low_freq = 0.06 * np.sin(2 * np.pi * time * Ts / 6.0)
        workload = piecewise + bursts + low_freq
    else:
        raise ValueError(f"Unknown workload profile '{profile}'")

    if jitter_seed is not None:
        rng = np.random.default_rng(seed=jitter_seed)
        workload += rng.normal(loc=0.0, scale=0.02, size=workload.shape)

    return np.clip(workload, 0.0, 1.0)


__all__ = ["generate_edge_ai_workload", "WorkloadGenerator", "_piecewise_levels"]
