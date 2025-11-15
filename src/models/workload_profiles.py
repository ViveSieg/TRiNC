"""Workload profiles for thermal control experiments."""

from __future__ import annotations

import numpy as np


def generate_edge_ai_workload(Ts: float, horizon: int) -> np.ndarray:
    """Generate a piecewise workload sequence favoring event-driven control.

    Parameters
    ----------
    Ts: float
        Sampling period [s].
    horizon: int
        Number of simulation steps.

    Returns
    -------
    np.ndarray
        Disturbance sequence ``d[k]`` normalized to ``[0, 1]``.

    The schedule contains three phases:
    1. 0--5 s low load (light inference).
    2. 5--12 s burst load (intensive batched processing).
    3. 12--20 s moderate load (post-processing / idle).
    """

    time = np.arange(horizon) * Ts
    workload = np.zeros_like(time)

    low_idx = time < 5.0
    burst_idx = (time >= 5.0) & (time < 12.0)
    medium_idx = time >= 12.0

    workload[low_idx] = 0.022
    workload[burst_idx] = 0.055
    workload[medium_idx] = 0.034

    # Add small random jitter to mimic stochastic workloads without biasing algorithms.
    rng = np.random.default_rng(seed=42)
    workload += rng.normal(loc=0.0, scale=0.003, size=workload.shape)
    return np.clip(workload, 0.0, 0.08)
