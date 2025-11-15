"""Performance and energy metrics for thermal control simulations."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_metrics(
    time: np.ndarray,
    temperature: np.ndarray,
    control: np.ndarray,
    T_ref: float,
    event_flags: Optional[np.ndarray] = None,
    control_events: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute tracking accuracy, energy proxies, and sparsity statistics."""

    error = T_ref - temperature
    overshoot = np.max(np.maximum(temperature - T_ref, 0.0))

    settling_idx = np.where(np.abs(error) <= 0.02 * T_ref)[0]
    if settling_idx.size > 0:
        settling_time = time[settling_idx[0]]
    else:
        settling_time = time[-1]

    steady_state_error = float(np.mean(error[int(0.9 * len(error)) :]))

    Ts = time[1] - time[0] if len(time) > 1 else 0.0
    E1 = float(np.sum(np.abs(control)) * Ts)
    E2 = float(np.sum(control**2) * Ts)

    if control_events is None:
        du = np.diff(control, prepend=control[0])
        N_du = int(np.sum(np.abs(du) > 0.02))
    else:
        N_du = int(np.sum(control_events))

    N_events = int(np.sum(np.abs(event_flags))) if event_flags is not None else N_du

    return {
        "overshoot": float(overshoot),
        "settling_time": float(settling_time),
        "steady_state_error": steady_state_error,
        "E1": E1,
        "E2": E2,
        "N_events": N_events,
        "N_du": N_du,
    }
