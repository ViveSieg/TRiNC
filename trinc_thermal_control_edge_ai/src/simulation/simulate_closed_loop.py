"""Closed-loop simulation utilities."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from src.models.thermal_model import ThermalModel


def run_closed_loop(
    model: ThermalModel,
    controller,
    workload: np.ndarray,
    T_ref: float,
    Ts: float,
    T0: float,
    algo_name: str,
) -> Dict[str, Any]:
    """Run a closed-loop simulation for a controller.

    Parameters
    ----------
    model: ThermalModel
        Thermal plant with ``step`` method.
    controller: object
        Controller with ``reset`` and ``compute_control`` methods.
    workload: np.ndarray
        Disturbance sequence ``d[k]``.
    T_ref: float
        Target temperature.
    Ts: float
        Sampling period.
    T0: float
        Initial temperature.
    algo_name: str
        Name of the controller (for logging).

    Returns
    -------
    Dict[str, Any]
        Dictionary with time series arrays and event indicators if available.
    """

    horizon = len(workload)
    time = np.arange(horizon) * Ts
    temperature = np.zeros(horizon)
    control_signal = np.zeros(horizon)
    error = np.zeros(horizon)

    model.reset(T0)
    controller.reset()

    T_k = T0
    u_k = 0.0

    extra = {}
    if hasattr(controller, "event_log"):
        extra = {key: [] for key in controller.event_log}

    for k in range(horizon):
        error[k] = T_ref - T_k
        u_k = controller.compute_control(error[k])
        control_signal[k] = u_k
        temperature[k] = T_k

        if extra:
            for key in extra:
                extra[key].append(controller.event_log[key][k])

        T_k = model.step(T_k, u_k, workload[k])

    results: Dict[str, Any] = {
        "time": time,
        "temperature": temperature,
        "control": control_signal,
        "error": error,
        "workload": workload,
        "algo_name": algo_name,
    }
    results.update(extra)
    return results
