"""Thermal plant model for edge-AI hotspot control.

The plant is intentionally simple and fully normalized to keep the
closed-loop simulations interpretable and reproducible. Temperature and
inputs are expressed in the unit interval and map to a physical
temperature span of roughly 40–90 °C (see class docstring for details).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalModel:
    """First-order discrete-time thermal model.

    The normalized temperature evolves according to

    .. math:: T_{k+1} = a T_k + (1 - a) (K_{heat} w_k - K_{cool} u_k),

    with ``a = exp(-T_s / tau_th)``. All variables are normalized to the
    unit interval ``[0, 1]``:

    * ``T_k`` is the hotspot temperature (0 ↔ ~40 °C, 1 ↔ ~90 °C).
    * ``w_k`` is the workload-induced heat input.
    * ``u_k`` is the cooling command (e.g., normalized fan duty cycle).

    Default parameters are representative of a few-watt edge AI
    accelerator module:

    * ``tau_th = 3 s`` places the thermal time constant within the
      1–5 s range observed on compact heat-spreaders.
    * ``K_heat = 1.0`` means full workload without cooling would settle
      close to ``T = 1``.
    * ``K_cool = 0.6`` means full cooling under full load drags the
      steady-state down to roughly ``T ≈ 0.4``.
    * ``T_s = 0.1 s`` matches a 10 Hz control loop often used on
      embedded monitoring threads.

    Physical mapping (commentary only):
        A normalized temperature ``T`` can be mapped to degrees Celsius
        using ``T_phys = 40 + T * (90 - 40)``. This is not implemented in
        code to keep the simulation algebra simple, but serves as a
        reference when interpreting plots.
    """

    tau_th: float
    K_heat: float
    K_cool: float
    Ts: float

    def __post_init__(self) -> None:
        self._a = math.exp(-self.Ts / self.tau_th)
        self._one_minus_a = 1.0 - self._a
        self._temperature = 0.0

    def reset(self, initial_temperature: float) -> None:
        """Reset internal temperature state ``T_k``."""

        self._temperature = float(initial_temperature)

    def step(self, temperature: float | None, control_signal: float, workload: float) -> float:
        """Propagate the plant by one sample.

        Parameters
        ----------
        temperature : float or None
            If provided, overrides the internal ``T_k`` before updating.
        control_signal : float
            Cooling input ``u_k`` in ``[0, 1]``.
        workload : float
            Normalized workload ``w_k`` in ``[0, 1]``.

        Returns
        -------
        float
            Updated temperature ``T_{k+1}``.
        """

        if temperature is not None:
            self._temperature = float(temperature)

        u_k = np.clip(control_signal, 0.0, 1.0)
        w_k = np.clip(workload, 0.0, 1.0)

        next_temperature = self._a * self._temperature + self._one_minus_a * (
            self.K_heat * w_k - self.K_cool * u_k
        )
        self._temperature = float(next_temperature)
        return self._temperature


__all__ = ["ThermalModel"]
