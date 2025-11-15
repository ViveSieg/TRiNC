"""Thermal plant model for edge AI chip hotspot control."""

from __future__ import annotations

import math


class ThermalModel:
    """First-order discrete-time thermal model.

    The continuous-time dynamics are given by:

    .. math:: \tau_{th} \dot T(t) = -T(t) + K_{cool} u(t) + d(t)

    where :math:`T` is the normalized hotspot temperature, :math:`u` is the
    normalized cooling effort (e.g., fan PWM duty cycle), and :math:`d` is the
    workload-induced heating disturbance. Discretizing with zero-order hold and
    sample time ``Ts`` yields:

    .. math:: T[k+1] = a T[k] + b u[k] + d[k]

    with ``a = exp(-Ts / tau_th)`` and ``b = K_cool * (1 - a)``.
    """

    def __init__(self, tau_th: float, K_cool: float, Ts: float) -> None:
        self.tau_th = tau_th
        self.K_cool = K_cool
        self.Ts = Ts
        self.a = math.exp(-Ts / tau_th)
        self.b = K_cool * (1.0 - self.a)
        self.temperature = 0.0

    def reset(self, initial_temperature: float) -> None:
        """Reset the internal temperature state."""

        self.temperature = initial_temperature

    def step(self, temperature: float | None, control_signal: float, workload: float) -> float:
        """Propagate the plant by one sample.

        Parameters
        ----------
        temperature: float or None
            Current hotspot temperature ``T_k``. If ``None`` the internal state is used.
        control_signal: float
            Cooling input ``u_k`` constrained to ``[0, 1]``.
        workload: float
            Disturbance term ``d_k`` representing heat generation.

        Returns
        -------
        float
            Next temperature ``T_{k+1}``.
        """

        T_k = self.temperature if temperature is None else temperature
        next_temperature = self.a * T_k + self.b * control_signal + workload
        self.temperature = next_temperature
        return next_temperature
