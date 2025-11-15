"""Tri-Reflex Neural Control (TRiNC) implementation.

TRiNC emulates three neural reflex pathways to regulate cooling effort:

- P gate reacts to magnitude deviations beyond a proportional threshold ``tau_p``.
- H gate integrates error with leakage and fires when habituation threshold ``theta_h``
  is exceeded, representing sustained offset compensation.
- S gate responds to rapid changes in error using a derivative threshold ``tau_s``.

Each gate emits {-1, 0, +1} events that are integrated through a synaptic decay with
factor ``rho``. The resulting control sequence delivers sparse pulses that quickly
counteract bursts yet remain energy efficient during steady phases.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TRINCController:
    """Event-driven controller leveraging tri-reflex neural modulation."""

    tau_p: float
    tau_s: float
    theta_h: float
    alpha_h: float
    w_h: float
    rho: float
    a_p: float
    a_h: float
    a_s: float
    refractory_steps: int = 0
    u_min: float = 0.0
    u_max: float = 1.0

    def __post_init__(self) -> None:
        self.h_state = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.refractory = 0
        self.event_log = {"gP": [], "gH": [], "gS": []}

    def reset(self) -> None:
        self.h_state = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.refractory = 0
        self.event_log = {"gP": [], "gH": [], "gS": []}

    def compute_control(self, error: float) -> float:
        """Compute the next cooling command based on tri-reflex logic."""

        delta_error = error - self.prev_error

        # P gate: magnitude-based reflex
        g_p = 0.0
        if abs(error) > self.tau_p:
            g_p = float(1.0 if error > 0 else -1.0)

        # H gate: leaky accumulator
        h_temp = (1.0 - self.alpha_h) * self.h_state + self.w_h * error
        g_h = 0.0
        if h_temp >= self.theta_h:
            g_h = 1.0
            h_temp -= self.theta_h
        elif h_temp <= -self.theta_h:
            g_h = -1.0
            h_temp += self.theta_h
        self.h_state = h_temp

        # S gate: surprise reflex for rapid changes
        g_s = 0.0
        if abs(delta_error) > self.tau_s:
            g_s = float(1.0 if delta_error > 0 else -1.0)

        # Optional refractory behavior to avoid chatter
        if self.refractory > 0:
            g_p = 0.0
            g_h = 0.0
            g_s = 0.0
            self.refractory -= 1
        elif g_p != 0.0 or g_h != 0.0 or g_s != 0.0:
            self.refractory = self.refractory_steps

        # Synaptic integration with decay
        control = (
            self.rho * self.prev_output
            + self.a_p * g_p
            + self.a_h * g_h
            + self.a_s * g_s
        )

        control = max(self.u_min, min(self.u_max, control))
        self.prev_error = error
        self.prev_output = control

        self.event_log["gP"].append(g_p)
        self.event_log["gH"].append(g_h)
        self.event_log["gS"].append(g_s)
        return control
