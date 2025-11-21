"""Leaky integrate-and-fire inspired controller."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseController
from .registry import register_controller


@register_controller("lif_snn")
@dataclass
class LIFSpikingController(BaseController):
    """Emit control pulses when the integrated error crosses a threshold."""

    leak: float
    threshold: float
    reset_value: float
    pulse_magnitude: float
    decay: float
    u_min: float = 0.0
    u_max: float = 1.0

    def __post_init__(self) -> None:
        self.membrane = 0.0
        self.prev_output = 0.0
        self.spike_count = 0

    def reset(self) -> None:
        self.membrane = 0.0
        self.prev_output = 0.0
        self.spike_count = 0

    def compute_control(self, error: float) -> float:
        """Integrate error and generate sparse pulses."""

        self.membrane = (1.0 - self.leak) * self.membrane + error
        control = (1.0 - self.decay) * self.prev_output

        if self.membrane > self.threshold:
            control += self.pulse_magnitude
            self.membrane = self.reset_value
            self.spike_count += 1
        elif self.membrane < -self.threshold:
            control -= self.pulse_magnitude
            self.membrane = -self.reset_value
            self.spike_count += 1

        control = max(self.u_min, min(self.u_max, control))
        self.prev_output = control
        return control
