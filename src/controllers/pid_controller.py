"""Discrete PID controller with anti-windup for thermal regulation."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseController
from .registry import register_controller


@register_controller("pid")
@dataclass
class PIDController(BaseController):
    """Conventional PID controller with conditional integration."""

    Kp: float
    Ki: float
    Kd: float
    Ts: float
    u_min: float = 0.0
    u_max: float = 1.0

    def __post_init__(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def reset(self) -> None:
        """Reset the controller state."""

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def compute_control(self, error: float) -> float:
        """Compute the control action for the current error."""

        proportional = self.Kp * error
        derivative = self.Kd * (error - self.prev_error) / self.Ts

        # Conditional integration to mitigate windup.
        tentative = proportional + self.integral + derivative
        if self.u_min < tentative < self.u_max:
            self.integral += self.Ki * error * self.Ts
        integral_term = self.integral

        output = proportional + integral_term + derivative
        output = max(self.u_min, min(self.u_max, output))

        self.prev_error = error
        self.prev_output = output
        return output
