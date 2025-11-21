"""PID controller with deadzone integration for sparse actuation."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseController
from .registry import register_controller


@register_controller("pid_deadzone")
@dataclass
class PIDDeadzoneController(BaseController):
    """PID variant that suppresses small-error responses."""

    Kp: float
    Ki: float
    Kd: float
    Ts: float
    deadband: float
    u_min: float = 0.0
    u_max: float = 1.0

    def __post_init__(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_control(self, error: float) -> float:
        """Compute PID action with deadzone on the integral term."""

        proportional = self.Kp * error
        derivative = self.Kd * (error - self.prev_error) / self.Ts

        if abs(error) > self.deadband:
            self.integral += self.Ki * error * self.Ts
        else:
            self.integral *= 0.9  # mild decay when inside deadband

        output = proportional + self.integral + derivative
        output = max(self.u_min, min(self.u_max, output))
        self.prev_error = error
        return output
