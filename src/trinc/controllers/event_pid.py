"""Event-triggered PID controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EventPIDController:
    """PID controller that updates only when error thresholds are exceeded."""

    Kp: float
    Ki: float
    Kd: float
    Ts: float
    error_threshold: float
    delta_threshold: float
    decay: float = 0.0
    u_min: float = 0.0
    u_max: float = 1.0

    def __post_init__(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.event_count = 0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.event_count = 0

    def compute_control(self, error: float) -> float:
        """Return control action, updating only on significant events."""

        error_change = error - self.prev_error
        triggered = abs(error) > self.error_threshold or abs(error_change) > self.delta_threshold
        if triggered:
            proportional = self.Kp * error
            derivative = self.Kd * error_change / self.Ts
            self.integral += self.Ki * error * self.Ts
            control = proportional + self.integral + derivative
            control = max(self.u_min, min(self.u_max, control))
            self.prev_output = control
            self.event_count += 1
        else:
            control = max(self.u_min, min(self.u_max, (1.0 - self.decay) * self.prev_output))

        self.prev_error = error
        return control
