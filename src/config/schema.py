"""Strongly-typed configuration system using dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""
    
    Ts: float  # Sampling time [s]
    duration: float  # Simulation duration [s]
    T_amb: float  # Ambient temperature (normalized)
    T_ref: float  # Reference temperature (normalized)
    T0: float  # Initial temperature (normalized)
    
    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        assert self.Ts > 0, "Sampling time must be greater than 0"
        assert self.duration > 0, "Simulation duration must be greater than 0"
        assert 0 <= self.T_amb <= 1, "Ambient temperature must be in [0, 1]"
        assert 0 <= self.T_ref <= 1, "Reference temperature must be in [0, 1]"
        assert 0 <= self.T0 <= 1, "Initial temperature must be in [0, 1]"


@dataclass
class ModelConfig:
    """Thermal model configuration parameters."""
    
    physics_gamma: float  # Physical cooling coefficient
    input_dim: int  # Workload feature dimension
    hidden_dim: int  # Network hidden layer width
    
    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        assert self.physics_gamma > 0, "Physical cooling coefficient must be greater than 0"
        assert self.input_dim > 0, "Input dimension must be greater than 0"
        assert self.hidden_dim > 0, "Hidden layer width must be greater than 0"


@dataclass
class ControllerConfig:
    """Controller configuration parameters (using dict for dynamic parameters)."""
    
    name: str  # Controller name
    params: Dict[str, Any]  # Controller parameter dictionary
    
    def __post_init__(self) -> None:
        """Validate parameters."""
        assert self.name, "Controller name cannot be empty"
        assert isinstance(self.params, dict), "Parameters must be a dictionary"


__all__ = ["SimulationConfig", "ModelConfig", "ControllerConfig"]
