"""Strongly-typed configuration system using dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class EnvironmentConfig:
    """Environment configuration: hardware parameters and physical properties.
    
    This configuration represents the fixed physical environment in which
    experiments are conducted. It includes hardware parameters and physical
    cooling coefficients that remain constant across different experiments.
    """
    
    physics_gamma: float  # Physical cooling coefficient
    input_dim: int  # Workload feature dimension
    hidden_dim: int  # Network hidden layer width (for PiNN models)
    T_amb: float  # Ambient temperature (normalized)
    Ts: float  # Sampling time [s] (hardware constraint)
    
    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        assert self.physics_gamma > 0, "Physical cooling coefficient must be greater than 0"
        assert self.input_dim > 0, "Input dimension must be greater than 0"
        assert self.hidden_dim > 0, "Hidden layer width must be greater than 0"
        assert 0 <= self.T_amb <= 1, "Ambient temperature must be in [0, 1]"
        assert self.Ts > 0, "Sampling time must be greater than 0"


@dataclass
class ExperimentConfig:
    """Experiment configuration: simulation parameters that vary per experiment.
    
    This configuration represents the experimental settings that change
    between different simulation runs, such as duration, target temperature,
    and initial conditions.
    """
    
    duration: float  # Simulation duration [s]
    T_ref: float  # Reference temperature (normalized) - target temperature
    T0: float  # Initial temperature (normalized)
    
    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        assert self.duration > 0, "Simulation duration must be greater than 0"
        assert 0 <= self.T_ref <= 1, "Reference temperature must be in [0, 1]"
        assert 0 <= self.T0 <= 1, "Initial temperature must be in [0, 1]"


@dataclass
class SimulationConfig:
    """Simulation configuration parameters (legacy compatibility).
    
    This class combines environment and experiment configs for backward
    compatibility. New code should use EnvironmentConfig and ExperimentConfig
    separately for better separation of concerns.
    """
    
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
    
    @classmethod
    def from_configs(
        cls,
        env_config: EnvironmentConfig,
        exp_config: ExperimentConfig,
    ) -> "SimulationConfig":
        """Create SimulationConfig from separate environment and experiment configs."""
        return cls(
            Ts=env_config.Ts,
            duration=exp_config.duration,
            T_amb=env_config.T_amb,
            T_ref=exp_config.T_ref,
            T0=exp_config.T0,
        )
    
    def to_environment_config(self) -> EnvironmentConfig:
        """Extract environment config from simulation config."""
        return EnvironmentConfig(
            physics_gamma=0.1,  # Default, should be set separately
            input_dim=8,  # Default, should be set separately
            hidden_dim=64,  # Default, should be set separately
            T_amb=self.T_amb,
            Ts=self.Ts,
        )
    
    def to_experiment_config(self) -> ExperimentConfig:
        """Extract experiment config from simulation config."""
        return ExperimentConfig(
            duration=self.duration,
            T_ref=self.T_ref,
            T0=self.T0,
        )


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


__all__ = [
    "EnvironmentConfig",
    "ExperimentConfig",
    "SimulationConfig",
    "ModelConfig",
    "ControllerConfig",
]
