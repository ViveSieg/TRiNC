"""Configuration presets for simulation, models, and controllers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from .schema import (
    SimulationConfig,
    ModelConfig,
    ControllerConfig,
    EnvironmentConfig,
    ExperimentConfig,
)


def get_default_environment_config() -> EnvironmentConfig:
    """Return default environment configuration.
    
    Returns
    -------
    EnvironmentConfig
        Default environment parameters (hardware and physical properties).
    """
    return EnvironmentConfig(
        physics_gamma=0.1,
        input_dim=8,
        hidden_dim=64,
        T_amb=0.4,
        Ts=0.1,
    )


def get_default_experiment_config() -> ExperimentConfig:
    """Return default experiment configuration.
    
    Returns
    -------
    ExperimentConfig
        Default experiment parameters (simulation settings).
    """
    return ExperimentConfig(
        duration=22.0,
        T_ref=0.62,
        T0=0.55,
    )


def get_default_simulation_config() -> SimulationConfig:
    """Return default simulation configuration.
    
    Returns
    -------
    SimulationConfig
        Default simulation parameters for edge-AI thermal control.
    """
    return SimulationConfig(
        Ts=0.1,
        duration=22.0,
        T_amb=0.4,
        T_ref=0.62,
        T0=0.55,
    )


def get_default_model_config(input_dim: int = 8) -> ModelConfig:
    """Return default model configuration.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension (default: 8)
    
    Returns
    -------
    ModelConfig
        Default model parameters for PiNN thermal model.
    """
    return ModelConfig(
        physics_gamma=0.1,
        input_dim=input_dim,
        hidden_dim=64,
    )


def get_default_controller_params() -> Dict[str, Dict[str, Any]]:
    """Return default controller parameters.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping controller names to their default parameters.
    """
    return {
        "pid": {"Kp": 3.1, "Ki": 0.9, "Kd": 0.24},
        "pid_deadzone": {"Kp": 3.2, "Ki": 0.8, "Kd": 0.25, "deadband": 0.02},
        "event_pid": {
            "Kp": 2.8,
            "Ki": 0.75,
            "Kd": 0.22,
            "error_threshold": 0.015,
            "delta_threshold": 0.01,
            "decay": 0.015,
        },
        "lif_snn": {
            "leak": 0.02,
            "threshold": 0.065,
            "reset_value": 0.012,
            "pulse_magnitude": 0.24,
            "decay": 0.07,
        },
        "trinc": {
            "tau_p": 0.032,
            "tau_s": 0.034,
            "theta_h": 0.06,
            "alpha_h": 0.032,
            "w_h": 0.26,
            "rho": 0.9,
            "a_p": 0.31,
            "a_h": 0.13,
            "a_s": 0.29,
            "refractory_steps": 8,
            "base_cooling": 0.32,
        },
    }


def get_search_space(algo_name: str) -> Dict[str, tuple[float, float]]:
    """Return parameter search space for a specific controller.
    
    Defines wide, fair ranges for all baselines to prevent "weak baseline" criticism.
    
    Parameters
    ----------
    algo_name : str
        Controller name
    
    Returns
    -------
    Dict[str, tuple[float, float]]
        Parameter search space with (min, max) bounds
    
    Raises
    ------
    KeyError
        If controller name is not supported
    """
    search_spaces = {
        "pid": {
            "Kp": (0.5, 8.0),  # Wide range for proportional gain
            "Ki": (0.0, 3.0),  # Integral gain
            "Kd": (0.0, 2.0),  # Derivative gain
        },
        "pid_deadzone": {
            "Kp": (0.5, 8.0),
            "Ki": (0.0, 3.0),
            "Kd": (0.0, 2.0),
            "deadband": (0.0, 0.12),  # Deadzone width
        },
        "event_pid": {
            "Kp": (0.5, 7.0),
            "Ki": (0.0, 2.5),
            "Kd": (0.0, 1.5),
            "error_threshold": (0.003, 0.08),  # Event triggering threshold
            "delta_threshold": (0.002, 0.08),
            "decay": (0.001, 0.08),  # Output decay rate
        },
        "lif_snn": {
            "leak": (0.001, 0.12),  # Membrane leak rate
            "threshold": (0.01, 0.25),  # Spike threshold
            "reset_value": (0.0, 0.15),
            "pulse_magnitude": (0.05, 0.7),  # Control pulse amplitude
            "decay": (0.01, 0.3),  # Output decay
        },
        "trinc": {
            "tau_p": (0.005, 0.12),  # Proportional gate threshold
            "tau_s": (0.005, 0.12),  # Surprise gate threshold
            "theta_h": (0.02, 0.18),  # Habituation threshold
            "alpha_h": (0.005, 0.12),  # Habituation leak
            "w_h": (0.05, 0.5),  # Habituation weight
            "rho": (0.4, 0.98),  # Synaptic decay
            "a_p": (0.05, 0.6),  # Proportional gate amplitude
            "a_h": (0.05, 0.35),  # Habituation gate amplitude
            "a_s": (0.05, 0.6),  # Surprise gate amplitude
            "refractory_steps": (2, 20),
            "base_cooling": (0.05, 0.6),
        },
    }
    
    if algo_name not in search_spaces:
        raise KeyError(
            f"Controller '{algo_name}' not supported. "
            f"Available: {list(search_spaces.keys())}"
        )
    
    return search_spaces[algo_name]


def get_tuning_search_spaces() -> Dict[str, Dict[str, tuple[float, float]]]:
    """Return parameter search spaces for all controllers.
    
    Returns
    -------
    Dict[str, Dict[str, tuple[float, float]]]
        Dictionary mapping controller names to their parameter bounds.
    """
    return {
        name: get_search_space(name)
        for name in ["pid", "pid_deadzone", "event_pid", "lif_snn", "trinc"]
    }


def get_golden_quartet_scenarios() -> list[Dict[str, Any]]:
    """Return the Golden Quartet scenarios for comprehensive evaluation.
    
    Returns
    -------
    list[Dict[str, Any]]
        List of scenario dictionaries with name, profile, description, and overrides.
    """
    return [
        {
            "name": "real_trace_replay",
            "profile": "real_trace_replay",
            "description": "Primary: Random sample from TPU test set. Used for optimization.",
            "sim_overrides": {},
        },
        {
            "name": "step_response",
            "profile": "step_response",
            "description": "Synthetic step for transient analysis (Rise Time, Overshoot).",
            "sim_overrides": {"duration": 15.0},
        },
        {
            "name": "sine_tracking",
            "profile": "sine_tracking",
            "description": "Synthetic sine wave for dynamic tracking (ITAE).",
            "sim_overrides": {"duration": 30.0, "T_ref": 0.6},
        },
        {
            "name": "steady_state",
            "profile": "steady_state",
            "description": "Constant high-load for efficiency (Energy).",
            "sim_overrides": {"duration": 20.0},
        },
    ]


__all__ = [
    "get_default_environment_config",
    "get_default_experiment_config",
    "get_default_simulation_config",
    "get_default_model_config",
    "get_default_controller_params",
    "get_search_space",
    "get_tuning_search_spaces",
    "get_golden_quartet_scenarios",
]
