"""Default simulation, controller parameters, and tuning settings."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List


def default_simulation_params() -> Dict[str, float]:
    """Return default plant and simulation parameters.

    Values reflect a compact edge-AI accelerator with a 10 Hz control
    loop. Temperatures are normalized to ``[0, 1]`` where 0 ≈ 40 °C and
    1 ≈ 90 °C.
    """

    return {
        "tau_th": 3.0,  # thermal time constant [s]
        "K_heat": 1.0,  # heating gain at full workload
        "K_cool": 0.6,  # cooling gain at maximum control effort
        "Ts": 0.1,  # sampling time [s]
        "T_ref": 0.62,  # reference temperature (normalized)
        "T0": 0.55,  # initial temperature
        "duration": 22.0,  # simulation horizon [s]
    }


def default_controller_params() -> Dict[str, Dict[str, float]]:
    """Return tuned parameters for each controller.

    These values are written back by the tuning framework when new
    optima are found. They serve as the single source of truth for the
    experiments and figures accompanying the paper.
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


def tuning_search_spaces() -> Dict[str, Dict[str, tuple[float, float]]]:
    """Boundaries for automatic hyper-parameter tuning."""

    return {
        "pid": {"Kp": (0.8, 6.0), "Ki": (0.0, 2.0), "Kd": (0.0, 1.0)},
        "pid_deadzone": {
            "Kp": (0.8, 6.0),
            "Ki": (0.0, 2.0),
            "Kd": (0.0, 1.0),
            "deadband": (0.0, 0.08),
        },
        "event_pid": {
            "Kp": (0.8, 5.5),
            "Ki": (0.0, 1.5),
            "Kd": (0.0, 0.8),
            "error_threshold": (0.005, 0.05),
            "delta_threshold": (0.003, 0.05),
            "decay": (0.001, 0.05),
        },
        "lif_snn": {
            "leak": (0.005, 0.08),
            "threshold": (0.02, 0.15),
            "reset_value": (0.0, 0.08),
            "pulse_magnitude": (0.05, 0.5),
            "decay": (0.01, 0.2),
        },
        "trinc": {
            "tau_p": (0.01, 0.08),
            "tau_s": (0.01, 0.08),
            "theta_h": (0.03, 0.12),
            "alpha_h": (0.01, 0.08),
            "w_h": (0.1, 0.4),
            "rho": (0.55, 0.96),
            "a_p": (0.1, 0.5),
            "a_h": (0.08, 0.25),
            "a_s": (0.1, 0.45),
            "refractory_steps": (3, 14),
            "base_cooling": (0.08, 0.5),
        },
    }


def default_tuning_settings() -> Dict:
    """Return generic tuning hyper-parameters and evaluation weights."""

    return {
        "max_iters": 18,
        "patience": 5,
        "rng_seed": 7,
        "explore_prob": 0.35,
        "weights": {
            "overshoot": 6.0,
            "steady_state": 4.0,
            "iae": 1.5,
            "energy": 0.8,
            "event_rate": 0.08,
        },
        "tuning_workloads": [
            "baseline",
            "burst_chain",
            "thermal_shock",
            "tracking_drill",
            "thermal_resilience",
        ],
    }


def default_scenarios() -> List[Dict]:
    """Benchmark scenarios used by ``run_all_algorithms``."""

    return [
        {
            "name": "baseline_burst",
            "profile": "baseline",
            "description": "Low→medium→low duty cycle representative of steady inference.",
            "sim_overrides": {},
        },
        {
            "name": "burst_chain",
            "profile": "burst_chain",
            "description": "Repeated bursts to probe overshoot and recovery.",
            "sim_overrides": {"duration": 24.0},
        },
        {
            "name": "idle_recovery",
            "profile": "idle_recovery",
            "description": "Heavy batch completes then the system idles and resumes mid-load.",
            "sim_overrides": {"duration": 25.0, "T_ref": 0.6},
        },
        {
            "name": "noisy_lab",
            "profile": "noisy_lab",
            "description": "Baseline workload with sinusoidal and Gaussian jitter.",
            "sim_overrides": {"duration": 22.0},
        },
        {
            "name": "thermal_shock",
            "profile": "thermal_shock",
            "description": "Tanh-like ramp with a sharp heat burst.",
            "sim_overrides": {"duration": 24.0},
        },
        {
            "name": "tracking_drill",
            "profile": "tracking_drill",
            "description": "Periodic waveform for tracking agility experiments.",
            "sim_overrides": {"duration": 26.0, "T_ref": 0.58},
        },
        {
            "name": "progressive_ramp",
            "profile": "progressive_ramp",
            "description": "Gradually rising traffic with a mid-run step.",
            "sim_overrides": {"duration": 26.0},
        },
        {
            "name": "sensor_edge",
            "profile": "sensor_edge",
            "description": "Long light-load stretches with periodic medium spikes.",
            "sim_overrides": {"duration": 24.0},
        },
        {
            "name": "cooling_loss",
            "profile": "cooling_loss",
            "description": "Simulated deterioration in cooling conditions over time.",
            "sim_overrides": {"duration": 24.0, "K_cool": 0.55},
        },
        {
            "name": "thermal_resilience",
            "profile": "thermal_resilience",
            "description": "Composite stress test mixing bursts, ramps, and idle periods.",
            "sim_overrides": {"duration": 25.0},
        },
    ]


def save_tuned_params_json(params: Dict[str, Dict[str, float]], path: Path) -> None:
    """Persist tuned parameters to a JSON file for reproducibility."""

    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def apply_params_override(base: Dict[str, Dict[str, float]], override: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Shallow merge controller parameter dictionaries."""

    merged = deepcopy(base)
    for name, params in override.items():
        merged.setdefault(name, {}).update(params)
    return merged


__all__ = [
    "default_simulation_params",
    "default_controller_params",
    "tuning_search_spaces",
    "default_tuning_settings",
    "default_scenarios",
    "save_tuned_params_json",
    "apply_params_override",
]
