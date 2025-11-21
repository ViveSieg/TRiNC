"""Performance and energy metrics for thermal control simulations."""

from __future__ import annotations

import time
from typing import Dict, Optional, Callable

import numpy as np

from ..controllers.base import BaseController


def compute_metrics(
    time: np.ndarray,
    temperature: np.ndarray,
    control: np.ndarray,
    T_ref: float,
    event_flags: Optional[np.ndarray] = None,
    control_events: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute tracking accuracy, energy proxies, and sparsity statistics."""

    error = T_ref - temperature
    overshoot = np.max(np.maximum(temperature - T_ref, 0.0))

    settling_idx = np.where(np.abs(error) <= 0.02 * T_ref)[0]
    if settling_idx.size > 0:
        settling_time = time[settling_idx[0]]
    else:
        settling_time = time[-1]

    steady_state_error = float(np.mean(error[int(0.9 * len(error)) :]))

    Ts = time[1] - time[0] if len(time) > 1 else 0.0
    E1 = float(np.sum(np.abs(control)) * Ts)
    E2 = float(np.sum(control**2) * Ts)

    if control_events is None:
        du = np.diff(control, prepend=control[0])
        N_du = int(np.sum(np.abs(du) > 0.02))
    else:
        N_du = int(np.sum(control_events))

    N_events = int(np.sum(np.abs(event_flags))) if event_flags is not None else N_du

    return {
        "overshoot": float(overshoot),
        "settling_time": float(settling_time),
        "steady_state_error": steady_state_error,
        "E1": E1,
        "E2": E2,
        "N_events": N_events,
        "N_du": N_du,
    }


def estimate_flops(controller: BaseController, algo_name: str) -> int:
    """Estimate FLOPs (Floating Point Operations) per control computation.
    
    This provides a rough estimate of computational complexity for embedded
    system deployment considerations.
    
    Parameters
    ----------
    controller : BaseController
        Controller instance
    algo_name : str
        Algorithm name
    
    Returns
    -------
    int
        Estimated FLOPs per control computation
    """
    # FLOPs estimates based on typical operations:
    # - Addition/Subtraction: 1 FLOP
    # - Multiplication: 1 FLOP
    # - Division: 1 FLOP
    # - Comparison: 0.5 FLOP (approximate)
    # - Conditional logic: minimal overhead
    
    if algo_name == "pid":
        # PID: P + I + D terms
        # P: 1 mult, I: 1 mult + 1 add, D: 1 sub + 1 div + 1 mult
        # Total: ~6 FLOPs
        return 6
    elif algo_name == "pid_deadzone":
        # Same as PID + deadzone check: ~7 FLOPs
        return 7
    elif algo_name == "event_pid":
        # Event check: 2 abs + 2 comparisons = ~3 FLOPs
        # PID computation when triggered: ~6 FLOPs
        # Average: ~4-5 FLOPs (depends on event rate)
        return 5
    elif algo_name == "lif_snn":
        # Membrane update: 2 mults + 1 add = 3 FLOPs
        # Decay: 1 mult = 1 FLOP
        # Threshold check: 2 comparisons = 1 FLOP
        # Total: ~5 FLOPs
        return 5
    elif algo_name == "trinc":
        # P gate: 2 comparisons = 1 FLOP
        # H gate: 2 mults + 1 add + 2 comparisons + conditional = ~6 FLOPs
        # S gate: 1 sub + 2 comparisons = ~2 FLOPs
        # Refractory: 3 comparisons = ~1.5 FLOPs
        # Synaptic integration: 3 mults + 2 adds = 5 FLOPs
        # Clipping: 2 comparisons = 1 FLOP
        # Total: ~16-17 FLOPs
        return 17
    else:
        # Default estimate for unknown controllers
        return 10


def measure_inference_latency(
    controller: BaseController,
    n_warmup: int = 10,
    n_trials: int = 1000,
) -> float:
    """Measure average inference latency (in seconds) for controller.
    
    This measures the actual wall-clock time for compute_control calls,
    which is critical for real-time embedded systems.
    
    Parameters
    ----------
    controller : BaseController
        Controller instance
    n_warmup : int
        Number of warmup iterations to avoid cold start effects
    n_trials : int
        Number of timing trials
    
    Returns
    -------
    float
        Average inference latency in seconds
    """
    # Warmup
    for _ in range(n_warmup):
        controller.compute_control(0.1)
    
    # Measure
    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        controller.compute_control(0.1)
        end = time.perf_counter()
        latencies.append(end - start)
    
    return float(np.mean(latencies))


def compute_complexity_metrics(
    controller: BaseController,
    algo_name: str,
    horizon: int,
) -> Dict[str, float]:
    """Compute computational complexity metrics.
    
    Parameters
    ----------
    controller : BaseController
        Controller instance
    algo_name : str
        Algorithm name
    horizon : int
        Simulation horizon (number of time steps)
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - flops_per_step: FLOPs per control computation
        - total_flops: Total FLOPs for entire simulation
        - inference_latency_ms: Average inference latency in milliseconds
        - total_compute_time_ms: Estimated total compute time in milliseconds
    """
    flops_per_step = estimate_flops(controller, algo_name)
    total_flops = flops_per_step * horizon
    
    # Measure actual latency
    latency_s = measure_inference_latency(controller)
    latency_ms = latency_s * 1000.0
    total_compute_time_ms = latency_ms * horizon
    
    return {
        "flops_per_step": float(flops_per_step),
        "total_flops": float(total_flops),
        "inference_latency_ms": latency_ms,
        "total_compute_time_ms": total_compute_time_ms,
    }
