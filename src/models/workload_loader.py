"""Workload loader and scenario adaptation: supports Golden Quartet scenarios."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .pinn_thermal import load_data

logger = logging.getLogger(__name__)

# Golden Quartet scenarios
GOLDEN_QUARTET = ["real_trace_replay", "step_response", "sine_tracking", "steady_state"]


def _piecewise_levels(time: np.ndarray, levels: list[float], breakpoints: list[int]) -> np.ndarray:
    """Build piecewise-constant load over the given time indices."""
    segments = np.zeros_like(time, dtype=float)
    idx_start = 0
    for level, bp in zip(levels, breakpoints):
        mask = (time >= idx_start) & (time < bp)
        segments[mask] = level
        idx_start = bp
    segments[time >= idx_start] = levels[len(breakpoints)]
    return segments


def generate_synthetic_signal(
    Ts: float,
    horizon: int,
    profile: str,
    jitter_seed: int | None = None,
) -> np.ndarray:
    """Generate synthetic 1D workload signals for Golden Quartet scenarios.
    
    Parameters
    ----------
    Ts : float
        Sampling time [s]
    horizon : int
        Number of simulation steps
    profile : str
        Signal profile name: "step_response", "sine_tracking", or "steady_state"
    jitter_seed : int | None, optional
        Seed for Gaussian jitter. None disables noise.
    
    Returns
    -------
    np.ndarray
        1D array of length horizon with values in [0, 1]
    """
    time = np.arange(horizon)
    workload = np.zeros_like(time, dtype=float)
    
    if profile == "step_response":
        # Step function for transient analysis (Rise Time, Overshoot)
        step_time = int(0.2 * horizon)
        workload[step_time:] = 0.75
    elif profile == "sine_tracking":
        # Sine wave for dynamic tracking (ITAE)
        workload = 0.5 + 0.3 * np.sin(2 * np.pi * time * Ts / 5.0)
    elif profile == "steady_state":
        # Constant high-load for efficiency (Energy)
        workload = np.full_like(time, 0.8, dtype=float)
    else:
        raise ValueError(f"Unknown synthetic profile '{profile}'. Available: step_response, sine_tracking, steady_state")
    
    if jitter_seed is not None:
        rng = np.random.default_rng(seed=jitter_seed)
        workload += rng.normal(loc=0.0, scale=0.02, size=workload.shape)
    
    return np.clip(workload, 0.0, 1.0)


def adapt_1d_to_nd(signal_1d: np.ndarray, template_nd: np.ndarray) -> np.ndarray:
    """Adapt 1D synthetic signal to multi-dimensional input using template.
    
    This function creates multi-dimensional workload data by broadcasting or multiplying
    a 1D synthetic signal (e.g., step function) onto a template from real data.
    This allows synthetic stress tests to match PiNN input dimensions.
    
    Parameters
    ----------
    signal_1d : np.ndarray
        1D signal, shape (horizon,)
    template_nd : np.ndarray
        Template feature vector from real data, shape (input_dim,)
    
    Returns
    -------
    np.ndarray
        Multi-dimensional workload, shape (horizon, input_dim)
    """
    horizon = len(signal_1d)
    input_dim = len(template_nd)
    
    # Broadcast template to all time steps and scale by 1D signal
    workload_nd = np.tile(template_nd, (horizon, 1))  # (horizon, input_dim)
    workload_nd = workload_nd * signal_1d[:, np.newaxis]  # Scale by signal
    
    return np.clip(workload_nd, 0.0, 1.0)


class WorkloadLoader:
    """Workload loader: handles Golden Quartet scenarios only."""
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        input_dim: int = 8,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Initialize workload loader.
        
        Parameters
        ----------
        data_path : Optional[Path]
            TPU data file path
        input_dim : int
            Required input dimension for model
        test_data : Optional[Tuple[np.ndarray, np.ndarray]]
            Optional test data (inputs, outputs) to avoid repeated loading
        """
        self.input_dim = input_dim
        self.test_inputs = None
        self.test_outputs = None
        
        if test_data is not None:
            self.test_inputs, self.test_outputs = test_data
            logger.info(f"Using provided test data: {self.test_inputs.shape}")
        elif data_path is not None:
            _, _, self.test_inputs, self.test_outputs = load_data(data_path)
            logger.info(f"Loaded test data from {data_path}")
    
    def get_workload(
        self,
        scenario_name: str,
        Ts: float,
        horizon: int,
        **kwargs,
    ) -> np.ndarray:
        """Get workload sequence for Golden Quartet scenarios.
        
        Parameters
        ----------
        scenario_name : str
            Scenario name (must be one of: real_trace_replay, step_response,
            sine_tracking, steady_state)
        Ts : float
            Sampling time
        horizon : int
            Number of time steps
        **kwargs
            Additional parameters (e.g., jitter_seed)
        
        Returns
        -------
        np.ndarray
            Workload sequence, shape (horizon, input_dim)
        
        Raises
        ------
        ValueError
            If scenario_name is not in Golden Quartet
        """
        if scenario_name not in GOLDEN_QUARTET:
            raise ValueError(
                f"Scenario '{scenario_name}' not in Golden Quartet. "
                f"Available: {GOLDEN_QUARTET}"
            )
        
        if scenario_name == "real_trace_replay":
            return self._get_real_trace(horizon)
        else:
            return self._get_synthetic_scenario(scenario_name, Ts, horizon, **kwargs)
    
    def _get_real_trace(self, horizon: int) -> np.ndarray:
        """Randomly sample real TPU workload sequence from test set.
        
        Parameters
        ----------
        horizon : int
            Required number of time steps
        
        Returns
        -------
        np.ndarray
            Real workload sequence, shape (horizon, input_dim)
        """
        if self.test_inputs is None:
            raise RuntimeError("Real data not loaded. Provide data_path or test_data")
        
        # Randomly select a sequence
        seq_idx = np.random.RandomState(42).randint(0, len(self.test_inputs))
        real_seq = self.test_inputs[seq_idx]  # (Seq, Feat)
        
        # If sequence is too short, tile it
        if len(real_seq) < horizon:
            repeats = (horizon // len(real_seq)) + 1
            real_seq = np.tile(real_seq, (repeats, 1))[:horizon]
        else:
            # Random starting position
            start_idx = np.random.RandomState(42).randint(0, len(real_seq) - horizon + 1)
            real_seq = real_seq[start_idx:start_idx + horizon]
        
        # Ensure dimension match
        if real_seq.shape[1] != self.input_dim:
            real_seq = self._adapt_dimensions(real_seq, self.input_dim)
        
        logger.debug(f"Real trace: shape {real_seq.shape}")
        return real_seq
    
    def _get_synthetic_scenario(
        self,
        scenario_name: str,
        Ts: float,
        horizon: int,
        **kwargs,
    ) -> np.ndarray:
        """Generate synthetic scenario and adapt to multi-dimensional input.
        
        Parameters
        ----------
        scenario_name : str
            Synthetic scenario name (step_response, sine_tracking, steady_state)
        Ts : float
            Sampling time
        horizon : int
            Number of time steps
        **kwargs
            Additional parameters
        
        Returns
        -------
        np.ndarray
            Adapted workload sequence, shape (horizon, input_dim)
        """
        # Generate 1D synthetic signal
        signal_1d = generate_synthetic_signal(
            Ts=Ts,
            horizon=horizon,
            profile=scenario_name,
            **kwargs,
        )
        
        # Adapt to multi-dimensional using template
        if self.test_inputs is not None and len(self.test_inputs) > 0:
            # Use first sample from test set as template
            template = self.test_inputs[0][0]  # (Feat,)
            
            # Adapt dimensions if needed
            if len(template) != self.input_dim:
                template = self._adapt_dimensions(template.reshape(1, -1), self.input_dim)[0]
            
            workload_nd = adapt_1d_to_nd(signal_1d, template)
        else:
            # Fallback: simple broadcasting if no real data available
            workload_nd = np.tile(signal_1d[:, np.newaxis], (1, self.input_dim))
        
        logger.debug(f"Synthetic scenario '{scenario_name}': shape {workload_nd.shape}")
        return workload_nd
    
    def _adapt_dimensions(self, data: np.ndarray, target_dim: int) -> np.ndarray:
        """Adapt data dimensions.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        target_dim : int
            Target dimension
        
        Returns
        -------
        np.ndarray
            Adapted data
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        current_dim = data.shape[-1]
        
        if current_dim == target_dim:
            return data
        elif current_dim > target_dim:
            # Downsample: take first target_dim features
            return data[..., :target_dim]
        else:
            # Upsample: repeat last feature
            padding = np.tile(data[..., -1:], (target_dim - current_dim,))
            if data.ndim == 2:
                padding = padding.reshape(1, -1)
                return np.concatenate([data, padding], axis=-1)
            else:
                return np.concatenate([data, padding], axis=-1)


__all__ = ["WorkloadLoader", "adapt_1d_to_nd", "GOLDEN_QUARTET", "generate_synthetic_signal"]
