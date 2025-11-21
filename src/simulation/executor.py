"""Universal simulation executor: eliminates code duplication and standardizes simulation flow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np

from ..controllers.base import BaseController
from ..config.schema import SimulationConfig
from ..utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Simulation result container."""
    
    time: np.ndarray
    temperature: np.ndarray
    control: np.ndarray
    error: np.ndarray
    workload: np.ndarray
    metrics: Dict[str, float]
    algo_name: str


class SimulationExecutor:
    """Universal simulation executor."""
    
    def __init__(self, config: SimulationConfig) -> None:
        """Initialize executor.
        
        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration
        """
        self.config = config
        self.horizon = int(round(config.duration / config.Ts))
    
    def run(
        self,
        controller: BaseController,
        model: Any,  # ThermalModel or GrayBoxThermalModel
        workload: np.ndarray,
        algo_name: str = "unknown",
        T_ref: Optional[float] = None,
    ) -> SimulationResult:
        """Execute closed-loop simulation.
        
        Standardized flow: Reset -> Loop (Compute Error -> Get Control u -> Model Step) -> Record -> Metrics
        
        Parameters
        ----------
        controller : BaseController
            Controller instance
        model : Any
            Thermal model (must implement reset(initial_temp) and step(current_temp, control_u, workload))
        workload : np.ndarray
            Workload sequence, shape (horizon, ...)
        algo_name : str
            Algorithm name
        T_ref : Optional[float]
            Reference temperature (defaults to config.T_ref)
        
        Returns
        -------
        SimulationResult
            Simulation result
        """
        T_ref = T_ref if T_ref is not None else self.config.T_ref
        Ts = self.config.Ts
        T0 = self.config.T0
        
        # Initialize
        time = np.arange(self.horizon) * Ts
        temperature = np.zeros(self.horizon)
        control_signal = np.zeros(self.horizon)
        error = np.zeros(self.horizon)
        
        model.reset(T0)
        controller.reset()
        
        T_k = T0
        u_k = 0.0
        
        # Main loop
        for k in range(self.horizon):
            error[k] = T_k - T_ref
            u_k = controller.compute_control(error[k])
            control_signal[k] = u_k
            temperature[k] = T_k
            
            # Get current workload (supports multi-dimensional)
            if workload.ndim == 1:
                w_k = workload[k]
            else:
                w_k = workload[k]
            
            # Model step
            if hasattr(model, 'step'):
                T_k = model.step(T_k, u_k, w_k)
            else:
                raise AttributeError(f"Model {type(model)} does not implement step method")
        
        # Compute metrics
        event_flags = None
        if hasattr(controller, 'event_log'):
            event_log = controller.event_log
            if event_log:
                # Compute event flags
                gP = np.array(event_log.get('gP', []))
                gH = np.array(event_log.get('gH', []))
                gS = np.array(event_log.get('gS', []))
                if len(gP) > 0:
                    event_flags = np.abs(gP) + np.abs(gH) + np.abs(gS)
        
        metrics = compute_metrics(
            time=time,
            temperature=temperature,
            control=control_signal,
            T_ref=T_ref,
            event_flags=event_flags,
        )
        
        # Compute ITAE (Integral Time-weighted Absolute Error)
        itae = float(np.sum(np.abs(error) * time) * Ts)
        metrics['ITAE'] = itae
        
        # Compute Energy (energy consumption: sum(u^2))
        energy = float(np.sum(control_signal**2) * Ts)
        metrics['Energy'] = energy
        
        # Compute Overshoot (safety metric)
        overshoot = float(np.max(np.maximum(temperature - T_ref, 0.0)))
        metrics['Overshoot'] = overshoot
        
        # Compute Rise Time (time to reach 90% of final value from 10%)
        if len(temperature) > 1:
            T_final = temperature[-1]
            T_range = T_final - temperature[0]
            if T_range > 0.01:  # Significant change
                T_10 = temperature[0] + 0.1 * T_range
                T_90 = temperature[0] + 0.9 * T_range
                rise_start_idx = np.where(temperature >= T_10)[0]
                rise_end_idx = np.where(temperature >= T_90)[0]
                if len(rise_start_idx) > 0 and len(rise_end_idx) > 0:
                    rise_time = float((rise_end_idx[0] - rise_start_idx[0]) * Ts)
                    metrics['RiseTime'] = rise_time
                else:
                    metrics['RiseTime'] = float('inf')
            else:
                metrics['RiseTime'] = 0.0
        else:
            metrics['RiseTime'] = 0.0
        
        # Settling Time is already computed in compute_metrics
        # Ensure it's in the metrics dict
        if 'settling_time' not in metrics:
            settling_idx = np.where(np.abs(error) <= 0.02 * T_ref)[0]
            if settling_idx.size > 0:
                metrics['settling_time'] = float(time[settling_idx[0]])
            else:
                metrics['settling_time'] = float(time[-1])
        
        logger.debug(f"Simulation completed: {algo_name}, ITAE={itae:.4f}, Energy={energy:.4f}")
        
        return SimulationResult(
            time=time,
            temperature=temperature,
            control=control_signal,
            error=error,
            workload=workload if workload.ndim == 1 else workload[:, 0],  # Flatten for storage
            metrics=metrics,
            algo_name=algo_name,
        )


__all__ = ["SimulationExecutor", "SimulationResult"]
