"""Universal simulation executor: eliminates code duplication and standardizes simulation flow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np

from ..controllers.base import BaseController
from ..config.schema import SimulationConfig
from ..utils.metrics import compute_metrics, compute_complexity_metrics

logger = logging.getLogger(__name__)


@dataclass
class RealismConfig:
    """Configuration for realistic simulation disturbances.
    
    This configuration enables injection of real-world disturbances to
    test controller robustness in Sim2Real scenarios.
    """
    
    sensor_noise_std: float = 0.0  # Standard deviation of Gaussian sensor noise
    actuator_delay_steps: int = 0  # Number of steps delay before control signal takes effect
    enable_noise: bool = False  # Enable/disable noise injection
    enable_delay: bool = False  # Enable/disable actuator delay
    
    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        assert self.sensor_noise_std >= 0, "Sensor noise std must be non-negative"
        assert self.actuator_delay_steps >= 0, "Actuator delay steps must be non-negative"
        if self.enable_noise:
            assert self.sensor_noise_std > 0, "Sensor noise std must be > 0 when noise is enabled"
        if self.enable_delay:
            assert self.actuator_delay_steps > 0, "Actuator delay steps must be > 0 when delay is enabled"


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
    """Universal simulation executor with optional realism disturbances."""
    
    def __init__(
        self,
        config: SimulationConfig,
        realism_config: Optional[RealismConfig] = None,
    ) -> None:
        """Initialize executor.
        
        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration
        realism_config : Optional[RealismConfig]
            Realism configuration for noise and delay injection (default: None, no disturbances)
        """
        self.config = config
        self.realism_config = realism_config or RealismConfig()
        self.horizon = int(round(config.duration / config.Ts))
        
        # Initialize control signal buffer for actuator delay
        if self.realism_config.enable_delay:
            self.control_buffer: list[float] = [0.0] * self.realism_config.actuator_delay_steps
        else:
            self.control_buffer = []
    
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
        
        # Reset control buffer for actuator delay
        if self.realism_config.enable_delay:
            self.control_buffer = [0.0] * self.realism_config.actuator_delay_steps
        
        T_k = T0
        u_k = 0.0
        
        # Main loop
        for k in range(self.horizon):
            # Inject sensor noise: controller sees noisy temperature reading
            T_measured = T_k
            if self.realism_config.enable_noise:
                noise = np.random.normal(0.0, self.realism_config.sensor_noise_std)
                T_measured = T_k + noise
                # Clip to valid range
                T_measured = np.clip(T_measured, 0.0, 1.0)
            
            # Compute error based on measured (possibly noisy) temperature
            error[k] = T_measured - T_ref
            u_k = controller.compute_control(error[k])
            control_signal[k] = u_k
            
            # Apply actuator delay: use delayed control signal
            if self.realism_config.enable_delay:
                u_effective = self.control_buffer[0] if len(self.control_buffer) > 0 else 0.0
                # Update buffer: shift and add new control
                self.control_buffer = self.control_buffer[1:] + [u_k]
            else:
                u_effective = u_k
            
            # Store actual (noise-free) temperature for metrics
            temperature[k] = T_k
            
            # Get current workload (supports multi-dimensional)
            if workload.ndim == 1:
                w_k = workload[k]
            else:
                w_k = workload[k]
            
            # Model step with effective (possibly delayed) control signal
            if hasattr(model, 'step'):
                T_k = model.step(T_k, u_effective, w_k)
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
        
        # Compute computational complexity metrics
        complexity_metrics = compute_complexity_metrics(
            controller=controller,
            algo_name=algo_name,
            horizon=self.horizon,
        )
        metrics.update(complexity_metrics)
        
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


__all__ = ["SimulationExecutor", "SimulationResult", "RealismConfig"]
