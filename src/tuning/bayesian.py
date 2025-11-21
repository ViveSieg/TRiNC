"""Bayesian optimizer: multi-objective optimization using Optuna."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import optuna
import pandas as pd

from ..controllers.registry import create_controller
from ..config.schema import SimulationConfig
from ..simulation.executor import SimulationExecutor
from ..models.workload_loader import WorkloadLoader

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """Bayesian optimizer: finds optimal controller parameters using Optuna."""
    
    def __init__(
        self,
        controller_name: str,
        param_bounds: Dict[str, tuple[float, float]],
        simulation_config: SimulationConfig,
        model: Any,
        workload_loader: WorkloadLoader,
        scenario_name: str = "real_trace_replay",  # Always optimize on real_trace_replay
        weights: Optional[Dict[str, float]] = None,
        n_trials: int = 50,
        random_seed: int = 42,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Initialize optimizer.
        
        Parameters
        ----------
        controller_name : str
            Controller name
        param_bounds : Dict[str, tuple[float, float]]
            Parameter search space, format {"param_name": (min, max)}
        simulation_config : SimulationConfig
            Simulation configuration
        model : Any
            Thermal model
        workload_loader : WorkloadLoader
            Workload loader
        scenario_name : str
            Optimization scenario name
        weights : Optional[Dict[str, float]]
            Objective function weights, default {"ITAE": 1.0, "Energy": 0.5}
        n_trials : int
            Number of optimization trials
        random_seed : int
            Random seed
        """
        self.controller_name = controller_name
        self.param_bounds = param_bounds
        self.simulation_config = simulation_config
        self.model = model
        self.workload_loader = workload_loader
        self.scenario_name = scenario_name
        self.weights = weights or {"ITAE": 1.0, "Energy": 0.5}
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.output_dir = output_dir or Path("artifacts/tuning")
        
        self.executor = SimulationExecutor(simulation_config)
        self.study: Optional[optuna.Study] = None
        self.results: list[Dict[str, Any]] = []
        
        # Normalization parameters (will be computed during optimization)
        self.itae_min: Optional[float] = None
        self.itae_max: Optional[float] = None
        self.energy_min: Optional[float] = None
        self.energy_max: Optional[float] = None
    
    def _normalize_metrics(self, itae: float, energy: float) -> tuple[float, float]:
        """Normalize metrics to [0, 1] range to prevent scale bias.
        
        Parameters
        ----------
        itae : float
            ITAE value
        energy : float
            Energy value
        
        Returns
        -------
        tuple[float, float]
            Normalized (ITAE, Energy) values
        """
        if self.itae_min is None or self.itae_max is None:
            # First trial: use raw values (will normalize in subsequent trials)
            return itae, energy
        
        # Normalize using min-max scaling
        itae_norm = (itae - self.itae_min) / (self.itae_max - self.itae_min + 1e-8)
        energy_norm = (energy - self.energy_min) / (self.energy_max - self.energy_min + 1e-8)
        
        return itae_norm, energy_norm
    
    def _update_normalization(self, itae: float, energy: float) -> None:
        """Update normalization parameters with new metric values.
        
        Parameters
        ----------
        itae : float
            ITAE value
        energy : float
            Energy value
        """
        if self.itae_min is None:
            self.itae_min = itae
            self.itae_max = itae
            self.energy_min = energy
            self.energy_max = energy
        else:
            self.itae_min = min(self.itae_min, itae)
            self.itae_max = max(self.itae_max, itae)
            self.energy_min = min(self.energy_min, energy)
            self.energy_max = max(self.energy_max, energy)
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function: w1 * normalized_ITAE + w2 * normalized_Energy.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        
        Returns
        -------
        float
            Weighted loss
        """
        # Sample parameters
        params = {}
        for param_name, (min_val, max_val) in self.param_bounds.items():
            params[param_name] = trial.suggest_float(param_name, min_val, max_val)
        
        # Create controller
        try:
            controller = create_controller(
                self.controller_name,
                params,
                Ts=self.simulation_config.Ts,
            )
        except Exception as e:
            logger.warning(f"Invalid parameters in trial {trial.number}: {e}")
            return 1e6
        
        # Get workload
        horizon = int(round(self.simulation_config.duration / self.simulation_config.Ts))
        workload = self.workload_loader.get_workload(
            self.scenario_name,
            self.simulation_config.Ts,
            horizon,
        )
        
        # Run simulation
        try:
            result = self.executor.run(
                controller=controller,
                model=self.model,
                workload=workload,
                algo_name=self.controller_name,
            )
            
            itae = result.metrics["ITAE"]
            energy = result.metrics["Energy"]
            
            # Update normalization parameters
            self._update_normalization(itae, energy)
            
            # Normalize metrics
            itae_norm, energy_norm = self._normalize_metrics(itae, energy)
            
            # Compute weighted loss with normalized metrics
            loss = (
                self.weights["ITAE"] * itae_norm
                + self.weights["Energy"] * energy_norm
            )
            
            # Record results (store both raw and normalized values)
            self.results.append({
                "trial": trial.number,
                **params,
                "ITAE": itae,
                "Energy": energy,
                "ITAE_normalized": itae_norm,
                "Energy_normalized": energy_norm,
                "Overshoot": result.metrics["Overshoot"],
                "loss": loss,
            })
            
            return loss
        except Exception as e:
            logger.warning(f"Simulation failed in trial {trial.number}: {e}")
            return 1e6
    
    def optimize(self) -> Dict[str, Any]:
        """Execute optimization.
        
        Returns
        -------
        Dict[str, Any]
            Best parameters
        """
        logger.info(f"Starting Bayesian optimization for {self.controller_name}")
        logger.info(f"Scenario: {self.scenario_name}, Trials: {self.n_trials}")
        
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.study = study
        best_params = study.best_params.copy()
        
        # Remove Ts if present (it's a configuration parameter)
        best_params.pop("Ts", None)
        
        logger.info(f"Optimization completed. Best loss: {study.best_value:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Auto-save results
        self.save_results()
        
        return best_params
    
    def save_results(self, output_path: Optional[Path] = None) -> None:
        """Save all optimization trial results to CSV.
        
        Saves to artifacts/tuning/tuning_history_{controller_name}.csv by default.
        This file is used directly for plotting Pareto Frontier in the paper.
        
        Parameters
        ----------
        output_path : Optional[Path]
            Output file path (default: tuning_history_{controller_name}.csv)
        """
        if not self.results:
            raise RuntimeError("No optimization results to save. Run optimize() first.")
        
        if output_path is None:
            output_path = self.output_dir / f"tuning_history_{self.controller_name}.csv"
        
        df = pd.DataFrame(self.results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Optimization results saved to: {output_path}")
        logger.info(f"Total trials: {len(df)}")
    
    def get_pareto_frontier(self) -> pd.DataFrame:
        """Get Pareto frontier (for plotting Pareto Frontier).
        
        Returns
        -------
        pd.DataFrame
            Pareto frontier data with ITAE and Energy columns
        """
        if not self.results:
            raise RuntimeError("No optimization results. Run optimize() first.")
        
        df = pd.DataFrame(self.results)
        
        # Simple Pareto frontier extraction
        pareto_mask = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j:
                    # If j dominates i in both ITAE and Energy, i is not on Pareto frontier
                    if (df.iloc[j]["ITAE"] <= df.iloc[i]["ITAE"] and
                        df.iloc[j]["Energy"] <= df.iloc[i]["Energy"] and
                        (df.iloc[j]["ITAE"] < df.iloc[i]["ITAE"] or
                         df.iloc[j]["Energy"] < df.iloc[i]["Energy"])):
                        pareto_mask[i] = False
                        break
        
        pareto_df = df[pareto_mask].sort_values("ITAE")
        logger.info(f"Pareto frontier: {len(pareto_df)} points out of {len(df)} trials")
        return pareto_df


__all__ = ["BayesianOptimizer"]
