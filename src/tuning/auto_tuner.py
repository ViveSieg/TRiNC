"""Automatic hyper-parameter tuning for all controllers.

The tuner implements a lightweight bandit-inspired search with early
stopping. Each iteration samples a candidate parameter set, simulates the
closed loop on a suite of workloads, and keeps the best-performing
configuration according to a weighted scalar score. If no improvement is
seen for ``patience`` iterations the search stops early.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from src.config.default_config import (
    default_controller_params,
    default_simulation_params,
    default_tuning_settings,
    tuning_search_spaces,
)
from src.controllers.event_pid import EventPIDController
from src.controllers.lif_snn_controller import LIFSpikingController
from src.controllers.pid_controller import PIDController
from src.controllers.pid_deadzone import PIDDeadzoneController
from src.controllers.trinc_controller import TRINCController
from src.models.thermal_model import ThermalModel
from src.models.workload_profiles import generate_edge_ai_workload
from src.simulation.metrics import compute_metrics
from src.simulation.simulate_closed_loop import run_closed_loop

CONTROLLERS = {
    "pid": PIDController,
    "pid_deadzone": PIDDeadzoneController,
    "event_pid": EventPIDController,
    "lif_snn": LIFSpikingController,
    "trinc": TRINCController,
}


@dataclass
class TuningHistoryEntry:
    iteration: int
    controller: str
    score: float
    metrics: Dict[str, float]
    params: Dict[str, Any]


def instantiate_controller(name: str, params: Dict[str, float], Ts: float):
    if name in {"pid", "pid_deadzone", "event_pid"}:
        return CONTROLLERS[name](Ts=Ts, **params)
    return CONTROLLERS[name](**params)


def _score_metrics(metrics: Dict[str, float], iae: float, horizon: int, weights: Dict[str, float]) -> float:
    overshoot_term = weights["overshoot"] * metrics["overshoot"]
    steady_term = weights["steady_state"] * abs(metrics["steady_state_error"])
    iae_term = weights["iae"] * iae
    energy_term = weights["energy"] * metrics["E2"]
    event_rate = metrics["N_events"] / max(1, horizon)
    event_term = weights["event_rate"] * event_rate
    return overshoot_term + steady_term + iae_term + energy_term + event_term


def evaluate_controller(
    controller_name: str,
    params: Dict[str, float],
    workloads: Iterable[str],
    sim_cfg: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Run a controller over several workloads and return averaged metrics."""

    Ts = sim_cfg["Ts"]
    horizon = int(round(sim_cfg["duration"] / Ts))
    model_kwargs = {k: sim_cfg[k] for k in ("tau_th", "K_heat", "K_cool")}

    aggregate_metrics: Dict[str, float] = {
        "overshoot": 0.0,
        "steady_state_error": 0.0,
        "E2": 0.0,
        "N_events": 0.0,
        "iae": 0.0,
    }

    workload_list = list(workloads)

    for profile in workload_list:
        workload = generate_edge_ai_workload(Ts, horizon, profile=profile)
        model = ThermalModel(**model_kwargs, Ts=Ts)
        controller = instantiate_controller(controller_name, params, Ts)

        results = run_closed_loop(
            model=model,
            controller=controller,
            workload=workload,
            T_ref=sim_cfg["T_ref"],
            Ts=Ts,
            T0=sim_cfg["T0"],
            algo_name=f"{controller_name}_tuning",
        )

        event_flags = (
            np.abs(results["gP"]) + np.abs(results["gH"]) + np.abs(results["gS"])
            if "gP" in results
            else None
        )
        metrics = compute_metrics(
            time=results["time"],
            temperature=results["temperature"],
            control=results["control"],
            T_ref=sim_cfg["T_ref"],
            event_flags=event_flags,
        )
        iae = float(np.sum(np.abs(results["error"])) * Ts)

        aggregate_metrics["overshoot"] += metrics["overshoot"]
        aggregate_metrics["steady_state_error"] += metrics["steady_state_error"]
        aggregate_metrics["E2"] += metrics["E2"]
        aggregate_metrics["N_events"] += metrics["N_events"]
        aggregate_metrics["iae"] += iae

    n = max(1, len(workload_list))
    for key in aggregate_metrics:
        aggregate_metrics[key] /= n

    score = _score_metrics(
        {k: aggregate_metrics[k] for k in ("overshoot", "steady_state_error", "E2", "N_events")},
        aggregate_metrics["iae"],
        horizon,
        weights,
    )
    aggregate_metrics["score"] = score
    return score, aggregate_metrics


def _sample_candidate(
    base_params: Dict[str, float],
    best_params: Dict[str, float],
    bounds: Dict[str, tuple[float, float]],
    rng: np.random.Generator,
    explore_prob: float,
) -> Dict[str, float]:
    candidate = deepcopy(best_params)
    for key, (low, high) in bounds.items():
        span = high - low
        if rng.random() < explore_prob:
            candidate[key] = float(rng.uniform(low, high))
        else:
            sigma = 0.2 * span
            candidate[key] = float(rng.normal(loc=best_params.get(key, base_params.get(key, low)), scale=sigma))
            candidate[key] = min(high, max(low, candidate[key]))
    return candidate


def tune_controllers(
    controller_subset: Iterable[str] | None = None,
    tuning_workloads: Iterable[str] | None = None,
    sim_overrides: Dict[str, float] | None = None,
    tuning_settings: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Dict[str, float]], List[TuningHistoryEntry]]:
    """Tune every controller (or a subset) using a simple adaptive search."""

    base_ctrl_params = default_controller_params()
    search_bounds = tuning_search_spaces()
    sim_cfg = default_simulation_params()
    if sim_overrides:
        sim_cfg.update(sim_overrides)

    settings = default_tuning_settings()
    if tuning_settings:
        settings.update(tuning_settings)

    workloads = list(tuning_workloads) if tuning_workloads else settings["tuning_workloads"]
    max_iters = settings["max_iters"]
    patience = settings["patience"]
    rng = np.random.default_rng(settings["rng_seed"])
    explore_prob = settings.get("explore_prob", 0.35)
    weights = settings["weights"]

    controllers = controller_subset or base_ctrl_params.keys()

    tuned_params: Dict[str, Dict[str, float]] = {}
    history: List[TuningHistoryEntry] = []

    for ctrl_name in controllers:
        print(f"\nTuning controller: {ctrl_name}")
        bounds = search_bounds[ctrl_name]
        base = base_ctrl_params[ctrl_name]
        best_params = deepcopy(base)
        best_score, best_metrics = evaluate_controller(ctrl_name, best_params, workloads, sim_cfg, weights)
        no_improve = 0

        history.append(
            TuningHistoryEntry(
                iteration=0,
                controller=ctrl_name,
                score=best_score,
                metrics=best_metrics,
                params=deepcopy(best_params),
            )
        )

        for iteration in range(1, max_iters + 1):
            candidate = _sample_candidate(base, best_params, bounds, rng, explore_prob)
            score, metrics = evaluate_controller(ctrl_name, candidate, workloads, sim_cfg, weights)
            history.append(
                TuningHistoryEntry(
                    iteration=iteration,
                    controller=ctrl_name,
                    score=score,
                    metrics=metrics,
                    params=deepcopy(candidate),
                )
            )

            if score + 1e-5 < best_score:
                best_score = score
                best_params = candidate
                no_improve = 0
                print(f"  Iter {iteration:02d}: improved score {score:.4f}")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  Early stopping after {iteration} iterations (patience reached)")
                break

        tuned_params[ctrl_name] = best_params

    return tuned_params, history


def history_to_rows(history: List[TuningHistoryEntry]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in history:
        row = {
            "iteration": entry.iteration,
            "controller": entry.controller,
            "score": entry.score,
        }
        row.update({f"metric_{k}": v for k, v in entry.metrics.items()})
        row.update({f"param_{k}": v for k, v in entry.params.items()})
        rows.append(row)
    return rows


__all__ = [
    "tune_controllers",
    "history_to_rows",
    "TuningHistoryEntry",
]
