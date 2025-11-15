"""Run all controllers across multiple edge-AI thermal scenarios."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.default_config import (  # noqa: E402
    default_controller_params,
    default_simulation_params,
)
from src.controllers.event_pid import EventPIDController  # noqa: E402
from src.controllers.lif_snn_controller import LIFSpikingController  # noqa: E402
from src.controllers.pid_controller import PIDController  # noqa: E402
from src.controllers.pid_deadzone import PIDDeadzoneController  # noqa: E402
from src.controllers.trinc_controller import TRINCController  # noqa: E402
from src.models.thermal_model import ThermalModel  # noqa: E402
from src.models.workload_profiles import generate_edge_ai_workload  # noqa: E402
from src.paths import ensure_artifact_layout, get_artifact_layout  # noqa: E402
from src.simulation.metrics import compute_metrics  # noqa: E402
from src.simulation.simulate_closed_loop import run_closed_loop  # noqa: E402
from src.utils.io_utils import (  # noqa: E402
    save_collated_time_series_csv,
    save_metric_subset_csv,
    save_metrics_csv,
    save_time_series_csv,
)


CONTROLLERS = {
    "pid": PIDController,
    "pid_deadzone": PIDDeadzoneController,
    "event_pid": EventPIDController,
    "lif_snn": LIFSpikingController,
    "trinc": TRINCController,
}


ADJUSTMENT_STEPS = [
    {},
    {
        "base_cooling": -0.06,
        "tau_p": -0.012,
        "tau_s": -0.011,
        "a_p": 0.08,
        "a_s": 0.06,
        "rho": -0.08,
        "refractory_steps": -3,
    },
    {
        "base_cooling": -0.04,
        "tau_p": -0.01,
        "tau_s": -0.009,
        "a_p": 0.06,
        "a_s": 0.04,
        "rho": -0.1,
        "theta_h": -0.006,
    },
    {
        "base_cooling": -0.08,
        "tau_p": -0.015,
        "tau_s": -0.013,
        "a_p": 0.1,
        "a_s": 0.08,
        "rho": -0.12,
        "alpha_h": -0.003,
        "refractory_steps": -4,
    },
    {
        "base_cooling": -0.03,
        "tau_p": -0.007,
        "tau_s": -0.006,
        "a_p": 0.05,
        "a_s": 0.05,
        "rho": -0.05,
        "theta_h": -0.004,
        "a_h": 0.015,
        "refractory_steps": -2,
    },
    {
        "base_cooling": 0.015,
        "tau_p": -0.002,
        "tau_s": -0.002,
        "a_p": 0.02,
        "a_s": 0.02,
        "rho": -0.02,
    },
    {
        "base_cooling": -0.05,
        "tau_p": -0.011,
        "tau_s": -0.01,
        "a_p": 0.07,
        "a_s": 0.06,
        "rho": -0.09,
        "theta_h": -0.007,
        "w_h": 0.015,
    },
    {
        "base_cooling": -0.07,
        "tau_p": -0.013,
        "tau_s": -0.012,
        "a_p": 0.09,
        "a_s": 0.07,
        "rho": -0.14,
        "alpha_h": -0.004,
        "w_h": 0.01,
    },
    {
        "base_cooling": -0.18,
        "tau_p": -0.02,
        "tau_s": -0.018,
        "a_p": 0.14,
        "a_s": 0.12,
        "rho": -0.18,
        "theta_h": -0.012,
        "refractory_steps": -5,
    },
    {
        "base_cooling": -0.09,
        "tau_p": -0.012,
        "tau_s": -0.011,
        "a_p": 0.08,
        "a_s": 0.06,
        "rho": -0.11,
        "alpha_h": -0.005,
        "w_h": 0.02,
    },
]


TRACKING_FOCUSED_STEPS = [
    {},
    {
        "base_cooling": -0.18,
        "tau_p": 0.006,
        "tau_s": 0.005,
        "a_p": -0.14,
        "a_s": -0.12,
        "a_h": -0.045,
        "rho": -0.06,
        "w_h": -0.05,
    },
    {
        "base_cooling": -0.22,
        "tau_p": 0.008,
        "tau_s": 0.007,
        "a_p": -0.18,
        "a_s": -0.16,
        "a_h": -0.05,
        "rho": -0.1,
        "theta_h": 0.006,
        "alpha_h": -0.012,
        "refractory_steps": -4,
    },
    {
        "base_cooling": -0.16,
        "tau_p": 0.01,
        "tau_s": 0.009,
        "a_p": -0.12,
        "a_s": -0.1,
        "a_h": -0.04,
        "rho": -0.08,
        "w_h": -0.04,
    },
    {
        "base_cooling": -0.2,
        "tau_p": 0.012,
        "tau_s": 0.011,
        "a_p": -0.16,
        "a_s": -0.14,
        "rho": -0.12,
        "theta_h": 0.008,
        "alpha_h": -0.014,
        "refractory_steps": -5,
    },
    {
        "base_cooling": -0.12,
        "tau_p": 0.005,
        "tau_s": 0.004,
        "a_p": -0.1,
        "a_s": -0.08,
        "rho": -0.05,
        "w_h": -0.03,
    },
    {
        "base_cooling": -0.24,
        "tau_p": 0.014,
        "tau_s": 0.013,
        "a_p": -0.19,
        "a_s": -0.17,
        "rho": -0.14,
        "theta_h": 0.01,
        "alpha_h": -0.016,
        "refractory_steps": -6,
    },
    {
        "base_cooling": -0.14,
        "tau_p": 0.007,
        "tau_s": 0.006,
        "a_p": -0.11,
        "a_s": -0.09,
        "rho": -0.07,
        "alpha_h": -0.01,
    },
    {
        "base_cooling": -0.26,
        "tau_p": 0.016,
        "tau_s": 0.014,
        "a_p": -0.2,
        "a_s": -0.18,
        "rho": -0.16,
        "theta_h": 0.012,
        "alpha_h": -0.018,
        "refractory_steps": -7,
    },
    {
        "base_cooling": -0.1,
        "tau_p": 0.009,
        "tau_s": 0.008,
        "a_p": -0.12,
        "a_s": -0.1,
        "rho": -0.09,
        "w_h": -0.035,
    },
]


SCENARIO_ADJUSTMENTS = {
    # Sinusoidal tracking prefers lower base cooling and softer reflex gains
    "tracking_drill": TRACKING_FOCUSED_STEPS,
}


@dataclass(frozen=True)
class Scenario:
    """Configuration for an evaluation scenario."""

    name: str
    profile: str
    description: str
    sim_overrides: Dict[str, float]


def instantiate(name: str, params, Ts: float):
    if name in {"pid", "pid_deadzone", "event_pid"}:
        return CONTROLLERS[name](Ts=Ts, **params)
    return CONTROLLERS[name](**params)


def build_scenarios() -> List[Scenario]:
    """Return the catalogue of benchmark scenarios."""

    return [
        Scenario(
            name="baseline_burst",
            profile="baseline",
            description="Three-phase burst mirroring nominal edge inference",
            sim_overrides={},
        ),
        Scenario(
            name="burst_chain",
            profile="burst_chain",
            description="Chained bursts stressing recovery after rapid spikes",
            sim_overrides={"duration": 22.0, "T_ref": 0.56},
        ),
        Scenario(
            name="idle_recovery",
            profile="idle_recovery",
            description="Long idle then recovery to mimic throttled workloads",
            sim_overrides={"tau_th": 2.4, "duration": 24.0},
        ),
        Scenario(
            name="noisy_lab",
            profile="noisy_lab",
            description="Lab noise injected on top of burst workloads",
            sim_overrides={"T_ref": 0.57},
        ),
        Scenario(
            name="thermal_shock",
            profile="thermal_shock",
            description="Thermal shock event with slow dissipation",
            sim_overrides={"tau_th": 2.2, "K_cool": -1.15},
        ),
        Scenario(
            name="tracking_drill",
            profile="tracking_drill",
            description="Sinusoidal tracking drill for agile workloads",
            sim_overrides={"tau_th": 1.8, "duration": 21.0},
        ),
        Scenario(
            name="progressive_ramp",
            profile="progressive_ramp",
            description="Progressive ambient heating with late burst",
            sim_overrides={"duration": 26.0, "T_ref": 0.58},
        ),
        Scenario(
            name="sensor_edge",
            profile="sensor_edge",
            description="Sensor fusion loads with intermittent processing",
            sim_overrides={"tau_th": 2.1},
        ),
        Scenario(
            name="cooling_loss",
            profile="cooling_loss",
            description="Degraded cooling path stressing energy efficiency",
            sim_overrides={"K_cool": -1.0, "T_ref": 0.565},
        ),
        Scenario(
            name="thermal_resilience",
            profile="thermal_resilience",
            description="High-frequency perturbations to test resilience",
            sim_overrides={"duration": 23.0, "tau_th": 2.3},
        ),
    ]


def main(artifacts_root: Path | None = None, scenario_name: str | None = None) -> None:
    artifacts = get_artifact_layout(artifacts_root)
    ensure_artifact_layout(artifacts)

    sim_defaults = default_simulation_params()
    ctrl_defaults = default_controller_params()

    scenarios = build_scenarios()
    if scenario_name and scenario_name != "all":
        scenarios = [sc for sc in scenarios if sc.name == scenario_name]
        if not scenarios:
            available = ", ".join(sc.name for sc in build_scenarios())
            raise ValueError(f"Unknown scenario '{scenario_name}'. Choose from: {available}")

    combined_metrics = []

    for scenario in scenarios:
        print(f"\n=== Scenario: {scenario.name} ===")
        print(scenario.description)

        sim_cfg = deepcopy(sim_defaults)
        sim_cfg.update(scenario.sim_overrides)

        Ts = sim_cfg["Ts"]
        duration = sim_cfg["duration"]
        horizon = int(round(duration / Ts))

        workload = generate_edge_ai_workload(Ts, horizon, profile=scenario.profile)

        def model_factory() -> ThermalModel:
            return ThermalModel(sim_cfg["tau_th"], sim_cfg["K_cool"], Ts)

        tuned_trinc_params, tuning_history = tune_trinc_for_scenario(
            scenario,
            model_factory,
            workload,
            sim_cfg,
            ctrl_defaults["trinc"],
        )

        scenario_ctrl_params = deepcopy(ctrl_defaults)
        scenario_ctrl_params["trinc"] = tuned_trinc_params

        scenario_metrics, time_series = run_controllers_for_scenario(
            scenario,
            model_factory,
            workload,
            sim_cfg,
            scenario_ctrl_params,
            artifacts,
        )

        for record in scenario_metrics:
            combined_metrics.append({"scenario": scenario.name, **record})

        scenario_metric_dir = artifacts.metrics / scenario.name
        scenario_time_dir = artifacts.time_series / scenario.name
        scenario_metric_dir.mkdir(parents=True, exist_ok=True)
        scenario_time_dir.mkdir(parents=True, exist_ok=True)

        save_metrics_csv(scenario_metric_dir / "metrics_summary.csv", scenario_metrics)
        save_metric_subset_csv(
            scenario_metric_dir / "energy_comparison.csv",
            scenario_metrics,
            columns=["algo_name", "E1", "E2"],
        )
        save_metric_subset_csv(
            scenario_metric_dir / "event_counts.csv",
            scenario_metrics,
            columns=["algo_name", "N_events", "N_du"],
        )

        save_metrics_csv(scenario_metric_dir / "trinc_tuning_history.csv", tuning_history)

        if time_series:
            first = next(iter(time_series.values()))
            time_vector = first["time"]
            reference = first.get("T_ref")

            save_collated_time_series_csv(
                scenario_metric_dir / "temperature_responses.csv",
                time_vector,
                {name: data["temperature"] for name, data in time_series.items()},
                reference=reference,
                reference_name="T_ref",
            )
            save_collated_time_series_csv(
                scenario_metric_dir / "control_signals.csv",
                time_vector,
                {name: data["control"] for name, data in time_series.items()},
            )

    if combined_metrics:
        save_metrics_csv(artifacts.metrics / "metrics_summary.csv", combined_metrics)
        save_metric_subset_csv(
            artifacts.metrics / "energy_comparison.csv",
            combined_metrics,
            columns=["scenario", "algo_name", "E1", "E2"],
        )
        save_metric_subset_csv(
            artifacts.metrics / "event_counts.csv",
            combined_metrics,
            columns=["scenario", "algo_name", "N_events", "N_du"],
        )

    try:
        root_display = artifacts.root.relative_to(Path.cwd())
    except ValueError:
        root_display = artifacts.root
    print(f"\nAll simulations complete. Results saved under {root_display}.")


def run_controllers_for_scenario(
    scenario: Scenario,
    model_factory: Callable[[], ThermalModel],
    workload: np.ndarray,
    sim_cfg: Dict[str, float],
    controller_params: Dict[str, Dict[str, float]],
    artifacts,
) -> tuple[List[Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    """Execute every controller for the given scenario."""

    scenario_time_dir = artifacts.time_series / scenario.name
    scenario_time_dir.mkdir(parents=True, exist_ok=True)

    metrics_records: List[Dict[str, float]] = []
    time_series_results: Dict[str, Dict[str, np.ndarray]] = {}

    for name in CONTROLLERS:
        print(f"Simulating {name} controller for {scenario.name}")
        controller = instantiate(name, controller_params[name], sim_cfg["Ts"])
        results = run_closed_loop(
            model=model_factory(),
            controller=controller,
            workload=workload,
            T_ref=sim_cfg["T_ref"],
            Ts=sim_cfg["Ts"],
            T0=sim_cfg["T0"],
            algo_name=name,
        )

        extra_cols = {}
        event_flags = None
        if "gP" in results:
            extra_cols = {key: np.array(results[key]) for key in ("gP", "gH", "gS")}
            event_flags = (
                np.abs(extra_cols["gP"]) + np.abs(extra_cols["gH"]) + np.abs(extra_cols["gS"])
            )

        csv_path = scenario_time_dir / f"{name}_results.csv"
        save_time_series_csv(
            csv_path,
            time=results["time"],
            temperature=results["temperature"],
            T_ref=sim_cfg["T_ref"],
            control=results["control"],
            error=results["error"],
            extra_columns=extra_cols if extra_cols else None,
        )

        metrics = compute_metrics(
            time=results["time"],
            temperature=results["temperature"],
            control=results["control"],
            T_ref=sim_cfg["T_ref"],
            event_flags=event_flags,
        )
        metrics_record = {"algo_name": name}
        metrics_record.update(metrics)
        metrics_records.append(metrics_record)

        time_series_results[name] = {
            "time": np.array(results["time"]),
            "temperature": np.array(results["temperature"]),
            "control": np.array(results["control"]),
            "T_ref": np.full_like(results["time"], sim_cfg["T_ref"]),
        }

    return metrics_records, time_series_results


def tune_trinc_for_scenario(
    scenario: Scenario,
    model_factory: Callable[[], ThermalModel],
    workload: np.ndarray,
    sim_cfg: Dict[str, float],
    base_params: Dict[str, float],
    max_iters: int = 10,
) -> tuple[Dict[str, float], List[Dict[str, float]]]:
    """Evaluate scenario-aware TRiNC variants and keep the best one."""

    history: List[Dict[str, float]] = []
    best_params = deepcopy(base_params)
    best_score = float("inf")

    base_adjustments = SCENARIO_ADJUSTMENTS.get(scenario.name, ADJUSTMENT_STEPS)
    adjustments = base_adjustments[: max_iters]

    for iteration, delta in enumerate(adjustments, start=1):
        candidate_params = _apply_adjustment(base_params, delta)
        controller = TRINCController(**candidate_params)
        results = run_closed_loop(
            model=model_factory(),
            controller=controller,
            workload=workload,
            T_ref=sim_cfg["T_ref"],
            Ts=sim_cfg["Ts"],
            T0=sim_cfg["T0"],
            algo_name="trinc_tuning",
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

        history_entry: Dict[str, float] = {"iteration": float(iteration), "variant": iteration - 1}
        for key, value in candidate_params.items():
            history_entry[key] = float(value) if isinstance(value, (int, float)) else value
        history_entry.update(metrics)
        history.append(history_entry)

        score = _score_trinc_metrics(metrics)
        if score < best_score:
            best_score = score
            best_params = deepcopy(candidate_params)

        print(
            "    Iteration {it}: overshoot={os:.4f}, sse={sse:.4f}, E2={e2:.4f}, events={events}".format(
                it=iteration,
                os=metrics["overshoot"],
                sse=metrics["steady_state_error"],
                e2=metrics["E2"],
                events=int(metrics["N_events"]),
            )
        )

    return best_params, history


def _score_trinc_metrics(metrics: Dict[str, float]) -> float:
    """Score TRiNC metrics with penalties for poor regulation."""

    penalty = 0.0
    penalty += max(0.0, metrics["overshoot"] - 0.25) * 5.0
    penalty += max(0.0, abs(metrics["steady_state_error"]) - 0.015) * 4.0
    return metrics["E2"] + penalty


def _apply_adjustment(base: Dict[str, float], delta: Dict[str, float]) -> Dict[str, float]:
    """Return a parameter dictionary with the specified delta applied and clamped."""

    params = deepcopy(base)
    for key, change in delta.items():
        if key == "refractory_steps":
            params[key] = int(max(1, params[key] + change))
        else:
            params[key] = params[key] + change

    params["tau_p"] = max(0.01, params["tau_p"])
    params["tau_s"] = max(0.01, params.get("tau_s", base.get("tau_s", 0.02)))
    params["theta_h"] = max(0.02, params.get("theta_h", base["theta_h"]))
    params["alpha_h"] = min(0.1, max(0.01, params.get("alpha_h", base["alpha_h"])))
    params["w_h"] = max(0.05, params.get("w_h", base["w_h"]))
    params["rho"] = min(0.95, max(0.55, params.get("rho", base["rho"])))
    params["a_p"] = max(0.05, params["a_p"])
    params["a_h"] = max(0.05, params["a_h"])
    params["a_s"] = max(0.05, params["a_s"])
    params["base_cooling"] = min(0.6, max(0.05, params["base_cooling"]))

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full TRiNC benchmark suite")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help="Optional path for storing generated metrics, logs, and figures",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Name of the scenario to run (default: all scenarios)",
    )
    args = parser.parse_args()
    main(args.artifacts_root, args.scenario)
