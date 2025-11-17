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
    apply_params_override,
    default_controller_params,
    default_scenarios,
    default_simulation_params,
    save_tuned_params_json,
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
from src.tuning import history_to_rows, tune_controllers  # noqa: E402
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
    return [Scenario(**entry) for entry in default_scenarios()]


def main(
    artifacts_root: Path | None = None,
    scenario_name: str | None = None,
    run_tuning: bool = False,
    tuning_workloads: List[str] | None = None,
) -> None:
    artifacts = get_artifact_layout(artifacts_root)
    ensure_artifact_layout(artifacts)

    sim_defaults = default_simulation_params()
    ctrl_defaults = default_controller_params()

    tuning_history_rows: List[Dict[str, float]] = []
    if run_tuning:
        tuned_params, history = tune_controllers(tuning_workloads=tuning_workloads)
        ctrl_defaults = apply_params_override(ctrl_defaults, tuned_params)
        tuning_history_rows = history_to_rows(history)
        save_tuned_params_json(tuned_params, artifacts.metrics / "tuned_params.json")
        save_metrics_csv(artifacts.metrics / "tuning_history.csv", tuning_history_rows)

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
            return ThermalModel(sim_cfg["tau_th"], sim_cfg["K_heat"], sim_cfg["K_cool"], Ts)

        scenario_metrics, time_series = run_controllers_for_scenario(
            scenario,
            model_factory,
            workload,
            sim_cfg,
            ctrl_defaults,
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

        if tuning_history_rows:
            save_metrics_csv(scenario_metric_dir / "tuning_history.csv", tuning_history_rows)

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
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run automatic tuning before executing scenarios",
    )
    parser.add_argument(
        "--tuning-workloads",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of workloads to use during tuning",
    )
    args = parser.parse_args()
    main(args.artifacts_root, args.scenario, args.tune, args.tuning_workloads)
