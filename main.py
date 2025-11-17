"""Interactive entry point for local thermal-control experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.config.default_config import (
    apply_params_override,
    default_controller_params,
    default_simulation_params,
    default_tuning_settings,
)
from src.controllers.event_pid import EventPIDController
from src.controllers.lif_snn_controller import LIFSpikingController
from src.controllers.pid_controller import PIDController
from src.controllers.pid_deadzone import PIDDeadzoneController
from src.controllers.trinc_controller import TRINCController
from src.models.thermal_model import ThermalModel
from src.models.workload_profiles import generate_edge_ai_workload
from src.paths import ensure_artifact_layout, get_artifact_layout
from src.simulation.metrics import compute_metrics
from src.simulation.simulate_closed_loop import run_closed_loop
from src.tuning import tune_controllers
from src.utils.io_utils import (
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

WORKLOAD_OPTIONS = [
    "baseline",
    "burst_chain",
    "idle_recovery",
    "noisy_lab",
    "thermal_shock",
    "tracking_drill",
    "progressive_ramp",
    "sensor_edge",
    "cooling_loss",
    "thermal_resilience",
]


def instantiate_controller(name: str, params: Dict[str, float], Ts: float):
    if name in {"pid", "pid_deadzone", "event_pid"}:
        return CONTROLLERS[name](Ts=Ts, **params)
    return CONTROLLERS[name](**params)


def prompt_list(prompt: str, options: List[str]) -> List[str]:
    print(prompt)
    for idx, opt in enumerate(options, start=1):
        print(f"  [{idx}] {opt}")
    selection = input("Enter comma-separated choices or 'all': ").strip().lower()
    if selection in {"all", ""}:
        return options
    indices = [int(x) for x in selection.split(",") if x.strip().isdigit()]
    chosen = [options[i - 1] for i in indices if 1 <= i <= len(options)]
    return chosen or options


def run_experiments(
    controllers: Iterable[str],
    workloads: Iterable[str],
    sim_cfg: Dict[str, float],
    controller_params: Dict[str, Dict[str, float]],
    artifacts_root: Path | None = None,
) -> None:
    artifacts = get_artifact_layout(artifacts_root)
    ensure_artifact_layout(artifacts)

    Ts = sim_cfg["Ts"]
    horizon = int(round(sim_cfg["duration"] / Ts))

    all_metrics = []
    time_series_records = {}

    for workload_name in workloads:
        workload = generate_edge_ai_workload(Ts, horizon, profile=workload_name)
        workload_dir = artifacts.time_series / workload_name
        workload_dir.mkdir(parents=True, exist_ok=True)

        for algo_name in controllers:
            params = controller_params[algo_name]
            controller = instantiate_controller(algo_name, params, Ts)
            model = ThermalModel(sim_cfg["tau_th"], sim_cfg["K_heat"], sim_cfg["K_cool"], Ts)

            results = run_closed_loop(
                model=model,
                controller=controller,
                workload=workload,
                T_ref=sim_cfg["T_ref"],
                Ts=Ts,
                T0=sim_cfg["T0"],
                algo_name=algo_name,
            )

            extra_cols = {}
            event_flags = None
            if "gP" in results:
                extra_cols = {key: np.array(results[key]) for key in ("gP", "gH", "gS")}
                event_flags = (
                    np.abs(extra_cols["gP"]) + np.abs(extra_cols["gH"]) + np.abs(extra_cols["gS"])
                )

            csv_path = workload_dir / f"{algo_name}.csv"
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
            metrics_record = {"algo_name": algo_name, "workload": workload_name, **metrics}
            all_metrics.append(metrics_record)

            time_series_records.setdefault(algo_name, {})[workload_name] = {
                "time": np.array(results["time"], dtype=float),
                "temperature": np.array(results["temperature"], dtype=float),
                "control": np.array(results["control"], dtype=float),
                "reference": np.full_like(results["time"], sim_cfg["T_ref"], dtype=float),
            }

    save_metrics_csv(artifacts.metrics / "interactive_metrics.csv", all_metrics)
    save_metric_subset_csv(
        artifacts.metrics / "interactive_energy.csv",
        all_metrics,
        columns=["algo_name", "workload", "E1", "E2"],
    )
    save_metric_subset_csv(
        artifacts.metrics / "interactive_events.csv",
        all_metrics,
        columns=["algo_name", "workload", "N_events", "N_du"],
    )

    # Collate responses for the first workload to visualize tracking and control.
    for workload_name in workloads:
        time_vector = next(iter(time_series_records.values()))[workload_name]["time"]
        save_collated_time_series_csv(
            artifacts.time_series / f"temperature_{workload_name}.csv",
            time_vector,
            {algo: series[workload_name]["temperature"] for algo, series in time_series_records.items()},
            reference=next(iter(time_series_records.values()))[workload_name]["reference"],
            reference_name="T_ref",
        )
        save_collated_time_series_csv(
            artifacts.time_series / f"control_{workload_name}.csv",
            time_vector,
            {algo: series[workload_name]["control"] for algo, series in time_series_records.items()},
        )

    print("\nInteractive run complete. Artifacts stored under", artifacts.root)


def main() -> None:
    print("TRiNC/Edge-AI Thermal Benchmark")
    controllers = prompt_list("Select controllers to run", list(CONTROLLERS.keys()))
    workloads = prompt_list("Select workload profiles", WORKLOAD_OPTIONS)

    mode = input("Run quick sanity check? (y/N): ").strip().lower()
    sim_cfg = default_simulation_params()
    if mode == "y":
        sim_cfg["duration"] = max(12.0, 0.5 * sim_cfg["duration"])

    should_tune = input("Run automatic tuning before simulation? (y/N): ").strip().lower() == "y"

    controller_params = default_controller_params()
    if should_tune:
        print("Running tuner across selected controllers...")
        tuning_settings = default_tuning_settings()
        tuned_params, _ = tune_controllers(
            controller_subset=controllers,
            tuning_workloads=workloads,
            sim_overrides=sim_cfg,
            tuning_settings=tuning_settings,
        )
        controller_params = apply_params_override(controller_params, tuned_params)

    run_experiments(controllers, workloads, sim_cfg, controller_params)


if __name__ == "__main__":
    main()
