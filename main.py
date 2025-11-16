"""CLI entry point for running thermal control experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.config.default_config import (
    default_controller_params,
    default_simulation_params,
)
from src.controllers.event_pid import EventPIDController
from src.controllers.lif_snn_controller import LIFSpikingController
from src.controllers.pid_controller import PIDController
from src.controllers.pid_deadzone import PIDDeadzoneController
from src.controllers.trinc_controller import TRINCController
from src.models.thermal_model import ThermalModel
from src.models.workload_profiles import generate_edge_ai_workload
from src.paths import ArtifactLayout, ensure_artifact_layout, get_artifact_layout
from src.simulation.metrics import compute_metrics
from src.simulation.simulate_closed_loop import run_closed_loop
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


def instantiate_controller(name: str, params: Dict[str, float], Ts: float):
    if name in {"pid", "pid_deadzone", "event_pid"}:
        return CONTROLLERS[name](Ts=Ts, **params)
    return CONTROLLERS[name](**params)


def run_single_algorithm(
    algo_name: str, artifacts: ArtifactLayout
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    sim_cfg = default_simulation_params()
    ctrl_cfg = default_controller_params()[algo_name]

    duration = sim_cfg["duration"]
    Ts = sim_cfg["Ts"]
    horizon = int(duration / Ts)

    model = ThermalModel(sim_cfg["tau_th"], sim_cfg["K_cool"], Ts)
    workload = generate_edge_ai_workload(Ts, horizon)

    controller = instantiate_controller(algo_name, ctrl_cfg, Ts)
    results = run_closed_loop(
        model=model,
        controller=controller,
        workload=workload,
        T_ref=sim_cfg["T_ref"],
        Ts=Ts,
        T0=sim_cfg["T0"],
        algo_name=algo_name,
    )

    single_time_dir = artifacts.time_series / "single"
    single_metric_dir = artifacts.metrics / "single"
    single_time_dir.mkdir(parents=True, exist_ok=True)
    single_metric_dir.mkdir(parents=True, exist_ok=True)

    extra_cols = {}
    if "gP" in results:
        extra_cols = {key: results[key] for key in ("gP", "gH", "gS")}
    csv_path = single_time_dir / f"{algo_name}.csv"
    save_time_series_csv(
        csv_path,
        time=results["time"],
        temperature=results["temperature"],
        T_ref=sim_cfg["T_ref"],
        control=results["control"],
        error=results["error"],
        extra_columns=extra_cols if extra_cols else None,
    )

    event_flags = None
    if "gP" in results:
        event_flags = np.abs(results["gP"]) + np.abs(results["gH"]) + np.abs(results["gS"])
    metrics = compute_metrics(
        time=results["time"],
        temperature=results["temperature"],
        control=results["control"],
        T_ref=sim_cfg["T_ref"],
        event_flags=event_flags,
    )

    metrics_record = {"algo_name": algo_name, **metrics}
    save_metrics_csv(single_metric_dir / f"{algo_name}.csv", [metrics_record])

    time_vector = np.array(results["time"], dtype=float)
    time_series_record = {
        "time": time_vector,
        "temperature": np.array(results["temperature"], dtype=float),
        "control": np.array(results["control"], dtype=float),
        "reference": np.full_like(time_vector, sim_cfg["T_ref"], dtype=float),
    }

    return metrics_record, time_series_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge AI thermal control benchmark")
    parser.add_argument(
        "--algo",
        type=str,
        choices=list(CONTROLLERS.keys()) + ["all"],
        default="trinc",
        help="Controller to execute",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help="Optional path for storing generated artifacts",
    )
    args = parser.parse_args()

    artifacts = get_artifact_layout(args.artifacts_root)
    ensure_artifact_layout(artifacts)

    if args.algo == "all":
        summary_rows = []
        time_series_records = {}

        for algo in CONTROLLERS:
            print(f"Running {algo}...")
            metrics_record, time_series_record = run_single_algorithm(algo, artifacts)
            summary_rows.append(metrics_record)
            time_series_records[algo] = time_series_record
            print(metrics_record)

        single_metric_dir = artifacts.metrics / "single"
        single_metric_dir.mkdir(parents=True, exist_ok=True)
        save_metrics_csv(single_metric_dir / "metrics_summary.csv", summary_rows)
        save_metric_subset_csv(
            single_metric_dir / "energy_comparison.csv",
            summary_rows,
            columns=["algo_name", "E1", "E2"],
        )
        save_metric_subset_csv(
            single_metric_dir / "event_counts.csv",
            summary_rows,
            columns=["algo_name", "N_events", "N_du"],
        )

        single_time_dir = artifacts.time_series / "single"
        single_time_dir.mkdir(parents=True, exist_ok=True)
        if time_series_records:
            first_algo = next(iter(time_series_records))
            time_vector = time_series_records[first_algo]["time"]
            reference = time_series_records[first_algo]["reference"]

            save_collated_time_series_csv(
                single_time_dir / "temperature_responses.csv",
                time_vector,
                {name: data["temperature"] for name, data in time_series_records.items()},
                reference=reference,
                reference_name="T_ref",
            )
            save_collated_time_series_csv(
                single_time_dir / "control_signals.csv",
                time_vector,
                {name: data["control"] for name, data in time_series_records.items()},
            )
    else:
        metrics_record, _ = run_single_algorithm(args.algo, artifacts)
        print(f"Metrics for {args.algo}: {metrics_record}")


if __name__ == "__main__":
    main()
