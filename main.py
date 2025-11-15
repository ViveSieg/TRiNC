"""CLI entry point for running thermal control experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from src.trinc.config.default_config import (
    default_controller_params,
    default_simulation_params,
)
from src.trinc.controllers.event_pid import EventPIDController
from src.trinc.controllers.lif_snn_controller import LIFSpikingController
from src.trinc.controllers.pid_controller import PIDController
from src.trinc.controllers.pid_deadzone import PIDDeadzoneController
from src.trinc.controllers.trinc_controller import TRINCController
from src.trinc.models.thermal_model import ThermalModel
from src.trinc.models.workload_profiles import generate_edge_ai_workload
from src.trinc.paths import ArtifactLayout, ensure_artifact_layout, get_artifact_layout
from src.trinc.simulation.metrics import compute_metrics
from src.trinc.simulation.simulate_closed_loop import run_closed_loop
from src.trinc.utils.io_utils import save_time_series_csv

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


def run_single_algorithm(algo_name: str, artifacts: ArtifactLayout) -> Dict[str, float]:
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

    extra_cols = {}
    if "gP" in results:
        extra_cols = {key: results[key] for key in ("gP", "gH", "gS")}
    csv_path = artifacts.time_series / f"{algo_name}_results.csv"
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
    return metrics


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
        metrics_summary = {}
        for algo in CONTROLLERS:
            print(f"Running {algo}...")
            metrics_summary[algo] = run_single_algorithm(algo, artifacts)
            print(metrics_summary[algo])
    else:
        metrics = run_single_algorithm(args.algo, artifacts)
        print(f"Metrics for {args.algo}: {metrics}")


if __name__ == "__main__":
    main()
