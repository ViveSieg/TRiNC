"""Run all controllers, collect metrics, and produce figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.paths import ensure_artifact_layout, get_artifact_layout
from src.simulation.metrics import compute_metrics
from src.simulation.simulate_closed_loop import run_closed_loop
from src.utils.io_utils import (
    save_collated_time_series_csv,
    save_metrics_csv,
    save_metric_subset_csv,
    save_time_series_csv,
)

CONTROLLERS = {
    "pid": PIDController,
    "pid_deadzone": PIDDeadzoneController,
    "event_pid": EventPIDController,
    "lif_snn": LIFSpikingController,
    "trinc": TRINCController,
}


def instantiate(name: str, params, Ts: float):
    if name in {"pid", "pid_deadzone", "event_pid"}:
        return CONTROLLERS[name](Ts=Ts, **params)
    return CONTROLLERS[name](**params)


def main(artifacts_root: Path | None = None) -> None:
    artifacts = get_artifact_layout(artifacts_root)
    ensure_artifact_layout(artifacts)

    sim_cfg = default_simulation_params()
    ctrl_cfgs = default_controller_params()

    duration = sim_cfg["duration"]
    Ts = sim_cfg["Ts"]
    horizon = int(duration / Ts)

    model = ThermalModel(sim_cfg["tau_th"], sim_cfg["K_cool"], Ts)
    workload = generate_edge_ai_workload(Ts, horizon)

    metrics_records = []
    time_series_results = {}

    for name in CONTROLLERS:
        print(f"Simulating {name} controller")
        controller = instantiate(name, ctrl_cfgs[name], Ts)
        results = run_closed_loop(
            model=model,
            controller=controller,
            workload=workload,
            T_ref=sim_cfg["T_ref"],
            Ts=Ts,
            T0=sim_cfg["T0"],
            algo_name=name,
        )

        extra_cols = {}
        if "gP" in results:
            extra_cols = {key: results[key] for key in ("gP", "gH", "gS")}
        csv_path = artifacts.time_series / f"{name}_results.csv"
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
        metrics_record = {"algo_name": name}
        metrics_record.update(metrics)
        metrics_records.append(metrics_record)

        time_series_results[name] = {
            "time": results["time"],
            "temperature": results["temperature"],
            "control": results["control"],
            "T_ref": np.full_like(results["time"], sim_cfg["T_ref"]),
        }

    save_metrics_csv(artifacts.metrics / "metrics_summary.csv", metrics_records)

    save_metric_subset_csv(
        artifacts.metrics / "energy_comparison.csv",
        metrics_records,
        columns=["algo_name", "E1", "E2"],
    )
    save_metric_subset_csv(
        artifacts.metrics / "event_counts.csv",
        metrics_records,
        columns=["algo_name", "N_events", "N_du"],
    )

    if time_series_results:
        first = next(iter(time_series_results.values()))
        time_vector = first["time"]
        reference = first.get("T_ref")

        save_collated_time_series_csv(
            artifacts.metrics / "temperature_responses.csv",
            time_vector,
            {name: data["temperature"] for name, data in time_series_results.items()},
            reference=reference,
            reference_name="T_ref",
        )
        save_collated_time_series_csv(
            artifacts.metrics / "control_signals.csv",
            time_vector,
            {name: data["control"] for name, data in time_series_results.items()},
        )

    try:
        root_display = artifacts.root.relative_to(Path.cwd())
    except ValueError:
        root_display = artifacts.root
    print(f"All simulations complete. Results saved under {root_display}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full TRiNC benchmark suite")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help="Optional path for storing generated metrics, logs, and figures",
    )
    args = parser.parse_args()
    main(args.artifacts_root)
