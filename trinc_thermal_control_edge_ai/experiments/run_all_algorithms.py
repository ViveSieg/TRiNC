"""Run all controllers, collect metrics, and produce figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config.default_config import default_simulation_params, default_controller_params
from src.controllers.pid_controller import PIDController
from src.controllers.pid_deadzone import PIDDeadzoneController
from src.controllers.event_pid import EventPIDController
from src.controllers.lif_snn_controller import LIFSpikingController
from src.controllers.trinc_controller import TRINCController
from src.models.thermal_model import ThermalModel
from src.models.workload_profiles import generate_edge_ai_workload
from src.simulation.simulate_closed_loop import run_closed_loop
from src.simulation.metrics import compute_metrics
from src.utils.io_utils import save_time_series_csv, save_metrics_csv
from src.utils.plotting import (
    plot_control_signals,
    plot_energy_bar,
    plot_event_counts,
    plot_temperature_responses,
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


def main() -> None:
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
        csv_path = Path("data/time_series") / f"{name}_results.csv"
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

    save_metrics_csv(Path("data/summary/metrics_summary.csv"), metrics_records)

    metrics_by_algo = {record["algo_name"]: record for record in metrics_records}

    plot_temperature_responses(time_series_results, Path("figures/temperature_responses.png"))
    plot_control_signals(time_series_results, Path("figures/control_signals.png"))
    plot_energy_bar(metrics_by_algo, Path("figures/energy_comparison_bar.png"))
    plot_event_counts(metrics_by_algo, Path("figures/event_counts_bar.png"))

    print("All simulations complete. Results saved to data/ and figures/ directories.")


if __name__ == "__main__":
    main()
