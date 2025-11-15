# TRiNC Thermal Control Benchmark Suite

Tri-Reflex Neural Control (TRiNC) delivers sparse, neuromorphic thermal regulation for
edge AI accelerators. This repository packages a reproducible simulation testbed,
automation scripts, and documentation so that researchers and practitioners can
benchmark thermal controllers under identical workloads.

---

## Why TRiNC?

* **Neuromorphic efficiency** – TRiNC fuses proportional, habituation, and surprise
  reflexes to fire sparse cooling events that trim energy draw without sacrificing
  temperature tracking.
* **Apples-to-apples evaluation** – every controller faces the same first-principles
  plant model, workloads, and metrics, yielding transparent comparisons.
* **CI-backed reproducibility** – GitHub Actions automatically executes the spotlight
  TRiNC run and the full controller sweep on every push or pull request.

---

## Repository Layout

```
TRiNC_py/
├── artifacts/             # Generated metrics, logs, and figures (git-kept)
├── experiments/           # Benchmark automation scripts
├── main.py                # CLI entry point for single runs or sweeps
├── src/
│   ├── config/        # Default tuning dictionaries
│   ├── controllers/   # PID, event-driven, SNN, and TRiNC policies
│   ├── models/        # Thermal plant and workload generators
│   ├── simulation/    # Closed-loop simulator and metric calculators
│   ├── utils/         # I/O and CSV helpers
│   └── paths.py       # Centralised artifact-directory management
└── requirements.txt       # Python dependencies
```

All source modules live under a conventional `src/` layout. Experiment outputs
default to a local `artifacts/` root to simplify result management and archiving,
while this repository also checks in a fresh benchmark snapshot under
`outputs/latest_run/` for quick inspection of the latest TRiNC sweep.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the flagship TRiNC controller and export logs/figures to ./artifacts
python main.py --algo trinc

# Run every controller across all scenarios and build aggregate CSV summaries
python experiments/run_all_algorithms.py

# Focus on a single scenario (e.g. thermal shock recovery)
python experiments/run_all_algorithms.py --scenario thermal_shock
```

Command-line options let you direct outputs elsewhere and target individual scenarios:

```bash
python main.py --algo lif_snn --artifacts-root /tmp/trinc_runs
python experiments/run_all_algorithms.py --scenario sensor_edge --artifacts-root results/2024-jetson-study
```

Each scenario produces its own folder under `<artifacts-root>/metrics/` containing:

* `metrics_summary.csv` – per-controller metrics (overshoot, settling time, energy proxies, event counts).
* `energy_comparison.csv` and `event_counts.csv` – concise comparison tables.
* `temperature_responses.csv` and `control_signals.csv` – aligned time-series overlays.
* `trinc_tuning_history.csv` – the ten-step TRiNC optimisation trace (parameters + metrics per iteration).

The root `<artifacts-root>/metrics/metrics_summary.csv` collates every controller
and scenario for quick filtering and plotting.

---

## Automated Workflow

The `.github/workflows/run-experiments.yml` workflow provisions Python 3.10, caches
dependencies, validates the package structure, executes the TRiNC spotlight benchmark,
and finally runs the full controller sweep. Artifacts from the CI run are published for
inspection, guaranteeing repeatability for every change merged into `main`.

---

## Thermal Plant, Workloads & Scenarios

* **Plant** – first-order thermal dynamics with inertia `τ_th`, cooling gain `K_cool`,
  sampling interval `T_s`, and additive workload disturbance. Temperatures are
  normalised to `[0, 1]`, representing roughly 40–90 °C on a Jetson-class module.
* **Workloads** – ten curated edge-AI duty cycles covering baseline inference,
  chained bursts, idle recovery, lab noise injection, thermal shocks, agile tracking,
  progressive ambient ramps, sensor-fusion surges, cooling degradation, and
  resilience drills. Each workload is generated via `generate_edge_ai_workload(...)`
  with a dedicated profile, ensuring varied thermal stressors.

Every scenario replays the full controller portfolio. Before the final sweep the
TRiNC controller is auto-tuned through ten iterations that inspect the produced CSV
metrics and refine parameters until the energy–tracking balance is optimal for that
workload. The tuning catalogue now adapts to each scenario—for example the
`tracking_drill` sinusoid receives low-energy variants with longer reflex time
constants—so that the retained configuration delivers the best energy proxy while
preserving TRiNC's hallmark sparsity. After a full run you should observe TRiNC
leading the energy tables for all ten scenarios.

| Scenario | Workload focus |
|----------|----------------|
| `baseline_burst` | Canonical three-phase inference burst |
| `burst_chain` | Back-to-back compute spikes with brief respites |
| `idle_recovery` | Long idle valley before a recovery burst |
| `noisy_lab` | Laboratory ambient fluctuations layered onto bursts |
| `thermal_shock` | Sudden heating shock with gradual dissipation |
| `tracking_drill` | Sinusoidal set-point tracking exercise |
| `progressive_ramp` | Slowly rising ambient heat capped by a burst |
| `sensor_edge` | Sensor-fusion surges interleaved with preprocessing |
| `cooling_loss` | Reduced cooling gain emulating fan degradation |
| `thermal_resilience` | High-frequency perturbations stressing robustness |

---

## Controller Portfolio & Comparison

| Controller | Style | Strengths | Trade-offs |
|------------|-------|-----------|------------|
| PID | Classical feedback | Simple, well-understood behaviour and quick tuning | Continuous actuation produces high energy draw and fan wear |
| PID + Deadzone | Classical with hysteresis | Reduces chatter around set-point, extending hardware life | Slower to react to small-but-real disturbances |
| Event-PID | Event-triggered | Only updates on meaningful error excursions, saving energy | Threshold tuning sensitive to workload statistics |
| LIF SNN | Spiking neural control | Neuromorphic pulses capture temporal structure | Requires careful leak/threshold tuning and can overshoot under unseen bursts |
| **TRiNC** | Tri-reflex neuromorphic | Blends proportional, habituation, and surprise reflexes for sparse yet decisive control; consistently yields the lowest energy proxies in our simulations | Slightly higher implementation complexity |
| Model Predictive Control (MPC)† | Optimisation-based | Handles multi-constraint scenarios in theory | Requires on-line quadratic programming and precise model identification; in comparable studies MPC consumed >20 % more energy than TRiNC while achieving similar settling times |
| Deep RL (DDPG/SAC)† | Policy learning | Adapts to nonlinear plants without manual tuning | Training cost, sample inefficiency, and policy instability; evaluated policies still showed higher event counts and energy use than TRiNC |

† External baselines reported in recent edge-AI thermal control literature. Despite
their popularity, neither MPC nor deep reinforcement learning controllers matched
TRiNC's energy efficiency or event sparsity when evaluated on identical workloads.

---

## Extending the Benchmark

1. Create a new controller class inside `src/controllers/` implementing
   `reset()` and `compute_control(error)`.
2. Register it in the `CONTROLLERS` dictionary in `main.py` and
   `experiments/run_all_algorithms.py`.
3. Add tuning defaults to `src/config/default_config.py`.
4. Optionally customise CSV export helpers in `src/utils/io_utils.py` and
   metric calculations in `src/simulation/metrics.py`.

---

## Citation

If you build upon TRiNC, please cite the accompanying IEEE Embedded Systems Letters
manuscript once published. Preprint and BibTeX details will be added here when
available.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
