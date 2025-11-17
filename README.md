# TRiNC Thermal Control Benchmark

This repository bundles a small simulation harness for comparing thermal
controllers on abstracted edge-AI workloads.  It focuses on the neuromorphic
Tri-Reflex Neural Control (TRiNC) policy and keeps a set of classical baselines
so you can reproduce the closed-loop experiments described in the accompanying
codebase.

## Features

- Deterministic plant and workload model implemented in `src/models/`.
- Controller portfolio (PID, dead-zone PID, event-triggered PID, leaky-integrate-
  and-fire SNN, and TRiNC) under `src/controllers/`.
- Metric computation utilities in `src/simulation/metrics.py` and CSV helpers in
  `src/utils/io_utils.py`.
- Command line entry points for running a single controller (`main.py`) or the
  full sweep across scenarios (`experiments/run_all_algorithms.py`).

## Simulation Stack

- **Thermal plant** – `src/models/thermal_model.py` provides a reduced-order
  model that captures heat accumulation, dissipation, and actuator response for
  embedded devices.
- **Workload generator** – `src/models/workload_profiles.py` synthesises activity
  traces ranging from calm idle behaviour to bursty inference so controllers can
  be evaluated under diverse edge deployments.
- **Metric pipeline** – `src/simulation/metrics.py` distils time-series outputs
  into regulation quality, energy proxies, and sparsity indicators that feed the
  comparison reports.

## Controller Portfolio

All controllers expose the same `compute_control` interface, making it easy to
swap policies during experiments:

- **PID** delivers a classic proportional–integral–derivative feedback loop for
  stable regulation in nominal conditions.
- **PID with dead zone** suppresses small error corrections to trade reactivity
  for actuator efficiency.
- **Event-triggered PID** evaluates the PID law on demand, reducing updates when
  thermal states remain steady.
- **Leaky integrate-and-fire SNN** emulates neuromorphic processing to deliver
  sparse control impulses for energy-sensitive workloads.
- **TRiNC** fuses reflex loops with adaptive gains so the controller stays agile
  under abrupt workload shifts.

## Scenario Catalogue

The batch runner exercises controllers on a family of workload narratives
defined in `experiments/run_all_algorithms.py`:

- `baseline_burst` – nominal inference bursts with recovery windows.
- `burst_chain` – consecutive bursts that stress cooling recovery.
- `idle_recovery` – prolonged idle phases followed by renewed demand.
- `noisy_lab` – disturbances injected on top of the baseline profile.
- `thermal_shock` – sudden heating events with sluggish dissipation.
- `tracking_drill` – smooth tracking of oscillatory targets.
- `progressive_ramp` – gradually rising ambient conditions ending in a burst.
- `sensor_edge` – intermittent sensor fusion workloads.
- `cooling_loss` – degraded actuator authority representing airflow issues.
- `thermal_resilience` – high-frequency thermal perturbations.

### Thermal Plant and Workload Modeling

We model the hotspot dynamics with a normalized first-order plant,
\(T_{k+1} = a T_k + (1 - a)(K_{\text{heat}} w_k - K_{\text{cool}} u_k)\),
where \(a = \exp(-T_s / \tau_{th})\). The default parameters reflect a
few-watt edge AI accelerator: \(\tau_{th} = 3\,\text{s}\) places the thermal
time constant in a typical 1–5 s envelope; \(K_{\text{heat}} = 1.0\) means a
fully loaded device without cooling drifts toward the upper end of the
normalized range; \(K_{\text{cool}} = 0.6\) means maximum cooling under full
load pulls the steady temperature down to roughly \(T \approx 0.4\). The
normalization anchors 0 to ~40 °C and 1 to ~90 °C, matching practical silicon
operating limits without embedding any hidden offsets in the code.

Workloads are generated via `generate_edge_ai_workload` as normalized traces
that mirror common edge-AI behaviours: bursty camera inference, idle recovery
after batch jobs, slow ambient ramps, and composite stress tests. Each profile
stays within \([0, 1]\) and is shared across all controllers, ensuring a fair
comparison. The signals are deliberately synthetic rather than fitted to a
specific TPU trace, giving a controlled benchmark that still stresses transient
and steady-state regulation.

## Repository Layout

```
TRiNC_py/
├── artifacts/             # Curated example outputs kept under version control
│   ├── metrics/
│   │   └── sample_run/    # Aggregate CSV summaries from a recorded run
│   └── time_series/
│       └── sample_run/    # Sample closed-loop trace
├── .github/workflows/     # Automation for GitHub Actions
├── experiments/           # Batch scripts and notebooks for comparisons
├── src/                   # Source package with configs, controllers, models
├── main.py                # CLI for running one controller
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

The files inside `artifacts/` illustrate the expected directory structure when
running the tooling.  New experiment runs can be redirected to another location
via the `--artifacts-root` flag if you do not want to overwrite the curated
snapshot.

## Automation

The repository ships with `.github/workflows/run-experiments.yml` so that every
push, pull request, or manual dispatch triggers the full evaluation sweep.  The
workflow installs dependencies, validates the source tree, runs the spotlight
TRiNC benchmark via `main.py`, and then executes the multi-scenario sweep in
`experiments/run_all_algorithms.py`.  Outputs are written under
`artifacts/ci/` during CI runs and uploaded as workflow artifacts for further
inspection.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the TRINC controller using the default settings
python main.py --algo trinc

# Execute every controller across all scenarios
python experiments/run_all_algorithms.py
```

Both entry points accept `--artifacts-root <path>` to choose where the generated
metrics, figures, and time-series CSV files should be written.  The default
location is `./artifacts` inside the repository root.  The single-controller CLI
stores its results under `artifacts/metrics/single/` and
`artifacts/time_series/single/`, while the batch runner organises outputs by
scenario beneath the same top-level folders.

## Extending the Benchmark

1. Implement a controller class in `src/controllers/` that exposes `reset()` and
   `compute_control(error)`.
2. Register it in the `CONTROLLERS` dictionary inside `main.py` and
   `experiments/run_all_algorithms.py`.
3. Provide default hyperparameters in `src/config/default_config.py` so the
   scripts can instantiate the controller without additional arguments.
