# TRiNC Thermal Control Benchmark

This repository bundles a small simulation harness for comparing thermal
controllers on an abstracted edge-AI workload.  It focuses on the neuromorphic
Tri-Reflex Neural Control (TRiNC) policy and keeps a set of classical baselines
so you can reproduce the closed-loop experiments described in the accompanying
codebase.

## Features

- Deterministic plant and workload model implemented in `src/models/`.
- Controller portfolio (PID, dead-zone PID, event-triggered PID, leaky-integrate-
  and-fire SNN, and TRINC) under `src/controllers/`.
- Metric computation utilities in `src/simulation/metrics.py` and CSV helpers in
  `src/utils/io_utils.py`.
- Command line entry points for running a single controller (`main.py`) or the
  full sweep across scenarios (`experiments/run_all_algorithms.py`).

## Repository Layout

```
TRiNC_py/
├── artifacts/             # Curated example outputs kept under version control
│   ├── metrics/           # Aggregate CSV summaries from a recorded run
│   └── time_series/       # Sample closed-loop trace
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
location is `./artifacts` inside the repository root.

## Extending the Benchmark

1. Implement a controller class in `src/controllers/` that exposes `reset()` and
   `compute_control(error)`.
2. Register it in the `CONTROLLERS` dictionary inside `main.py` and
   `experiments/run_all_algorithms.py`.
3. Provide default hyperparameters in `src/config/default_config.py` so the
   scripts can instantiate the controller without additional arguments.
