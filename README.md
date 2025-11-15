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
│   └── trinc/
│       ├── config/        # Default tuning dictionaries
│       ├── controllers/   # PID, event-driven, SNN, and TRiNC policies
│       ├── models/        # Thermal plant and workload generators
│       ├── simulation/    # Closed-loop simulator and metric calculators
│       ├── utils/         # I/O and plotting helpers
│       └── paths.py       # Centralised artifact-directory management
└── requirements.txt       # Python dependencies
```

The new `src/trinc` package consolidates all source modules in a conventional
`src/`-layout, and experiment outputs now land under a single `artifacts/` root to
simplify result management and archiving.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the flagship TRiNC controller and export logs/figures to ./artifacts
python main.py --algo trinc

# Run every controller and build aggregate figures/metrics
python experiments/run_all_algorithms.py
```

Command-line options let you direct outputs elsewhere:

```bash
python main.py --algo lif_snn --artifacts-root /tmp/trinc_runs
python experiments/run_all_algorithms.py --artifacts-root results/2024-jetson-study
```

All CSV logs are stored in `<artifacts-root>/time_series/`, aggregate metrics in
`<artifacts-root>/metrics/metrics_summary.csv`, and publication-quality figures in
`<artifacts-root>/figures/`.

---

## Automated Workflow

The `.github/workflows/run-experiments.yml` workflow provisions Python 3.10, caches
dependencies, validates the package structure, executes the TRiNC spotlight benchmark,
and finally runs the full controller sweep. Artifacts from the CI run are published for
inspection, guaranteeing repeatability for every change merged into `main`.

---

## Thermal Plant & Workloads

* **Plant** – first-order thermal dynamics with inertia `τ_th`, cooling gain `K_cool`,
  sampling interval `T_s`, and additive workload disturbance. Temperatures are
  normalised to `[0, 1]`, representing roughly 40–90 °C on a Jetson-class module.
* **Workload** – a bursty on-device learning scenario: calm background load, followed
  by intense inference spikes, then a moderate recovery phase. The profile rewards
  controllers that react rapidly yet remain quiescent during steady periods.

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

1. Create a new controller class inside `src/trinc/controllers/` implementing
   `reset()` and `compute_control(error)`.
2. Register it in the `CONTROLLERS` dictionary in `main.py` and
   `experiments/run_all_algorithms.py`.
3. Add tuning defaults to `src/trinc/config/default_config.py`.
4. Optionally customise plots or metrics in `src/trinc/utils/plotting.py` and
   `src/trinc/simulation/metrics.py`.

---

## Citation

If you build upon TRiNC, please cite the accompanying IEEE Embedded Systems Letters
manuscript once published. Preprint and BibTeX details will be added here when
available.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
