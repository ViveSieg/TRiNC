# TRiNC: Multi-Controller Thermal Control Benchmarking Platform

A comprehensive simulation platform for comparing thermal controllers on edge-AI workloads using a gray-box physics-informed neural network (PiNN) digital twin. This repository implements the Tri-Reflex Neural Control (TRiNC) algorithm alongside classical baselines (PID variants, SNN) to enable rigorous, reproducible closed-loop experiments for thermal regulation in embedded systems.

## Overview

TRiNC addresses the challenge of thermal control in edge-AI accelerators by combining:
- **Gray-box digital twin**: Physics-informed neural network (PiNN) that learns natural heating patterns from real TPU data and injects physical cooling equations
- **All-Pareto optimization**: Bayesian hyperparameter optimization for all controllers to ensure fair comparison
- **Golden Quartet scenarios**: Four standardized test scenarios for comprehensive evaluation
- **Unified CLI**: Professional command-line interface built with Typer, Rich, and Questionary

## Key Features

- **Multi-controller portfolio**: Five distinct controllers (PID, PID with deadzone, Event-triggered PID, LIF SNN, and TRiNC) with standardized interfaces
- **PiNN thermal model**: Gray-box model combining learned natural heating with physics-based cooling control
- **Bayesian optimization**: Multi-objective optimization using Optuna to find Pareto frontiers for all controllers
- **Real-world data integration**: Support for Google Edge TPU workload traces (`TPU_Google_Edge.pkl`)
- **Comprehensive metrics**: ITAE, Energy, Overshoot, Rise Time, and Settling Time computation
- **Professional CLI**: Interactive command-line interface with Rich tables and progress indicators

## Architecture

### Directory Structure

```
TRiNC/
├── data/                    # TPU data files (TPU_Google_Edge.pkl)
├── artifacts/               # Generated outputs
│   ├── tuning/             # Optimization results (tuning_history_*.csv)
│   ├── pinn_model.pt       # Trained PiNN model weights
│   └── validation_results.csv
├── src/
│   ├── config/
│   │   ├── schema.py       # Strongly-typed configuration (dataclasses)
│   │   └── presets.py      # Configuration presets and search spaces
│   ├── controllers/
│   │   ├── base.py         # BaseController abstract base class
│   │   ├── registry.py     # Factory pattern for controller instantiation
│   │   └── *.py            # Controller implementations (5 controllers)
│   ├── models/
│   │   ├── pinn_thermal.py # Gray-box PiNN thermal model
│   │   ├── thermal_model.py # Traditional first-order linear model
│   │   └── workload_loader.py # Golden Quartet scenario loader
│   ├── simulation/
│   │   └── executor.py     # Universal simulation executor
│   ├── tuning/
│   │   └── bayesian.py     # Bayesian optimizer (Optuna)
│   └── utils/
│       ├── metrics.py      # Performance metrics computation
│       ├── io_utils.py     # CSV I/O utilities
│       └── plotting.py    # Visualization tools
├── main.py                  # Unified CLI entry point
├── requirements.txt        # Python dependencies
└── README.md
```

### Core Components

#### Gray-Box PiNN Thermal Model

The `GrayBoxThermalModel` implements the physics fusion formula:

\[
\frac{dT}{dt} = \text{NN}(T_{\text{curr}}, W_{\text{curr}}) - \gamma \cdot u
\]

where:
- `NN(T, W)` learns natural heating patterns from real TPU data
- \(-\gamma \cdot u\) is the manually injected physical cooling term
- This approach solves the challenge of training on data without control signals

The model uses Min-Max normalization for both inputs and outputs to ensure neural network convergence.

#### Controller Registry System

All controllers inherit from `BaseController` and are registered via the `@register_controller` decorator:

- **PID**: Classic proportional-integral-derivative feedback
- **PID with deadzone**: Suppresses small error corrections for efficiency
- **Event-triggered PID**: Evaluates PID law on-demand to reduce updates
- **LIF SNN**: Leaky integrate-and-fire spiking neural network
- **TRiNC**: Tri-Reflex Neural Control with P, H, and S gates

#### Golden Quartet Scenarios

Four standardized test scenarios for comprehensive evaluation:

1. **real_trace_replay**: Primary scenario for optimization. Randomly samples real TPU workload sequences from the test set.
2. **step_response**: Synthetic step function for transient analysis (Rise Time, Overshoot).
3. **sine_tracking**: Synthetic sine wave for dynamic tracking performance (ITAE).
4. **steady_state**: Constant high-load for efficiency evaluation (Energy).

Synthetic scenarios (1D signals) are automatically adapted to multi-dimensional inputs using the `adapt_1d_to_nd()` function, which broadcasts synthetic signals onto templates from real data.

#### Bayesian Optimization

The `BayesianOptimizer` class:
- Optimizes all controllers on `real_trace_replay` to ensure real-world applicability
- Uses normalized objectives: `loss = w1 * normalized_ITAE + w2 * normalized_Energy`
- Saves all trial results to `artifacts/tuning/tuning_history_{controller_name}.csv`
- Extracts Pareto frontiers for multi-controller comparison

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- NumPy >= 1.24
- Pandas >= 1.5
- Matplotlib >= 3.7
- PyTorch >= 2.0.0
- Optuna >= 3.0.0
- Typer >= 0.9.0
- Rich >= 13.0.0
- Questionary >= 2.0.0

## Quick Start

### Execute Full Scientific Protocol

Run the complete benchmarking workflow:

```bash
python main.py execute-protocol
```

This command:
1. Trains the PiNN model (if not already trained)
2. Optimizes all 5 controllers using Bayesian optimization
3. Validates optimized controllers on Golden Quartet scenarios
4. Generates Pareto frontier comparison plot

### Individual Commands

```bash
# Train PiNN digital twin
python main.py train-digital-twin

# Optimize all controllers (batch run for overnight execution)
python main.py optimize

# Run simulation experiments
python main.py simulate

# Generate Pareto frontier plot
python main.py analyze

# Clean generated artifacts
python main.py clean
```

## Usage Examples

### Hyperparameter Optimization

```bash
python main.py optimize
```

The CLI will prompt you to:
- Choose optimization scope (all controllers or specific ones)
- Select controllers to optimize
- Results are automatically saved to `artifacts/tuning/tuning_history_{controller_name}.csv`

### Running Simulations

```bash
python main.py simulate
```

Interactive prompts allow you to:
- Select controller(s): single or all controllers
- Choose scenario(s): single scenario or all Golden Quartet
- Results are displayed in a Rich table and can be saved

### Analysis and Visualization

```bash
python main.py analyze
```

Generates a multi-controller Pareto frontier plot showing:
- ITAE vs Energy trade-offs for all controllers
- TRiNC dominance visualization
- Saved to `artifacts/pareto_comparison.png`

## Controller Portfolio

All controllers implement the `BaseController` interface with `reset()` and `compute_control(error)` methods:

| Controller | Description | Key Parameters |
|------------|-------------|----------------|
| **PID** | Classic proportional-integral-derivative feedback | Kp, Ki, Kd |
| **PID_Deadzone** | PID with deadzone integration for sparse actuation | Kp, Ki, Kd, deadband |
| **Event_PID** | Event-triggered PID that updates on-demand | Kp, Ki, Kd, error_threshold, decay |
| **LIF_SNN** | Leaky integrate-and-fire spiking neural network | leak, threshold, pulse_magnitude |
| **TRiNC** | Tri-Reflex Neural Control with adaptive gains | rho, theta_h, tau_p, w_h, a_p, a_h, a_s |

## Configuration

### Simulation Configuration

Default simulation parameters (via `get_default_simulation_config()`):
- Sampling time: `Ts = 0.1 s` (10 Hz control loop)
- Duration: `duration = 22.0 s`
- Ambient temperature: `T_amb = 0.4` (normalized)
- Reference temperature: `T_ref = 0.62` (normalized)
- Initial temperature: `T0 = 0.55` (normalized)

### Model Configuration

PiNN model parameters (via `get_default_model_config()`):
- Physics cooling coefficient: `physics_gamma = 0.1`
- Input dimension: `input_dim = 8` (from TPU data)
- Hidden layer width: `hidden_dim = 64`

### Search Spaces

Wide, fair parameter search spaces are defined in `get_search_space()` for all controllers to prevent "weak baseline" criticism. Each controller has carefully tuned bounds that allow comprehensive exploration of the design space.

## Metrics

The platform computes comprehensive performance metrics:

- **ITAE** (Integral Time-weighted Absolute Error): Tracking accuracy
- **Energy**: Control effort consumption (sum of u²)
- **Overshoot**: Safety metric (maximum temperature above reference)
- **Rise Time**: Time to reach 90% of final value from 10%
- **Settling Time**: Time to reach within 2% of reference

## Extending the Platform

### Adding a New Controller

1. Create a new controller class in `src/controllers/`:
   ```python
   from .base import BaseController
   from .registry import register_controller
   
   @register_controller("my_controller")
   @dataclass
   class MyController(BaseController):
       # Parameters
       param1: float
       
       def reset(self) -> None:
           # Initialize state
           pass
       
       def compute_control(self, error: float) -> float:
           # Control logic
           return control_signal
   ```

2. Add default parameters to `src/config/presets.py`:
   ```python
   def get_default_controller_params():
       return {
           # ... existing controllers
           "my_controller": {"param1": 0.5, ...},
       }
   ```

3. Define search space in `get_search_space()`:
   ```python
   def get_search_space(algo_name: str):
       search_spaces = {
           # ... existing spaces
           "my_controller": {"param1": (0.0, 1.0), ...},
       }
   ```

The controller will automatically be available in the CLI and optimization workflows.

## Data Format

The platform expects TPU data in the following format:
- **Input**: Shape `(N, Seq, Feat)` - Multi-dimensional workload metrics
- **Output**: Shape `(N, Seq, H, W)` - Thermal maps (automatically extracts T_max)

Data is automatically split into 80% training and 20% test sets. The PiNN model learns natural heating patterns from training data and is validated on the test set.
