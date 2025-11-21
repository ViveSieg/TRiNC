"""Unified CLI for TRiNC multi-controller benchmarking platform."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import questionary
import pandas as pd
import numpy as np

from src.config.schema import SimulationConfig
from src.config.presets import (
    get_default_simulation_config,
    get_default_model_config,
    get_default_controller_params,
    get_search_space,
    get_golden_quartet_scenarios,
)
from src.controllers.registry import create_controller, list_controllers
from src.models.thermal_model import ThermalModel
from src.models.pinn_thermal import GrayBoxThermalModel, load_data
from src.models.workload_loader import WorkloadLoader, GOLDEN_QUARTET
from src.simulation.executor import SimulationExecutor
from src.tuning.bayesian import BayesianOptimizer
from src.utils.io_utils import save_metrics_csv, save_time_series_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="TRiNC Multi-Controller Benchmarking Platform")
console = Console()

# Global paths
DATA_PATH = Path("data/TPU_Google_Edge.pkl")
ARTIFACTS_DIR = Path("artifacts")
TUNING_DIR = ARTIFACTS_DIR / "tuning"
MODEL_PATH = ARTIFACTS_DIR / "pinn_model.pt"


def ensure_pinn_model() -> tuple[GrayBoxThermalModel, WorkloadLoader]:
    """Ensure PiNN model is trained and loaded.
    
    Returns
    -------
    tuple[GrayBoxThermalModel, WorkloadLoader]
        Trained model and workload loader
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    if MODEL_PATH.exists():
        console.print(f"[green]Loading trained PiNN model from {MODEL_PATH}[/green]")
        model = GrayBoxThermalModel.load(MODEL_PATH)
        # Load test data for workload loader
        _, _, test_inputs, test_outputs = load_data(DATA_PATH)
        workload_loader = WorkloadLoader(
            test_data=(test_inputs, test_outputs),
            input_dim=model.input_dim,
        )
    else:
        console.print("[yellow]Training PiNN model...[/yellow]")
        train_inputs, train_outputs, test_inputs, test_outputs = load_data(DATA_PATH)
        input_dim = train_inputs.shape[-1]
        
        model_config = get_default_model_config(input_dim=input_dim)
        model = GrayBoxThermalModel(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            physics_gamma=model_config.physics_gamma,
            Ts=get_default_simulation_config().Ts,
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training PiNN...", total=None)
            model.train(train_inputs, train_outputs, epochs=100)
            progress.update(task, completed=True)
        
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)
        console.print(f"[green]Model saved to {MODEL_PATH}[/green]")
        
        workload_loader = WorkloadLoader(
            test_data=(test_inputs, test_outputs),
            input_dim=input_dim,
        )
    
    return model, workload_loader


@app.command()
def execute_protocol() -> None:
    """Execute Full Scientific Protocol: Train PiNN, optimize all controllers, validate."""
    console.print("[bold cyan]🚀 Executing Full Scientific Protocol[/bold cyan]")
    
    # Step 1: Train PiNN
    console.print("\n[bold]Step 1: Training Digital Twin (PiNN)[/bold]")
    model, workload_loader = ensure_pinn_model()
    
    # Step 2: Global Optimization Loop
    console.print("\n[bold]Step 2: Global Optimization Loop[/bold]")
    controllers = list_controllers()
    sim_config = get_default_simulation_config()
    best_params_all: Dict[str, Dict[str, Any]] = {}
    
    for controller_name in controllers:
        console.print(f"\n[cyan]Optimizing {controller_name}...[/cyan]")
        search_space = get_search_space(controller_name)
        
        optimizer = BayesianOptimizer(
            controller_name=controller_name,
            param_bounds=search_space,
            simulation_config=sim_config,
            model=model,
            workload_loader=workload_loader,
            scenario_name="real_trace_replay",  # Always optimize on real trace
            n_trials=50,
            output_dir=TUNING_DIR,
        )
        
        best_params = optimizer.optimize()
        best_params_all[controller_name] = best_params
    
    # Step 3: Validation on Golden Quartet
    console.print("\n[bold]Step 3: Validation on Golden Quartet[/bold]")
    scenarios = get_golden_quartet_scenarios()
    all_results: List[Dict[str, Any]] = []
    
    executor = SimulationExecutor(sim_config)
    for scenario in scenarios:
        scenario_name = scenario["name"]
        profile_name = scenario["profile"]
        
        console.print(f"\n[cyan]Scenario: {scenario_name}[/cyan]")
        
        # Apply scenario overrides
        scenario_sim_config = sim_config
        if scenario.get("sim_overrides"):
            from dataclasses import asdict
            sim_dict = asdict(sim_config)
            sim_dict.update(scenario["sim_overrides"])
            scenario_sim_config = SimulationConfig(**sim_dict)
            executor = SimulationExecutor(scenario_sim_config)
        
        horizon = int(round(scenario_sim_config.duration / scenario_sim_config.Ts))
        workload = workload_loader.get_workload(
            profile_name,
            scenario_sim_config.Ts,
            horizon,
        )
        
        for controller_name in controllers:
            params = best_params_all.get(controller_name, get_default_controller_params()[controller_name])
            controller = create_controller(controller_name, params, Ts=scenario_sim_config.Ts)
            
            result = executor.run(
                controller=controller,
                model=model,
                workload=workload,
                algo_name=controller_name,
                T_ref=scenario_sim_config.T_ref,
            )
            
            all_results.append({
                "controller": controller_name,
                "scenario": scenario_name,
                **result.metrics,
            })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = ARTIFACTS_DIR / "validation_results.csv"
    results_df.to_csv(results_path, index=False)
    console.print(f"\n[green]Validation results saved to {results_path}[/green]")
    
    # Step 4: Generate Pareto plot
    console.print("\n[bold]Step 4: Generating Pareto Plot[/bold]")
    plot_multi_controller_pareto()
    
    console.print("\n[bold green]✅ Full Scientific Protocol Completed![/bold green]")


@app.command()
def train_digital_twin() -> None:
    """Train Digital Twin: Train PiNN on TPU_Google_Edge.pkl."""
    console.print("[bold cyan]🧠 Training Digital Twin[/bold cyan]")
    ensure_pinn_model()
    console.print("[green]✅ Training completed![/green]")


@app.command()
def optimize() -> None:
    """Hyperparameter Optimization: Optimize controller parameters."""
    console.print("[bold cyan]🎯 Hyperparameter Optimization[/bold cyan]")
    
    model, workload_loader = ensure_pinn_model()
    controllers = list_controllers()
    sim_config = get_default_simulation_config()
    
    # Selection logic
    choice = questionary.select(
        "Optimization scope:",
        choices=[
            "Optimize All Controllers (Batch run for overnight execution)",
            "Optimize Specific Controller",
        ],
    ).ask()
    
    if choice == "Optimize All Controllers (Batch run for overnight execution)":
        controller_list = controllers
    else:
        controller_list = questionary.checkbox(
            "Select controllers to optimize:",
            choices=controllers,
        ).ask()
    
    for controller_name in controller_list:
        console.print(f"\n[cyan]Optimizing {controller_name}...[/cyan]")
        search_space = get_search_space(controller_name)
        
        optimizer = BayesianOptimizer(
            controller_name=controller_name,
            param_bounds=search_space,
            simulation_config=sim_config,
            model=model,
            workload_loader=workload_loader,
            scenario_name="real_trace_replay",
            n_trials=50,
            output_dir=TUNING_DIR,
        )
        
        best_params = optimizer.optimize()
        console.print(f"[green]Best parameters for {controller_name}: {best_params}[/green]")


@app.command()
def simulate() -> None:
    """Run Simulation Experiments: Run simulations with best or default parameters."""
    console.print("[bold cyan]🧪 Run Simulation Experiments[/bold cyan]")
    
    model, workload_loader = ensure_pinn_model()
    controllers = list_controllers()
    scenarios = get_golden_quartet_scenarios()
    sim_config = get_default_simulation_config()
    
    # Matrix selection
    scope = questionary.select(
        "Scope:",
        choices=["Single Controller", "All Controllers"],
    ).ask()
    
    if scope == "Single Controller":
        controller_list = [questionary.select("Select controller:", choices=controllers).ask()]
    else:
        controller_list = controllers
    
    scenario_choice = questionary.select(
        "Scenario:",
        choices=["Single Scenario", "All Golden Quartet"],
    ).ask()
    
    if scenario_choice == "Single Scenario":
        scenario_list = [questionary.select("Select scenario:", choices=[s["name"] for s in scenarios]).ask()]
    else:
        scenario_list = [s["name"] for s in scenarios]
    
    # Load best params if available
    best_params_all: Dict[str, Dict[str, Any]] = {}
    for controller_name in controller_list:
        tuning_file = TUNING_DIR / f"tuning_history_{controller_name}.csv"
        if tuning_file.exists():
            df = pd.read_csv(tuning_file)
            if len(df) > 0:
                best_idx = df["loss"].idxmin()
                best_params = df.iloc[best_idx].to_dict()
                # Extract only parameter columns (exclude metrics)
                param_cols = [c for c in df.columns if c not in ["trial", "ITAE", "Energy", "ITAE_normalized", "Energy_normalized", "Overshoot", "loss"]]
                best_params_all[controller_name] = {k: best_params[k] for k in param_cols if k in best_params}
                console.print(f"[green]Loaded best params for {controller_name}[/green]")
            else:
                best_params_all[controller_name] = get_default_controller_params()[controller_name]
        else:
            best_params_all[controller_name] = get_default_controller_params()[controller_name]
    
    # Run simulations
    all_results: List[Dict[str, Any]] = []
    executor = SimulationExecutor(sim_config)
    
    for scenario_name in scenario_list:
        scenario = next(s for s in scenarios if s["name"] == scenario_name)
        profile_name = scenario["profile"]
        
        # Apply scenario overrides
        scenario_sim_config = sim_config
        if scenario.get("sim_overrides"):
            from dataclasses import asdict
            sim_dict = asdict(sim_config)
            sim_dict.update(scenario["sim_overrides"])
            scenario_sim_config = SimulationConfig(**sim_dict)
            executor = SimulationExecutor(scenario_sim_config)
        
        horizon = int(round(scenario_sim_config.duration / scenario_sim_config.Ts))
        workload = workload_loader.get_workload(
            profile_name,
            scenario_sim_config.Ts,
            horizon,
        )
        
        for controller_name in controller_list:
            params = best_params_all[controller_name]
            controller = create_controller(controller_name, params, Ts=scenario_sim_config.Ts)
            
            result = executor.run(
                controller=controller,
                model=model,
                workload=workload,
                algo_name=controller_name,
                T_ref=scenario_sim_config.T_ref,
            )
            
            all_results.append({
                "controller": controller_name,
                "scenario": scenario_name,
                **result.metrics,
            })
    
    # Display results table
    table = Table(title="Simulation Results")
    table.add_column("Controller")
    table.add_column("Scenario")
    table.add_column("ITAE", justify="right")
    table.add_column("Energy", justify="right")
    table.add_column("Overshoot", justify="right")
    
    for result in all_results:
        table.add_row(
            result["controller"],
            result["scenario"],
            f"{result['ITAE']:.4f}",
            f"{result['Energy']:.4f}",
            f"{result['Overshoot']:.4f}",
        )
    
    console.print(table)


@app.command()
def analyze() -> None:
    """Analysis & Plotting: Generate multi-controller Pareto plot."""
    console.print("[bold cyan]📊 Analysis & Plotting[/bold cyan]")
    plot_multi_controller_pareto()


def plot_multi_controller_pareto() -> None:
    """Plot multi-controller Pareto frontier showing TRiNC dominance."""
    import matplotlib.pyplot as plt
    from src.utils.plotting import create_pareto_3d
    
    controllers = list_controllers()
    colors = {
        "pid": "blue",
        "pid_deadzone": "cyan",
        "event_pid": "green",
        "lif_snn": "orange",
        "trinc": "red",
    }
    
    # 2D Pareto plot (matplotlib)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Collect metrics for 3D plot
    metrics_3d: Dict[str, Dict[str, float]] = {}
    
    for controller_name in controllers:
        tuning_file = TUNING_DIR / f"tuning_history_{controller_name}.csv"
        if not tuning_file.exists():
            console.print(f"[yellow]Warning: No tuning data for {controller_name}[/yellow]")
            continue
        
        df = pd.read_csv(tuning_file)
        if len(df) == 0:
            continue
        
        # Extract Pareto frontier
        pareto_mask = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j:
                    if (df.iloc[j]["ITAE"] <= df.iloc[i]["ITAE"] and
                        df.iloc[j]["Energy"] <= df.iloc[i]["Energy"] and
                        (df.iloc[j]["ITAE"] < df.iloc[i]["ITAE"] or
                         df.iloc[j]["Energy"] < df.iloc[i]["Energy"])):
                        pareto_mask[i] = False
                        break
        
        pareto_df = df[pareto_mask].sort_values("ITAE")
        
        # Get best point (lowest ITAE)
        if len(pareto_df) > 0:
            best_idx = pareto_df["ITAE"].idxmin()
            best_row = pareto_df.loc[best_idx]
            metrics_3d[controller_name] = {
                "ITAE": float(best_row["ITAE"]),
                "Energy": float(best_row["Energy"]),
                "inference_latency_ms": float(best_row.get("inference_latency_ms", 0.0)),
            }
        
        color = colors.get(controller_name, "gray")
        ax.scatter(
            pareto_df["ITAE"],
            pareto_df["Energy"],
            label=controller_name.upper(),
            color=color,
            alpha=0.7,
            s=50,
        )
        ax.plot(
            pareto_df["ITAE"],
            pareto_df["Energy"],
            color=color,
            alpha=0.5,
            linestyle="--",
        )
    
    ax.set_xlabel("ITAE (Integral Time-weighted Absolute Error)", fontsize=12)
    ax.set_ylabel("Energy (Control Effort)", fontsize=12)
    ax.set_title("Battle of Controllers: Pareto Frontier Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = ARTIFACTS_DIR / "pareto_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    console.print(f"[green]2D Pareto plot saved to {plot_path}[/green]")
    plt.close()
    
    # 3D Pareto plot (Plotly)
    if metrics_3d:
        plot_3d_path = ARTIFACTS_DIR / "pareto_3d.html"
        create_pareto_3d(metrics_3d, plot_3d_path)
        console.print(f"[green]3D Pareto plot saved to {plot_3d_path}[/green]")


@app.command()
def ablation() -> None:
    """Ablation Study: Test TRiNC with different gate combinations."""
    console.print("[bold cyan]🔬 TRiNC Ablation Study[/bold cyan]")
    
    model, workload_loader = ensure_pinn_model()
    sim_config = get_default_simulation_config()
    executor = SimulationExecutor(sim_config)
    
    # Get default TRiNC parameters
    trinc_params = get_default_controller_params()["trinc"]
    
    # Define ablation configurations
    ablation_configs = [
        {"name": "Full TRiNC", "enable_p": True, "enable_h": True, "enable_s": True},
        {"name": "P+H only", "enable_p": True, "enable_h": True, "enable_s": False},
        {"name": "P+S only", "enable_p": True, "enable_h": False, "enable_s": True},
        {"name": "H+S only", "enable_p": False, "enable_h": True, "enable_s": True},
        {"name": "P only", "enable_p": True, "enable_h": False, "enable_s": False},
        {"name": "H only", "enable_p": False, "enable_h": True, "enable_s": False},
        {"name": "S only", "enable_p": False, "enable_h": False, "enable_s": True},
    ]
    
    # Select scenario
    scenarios = get_golden_quartet_scenarios()
    scenario_choice = questionary.select(
        "Select scenario:",
        choices=[s["name"] for s in scenarios],
    ).ask()
    
    scenario = next(s for s in scenarios if s["name"] == scenario_choice)
    profile_name = scenario["profile"]
    
    # Apply scenario overrides
    scenario_sim_config = sim_config
    if scenario.get("sim_overrides"):
        from dataclasses import asdict
        sim_dict = asdict(sim_config)
        sim_dict.update(scenario["sim_overrides"])
        scenario_sim_config = SimulationConfig(**sim_dict)
        executor = SimulationExecutor(scenario_sim_config)
    
    horizon = int(round(scenario_sim_config.duration / scenario_sim_config.Ts))
    workload = workload_loader.get_workload(
        profile_name,
        scenario_sim_config.Ts,
        horizon,
    )
    
    # Run ablation experiments
    all_results: List[Dict[str, Any]] = []
    
    for config in ablation_configs:
        console.print(f"\n[cyan]Testing: {config['name']}[/cyan]")
        
        # Create controller with ablation settings
        controller_params = {
            **trinc_params,
            "enable_p_gate": config["enable_p"],
            "enable_h_gate": config["enable_h"],
            "enable_s_gate": config["enable_s"],
        }
        controller = create_controller("trinc", controller_params, Ts=scenario_sim_config.Ts)
        
        result = executor.run(
            controller=controller,
            model=model,
            workload=workload,
            algo_name=f"trinc_{config['name'].lower().replace(' ', '_')}",
            T_ref=scenario_sim_config.T_ref,
        )
        
        all_results.append({
            "configuration": config["name"],
            "P_gate": config["enable_p"],
            "H_gate": config["enable_h"],
            "S_gate": config["enable_s"],
            **result.metrics,
        })
    
    # Display results
    table = Table(title=f"Ablation Study Results - {scenario_choice}")
    table.add_column("Configuration")
    table.add_column("Gates", justify="center")
    table.add_column("ITAE", justify="right")
    table.add_column("Energy", justify="right")
    table.add_column("Overshoot", justify="right")
    table.add_column("Latency (ms)", justify="right")
    
    for result in all_results:
        gates = ""
        if result["P_gate"]:
            gates += "P"
        if result["H_gate"]:
            gates += "H"
        if result["S_gate"]:
            gates += "S"
        
        table.add_row(
            result["configuration"],
            gates,
            f"{result['ITAE']:.4f}",
            f"{result['Energy']:.4f}",
            f"{result['Overshoot']:.4f}",
            f"{result.get('inference_latency_ms', 0.0):.4f}",
        )
    
    console.print(table)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = ARTIFACTS_DIR / f"ablation_{scenario_choice}.csv"
    results_df.to_csv(results_path, index=False)
    console.print(f"\n[green]Ablation results saved to {results_path}[/green]")


@app.command()
def clean() -> None:
    """Clean Artifacts: Remove generated files."""
    console.print("[bold cyan]🗑️ Clean Artifacts[/bold cyan]")
    
    if questionary.confirm("Are you sure you want to clean all artifacts?").ask():
        import shutil
        if ARTIFACTS_DIR.exists():
            shutil.rmtree(ARTIFACTS_DIR)
            console.print(f"[green]Cleaned {ARTIFACTS_DIR}[/green]")
        else:
            console.print("[yellow]No artifacts to clean[/yellow]")


if __name__ == "__main__":
    app()
