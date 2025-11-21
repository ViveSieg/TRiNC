"""Plotting utilities for thermal control benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


COLORS = {
    "pid": "tab:blue",
    "pid_deadzone": "tab:orange",
    "event_pid": "tab:green",
    "lif_snn": "tab:red",
    "trinc": "tab:purple",
}


def plot_temperature_responses(results: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
    """Plot temperature trajectories for all algorithms."""

    plt.figure(figsize=(8, 4))
    for name, data in results.items():
        plt.plot(data["time"], data["temperature"], label=name, color=COLORS.get(name, None))
    ref = next(iter(results.values()))
    plt.plot(ref["time"], ref["T_ref"], "k--", label="Reference")
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized Temperature")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_control_signals(results: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
    """Plot control actions for all algorithms."""

    plt.figure(figsize=(8, 4))
    for name, data in results.items():
        plt.plot(data["time"], data["control"], label=name, color=COLORS.get(name, None))
    plt.xlabel("Time [s]")
    plt.ylabel("Cooling Effort (PWM)")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_energy_bar(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Plot energy proxy comparison."""

    labels = list(metrics.keys())
    E1 = [metrics[name]["E1"] for name in labels]
    E2 = [metrics[name]["E2"] for name in labels]

    x = range(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar([i - 0.15 for i in x], E1, width=0.3, label="∑|u|Ts")
    plt.bar([i + 0.15 for i in x], E2, width=0.3, label="∑u²Ts")
    plt.xticks(list(x), labels, rotation=30)
    plt.ylabel("Energy Proxy")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_event_counts(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Plot event counts for event-driven controllers."""

    labels = list(metrics.keys())
    events = [metrics[name]["N_events"] for name in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, events, color=[COLORS.get(name, "gray") for name in labels])
    plt.ylabel("Event Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_interactive_report(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: str = "Thermal Control Simulation Report",
) -> None:
    """Create interactive HTML report with Plotly.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Dictionary mapping controller names to their simulation results.
        Each result should contain: time, temperature, control, error, metrics
    output_path : Path
        Path to save HTML report
    title : str
        Report title
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive reports. "
            "Install with: pip install plotly>=5.0.0"
        )
    
    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Temperature Response", "Control Signal", "Error"),
        vertical_spacing=0.1,
        shared_xaxes=True,
    )
    
    # Plot temperature responses
    for name, data in results.items():
        time = data["time"]
        temperature = data["temperature"]
        color = COLORS.get(name, "gray")
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=temperature,
                name=f"{name} (Temp)",
                line=dict(color=color),
                legendgroup=name,
            ),
            row=1,
            col=1,
        )
    
    # Add reference line
    if results:
        first_result = next(iter(results.values()))
        time = first_result["time"]
        T_ref = first_result.get("T_ref", [0.62] * len(time))
        if isinstance(T_ref, (int, float)):
            T_ref = [T_ref] * len(time)
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=T_ref,
                name="Reference",
                line=dict(color="black", dash="dash"),
                legendgroup="ref",
            ),
            row=1,
            col=1,
        )
    
    # Plot control signals
    for name, data in results.items():
        time = data["time"]
        control = data["control"]
        color = COLORS.get(name, "gray")
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=control,
                name=f"{name} (Control)",
                line=dict(color=color),
                legendgroup=name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    
    # Plot errors
    for name, data in results.items():
        time = data["time"]
        error = data["error"]
        color = COLORS.get(name, "gray")
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=error,
                name=f"{name} (Error)",
                line=dict(color=color),
                legendgroup=name,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=900,
        hovermode="x unified",
    )
    
    fig.update_xaxes(title_text="Time [s]", row=3, col=1)
    fig.update_yaxes(title_text="Normalized Temperature", row=1, col=1)
    fig.update_yaxes(title_text="Cooling Effort", row=2, col=1)
    fig.update_yaxes(title_text="Error", row=3, col=1)
    
    # Save HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    
    print(f"Interactive report saved to {output_path}")


def create_pareto_3d(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
    x_metric: str = "ITAE",
    y_metric: str = "Energy",
    z_metric: str = "inference_latency_ms",
) -> None:
    """Create 3D Pareto plot with ITAE, Energy, and Latency.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Dictionary mapping controller names to their metrics
    output_path : Path
        Path to save HTML plot
    x_metric : str
        Metric for x-axis (default: ITAE)
    y_metric : str
        Metric for y-axis (default: Energy)
    z_metric : str
        Metric for z-axis (default: inference_latency_ms)
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for 3D plots. "
            "Install with: pip install plotly>=5.0.0"
        )
    
    fig = go.Figure()
    
    for name, metrics in results.items():
        x = metrics.get(x_metric, 0.0)
        y = metrics.get(y_metric, 0.0)
        z = metrics.get(z_metric, 0.0)
        color = COLORS.get(name, "gray")
        
        fig.add_trace(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode="markers",
                name=name,
                marker=dict(
                    size=10,
                    color=color,
                    opacity=0.8,
                ),
                text=[f"{name}<br>ITAE: {x:.4f}<br>Energy: {y:.4f}<br>Latency: {z:.4f} ms"],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    
    fig.update_layout(
        title="3D Pareto Frontier: ITAE vs Energy vs Latency",
        scene=dict(
            xaxis_title=x_metric,
            yaxis_title=y_metric,
            zaxis_title=z_metric,
        ),
        height=800,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    
    print(f"3D Pareto plot saved to {output_path}")
