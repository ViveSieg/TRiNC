"""Plotting utilities for thermal control benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


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
