"""Input/output helpers for simulation results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd


def save_time_series_csv(
    filepath: Path,
    time: Iterable[float],
    temperature: Iterable[float],
    T_ref: float,
    control: Iterable[float],
    error: Iterable[float],
    extra_columns: Dict[str, Iterable[float]] | None = None,
) -> None:
    """Save time-series data to CSV."""

    time_list = list(time)
    data = {
        "time": time_list,
        "temperature": list(temperature),
        "T_ref": [T_ref] * len(time_list),
        "control": list(control),
        "error": list(error),
    }
    if extra_columns:
        for key, values in extra_columns.items():
            data[key] = list(values)

    df = pd.DataFrame(data)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_metrics_csv(filepath: Path, metrics_list: List[Dict[str, float]]) -> None:
    """Save aggregated metrics for all controllers."""

    df = pd.DataFrame(metrics_list)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_metric_subset_csv(
    filepath: Path, metrics_list: Sequence[Mapping[str, float]], columns: Sequence[str]
) -> None:
    """Persist selected metric columns for quick comparisons."""

    if not metrics_list:
        return
    df = pd.DataFrame(metrics_list)
    subset = df.loc[:, [col for col in columns if col in df.columns]]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(filepath, index=False)


def save_collated_time_series_csv(
    filepath: Path,
    time: Iterable[float],
    series_by_algo: Mapping[str, Iterable[float]],
    reference: Iterable[float] | None = None,
    reference_name: str = "reference",
) -> None:
    """Export multi-controller time series in a single wide CSV."""

    time_list = list(time)
    data = {"time": time_list}
    for name, values in series_by_algo.items():
        data[name] = list(values)
    if reference is not None:
        data[reference_name] = list(reference)

    df = pd.DataFrame(data)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
