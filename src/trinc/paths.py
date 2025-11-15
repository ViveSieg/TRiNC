"""Centralised project paths for artifact management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Repository root directory."""

DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
"""Default location for generated experiment outputs."""


@dataclass(frozen=True)
class ArtifactLayout:
    """Container for directories used to persist experiment artifacts."""

    root: Path
    time_series: Path
    metrics: Path
    figures: Path


def _normalise_root(override: Path | None) -> Path:
    if override is None:
        return DEFAULT_ARTIFACTS_ROOT

    root = Path(override).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    else:
        root = root.resolve()
    return root


def get_artifact_layout(override: Path | None = None) -> ArtifactLayout:
    """Return the directory layout for storing experiment artifacts."""

    root = _normalise_root(override)
    return ArtifactLayout(
        root=root,
        time_series=root / "time_series",
        metrics=root / "metrics",
        figures=root / "figures",
    )


def ensure_artifact_layout(layout: ArtifactLayout) -> None:
    """Create artifact directories if they do not already exist."""

    for path in _iter_artifact_paths(layout):
        path.mkdir(parents=True, exist_ok=True)


def _iter_artifact_paths(layout: ArtifactLayout) -> Iterable[Path]:
    yield layout.time_series
    yield layout.metrics
    yield layout.figures


__all__ = [
    "ArtifactLayout",
    "DEFAULT_ARTIFACTS_ROOT",
    "PROJECT_ROOT",
    "ensure_artifact_layout",
    "get_artifact_layout",
]
