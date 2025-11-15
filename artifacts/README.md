# Curated Artifacts

This folder stores a lightweight snapshot of generated experiment outputs so the
project tree reflects the default layout expected by the CLI utilities.

- `metrics/` contains run-level CSV summaries that were produced by executing
  the controller suite.
- `time_series/baseline_burst_trinc.csv` captures the closed-loop response of
  the TRINC controller on the `baseline_burst` scenario and serves as a sample
  trace for plotting utilities.

New experiment runs can be directed to another location via the
`--artifacts-root` flag to avoid polluting the repository with large result
sets.
