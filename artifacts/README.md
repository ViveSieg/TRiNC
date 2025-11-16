# Curated Artifacts

This directory illustrates the layout that automation scripts expect when they
persist experiment results.  The sample files capture one previously recorded
run and help downstream tooling (for example plotting utilities) initialise the
correct folder structure.

- `metrics/sample_run/` stores aggregate CSV summaries from the controller
  sweep.
- `time_series/sample_run/` provides a single closed-loop trace for quick
  inspection.

Fresh experiment runs can be redirected to another location via the
`--artifacts-root` flag if you want to keep the curated snapshot untouched.
