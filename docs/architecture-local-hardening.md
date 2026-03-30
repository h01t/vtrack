# Local Architecture Hardening Notes

This document summarizes the local reliability/maintainability hardening layers introduced across the vtrack codebase.

## Goals

1. **Operational robustness**
   - clear exception boundaries
   - deterministic failure semantics
2. **Observability**
   - structured contextual logging across runtime modules
3. **Maintainability**
   - modular CLI parser/handler architecture
4. **Compatibility**
   - preserve existing scripts and command UX

---

## Layered structure

```text
scripts/*.py wrappers
        │
        ▼
    vtrack.cli (facade)
        │
        ├── vtrack.cli_parser     (argparse command graph)
        ├── vtrack.cli_handlers   (command execution handlers)
        └── vtrack.cli_args       (typed mapping/parsing helpers)
        │
        ▼
   vtrack.workflows (high-level orchestration)
        │
        ├── detect / track / pipeline
        ├── benchmarking
        ├── artifacts
        └── remote
```

---

## New foundation modules

## `src/vtrack/errors.py`
Typed domain exceptions:
- `VTrackError`
- `SourceValidationError`
- `ModelLoadError`
- `PipelineRuntimeError`
- `ArtifactPublishError`
- `RemoteExecutionError`

Why:
- avoids leaking low-level exceptions across boundaries
- improves operator-facing diagnostics and testability

## `src/vtrack/runtime_types.py`
Shared runtime dataclasses:
- `SourceSpec`
- `RunLimits`
- `RuntimeContext`
- `RunStats`

Why:
- explicit data contracts across module boundaries

## `src/vtrack/runtime_validation.py`
Source and limits validation helpers:
- source parsing/validation
- run-limit validation guardrails

Why:
- fail fast on invalid runtime inputs
- reduce undefined runtime behavior

## `src/vtrack/logging_utils.py`
Structured logging helper layer:
- logger config by env (`VTRACK_LOG_LEVEL`, `VTRACK_LOG_FORMAT`)
- contextual logging events for runtime lifecycle/failure paths

Why:
- predictable logs for local debugging and automation integration

---

## CLI modularization (Step 3)

Previously, `cli.py` was a monolith with parser wiring + arg mapping + handlers.

Now:
- `cli_parser.py` owns command graph and arguments
- `cli_handlers.py` owns command execution handlers
- `cli_args.py` owns typed argument-to-config adapters and geometry/source parsing
- `cli.py` is a backward-compatible facade for entrypoints and compatibility imports

Benefits:
- lower cognitive load
- clearer ownership boundaries
- easier incremental testing/refactoring
- preserved wrapper compatibility (`scripts/*.py`)

---

## Runtime error strategy

## Detect/track/pipeline hardening
- wrap model init failures in `ModelLoadError`
- wrap inference/runtime loop failures in `PipelineRuntimeError`
- include contextual logging payload with source/model/tracker/device metadata

## Remote/artifacts boundaries (planned/ongoing)
- map subprocess and publish failures to typed domain errors
- avoid broad exception swallowing

---

## Testing strategy

Validation approach used:
1. Ruff lint on changed modules
2. Targeted regression tests for changed runtime/CLI paths
3. Full `pytest -q` suite checks
4. CLI help/argument behavior checks across all subcommands and wrappers

This strategy ensures:
- refactor safety
- behavior parity
- compatibility preservation

---

## Benchmark governance

`tasks/benchmark_regression.py` provides lightweight local regression checks:
- compare current benchmark CSV against baseline
- enforce configurable thresholds for:
  - throughput FPS drop
  - p95 latency increase
  - median latency increase

This enables a CI-friendly quality gate without introducing heavy dependencies.

---

## Compatibility guarantees

The hardening/modularization path is intentionally non-breaking for:
- top-level command usage (`vtrack ...`)
- wrapper scripts under `scripts/`
- existing tests relying on selected facade symbols

Where needed, facade aliases are preserved to avoid churn during migration.

---

## Future local improvements

1. Expand explicit error mapping in workflows/remote/artifacts
2. Add dedicated tests for structured logging payload contracts
3. Add benchmark regression checks into CI pipeline
4. Add CLI contract snapshot tests for argument/help stability
