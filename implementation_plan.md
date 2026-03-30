# Implementation Plan

[Overview]
Stabilize and harden the local vtrack codebase by delivering production-grade runtime robustness, structured observability, maintainable CLI architecture, and comprehensive test/quality guardrails.

This implementation focuses exclusively on local-code improvements inside the current repository and intentionally excludes future platform expansion (ONNX/CoreML/RPi/web dashboard). The current codebase already has strong modular boundaries (`cli` → `workflows` → runtime modules), good tests, and clear docs, so the most valuable next step is reliability hardening and maintainability refactoring rather than feature expansion.

The plan is organized into phases that reduce operational risk first (runtime safety + diagnostics), then reduce maintenance cost (CLI modularization + shared validation), and finally increase confidence and governance (expanded test coverage + performance/quality gates). Each phase is designed to be incrementally shippable and backward-compatible with existing commands and wrappers in `scripts/*.py`.

[Types]
Type changes will introduce explicit shared argument/command schemas, structured runtime diagnostics models, and typed logging contexts to remove ambiguity and enforce consistent error/report behavior across commands.

1. New data structures (dataclasses/TypedDicts)
- `src/vtrack/runtime_types.py` (new)
  - `SourceSpec` (dataclass)
    - `raw: str | int | Path`
    - `kind: Literal["webcam", "file", "url", "stream"]`
    - `resolved_path: Path | None`
    - Validation:
      - local file sources must exist when `kind == "file"`
      - webcam index must be non-negative int
  - `RunLimits` (dataclass)
    - `max_frames: int | None`
    - `max_seconds: float | None`
    - Validation:
      - if set, both must be positive
  - `RuntimeContext` (dataclass)
    - `command: str`
    - `model_path: str`
    - `tracker: str | None`
    - `device: str | None`
    - `source_kind: str`
  - `RunStats` (dataclass)
    - `frames_processed: int`
    - `wall_time_sec: float`
    - `avg_fps: float`

- `src/vtrack/errors.py` (new)
  - `VTrackError` (base)
  - `SourceValidationError(VTrackError)`
  - `ModelLoadError(VTrackError)`
  - `PipelineRuntimeError(VTrackError)`
  - `ArtifactPublishError(VTrackError)`
  - `RemoteExecutionError(VTrackError)`

2. Logging payload types
- `src/vtrack/logging_utils.py` (new)
  - `LogContext` as `TypedDict(total=False)`
    - `command`, `source`, `source_kind`, `model`, `tracker`, `device`, `frame`, `elapsed_ms`, `event`, `error_type`

3. CLI parser typing helpers
- `src/vtrack/cli_args.py` (new)
  - typed parser attachment helpers returning `argparse.ArgumentParser`
  - shared argument group builders to guarantee consistent defaults/signatures across commands

[Files]
File updates will add focused support modules, split large CLI responsibilities into maintainable submodules, and modify existing runtime/workflow files to consume shared validation + logging + error infrastructure.

New files to create:
1. `src/vtrack/errors.py`
- Centralized domain exceptions for clear user-facing failures and testability.

2. `src/vtrack/logging_utils.py`
- Logger factory/configuration (human + optional JSON formatter modes).
- Context-aware logging helpers.

3. `src/vtrack/runtime_validation.py`
- Source parsing/validation helpers.
- Runtime guardrails (`max_frames`, optional `max_seconds`) validators.

4. `src/vtrack/runtime_types.py`
- Shared dataclasses and typed runtime payloads.

5. `src/vtrack/cli_args.py`
- Shared CLI argument registration functions (common inference args, source args, run-limit args).

6. `src/vtrack/cli_handlers.py`
- Command handler functions migrated from `cli.py`.

7. `src/vtrack/cli_parser.py`
- Parser construction migrated from `cli.py`.

8. `tests/test_runtime_validation.py`
- Source validation and run-limit validation tests.

9. `tests/test_logging_utils.py`
- Structured logging behavior tests (context fields, format mode selection).

10. `tests/test_cli_modular.py`
- Parser + handler wiring regression tests for split CLI modules.

11. `tests/test_error_paths.py`
- Negative-path tests across detect/track/pipeline/workflow boundaries.

12. `tests/test_performance_regression_contract.py`
- Baseline comparator tests for benchmark report contract/tolerance logic.

13. `tasks/benchmark_regression.py`
- Local script to compare benchmark output against baseline thresholds.

14. `docs/troubleshooting.md`
- Runtime failure modes and recovery guidance.

15. `docs/architecture-local-hardening.md`
- Local architecture notes and rationale for new guardrails/logging boundaries.

Existing files to modify:
1. `src/vtrack/cli.py`
- Reduce to thin facade (`build_parser`, `main`, wrappers), delegating parser/handlers to new modules.
- Preserve external API and script wrapper compatibility.

2. `src/vtrack/workflows.py`
- Integrate standardized runtime validation and structured error translation.
- Emit consistent contextual logs around run start/end and failure.

3. `src/vtrack/pipeline.py`
- Add run limits support (`max_frames`, optional `max_seconds`).
- Convert prints to logger events.
- Improve exception boundaries and deterministic cleanup messaging.

4. `src/vtrack/detect.py`
- Add model-load/source error handling wrappers.
- Enforce validated source contract for local files.

5. `src/vtrack/track.py`
- Harden model initialization and tracker config errors with typed exceptions.
- Add contextual logging around tracker initialization.

6. `src/vtrack/benchmarking.py`
- Standardize benchmark event logs and optional baseline-comparison hook outputs.

7. `src/vtrack/artifacts.py`
- Replace broad exception swallowing in selected sections with bounded, logged handling where feasible.
- Keep behavior backward-compatible.

8. `src/vtrack/remote.py`
- Wrap subprocess failures with `RemoteExecutionError` and include sanitized command context.

9. `README.md`
- Update command docs with new local guardrail flags (if introduced).
- Add link to troubleshooting and benchmark-regression workflow.

10. `pyproject.toml`
- Add pytest markers/options for new perf contract tests if needed.
- Keep dependency footprint minimal (prefer stdlib logging/json).

Files potentially deleted/moved:
- No deletions planned.
- Logical migration of parser/handler code from `cli.py` into `cli_parser.py` and `cli_handlers.py`.

Configuration updates:
- Optional env var for log format, e.g. `VTRACK_LOG_FORMAT=json|text` (no dependency on external logging libs).
- Optional env var for log level, e.g. `VTRACK_LOG_LEVEL=INFO|DEBUG|...`.

[Functions]
Function changes will centralize validation and error handling, standardize parser construction, and enforce consistent runtime controls.

New functions:
1. `src/vtrack/runtime_validation.py`
- `parse_source_spec(value: str) -> SourceSpec`
- `validate_source_for_command(source: SourceSpec, *, allow_stream: bool = True) -> SourceSpec`
- `validate_run_limits(max_frames: int | None, max_seconds: float | None) -> RunLimits`

2. `src/vtrack/logging_utils.py`
- `get_logger(name: str = "vtrack") -> logging.Logger`
- `build_log_context(**kwargs) -> LogContext`
- `log_event(logger, level, message, context: LogContext | None = None) -> None`

3. `src/vtrack/cli_args.py`
- `add_inference_args(parser: argparse.ArgumentParser, *, tracking: bool) -> None`
- `add_source_arg(parser: argparse.ArgumentParser, *, allow_url: bool = True) -> None`
- `add_run_limit_args(parser: argparse.ArgumentParser) -> None`

4. `src/vtrack/cli_parser.py`
- `build_parser() -> argparse.ArgumentParser` (migrated target)

5. `src/vtrack/cli_handlers.py`
- `cmd_demo(args) -> int`
- `cmd_detect_image(args) -> int`
- `cmd_detect_video(args) -> int`
- `cmd_benchmark_track(args) -> int`
- `cmd_train(args) -> int`
- `cmd_evaluate(args) -> int`
- `cmd_train_remote(args) -> int`

Modified functions:
1. `src/vtrack/cli.py`
- `main(...)`: route through new parser/handlers, unify exception mapping.
- wrapper mains remain unchanged semantically.

2. `src/vtrack/pipeline.py`
- `VehiclePipeline.run(...)`: add optional `max_frames`, `max_seconds`; structured lifecycle logging.

3. `src/vtrack/workflows.py`
- `run_demo`, `run_detect_image`, `run_detect_video`, `run_tracking_benchmark`: validate source/limits first and log start/end.

4. `src/vtrack/benchmarking.py`
- `_benchmark_run(...)`: richer event logging + stable report metadata.

5. `src/vtrack/remote.py`
- `run_remote_training_commands(...)`: typed error wrapping around subprocess failures.

Removed functions:
- None removed publicly.
- Internal helper movement from `cli.py` to `cli_args.py`/`cli_handlers.py`/`cli_parser.py` with re-export compatibility where required by tests.

[Classes]
Class changes will preserve domain APIs while adding reliability hooks and clearer operational boundaries.

New classes:
1. `src/vtrack/errors.py`
- `VTrackError` hierarchy (listed above).

2. `src/vtrack/runtime_types.py`
- `SourceSpec`, `RunLimits`, `RuntimeContext`, `RunStats`.

Modified classes:
1. `src/vtrack/pipeline.py::VehiclePipeline`
- Add optional run-limit controls and structured lifecycle diagnostics.
- Ensure deterministic close/cleanup and explicit summary object generation.

2. `src/vtrack/track.py::VehicleTracker`
- Harden initialization path with better exception mapping and context logging.

3. `src/vtrack/detect.py::VehicleDetector`
- Similar hardening for model/source validation boundary.

4. `src/vtrack/settings.py::InferenceConfig` (optional minimal extension)
- If needed, include optional run-limit defaults while preserving backward compatibility.

Removed classes:
- None.

[Dependencies]
Dependency strategy keeps the stack stable and avoids new heavy packages; improvements rely primarily on stdlib modules and existing dependencies.

- No required new third-party runtime dependencies planned.
- Continue using stdlib `logging`, `json`, `dataclasses`, `typing`.
- Optional: if JSON logs need stronger formatting, consider `python-json-logger` as a deferred choice (not required in this plan).
- Existing dependency versions in `pyproject.toml` remain unchanged unless a specific compatibility fix is later required.

[Testing]
Testing will prioritize regressions in behavior, reliability under failure paths, and maintainability guarantees for the CLI refactor while preserving existing successful coverage.

Required test updates:
1. Keep existing suite green (`uv run pytest`).
2. Add targeted tests:
- source validation (file exists, invalid webcam index, malformed URL/source strings where applicable)
- run-limit validation and enforcement behavior
- typed exception mapping from core modules
- structured logging payload presence (without brittle full-line assertions)
- CLI modularization parity tests (same options/defaults/help behavior)
- remote command failure translation tests
- benchmark contract test with tolerance checks against baseline schema

3. Add/maintain smoke tests as opt-in:
- no change to marker strategy, but ensure compatibility with new CLI wiring.

4. Quality checks:
- `uv run ruff check src scripts tests`
- `uv run pytest`
- optional benchmark contract script invocation for local regression evidence.

[Implementation Order]
Implementation proceeds in low-risk, dependency-aware phases so each phase can be validated independently and merged safely.

1. Phase 1 — Foundations (types/errors/logging/validation)
   1.1 Add `errors.py`, `runtime_types.py`, `logging_utils.py`, `runtime_validation.py`.
   1.2 Add unit tests for these modules.
   1.3 Integrate minimally into workflows for initial coverage.

2. Phase 2 — Runtime robustness hardening
   2.1 Integrate source/run-limit validation in workflows and pipeline.
   2.2 Add typed exception mapping in detect/track/remote/artifacts touchpoints.
   2.3 Replace critical prints with logger events while preserving user-facing summaries.

3. Phase 3 — CLI modularization
   3.1 Create `cli_args.py`, `cli_parser.py`, `cli_handlers.py`.
   3.2 Reduce `cli.py` to facade and ensure wrapper compatibility.
   3.3 Add parser/help parity tests and handler wiring tests.

4. Phase 4 — Quality governance and regression controls
   4.1 Add benchmark regression helper script and contract tests.
   4.2 Add docs: troubleshooting + local hardening architecture notes.
   4.3 Update README references to new local robustness workflows.

5. Phase 5 — Full validation and release readiness
   5.1 Run lint + full tests + selected CLI runtime checks.
   5.2 Fix discovered regressions.
   5.3 Final verification of backward compatibility for `scripts/*.py` wrappers.
