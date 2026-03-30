"""Project-specific exception hierarchy for runtime and workflow failures."""


class VTrackError(RuntimeError):
    """Base class for vtrack domain errors."""


class SourceValidationError(VTrackError):
    """Raised when an input source is invalid for the requested command."""


class ModelLoadError(VTrackError):
    """Raised when a model cannot be loaded or initialized."""


class PipelineRuntimeError(VTrackError):
    """Raised when pipeline execution fails during processing."""


class ArtifactPublishError(VTrackError):
    """Raised when artifact publishing/copying fails."""


class RemoteExecutionError(VTrackError):
    """Raised when remote training command execution fails."""
