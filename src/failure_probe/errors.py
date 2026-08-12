"""Package-specific exceptions."""


class FailureProbeError(Exception):
    """Base error shown to CLI users without a traceback."""


class DatasetFormatError(FailureProbeError):
    """Raised when a dataset or annotation file is invalid."""


class UnsafePathError(FailureProbeError):
    """Raised when a path escapes an allowed root."""


class RunFormatError(FailureProbeError):
    """Raised when a directory is not a valid Failure Probe run."""
