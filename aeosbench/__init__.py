"""AEOSBench reimplementation package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aeosbench-repro")
except PackageNotFoundError:  # pragma: no cover - fallback for editable source tree use
    __version__ = "0.1.0"

