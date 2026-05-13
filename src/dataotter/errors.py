from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataotter.types import MapResult


class DataOtterError(Exception):
    """Base class for dataotter errors."""


class CacheMismatchError(DataOtterError):
    def __init__(self, mismatches: list[dict[str, object]]) -> None:
        super().__init__("Cached row inputs do not match the current data")
        self.mismatches = mismatches


class ConfigMismatchError(DataOtterError):
    def __init__(self, *, name: str) -> None:
        super().__init__(
            f"Cached map {name!r} was created with a different config"
        )
        self.name = name


class MapFailedError(DataOtterError):
    def __init__(self, result: MapResult) -> None:
        failed_rows = result.stats.failed_rows
        super().__init__(f"Map failed with {failed_rows} failed row(s)")
        self.result = result


class InvalidFunctionError(DataOtterError):
    """Raised when the provided function cannot be called by dataotter."""


class InvalidRowIdError(DataOtterError):
    """Raised when the row ID column is missing or invalid."""
