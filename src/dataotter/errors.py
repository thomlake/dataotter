from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from dataotter.types import MapResult


class DataOtterError(Exception):
    """Base class for dataotter errors."""


class CacheMismatchError(DataOtterError):
    def __init__(self, mismatches: pd.DataFrame) -> None:
        super().__init__("Cached row inputs do not match the current data")
        self.mismatches = mismatches


class MapFailedError(DataOtterError):
    def __init__(self, result: MapResult) -> None:
        failed_rows = result.stats.failed_rows
        super().__init__(f"Map failed with {failed_rows} failed row(s)")
        self.result = result


class InvalidFunctionError(DataOtterError):
    """Raised when the provided function cannot be called by dataotter."""


class InvalidBindingError(DataOtterError):
    """Raised when input or output bindings are invalid."""


class InvalidRowIdError(DataOtterError):
    """Raised when the row ID column is missing or invalid."""
