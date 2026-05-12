from dataclasses import dataclass
from typing import Literal

import pandas as pd

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_REUSED = "reused"

RowStatus = Literal["success", "error", "reused"]


@dataclass(frozen=True)
class MapStats:
    total_rows: int
    reused_rows: int
    eligible_rows: int
    attempted_rows: int
    succeeded_rows: int
    failed_rows: int
    not_started_rows: int
    stopped_early: bool
    started_at: str
    finished_at: str
    duration_seconds: float


@dataclass(frozen=True)
class RowEvent:
    row_id: str
    status: RowStatus
    attempted: bool
    reused: bool
    error_type: str | None
    duration_seconds: float
    completed_rows: int
    attempted_rows: int
    succeeded_rows: int
    failed_rows: int


@dataclass(frozen=True)
class MapResult:
    output: pd.DataFrame
    errors: pd.DataFrame
    stats: MapStats
    run_id: str
    map_id: str
