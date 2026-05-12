from typing import Any, Awaitable, Callable

import pandas as pd

from dataotter._bindings import BindingsInput
from dataotter._runner import (
    DEFAULT_CONCURRENCY,
    DEFAULT_ERRORS,
    DEFAULT_MAX_FAILURES,
    DEFAULT_VERSION,
    ErrorsMode,
)
from dataotter.engine import Engine
from dataotter.types import MapResult, RowEvent


async def map(
    *,
    data: pd.DataFrame,
    row_id: str,
    step_name: str,
    fn: Callable[..., Awaitable[dict[str, Any]]],
    inputs: BindingsInput,
    outputs: BindingsInput,
    config: dict[str, Any] | None = None,
    engine: Engine | None = None,
    version: str = DEFAULT_VERSION,
    concurrency: int = DEFAULT_CONCURRENCY,
    errors: ErrorsMode = DEFAULT_ERRORS,
    max_failures: int | None = DEFAULT_MAX_FAILURES,
    on_row_complete: Callable[[RowEvent], None | Awaitable[None]] | None = None,
) -> MapResult:
    engine = engine or Engine()
    return await engine.map(
        data=data,
        row_id=row_id,
        step_name=step_name,
        fn=fn,
        inputs=inputs,
        outputs=outputs,
        config=config,
        version=version,
        concurrency=concurrency,
        errors=errors,
        max_failures=max_failures,
        on_row_complete=on_row_complete,
    )
