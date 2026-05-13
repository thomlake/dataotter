from typing import Any, Awaitable, Callable

from dataotter._runner import (
    DEFAULT_CONCURRENCY,
    DEFAULT_ERRORS,
    DEFAULT_MAX_FAILURES,
    ErrorsMode,
    run_map,
)
from dataotter.stores import JsonlStore, Store
from dataotter.types import MapResult, RowEvent


async def map(
    *,
    data: list[dict[str, Any]],
    row_id: str,
    name: str,
    fn: Callable[..., Awaitable[dict[str, Any]]],
    config: dict[str, Any] | None = None,
    store: Store | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    errors: ErrorsMode = DEFAULT_ERRORS,
    max_failures: int | None = DEFAULT_MAX_FAILURES,
    on_row_complete: (
        Callable[[RowEvent], None | Awaitable[None]] | None
    ) = None,
) -> MapResult:
    store = store or JsonlStore()
    return await run_map(
        store=store,
        data=data,
        row_id=row_id,
        name=name,
        fn=fn,
        config=config,
        concurrency=concurrency,
        errors=errors,
        max_failures=max_failures,
        on_row_complete=on_row_complete,
    )
