import asyncio
import inspect
import re
import time
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from dataotter._normalize import (
    canonical_json,
    normalize_value,
    now_iso,
    stable_hash,
)
from dataotter.errors import (
    CacheMismatchError,
    InvalidFunctionError,
    InvalidRowIdError,
    MapFailedError,
)
from dataotter.stores import (
    RECORD_TYPE_ROW_RESULT,
    RowState,
    Store,
    _record_to_row_state,
    normalize_config,
)
from dataotter.types import (
    STATUS_ERROR,
    STATUS_REUSED,
    STATUS_SUCCESS,
    MapResult,
    MapStats,
    RowEvent,
    RowStatus,
)

DEFAULT_CONCURRENCY = 10
DEFAULT_MAX_FAILURES = 10
ERRORS_RAISE_AFTER = "raise_after"
ERRORS_RETURN = "return"
DEFAULT_ERRORS = ERRORS_RAISE_AFTER
ErrorsMode = Literal["raise_after", "return"]
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class _RowContext:
    row_id: str
    row_key: str
    input_hash: str
    record: dict[str, Any]


async def run_map(
    *,
    store: Store,
    data: list[dict[str, Any]],
    row_id: str,
    name: str,
    fn: Callable[..., Awaitable[dict[str, Any]]],
    config: dict[str, Any] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    errors: ErrorsMode = DEFAULT_ERRORS,
    max_failures: int | None = DEFAULT_MAX_FAILURES,
    on_row_complete: (
        Callable[[RowEvent], None | Awaitable[None]] | None
    ) = None,
) -> MapResult:
    started_at = now_iso()
    started_perf = time.perf_counter()
    normalized_config = normalize_config(config)

    _validate_args(
        data=data,
        row_id=row_id,
        name=name,
        fn=fn,
        concurrency=concurrency,
        errors=errors,
        max_failures=max_failures,
    )

    run_id = _new_run_id()

    store_run, states = await store.begin_run(
        name=name,
        row_id_column=row_id,
        config=normalized_config,
    )
    try:
        rows = _prepare_rows(data=data, row_id=row_id)

        mismatches = _find_mismatches(name=name, rows=rows, states=states)
        if mismatches:
            raise CacheMismatchError(mismatches)

        eligible = [
            row
            for row in rows
            if (
                row.row_key not in states
                or states[row.row_key].status == STATUS_ERROR
            )
        ]
        reused = [
            row
            for row in rows
            if (
                row.row_key in states
                and states[row.row_key].status == STATUS_SUCCESS
            )
        ]
        reused_rows = len(reused)

        attempted_keys: set[str] = set()
        stop_event = asyncio.Event()
        next_lock = asyncio.Lock()
        progress_lock = asyncio.Lock()
        next_index = 0
        completed_count = 0
        attempted_count = 0
        succeeded_count = 0
        failed_count = 0

        async def emit_row_event(
            *,
            row_id_value: str,
            status: RowStatus,
            attempted: bool,
            reused_value: bool,
            error_type: str | None,
            duration_seconds: float,
        ) -> None:
            nonlocal attempted_count
            nonlocal completed_count
            nonlocal failed_count
            nonlocal succeeded_count
            if attempted:
                attempted_count += 1
            completed_count += 1
            if status == STATUS_SUCCESS:
                succeeded_count += 1
            elif status == STATUS_ERROR:
                failed_count += 1

            if on_row_complete is None:
                return

            event = RowEvent(
                row_id=row_id_value,
                status=status,
                attempted=attempted,
                reused=reused_value,
                error_type=error_type,
                duration_seconds=duration_seconds,
                completed_rows=completed_count,
                attempted_rows=attempted_count,
                succeeded_rows=succeeded_count,
                failed_rows=failed_count,
            )
            maybe_awaitable = on_row_complete(event)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable

        for row in reused:
            async with progress_lock:
                await emit_row_event(
                    row_id_value=row.row_id,
                    status=STATUS_REUSED,
                    attempted=False,
                    reused_value=True,
                    error_type=None,
                    duration_seconds=0.0,
                )

        async def worker() -> None:
            nonlocal next_index
            while True:
                async with next_lock:
                    if stop_event.is_set() or next_index >= len(eligible):
                        return
                    row = eligible[next_index]
                    next_index += 1

                attempted_keys.add(row.row_key)
                row_started = time.perf_counter()
                record = await _run_one(
                    row=row,
                    fn=fn,
                    run_id=run_id,
                    name=name,
                    row_id_column=row_id,
                )
                await store_run.append_row_result(record)
                states[row.row_key] = _record_to_row_state(record)
                duration_seconds = time.perf_counter() - row_started
                async with progress_lock:
                    error = record.get("error") or {}
                    await emit_row_event(
                        row_id_value=row.row_id,
                        status=record["status"],
                        attempted=True,
                        reused_value=False,
                        error_type=error.get("type"),
                        duration_seconds=duration_seconds,
                    )
                    should_stop = (
                        record["status"] == STATUS_ERROR
                        and _should_stop_for_failures(
                            failed_count=failed_count,
                            max_failures=max_failures,
                        )
                    )
                    if should_stop:
                        stop_event.set()

        worker_count = min(concurrency, len(eligible))
        if worker_count:
            worker_tasks = [
                asyncio.create_task(worker())
                for _ in range(worker_count)
            ]
            try:
                await asyncio.gather(*worker_tasks)
            except (asyncio.CancelledError, KeyboardInterrupt):
                stop_event.set()
                for task in worker_tasks:
                    task.cancel()
                await asyncio.gather(*worker_tasks, return_exceptions=True)
                raise

        result = _build_result(
            rows=rows,
            states=states,
            run_id=run_id,
            name=name,
            row_id_column=row_id,
            total_rows=len(rows),
            reused_rows=reused_rows,
            eligible_rows=len(eligible),
            attempted_rows=len(attempted_keys),
            stopped_early=stop_event.is_set(),
            started_at=started_at,
            started_perf=started_perf,
        )

        if result.stats.failed_rows and errors == ERRORS_RAISE_AFTER:
            raise MapFailedError(result)

        return result
    finally:
        await store_run.finish()


def _validate_args(
    *,
    data: list[dict[str, Any]],
    row_id: str,
    name: str,
    fn: Callable[..., Awaitable[dict[str, Any]]],
    concurrency: int,
    errors: str,
    max_failures: int | None,
) -> None:
    if not isinstance(data, list):
        raise TypeError("data must be a list of dict records")
    if not data:
        raise ValueError("data must contain at least one row")
    if not all(isinstance(record, Mapping) for record in data):
        raise TypeError("data must be a list of dict records")
    for index, record in enumerate(data):
        invalid_keys = [
            key for key in record
            if not isinstance(key, str) or not key
        ]
        if invalid_keys:
            raise TypeError(
                f"row {index} contains non-string or empty keys: "
                f"{invalid_keys!r}"
            )
    if not isinstance(row_id, str) or not row_id:
        raise InvalidRowIdError("row_id must be a non-empty string")
    _validate_name("name", name)
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    if errors not in {ERRORS_RAISE_AFTER, ERRORS_RETURN}:
        raise ValueError(
            f"errors must be {ERRORS_RAISE_AFTER!r} "
            f"or {ERRORS_RETURN!r}"
        )
    if max_failures is not None and max_failures < 1:
        raise ValueError("max_failures must be >= 1 or None")

    seen: set[str] = set()
    for index, record in enumerate(data):
        if row_id not in record:
            raise InvalidRowIdError(
                f"row {index} does not contain row_id key {row_id!r}"
            )
        normalized_row_id = normalize_value(record[row_id])
        if normalized_row_id is None:
            raise InvalidRowIdError(
                f"row_id key {row_id!r} contains null values"
            )
        if not isinstance(normalized_row_id, str):
            raise InvalidRowIdError(
                f"row_id key {row_id!r} values must be strings"
            )
        if normalized_row_id in seen:
            raise InvalidRowIdError(
                f"row_id key {row_id!r} contains duplicate values"
            )
        seen.add(normalized_row_id)

    key_names = sorted({str(key) for record in data for key in record})
    _validate_async_function(fn=fn, arg_names=key_names)


def _validate_name(label: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if NAME_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{label} may only contain letters, numbers, dots, "
            "underscores, and hyphens"
        )


def _validate_async_function(
    *,
    fn: Callable[..., Awaitable[dict[str, Any]]],
    arg_names: list[str],
) -> None:
    is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )
    if not is_async:
        raise InvalidFunctionError("fn must be an async callable")

    signature = inspect.signature(fn)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    for name in arg_names:
        parameter = parameters.get(name)
        if parameter is None:
            if not accepts_kwargs:
                raise InvalidFunctionError(
                    f"fn does not accept keyword argument {name!r}"
                )
            continue
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise InvalidFunctionError(
                f"fn argument {name!r} is positional-only"
            )

    provided = set(arg_names)
    for name, parameter in parameters.items():
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            continue
        required = parameter.default is inspect.Parameter.empty
        if required and name not in provided:
            raise InvalidFunctionError(
                f"fn requires argument {name!r} that is not bound"
            )


def _prepare_rows(
    *,
    data: list[dict[str, Any]],
    row_id: str,
) -> list[_RowContext]:
    rows: list[_RowContext] = []
    for row in data:
        record = normalize_value(dict(row))
        if not isinstance(record, dict):
            raise TypeError("data records must normalize to dicts")
        normalized_row_id = record[row_id]
        if not isinstance(normalized_row_id, str):
            raise InvalidRowIdError(
                f"row_id key {row_id!r} values must be strings"
            )
        rows.append(
            _RowContext(
                row_id=normalized_row_id,
                row_key=canonical_json(normalized_row_id),
                input_hash=stable_hash(record),
                record=record,
            )
        )
    return rows


def _find_mismatches(
    *,
    name: str,
    rows: list[_RowContext],
    states: dict[str, RowState],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        state = states.get(row.row_key)
        if state is None or state.input_hash == row.input_hash:
            continue
        mismatches.append(
            {
                "name": name,
                "row_id": row.row_id,
                "mismatch_type": "input_value_changed",
                "cached_input_hash": state.input_hash,
                "current_input_hash": row.input_hash,
            }
        )
    return mismatches


async def _run_one(
    *,
    row: _RowContext,
    fn: Callable[..., Awaitable[dict[str, Any]]],
    run_id: str,
    name: str,
    row_id_column: str,
) -> dict[str, Any]:
    created_at = now_iso()
    base = {
        "type": RECORD_TYPE_ROW_RESULT,
        "run_id": run_id,
        "name": name,
        "row_id": row.row_id,
        "input_hash": row.input_hash,
        "created_at": created_at,
    }
    try:
        output_values = _validate_and_materialize_outputs(
            result=await fn(**row.record),
            row_id_column=row_id_column,
        )
    except Exception as exc:
        return {
            **base,
            "status": STATUS_ERROR,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }

    return {
        **base,
        "status": STATUS_SUCCESS,
        "outputs": output_values,
    }


def _validate_and_materialize_outputs(
    *,
    result: Any,
    row_id_column: str,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError("fn must return a dict")
    if row_id_column in result:
        raise ValueError(
            f"fn result may not include row_id key {row_id_column!r}"
        )

    output_values = normalize_value(result)
    if not isinstance(output_values, dict):
        raise TypeError("fn result must normalize to a dict")
    return output_values


def _build_result(
    *,
    rows: list[_RowContext],
    states: dict[str, RowState],
    run_id: str,
    name: str,
    row_id_column: str,
    total_rows: int,
    reused_rows: int,
    eligible_rows: int,
    attempted_rows: int,
    stopped_early: bool,
    started_at: str,
    started_perf: float,
) -> MapResult:
    output_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for row in rows:
        state = states.get(row.row_key)
        if state is None:
            continue
        if state.status == STATUS_SUCCESS:
            output_rows.append(
                {row_id_column: row.row_id, **(state.outputs or {})}
            )
        elif state.status == STATUS_ERROR:
            error = state.error or {}
            error_rows.append(
                {
                    "row_id": row.row_id,
                    "error_type": error.get("type"),
                    "error_message": error.get("message"),
                    "traceback": error.get("traceback"),
                    "attempted_at": state.created_at,
                    "input_hash": state.input_hash,
                }
            )

    finished_at = now_iso()
    stats = MapStats(
        total_rows=total_rows,
        reused_rows=reused_rows,
        eligible_rows=eligible_rows,
        attempted_rows=attempted_rows,
        succeeded_rows=len(output_rows),
        failed_rows=len(error_rows),
        not_started_rows=eligible_rows - attempted_rows,
        stopped_early=stopped_early,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.perf_counter() - started_perf,
    )
    return MapResult(
        output=output_rows,
        errors=error_rows,
        stats=stats,
        run_id=run_id,
        name=name,
    )


def _new_run_id() -> str:
    return f"{now_iso()}-{uuid.uuid4().hex[:12]}"


def _should_stop_for_failures(
    *,
    failed_count: int,
    max_failures: int | None,
) -> bool:
    if max_failures is None:
        return False
    return failed_count >= max_failures
