# dataotter

`dataotter` runs resumable async row-wise workflows over pandas DataFrames.

It is useful when each row calls something slow or failure-prone, such as an
LLM API, classifier, parser, scraper, or enrichment service. Successful rows are
persisted as they complete, so reruns reuse prior successes and retry only
missing or failed rows.

`dataotter` is not a DAG orchestrator and is not a pandas replacement. Use
pandas for filtering, joining, sequencing, and merging intermediate results.

## Install

This project uses `uv`.

```bash
uv sync
```

## Quick Start

```python
import dataotter
import pandas as pd


df = pd.DataFrame(
    {
        "doc_id": ["1", "2", "3"],
        "body_text": ["a job post", "a recipe", "another job post"],
    }
)

engine = dataotter.Engine()


async def classify(text: str) -> dict[str, object]:
    # Call any async service here.
    return {
        "label": "job" if "job" in text else "other",
        "confidence": 0.9,
    }


result = await dataotter.map(
    data=df,
    row_id="doc_id",
    step_name="classify_document",
    fn=classify,
    inputs={"body_text": "text"},
    outputs={
        "label": "classification_label",
        "confidence": "classification_confidence",
    },
    config={"classifier_version": "v1"},
    engine=engine,
    concurrency=10,
)

df = df.merge(result.output, on="doc_id", how="left")
```

`result.output` contains only the row ID column and declared output columns.
Merge it back into your own DataFrame explicitly.

## API

```python
await dataotter.map(
    *,
    data: pd.DataFrame,
    row_id: str,
    step_name: str,
    fn: Callable[..., Awaitable[dict[str, Any]]],
    inputs: list[str] | dict[str, str],
    outputs: list[str] | dict[str, str],
    config: dict[str, Any] | None = None,
    engine: dataotter.Engine | None = None,
    version: str = "1",
    concurrency: int = 10,
    errors: Literal["raise_after", "return"] = "raise_after",
    max_failures: int | None = 10,
    on_row_complete: Callable[[dataotter.RowEvent], None | Awaitable[None]] | None = None,
) -> dataotter.MapResult
```

Important rules:

- `row_id` is required, must name an existing column, and all row ID values must be unique non-null strings.
- `fn` must be async and must return a `dict`.
- `inputs` maps DataFrame columns to function keyword arguments.
- `outputs` maps returned dict keys to output DataFrame columns.
- Returned keys must exactly match declared output keys.
- Output columns may not include the row ID column.

List shorthand is supported:

```python
inputs=["text"]
outputs=["label"]
```

is equivalent to:

```python
inputs={"text": "text"}
outputs={"label": "label"}
```

## Cache Identity

`dataotter` caches row outcomes under a derived `map_id`.

The `map_id` is based on:

- `step_name`
- `version`
- function input argument names
- output bindings
- normalized `config`

Row-level reuse is based on:

- same `map_id`
- same row ID value
- same normalized kwargs passed to `fn`

Source DataFrame column names are adapter details. Renaming `body_text` to
`raw_text` does not invalidate cache state if both invocations call
`fn(text=...)` with the same value for the same row ID.

`dataotter` does not hash Python function code. Use `version` or `config` to
record changes that affect correctness:

```python
result = await dataotter.map(
    ...,
    step_name="extract_fields",
    version="2",
    config={"prompt_hash": "sha256:..."},
)
```

If a row already has cached state for the same `map_id` and row ID, but the
current kwargs hash differs, `dataotter` raises `CacheMismatchError` before
starting new work.

## Failure Behavior

Rows may fail independently. Successful rows are persisted and reused on later
runs. Failed rows are retried on later runs.

By default, `map` stops starting new work after 10 row failures:

```python
result = await dataotter.map(..., max_failures=10)
```

Useful options:

- `max_failures=1`: stop after the first failed row.
- `max_failures=None`: keep running all eligible rows regardless of failures.
- `errors="raise_after"`: raise `MapFailedError` after in-flight rows finish.
- `errors="return"`: return `MapResult` even when rows failed.

`max_failures` must be `>= 1` or `None`; any other value raises `ValueError`.

When `errors="raise_after"`, the exception contains the partial result:

```python
try:
    result = await dataotter.map(...)
except dataotter.MapFailedError as exc:
    result = exc.result
    print(result.errors)
```

## Progress

Pass `on_row_complete` to receive one event per reused, successful, or failed row.

```python
def report(event: dataotter.RowEvent) -> None:
    print(event.completed_rows, event.row_id, event.status)


result = await dataotter.map(
    ...,
    on_row_complete=report,
)
```

The callback may be sync or async.

## Results

`MapResult` contains:

- `output`: DataFrame with `row_id + declared output columns`.
- `errors`: DataFrame with failed row details.
- `stats`: row counts and timing fields.
- `run_id`: unique ID for the current run.
- `map_id`: derived cache identity.

`MapStats` includes:

- `total_rows`
- `reused_rows`
- `eligible_rows`
- `attempted_rows`
- `succeeded_rows`
- `failed_rows`
- `not_started_rows`
- `stopped_early`
- `started_at`
- `finished_at`
- `duration_seconds`

## Cache Management

```python
engine = dataotter.Engine()

maps = engine.list_maps()
manifest = engine.get_map(result.map_id)

engine.delete_map(result.map_id)
engine.delete_maps(step_name="classify_document")
```

The default store is a local JSONL store rooted at `.dataotter/` in the current
working directory. It is safe within a single async process, but it is not
intended for cross-process coordination. Pass a custom store (or a `JsonlStore`
with a different path) via:

```python
engine = dataotter.Engine(store=dataotter.JsonlStore("/path/to/cache"))
```

## Development

```bash
uv run python -m pytest
uv run python -m compileall src tests
```
