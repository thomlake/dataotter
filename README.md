# dataotter

`dataotter` provides local persistence for async maps over records.

It is useful when each record calls something slow or failure-prone, such as an
LLM API, classifier, parser, scraper, or enrichment service. Successful rows are
persisted as they complete, so reruns reuse prior successes and retry only
missing or failed rows.

`dataotter` is not a DAG orchestrator or a DataFrame library. Prepare, filter,
join, and merge your data outside `dataotter`; pass records in as plain
`list[dict[str, Any]]`.

## Install

This project uses `uv`.

```bash
uv sync
```

## Quick Start

```python
import dataotter


records = [
    {"doc_id": "1", "body_text": "a job post"},
    {"doc_id": "2", "body_text": "a recipe"},
    {"doc_id": "3", "body_text": "another job post"},
]

store = dataotter.JsonlStore()


async def classify(doc_id: str, body_text: str) -> dict[str, object]:
    # Call any async service here.
    return {
        "label": "job" if "job" in body_text else "other",
        "confidence": 0.9,
    }


result = await dataotter.map(
    data=records,
    row_id="doc_id",
    name="classify_document-v1",
    fn=classify,
    config={"prompt_hash": "sha256:..."},
    store=store,
    concurrency=10,
)
```

`result.output` contains successful rows as plain dictionaries:

```python
[
    {"doc_id": "1", "label": "job", "confidence": 0.9},
    {"doc_id": "2", "label": "other", "confidence": 0.9},
    {"doc_id": "3", "label": "job", "confidence": 0.9},
]
```

If your data starts in pandas, convert it before calling `map`:

```python
records = df.to_dict(orient="records")
```

## API

```python
await dataotter.map(
    *,
    data: list[dict[str, Any]],
    row_id: str,
    name: str,
    fn: Callable[..., Awaitable[dict[str, Any]]],
    config: dict[str, Any] | None = None,
    store: dataotter.Store | None = None,
    concurrency: int = 10,
    errors: Literal["raise_after", "return"] = "raise_after",
    max_failures: int | None = 10,
    on_row_complete: Callable[[dataotter.RowEvent], None | Awaitable[None]] | None = None,
) -> dataotter.MapResult
```

Important rules:

- `data` must be a non-empty list of dict records.
- `row_id` is required, must name an existing key, and all row ID values must be unique non-null strings.
- `name` is the cache identity and may contain letters, numbers, dots, underscores, and hyphens.
- `fn` must be async and must return a `dict`.
- `dataotter` calls `fn` with expanded keyword arguments from each normalized record.
- Returned keys are persisted as output keys.
- Returned keys may not include the `row_id` key.

## Cache Identity

`name` identifies a persisted map. Use whatever naming convention fits your
workflow, for example `name=f"{step_name}-{version}"`.

Map-level validation is based on `config`. Reusing a `name` with a different
normalized config raises `ConfigMismatchError` before row work starts.

Row-level reuse is based on:

- same `name`
- same row ID value
- same normalized full record hash

If a row already has cached state for the same `name` and row ID, but the
current full record hash differs, `dataotter` raises `CacheMismatchError` before
starting new work.

`dataotter` does not hash Python function code. Put prompt hashes, model names,
or other correctness-affecting settings in `config`, or include them in `name`.

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

- `output`: list of successful row dicts with `row_id` plus returned keys.
- `errors`: list of failed row details.
- `stats`: row counts and timing fields.
- `run_id`: unique ID for the current run.
- `name`: cache identity.

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
store = dataotter.JsonlStore()

maps = store.list_maps()
manifest = store.get_map(result.name)

store.delete_map(result.name)
store.delete_maps(name="classify_document-v1")
```

The default store is a local JSONL store rooted at `.dataotter/` in the current
working directory. It is safe within a single async process, but it is not
intended for cross-process coordination. Pass a custom store or path via:

```python
store = dataotter.JsonlStore("/path/to/cache")
result = await dataotter.map(..., store=store)
```

## Development

```bash
uv run python -m pytest
uv run python -m compileall src tests
```
