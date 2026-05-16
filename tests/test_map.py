import asyncio
import json

import pytest

import dataotter


async def test_successful_map_and_cache_reuse(tmp_path):
    calls = 0

    async def classify(id: str, body: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"classification_label": body.upper()}

    data = [{"id": "1", "body": "a"}, {"id": "2", "body": "b"}]
    store = dataotter.JsonlStore(tmp_path)

    result = await dataotter.map(
        data=data,
        row_id="id",
        name="classify",
        fn=classify,
        store=store,
    )

    assert calls == 2
    assert result.name == "classify"
    assert result.output == [
        {"id": "1", "classification_label": "A"},
        {"id": "2", "classification_label": "B"},
    ]
    assert result.errors == []

    second = await dataotter.map(
        data=data,
        row_id="id",
        name="classify",
        fn=classify,
        store=store,
    )

    assert calls == 2
    assert second.stats.reused_rows == 2
    assert second.stats.attempted_rows == 0


async def test_config_mismatch_raises_before_running_work(tmp_path):
    calls = 0

    async def fn(id: str, text: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": text}

    store = dataotter.JsonlStore(tmp_path)
    data = [{"id": "1", "text": "x"}]

    await dataotter.map(
        data=data,
        row_id="id",
        name="config",
        fn=fn,
        config={"prompt": "a"},
        store=store,
    )

    with pytest.raises(dataotter.ConfigMismatchError):
        await dataotter.map(
            data=data,
            row_id="id",
            name="config",
            fn=fn,
            config={"prompt": "b"},
            store=store,
        )

    assert calls == 1


async def test_cache_mismatch_raises_before_running_work(tmp_path):
    calls = 0

    async def echo(id: str, text: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": text}

    store = dataotter.JsonlStore(tmp_path)

    await dataotter.map(
        data=[{"id": "1", "text": "old"}],
        row_id="id",
        name="echo",
        fn=echo,
        store=store,
    )

    with pytest.raises(dataotter.CacheMismatchError) as exc_info:
        await dataotter.map(
            data=[{"id": "1", "text": "new"}],
            row_id="id",
            name="echo",
            fn=echo,
            store=store,
        )

    assert calls == 1
    assert (
        exc_info.value.mismatches[0]["mismatch_type"]
        == "input_value_changed"
    )


async def test_failed_rows_are_retried_and_success_supersedes_error(tmp_path):
    seen: dict[int, int] = {}

    async def sometimes_fails(id: str, text: str) -> dict[str, object]:
        row_number = int(text)
        seen[row_number] = seen.get(row_number, 0) + 1
        if row_number == 1 and seen[row_number] == 1:
            raise RuntimeError("temporary")
        return {"value": row_number * 10}

    data = [{"id": "1", "text": "1"}, {"id": "2", "text": "2"}]
    store = dataotter.JsonlStore(tmp_path)

    first = await dataotter.map(
        data=data,
        row_id="id",
        name="numbers",
        fn=sometimes_fails,
        store=store,
        errors="return",
    )

    assert first.stats.failed_rows == 1
    assert [error["row_id"] for error in first.errors] == ["1"]

    second = await dataotter.map(
        data=data,
        row_id="id",
        name="numbers",
        fn=sometimes_fails,
        store=store,
    )

    assert second.errors == []
    assert sorted(second.output, key=lambda row: row["id"]) == [
        {"id": "1", "value": 10},
        {"id": "2", "value": 20},
    ]


async def test_full_record_hash_controls_cache_mismatch(tmp_path):
    calls = 0

    async def fn(**record: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": record["text"]}

    store = dataotter.JsonlStore(tmp_path)
    await dataotter.map(
        data=[{"id": "1", "text": "x", "ignored": "a"}],
        row_id="id",
        name="full_record",
        fn=fn,
        store=store,
    )

    with pytest.raises(dataotter.CacheMismatchError):
        await dataotter.map(
            data=[{"id": "1", "text": "x", "ignored": "b"}],
            row_id="id",
            name="full_record",
            fn=fn,
            store=store,
        )

    assert calls == 1


async def test_config_is_normalized_for_manifest(tmp_path):
    store = dataotter.JsonlStore(tmp_path)
    manifest = await store.ensure_map(
        name="config",
        row_id_column="id",
        config={"items": (1, 2), "nested": {"value": None}},
    )

    assert manifest["config"] == {"items": [1, 2], "nested": {"value": None}}
    assert store.get_map("config")["config_hash"] == manifest["config_hash"]


async def test_map_accepts_swappable_store_and_lists_maps(tmp_path):
    async def fn(id: str, text: str) -> dict[str, object]:
        return {"value": text}

    store = dataotter.JsonlStore(tmp_path)
    result = await dataotter.map(
        data=[{"id": "1", "text": "a"}],
        row_id="id",
        name="store_injection",
        fn=fn,
        store=store,
    )

    assert result.output == [{"id": "1", "value": "a"}]
    assert (
        store.list_maps(name="store_injection")[0]["name"]
        == "store_injection"
    )


async def test_map_uses_default_jsonl_store(monkeypatch, tmp_path):
    async def fn(id: str, text: str) -> dict[str, object]:
        return {"value": text}

    monkeypatch.chdir(tmp_path)

    result = await dataotter.map(
        data=[{"id": "1", "text": "a"}],
        row_id="id",
        name="default_store",
        fn=fn,
    )

    assert result.output == [{"id": "1", "value": "a"}]
    assert (tmp_path / ".dataotter" / "maps" / "default_store").exists()


async def test_jsonl_store_run_rejects_append_after_finish(tmp_path):
    store = dataotter.JsonlStore(tmp_path)
    store_run, _ = await store.begin_run(
        name="store_run",
        row_id_column="id",
        config={},
    )

    await store_run.finish()

    with pytest.raises(RuntimeError):
        await store_run.append_row_result({"type": "row_result"})


async def test_max_failures_allows_intermittent_failures(tmp_path):
    async def fn(id: str, text: str) -> dict[str, object]:
        value = int(text)
        if value in {0, 2}:
            raise RuntimeError(f"bad {value}")
        return {"value": value}

    result = await dataotter.map(
        data=[{"id": str(i), "text": str(i)} for i in range(5)],
        row_id="id",
        name="intermittent",
        fn=fn,
        store=dataotter.JsonlStore(tmp_path),
        concurrency=1,
        errors="return",
        max_failures=3,
    )

    assert result.stats.attempted_rows == 5
    assert result.stats.failed_rows == 2
    assert result.stats.stopped_early is False


async def test_on_row_complete_receives_attempted_and_reused_events(tmp_path):
    events: list[dataotter.RowEvent] = []

    async def fn(id: str, text: str) -> dict[str, object]:
        return {"value": text}

    data = [{"id": "1", "text": "a"}]
    store = dataotter.JsonlStore(tmp_path)

    first = await dataotter.map(
        data=data,
        row_id="id",
        name="events",
        fn=fn,
        store=store,
        on_row_complete=events.append,
    )
    await dataotter.map(
        data=data,
        row_id="id",
        name="events",
        fn=fn,
        store=store,
        on_row_complete=events.append,
    )

    assert [event.status for event in events] == ["success", "reused"]
    assert [event.attempted for event in events] == [True, False]
    assert first.stats.duration_seconds >= 0
    assert first.stats.started_at.endswith("Z")
    assert first.stats.finished_at.endswith("Z")


async def test_on_row_complete_accepts_async_callback(tmp_path):
    events: list[tuple[str, str]] = []

    async def fn(id: str, text: str) -> dict[str, object]:
        if text == "bad":
            raise RuntimeError("bad row")
        return {"value": text}

    async def on_row_complete(event: dataotter.RowEvent) -> None:
        await asyncio.sleep(0)
        events.append((event.row_id, event.status))

    result = await dataotter.map(
        data=[{"id": "1", "text": "ok"}, {"id": "2", "text": "bad"}],
        row_id="id",
        name="async_events",
        fn=fn,
        store=dataotter.JsonlStore(tmp_path),
        errors="return",
        max_failures=None,
        on_row_complete=on_row_complete,
    )

    assert sorted(events) == [("1", "success"), ("2", "error")]
    assert result.stats.failed_rows == 1


async def test_jsonl_store_compacts_on_begin_run(tmp_path):
    store = dataotter.JsonlStore(
        tmp_path,
        compact_min_records=1,
        compact_ratio=1.1,
    )
    store_run, _ = await store.begin_run(
        name="compact",
        row_id_column="id",
        config={},
    )
    for value in ["a", "b", "c"]:
        await store_run.append_row_result(
            {
                "type": "row_result",
                "run_id": "run",
                "name": "compact",
                "row_id": "1",
                "input_hash": "sha256:abc",
                "status": "success",
                "outputs": {"value": value},
                "created_at": "2026-05-12T00:00:00Z",
            }
        )
    await store_run.finish()

    second_run, _ = await store.begin_run(
        name="compact",
        row_id_column="id",
        config={},
    )
    await second_run.finish()

    results_path = next(tmp_path.rglob("results.jsonl"))
    lines = results_path.read_text().splitlines()
    assert len(lines) == 1
    assert '"value":"c"' in lines[0]


async def test_jsonl_store_ignores_incomplete_trailing_lines(tmp_path):
    store = dataotter.JsonlStore(tmp_path)
    store_run, _ = await store.begin_run(
        name="recovery",
        row_id_column="id",
        config={},
    )
    await store_run.append_row_result(
        {
            "type": "row_result",
            "run_id": "run",
            "name": "recovery",
            "row_id": "1",
            "input_hash": "sha256:ok",
            "status": "success",
            "outputs": {"value": "ok"},
            "created_at": "2026-05-12T00:00:00Z",
        }
    )
    await store_run.finish()
    results_path = next(tmp_path.rglob("results.jsonl"))
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write('{"type":"row_result",')

    states = await store.load_states(name="recovery")

    assert list(states) == ['"1"']
    assert states['"1"'].outputs == {"value": "ok"}


async def test_invalid_rows_names_and_functions(tmp_path):
    store = dataotter.JsonlStore(tmp_path)

    async def fn(id: str, text: str) -> dict[str, object]:
        return {"value": text}

    with pytest.raises(dataotter.InvalidRowIdError):
        await dataotter.map(
            data=[{"id": "1", "text": "a"}, {"id": "1", "text": "b"}],
            row_id="id",
            name="bad_rows",
            fn=fn,
            store=store,
        )

    def sync_fn(id: str, text: str) -> dict[str, object]:
        return {"value": text}

    with pytest.raises(dataotter.InvalidFunctionError):
        await dataotter.map(
            data=[{"id": "1", "text": "a"}],
            row_id="id",
            name="bad_fn",
            fn=sync_fn,
            store=store,
        )

    with pytest.raises(dataotter.InvalidRowIdError):
        await dataotter.map(
            data=[{"id": 1, "text": "a"}],
            row_id="id",
            name="bad_row_type",
            fn=fn,
            store=store,
        )

    with pytest.raises(dataotter.InvalidRowIdError):
        await dataotter.map(
            data=[{"text": "a"}],
            row_id="id",
            name="missing_row_id",
            fn=fn,
            store=store,
        )

    with pytest.raises(ValueError):
        await dataotter.map(
            data=[{"id": "1", "text": "a"}],
            row_id="id",
            name="../bad",
            fn=fn,
            store=store,
        )

    with pytest.raises(ValueError):
        await dataotter.map(
            data=[],
            row_id="id",
            name="empty",
            fn=fn,
            store=store,
        )

    with pytest.raises(TypeError):
        await dataotter.map(
            data=[{"id": "1", "text": "a", "": "bad"}],
            row_id="id",
            name="bad_key",
            fn=fn,
            store=store,
        )

    with pytest.raises(dataotter.InvalidFunctionError):
        await dataotter.map(
            data=[{"id": "1", "text": "a", "extra": "b"}],
            row_id="id",
            name="unexpected_arg",
            fn=fn,
            store=store,
        )

    with pytest.raises(dataotter.InvalidFunctionError):
        await dataotter.map(
            data=[{"id": "1", "other": "a"}],
            row_id="id",
            name="missing_required_arg",
            fn=fn,
            store=store,
        )

    with pytest.raises(ValueError, match="max_failures must be >= 1"):
        await dataotter.map(
            data=[{"id": "1", "text": "a"}],
            row_id="id",
            name="bad_max_failures",
            fn=fn,
            store=store,
            max_failures=0,
        )


async def test_output_validation_is_persisted_as_row_error(tmp_path):
    async def collides(id: str, text: str) -> dict[str, object]:
        return {"id": "other"}

    result = await dataotter.map(
        data=[{"id": "1", "text": "a"}],
        row_id="id",
        name="row_id_collision",
        fn=collides,
        store=dataotter.JsonlStore(tmp_path),
        errors="return",
    )

    assert result.output == []
    assert [error["error_type"] for error in result.errors] == ["ValueError"]
    assert result.stats.failed_rows == 1


async def test_non_dict_return_is_persisted_as_row_error(tmp_path):
    async def fn(id: str, text: str) -> list[str]:
        return [text]

    result = await dataotter.map(
        data=[{"id": "1", "text": "a"}],
        row_id="id",
        name="non_dict_output",
        fn=fn,
        store=dataotter.JsonlStore(tmp_path),
        errors="return",
    )

    assert [error["error_type"] for error in result.errors] == ["TypeError"]
    assert result.output == []


async def test_semi_fail_fast_bounds_started_rows(tmp_path):
    active = 0
    max_active = 0
    attempted: list[int] = []

    async def fn(id: str, text: str) -> dict[str, object]:
        nonlocal active, max_active
        row_number = int(text)
        attempted.append(row_number)
        active += 1
        max_active = max(max_active, active)
        try:
            if row_number == 0:
                await asyncio.sleep(0.01)
                raise RuntimeError("stop")
            await asyncio.sleep(0.05)
            return {"value": row_number}
        finally:
            active -= 1

    result = await dataotter.map(
        data=[{"id": str(i), "text": str(i)} for i in range(8)],
        row_id="id",
        name="fail_fast",
        fn=fn,
        store=dataotter.JsonlStore(tmp_path),
        concurrency=2,
        errors="return",
        max_failures=1,
    )

    assert max_active <= 2
    assert result.stats.stopped_early is True
    assert result.stats.attempted_rows == 2
    assert result.stats.not_started_rows == 6
    assert attempted == [0, 1]


async def test_max_failures_none_attempts_all_rows(tmp_path):
    async def fn(id: str, text: str) -> dict[str, object]:
        raise RuntimeError(text)

    result = await dataotter.map(
        data=[{"id": str(i), "text": str(i)} for i in range(4)],
        row_id="id",
        name="all_failures",
        fn=fn,
        store=dataotter.JsonlStore(tmp_path),
        concurrency=2,
        errors="return",
        max_failures=None,
    )

    assert result.stats.attempted_rows == 4
    assert result.stats.failed_rows == 4
    assert result.stats.stopped_early is False


async def test_map_failed_error_contains_partial_result(tmp_path):
    async def fn(id: str, text: str) -> dict[str, object]:
        raise RuntimeError(text)

    with pytest.raises(dataotter.MapFailedError) as exc_info:
        await dataotter.map(
            data=[{"id": "1", "text": "boom"}],
            row_id="id",
            name="raises",
            fn=fn,
            store=dataotter.JsonlStore(tmp_path),
        )

    assert exc_info.value.result.stats.failed_rows == 1
    assert [
        error["error_message"]
        for error in exc_info.value.result.errors
    ] == ["boom"]


async def test_cancellation_cancels_in_flight_rows_and_reraises(tmp_path):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fn(id: str, text: str) -> dict[str, object]:
        started.set()
        if cancelled.is_set():
            return {"value": text}
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {"value": text}

    store = dataotter.JsonlStore(tmp_path)
    task = asyncio.create_task(
        dataotter.map(
            data=[{"id": "1", "text": "a"}],
            row_id="id",
            name="cancel",
            fn=fn,
            store=store,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert cancelled.is_set()
    assert await store.load_states(name="cancel") == {}

    retry = await dataotter.map(
        data=[{"id": "1", "text": "a"}],
        row_id="id",
        name="cancel",
        fn=fn,
        store=store,
    )
    assert retry.stats.reused_rows == 0
    assert retry.stats.attempted_rows == 1


async def test_delete_maps(tmp_path):
    async def fn(id: str, text: str) -> dict[str, object]:
        return {"value": text}

    store = dataotter.JsonlStore(tmp_path)
    result = await dataotter.map(
        data=[{"id": "1", "text": "a"}],
        row_id="id",
        name="delete_me",
        fn=fn,
        store=store,
    )

    assert store.get_map(result.name)["name"] == "delete_me"
    assert store.delete_map(result.name) is True
    assert store.delete_map(result.name) is False

    await dataotter.map(
        data=[{"id": "1", "text": "a"}],
        row_id="id",
        name="delete_family",
        fn=fn,
        store=store,
    )
    assert store.delete_maps(name="delete_family") == 1


async def test_results_jsonl_contains_one_record_per_attempted_row(tmp_path):
    async def fn(id: str, text: str) -> dict[str, object]:
        return {"value": text}

    store = dataotter.JsonlStore(tmp_path)
    result = await dataotter.map(
        data=[{"id": "1", "text": "a"}, {"id": "2", "text": "b"}],
        row_id="id",
        name="jsonl_records",
        fn=fn,
        store=store,
    )

    records = [
        json.loads(line)
        for line in next(
            tmp_path.rglob("results.jsonl")
        ).read_text().splitlines()
    ]
    assert [record["row_id"] for record in records] == ["1", "2"]
    assert {record["name"] for record in records} == {result.name}
