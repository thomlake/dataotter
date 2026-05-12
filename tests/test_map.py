import asyncio
import json

import pandas as pd
import pytest

import dataotter


async def test_successful_map_and_cache_reuse(tmp_path):
    calls = 0

    async def classify(text: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"label": text.upper()}

    df = pd.DataFrame({"id": ["1", "2"], "body": ["a", "b"]})
    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))

    result = await dataotter.map(
        data=df,
        row_id="id",
        step_name="classify",
        fn=classify,
        inputs={"body": "text"},
        outputs={"label": "classification_label"},
        engine=engine,
    )

    assert calls == 2
    assert result.output.to_dict(orient="records") == [
        {"id": "1", "classification_label": "A"},
        {"id": "2", "classification_label": "B"},
    ]
    assert result.errors.empty

    second = await dataotter.map(
        data=df,
        row_id="id",
        step_name="classify",
        fn=classify,
        inputs={"body": "text"},
        outputs={"label": "classification_label"},
        engine=engine,
    )

    assert calls == 2
    assert second.stats.reused_rows == 2
    assert second.stats.attempted_rows == 0


async def test_failed_rows_are_retried_and_success_supersedes_error(tmp_path):
    seen: dict[int, int] = {}

    async def sometimes_fails(text: str) -> dict[str, object]:
        row_number = int(text)
        seen[row_number] = seen.get(row_number, 0) + 1
        if row_number == 1 and seen[row_number] == 1:
            raise RuntimeError("temporary")
        return {"value": row_number * 10}

    df = pd.DataFrame({"id": ["1", "2"], "text": ["1", "2"]})
    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))

    first = await dataotter.map(
        data=df,
        row_id="id",
        step_name="numbers",
        fn=sometimes_fails,
        inputs=["text"],
        outputs=["value"],
        engine=engine,
        errors="return",
    )

    assert first.stats.failed_rows == 1
    assert first.errors["row_id"].tolist() == ["1"]

    second = await dataotter.map(
        data=df,
        row_id="id",
        step_name="numbers",
        fn=sometimes_fails,
        inputs=["text"],
        outputs=["value"],
        engine=engine,
    )

    assert second.errors.empty
    assert second.output.sort_values("id").to_dict(orient="records") == [
        {"id": "1", "value": 10},
        {"id": "2", "value": 20},
    ]


async def test_cache_mismatch_raises_before_running_work(tmp_path):
    calls = 0

    async def echo(text: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": text}

    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))
    df = pd.DataFrame({"id": ["1"], "text": ["old"]})
    changed = pd.DataFrame({"id": ["1"], "text": ["new"]})

    await dataotter.map(
        data=df,
        row_id="id",
        step_name="echo",
        fn=echo,
        inputs=["text"],
        outputs=["value"],
        engine=engine,
    )

    with pytest.raises(dataotter.CacheMismatchError) as exc_info:
        await dataotter.map(
            data=changed,
            row_id="id",
            step_name="echo",
            fn=echo,
            inputs=["text"],
            outputs=["value"],
            engine=engine,
        )

    assert calls == 1
    assert exc_info.value.mismatches["mismatch_type"].tolist() == ["input_value_changed"]


async def test_kwargs_normalization_controls_cache_mismatch(tmp_path):
    calls = 0

    async def echo(value: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": value}

    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))
    await dataotter.map(
        data=pd.DataFrame({"id": ["1"], "value": [pd.NA]}),
        row_id="id",
        step_name="normalize_kwargs",
        fn=echo,
        inputs=["value"],
        outputs=["value"],
        engine=engine,
    )
    reused = await dataotter.map(
        data=pd.DataFrame({"id": ["1"], "value": [None]}),
        row_id="id",
        step_name="normalize_kwargs",
        fn=echo,
        inputs=["value"],
        outputs=["value"],
        engine=engine,
    )

    assert calls == 1
    assert reused.stats.reused_rows == 1


async def test_config_and_output_bindings_change_map_id(tmp_path):
    async def fn(text: str) -> dict[str, object]:
        return {"label": text}

    df = pd.DataFrame({"id": ["1"], "text": ["x"]})
    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))

    first = await dataotter.map(
        data=df,
        row_id="id",
        step_name="experiment",
        fn=fn,
        inputs=["text"],
        outputs={"label": "label_a"},
        config={"prompt": "a"},
        engine=engine,
    )
    second = await dataotter.map(
        data=df,
        row_id="id",
        step_name="experiment",
        fn=fn,
        inputs=["text"],
        outputs={"label": "label_b"},
        config={"prompt": "b"},
        engine=engine,
    )

    assert first.map_id != second.map_id
    assert engine.list_maps(step_name="experiment").shape[0] == 2


async def test_source_column_and_row_id_column_renames_reuse_cache(tmp_path):
    calls = 0

    async def fn(text: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": text.upper()}

    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))
    first = await dataotter.map(
        data=pd.DataFrame({"doc_id": ["1"], "raw_text": ["hello"]}),
        row_id="doc_id",
        step_name="rename_safe",
        fn=fn,
        inputs={"raw_text": "text"},
        outputs=["value"],
        engine=engine,
    )
    second = await dataotter.map(
        data=pd.DataFrame({"id": ["1"], "renamed_text": ["hello"]}),
        row_id="id",
        step_name="rename_safe",
        fn=fn,
        inputs={"renamed_text": "text"},
        outputs=["value"],
        engine=engine,
    )

    assert calls == 1
    assert first.map_id == second.map_id
    assert second.stats.reused_rows == 1
    assert second.output.to_dict(orient="records") == [{"id": "1", "value": "HELLO"}]


async def test_changing_function_argument_name_creates_new_map_id(tmp_path):
    async def fn(**kwargs: object) -> dict[str, object]:
        return {"value": next(iter(kwargs.values()))}

    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))
    first = await dataotter.map(
        data=pd.DataFrame({"id": ["1"], "text": ["hello"]}),
        row_id="id",
        step_name="arg_identity",
        fn=fn,
        inputs={"text": "text"},
        outputs=["value"],
        engine=engine,
    )
    second = await dataotter.map(
        data=pd.DataFrame({"id": ["1"], "text": ["hello"]}),
        row_id="id",
        step_name="arg_identity",
        fn=fn,
        inputs={"text": "renamed_text"},
        outputs=["value"],
        engine=engine,
    )

    assert first.map_id != second.map_id
    assert second.stats.attempted_rows == 1


async def test_engine_normalizes_config_for_map_id_and_manifest(tmp_path):
    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))
    config = {"nested": {"value": pd.NA}, "items": (1, 2)}

    map_id = engine.derive_map_id(
        step_name="config",
        version="1",
        input_args=["text"],
        outputs={"value": "value"},
        config=config,
    )
    manifest = await engine.store.ensure_map(
        map_id=map_id,
        step_name="config",
        version="1",
        row_id_column="id",
        inputs={"text": "text"},
        outputs={"value": "value"},
        config=config,
    )

    assert manifest["config"] == {"items": [1, 2], "nested": {"value": None}}
    assert manifest["config_hash"] == engine.get_map(map_id)["config_hash"]


async def test_engine_accepts_swappable_store(tmp_path):
    async def fn(text: str) -> dict[str, object]:
        return {"value": text}

    store = dataotter.JsonlStore(tmp_path)
    engine = dataotter.Engine(store=store)
    result = await engine.map(
        data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
        row_id="id",
        step_name="store_injection",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
    )

    assert result.output.to_dict(orient="records") == [{"id": "1", "value": "a"}]
    assert engine.list_maps(step_name="store_injection").shape[0] == 1


async def test_jsonl_store_run_rejects_append_after_finish(tmp_path):
    store = dataotter.JsonlStore(tmp_path)
    engine = dataotter.Engine(store=store)
    map_id = engine.derive_map_id(
        step_name="store_run",
        version="1",
        input_args=["text"],
        outputs={"value": "value"},
        config={},
    )
    store_run, _ = await store.begin_run(
        map_id=map_id,
        step_name="store_run",
        version="1",
        row_id_column="id",
        inputs={"text": "text"},
        outputs={"value": "value"},
        config={},
    )

    await store_run.finish()

    with pytest.raises(RuntimeError):
        await store_run.append_row_result({"type": "row_result"})


async def test_max_failures_allows_intermittent_failures(tmp_path):
    async def fn(text: str) -> dict[str, object]:
        value = int(text)
        if value in {0, 2}:
            raise RuntimeError(f"bad {value}")
        return {"value": value}

    result = await dataotter.map(
        data=pd.DataFrame({"id": [str(i) for i in range(5)], "text": [str(i) for i in range(5)]}),
        row_id="id",
        step_name="intermittent",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=dataotter.Engine(store=dataotter.JsonlStore(tmp_path)),
        concurrency=1,
        errors="return",
        max_failures=3,
    )

    assert result.stats.attempted_rows == 5
    assert result.stats.failed_rows == 2
    assert result.stats.stopped_early is False


async def test_on_row_complete_receives_attempted_and_reused_events(tmp_path):
    events: list[dataotter.RowEvent] = []

    async def fn(text: str) -> dict[str, object]:
        return {"value": text}

    df = pd.DataFrame({"id": ["1"], "text": ["a"]})
    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))

    first = await dataotter.map(
        data=df,
        row_id="id",
        step_name="events",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=engine,
        on_row_complete=events.append,
    )
    await dataotter.map(
        data=df,
        row_id="id",
        step_name="events",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=engine,
        on_row_complete=events.append,
    )

    assert [event.status for event in events] == ["success", "reused"]
    assert [event.attempted for event in events] == [True, False]
    assert first.stats.duration_seconds >= 0
    assert first.stats.started_at.endswith("Z")
    assert first.stats.finished_at.endswith("Z")


async def test_on_row_complete_accepts_async_callback(tmp_path):
    events: list[tuple[str, str]] = []

    async def fn(text: str) -> dict[str, object]:
        if text == "bad":
            raise RuntimeError("bad row")
        return {"value": text}

    async def on_row_complete(event: dataotter.RowEvent) -> None:
        await asyncio.sleep(0)
        events.append((event.row_id, event.status))

    result = await dataotter.map(
        data=pd.DataFrame({"id": ["1", "2"], "text": ["ok", "bad"]}),
        row_id="id",
        step_name="async_events",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=dataotter.Engine(store=dataotter.JsonlStore(tmp_path)),
        errors="return",
        max_failures=None,
        on_row_complete=on_row_complete,
    )

    assert sorted(events) == [("1", "success"), ("2", "error")]
    assert result.stats.failed_rows == 1


async def test_jsonl_store_compacts_on_begin_run(tmp_path):
    store = dataotter.JsonlStore(tmp_path, compact_min_records=1, compact_ratio=1.1)
    engine = dataotter.Engine(store=store)
    map_id = engine.derive_map_id(
        step_name="compact",
        version="1",
        input_args=["text"],
        outputs={"value": "value"},
        config={},
    )
    store_run, _ = await store.begin_run(
        map_id=map_id,
        step_name="compact",
        version="1",
        row_id_column="id",
        inputs={"text": "text"},
        outputs={"value": "value"},
        config={},
    )
    for value in ["a", "b", "c"]:
        await store_run.append_row_result(
            {
                "type": "row_result",
                "run_id": "run",
                "map_id": map_id,
                "step_name": "compact",
                "version": "1",
                "row_id": "1",
                "input_hash": "sha256:abc",
                "status": "success",
                "outputs": {"value": value},
                "created_at": "2026-05-12T00:00:00Z",
            }
        )
    await store_run.finish()

    second_run, _ = await store.begin_run(
        map_id=map_id,
        step_name="compact",
        version="1",
        row_id_column="id",
        inputs={"text": "text"},
        outputs={"value": "value"},
        config={},
    )
    await second_run.finish()

    results_path = next(tmp_path.rglob("results.jsonl"))
    lines = results_path.read_text().splitlines()
    assert len(lines) == 1
    assert '"value":"c"' in lines[0]


async def test_jsonl_store_ignores_incomplete_trailing_lines(tmp_path):
    store = dataotter.JsonlStore(tmp_path)
    engine = dataotter.Engine(store=store)
    map_id = engine.derive_map_id(
        step_name="recovery",
        version="1",
        input_args=["text"],
        outputs={"value": "value"},
        config={},
    )
    store_run, _ = await store.begin_run(
        map_id=map_id,
        step_name="recovery",
        version="1",
        row_id_column="id",
        inputs={"text": "text"},
        outputs={"value": "value"},
        config={},
    )
    await store_run.append_row_result(
        {
            "type": "row_result",
            "run_id": "run",
            "map_id": map_id,
            "step_name": "recovery",
            "version": "1",
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

    states = await store.load_states(step_name="recovery", version="1", map_id=map_id)

    assert list(states) == ['"1"']
    assert states['"1"'].outputs == {"value": "ok"}


async def test_invalid_rows_bindings_and_functions(tmp_path):
    df = pd.DataFrame({"id": ["1", "1"], "text": ["a", "b"]})
    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))

    async def fn(text: str) -> dict[str, object]:
        return {"value": text}

    with pytest.raises(dataotter.InvalidRowIdError):
        await dataotter.map(
            data=df,
            row_id="id",
            step_name="bad_rows",
            fn=fn,
            inputs=["text"],
            outputs=["value"],
            engine=engine,
        )

    with pytest.raises(dataotter.InvalidBindingError):
        await dataotter.map(
            data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
            row_id="id",
            step_name="bad_outputs",
            fn=fn,
            inputs=["text"],
            outputs={"value": "id"},
            engine=engine,
        )

    def sync_fn(text: str) -> dict[str, object]:
        return {"value": text}

    with pytest.raises(dataotter.InvalidFunctionError):
        await dataotter.map(
            data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
            row_id="id",
            step_name="bad_fn",
            fn=sync_fn,
            inputs=["text"],
            outputs=["value"],
            engine=engine,
        )

    with pytest.raises(dataotter.InvalidRowIdError):
        await dataotter.map(
            data=pd.DataFrame({"id": [1], "text": ["a"]}),
            row_id="id",
            step_name="bad_row_type",
            fn=fn,
            inputs=["text"],
            outputs=["value"],
            engine=engine,
        )

    with pytest.raises(ValueError):
        await dataotter.map(
            data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
            row_id="id",
            step_name="../bad",
            fn=fn,
            inputs=["text"],
            outputs=["value"],
            engine=engine,
        )

    with pytest.raises(ValueError):
        await dataotter.map(
            data=pd.DataFrame({"id": [], "text": []}),
            row_id="id",
            step_name="empty",
            fn=fn,
            inputs=["text"],
            outputs=["value"],
            engine=engine,
        )

    with pytest.raises(dataotter.InvalidBindingError):
        await dataotter.map(
            data=pd.DataFrame({"id": ["1"], "a": ["a"], "b": ["b"]}),
            row_id="id",
            step_name="duplicate_args",
            fn=fn,
            inputs={"a": "text", "b": "text"},
            outputs=["value"],
            engine=engine,
        )

    with pytest.raises(dataotter.InvalidBindingError):
        await dataotter.map(
            data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
            row_id="id",
            step_name="missing_input",
            fn=fn,
            inputs=["missing"],
            outputs=["value"],
            engine=engine,
        )

    with pytest.raises(dataotter.InvalidFunctionError):
        await dataotter.map(
            data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
            row_id="id",
            step_name="missing_required_arg",
            fn=fn,
            inputs={"text": "other"},
            outputs=["value"],
            engine=engine,
        )

    with pytest.raises(ValueError, match="max_failures must be >= 1"):
        await dataotter.map(
            data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
            row_id="id",
            step_name="bad_max_failures",
            fn=fn,
            inputs=["text"],
            outputs=["value"],
            engine=engine,
            max_failures=0,
        )


async def test_strict_output_validation_is_persisted_as_row_error(tmp_path):
    async def fn(text: str) -> dict[str, object]:
        return {"wrong": text}

    result = await dataotter.map(
        data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
        row_id="id",
        step_name="strict_outputs",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=dataotter.Engine(store=dataotter.JsonlStore(tmp_path)),
        errors="return",
    )

    assert result.output.empty
    assert result.errors["error_type"].tolist() == ["ValueError"]
    assert result.stats.failed_rows == 1


async def test_non_dict_return_is_persisted_as_row_error(tmp_path):
    async def fn(text: str) -> list[str]:
        return [text]

    result = await dataotter.map(
        data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
        row_id="id",
        step_name="non_dict_output",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=dataotter.Engine(store=dataotter.JsonlStore(tmp_path)),
        errors="return",
    )

    assert result.errors["error_type"].tolist() == ["TypeError"]
    assert result.output.empty


async def test_semi_fail_fast_bounds_started_rows(tmp_path):
    active = 0
    max_active = 0
    attempted: list[int] = []

    async def fn(text: str) -> dict[str, object]:
        nonlocal active, max_active
        row_id = int(text)
        attempted.append(row_id)
        active += 1
        max_active = max(max_active, active)
        try:
            if row_id == 0:
                await asyncio.sleep(0.01)
                raise RuntimeError("stop")
            await asyncio.sleep(0.05)
            return {"value": row_id}
        finally:
            active -= 1

    df = pd.DataFrame({"id": [str(i) for i in range(8)], "text": [str(i) for i in range(8)]})
    result = await dataotter.map(
        data=df,
        row_id="id",
        step_name="fail_fast",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=dataotter.Engine(store=dataotter.JsonlStore(tmp_path)),
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
    async def fn(text: str) -> dict[str, object]:
        raise RuntimeError(text)

    result = await dataotter.map(
        data=pd.DataFrame({"id": [str(i) for i in range(4)], "text": [str(i) for i in range(4)]}),
        row_id="id",
        step_name="all_failures",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=dataotter.Engine(store=dataotter.JsonlStore(tmp_path)),
        concurrency=2,
        errors="return",
        max_failures=None,
    )

    assert result.stats.attempted_rows == 4
    assert result.stats.failed_rows == 4
    assert result.stats.stopped_early is False


async def test_map_failed_error_contains_partial_result(tmp_path):
    async def fn(text: str) -> dict[str, object]:
        raise RuntimeError(text)

    with pytest.raises(dataotter.MapFailedError) as exc_info:
        await dataotter.map(
            data=pd.DataFrame({"id": ["1"], "text": ["boom"]}),
            row_id="id",
            step_name="raises",
            fn=fn,
            inputs=["text"],
            outputs=["value"],
            engine=dataotter.Engine(store=dataotter.JsonlStore(tmp_path)),
        )

    assert exc_info.value.result.stats.failed_rows == 1
    assert exc_info.value.result.errors["error_message"].tolist() == ["boom"]


async def test_cancellation_drains_in_flight_rows_and_reraises(tmp_path):
    started = asyncio.Event()

    async def fn(text: str) -> dict[str, object]:
        started.set()
        await asyncio.sleep(0.02)
        return {"value": text}

    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))
    task = asyncio.create_task(
        dataotter.map(
            data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
            row_id="id",
            step_name="cancel",
            fn=fn,
            inputs=["text"],
            outputs=["value"],
            engine=engine,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    retry = await dataotter.map(
        data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
        row_id="id",
        step_name="cancel",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=engine,
    )
    assert retry.stats.reused_rows == 1


async def test_delete_maps(tmp_path):
    async def fn(text: str) -> dict[str, object]:
        return {"value": text}

    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))
    result = await dataotter.map(
        data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
        row_id="id",
        step_name="delete_me",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=engine,
    )

    assert engine.get_map(result.map_id)["step_name"] == "delete_me"
    assert engine.delete_map(result.map_id) is True
    assert engine.delete_map(result.map_id) is False

    await dataotter.map(
        data=pd.DataFrame({"id": ["1"], "text": ["a"]}),
        row_id="id",
        step_name="delete_family",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=engine,
    )
    assert engine.delete_maps(step_name="delete_family") == 1


async def test_results_jsonl_contains_one_record_per_attempted_row(tmp_path):
    async def fn(text: str) -> dict[str, object]:
        return {"value": text}

    engine = dataotter.Engine(store=dataotter.JsonlStore(tmp_path))
    result = await dataotter.map(
        data=pd.DataFrame({"id": ["1", "2"], "text": ["a", "b"]}),
        row_id="id",
        step_name="jsonl_records",
        fn=fn,
        inputs=["text"],
        outputs=["value"],
        engine=engine,
    )

    records = [
        json.loads(line)
        for line in next(tmp_path.rglob("results.jsonl")).read_text().splitlines()
    ]
    assert [record["row_id"] for record in records] == ["1", "2"]
    assert {record["map_id"] for record in records} == {result.map_id}
