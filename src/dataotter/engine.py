from typing import Any, Awaitable, Callable

import pandas as pd

from dataotter._bindings import BindingsInput
from dataotter._normalize import stable_hash
from dataotter._runner import (
    DEFAULT_CONCURRENCY,
    DEFAULT_ERRORS,
    DEFAULT_MAX_FAILURES,
    DEFAULT_VERSION,
    ErrorsMode,
    run_engine_map,
)
from dataotter.stores import JsonlStore, Store, normalize_config
from dataotter.types import MapResult, RowEvent


class Engine:
    def __init__(self, *, store: Store | None = None) -> None:
        self.store = store if store is not None else JsonlStore()

    async def map(
        self,
        *,
        data: pd.DataFrame,
        row_id: str,
        step_name: str,
        fn: Callable[..., Awaitable[dict[str, Any]]],
        inputs: BindingsInput,
        outputs: BindingsInput,
        config: dict[str, Any] | None = None,
        version: str = DEFAULT_VERSION,
        concurrency: int = DEFAULT_CONCURRENCY,
        errors: ErrorsMode = DEFAULT_ERRORS,
        max_failures: int | None = DEFAULT_MAX_FAILURES,
        on_row_complete: Callable[[RowEvent], None | Awaitable[None]] | None = None,
    ) -> MapResult:
        return await run_engine_map(
            engine=self,
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

    def derive_map_id(
        self,
        *,
        step_name: str,
        version: str,
        input_args: list[str],
        outputs: dict[str, str],
        config: dict[str, Any],
    ) -> str:
        normalized_config = normalize_config(config)
        return stable_hash(
            {
                "step_name": step_name,
                "version": version,
                "input_args": input_args,
                "outputs": outputs,
                "config": normalized_config,
            }
        )

    def list_maps(self, *, step_name: str | None = None) -> pd.DataFrame:
        return self.store.list_maps(step_name=step_name)

    def get_map(self, map_id: str) -> dict[str, Any]:
        return self.store.get_map(map_id)

    def delete_map(self, map_id: str) -> bool:
        return self.store.delete_map(map_id)

    def delete_maps(
        self,
        *,
        step_name: str | None = None,
        version: str | None = None,
    ) -> int:
        return self.store.delete_maps(step_name=step_name, version=version)
