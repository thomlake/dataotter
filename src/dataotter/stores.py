import asyncio
import json
import shutil
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Protocol

from dataotter._normalize import (
    canonical_json,
    normalize_value,
    now_iso,
    stable_hash,
)
from dataotter.errors import ConfigMismatchError
from dataotter.types import STATUS_ERROR

MANIFEST_FILE = "manifest.json"
RESULTS_FILE = "results.jsonl"
MAPS_DIR = "maps"
PACKAGE_VERSION = _pkg_version("dataotter")
RECORD_TYPE_ROW_RESULT = "row_result"
RECORD_TYPE_MAP_MANIFEST = "map_manifest"
DEFAULT_COMPACT_MIN_RECORDS = 10_000
DEFAULT_COMPACT_RATIO = 2.0


@dataclass(frozen=True)
class RowState:
    row_id: Any
    row_key: str
    input_hash: str
    status: str
    outputs: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: str


class Store(Protocol):
    async def begin_run(
        self,
        *,
        name: str,
        row_id_column: str,
        config: dict[str, Any],
    ) -> tuple[StoreRun, dict[str, RowState]]: ...

    async def ensure_map(
        self,
        *,
        name: str,
        row_id_column: str,
        config: dict[str, Any],
    ) -> dict[str, Any]: ...

    async def load_states(
        self,
        *,
        name: str,
    ) -> dict[str, RowState]: ...

    def list_maps(
        self,
        *,
        name: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_map(self, name: str) -> dict[str, Any]: ...

    def delete_map(self, name: str) -> bool: ...

    def delete_maps(self, *, name: str | None = None) -> int: ...


class StoreRun(Protocol):
    async def append_row_result(self, record: dict[str, Any]) -> None: ...

    async def finish(self) -> None: ...


class JsonlStore:
    def __init__(
        self,
        cache: str | Path = ".dataotter",
        *,
        compact_min_records: int = DEFAULT_COMPACT_MIN_RECORDS,
        compact_ratio: float = DEFAULT_COMPACT_RATIO,
    ) -> None:
        if compact_min_records < 0:
            raise ValueError("compact_min_records must be >= 0")
        if compact_ratio <= 1.0:
            raise ValueError("compact_ratio must be > 1.0")
        self.cache = Path(cache)
        self._locks: dict[str, asyncio.Lock] = {}
        self.compact_min_records = compact_min_records
        self.compact_ratio = compact_ratio

    async def begin_run(
        self,
        *,
        name: str,
        row_id_column: str,
        config: dict[str, Any],
    ) -> tuple[JsonlStoreRun, dict[str, RowState]]:
        await self.ensure_map(
            name=name,
            row_id_column=row_id_column,
            config=config,
        )
        lock = self._locks.setdefault(name, asyncio.Lock())
        path = self._results_path(name=name)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with lock:
            records, total_records = _read_latest_records(path)
            should_compact = self._should_compact(
                total_records=total_records,
                latest_count=len(records),
            )
            if should_compact:
                _rewrite_results(path, records.values())
        states = {
            row_key: _record_to_row_state(record)
            for row_key, record in records.items()
        }
        handle = path.open("a", encoding="utf-8")
        return (
            JsonlStoreRun(
                store=self,
                name=name,
                lock=lock,
                handle=handle,
            ),
            states,
        )

    async def ensure_map(
        self,
        *,
        name: str,
        row_id_column: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_config = normalize_config(config)
        path = self._map_path(name=name)
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / MANIFEST_FILE

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["config"] != normalized_config:
                raise ConfigMismatchError(name=name)
            return manifest

        now = now_iso()
        manifest = {
            "type": RECORD_TYPE_MAP_MANIFEST,
            "dataotter_version": PACKAGE_VERSION,
            "name": name,
            "row_id_column": row_id_column,
            "config_hash": stable_hash(normalized_config),
            "config": normalized_config,
            "created_at": now,
            "updated_at": now,
        }
        _atomic_write_text(manifest_path, canonical_json(manifest) + "\n")
        return manifest

    async def load_states(
        self,
        *,
        name: str,
    ) -> dict[str, RowState]:
        results_path = self._results_path(name=name)
        records, _ = _read_latest_records(results_path)
        return {
            row_key: _record_to_row_state(record)
            for row_key, record in records.items()
        }

    def list_maps(self, *, name: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for manifest_path in self.cache.glob(f"{MAPS_DIR}/*/{MANIFEST_FILE}"):
            manifest = json.loads(manifest_path.read_text())
            if name is not None and manifest["name"] != name:
                continue
            states = self._load_states_for_manifest(manifest)
            failures = sum(
                1 for state in states.values()
                if state.status == STATUS_ERROR
            )
            rows.append(
                {
                    "name": manifest["name"],
                    "row_id_column": manifest["row_id_column"],
                    "config": manifest["config"],
                    "rows": len(states),
                    "failures": failures,
                    "created_at": manifest["created_at"],
                    "updated_at": manifest["updated_at"],
                }
            )
        return rows

    def get_map(self, name: str) -> dict[str, Any]:
        manifest_path = self._map_path(name=name) / MANIFEST_FILE
        if not manifest_path.exists():
            raise KeyError(f"No cached map found for name {name!r}")
        return json.loads(manifest_path.read_text())

    def delete_map(self, name: str) -> bool:
        path = self._map_path(name=name)
        if not path.exists():
            return False
        shutil.rmtree(path)
        return True

    def delete_maps(self, *, name: str | None = None) -> int:
        deleted = 0
        for manifest_path in list(
            self.cache.glob(f"{MAPS_DIR}/*/{MANIFEST_FILE}")
        ):
            manifest = json.loads(manifest_path.read_text())
            if name is not None and manifest["name"] != name:
                continue
            shutil.rmtree(manifest_path.parent)
            deleted += 1
        return deleted

    def _map_path(self, *, name: str) -> Path:
        return self.cache / MAPS_DIR / _safe_path(name)

    def _results_path(self, *, name: str) -> Path:
        return self._map_path(name=name) / RESULTS_FILE

    async def _touch_manifest(self, *, name: str) -> None:
        manifest_path = self._map_path(name=name) / MANIFEST_FILE
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        manifest["updated_at"] = now_iso()
        _atomic_write_text(manifest_path, canonical_json(manifest) + "\n")

    def _load_states_for_manifest(
        self,
        manifest: dict[str, Any],
    ) -> dict[str, RowState]:
        path = self._results_path(name=manifest["name"])
        records, _ = _read_latest_records(path)
        return {
            row_key: _record_to_row_state(record)
            for row_key, record in records.items()
        }

    def _should_compact(
        self,
        *,
        total_records: int,
        latest_count: int,
    ) -> bool:
        if latest_count == 0:
            return False
        if total_records < self.compact_min_records:
            return False
        if total_records <= latest_count * self.compact_ratio:
            return False
        return True


class JsonlStoreRun:
    def __init__(
        self,
        *,
        store: JsonlStore,
        name: str,
        lock: asyncio.Lock,
        handle: Any,
    ) -> None:
        self._store = store
        self._name = name
        self._lock = lock
        self._handle = handle
        self._closed = False

    async def append_row_result(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError(
                "Cannot append row result after run is finished"
            )
        async with self._lock:
            self._handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
            )
            self._handle.write("\n")
            self._handle.flush()

    async def finish(self) -> None:
        if self._closed:
            return
        async with self._lock:
            self._handle.close()
            self._closed = True
            await self._store._touch_manifest(name=self._name)


def normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_value(config or {})
    if not isinstance(normalized, dict):
        raise ValueError("config must normalize to a dict")
    canonical_json(normalized)
    return normalized


def _read_latest_records(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    total = 0
    if not path.exists():
        return latest, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != RECORD_TYPE_ROW_RESULT:
            continue
        total += 1
        row_key = canonical_json(record["row_id"])
        latest[row_key] = record
    return latest, total


def _record_to_row_state(record: dict[str, Any]) -> RowState:
    row_id = record["row_id"]
    return RowState(
        row_id=row_id,
        row_key=canonical_json(row_id),
        input_hash=record["input_hash"],
        status=record["status"],
        outputs=record.get("outputs"),
        error=record.get("error"),
        created_at=record["created_at"],
    )


def _rewrite_results(path: Path, records: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
    tmp_path.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _safe_path(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in value
    )
    return safe or "_"
