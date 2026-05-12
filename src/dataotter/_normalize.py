import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

HASH_PREFIX = "sha256:"


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def normalize_value(value: Any) -> Any:
    if value is None:
        return None

    if value is pd.NA:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, datetime | date):
        return value.isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): normalize_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }

    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]

    if isinstance(value, list):
        return [normalize_value(item) for item in value]

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [normalize_value(item) for item in value]

    if hasattr(value, "item"):
        try:
            return normalize_value(value.item())
        except (TypeError, ValueError):
            pass

    if isinstance(value, bool | int | str):
        return value

    if isinstance(value, float):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def canonical_json(value: Any) -> str:
    normalized = normalize_value(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    payload = canonical_json(value).encode("utf-8")
    return f"{HASH_PREFIX}{hashlib.sha256(payload).hexdigest()}"


def short_hash(value: str, *, length: int = 12) -> str:
    if value.startswith(HASH_PREFIX):
        value = value.removeprefix(HASH_PREFIX)
    return value[:length]
