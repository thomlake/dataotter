from collections.abc import Mapping

from dataotter.errors import InvalidBindingError

BindingsInput = list[str] | dict[str, str]


def normalize_inputs(inputs: BindingsInput) -> dict[str, str]:
    return _normalize_bindings(inputs, label="inputs")


def normalize_outputs(outputs: BindingsInput) -> dict[str, str]:
    return _normalize_bindings(outputs, label="outputs")


def _normalize_bindings(bindings: BindingsInput, *, label: str) -> dict[str, str]:
    if isinstance(bindings, list):
        normalized = {item: item for item in bindings}
    elif isinstance(bindings, Mapping):
        normalized = dict(bindings)
    else:
        raise InvalidBindingError(f"{label} must be a list[str] or dict[str, str]")

    if not normalized:
        raise InvalidBindingError(f"{label} may not be empty")

    for source, target in normalized.items():
        if not isinstance(source, str) or not source:
            raise InvalidBindingError(f"{label} keys must be non-empty strings")
        if not isinstance(target, str) or not target:
            raise InvalidBindingError(f"{label} values must be non-empty strings")

    targets = list(normalized.values())
    if len(set(targets)) != len(targets):
        raise InvalidBindingError(f"{label} may not map multiple keys to the same value")

    return normalized
