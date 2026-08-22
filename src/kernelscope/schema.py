from __future__ import annotations

from typing import Any

from .models import DecisionStep, Generation


class SchemaError(ValueError):
    pass


def _require(obj: dict[str, Any], key: str, kind: type) -> Any:
    if key not in obj:
        raise SchemaError(f"missing required field: {key}")
    value = obj[key]
    if not isinstance(value, kind):
        raise SchemaError(f"field {key} must be {kind.__name__}")
    return value


def parse_generation(payload: dict[str, Any]) -> Generation:
    understanding = _require(payload, "task_understanding", dict)
    raw_steps = _require(payload, "steps", list)
    complexity = _require(payload, "complexity", dict)
    final_kernel = _require(payload, "final_kernel", str)
    launch_config = _require(payload, "launch_config", dict)

    steps: list[DecisionStep] = []
    seen: set[str] = set()
    for raw in raw_steps:
        if not isinstance(raw, dict):
            raise SchemaError("each step must be an object")
        step_id = _require(raw, "id", str)
        if step_id in seen:
            raise SchemaError(f"duplicate step id: {step_id}")
        seen.add(step_id)
        steps.append(
            DecisionStep(
                step_id=step_id,
                step_type=_require(raw, "type", str),
                claim=_require(raw, "claim", str),
                depends_on=_list_of_strings(raw, "depends_on"),
                code_symbols=_list_of_strings(raw, "code_symbols"),
                evidence_expected=_list_of_strings(raw, "evidence_expected"),
            )
        )

    ids = {step.step_id for step in steps}
    missing_deps = {
        dep for step in steps for dep in step.depends_on if dep not in ids
    }
    if missing_deps:
        raise SchemaError(f"unknown step dependency: {sorted(missing_deps)}")
    return Generation(understanding, steps, complexity, final_kernel, launch_config)


def _list_of_strings(obj: dict[str, Any], key: str) -> list[str]:
    value = obj.get(key, [])
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise SchemaError(f"field {key} must be a list of strings")
    return value
