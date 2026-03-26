"""Render and validate command parameters using runtime CLI schema."""

from __future__ import annotations

from collections.abc import Mapping

from entomokit.cli_schema import get_command_schema


TRUE_VALUES = {"1", "true", "yes", "on", "y"}
FALSE_VALUES = {"0", "false", "no", "off", "n"}


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_VALUES:
            return True
        if lowered in FALSE_VALUES:
            return False
    return None


def _coerce_value(value: object, value_type: str) -> tuple[object | None, str | None]:
    if value_type == "bool":
        if value is None:
            return True, None
        parsed = _coerce_bool(value)
        if parsed is None:
            return None, "expects boolean (true/false)"
        return parsed, None

    if value is None:
        return None, None

    if value_type == "int":
        try:
            return int(value), None
        except (TypeError, ValueError):
            return None, "expects integer"

    if value_type == "float":
        try:
            return float(value), None
        except (TypeError, ValueError):
            return None, "expects float"

    return str(value), None


def _schema_lookup(
    parameters: list[dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    by_name: dict[str, dict[str, object]] = {}
    alias_to_name: dict[str, str] = {}

    for param in parameters:
        name = str(param["name"])
        by_name[name] = param
        alias_to_name[name] = name
        alias_to_name[str(param["dest"])] = name
        for opt in param.get("options") or []:
            alias_to_name[str(opt)] = name

    return by_name, alias_to_name


def validate_parameters(
    command: str, user_inputs: Mapping[str, object]
) -> dict[str, object]:
    """Validate proposed parameters before command execution."""
    schema = get_command_schema(command)
    if schema is None:
        return {
            "passed": False,
            "errors": [f"Unknown command schema: {command}"],
            "final_values": {},
            "value_sources": {},
            "schema_source": "runtime CLI export",
        }

    params = schema["parameters"]
    by_name, alias_to_name = _schema_lookup(params)
    errors: list[str] = []
    unknown_keys: list[str] = []
    normalized_inputs: dict[str, object] = {}

    for raw_key, raw_value in user_inputs.items():
        key = str(raw_key)
        canonical = alias_to_name.get(key)
        if canonical is None:
            unknown_keys.append(key)
            continue
        normalized_inputs[canonical] = raw_value

    for key in unknown_keys:
        errors.append(f"Unknown parameter: {key}")

    final_values: dict[str, object] = {}
    value_sources: dict[str, str] = {}

    for name, param in by_name.items():
        required = bool(param.get("required", False))
        value_type = str(param.get("value_type", "str"))
        default = param.get("default")

        if name in normalized_inputs:
            parsed, err = _coerce_value(normalized_inputs[name], value_type)
            if err:
                errors.append(f"{name} {err}")
                continue
            value = parsed
            source = "user"
        else:
            value = default
            source = "default"

        if required and (value is None or value == ""):
            errors.append(f"Missing required parameter: {name}")

        choices = param.get("choices")
        if choices and value is not None and str(value) not in choices:
            allowed = " | ".join(str(c) for c in choices)
            errors.append(f"{name} must be one of: {allowed}")

        final_values[name] = value
        value_sources[name] = source

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "final_values": final_values,
        "value_sources": value_sources,
        "schema_source": "runtime CLI export",
    }


def render_parameter_card(command: str, user_inputs: Mapping[str, object]) -> str:
    """Render a full parameter card for user confirmation."""
    schema = get_command_schema(command)
    if schema is None:
        return (
            f"步骤：{command}\n"
            "参数卡：\n"
            "- 无法加载该命令的参数 schema\n"
            "参数来源：runtime CLI export\n"
            "校验状态：blocked"
        )

    validation = validate_parameters(command, user_inputs)
    lines: list[str] = [f"步骤：{command}", "参数卡："]

    for param in schema["parameters"]:
        name = str(param["name"])
        meaning = str(param.get("help") or "(no help text)")
        options = str(param.get("value_hint") or "<value>")
        value = validation["final_values"].get(name)
        source = validation["value_sources"].get(name, "default")
        lines.extend(
            [
                f"- {name}",
                f"  - 含义：{meaning}",
                f"  - 可选：{options}",
                f"  - 当前：{value!r} ({source})",
            ]
        )

    lines.append(f"参数来源：{validation['schema_source']}")
    if validation["passed"]:
        lines.append("校验状态：passed")
    else:
        lines.append("校验状态：blocked")
        lines.append("校验错误：")
        for err in validation["errors"]:
            lines.append(f"- {err}")

    return "\n".join(lines)
