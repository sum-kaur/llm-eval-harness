"""Unit tests for evaluators — no LLM calls required."""
import json

import pytest

from eval_harness.evaluators import run_evaluators
from eval_harness.models import EvalStatus, EvaluatorConfig, EvaluatorType


def cfg(type_: EvaluatorType, **kw) -> EvaluatorConfig:
    return EvaluatorConfig(type=type_, **kw)


# ---------------------------------------------------------------------------
# ExactEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exact_pass():
    results = await run_evaluators("42", [cfg(EvaluatorType.EXACT, value="42")])
    assert results[0].status == EvalStatus.PASS
    assert results[0].score == 1.0


@pytest.mark.asyncio
async def test_exact_strips_whitespace():
    results = await run_evaluators("  42  ", [cfg(EvaluatorType.EXACT, value="42")])
    assert results[0].status == EvalStatus.PASS


@pytest.mark.asyncio
async def test_exact_case_insensitive():
    results = await run_evaluators("POSITIVE", [cfg(EvaluatorType.EXACT, value="positive")])
    assert results[0].status == EvalStatus.PASS


@pytest.mark.asyncio
async def test_exact_case_sensitive_fail():
    results = await run_evaluators(
        "POSITIVE", [cfg(EvaluatorType.EXACT, value="positive", case_sensitive=True)]
    )
    assert results[0].status == EvalStatus.FAIL


@pytest.mark.asyncio
async def test_exact_fail():
    results = await run_evaluators("43", [cfg(EvaluatorType.EXACT, value="42")])
    assert results[0].status == EvalStatus.FAIL
    assert results[0].score == 0.0


# ---------------------------------------------------------------------------
# ContainsEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contains_pass():
    results = await run_evaluators(
        "The capital of France is Paris.",
        [cfg(EvaluatorType.CONTAINS, value="Paris")],
    )
    assert results[0].status == EvalStatus.PASS


@pytest.mark.asyncio
async def test_contains_case_insensitive():
    results = await run_evaluators(
        "paris is great",
        [cfg(EvaluatorType.CONTAINS, value="Paris", case_sensitive=False)],
    )
    assert results[0].status == EvalStatus.PASS


@pytest.mark.asyncio
async def test_contains_fail():
    results = await run_evaluators(
        "London is the capital.",
        [cfg(EvaluatorType.CONTAINS, value="Paris")],
    )
    assert results[0].status == EvalStatus.FAIL


# ---------------------------------------------------------------------------
# NotContainsEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_not_contains_pass():
    results = await run_evaluators(
        "Hello, world!",
        [cfg(EvaluatorType.NOT_CONTAINS, value="harm")],
    )
    assert results[0].status == EvalStatus.PASS


@pytest.mark.asyncio
async def test_not_contains_fail():
    results = await run_evaluators(
        "I will harm you.",
        [cfg(EvaluatorType.NOT_CONTAINS, value="harm")],
    )
    assert results[0].status == EvalStatus.FAIL


# ---------------------------------------------------------------------------
# RegexEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regex_pass():
    results = await run_evaluators(
        "42",
        [cfg(EvaluatorType.REGEX, pattern=r"^\d+$")],
    )
    assert results[0].status == EvalStatus.PASS


@pytest.mark.asyncio
async def test_regex_fail():
    results = await run_evaluators(
        "forty-two",
        [cfg(EvaluatorType.REGEX, pattern=r"^\d+$")],
    )
    assert results[0].status == EvalStatus.FAIL


@pytest.mark.asyncio
async def test_regex_invalid_pattern():
    results = await run_evaluators(
        "anything",
        [cfg(EvaluatorType.REGEX, pattern=r"[invalid")],
    )
    assert results[0].status == EvalStatus.ERROR


@pytest.mark.asyncio
async def test_regex_case_insensitive():
    results = await run_evaluators(
        "HELLO WORLD",
        [cfg(EvaluatorType.REGEX, pattern=r"hello", case_sensitive=False)],
    )
    assert results[0].status == EvalStatus.PASS


# ---------------------------------------------------------------------------
# StartsWithEvaluator / EndsWithEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_starts_with_pass():
    results = await run_evaluators(
        "  Hello there",
        [cfg(EvaluatorType.STARTS_WITH, value="Hello")],
    )
    assert results[0].status == EvalStatus.PASS


@pytest.mark.asyncio
async def test_ends_with_pass():
    results = await run_evaluators(
        "Have a great day!  ",
        [cfg(EvaluatorType.ENDS_WITH, value="day!")],
    )
    assert results[0].status == EvalStatus.PASS


# ---------------------------------------------------------------------------
# JsonSchemaEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_schema_valid():
    payload = json.dumps({"name": "Alice", "age": 30, "city": "London"})
    schema = {
        "type": "object",
        "required": ["name", "age", "city"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        },
    }
    results = await run_evaluators(
        payload,
        [EvaluatorConfig(type=EvaluatorType.JSON_SCHEMA, **{"schema": schema})],
    )
    assert results[0].status == EvalStatus.PASS


@pytest.mark.asyncio
async def test_json_schema_missing_field():
    payload = json.dumps({"name": "Alice", "age": 30})  # city missing
    schema = {
        "type": "object",
        "required": ["name", "age", "city"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        },
    }
    results = await run_evaluators(
        payload,
        [EvaluatorConfig(type=EvaluatorType.JSON_SCHEMA, **{"schema": schema})],
    )
    assert results[0].status == EvalStatus.FAIL


@pytest.mark.asyncio
async def test_json_schema_invalid_json():
    results = await run_evaluators(
        "not json at all",
        [EvaluatorConfig(type=EvaluatorType.JSON_SCHEMA, **{"schema": {"type": "object"}})],
    )
    assert results[0].status == EvalStatus.FAIL


@pytest.mark.asyncio
async def test_json_schema_strips_code_fence():
    payload = '```json\n{"key": "value"}\n```'
    results = await run_evaluators(
        payload,
        [EvaluatorConfig(type=EvaluatorType.JSON_SCHEMA, **{"schema": {"type": "object"}})],
    )
    assert results[0].status == EvalStatus.PASS


# ---------------------------------------------------------------------------
# Multiple evaluators — all must pass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_evaluators_all_pass():
    results = await run_evaluators(
        "Paris is the capital of France.",
        [
            cfg(EvaluatorType.CONTAINS, value="Paris"),
            cfg(EvaluatorType.CONTAINS, value="France"),
            cfg(EvaluatorType.NOT_CONTAINS, value="London"),
        ],
    )
    assert all(r.status == EvalStatus.PASS for r in results)


@pytest.mark.asyncio
async def test_multiple_evaluators_one_fails():
    results = await run_evaluators(
        "Paris is the capital of France.",
        [
            cfg(EvaluatorType.CONTAINS, value="Paris"),
            cfg(EvaluatorType.CONTAINS, value="London"),  # will fail
        ],
    )
    statuses = {r.status for r in results}
    assert EvalStatus.PASS in statuses
    assert EvalStatus.FAIL in statuses
