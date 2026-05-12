from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .providers import LLMProvider

from .models import EvalStatus, EvaluatorConfig, EvaluatorResult, EvaluatorType


class BaseEvaluator(ABC):
    @abstractmethod
    async def evaluate(
        self,
        response: str,
        config: EvaluatorConfig,
        provider: Optional[LLMProvider] = None,
    ) -> EvaluatorResult: ...


class ExactEvaluator(BaseEvaluator):
    async def evaluate(self, response, config, provider=None):
        expected = config.value or ""
        a = response.strip() if not config.case_sensitive else response
        b = expected.strip() if not config.case_sensitive else expected
        if not config.case_sensitive:
            a, b = a.lower(), b.lower()
        passed = a == b
        return EvaluatorResult(
            type=EvaluatorType.EXACT,
            status=EvalStatus.PASS if passed else EvalStatus.FAIL,
            score=1.0 if passed else 0.0,
            message="Exact match" if passed else f"Expected exactly {expected!r}, got {response.strip()!r}",
        )


class ContainsEvaluator(BaseEvaluator):
    async def evaluate(self, response, config, provider=None):
        needle = config.value or ""
        hay = response if config.case_sensitive else response.lower()
        pin = needle if config.case_sensitive else needle.lower()
        passed = pin in hay
        return EvaluatorResult(
            type=EvaluatorType.CONTAINS,
            status=EvalStatus.PASS if passed else EvalStatus.FAIL,
            score=1.0 if passed else 0.0,
            message=f"Contains {needle!r}" if passed else f"Expected to contain {needle!r}",
        )


class NotContainsEvaluator(BaseEvaluator):
    async def evaluate(self, response, config, provider=None):
        needle = config.value or ""
        hay = response if config.case_sensitive else response.lower()
        pin = needle if config.case_sensitive else needle.lower()
        passed = pin not in hay
        return EvaluatorResult(
            type=EvaluatorType.NOT_CONTAINS,
            status=EvalStatus.PASS if passed else EvalStatus.FAIL,
            score=1.0 if passed else 0.0,
            message=f"Does not contain {needle!r}" if passed else f"Must not contain {needle!r}",
        )


class RegexEvaluator(BaseEvaluator):
    async def evaluate(self, response, config, provider=None):
        pattern = config.pattern or ""
        flags = 0 if config.case_sensitive else re.IGNORECASE
        try:
            match = re.search(pattern, response, flags)
            passed = match is not None
            return EvaluatorResult(
                type=EvaluatorType.REGEX,
                status=EvalStatus.PASS if passed else EvalStatus.FAIL,
                score=1.0 if passed else 0.0,
                message=f"Pattern matched: {match.group(0)!r}" if passed else f"Pattern {pattern!r} not found",
                details={"match": match.group(0) if match else None},
            )
        except re.error as exc:
            return EvaluatorResult(
                type=EvaluatorType.REGEX,
                status=EvalStatus.ERROR,
                score=0.0,
                message=f"Invalid regex: {exc}",
            )


class StartsWithEvaluator(BaseEvaluator):
    async def evaluate(self, response, config, provider=None):
        prefix = config.value or ""
        text = response.lstrip()
        a = text if config.case_sensitive else text.lower()
        b = prefix if config.case_sensitive else prefix.lower()
        passed = a.startswith(b)
        return EvaluatorResult(
            type=EvaluatorType.STARTS_WITH,
            status=EvalStatus.PASS if passed else EvalStatus.FAIL,
            score=1.0 if passed else 0.0,
            message=f"Starts with {prefix!r}" if passed else f"Expected to start with {prefix!r}",
        )


class EndsWithEvaluator(BaseEvaluator):
    async def evaluate(self, response, config, provider=None):
        suffix = config.value or ""
        text = response.rstrip()
        a = text if config.case_sensitive else text.lower()
        b = suffix if config.case_sensitive else suffix.lower()
        passed = a.endswith(b)
        return EvaluatorResult(
            type=EvaluatorType.ENDS_WITH,
            status=EvalStatus.PASS if passed else EvalStatus.FAIL,
            score=1.0 if passed else 0.0,
            message=f"Ends with {suffix!r}" if passed else f"Expected to end with {suffix!r}",
        )


class JsonSchemaEvaluator(BaseEvaluator):
    async def evaluate(self, response, config, provider=None):
        import jsonschema

        text = response.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return EvaluatorResult(
                type=EvaluatorType.JSON_SCHEMA,
                status=EvalStatus.FAIL,
                score=0.0,
                message=f"Response is not valid JSON: {exc}",
                details={"parse_error": str(exc)},
            )

        if config.schema_def is None:
            return EvaluatorResult(
                type=EvaluatorType.JSON_SCHEMA,
                status=EvalStatus.PASS,
                score=1.0,
                message="Valid JSON (no schema provided)",
            )

        try:
            jsonschema.validate(instance=data, schema=config.schema_def)
            return EvaluatorResult(
                type=EvaluatorType.JSON_SCHEMA,
                status=EvalStatus.PASS,
                score=1.0,
                message="Valid JSON matching schema",
            )
        except jsonschema.ValidationError as exc:
            return EvaluatorResult(
                type=EvaluatorType.JSON_SCHEMA,
                status=EvalStatus.FAIL,
                score=0.0,
                message=f"Schema validation failed: {exc.message}",
                details={
                    "path": list(exc.absolute_path),
                    "schema_path": list(exc.absolute_schema_path),
                },
            )


_JUDGE_SYSTEM = "You are an expert evaluator. Respond only with valid JSON."

_JUDGE_PROMPT = """\
Evaluate the following AI response against the criteria below.

## Criteria
{criteria}

## Response
{response}

Return a JSON object with exactly these fields:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}

Where 1.0 = perfectly meets all criteria, 0.0 = completely fails.
"""


class LLMJudgeEvaluator(BaseEvaluator):
    async def evaluate(self, response, config, provider=None):
        if provider is None:
            return EvaluatorResult(
                type=EvaluatorType.LLM_JUDGE,
                status=EvalStatus.ERROR,
                score=0.0,
                message="LLM judge requires a provider",
            )

        criteria = config.criteria or "Is this a high-quality, helpful, accurate response?"
        prompt = _JUDGE_PROMPT.format(criteria=criteria, response=response)

        try:
            raw, _ = await provider.complete(
                prompt=prompt,
                system=_JUDGE_SYSTEM,
                max_tokens=256,
                temperature=0.0,
                timeout=30.0,
            )
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
                text = "\n".join(inner)

            data = json.loads(text)
            score = float(data["score"])
            reasoning = str(data.get("reasoning", ""))
            passed = score >= config.threshold
            return EvaluatorResult(
                type=EvaluatorType.LLM_JUDGE,
                status=EvalStatus.PASS if passed else EvalStatus.FAIL,
                score=score,
                message=reasoning,
                details={"threshold": config.threshold},
            )
        except Exception as exc:
            return EvaluatorResult(
                type=EvaluatorType.LLM_JUDGE,
                status=EvalStatus.ERROR,
                score=0.0,
                message=f"Judge evaluation failed: {exc}",
            )


_REGISTRY: dict[EvaluatorType, type[BaseEvaluator]] = {
    EvaluatorType.EXACT: ExactEvaluator,
    EvaluatorType.CONTAINS: ContainsEvaluator,
    EvaluatorType.NOT_CONTAINS: NotContainsEvaluator,
    EvaluatorType.REGEX: RegexEvaluator,
    EvaluatorType.STARTS_WITH: StartsWithEvaluator,
    EvaluatorType.ENDS_WITH: EndsWithEvaluator,
    EvaluatorType.JSON_SCHEMA: JsonSchemaEvaluator,
    EvaluatorType.LLM_JUDGE: LLMJudgeEvaluator,
}


async def run_evaluators(
    response: str,
    configs: list[EvaluatorConfig],
    provider: Optional[LLMProvider] = None,
) -> list[EvaluatorResult]:
    results = []
    for config in configs:
        cls = _REGISTRY.get(config.type)
        if cls is None:
            results.append(
                EvaluatorResult(
                    type=config.type,
                    status=EvalStatus.ERROR,
                    score=0.0,
                    message=f"Unknown evaluator type: {config.type}",
                )
            )
        else:
            results.append(await cls().evaluate(response, config, provider))
    return results
