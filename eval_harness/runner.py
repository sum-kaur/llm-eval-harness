from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Callable, Optional

from .evaluators import run_evaluators
from .models import EvalStatus, RunResult, TestCase, TestResult, TestSuite
from .providers import LLMProvider


async def _run_single(
    test: TestCase,
    provider: LLMProvider,
    judge_provider: Optional[LLMProvider],
    semaphore: asyncio.Semaphore,
    on_result: Optional[Callable[[TestResult], None]],
) -> TestResult:
    async with semaphore:
        if test.skip:
            result = TestResult(
                test_id=test.id,
                test_name=test.name,
                status=EvalStatus.SKIP,
            )
            if on_result:
                on_result(result)
            return result

        start = time.perf_counter()
        response: Optional[str] = None
        tokens: Optional[int] = None
        error: Optional[str] = None

        for attempt in range(3):
            try:
                response, tokens = await provider.complete(
                    prompt=test.prompt,
                    system=test.system_prompt,
                    messages=test.messages,
                    max_tokens=test.max_tokens or 1024,
                    temperature=test.temperature if test.temperature is not None else 0.0,
                    timeout=test.timeout,
                )
                error = None
                break
            except asyncio.TimeoutError:
                error = f"Request timed out after {test.timeout}s"
                break
            except Exception as exc:
                error = str(exc)
                if attempt < 2:
                    await asyncio.sleep(2**attempt)

        latency_ms = (time.perf_counter() - start) * 1000

        if error:
            result = TestResult(
                test_id=test.id,
                test_name=test.name,
                status=EvalStatus.ERROR,
                response=response,
                latency_ms=latency_ms,
                tokens_used=tokens,
                error=error,
            )
        else:
            eval_results = await run_evaluators(
                response=response or "",
                configs=test.evaluators,
                provider=judge_provider or provider,
            )
            any_error = any(r.status == EvalStatus.ERROR for r in eval_results)
            all_passed = all(r.status == EvalStatus.PASS for r in eval_results)
            if any_error:
                status = EvalStatus.ERROR
            elif all_passed:
                status = EvalStatus.PASS
            else:
                status = EvalStatus.FAIL

            result = TestResult(
                test_id=test.id,
                test_name=test.name,
                status=status,
                response=response,
                evaluator_results=eval_results,
                latency_ms=latency_ms,
                tokens_used=tokens,
            )

        if on_result:
            on_result(result)
        return result


async def run_suite(
    suite: TestSuite,
    provider: LLMProvider,
    judge_provider: Optional[LLMProvider] = None,
    concurrency: int = 5,
    tags: Optional[list[str]] = None,
    on_result: Optional[Callable[[TestResult], None]] = None,
) -> RunResult:
    tests = suite.tests
    if tags:
        tests = [t for t in tests if any(tag in t.tags for tag in tags)]

    semaphore = asyncio.Semaphore(concurrency)
    results = list(
        await asyncio.gather(
            *[_run_single(t, provider, judge_provider, semaphore, on_result) for t in tests]
        )
    )

    passed = sum(1 for r in results if r.status == EvalStatus.PASS)
    failed = sum(1 for r in results if r.status == EvalStatus.FAIL)
    errors = sum(1 for r in results if r.status == EvalStatus.ERROR)
    skipped = sum(1 for r in results if r.status == EvalStatus.SKIP)
    active = len(results) - skipped

    return RunResult(
        run_id=str(uuid.uuid4()),
        suite_name=suite.name,
        model=getattr(provider, "model", "unknown"),
        provider=provider.name,
        timestamp=datetime.utcnow(),
        results=results,
        total_tests=len(results),
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        total_latency_ms=sum(r.latency_ms for r in results),
        pass_rate=passed / active if active > 0 else 0.0,
    )
