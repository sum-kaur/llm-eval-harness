import pytest

from eval_harness.models import EvaluatorConfig, EvaluatorType, TestCase


@pytest.fixture
def make_config():
    def _make(type_: EvaluatorType, **kwargs) -> EvaluatorConfig:
        return EvaluatorConfig(type=type_, **kwargs)
    return _make


@pytest.fixture
def make_test():
    def _make(prompt: str, evaluators, **kwargs) -> TestCase:
        return TestCase(
            id="test_fixture",
            name="Fixture test",
            prompt=prompt,
            evaluators=evaluators,
            **kwargs,
        )
    return _make
