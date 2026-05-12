from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvalStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class EvaluatorType(str, Enum):
    EXACT = "exact"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    JSON_SCHEMA = "json_schema"
    LLM_JUDGE = "llm_judge"


class EvaluatorConfig(BaseModel):
    type: EvaluatorType
    value: Optional[str] = None
    pattern: Optional[str] = None
    schema_def: Optional[dict[str, Any]] = Field(None, alias="schema")
    criteria: Optional[str] = None
    threshold: float = 0.7
    judge_model: Optional[str] = None
    case_sensitive: bool = False

    model_config = {"populate_by_name": True}


class TestCase(BaseModel):
    id: str
    name: str
    prompt: str
    evaluators: list[EvaluatorConfig]
    tags: list[str] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    messages: Optional[list[dict[str, str]]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: float = 30.0
    skip: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestSuite(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0.0"
    defaults: dict[str, Any] = Field(default_factory=dict)
    tests: list[TestCase]


class EvaluatorResult(BaseModel):
    type: EvaluatorType
    status: EvalStatus
    score: float
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class TestResult(BaseModel):
    test_id: str
    test_name: str
    status: EvalStatus
    response: Optional[str] = None
    evaluator_results: list[EvaluatorResult] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    run_id: str
    suite_name: str
    model: str
    provider: str
    timestamp: datetime
    results: list[TestResult] = Field(default_factory=list)
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total_latency_ms: float = 0.0
    pass_rate: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaselineComparison(BaseModel):
    test_id: str
    test_name: str
    baseline_status: EvalStatus
    current_status: EvalStatus
    regressed: bool   # pass -> fail/error
    improved: bool    # fail/error -> pass
