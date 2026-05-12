from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from .models import (
    BaselineComparison,
    EvalStatus,
    EvaluatorResult,
    EvaluatorType,
    RunResult,
    TestResult,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    suite_name  TEXT NOT NULL,
    model       TEXT NOT NULL,
    provider    TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    total_tests INTEGER NOT NULL,
    passed      INTEGER NOT NULL,
    failed      INTEGER NOT NULL,
    errors      INTEGER NOT NULL,
    skipped     INTEGER NOT NULL,
    pass_rate   REAL NOT NULL,
    latency_ms  REAL NOT NULL,
    is_baseline INTEGER NOT NULL DEFAULT 0,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS test_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    test_id     TEXT NOT NULL,
    test_name   TEXT NOT NULL,
    status      TEXT NOT NULL,
    response    TEXT,
    latency_ms  REAL NOT NULL,
    tokens_used INTEGER,
    error       TEXT,
    timestamp   TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS evaluator_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    test_result_id  INTEGER NOT NULL,
    evaluator_type  TEXT NOT NULL,
    status          TEXT NOT NULL,
    score           REAL NOT NULL,
    message         TEXT NOT NULL,
    details         TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (test_result_id) REFERENCES test_results(id)
);

CREATE INDEX IF NOT EXISTS idx_tr_run_id  ON test_results(run_id);
CREATE INDEX IF NOT EXISTS idx_er_tr_id   ON evaluator_results(test_result_id);
"""


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def save_run(self, run: RunResult) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
                (
                    run.run_id,
                    run.suite_name,
                    run.model,
                    run.provider,
                    run.timestamp.isoformat(),
                    run.total_tests,
                    run.passed,
                    run.failed,
                    run.errors,
                    run.skipped,
                    run.pass_rate,
                    run.total_latency_ms,
                    json.dumps(run.metadata),
                ),
            )
            for tr in run.results:
                cursor = await db.execute(
                    """INSERT INTO test_results
                       (run_id, test_id, test_name, status, response, latency_ms,
                        tokens_used, error, timestamp, metadata)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run.run_id,
                        tr.test_id,
                        tr.test_name,
                        tr.status.value,
                        tr.response,
                        tr.latency_ms,
                        tr.tokens_used,
                        tr.error,
                        tr.timestamp.isoformat(),
                        json.dumps(tr.metadata),
                    ),
                )
                tr_id = cursor.lastrowid
                for er in tr.evaluator_results:
                    await db.execute(
                        """INSERT INTO evaluator_results
                           (test_result_id, evaluator_type, status, score, message, details)
                           VALUES (?,?,?,?,?,?)""",
                        (
                            tr_id,
                            er.type.value,
                            er.status.value,
                            er.score,
                            er.message,
                            json.dumps(er.details),
                        ),
                    )
            await db.commit()

    async def set_baseline(self, run_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE runs SET is_baseline = 0")
            await db.execute(
                "UPDATE runs SET is_baseline = 1 WHERE run_id = ?", (run_id,)
            )
            await db.commit()

    async def get_baseline_run_id(self, suite_name: Optional[str] = None) -> Optional[str]:
        async with aiosqlite.connect(self.db_path) as db:
            if suite_name:
                row = await (
                    await db.execute(
                        "SELECT run_id FROM runs WHERE is_baseline=1 AND suite_name=?"
                        " ORDER BY timestamp DESC LIMIT 1",
                        (suite_name,),
                    )
                ).fetchone()
            else:
                row = await (
                    await db.execute(
                        "SELECT run_id FROM runs WHERE is_baseline=1"
                        " ORDER BY timestamp DESC LIMIT 1"
                    )
                ).fetchone()
            return row[0] if row else None

    async def list_runs(
        self, suite_name: Optional[str] = None, limit: int = 20
    ) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if suite_name:
                rows = await (
                    await db.execute(
                        "SELECT * FROM runs WHERE suite_name=? ORDER BY timestamp DESC LIMIT ?",
                        (suite_name, limit),
                    )
                ).fetchall()
            else:
                rows = await (
                    await db.execute(
                        "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
                    )
                ).fetchall()
            return [dict(r) for r in rows]

    async def get_run(self, run_id: str) -> Optional[RunResult]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            run_row = await (
                await db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
            ).fetchone()
            if not run_row:
                return None
            run_dict = dict(run_row)

            tr_rows = await (
                await db.execute(
                    "SELECT * FROM test_results WHERE run_id=? ORDER BY id", (run_id,)
                )
            ).fetchall()

            results = []
            for tr_row in tr_rows:
                tr = dict(tr_row)
                er_rows = await (
                    await db.execute(
                        "SELECT * FROM evaluator_results WHERE test_result_id=? ORDER BY id",
                        (tr["id"],),
                    )
                ).fetchall()
                eval_results = [
                    EvaluatorResult(
                        type=EvaluatorType(r["evaluator_type"]),
                        status=EvalStatus(r["status"]),
                        score=r["score"],
                        message=r["message"],
                        details=json.loads(r["details"]),
                    )
                    for r in er_rows
                ]
                results.append(
                    TestResult(
                        test_id=tr["test_id"],
                        test_name=tr["test_name"],
                        status=EvalStatus(tr["status"]),
                        response=tr["response"],
                        evaluator_results=eval_results,
                        latency_ms=tr["latency_ms"],
                        tokens_used=tr["tokens_used"],
                        error=tr["error"],
                        timestamp=datetime.fromisoformat(tr["timestamp"]),
                        metadata=json.loads(tr["metadata"]),
                    )
                )

            return RunResult(
                run_id=run_dict["run_id"],
                suite_name=run_dict["suite_name"],
                model=run_dict["model"],
                provider=run_dict["provider"],
                timestamp=datetime.fromisoformat(run_dict["timestamp"]),
                results=results,
                total_tests=run_dict["total_tests"],
                passed=run_dict["passed"],
                failed=run_dict["failed"],
                errors=run_dict["errors"],
                skipped=run_dict["skipped"],
                total_latency_ms=run_dict["latency_ms"],
                pass_rate=run_dict["pass_rate"],
                metadata=json.loads(run_dict["metadata"]),
            )

    async def compare_runs(
        self, baseline_id: str, current_id: str
    ) -> list[BaselineComparison]:
        baseline = await self.get_run(baseline_id)
        current = await self.get_run(current_id)
        if not baseline or not current:
            return []

        b_map = {r.test_id: r.status for r in baseline.results}
        comparisons = []
        for r in current.results:
            b_status = b_map.get(r.test_id, EvalStatus.SKIP)
            comparisons.append(
                BaselineComparison(
                    test_id=r.test_id,
                    test_name=r.test_name,
                    baseline_status=b_status,
                    current_status=r.status,
                    regressed=(
                        b_status == EvalStatus.PASS
                        and r.status in (EvalStatus.FAIL, EvalStatus.ERROR)
                    ),
                    improved=(
                        b_status in (EvalStatus.FAIL, EvalStatus.ERROR)
                        and r.status == EvalStatus.PASS
                    ),
                )
            )
        return comparisons
