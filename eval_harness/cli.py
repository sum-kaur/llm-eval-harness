from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows so Unicode symbols (✓ ✗ ⚠) render correctly.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import click
import yaml
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .models import EvaluatorConfig, TestCase, TestSuite
from .providers import make_provider
from .reporter import HtmlReporter, JsonReporter, JUnitReporter, TerminalReporter
from .runner import run_suite
from .storage import Storage

load_dotenv()

_DEFAULT_DB = Path(".eval_results.db")
_err = Console(stderr=True, legacy_windows=False)


# ---------------------------------------------------------------------------
# Suite loader
# ---------------------------------------------------------------------------


def _load_suite(path: Path) -> TestSuite:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    defaults = data.get("defaults", {})
    tests: list[TestCase] = []
    for td in data.get("tests", []):
        merged = {**defaults, **td}
        # Evaluators need special handling — don't let defaults clobber them
        merged["evaluators"] = [
            EvaluatorConfig.model_validate(e) for e in td.get("evaluators", [])
        ]
        # Strip keys not in TestCase (e.g. provider, model from defaults)
        allowed = TestCase.model_fields.keys()
        filtered = {k: v for k, v in merged.items() if k in allowed}
        tests.append(TestCase.model_validate(filtered))

    return TestSuite(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        version=data.get("version", "1.0.0"),
        defaults=defaults,
        tests=tests,
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """LLM Eval Harness — test your LLMs with confidence."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@main.command("run")
@click.argument("suite_file", type=click.Path(exists=True, path_type=Path))
@click.option("--model", "-m", default=None, help="Override model")
@click.option("--provider", "-p", default=None, help="anthropic | openai | openai_compatible")
@click.option("--base-url", default=None, help="Base URL for openai_compatible provider")
@click.option("--tags", "-t", multiple=True, help="Run only tests with these tags")
@click.option("--concurrency", "-j", default=5, show_default=True, help="Parallel test limit")
@click.option(
    "--output", "-o",
    type=click.Choice(["table", "json", "html", "junit"]),
    default="table",
    show_default=True,
)
@click.option("--out-file", type=click.Path(path_type=Path), default=None)
@click.option("--compare-baseline", is_flag=True, help="Compare vs stored baseline")
@click.option(
    "--fail-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Exit 1 if pass rate is below this (0–1)",
)
@click.option("--fail-on-regression", is_flag=True, help="Exit 1 only when regressions found")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB, show_default=True)
@click.option("--dry-run", is_flag=True, help="Validate suite without calling the LLM")
@click.option("--no-save", is_flag=True, help="Do not persist results to database")
def cmd_run(
    suite_file: Path,
    model: Optional[str],
    provider: Optional[str],
    base_url: Optional[str],
    tags: tuple[str, ...],
    concurrency: int,
    output: str,
    out_file: Optional[Path],
    compare_baseline: bool,
    fail_threshold: Optional[float],
    fail_on_regression: bool,
    verbose: bool,
    db: Path,
    dry_run: bool,
    no_save: bool,
) -> None:
    """Run an eval suite against an LLM."""
    suite = _load_suite(suite_file)

    if dry_run:
        _err.print(f"[green]✓[/green] Suite [bold]{suite.name}[/bold] — {len(suite.tests)} tests")
        for t in suite.tests:
            skip_note = "  [dim](skip)[/dim]" if t.skip else ""
            _err.print(f"  [dim]{t.id:<30}[/dim] {t.name}{skip_note}")
        return

    effective_provider = provider or suite.defaults.get("provider", "anthropic")
    effective_model = model or suite.defaults.get("model", "claude-haiku-4-5-20251001")

    env_key = f"{effective_provider.upper()}_API_KEY"
    api_key = os.environ.get(env_key)

    try:
        llm = make_provider(
            effective_provider,
            effective_model,
            api_key=api_key,
            base_url=base_url,
        )
    except ValueError as exc:
        _err.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    storage = Storage(db)

    async def _execute():
        await storage.initialize()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=_err,
            transient=True,
        ) as progress:
            active_tests = [t for t in suite.tests if not t.skip]
            task_id = progress.add_task(
                f"Running [bold]{suite.name}[/bold]", total=len(active_tests)
            )

            def on_result(result):
                from .models import EvalStatus
                if result.status != EvalStatus.SKIP:
                    progress.advance(task_id)

            run_result = await run_suite(
                suite=suite,
                provider=llm,
                concurrency=concurrency,
                tags=list(tags) if tags else None,
                on_result=on_result,
            )

        if not no_save:
            await storage.save_run(run_result)

        comparisons = None
        if compare_baseline:
            bl_id = await storage.get_baseline_run_id(suite.name)
            if bl_id:
                comparisons = await storage.compare_runs(bl_id, run_result.run_id)
            else:
                _err.print(
                    "[yellow]Warning:[/yellow] No baseline found for this suite. "
                    "Run [bold]llm-eval baseline set <run_id>[/bold] to create one."
                )

        return run_result, comparisons

    run_result, comparisons = asyncio.run(_execute())

    # ---- output ----
    if output == "table":
        TerminalReporter(verbose=verbose).report(run_result, comparisons)
        _err.print(f"[dim]Run ID: {run_result.run_id}[/dim]")
    elif output == "json":
        out = JsonReporter().report(run_result, comparisons)
        if out_file:
            out_file.write_text(out, encoding="utf-8")
            _err.print(f"[green]JSON written to {out_file}[/green]")
        else:
            print(out)
    elif output == "html":
        out = HtmlReporter().report(run_result, comparisons)
        if out_file:
            out_file.write_text(out, encoding="utf-8")
            _err.print(f"[green]HTML report: {out_file}[/green]")
        else:
            print(out)
    elif output == "junit":
        out = JUnitReporter().report(run_result)
        if out_file:
            out_file.write_text(out, encoding="utf-8")
            _err.print(f"[green]JUnit XML: {out_file}[/green]")
        else:
            print(out)

    # ---- exit code ----
    regressions = sum(1 for c in (comparisons or []) if c.regressed)

    if fail_on_regression:
        sys.exit(1 if regressions > 0 else 0)
    if fail_threshold is not None:
        sys.exit(1 if run_result.pass_rate < fail_threshold else 0)
    # Default: fail if any test failed or errored
    sys.exit(1 if (run_result.failed + run_result.errors) > 0 else 0)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


@main.command("baseline")
@click.argument("run_id")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
def cmd_baseline(run_id: str, db: Path) -> None:
    """Mark a run as the regression baseline."""

    async def _set():
        s = Storage(db)
        await s.initialize()
        await s.set_baseline(run_id)

    asyncio.run(_set())
    _err.print(f"[green]✓[/green] Baseline set to [bold]{run_id}[/bold]")


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


@main.command("history")
@click.option("--suite", "-s", default=None, help="Filter by suite name")
@click.option("--limit", "-n", default=20, show_default=True)
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
def cmd_history(suite: Optional[str], limit: int, db: Path) -> None:
    """List recent eval runs."""

    async def _list():
        s = Storage(db)
        await s.initialize()
        return await s.list_runs(suite_name=suite, limit=limit)

    runs = asyncio.run(_list())
    if not runs:
        _err.print("[dim]No runs found.[/dim]")
        return

    table = Table(box=box.ROUNDED, header_style="bold")
    table.add_column("Run ID", style="dim", no_wrap=True, max_width=12)
    table.add_column("Suite")
    table.add_column("Model")
    table.add_column("Pass Rate", justify="right")
    table.add_column("P/F/E/S", justify="right")
    table.add_column("Timestamp")
    table.add_column("BL", width=3, justify="center")

    for r in runs:
        pr = r["pass_rate"]
        pr_s = "green" if pr >= 0.9 else ("yellow" if pr >= 0.7 else "red")
        table.add_row(
            r["run_id"][:8] + "…",
            r["suite_name"],
            r["model"],
            f"[{pr_s}]{pr:.0%}[/{pr_s}]",
            f"[green]{r['passed']}[/green]/[red]{r['failed']}[/red]"
            f"/[yellow]{r['errors']}[/yellow]/[dim]{r['skipped']}[/dim]",
            r["timestamp"][:16].replace("T", " "),
            "[yellow]★[/yellow]" if r["is_baseline"] else "",
        )

    Console().print(table)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@main.command("compare")
@click.argument("run_id_a")
@click.argument("run_id_b")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
def cmd_compare(run_id_a: str, run_id_b: str, db: Path) -> None:
    """Compare two runs (A = baseline, B = current)."""

    async def _cmp():
        s = Storage(db)
        await s.initialize()
        return await s.compare_runs(run_id_a, run_id_b)

    comparisons = asyncio.run(_cmp())
    if not comparisons:
        _err.print("[red]Could not compare (runs not found?).[/red]")
        sys.exit(1)

    table = Table(
        box=box.ROUNDED,
        header_style="bold",
        title=f"[dim]{run_id_a[:8]}[/dim] → [dim]{run_id_b[:8]}[/dim]",
    )
    table.add_column("Test ID", style="dim")
    table.add_column("Name")
    table.add_column("Baseline")
    table.add_column("Current")
    table.add_column("Change", justify="center")

    regressions = 0
    improvements = 0
    for c in comparisons:
        delta = ""
        if c.regressed:
            delta = "[red]↓ REGRESSED[/red]"
            regressions += 1
        elif c.improved:
            delta = "[green]↑ IMPROVED[/green]"
            improvements += 1

        b_s = "green" if c.baseline_status.value == "pass" else "red"
        c_s = "green" if c.current_status.value == "pass" else "red"
        table.add_row(
            c.test_id,
            c.test_name,
            f"[{b_s}]{c.baseline_status.value}[/{b_s}]",
            f"[c_s]{c.current_status.value}[/c_s]".replace("c_s", c_s),
            delta,
        )

    Console().print(table)
    Console().print(
        f"\n  [red]{regressions} regression(s)[/red]  "
        f"[green]{improvements} improvement(s)[/green]"
    )
    if regressions > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@main.command("show")
@click.argument("run_id")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--verbose", "-v", is_flag=True)
def cmd_show(run_id: str, db: Path, verbose: bool) -> None:
    """Show full details of a stored run."""

    async def _get():
        s = Storage(db)
        await s.initialize()
        return await s.get_run(run_id)

    run = asyncio.run(_get())
    if not run:
        _err.print(f"[red]Run {run_id!r} not found.[/red]")
        sys.exit(1)

    TerminalReporter(verbose=verbose).report(run)
