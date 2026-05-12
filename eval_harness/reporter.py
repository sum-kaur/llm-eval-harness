from __future__ import annotations

import json
import sys
from io import StringIO
from typing import Optional, TextIO
from xml.etree import ElementTree as ET

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .models import BaselineComparison, EvalStatus, RunResult

_STYLE = {
    EvalStatus.PASS: "green",
    EvalStatus.FAIL: "red",
    EvalStatus.ERROR: "yellow",
    EvalStatus.SKIP: "dim",
}
_ICON = {
    EvalStatus.PASS: "✓",
    EvalStatus.FAIL: "✗",
    EvalStatus.ERROR: "⚠",
    EvalStatus.SKIP: "–",
}


class TerminalReporter:
    def __init__(self, file: TextIO = sys.stdout, verbose: bool = False) -> None:
        self.console = Console(file=file, highlight=False, legacy_windows=False)
        self.verbose = verbose

    def report(
        self,
        run: RunResult,
        comparisons: Optional[list[BaselineComparison]] = None,
    ) -> None:
        c = self.console
        c.rule(f"[bold]{run.suite_name}[/bold]")
        c.print(f"  Model   [cyan]{run.model}[/cyan] via [cyan]{run.provider}[/cyan]")
        c.print(f"  Run ID  [dim]{run.run_id}[/dim]")
        c.print(f"  Time    {run.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        c.print()

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        table.add_column("", width=3, justify="center")
        table.add_column("Test ID", style="dim", no_wrap=True)
        table.add_column("Name")
        table.add_column("Latency", justify="right", width=9)
        if comparisons:
            table.add_column("Δ", width=4, justify="center")
        if self.verbose:
            table.add_column("Evaluators")

        comp_map = {c_.test_id: c_ for c_ in (comparisons or [])}
        for r in run.results:
            icon = Text(_ICON[r.status], style=_STYLE[r.status])
            lat = f"{r.latency_ms:.0f}ms" if r.status != EvalStatus.SKIP else "–"
            row: list = [icon, r.test_id, r.test_name, lat]

            if comparisons:
                comp = comp_map.get(r.test_id)
                if comp and comp.regressed:
                    row.append(Text("↓", style="bold red"))
                elif comp and comp.improved:
                    row.append(Text("↑", style="bold green"))
                else:
                    row.append(Text(""))

            if self.verbose:
                parts = [
                    f"[{_STYLE[er.status]}]{_ICON[er.status]}[/{_STYLE[er.status]}] {er.type.value}"
                    for er in r.evaluator_results
                ]
                row.append(" · ".join(parts) if parts else "–")

            table.add_row(*row)

        c.print(table)
        c.print()

        pass_rate = run.pass_rate
        pr_style = "green" if pass_rate >= 0.9 else ("yellow" if pass_rate >= 0.7 else "red")
        c.print(
            f"  [bold]Results:[/bold]  "
            f"[green]{run.passed} passed[/green]  "
            f"[red]{run.failed} failed[/red]  "
            f"[yellow]{run.errors} errors[/yellow]  "
            f"[dim]{run.skipped} skipped[/dim]  "
            f"[bold {pr_style}]({pass_rate:.0%})[/bold {pr_style}]"
        )

        if comparisons:
            regressions = sum(1 for x in comparisons if x.regressed)
            improvements = sum(1 for x in comparisons if x.improved)
            c.print(
                f"  [bold]Baseline:[/bold] "
                f"[red]{regressions} regressed[/red]  "
                f"[green]{improvements} improved[/green]"
            )

        active = run.total_tests - run.skipped
        avg_lat = run.total_latency_ms / max(active, 1)
        total_tok = sum(r.tokens_used or 0 for r in run.results)
        c.print(f"  [bold]Perf:[/bold]    avg {avg_lat:.0f}ms  {total_tok:,} tokens total")
        c.print()


class JsonReporter:
    def report(
        self,
        run: RunResult,
        comparisons: Optional[list[BaselineComparison]] = None,
    ) -> str:
        data = run.model_dump(mode="json")
        if comparisons:
            data["comparisons"] = [x.model_dump(mode="json") for x in comparisons]
        return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# HTML reporter — self-contained, no external assets
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eval: {suite_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;padding:2rem;line-height:1.5}}
h1{{font-size:1.4rem;font-weight:700;margin-bottom:.2rem}}
.meta{{color:#7d8590;font-size:.85rem;margin-bottom:1.75rem}}
.meta code{{background:#161b22;padding:.1rem .4rem;border-radius:4px;font-size:.8rem}}
.cards{{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:2rem}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.9rem 1.4rem;min-width:100px}}
.cv{{font-size:1.75rem;font-weight:700;line-height:1}}
.cl{{font-size:.7rem;color:#7d8590;text-transform:uppercase;letter-spacing:.06em;margin-top:.25rem}}
.g{{color:#3fb950}}.r{{color:#f85149}}.y{{color:#d29922}}.d{{color:#7d8590}}.b{{color:#58a6ff}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{text-align:left;padding:.6rem 1rem;background:#161b22;color:#7d8590;border-bottom:1px solid #30363d;
    font-weight:600;text-transform:uppercase;letter-spacing:.05em;font-size:.72rem}}
td{{padding:.65rem 1rem;border-bottom:1px solid #161b22;vertical-align:top}}
tr:hover td{{background:#161b22}}
.badge{{display:inline-block;padding:.1rem .5rem;border-radius:9999px;font-size:.72rem;font-weight:700}}
.bp{{background:#0d2818;color:#3fb950}}.bf{{background:#3d0f0f;color:#f85149}}
.be{{background:#3d2600;color:#d29922}}.bs{{background:#1c2128;color:#7d8590}}
.resp{{max-width:420px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#7d8590;font-size:.8rem}}
.evals{{list-style:none;font-size:.75rem;color:#7d8590}}
.evals li{{white-space:nowrap}}
</style>
</head>
<body>
<h1>LLM Eval &mdash; {suite_name}</h1>
<div class="meta">
  Model: <strong>{model}</strong> &nbsp;&middot;&nbsp;
  Provider: <strong>{provider}</strong> &nbsp;&middot;&nbsp;
  Run: <code>{run_id}</code> &nbsp;&middot;&nbsp;
  {timestamp}
</div>
<div class="cards">
  <div class="card"><div class="cv {pr_cls}">{pass_pct}%</div><div class="cl">Pass Rate</div></div>
  <div class="card"><div class="cv g">{passed}</div><div class="cl">Passed</div></div>
  <div class="card"><div class="cv r">{failed}</div><div class="cl">Failed</div></div>
  <div class="card"><div class="cv y">{errors}</div><div class="cl">Errors</div></div>
  <div class="card"><div class="cv d">{skipped}</div><div class="cl">Skipped</div></div>
  <div class="card"><div class="cv b">{avg_lat}ms</div><div class="cl">Avg Latency</div></div>
  <div class="card"><div class="cv" style="color:#bc8cff">{total_tok}</div><div class="cl">Tokens</div></div>
</div>
<table>
<thead><tr>
  <th>Status</th><th>Test ID</th><th>Name</th>
  <th>Latency</th><th>Tokens</th><th>Response</th><th>Evaluators</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
</body>
</html>
"""

_ROW = (
    "<tr>"
    "<td><span class='badge b{s}'>{su}</span></td>"
    "<td style='font-family:monospace;font-size:.75rem'>{tid}</td>"
    "<td>{tname}</td>"
    "<td style='white-space:nowrap;font-size:.8rem'>{lat}</td>"
    "<td style='font-size:.8rem'>{tok}</td>"
    "<td><div class='resp' title='{resp_full}'>{resp_short}</div></td>"
    "<td><ul class='evals'>{evals}</ul></td>"
    "</tr>"
)


class HtmlReporter:
    def report(
        self,
        run: RunResult,
        comparisons: Optional[list[BaselineComparison]] = None,
    ) -> str:
        active = run.total_tests - run.skipped
        avg_lat = run.total_latency_ms / max(active, 1)
        total_tok = sum(r.tokens_used or 0 for r in run.results)
        pct = run.pass_rate * 100
        pr_cls = "g" if pct >= 90 else ("y" if pct >= 70 else "r")

        rows = []
        for r in run.results:
            evals_html = "".join(
                f"<li><span class='{_STYLE[er.status]}'>"
                f"{'✓' if er.status == EvalStatus.PASS else '✗'} {er.type.value}:</span>"
                f" {er.message[:80]}</li>"
                for er in r.evaluator_results
            ) or "<li>–</li>"
            raw = r.response or r.error or ""
            safe_full = raw.replace('"', "&quot;").replace("<", "&lt;")
            safe_short = (raw[:80] + "…" if len(raw) > 80 else raw).replace("<", "&lt;")
            rows.append(
                _ROW.format(
                    s=r.status.value[0],
                    su=r.status.value.upper(),
                    tid=r.test_id,
                    tname=r.test_name,
                    lat=f"{r.latency_ms:.0f}ms" if r.status != EvalStatus.SKIP else "–",
                    tok=r.tokens_used if r.tokens_used is not None else "–",
                    resp_full=safe_full,
                    resp_short=safe_short,
                    evals=evals_html,
                )
            )

        return _HTML.format(
            suite_name=run.suite_name,
            model=run.model,
            provider=run.provider,
            run_id=run.run_id,
            timestamp=run.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            pass_pct=f"{pct:.0f}",
            pr_cls=pr_cls,
            passed=run.passed,
            failed=run.failed,
            errors=run.errors,
            skipped=run.skipped,
            avg_lat=f"{avg_lat:.0f}",
            total_tok=f"{total_tok:,}",
            rows="\n".join(rows),
        )


class JUnitReporter:
    def report(self, run: RunResult) -> str:
        suite_el = ET.Element("testsuite")
        suite_el.set("name", run.suite_name)
        suite_el.set("tests", str(run.total_tests))
        suite_el.set("failures", str(run.failed))
        suite_el.set("errors", str(run.errors))
        suite_el.set("skipped", str(run.skipped))
        suite_el.set("timestamp", run.timestamp.isoformat())
        suite_el.set("time", f"{run.total_latency_ms / 1000:.3f}")

        for r in run.results:
            case = ET.SubElement(suite_el, "testcase")
            case.set("name", r.test_name)
            case.set("classname", run.suite_name)
            case.set("time", f"{r.latency_ms / 1000:.3f}")

            if r.status == EvalStatus.SKIP:
                ET.SubElement(case, "skipped")
            elif r.status == EvalStatus.ERROR:
                err_el = ET.SubElement(case, "error")
                err_el.set("message", r.error or "Unknown error")
                err_el.text = r.error
            elif r.status == EvalStatus.FAIL:
                msgs = [er.message for er in r.evaluator_results if er.status == EvalStatus.FAIL]
                fail_el = ET.SubElement(case, "failure")
                fail_el.set("message", "; ".join(msgs) or "Test failed")
                fail_el.text = (
                    f"Response: {r.response or ''}\n\nFailed evaluators:\n" + "\n".join(msgs)
                )

        ET.indent(suite_el, space="  ")
        buf = StringIO()
        buf.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        buf.write(ET.tostring(suite_el, encoding="unicode"))
        return buf.getvalue()
