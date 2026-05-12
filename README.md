# LLM Eval Harness

A production-quality CLI for running prompt-based test suites against LLMs, tracking results over time, and catching regressions in CI/CD.

---

## Features

- **YAML test definitions** — readable, version-controllable test suites
- **8 evaluator types** — exact match, contains, not-contains, regex, starts/ends-with, JSON schema, and LLM-as-judge
- **Async runner** — parallel test execution with configurable concurrency, retries, and per-test timeouts
- **Persistent history** — all runs saved to SQLite; compare any two runs at any time
- **Baseline regression detection** — mark a passing run as the baseline; future CI runs exit non-zero only when tests regress
- **4 output formats** — rich terminal table, JSON, self-contained HTML report, JUnit XML
- **Multi-provider** — Anthropic and OpenAI (including any OpenAI-compatible endpoint)

---

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/YOUR_USERNAME/llm-eval-harness.git
cd llm-eval-harness
pip install -e .
```

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY or OPENAI_API_KEY
```

---

## Quick start

```bash
# Validate a suite without calling the API
llm-eval run examples/basic_suite.yaml --dry-run

# Run against the default model (claude-haiku)
llm-eval run examples/basic_suite.yaml

# Override model and provider
llm-eval run examples/basic_suite.yaml --model claude-sonnet-4-6 --provider anthropic

# Generate an HTML report
llm-eval run examples/basic_suite.yaml --output html --out-file report.html
```

---

## Writing test suites

Tests are defined in YAML. The `defaults` block applies to every test unless overridden.

```yaml
name: "My Suite"
description: "Core capability checks"
version: "1.0.0"

defaults:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  max_tokens: 256
  temperature: 0.0
  timeout: 30.0

tests:

  - id: arithmetic_basic
    name: "Simple addition"
    tags: [math]
    prompt: "What is 15 + 27? Reply with just the number."
    evaluators:
      - type: exact
        value: "42"

  - id: capital_france
    name: "Geography"
    prompt: "What is the capital of France?"
    evaluators:
      - type: contains
        value: "Paris"
        case_sensitive: false

  - id: json_output
    name: "Structured output"
    system_prompt: "Respond with valid JSON only."
    prompt: "Return a book object with title, author, and year."
    evaluators:
      - type: json_schema
        schema:
          type: object
          required: [title, author, year]
          properties:
            title: { type: string }
            author: { type: string }
            year: { type: integer }

  - id: essay_quality
    name: "LLM-as-judge"
    prompt: "In 2–3 sentences, explain why tests matter in software."
    evaluators:
      - type: llm_judge
        criteria: |
          Is this a clear, accurate, concise answer (2–3 sentences)
          that explains why software testing matters?
        threshold: 0.8
```

### All evaluator types

| Type | Required fields | Description |
|---|---|---|
| `exact` | `value` | Trimmed, case-insensitive string equality (set `case_sensitive: true` to override) |
| `contains` | `value` | Response must contain the substring |
| `not_contains` | `value` | Response must not contain the substring |
| `regex` | `pattern` | Python regex must match somewhere in the response |
| `starts_with` | `value` | Response must start with the value (leading whitespace ignored) |
| `ends_with` | `value` | Response must end with the value (trailing whitespace ignored) |
| `json_schema` | `schema` | Response must be valid JSON matching the given JSON Schema |
| `llm_judge` | `criteria`, `threshold` | Ask the model to score 0–1 against free-text criteria; pass if score ≥ threshold |

### Multi-turn conversations

Supply a `messages` list instead of `prompt`:

```yaml
- id: context_retention
  name: "Multi-turn memory"
  messages:
    - role: user
      content: "My name is Alice."
    - role: assistant
      content: "Got it, Alice!"
    - role: user
      content: "What is my name?"
  prompt: ""
  evaluators:
    - type: contains
      value: "Alice"
```

---

## CLI reference

### `llm-eval run`

```
llm-eval run SUITE_FILE [OPTIONS]

Options:
  -m, --model TEXT              Override model
  -p, --provider TEXT           anthropic | openai | openai_compatible
      --base-url TEXT           Base URL for openai_compatible
  -t, --tags TEXT               Run only tests with these tags (repeatable)
  -j, --concurrency INT         Parallel test limit  [default: 5]
  -o, --output [table|json|html|junit]  [default: table]
      --out-file PATH           Write output to file
      --compare-baseline        Compare against stored baseline
      --fail-threshold FLOAT    Exit 1 if pass rate < threshold (0–1)
      --fail-on-regression      Exit 1 only when regressions are found
  -v, --verbose                 Show per-evaluator detail
      --db PATH                 SQLite database path  [default: .eval_results.db]
      --dry-run                 Validate suite without calling the LLM
      --no-save                 Do not persist results
```

### `llm-eval baseline`

```
llm-eval baseline RUN_ID     Mark a run as the regression baseline
```

### `llm-eval history`

```
llm-eval history [--suite NAME] [--limit N]    List recent runs
```

### `llm-eval compare`

```
llm-eval compare RUN_ID_A RUN_ID_B    Diff two runs; exits 1 if regressions found
```

### `llm-eval show`

```
llm-eval show RUN_ID [--verbose]    Full detail of a stored run
```

---

## CI/CD integration

### GitHub Actions example

```yaml
name: LLM Regression Tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install
        run: pip install -e .

      - name: Run eval suite
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          llm-eval run examples/basic_suite.yaml \
            --output junit --out-file results.xml \
            --compare-baseline \
            --fail-on-regression

      - name: Publish test results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: results.xml
```

### Exit codes

| Scenario | Exit code |
|---|---|
| All tests pass | `0` |
| Any test failed / errored (default) | `1` |
| `--fail-on-regression`: no regressions | `0` |
| `--fail-on-regression`: regressions found | `1` |
| `--fail-threshold N`: pass rate ≥ N | `0` |
| `--fail-threshold N`: pass rate < N | `1` |

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite covers all evaluator types and requires no API keys (no LLM calls).

---

## Project structure

```
eval_harness/
├── models.py       Pydantic v2 data models
├── providers.py    Async LLM provider abstraction (Anthropic, OpenAI)
├── evaluators.py   Evaluator implementations
├── runner.py       Async test runner with concurrency + retry
├── storage.py      SQLite persistence via aiosqlite
├── reporter.py     Terminal / JSON / HTML / JUnit reporters
└── cli.py          Click CLI (run, baseline, history, compare, show)
examples/
├── basic_suite.yaml     Quick-start examples
└── advanced_suite.yaml  LLM-as-judge and multi-turn examples
tests/
└── test_evaluators.py   Unit tests (22 tests, no API keys needed)
```

---

## License

MIT
