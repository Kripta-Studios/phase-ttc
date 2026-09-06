# Bounded baseline and current-source test accounting

The source-pin checkout collected 1,803 literal pytest node IDs without executing
tests. This is inventory evidence, not a passing suite or a legacy failure list.
The complete baseline/current failure comparison remains required before freeze.

`run_stage70_test_audit.py` runs a clean committed checkout on CPU and reserves its
entire optimizer allowance in the campaign update ledger first. An isolated
`sitecustomize` installs public PyTorch optimizer pre/post hooks in the test process
and inherited Python subprocesses. SQLite serializes charged intents, confirmations,
process identities and optimizer instances. Hook initialization failure terminates
that Python process instead of continuing without accounting. No model seed,
loss, optimizer or target is changed by the hooks.

The parent writes progress while pytest runs. It closes output writers and checks
recorded descendant process identities before settling the receipt. Interrupted
intents count as uncertain attempts. Unused allowance is released only on settlement.
A completed test suite does not certify a scientific model endpoint; optimizer
instances observed in tests are reported separately from fixed model endpoints.

The capture plugin records exact node IDs and setup/call/teardown failures, including
collection failures and skips. It imports no project module, so the baseline imports
its own source. A deliberately sanitized subprocess environment can suppress inherited
instrumentation; the known training subprocesses inherit the environment. This is an
audited application contract, not an operating-system security boundary.

Thirteen accounting and composite-job tests pass without actual optimizer work.
The real synthetic CPU training-flow test passed with exactly one charged and
confirmed update, one optimizer instance and no uncertain attempt. The receipt is
`baseline_qa/instrumentation_cpu_acceptance/CLOSED_ACCOUNTING.json`. Inherited child
process coverage will also be inspected in the complete baseline suite. The resource reservation must
include outstanding concurrent cache writes as well as the suite's own allocation.

Scoped Pyright passed all 48 files with the explicit shared Python interpreter.
The initial invocation without `--pythonpath` reported nine pandas typing errors;
it resolved a different environment. No source or type suppression was needed.
Use `--pythonpath ../e-jepa-ttc/.venv/Scripts/python.exe` for this worktree.

The composite architecture executor reserves pretext and supervised budgets before
either phase starts, requires a signed scientific lock and physically bound source
view, and only publishes the prescribed final endpoint. It is not execution-ready:
the full lock compiler, conditional source-bound orchestration and remaining
diagnostic/packaging integration must still be completed before scientific fitting.
