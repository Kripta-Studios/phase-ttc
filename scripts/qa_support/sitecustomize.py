"""Opt-in instrumentation inherited by Python children of the Stage70 QA runner."""

import os
import sys

if os.environ.get("STAGE70_QA_OPTIMIZER_DB"):
    try:
        if sys.version.split()[0] != os.environ.get("STAGE70_QA_PYTHON_VERSION"):
            raise RuntimeError("QA child Python version differs from the registered interpreter")
        from stage70_optimizer_audit import install

        install()
    except BaseException as error:
        # Python otherwise prints a sitecustomize exception and continues without
        # hooks, which would allow unaccounted optimizer work.
        sys.stderr.write(f"Fatal Stage70 QA accounting initialization: {error!r}\n")
        sys.stderr.flush()
        os._exit(86)
