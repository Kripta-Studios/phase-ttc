"""Zero-shot reproduction of published optical-expansion / motion-in-depth baselines on eAP.

Only our own adapter and evaluation code lives here; third-party source and weights are read
from ``PHASE_TTC_BASELINES_ROOT`` at run time and are never copied into the repository.
``common`` and ``frames`` are numpy/Pillow-only so they can run inside every baseline venv;
``adapters``, ``run`` and ``sanity`` need torch; ``manifest``, ``evaluate`` and ``report``
run in the repository environment.  The protocol is
``docs/sota_20261118/literature_baselines/PROTOCOL.json``.
"""
