"""Matched-information comparators for the GarlTTC test claim (SOTA_20261118 follow-up).

A: geometry-only head (supplied boxes only) packaged for Test12.
B: Garl-TTC predictions plus the same causal box-expansion geometry, stacked in the same head
family, loss and grouped CV as ``RGB_GEO_BAND``.

Numpy-only modules (``features``, ``projection``, ``sources``) carry no torch dependency so
their unit tests run in the lightweight repository environment.
"""
