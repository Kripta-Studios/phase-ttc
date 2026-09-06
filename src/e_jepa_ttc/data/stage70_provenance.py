"""Check ALL ancestors' fit/selection/normalization sets, not only direct parent fits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def assert_excludes(
    nodes: Mapping[str, Mapping[str, Any]], root: str, held_out: set[str]
) -> set[str]:
    visited: set[str] = set()
    active: set[str] = set()

    def visit(name: str) -> None:
        if name in active:
            raise ValueError("cycle in producer ancestry")
        if name in visited:
            return
        if name not in nodes:
            raise ValueError(f"missing ancestor {name}")
        active.add(name)
        node = nodes[name]
        for field in ("fit_sequences", "selection_sequences", "normalization_sequences"):
            if field not in node:
                raise ValueError(f"unrecorded provenance field {name}:{field}")
            if held_out.intersection(node[field]):
                raise ValueError(f"held-out sequence exposed through {name}:{field}")
        for parent in node.get("parents", []):
            visit(parent)
        active.remove(name)
        visited.add(name)

    visit(root)
    return visited
