"""Sequence-block fold rules shared by the released rotations and the fusion refit."""

from __future__ import annotations

BLOCKS = 5


def refiner_block(generation_block: int) -> int:
    """Released rotation ``r`` generates ``B_r`` with a refiner trained on ``B_{r+1}``."""
    return (generation_block + 1) % BLOCKS


def producer_blocks(generation_block: int) -> tuple[int, ...]:
    """Released rotation ``r`` trains its producers on ``B_{r+2}, B_{r+3}, B_{r+4}``."""
    return tuple((generation_block + offset) % BLOCKS for offset in (2, 3, 4))


def fusion_train_blocks(test_block: int) -> tuple[int, ...]:
    """Fusion training blocks for a test block.

    Excludes the test block ``B_b`` and ``B_{b-1}``, whose context features were
    produced by a refiner fitted on ``B_b``'s labels.
    """
    if not 0 <= test_block < BLOCKS:
        raise ValueError("Unknown block")
    return tuple((test_block + offset) % BLOCKS for offset in (1, 2, 3))


def chain_blocks(test_block: int) -> set[int]:
    """Every block whose labels trained a model in the test block's own feature chain."""
    return (
        set(producer_blocks(test_block))
        | {refiner_block(test_block)}
        | set(fusion_train_blocks(test_block))
    )


__all__ = [
    "BLOCKS",
    "chain_blocks",
    "fusion_train_blocks",
    "producer_blocks",
    "refiner_block",
]
