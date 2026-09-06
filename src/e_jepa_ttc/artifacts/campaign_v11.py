"""Stage70–76 campaign contracts and fixed resource reservations."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from e_jepa_ttc.artifacts.risk_geometry_v10 import CampaignOwner


@dataclass(frozen=True)
class ResourcePolicy:
    """User-authorized resource amendment 001, expressed in absolute bytes."""

    disk_free_bytes: int = 120_000_000_000
    host_ram_available_bytes: int = 4 * 1024**3

    def __post_init__(self) -> None:
        if self.disk_free_bytes < 0 or self.host_ram_available_bytes < 0:
            raise ValueError("resource minima cannot be negative")


@dataclass(frozen=True)
class ResourceReservation:
    """Measured headroom after the entire proposed incremental allocation."""

    resource: str
    available_bytes: int
    planned_bytes: int
    required_free_after_bytes: int

    def __post_init__(self) -> None:
        if min(self.available_bytes, self.planned_bytes, self.required_free_after_bytes) < 0:
            raise ValueError("negative reservation or resource quantity")

    @property
    def projected_available_bytes(self) -> int:
        """Return the remaining bytes after all planned allocations."""
        return self.available_bytes - self.planned_bytes

    @property
    def passes(self) -> bool:
        """Accept the inclusive fixed-byte boundary, independent of volume size."""
        return self.projected_available_bytes >= self.required_free_after_bytes

    def record(self) -> dict[str, str | int | bool]:
        """Export a small JSON-compatible operational receipt."""
        return {
            **asdict(self),
            "projected_available_bytes": self.projected_available_bytes,
            "additional_free_bytes_needed": max(
                0, self.required_free_after_bytes - self.projected_available_bytes
            ),
            "passes": self.passes,
        }


DEFAULT_RESOURCE_POLICY = ResourcePolicy()


def disk_reservation(
    path: Path, planned_bytes: int, policy: ResourcePolicy = DEFAULT_RESOURCE_POLICY
) -> ResourceReservation:
    """Measure only a volume on which the proposed job will allocate storage."""
    return ResourceReservation(
        str(path.resolve()), shutil.disk_usage(path).free, planned_bytes, policy.disk_free_bytes
    )


def host_ram_reservation(
    planned_bytes: int, policy: ResourcePolicy = DEFAULT_RESOURCE_POLICY
) -> ResourceReservation:
    """Measure available host RAM with a fixed post-allocation minimum."""
    return ResourceReservation(
        "host_RAM",
        psutil.virtual_memory().available,
        planned_bytes,
        policy.host_ram_available_bytes,
    )


class ResourceOwner(CampaignOwner):
    """Reuse atomic writer ownership/telemetry with the amended fixed margins."""

    def __init__(self, root: Path, *, remaining_bytes: int, planned_ram_bytes: int) -> None:
        if min(remaining_bytes, planned_ram_bytes) < 0:
            raise ValueError("negative resource reservation")
        super().__init__(root)
        self.estimated_remaining_bytes = remaining_bytes
        self.planned_ram_bytes = planned_ram_bytes

    def check(self) -> None:
        """Monitor live RAM and disk remaining after the outstanding write budget."""
        if self.failure:
            raise RuntimeError(self.failure)
        if not host_ram_reservation(0).passes:
            raise RuntimeError("RESOURCE_BLOCKED: available host RAM below fixed 4 GiB")
        if not disk_reservation(self.root, self.estimated_remaining_bytes).passes:
            raise RuntimeError("RESOURCE_BLOCKED: projected free disk below fixed 120 GB")

    def __enter__(self) -> ResourceOwner:
        """Reserve the full maximum incremental RAM before starting allocations."""
        self.root.mkdir(parents=True, exist_ok=True)
        if not host_ram_reservation(self.planned_ram_bytes).passes:
            raise RuntimeError("RESOURCE_BLOCKED: full RAM reservation leaves less than 4 GiB")
        super().__enter__()
        return self
