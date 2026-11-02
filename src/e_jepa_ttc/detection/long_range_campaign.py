"""Separate admission for the all-observed-target 300m continuation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from e_jepa_ttc.detection.range_campaign import RangeConfig


class LongRangeConfig(RangeConfig):
    """Preserve the executed 60/150m producer contract without weakening it."""

    @classmethod
    def load(cls, path: Path) -> LongRangeConfig:
        """Keep the subclass return type while sharing strict field parsing."""
        return cast(LongRangeConfig, super().load(path))

    def validate(self) -> None:
        """Admit only the preregistered 300m extension with unchanged other settings."""
        if self.schema != "phase_eap_long_range_campaign_v1" or self.depth_max_m != 300.0:
            raise ValueError("Long-range continuation requires its own schema and 300m limit")
        RangeConfig.validate(replace(self, schema="phase_eap_range_campaign_v1", depth_max_m=150.0))
