"""Fixed-shape CUDA graphs for exact persistent Garl RGB reuse and current events."""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
import torch


class GraphPairedBackboneCache:
    """Replay native FP32 branches; cache only the exact paired RGB feature."""

    def __init__(self, model: torch.nn.Module) -> None:
        started = time.perf_counter()
        native = cast(Any, model)
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise ValueError("Persistent Garl graphs require CUDA")
        self.model = model
        self.key: tuple | None = None
        self.rgb: np.ndarray | None = None
        self.feature: torch.Tensor | None = None
        self.hits = self.misses = 0
        self.rgb_input = torch.zeros((1, 6, 128, 128), device=device)
        self.event_input = torch.zeros((1, 40, 128, 128), device=device)

        def rgb_forward() -> torch.Tensor:
            features, _ = native.backbone_rgb(self.rgb_input)
            return native.pool_layer(features).reshape(1, -1)

        def event_forward() -> torch.Tensor:
            features, _ = native.backbone_event(self.event_input)
            features = native.pool_layer(features).reshape(1, -1)
            return native.final_layer(
                native.middle_layer(torch.cat([self.rgb_feature_input, features], dim=1))
            )

        with torch.inference_mode():
            self.rgb_feature_input = torch.zeros_like(rgb_forward())
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    rgb_forward()
                    event_forward()
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            self.rgb_graph = torch.cuda.CUDAGraph()
            self.event_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.rgb_graph):
                self.rgb_output = rgb_forward()
            with torch.cuda.graph(self.event_graph):
                self.output = event_forward()
            torch.cuda.synchronize(device)
        self.startup = {
            "abi": "garl_native_fp32_separate_rgb_event_graphs_v1",
            "capture_seconds": time.perf_counter() - started,
            "capture_inputs": "synthetic zeros only; no evaluation queries",
            "rgb_shape": list(self.rgb_input.shape),
            "event_shape": list(self.event_input.shape),
            "cache": "one exact RGB pair; every call replays event backbone and TTC heads",
        }

    def predict(self, sensor: np.ndarray, key: tuple) -> torch.Tensor:
        """Return current native heights; output buffer is valid until the next call."""
        if sensor.shape != (46, 128, 128) or sensor.dtype != np.float32:
            raise ValueError("Expected native FP32 Garl input [46,128,128]")
        with torch.inference_mode():
            if key == self.key:
                if self.rgb is None or not np.array_equal(sensor[:6], self.rgb):
                    raise ValueError("Immutable cache key reused with different RGB bytes")
                self.hits += 1
            else:
                self.rgb_input.copy_(torch.from_numpy(np.ascontiguousarray(sensor[:6])[None]))
                self.rgb_graph.replay()
                # Own this cache: subsequent graph replays overwrite rgb_output.
                self.feature = self.rgb_output.clone()
                self.key, self.rgb = key, sensor[:6].copy()
                self.misses += 1
            assert self.feature is not None
            self.rgb_feature_input.copy_(self.feature)
            self.event_input.copy_(torch.from_numpy(np.ascontiguousarray(sensor[6:])[None]))
            self.event_graph.replay()
            return self.output
