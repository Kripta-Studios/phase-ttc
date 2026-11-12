"""CUDA Graph reuse must preserve current event work and exact paired-RGB identity."""

import numpy as np
import pytest
import torch
from torch import nn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_graph_matches_eager_on_rgb_miss_hit_and_replacement():
    from operational.local_comparison.garl_graph import GraphPairedBackboneCache

    class Backbone(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Conv2d(channels, 8, 3, padding=1)

        def forward(self, value):
            return self.conv(value).relu(), None

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_rgb = Backbone(6)
            self.backbone_event = Backbone(40)
            self.pool_layer = nn.AdaptiveAvgPool2d(1)
            self.middle_layer = nn.Linear(16, 8)
            self.final_layer = nn.Linear(8, 2)

        def forward(self, value):
            rgb = self.pool_layer(self.backbone_rgb(value[:, :6])[0]).flatten(1)
            event = self.pool_layer(self.backbone_event(value[:, 6:])[0]).flatten(1)
            return self.final_layer(self.middle_layer(torch.cat([rgb, event], dim=1)))

    torch.manual_seed(7)
    model = Model().cuda().eval().requires_grad_(False)
    cache = GraphPairedBackboneCache(model)
    rng = np.random.default_rng(11)
    value = rng.normal(size=(46, 128, 128)).astype(np.float32)
    for key, channel in [("a", 0), ("a", 6), ("b", 0), ("a", 0)]:
        value[channel:] += np.float32(0.01)
        with torch.inference_mode():
            expected = model(torch.from_numpy(value[None]).cuda()).cpu()
            actual = cache.predict(value, (key,)).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert cache.hits == 1
    assert cache.misses == 3
    value[:6] += 1
    with pytest.raises(ValueError, match="RGB bytes"):
        cache.predict(value, ("a",))
