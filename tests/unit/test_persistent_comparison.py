"""Causal service and historical-runtime reconstruction contracts."""

import hashlib

import numpy as np
import pytest


def test_reference_relocation_keeps_hashes_and_rejects_outside_paths(tmp_path):
    from operational.local_comparison.persistent import relocate_reference_sources

    root = tmp_path / "original"
    original = {"sources": {str(root / "src/model.py"): "abc"}, "other": "unchanged"}
    relocated = relocate_reference_sources(original, root)
    assert relocated == {"sources": {"src/model.py": "abc"}, "other": "unchanged"}
    assert str(root / "src/model.py") in original["sources"]
    with pytest.raises(ValueError, match="outside"):
        relocate_reference_sources({"sources": {str(tmp_path / "other.py"): "abc"}}, root)


def test_materialization_restores_exact_archived_sources_without_mutating_checkout(tmp_path):
    from operational.local_comparison.persistent import materialize_runtime

    root = tmp_path / "repo"
    (root / "src/pkg").mkdir(parents=True)
    current = root / "src/pkg/model.py"
    current.write_bytes(b"new implementation\n")
    (root / "configs").mkdir()
    (root / "configs/runtime.json").write_bytes(b"{}")
    archive = tmp_path / "archive"
    archive.mkdir()
    original = b"original\r\n"
    (archive / "src__pkg__model.py.txt").write_bytes(original)
    binding = {"sources": {"src\\pkg\\model.py": hashlib.sha256(original).hexdigest()}}
    target = root / "output/runtime"
    receipt = materialize_runtime(root, target, archive, binding)
    assert (target / "src/pkg/model.py").read_bytes() == original
    assert current.read_bytes() == b"new implementation\n"
    assert (target / "configs/runtime.json").read_bytes() == b"{}"
    assert receipt["restored_sources"] == 1
    with pytest.raises(FileExistsError):
        materialize_runtime(root, target, archive, binding)
    (archive / "src__pkg__model.py.txt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        materialize_runtime(root, root / "output/bad", archive, binding)


def test_event_ring_preserves_native_window_bytes_and_rejects_gaps():
    from operational.local_comparison.persistent import EventRing

    def packet(ts):
        t = np.asarray(ts, dtype=np.int64)
        return {
            "x": np.ones(len(t), np.int32),
            "y": np.zeros(len(t), np.int32),
            "t": t,
            "p": np.ones(len(t), np.int8),
        }

    ring = EventRing(packet([2, 6, 9]), 0, 10, history_us=20)
    ring.append(packet([10, 19]), 10, 20)
    np.testing.assert_array_equal(ring.window(6, 19)["t"], [6, 9, 10])
    with pytest.raises(ValueError, match="gap"):
        ring.append(packet([25]), 25, 30)
    with pytest.raises(ValueError, match="future"):
        ring.window(10, 21)
    ring.append(packet([22, 29]), 20, 30)
    with pytest.raises(ValueError, match="history"):
        ring.window(0, 6)
    np.testing.assert_array_equal(ring.window(10, 30)["t"], [10, 19, 22, 29])


def test_paced_runner_accepts_declared_garl_output_and_rejects_missing_output():
    from operational.train40_runtime.paced import run_paced_replay

    for predictions, passed in [({"GARL_TTC": 2.0}, True), ({"other": 2.0}, False)]:
        result = run_paced_replay(
            lambda index: index,
            lambda arrival, predictions=predictions: {
                "roi": {"success": True, "predictions": predictions}
            },
            ["roi"],
            updates=1,
            startup_lead_s=0,
            prediction_components=("GARL_TTC",),
        )
        assert result["grouped"]["passed"] is passed
        assert result["prediction_components"] == ["GARL_TTC"]


def test_paired_backbone_cache_preserves_output_and_binds_exact_rgb_bytes():
    import torch
    from torch import nn

    from operational.local_comparison.garl_persistent import PairedBackboneCache

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x.mean(dim=1, keepdim=True), None

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_rgb = Backbone()
            self.backbone_event = Backbone()
            self.pool_layer = nn.AdaptiveAvgPool2d(1)
            self.middle_layer = nn.Identity()
            self.final_layer = nn.Identity()

    model = Model()
    cache = PairedBackboneCache(model)
    sensor = np.ones((46, 4, 4), np.float32)
    a = cache.predict(sensor, ("frame-pair",))
    sensor[6:] *= 2
    b = cache.predict(sensor, ("frame-pair",))
    torch.testing.assert_close(a, torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(b, torch.tensor([[1.0, 2.0]]))
    assert model.backbone_rgb.calls == 1
    assert model.backbone_event.calls == 2
    sensor[:6] *= 2
    with pytest.raises(ValueError, match="RGB bytes"):
        cache.predict(sensor, ("frame-pair",))
    c = cache.predict(sensor, ("different-ROI",))
    torch.testing.assert_close(c, torch.tensor([[2.0, 2.0]]))
    assert model.backbone_rgb.calls == 2


@pytest.mark.parametrize("square", [(3, 2, 21, 20), (-4, -2, 18, 20), (18, 16, 40, 38)])
def test_sparse_rgb_normalization_is_byte_exact_to_full_frame(square):
    from e_jepa_ttc.data.garl_official_preprocessing import official_resize_roi
    from operational.garl_evttc_competition.prepare import IMAGENET_MEAN, IMAGENET_STD
    from operational.local_comparison.garl_persistent import exact_rgb_roi

    rng = np.random.default_rng(19)
    frames = [rng.integers(0, 256, (29, 37, 3), dtype=np.uint8) for _ in range(2)]
    normalized = np.concatenate(
        [
            (f.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0) - IMAGENET_MEAN)
            / IMAGENET_STD
            for f in frames
        ]
    )
    expected = official_resize_roi(normalized, square, (128, 128)).numpy()
    actual = exact_rgb_roi(frames, square)
    assert actual.tobytes() == expected.tobytes()
