"""The evaluator identifies each object by position before looking at TTC."""

from operational.per_object_events import ARMS, score


def test_swapped_times_do_not_resolve_spatial_association() -> None:
    left, right = [0, 0, 20, 20], [40, 0, 60, 20]
    truth = {
        "s": {
            "100": [
                {"id": "a", "box": left, "ttc": 2.0},
                {"id": "b", "box": right, "ttc": 7.0},
            ]
        }
    }
    pred = {
        "s": {
            "100": [
                {"id": "p", "box": left, "ttc": dict.fromkeys(ARMS, 7.0)},
                {"id": "q", "box": right, "ttc": dict.fromkeys(ARMS, 2.0)},
            ]
        }
    }
    result = score(pred, truth, 0.5)
    assert result["matched"] == 2
    assert result["arms"][ARMS[0]]["mae_seconds"] == 5.0


def test_misses_and_unconfirmed_tracks_remain_in_denominator() -> None:
    ref = {"id": "a", "box": [0, 0, 20, 20], "ttc": 2.0}
    truth = {"s": {"100": [ref], "200": [ref]}}
    pred = {"s": {"100": [], "200": [{"id": "p", "box": ref["box"], "ttc": {}}]}}
    result = score(pred, truth, 0.5)
    assert result["objects"] == 2
    assert result["matched"] == 1
    assert result["arms"][ARMS[0]] == {"with_ttc": 0, "mae_seconds": None}
