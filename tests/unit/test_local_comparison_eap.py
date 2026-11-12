"""Keep the eAP timing cohort physical, deterministic and causally aligned."""

import pytest

from operational.local_comparison.eap import current_garl_row, select_physical


def test_selection_collapses_aliases_before_prefix_selection():
    data = {
        "physical_aliases": {"alias": "real", "real": "real"},
        "queries": [
            {"sample_token": "alias", "sequence_id": "s", "query_time_us": 1},
            {"sample_token": "real", "sequence_id": "s", "query_time_us": 1},
        ],
    }
    assert [q["sample_token"] for q in select_physical(data, 10)] == ["real"]


@pytest.mark.parametrize("sensor_interval", [100001, 100002])
def test_garl_row_uses_current_pair_not_an_older_rgb_history(sensor_interval):
    query = {
        "sample_token": "q",
        "sequence_id": "s",
        "query_time_us": 1000,
        "event_history": ["q"],
        "rgb_history": ["r"],
    }
    data = {
        "event_observations": {
            "q": {
                "windows_us": [
                    [0, sensor_interval],
                    [sensor_interval, 2 * sensor_interval],
                    [2 * sensor_interval, 3 * sensor_interval],
                ],
                "sensor_anchor_us": 3 * sensor_interval,
            }
        },
        "rgb_observations": {
            "r": {
                "sample_token": "q",
                "anchor_us": 1000,
                "producer_times_us": [-99000, 1000],
                "producer_sensor_times_us": [
                    sensor_interval,
                    2 * sensor_interval,
                    3 * sensor_interval,
                ],
                "producer_shards": ["a", "b", "c"],
                "producer_members": ["x", "y", "z"],
                "producer_boxes_xyxy": [[0, 0, 5, 5]] * 3,
            }
        },
    }
    row = current_garl_row(data, query)
    assert row["rgb_member_paths"] == ["y", "z"]
    assert row["event_windows_us"] == [
        [sensor_interval, 2 * sensor_interval],
        [2 * sensor_interval, 3 * sensor_interval],
    ]
    assert row["native_delta_t_s"] == 0.1
    assert row["sensor_delta_t_s"] == sensor_interval / 1e6
    data["rgb_observations"]["r"]["anchor_us"] -= 1
    with pytest.raises(ValueError, match="annotation anchor"):
        current_garl_row(data, query)
    data["rgb_observations"]["r"]["anchor_us"] += 1
    data["event_observations"]["q"]["sensor_anchor_us"] += 1
    with pytest.raises(ValueError, match="sensor anchor"):
        current_garl_row(data, query)
    data["event_observations"]["q"]["sensor_anchor_us"] -= 1
    data["rgb_observations"]["r"]["producer_sensor_times_us"][-1] += 1
    with pytest.raises(ValueError, match="endpoint"):
        current_garl_row(data, query)
