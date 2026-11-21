import numpy as np

from scripts.audit_train40_box_provenance import compare_boxes


def test_integer_equivalence_is_distinct_from_continuous_iou() -> None:
    corners = np.array([[[0.3, 1.8, 1.0], [4.8, 6.9, 1.0]] * 4])
    valid, iou, error, exact = compare_boxes(corners, np.eye(3)[None], np.array([[0, 1, 4, 6]]))
    assert valid.item() and exact.item()
    assert 0 < iou.item() < 1
    assert np.isclose(error.item(), 0.9)


def test_clip_to_frame_and_reject_corner_behind_camera() -> None:
    vertices = [[-2.0, 4.0, 1.0], [1500.0, 900.0, 1.0]] * 4
    corners = np.array([vertices, vertices])
    corners[1, 1, 2] = -1
    k = np.broadcast_to(np.eye(3), (2, 3, 3))
    supplied = np.array([[0, 4, 1279, 719]] * 2)
    valid, iou, _, exact = compare_boxes(corners, k, supplied)
    assert valid.tolist() == [True, False]
    assert iou[0] == 1 and exact[0]
