from failure_probe.geometry import area, iou, xywh_to_xyxy


def test_geometry_helpers() -> None:
    box = xywh_to_xyxy([10, 20, 30, 40])
    assert box == (10, 20, 40, 60)
    assert area(box) == 1200
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 1 / 3
