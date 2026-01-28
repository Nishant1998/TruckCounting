from src.tracklets import RecentEnding, should_merge_tracklets


def test_should_merge_tracklets() -> None:
    ending = RecentEnding(vehicle_uid=1, end_time=1.0, end_point=(100.0, 100.0), end_area=400.0)
    assert should_merge_tracklets(
        start_time=1.5,
        start_point=(120.0, 105.0),
        start_area=380.0,
        ending=ending,
        merge_window=1.0,
        merge_distance=60.0,
        area_ratio_tol=0.5,
    )
    assert not should_merge_tracklets(
        start_time=3.0,
        start_point=(120.0, 105.0),
        start_area=380.0,
        ending=ending,
        merge_window=1.0,
        merge_distance=60.0,
        area_ratio_tol=0.5,
    )


if __name__ == "__main__":
    test_should_merge_tracklets()
    print("merge test passed")
