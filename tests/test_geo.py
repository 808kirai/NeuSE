import math

import pytest

from src.geo import ground_offset, ground_position, metres_per_degree

frame = (640, 512)
fov = (45, 37)
centre = [310, 246, 20, 20]


def test_straight_down_puts_the_frame_centre_under_the_unit():
    ahead, right = ground_offset(centre, frame, 100, 90, fov)
    assert ahead == pytest.approx(0, abs=1e-6) and right == pytest.approx(0, abs=1e-6)


def test_straight_down_the_top_of_the_frame_is_ahead_and_the_right_edge_is_right():
    top = ground_offset([310, -10, 20, 20], frame, 100, 90, fov)
    # half the view angle down the frame, from 100 m up
    assert top[0] == pytest.approx(100 * math.tan(math.radians(37 / 2)), rel=1e-6) and top[1] == pytest.approx(0)
    side = ground_offset([630, 246, 20, 20], frame, 100, 90, fov)
    assert side[1] == pytest.approx(100 * math.tan(math.radians(45 / 2)), rel=1e-6) and side[0] == pytest.approx(0)


def test_tilted_camera_looks_further_ahead():
    # 45 degrees down from 100 m meets the ground 100 m ahead
    ahead, right = ground_offset(centre, frame, 100, 45, fov)
    assert ahead == pytest.approx(100) and right == pytest.approx(0)
    # 30 degrees down from 130 m is about 225 m ahead, the error when a sighting is placed at the unit
    assert ground_offset(centre, frame, 130, 30, fov)[0] == pytest.approx(225, abs=1)


def test_a_box_above_the_horizon_has_no_ground_position():
    assert ground_offset([310, -10, 20, 20], frame, 100, 10, fov) is None


def test_heading_turns_ahead_into_north_or_east():
    here = (-1.0, 36.0)
    pose = {"altitude": 100, "heading": 0, "tilt": 45}
    lat, lon = ground_position(here, pose, centre, frame, fov, 1500)
    assert (lat - here[0]) * metres_per_degree == pytest.approx(100, abs=0.01) and lon == pytest.approx(here[1])

    lat, lon = ground_position(here, {**pose, "heading": 90}, centre, frame, fov, 1500)
    east = (lon - here[1]) * metres_per_degree * math.cos(math.radians(here[0]))
    assert east == pytest.approx(100, abs=0.01) and lat == pytest.approx(here[0])

    lat, lon = ground_position(here, {**pose, "heading": 180}, centre, frame, fov, 1500)
    assert (lat - here[0]) * metres_per_degree == pytest.approx(-100, abs=0.01)


@pytest.mark.parametrize("pose", [
    None,
    {},
    {"altitude": 100, "heading": 0},
    {"altitude": 0, "heading": 0, "tilt": 45},
    {"altitude": 100, "heading": 0, "tilt": 0},
    {"altitude": 100, "heading": 0, "tilt": 120},
    {"altitude": "high", "heading": 0, "tilt": 45},
    {"altitude": 100, "heading": float("nan"), "tilt": 45},
])
def test_a_pose_that_cannot_be_used_gives_no_position(pose):
    assert ground_position((-1.0, 36.0), pose, centre, frame, fov, 1500) is None


def test_positions_too_far_to_trust_are_dropped():
    # just below the horizon from 100 m lands several km away
    far = {"altitude": 100, "heading": 0, "tilt": 2}
    assert ground_position((-1.0, 36.0), far, centre, frame, fov, 1500) is None
    assert ground_position((-1.0, 36.0), far, centre, frame, fov, 10000) is not None
