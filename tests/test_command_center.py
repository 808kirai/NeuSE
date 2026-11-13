import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from command_center import store
from command_center.app import create_app
from demo_units import flight_path, link, metres_per_degree, run_unit
from src import sync
from src.config import config
from src.logger import data_logger, read_status
from src.pipeline import dual_spectral_pipeline


def detection(uid, unit="unit-1", minutes_ago=0, lat=-1.28, lon=36.8):
    moment = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {"uid": uid, "unit_id": unit, "timestamp": moment.isoformat(timespec="seconds"),
            "latitude": lat, "longitude": lon, "confidence": 0.9, "thermal_peak": 200,
            "bbox": [1, 2, 3, 4], "detector": "classical-person-check", "latency_ms": 20.0}


def note(unit="unit-1", queued=0, lat=-1.28, lon=36.8):
    return {"unit_id": unit, "queued": queued, "latitude": lat, "longitude": lon,
            "detector": "classical-person-check", "frames": 10, "updated_at": "2026-11-13T10:00:00+00:00"}


@pytest.fixture
def client(tmp_path):
    return create_app(tmp_path / "center.db", token="").test_client()


# status notes

def test_status_note_shows_a_unit_before_any_detections(client):
    assert client.post("/api/v1/status", json=note("unit-9", queued=12)).status_code == 200
    unit = client.get("/api/v1/units").get_json()["units"][0]
    assert unit["unit_id"] == "unit-9" and unit["detections"] == 0
    assert unit["queued"] == 12 and unit["online"] is True


@pytest.mark.parametrize("bad", [
    {**note(), "unit_id": "bad unit"},
    {**note(), "latitude": 100},
    {**note(), "queued": -1},
    {**note(), "queued": 1.5},
    {**note(), "frames": True},
    {**note(), "detector": "x" * 65},
    [note()],
])
def test_bad_status_notes_are_refused(client, bad):
    assert client.post("/api/v1/status", json=bad).status_code == 400


def test_status_needs_the_token_when_set(tmp_path):
    secured = create_app(tmp_path / "c.db", token="secret").test_client()
    assert secured.post("/api/v1/status", json=note()).status_code == 401
    assert secured.get("/api/v1/detections").status_code == 401


def test_units_go_offline_when_quiet(client, tmp_path, monkeypatch):
    client.post("/api/v1/status", json=note("unit-1"))
    client.post("/api/v1/status", json=note("unit-2"))
    # pretend unit-2 was last heard ten minutes ago
    with store.connect(tmp_path / "center.db") as conn:
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(timespec="seconds")
        conn.execute("update units set last_status_at = ? where unit_id = 'unit-2'", (old,))
    units = {u["unit_id"]: u for u in client.get("/api/v1/units").get_json()["units"]}
    assert units["unit-1"]["online"] is True and units["unit-2"]["online"] is False


def test_unit_position_comes_from_its_latest_status(client):
    client.post("/api/v1/detections", json={"detections": [detection("a" * 32, lat=-1.0)]})
    client.post("/api/v1/status", json=note(lat=-2.0))
    assert client.get("/api/v1/units").get_json()["units"][0]["latitude"] == -2.0


# detections for the map

def test_map_follows_arrival_order_so_late_backlog_is_not_missed(client):
    # a new detection arrives first, then an older one from a backlog
    client.post("/api/v1/detections", json={"detections": [detection("a" * 32, minutes_ago=0)]})
    first = client.get("/api/v1/detections").get_json()["detections"]
    assert [d["uid"] for d in first] == ["a" * 32]

    client.post("/api/v1/detections", json={"detections": [detection("b" * 32, minutes_ago=30)]})
    later = client.get(f"/api/v1/detections?after={first[-1]['seq']}").get_json()["detections"]
    assert [d["uid"] for d in later] == ["b" * 32]


def test_map_pages_through_a_long_backlog(client):
    uids = [f"{i:032x}" for i in range(30)]
    client.post("/api/v1/detections", json={"detections": [detection(u) for u in uids]})
    seen, after = [], 0
    while True:
        page = client.get(f"/api/v1/detections?after={after}&limit=7").get_json()["detections"]
        if not page:
            break
        seen += [d["uid"] for d in page]
        after = page[-1]["seq"]
    assert sorted(seen) == sorted(uids)


def test_map_returns_boxes_as_lists(client):
    client.post("/api/v1/detections", json={"detections": [detection("c" * 32)]})
    assert client.get("/api/v1/detections").get_json()["detections"][0]["bbox"] == [1, 2, 3, 4]


@pytest.mark.parametrize("query", ["?after=x", "?limit=many"])
def test_bad_map_queries_are_refused(client, query):
    assert client.get("/api/v1/detections" + query).status_code == 400


def test_coordinator_page_loads_a_pinned_map_library(client):
    page = client.get("/").get_data(as_text=True)
    assert 'id="map"' in page
    assert "leaflet@1.9.4" in page and 'integrity="sha256-' in page
    # unit text is only ever written with textContent
    assert "innerHTML" not in page


# unit side

def test_status_row_tracks_position_and_frames(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "status_every", 0.0)
    pipe = dual_spectral_pipeline()
    pipe.process_frames(thermal_frame=np.zeros((480, 640), np.uint8), position=(-1.5, 36.5))
    pipe.process_frames(thermal_frame=np.zeros((480, 640), np.uint8), position=(-1.6, 36.6))
    status = read_status()
    assert (status["latitude"], status["longitude"]) == (-1.6, 36.6)
    assert status["frames"] == 2 and status["detector"] == "warm-spots"


def test_detections_are_saved_where_the_unit_was(isolated_db):
    frame = np.zeros((480, 640), np.uint8)
    cv2.rectangle(frame, (200, 150), (260, 250), 255, -1)
    dual_spectral_pipeline().process_frames(thermal_frame=frame, position=(-1.5, 36.5))
    detection = sync.pending(1)[0]
    assert (detection["latitude"], detection["longitude"]) == (-1.5, 36.5)


def test_status_note_carries_the_queue(isolated_db):
    data_logger().log_events([{"bbox": [0, 0, 1, 1], "confidence": None, "thermal_peak": 1, "detector": "x"}], 1.0)
    current = sync.status_note()
    assert current["queued"] == 1 and current["unit_id"] == config.unit_id


def test_sync_loop_sends_status_notes(isolated_db, tmp_path):
    # the sync program sets up the unit's database before its first round
    data_logger()
    test_client = create_app(tmp_path / "center.db", token="").test_client()
    rounds = []
    sync.run(lambda d: {"accepted": [x["uid"] for x in d]}, interval=1, stop=lambda: len(rounds) >= 2,
             sleep=rounds.append,
             send_status=lambda n: test_client.post("/api/v1/status", json=n).get_json())
    assert test_client.get("/api/v1/units").get_json()["units"][0]["unit_id"] == config.unit_id


# demo

def test_flight_paths_stay_in_each_units_own_area():
    base = (-1.286389, 36.817223)
    for index in range(3):
        points = [flight_path(index, 3, base, step) for step in range(300)]
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        # each square is 300 m across
        assert (max(lats) - min(lats)) * metres_per_degree <= 301
        assert (max(lons) - min(lons)) * metres_per_degree * math.cos(math.radians(base[0])) <= 301
    # different units search different places
    assert flight_path(0, 3, base, 0) != flight_path(1, 3, base, 0)


def test_link_drops_and_comes_back():
    radio = link(random.Random(1), drop_chance=0.2, shortest=2, longest=3)
    states = [radio.tick() for _ in range(200)]
    assert True in states and False in states
    assert link(random.Random(1), drop_chance=0.0).tick() is True


def test_demo_unit_ends_with_everything_sent(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "detector", "person-check")
    images = []
    for i in range(3):
        frame = np.random.default_rng(i).integers(30, 50, (512, 640), dtype=np.uint8)
        cv2.ellipse(frame, (100 + 50 * i, 100), (6, 8), 0, 0, 360, 200, -1)
        images.append(tmp_path / f"{i}.jpg")
        cv2.imwrite(str(images[-1]), frame)

    center = create_app(tmp_path / "center.db", token="").test_client()
    send = lambda d: center.post("/api/v1/detections", json={"detections": d}).get_json()
    send_status = lambda n: center.post("/api/v1/status", json=n).get_json()
    radio = link(random.Random(0), drop_chance=0.5, shortest=1, longest=2)
    run_unit(images, 0, 2, (-1.28, 36.8), send, send_status, radio, frames=12, sleep=lambda s: None,
             say=lambda text: None)

    assert sync.pending_count() == 0
    units = center.get("/api/v1/units").get_json()["units"]
    assert units[0]["unit_id"] == config.unit_id and units[0]["queued"] == 0
