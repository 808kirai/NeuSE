import base64
import hashlib
import json
import math
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from command_center import reports, store
from command_center.app import create_app, parse_unit_tokens
from demo_units import flight_heading, flight_path, frame_pose, link, metres_per_degree, run_unit
from src import sync
from src.config import config
from src.logger import data_logger, read_status
from src.pipeline import thermal_pipeline


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


def test_a_unit_that_syncs_but_no_longer_detects_is_not_shown_as_searching(client):
    # its sync program is alive, but its detector last handled a frame 3 hours ago
    client.post("/api/v1/status", json={**note("unit-1"), "detector_idle_s": 10800.0})
    client.post("/api/v1/status", json={**note("unit-2"), "detector_idle_s": 2.0})
    client.post("/api/v1/status", json=note("unit-3"))
    units = {u["unit_id"]: u for u in client.get("/api/v1/units").get_json()["units"]}
    assert units["unit-1"]["online"] is True and units["unit-1"]["detecting"] is False
    assert units["unit-1"]["detector_idle_s"] >= 10800
    assert units["unit-2"]["detecting"] is True
    # a unit that does not say stays unknown instead of being guessed
    assert units["unit-3"]["detecting"] is None


def test_unit_status_note_says_how_long_the_detector_has_been_quiet(isolated_db):
    logger = data_logger()
    logger.note_status(None, "person-check", 10, 20.0)
    quiet = (datetime.now().astimezone() - timedelta(minutes=5)).isoformat(timespec="seconds")
    with sqlite3.connect(isolated_db) as conn:
        conn.execute("update unit_status set updated_at = ?", (quiet,))
    current = sync.status_note()
    assert 295 <= current["detector_idle_s"] <= 310 and current["refused"] == 0


# who may do what

def far_away(app):
    # a client that looks like it comes from another computer
    return app.test_client(), {"REMOTE_ADDR": "203.0.113.9"}


def test_with_no_token_only_this_computer_is_answered(tmp_path):
    app = create_app(tmp_path / "c.db", token="", unit_tokens={})
    remote, elsewhere = far_away(app)
    assert remote.get("/api/v1/units", environ_base=elsewhere).status_code == 401
    assert remote.post("/api/v1/detections", json={"detections": []}, environ_base=elsewhere).status_code == 401
    assert remote.post("/api/v1/status", json=note(), environ_base=elsewhere).status_code == 401
    # this computer still gets in, unless the request was passed on by a proxy
    assert remote.get("/api/v1/units").status_code == 200
    assert remote.get("/api/v1/units", headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 401
    # the health check and the page itself carry no data
    assert remote.get("/api/v1/health", environ_base=elsewhere).status_code == 200


def test_a_unit_token_can_send_for_its_own_unit_only_and_cannot_read(tmp_path):
    app = create_app(tmp_path / "c.db", token="coordinator", unit_tokens={"unit-1": "one", "unit-2": "two"})
    client = app.test_client()
    one = {"Authorization": "Bearer one"}
    mine, theirs = detection("a" * 32, unit="unit-1"), detection("b" * 32, unit="unit-2")
    reply = client.post("/api/v1/detections", json={"detections": [mine, theirs]}, headers=one).get_json()
    assert reply["accepted"] == ["a" * 32]
    assert reply["rejected"] == {"b" * 32: "this token only sends for unit-1"}
    assert client.post("/api/v1/status", json=note("unit-1"), headers=one).status_code == 200
    assert client.post("/api/v1/status", json=note("unit-2"), headers=one).status_code == 403
    # a lost unit must not be able to read where people are
    for path in ("/api/v1/units", "/api/v1/detections", "/api/v1/report"):
        assert client.get(path, headers=one).status_code == 403
        assert client.get(path, headers={"Authorization": "Bearer coordinator"}).status_code == 200
    assert client.get("/api/v1/units", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_a_unit_named_all_gets_no_extra_rights(tmp_path):
    client = create_app(tmp_path / "c.db", token="coordinator", unit_tokens={"all": "sneaky"}).test_client()
    assert client.get("/api/v1/units", headers={"Authorization": "Bearer sneaky"}).status_code == 403


def test_unit_tokens_setting_is_checked():
    assert parse_unit_tokens("unit-1=abc, unit-2=def") == {"unit-1": "abc", "unit-2": "def"}
    assert parse_unit_tokens(None) == {}
    for bad in ("unit 1=abc", "unit-1", "unit-1="):
        with pytest.raises(ValueError):
            parse_unit_tokens(bad)


# inputs that used to make the server fail

def test_strange_numbers_and_times_are_refused_not_crashed_on(client):
    cases = {
        "c" * 32: {"latitude": int("9" * 400)},
        "d" * 32: {"bbox": [1, 2, 3, int("9" * 400)]},
        "e" * 32: {"timestamp": "0001-01-01T00:00:00+05:00"},
        "f" * 32: {"timestamp": "9999-12-31T23:59:59-05:00"},
        "1" * 32: {"thermal_peak": int("9" * 400)},
        "2" * 32: {"position_source": "the moon"},
    }
    batch = [{**detection(uid), **change} for uid, change in cases.items()]
    reply = client.post("/api/v1/detections", data=json.dumps({"detections": batch}),
                        content_type="application/json")
    assert reply.status_code == 200
    assert set(reply.get_json()["rejected"]) == set(cases) and reply.get_json()["accepted"] == []


def test_strange_requests_get_a_plain_refusal(client):
    assert client.post("/api/v1/status", json={**note(), "queued": 2 ** 70}).status_code == 400
    assert client.post("/api/v1/status", json={**note(), "detector_idle_s": -1}).status_code == 400
    deep = "[" * 100000 + "]" * 100000
    assert client.post("/api/v1/detections", data=deep, content_type="application/json").status_code == 400
    assert client.post("/api/v1/detections", json={"detections": [], "sent_at": "noon"}).status_code == 400
    assert client.get(f"/api/v1/detections?after={10 ** 30}").status_code == 400


# unit clocks

def test_times_from_a_unit_with_a_wrong_clock_are_corrected(tmp_path):
    # field computers often have no battery clock, so a unit can think it is hours ahead or years behind
    # each batch carries the unit's own clock reading, and the gap to this computer's clock is taken off
    now = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
    for label, wrong_by in (("fast", timedelta(hours=2)), ("slow", timedelta(days=-2200)), ("right", timedelta(0))):
        db = tmp_path / f"{label}.db"
        store.setup(db)
        unit_now = now + wrong_by
        # seen 10 minutes before the batch was sent, by the unit's own clock
        seen = {**detection("a" * 32), "timestamp": (unit_now - timedelta(minutes=10)).isoformat()}
        accepted, rejected = store.save(db, [seen], sent_at=unit_now, now=now)
        assert accepted == ["a" * 32] and not rejected
        stored = store.recent(db)[0]
        assert stored["timestamp_utc"] == (now - timedelta(minutes=10)).isoformat(timespec="seconds")
        # the time the unit gave is kept as it was, next to the correction
        assert stored["timestamp"] == seen["timestamp"]
        assert stored["clock_offset_s"] == pytest.approx(-wrong_by.total_seconds())
        # and the report for the last half hour now includes it
        assert reports.gather(db, minutes=30, now=now)["sightings"] == 1


def test_a_corrected_time_is_never_in_the_future(tmp_path):
    db = tmp_path / "c.db"
    store.setup(db)
    now = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
    unit_now = now - timedelta(hours=5)
    # stamped after the unit says it sent the batch, which cannot be right
    odd = {**detection("a" * 32), "timestamp": (unit_now + timedelta(hours=1)).isoformat()}
    store.save(db, [odd], sent_at=unit_now, now=now)
    assert store.recent(db)[0]["timestamp_utc"] == now.isoformat(timespec="seconds")


def test_a_database_from_the_first_version_is_moved_over(tmp_path):
    # the first version had no seq column and paged by sqlite's hidden row number
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""create table detections (uid text primary key, unit_id text not null, timestamp text not null,
                        timestamp_utc text not null, latitude real not null, longitude real not null, confidence real,
                        thermal_peak real, bbox text, detector text, latency_ms real, received_at text not null)""")
        conn.execute("create index detections_time on detections (timestamp_utc)")
        conn.execute("""create table units (unit_id text primary key, last_status_at text not null, queued integer,
                        latitude real, longitude real, detector text, frames integer)""")
        for i, uid in enumerate(("b" * 32, "a" * 32, "c" * 32)):
            conn.execute("insert into detections values (?, 'unit-1', ?, ?, 1, 2, 0.9, 200, '[1, 2, 3, 4]', 'x', 1, ?)",
                         (uid, f"2026-11-13T10:00:0{i}+00:00", f"2026-11-13T10:00:0{i}+00:00",
                          "2026-11-13T10:00:09+00:00"))
        conn.execute("insert into units values ('unit-1', '2026-11-13T10:00:09+00:00', 0, 1, 2, 'x', 5)")

    client = create_app(db, token="").test_client()
    kept = client.get("/api/v1/detections").get_json()["detections"]
    # same rows, in the order they first arrived, now with a running number
    assert [d["uid"] for d in kept] == ["b" * 32, "a" * 32, "c" * 32] and [d["seq"] for d in kept] == [1, 2, 3]
    assert client.post("/api/v1/detections", json={"detections": [detection("d" * 32)]}).status_code == 200
    assert client.get("/api/v1/detections?after=3").get_json()["detections"][0]["uid"] == "d" * 32
    assert client.get("/api/v1/units").get_json()["units"][0]["unit_id"] == "unit-1"


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


def test_coordinator_page_serves_its_own_map_library(client):
    page = client.get("/").get_data(as_text=True)
    assert 'id="map"' in page
    # nothing the page needs to start comes from another site, so it works where those are blocked
    assert "/static/leaflet/leaflet.js" in page and "/static/leaflet/leaflet.css" in page
    assert "unpkg" not in page and "cdn" not in page.lower()
    # the copies in the repository are the published leaflet 1.9.4 files
    for name, digest in (("leaflet.js", "20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="),
                         ("leaflet.css", "p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=")):
        body = client.get(f"/static/leaflet/{name}").get_data()
        assert base64.b64encode(hashlib.sha256(body).digest()).decode() == digest
    # unit text is only ever written with textContent, and escaped before the map library sees it
    assert "innerHTML" not in page
    # the lists still work when the map cannot load
    assert "if (window.L)" in page and "tileerror" in page


def test_map_background_can_be_set(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUSE_TILE_URL", "https://tiles.example.org/{z}/{x}/{y}.png")
    page = create_app(tmp_path / "c.db", token="").test_client().get("/").get_data(as_text=True)
    assert "tiles.example.org" in page and "openstreetmap" not in page.lower()


# unit side

def test_status_row_tracks_position_and_frames(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "status_every", 0.0)
    pipe = thermal_pipeline()
    pipe.process_frames(thermal_frame=np.zeros((480, 640), np.uint8), position=(-1.5, 36.5))
    pipe.process_frames(thermal_frame=np.zeros((480, 640), np.uint8), position=(-1.6, 36.6))
    status = read_status()
    assert (status["latitude"], status["longitude"]) == (-1.6, 36.6)
    assert status["frames"] == 2 and status["detector"] == "warm-spots"


def test_detections_are_saved_where_the_unit_was(isolated_db):
    frame = np.zeros((480, 640), np.uint8)
    cv2.rectangle(frame, (200, 150), (260, 250), 255, -1)
    thermal_pipeline().process_frames(thermal_frame=frame, position=(-1.5, 36.5))
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

def test_sightings_are_placed_on_the_ground_when_the_unit_gives_its_pose(isolated_db, tmp_path):
    frame = np.zeros((480, 640), np.uint8)
    cv2.rectangle(frame, (290, 190), (350, 290), 255, -1)
    pipe = thermal_pipeline()
    here = (-1.5, 36.5)
    # looking 45 degrees down from 100 m while facing north puts the frame centre 100 m to the north
    hits = pipe.process_frames(thermal_frame=frame, position=here, pose={"altitude": 100, "heading": 0, "tilt": 45})
    assert hits[0]["position_source"] == "projected"
    north = (hits[0]["position"][0] - here[0]) * metres_per_degree
    assert north == pytest.approx(100, abs=3) and hits[0]["position"][1] == pytest.approx(here[1], abs=1e-4)

    # the position and how it was worked out travel all the way to the command center
    center = create_app(tmp_path / "center.db", token="").test_client()
    sync.sync_once(lambda d: center.post("/api/v1/detections", json={"detections": d}).get_json())
    stored = center.get("/api/v1/detections").get_json()["detections"][0]
    assert stored["position_source"] == "projected" and stored["latitude"] == pytest.approx(hits[0]["position"][0])


def test_without_a_pose_a_sighting_is_marked_as_placed_at_the_unit(isolated_db):
    frame = np.zeros((480, 640), np.uint8)
    cv2.rectangle(frame, (200, 150), (260, 250), 255, -1)
    hits = thermal_pipeline().process_frames(thermal_frame=frame, position=(-1.5, 36.5))
    assert hits[0]["position_source"] == "unit" and hits[0]["position"] == (-1.5, 36.5)
    assert sync.pending(1)[0]["position_source"] == "unit"


def test_demo_frames_give_their_height_and_tilt():
    assert frame_pose("1_130_60_0_04077.jpg", 90.0) == {"altitude": 130.0, "heading": 90.0, "tilt": 60.0}
    assert frame_pose("0.jpg", 0.0) is None


def test_demo_flights_never_jump():
    # the path used to jump 280 m back to its start after the last line
    base = (-1.286389, 36.817223)
    for index in range(3):
        points = [flight_path(index, 3, base, step) for step in range(1200)]
        for a, b in zip(points, points[1:]):
            north = (b[0] - a[0]) * metres_per_degree
            east = (b[1] - a[1]) * metres_per_degree * math.cos(math.radians(base[0]))
            # 10 m along a line, or that plus 40 m across to the next one
            assert math.hypot(north, east) <= 42
    assert flight_heading(0, 3, base, 0) == pytest.approx(90, abs=1)


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


def test_a_stopped_table_move_loses_nothing(tmp_path, monkeypatch):
    # a power cut part way through moving the first version's table used to strand its rows
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""create table detections (uid text primary key, unit_id text not null, timestamp text not null,
                        timestamp_utc text not null, latitude real not null, longitude real not null, confidence real,
                        thermal_peak real, bbox text, detector text, latency_ms real, received_at text not null)""")
        conn.execute("insert into detections values ('a', 'unit-1', 't', 't', 1, 2, null, null, null, null, null, 'r')")

    real_connect = store.connect

    class power_cut:
        # a connection that dies straight after the table is renamed
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, *args):
            result = self.conn.execute(sql, *args)
            if "rename to detections_old" in sql:
                raise KeyboardInterrupt("power cut")
            return result

        def __getattr__(self, name):
            return getattr(self.conn, name)

    monkeypatch.setattr(store, "connect", lambda path: power_cut(real_connect(path)))
    with pytest.raises(KeyboardInterrupt):
        store.setup(db)
    monkeypatch.setattr(store, "connect", real_connect)

    # the move was undone as a whole, so the next start moves the row over
    store.setup(db)
    assert store.count(db) == 1 and store.recent(db)[0]["uid"] == "a"
    with sqlite3.connect(db) as conn:
        assert "detections_old" not in {r[0] for r in conn.execute("select name from sqlite_master")}


def test_rows_stranded_by_an_earlier_stopped_move_are_recovered(tmp_path):
    db = tmp_path / "stranded.db"
    store.setup(db)
    with sqlite3.connect(db) as conn:
        conn.execute("create table detections_old as select * from detections where 0")
        conn.execute("alter table detections_old drop column seq")
        conn.execute("""insert into detections_old (uid, unit_id, timestamp, timestamp_utc, latitude, longitude,
                        received_at) values ('z', 'unit-1', 't', 't', 1, 2, 'r')""")
    store.setup(db)
    assert store.count(db) == 1


def test_unit_tokens_without_a_coordinator_token_say_why_nothing_can_be_read(tmp_path):
    client = create_app(tmp_path / "c.db", token="", unit_tokens={"unit-1": "one"}).test_client()
    reply = client.get("/api/v1/units")
    assert reply.status_code == 401 and "NEUSE_CC_TOKEN" in reply.get_json()["error"]
    # units can still send
    assert client.post("/api/v1/status", json=note("unit-1"), headers={"Authorization": "Bearer one"}).status_code == 200


def test_empty_settings_from_a_deployment_file_count_as_not_set(tmp_path, monkeypatch):
    # docker passes NEUSE_TILE_URL= as an empty value, which used to leave the map with no background
    for name in ("NEUSE_TILE_URL", "NEUSE_TILE_CREDIT", "NEUSE_CC_DATABASE_URL", "NEUSE_LLM_URL", "NEUSE_CC_UNIT_TOKENS"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("NEUSE_CC_DB", str(tmp_path / "c.db"))
    app = create_app(token="secret")
    page = app.test_client().get("/").get_data(as_text=True)
    assert "tile.openstreetmap.org" in page and "OpenStreetMap contributors" in page
    assert (tmp_path / "c.db").exists()
    report = app.test_client().get("/api/v1/report", headers={"Authorization": "Bearer secret"}).get_json()
    assert report["source"] == "template"
