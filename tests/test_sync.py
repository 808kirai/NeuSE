import gzip
import json
import socket
import sqlite3
import threading

import cv2
import numpy as np
import pytest
from werkzeug.serving import make_server

from command_center import store
from command_center.app import create_app, unzip
from src import sync
from src.config import config
from src.logger import data_logger
from src.pipeline import dual_spectral_pipeline


def log_hits(count, start=0):
    # saves count made up detections in one go, their ids follow the order of the boxes
    data_logger().log_events([{"bbox": [i, i, 10, 12], "confidence": 0.9, "thermal_peak": 200, "detector": "test"}
                              for i in range(start, start + count)], 5.0)


def rows(db):
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("select * from detections order by id")]


@pytest.fixture
def center(tmp_path):
    # a command center on its own database, reached through flask's test client
    app = create_app(tmp_path / "center.db", token="")
    client = app.test_client()

    def send(detections):
        reply = client.post("/api/v1/detections", json={"detections": detections})
        if reply.status_code != 200:
            raise sync.send_error(f"answered {reply.status_code}")
        return reply.get_json()

    send.client = client
    send.db = tmp_path / "center.db"
    return send


def offline(detections):
    raise sync.send_error("no link")


# logging

def test_each_detection_gets_an_id_and_unit(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "unit_id", "unit-7")
    log_hits(3)
    saved = rows(isolated_db)
    assert len({r["uid"] for r in saved}) == 3
    assert all(len(r["uid"]) == 32 and r["unit_id"] == "unit-7" and r["synced_at"] is None for r in saved)


def test_old_database_rows_get_ids_and_can_be_sent(isolated_db, center):
    with sqlite3.connect(isolated_db) as conn:
        conn.execute("""create table detections (id integer primary key autoincrement, timestamp text,
                        latitude real, longitude real, confidence real, thermal_peak real, bbox text)""")
        conn.executemany("insert into detections (timestamp, latitude, longitude, bbox) values (?, 1, 2, '[1, 2, 3, 4]')",
                         [("2026-11-01T10:00:00+02:00",), ("2026-11-01T10:00:01+02:00",)])
    data_logger()
    saved = rows(isolated_db)
    assert all(r["uid"] and r["unit_id"] == config.unit_id for r in saved)
    assert sync.sync_once(center)["synced"] == 2


# sending

def test_everything_queued_is_sent_and_marked(isolated_db, center):
    log_hits(5)
    result = sync.sync_once(center)
    assert result == {"synced": 5, "pending": 0, "error": None}
    assert all(r["synced_at"] for r in rows(isolated_db))
    assert store.count(center.db) == 5


def test_nothing_is_lost_while_the_link_is_down(isolated_db, center):
    log_hits(4)
    result = sync.sync_once(offline)
    assert result["synced"] == 0 and result["pending"] == 4 and result["error"]
    # the unit keeps detecting while offline
    log_hits(3, start=4)
    assert sync.sync_once(center)["synced"] == 7
    assert store.count(center.db) == 7


def test_resending_after_a_lost_reply_does_not_duplicate(isolated_db, center):
    log_hits(3)

    def reply_lost(detections):
        # the command center stores the batch but the answer never reaches the unit
        center(detections)
        raise sync.send_error("reply lost")

    assert sync.sync_once(reply_lost)["pending"] == 3
    assert sync.sync_once(center)["synced"] == 3
    assert store.count(center.db) == 3


def test_newest_detections_go_first_in_batches(isolated_db, monkeypatch):
    log_hits(250)
    batches = []

    def record(detections):
        batches.append([d["bbox"][0] for d in detections])
        return {"accepted": [d["uid"] for d in detections]}

    assert sync.sync_once(record, batch=100)["synced"] == 250
    assert [len(b) for b in batches] == [100, 100, 50]
    assert batches[0][0] == 249 and batches[-1][-1] == 0


def test_boxes_are_sent_as_lists(isolated_db):
    log_hits(1)
    assert sync.pending(10)[0]["bbox"] == [0, 0, 10, 12]


def test_only_confirmed_detections_are_marked(isolated_db):
    log_hits(4)

    def take_half(detections):
        return {"accepted": [d["uid"] for d in detections[:2]]}

    result = sync.sync_once(take_half, batch=4)
    # two taken, then the next batch only takes two more
    assert result["synced"] == 4 and result["pending"] == 0


def test_a_command_center_that_takes_nothing_does_not_loop(isolated_db):
    log_hits(2)
    result = sync.sync_once(lambda detections: {"accepted": []})
    assert result["synced"] == 0 and result["pending"] == 2 and result["error"]


def test_rejected_detections_are_kept_but_not_resent(isolated_db):
    log_hits(2)
    calls = []

    def refuse_one(detections):
        calls.append(len(detections))
        return {"accepted": [detections[0]["uid"]], "rejected": {detections[1]["uid"]: "bad box"}}

    assert sync.sync_once(refuse_one)["synced"] == 1
    assert sync.sync_once(refuse_one)["synced"] == 0
    assert calls == [2]
    assert [r["sync_error"] for r in rows(isolated_db)].count("bad box") == 1


@pytest.mark.parametrize("reply", [None, [], "ok", {"accepted": "all"}, {"accepted": [["x"]]}, {"rejected": ["x"]}])
def test_strange_replies_leave_the_queue_alone(isolated_db, reply):
    log_hits(2)
    result = sync.sync_once(lambda detections: reply)
    assert result["pending"] == 2 and result["error"]


# retry timing

def test_waits_double_after_failures_and_reset_after_success(isolated_db, center):
    log_hits(1)
    outcomes = iter([False, False, False, True, False])
    waits = []

    def flaky(detections):
        if not next(outcomes):
            raise sync.send_error("no link")
        return center(detections)

    def sleep(seconds):
        waits.append(seconds)
        # new detections arrive each round so every round has something to send
        log_hits(1, start=len(waits))

    sync.run(flaky, interval=5, max_wait=15, stop=lambda: len(waits) >= 5, sleep=sleep, jitter=lambda: 1.0)
    assert waits == [10, 15, 15, 5, 10]


# command center

def test_command_center_checks_each_detection(center):
    good = {"uid": "a" * 32, "unit_id": "unit-1", "timestamp": "2026-11-13T10:00:00+02:00",
            "latitude": -1.28, "longitude": 36.8, "confidence": 0.9, "thermal_peak": 200,
            "bbox": [1, 2, 3, 4], "detector": "x", "latency_ms": 3.0}
    bad = [
        {**good, "uid": "b" * 32, "timestamp": "2026-11-13T10:00:00"},
        {**good, "uid": "c" * 32, "latitude": 95},
        {**good, "uid": "d" * 32, "confidence": 1.5},
        {**good, "uid": "e" * 32, "bbox": [1, 2, 3]},
        {**good, "uid": "f" * 32, "unit_id": "unit 1; drop table"},
        {**good, "uid": "1" * 32, "latitude": True},
        {**good, "uid": "nothex"},
        "not a detection",
    ]
    reply = center([good] + bad)
    assert reply["accepted"] == [good["uid"]]
    assert set(reply["rejected"]) == {"b" * 32, "c" * 32, "d" * 32, "e" * 32, "f" * 32, "1" * 32, "nothex"}
    assert store.count(center.db) == 1


def test_command_center_summarises_units(isolated_db, center, monkeypatch):
    for unit, count in (("unit-a", 2), ("unit-b", 3)):
        monkeypatch.setattr(config, "unit_id", unit)
        log_hits(count)
    sync.sync_once(center)
    body = center.client.get("/api/v1/units").get_json()
    assert body["detections"] == 5
    assert {u["unit_id"]: u["detections"] for u in body["units"]} == {"unit-a": 2, "unit-b": 3}
    assert body["units"][0]["latitude"] == config.default_lat


def test_command_center_needs_the_token_when_set(tmp_path):
    client = create_app(tmp_path / "c.db", token="secret").test_client()
    assert client.post("/api/v1/detections", json={"detections": []}).status_code == 401
    assert client.get("/api/v1/units").status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/api/v1/detections", json={"detections": []}, headers=wrong).status_code == 401
    right = {"Authorization": "Bearer secret"}
    assert client.post("/api/v1/detections", json={"detections": []}, headers=right).status_code == 200
    assert client.get("/api/v1/health").status_code == 200


def test_command_center_refuses_bad_requests(center):
    client = center.client
    assert client.post("/api/v1/detections", data="not json").status_code == 400
    assert client.post("/api/v1/detections", json={"detections": "x"}).status_code == 400
    assert client.post("/api/v1/detections", json=[1]).status_code == 400
    assert client.post("/api/v1/detections", json={"detections": [{}] * 501}).status_code == 413
    zipped = {"Content-Encoding": "gzip"}
    assert client.post("/api/v1/detections", data=b"not gzip", headers=zipped).status_code == 400


def test_unzip_stops_oversized_bodies():
    with pytest.raises(ValueError):
        unzip(gzip.compress(b"0" * (17 * 1024 * 1024)))


# over real http

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_center(tmp_path):
    # the command center on a real port, so the unit's http sender is tested end to end
    server = make_server("127.0.0.1", 0, create_app(tmp_path / "live.db", token="secret"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", tmp_path / "live.db"
    server.shutdown()


def test_pipeline_detections_reach_the_command_center_over_http(isolated_db, live_center, monkeypatch):
    url, center_db = live_center
    monkeypatch.setattr(config, "unit_id", "unit-field-1")
    frame = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (260, 250), 255, -1)
    dual_spectral_pipeline().process_frames(thermal_frame=frame)

    result = sync.sync_once(sync.http_sender(url, token="secret"))
    assert result == {"synced": 1, "pending": 0, "error": None}
    assert store.units(center_db)[0]["unit_id"] == "unit-field-1"


def test_wrong_token_over_http_keeps_the_queue(isolated_db, live_center):
    url, _ = live_center
    log_hits(2)
    result = sync.sync_once(sync.http_sender(url, token="wrong"))
    assert result["pending"] == 2 and "401" in result["error"]


def test_unreachable_command_center_keeps_the_queue(isolated_db):
    log_hits(2)
    result = sync.sync_once(sync.http_sender(f"http://127.0.0.1:{free_port()}", timeout=2))
    assert result["pending"] == 2 and "could not reach" in result["error"]
