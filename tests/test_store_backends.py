"""
the same storage checks against sqlite and mysql
mysql runs when NEUSE_TEST_MYSQL_URL points at an empty test database, as it does in ci
it drops and recreates the command center tables, so never point it at a real database
"""
import os
from datetime import datetime, timedelta, timezone

import pytest

from command_center import reports, store
from command_center.app import create_app

mysql_url = os.environ.get("NEUSE_TEST_MYSQL_URL")


@pytest.fixture(params=["sqlite", "mysql"])
def db(request, tmp_path):
    if request.param == "sqlite":
        path = tmp_path / "center.db"
        store.setup(path)
        return path
    if not mysql_url:
        # ci sets NEUSE_TEST_MYSQL_REQUIRED so a missing database fails the run instead of passing quietly
        if os.environ.get("NEUSE_TEST_MYSQL_REQUIRED"):
            pytest.fail("NEUSE_TEST_MYSQL_REQUIRED is set but NEUSE_TEST_MYSQL_URL is not")
        pytest.skip("set NEUSE_TEST_MYSQL_URL to an empty test database to run the mysql checks")
    with store.session(mysql_url) as s:
        s.run("drop table if exists detections")
        s.run("drop table if exists units")
        s.run("drop table if exists coverage")
    store.setup(mysql_url)
    # running setup twice must be harmless
    store.setup(mysql_url)
    return mysql_url


def detection(uid, unit="unit-1", minutes_ago=0, lat=-1.28, lon=36.8):
    moment = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {"uid": uid, "unit_id": unit, "timestamp": moment.isoformat(timespec="seconds"), "latitude": lat,
            "longitude": lon, "confidence": 0.9, "thermal_peak": 200, "bbox": [1, 2, 3, 4], "detector": "person-check",
            "latency_ms": 20.0, "position_source": "projected"}


def note(unit="unit-1", queued=0, lat=-1.28, lon=36.8):
    return {"unit_id": unit, "queued": queued, "latitude": lat, "longitude": lon, "detector": "person-check",
            "frames": 10, "refused": 0, "detector_idle_s": 2.5}


def test_each_detection_is_stored_once(db):
    accepted, rejected = store.save(db, [detection("a" * 32), detection("b" * 32)])
    assert accepted == ["a" * 32, "b" * 32] and rejected == {}
    # resending is safe
    assert store.save(db, [detection("a" * 32)])[0] == ["a" * 32]
    assert store.count(db) == 2


def test_bad_detections_are_refused(db):
    accepted, rejected = store.save(db, [detection("c" * 32), {**detection("d" * 32), "latitude": 200}])
    assert accepted == ["c" * 32] and list(rejected) == ["d" * 32] and store.count(db) == 1


def test_stored_fields_come_back_the_same(db):
    store.save(db, [detection("e" * 32)])
    row = store.recent(db)[0]
    assert row["uid"] == "e" * 32 and row["bbox"] == [1, 2, 3, 4] and row["position_source"] == "projected"
    assert row["latitude"] == pytest.approx(-1.28) and row["confidence"] == pytest.approx(0.9)
    assert isinstance(row["seq"], int)


def test_map_pages_by_arrival(db):
    store.save(db, [detection(f"{i:032x}") for i in range(12)])
    seen, after = [], 0
    while True:
        page = store.recent(db, after, 5)
        if not page:
            break
        seen += [d["uid"] for d in page]
        after = page[-1]["seq"]
    assert seen == [f"{i:032x}" for i in range(12)]
    # without after, the newest few in arrival order
    assert [d["uid"] for d in store.recent(db, None, 3)] == [f"{i:032x}" for i in range(9, 12)]


def test_status_notes_replace_the_last_one(db):
    store.save_status(db, note(queued=5))
    store.save_status(db, note(queued=2, lat=-1.5))
    unit = store.units(db)[0]
    assert unit["unit_id"] == "unit-1" and unit["queued"] == 2 and unit["latitude"] == pytest.approx(-1.5)
    assert unit["detector_idle_s"] == pytest.approx(2.5) and unit["detections"] == 0


def test_units_combine_detections_and_status(db):
    store.save(db, [detection("f" * 32, unit="unit-2", lat=-2.0)])
    store.save_status(db, note("unit-1"))
    units = {u["unit_id"]: u for u in store.units(db)}
    assert units["unit-2"]["detections"] == 1 and units["unit-2"]["latitude"] == pytest.approx(-2.0)
    assert units["unit-1"]["detections"] == 0


def test_report_facts_read_the_same(db):
    store.save(db, [detection("1" * 32, minutes_ago=2), detection("2" * 32, minutes_ago=90)])
    store.save_status(db, note())
    facts = reports.gather(db, minutes=30)
    assert facts["sightings"] == 1 and facts["units"][0]["online"] is True


def test_the_whole_api_runs_on_it(db):
    client = create_app(db, token="secret").test_client()
    auth = {"Authorization": "Bearer secret"}
    reply = client.post("/api/v1/detections", json={"detections": [detection("9" * 32)]}, headers=auth)
    assert reply.get_json()["accepted"] == ["9" * 32]
    assert client.post("/api/v1/status", json=note(), headers=auth).status_code == 200
    assert client.get("/api/v1/units", headers=auth).get_json()["detections"] == 1
    assert client.get("/api/v1/detections?after=0", headers=auth).get_json()["detections"][0]["uid"] == "9" * 32
    assert "1 sighting" in client.get("/api/v1/report", headers=auth).get_json()["text"]


def test_mysql_address_is_read_with_coded_characters():
    settings = store.mysql_settings("mysql://neuse:p%40ss%3Aword@192.168.0.10:3307/neuse_db")
    assert settings == {"host": "192.168.0.10", "port": 3307, "user": "neuse", "password": "p@ss:word",
                        "database": "neuse_db"}
    assert store.mysql_settings("mysql://neuse:x@db.example.org/neuse")["port"] == 3306
    for bad in ("mysql://db.example.org/neuse", "mysql://neuse:x@db.example.org/", "mysql:///neuse"):
        with pytest.raises(ValueError):
            store.mysql_settings(bad)


def test_coverage_keeps_the_newest_time_and_counts_units(db):
    now = datetime.now(timezone.utc)
    newer = (now - timedelta(minutes=1)).isoformat(timespec="seconds")
    older = (now - timedelta(minutes=40)).isoformat(timespec="seconds")
    assert store.save_coverage(db, "unit-1", [[1, 2, newer], [1, 3, older]]) == 2
    store.save_coverage(db, "unit-1", [[1, 2, older]])
    store.save_coverage(db, "unit-2", [[1, 2, older]])
    cells = {(c["cell_row"], c["cell_col"]): c for c in store.coverage(db)}
    assert cells[(1, 2)]["last_seen"] == datetime.fromisoformat(newer).isoformat(timespec="seconds")
    assert int(cells[(1, 2)]["units"]) == 2 and int(cells[(1, 3)]["units"]) == 1
    # only squares covered since a time
    assert len(store.coverage(db, (now - timedelta(minutes=10)).isoformat(timespec="seconds"))) == 1

