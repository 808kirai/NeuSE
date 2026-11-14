import json
import socket
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from command_center import reports, store
from command_center.app import create_app

now = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
base = (-1.2864, 36.8172)


def metres_north(lat, metres):
    return lat + metres / reports.metres_per_degree


def sighting(uid, unit, minutes_ago, lat, lon, confidence=0.9):
    moment = now - timedelta(minutes=minutes_ago)
    return {"uid": uid, "unit_id": unit, "timestamp": moment.isoformat(), "latitude": lat, "longitude": lon,
            "confidence": confidence, "thermal_peak": 200, "bbox": [1, 2, 3, 4], "detector": "x", "latency_ms": 1.0}


def status(db, unit, lat, lon, queued, minutes_ago):
    store.save_status(db, {"unit_id": unit, "latitude": lat, "longitude": lon, "queued": queued})
    seen = (now - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")
    with store.connect(db) as conn:
        conn.execute("update units set last_status_at = ? where unit_id = ?", (seen, unit))


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "center.db"
    store.setup(path)
    # two sightings 30 m apart make one area, one 2 km away makes another, and one is too old to count
    store.save(path, [
        sighting("a" * 32, "unit-1", 2, base[0], base[1], 0.97),
        sighting("b" * 32, "unit-1", 3, metres_north(base[0], 30), base[1], 0.8),
        sighting("c" * 32, "unit-2", 10, metres_north(base[0], 2000), base[1], 0.7),
        sighting("d" * 32, "unit-2", 90, base[0], base[1]),
    ])
    # pretend each sighting arrived when it was seen, the test clock is fixed at noon
    with store.connect(path) as conn:
        conn.execute("update detections set received_at = timestamp_utc")
    status(path, "unit-1", base[0], base[1], 0, minutes_ago=0)
    status(path, "unit-2", metres_north(base[0], 2000), base[1], 16, minutes_ago=12)
    return path


def test_sightings_close_together_form_one_area():
    points = [{"timestamp_utc": "1", "latitude": base[0], "longitude": base[1]},
              {"timestamp_utc": "2", "latitude": metres_north(base[0], 50), "longitude": base[1]},
              {"timestamp_utc": "3", "latitude": metres_north(base[0], 500), "longitude": base[1]}]
    areas = reports.group_areas(points)
    assert sorted(len(a["points"]) for a in areas) == [1, 2]


def test_distance_is_close_to_true_metres():
    assert reports.distance(base, (metres_north(base[0], 1000), base[1])) == pytest.approx(1000, rel=0.01)


def test_facts_cover_only_the_window(db):
    facts = reports.gather(db, minutes=30, now=now)
    assert facts["sightings"] == 3
    assert [a["sightings"] for a in facts["areas"]] == [2, 1]
    first = facts["areas"][0]
    assert first["minutes_since_latest"] == 2 and first["highest_confidence"] == 0.97
    assert first["nearest_unit"] == "unit-1" and first["nearest_unit_metres"] < 50


def test_facts_flag_silent_units_with_a_backlog(db):
    units = {u["unit_id"]: u for u in reports.gather(db, minutes=30, now=now)["units"]}
    assert units["unit-1"]["online"] is True
    assert units["unit-2"]["online"] is False and units["unit-2"]["waiting_on_unit"] == 16


def test_template_report_says_what_matters(db):
    text = reports.write_template(reports.gather(db, minutes=30, now=now))
    assert "1 of 2 units online" in text
    # its last sighting arrived 10 min ago, after its last status note at 12 min
    assert "unit-2 has been silent for 10 min" in text and "16 sightings still waiting" in text
    assert "3 sightings in 2 areas" in text and "not a head count" in text
    assert "Area 1" in text and "highest confidence 0.97" in text
    # never a count of people
    assert "people found" not in text


def test_report_says_when_a_unit_is_in_contact_but_not_searching(db):
    # unit-1 still syncs, but its detector last handled a frame 20 minutes before its last note
    with store.connect(db) as conn:
        conn.execute("update units set detector_idle_s = 1200, refused = 3 where unit_id = 'unit-1'")
    facts = reports.gather(db, minutes=30, now=now)
    unit = next(u for u in facts["units"] if u["unit_id"] == "unit-1")
    assert unit["online"] is True and unit["detecting"] is False and unit["detector_quiet_minutes"] == 20
    text = reports.write_template(facts)
    assert "unit-1 is in contact but its detector has been quiet for 20 min" in text
    assert "unit-1 holds 3 sightings the command center refused" in text


def test_report_says_when_sightings_are_only_placed_at_the_unit(db):
    text = reports.write_template(reports.gather(db, minutes=30, now=now))
    assert "3 sightings could only be placed at the unit's own position" in text
    with store.connect(db) as conn:
        conn.execute("update detections set position_source = 'projected'")
    assert "placed at the unit" not in reports.write_template(reports.gather(db, minutes=30, now=now))


def test_language_model_gets_a_short_list_of_areas():
    facts = {"areas": [{"n": i} for i in range(50)], "sightings": 50}
    trimmed = reports.for_model(facts)
    assert len(trimmed["areas"]) == reports.model_areas and trimmed["areas_not_listed"] == 30


def test_template_report_with_nothing_seen(tmp_path):
    path = tmp_path / "empty.db"
    store.setup(path)
    store.save_status(path, {"unit_id": "unit-1", "latitude": 0.0, "longitude": 0.0, "queued": 0})
    text = reports.write_template(reports.gather(path, minutes=30))
    assert "No sightings of people" in text


# language model

class fake_model(BaseHTTPRequestHandler):
    # answers like an openai style chat api, or fails when told to
    reply = {"choices": [{"message": {"content": "Area 1 needs attention first."}}]}
    status = 200
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        fake_model.seen.append((self.path, self.headers["Authorization"], body))
        data = json.dumps(fake_model.reply).encode()
        self.send_response(fake_model.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def model_url():
    fake_model.seen = []
    fake_model.status = 200
    fake_model.reply = {"choices": [{"message": {"content": "Area 1 needs attention first."}}]}
    server = HTTPServer(("127.0.0.1", 0), fake_model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_model_writes_the_report_from_the_facts(db, model_url):
    result = reports.report(db, 30, (model_url, "key-1", "model-x"), now=now)
    assert result["source"] == "model" and result["text"] == "Area 1 needs attention first."
    path, auth, body = fake_model.seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer key-1" and body["model"] == "model-x"
    # the model is told not to count people and only gets the measured facts
    assert "never turn sightings into a count of people" in body["messages"][0]["content"]
    assert json.loads(body["messages"][1]["content"]) == reports.for_model(result["facts"])


@pytest.mark.parametrize("status, reply", [
    (500, {"error": "down"}),
    (200, {"choices": []}),
    (200, {"choices": [{"message": {"content": "  "}}]}),
    (200, "not an object"),
])
def test_model_problems_fall_back_to_the_template(db, model_url, status, reply):
    fake_model.status, fake_model.reply = status, reply
    result = reports.report(db, 30, (model_url, "key", "model"), now=now)
    assert result["source"] == "template" and "could not be used" in result["note"]
    assert "Area 1" in result["text"]


def test_unreachable_model_falls_back_to_the_template(db):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    result = reports.report(db, 30, (f"http://127.0.0.1:{port}/v1", "key", "model"), now=now)
    assert result["source"] == "template"


def test_no_model_settings_use_the_template(db):
    assert reports.report(db, 30, (None, None, None), now=now)["source"] == "template"


# api

def test_report_api(db):
    client = create_app(db, token="secret", llm=(None, None, None)).test_client()
    assert client.get("/api/v1/report").status_code == 401
    auth = {"Authorization": "Bearer secret"}
    body = client.get("/api/v1/report?minutes=600", headers=auth).get_json()
    assert body["source"] == "template" and body["facts"]["window_minutes"] == 600
    for bad in ("0", "1441", "soon"):
        assert client.get(f"/api/v1/report?minutes={bad}", headers=auth).status_code == 400


def test_page_has_the_report_panel_and_counts_sightings(tmp_path):
    page = create_app(tmp_path / "c.db", token="").test_client().get("/").get_data(as_text=True)
    assert 'id="report-button"' in page
    assert "people found" not in page
