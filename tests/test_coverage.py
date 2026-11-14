"""
checks the search coverage, which ground each unit's camera has covered
from the footprint maths on the unit, through sync, to the command center's map data and reports
"""
import math
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from command_center import reports, store
from command_center.app import create_app
from src import geo, sync
from src.config import config
from src.logger import data_logger
from src.pipeline import thermal_pipeline

here = (-1.2864, 36.8172)
down = {"altitude": 100, "heading": 0, "tilt": 90}


def test_straight_down_footprint_is_centred_and_the_right_size():
    corners = geo.footprint(here, down, (640, 512), (45, 37), 1500)
    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]
    # 2 x 100 x tan(18.5 degrees) north to south, 2 x 100 x tan(22.5 degrees) east to west
    depth = (max(lats) - min(lats)) * geo.metres_per_degree
    width = (max(lons) - min(lons)) * geo.metres_per_degree * math.cos(math.radians(here[0]))
    assert depth == pytest.approx(66.9, abs=0.5) and width == pytest.approx(82.8, abs=0.5)
    assert np.mean(lats) == pytest.approx(here[0], abs=1e-6) and np.mean(lons) == pytest.approx(here[1], abs=1e-6)


def test_a_camera_looking_at_the_horizon_has_no_closed_footprint():
    assert geo.footprint(here, {**down, "tilt": 10}, (640, 512), (45, 37), 1500) is None


def test_squares_cover_about_the_footprint_area():
    corners = geo.footprint(here, {**down, "altitude": 400}, (640, 512), (45, 37), 1500)
    cells = geo.cells_in(corners)
    covered = sum(geo.cell_area(row) for row, _ in cells)
    true_area = (2 * 400 * math.tan(math.radians(22.5))) * (2 * 400 * math.tan(math.radians(18.5)))
    assert covered == pytest.approx(true_area, rel=0.1)


def test_point_in_polygon():
    square = [(0, 0), (0, 1), (1, 1), (1, 0)]
    assert geo.inside((0.5, 0.5), square) and not geo.inside((1.5, 0.5), square)


def test_a_huge_footprint_is_refused_instead_of_filling_memory():
    assert geo.cells_in([(0, 0), (0, 10), (10, 10), (10, 0)]) == set()


# unit side

def frames(pipe, count, pose=down):
    for _ in range(count):
        pipe.process_frames(thermal_frame=np.zeros((512, 640), np.uint8), position=here, pose=pose)


def saved_cells(db):
    with sqlite3.connect(db) as conn:
        return conn.execute("select count(*), sum(frames), min(synced) from coverage").fetchone()


def test_the_pipeline_records_covered_ground_with_its_status(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "status_every", 0.0)
    pipe = thermal_pipeline()
    frames(pipe, 3)
    count, total, synced = saved_cells(isolated_db)
    # the same 16 squares, three frames each, none sent yet
    assert count == 16 and total == 48 and synced == 0


def test_no_pose_means_no_coverage(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "status_every", 0.0)
    frames(thermal_pipeline(), 2, pose=None)
    assert saved_cells(isolated_db)[0] == 0


def test_flush_saves_what_is_still_in_memory(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "status_every", 3600.0)
    pipe = thermal_pipeline()
    frames(pipe, 3)
    # the first frame wrote its note, the next two are only in memory
    assert saved_cells(isolated_db)[1] == 16
    pipe.flush()
    assert saved_cells(isolated_db)[1] == 48


@pytest.fixture
def center(tmp_path):
    app = create_app(tmp_path / "center.db", token="", unit_tokens={})
    client = app.test_client()

    def send_coverage(cells):
        reply = client.post("/api/v1/coverage", json={"unit_id": config.unit_id, "cells": cells})
        if reply.status_code != 200:
            raise sync.send_error(f"answered {reply.status_code}")
        return reply.get_json()

    send_coverage.client = client
    send_coverage.db = tmp_path / "center.db"
    return send_coverage


def test_covered_ground_reaches_the_command_center(isolated_db, monkeypatch, center):
    monkeypatch.setattr(config, "status_every", 0.0)
    frames(thermal_pipeline(), 2)
    assert sync.sync_coverage(center) == 16
    assert saved_cells(isolated_db)[2] == 1
    # nothing new, nothing sent
    assert sync.sync_coverage(center) == 0
    data = center.client.get("/api/v1/coverage").get_json()
    assert len(data["cells"]) == 16 and data["cell_degrees"] == geo.cell_degrees


def test_ground_seen_again_is_sent_again_with_the_newer_time(isolated_db, monkeypatch, center):
    monkeypatch.setattr(config, "status_every", 0.0)
    pipe = thermal_pipeline()
    frames(pipe, 1)
    sync.sync_coverage(center)
    frames(pipe, 1)
    assert sync.sync_coverage(center) == 16


def test_a_failed_link_keeps_coverage_waiting(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "status_every", 0.0)
    frames(thermal_pipeline(), 1)

    def offline(cells):
        raise sync.send_error("no link")

    assert sync.sync_coverage(offline) is None
    assert saved_cells(isolated_db)[2] == 0


# command center side

def cell(row=-6432, col=184086, minutes_ago=0):
    moment = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return [row, col, moment.isoformat(timespec="seconds")]


def test_newer_times_win_and_older_ones_do_not_replace_them(tmp_path):
    db = tmp_path / "c.db"
    store.setup(db)
    store.save_coverage(db, "unit-1", [cell(minutes_ago=0)])
    store.save_coverage(db, "unit-1", [cell(minutes_ago=30)])
    newest = store.coverage(db)[0]["last_seen"]
    assert datetime.fromisoformat(newest) > datetime.now(timezone.utc) - timedelta(minutes=1)


def test_squares_covered_by_two_units_are_counted_once(tmp_path):
    db = tmp_path / "c.db"
    store.setup(db)
    store.save_coverage(db, "unit-1", [cell()])
    store.save_coverage(db, "unit-2", [cell()])
    rows = store.coverage(db)
    assert len(rows) == 1 and rows[0]["units"] == 2


@pytest.mark.parametrize("bad", [
    {"unit_id": "bad name", "cells": []},
    {"unit_id": "unit-1", "cells": "x"},
    {"unit_id": "unit-1", "cells": [[1, 2]]},
    {"unit_id": "unit-1", "cells": [[1.5, 2, "2026-11-14T10:00:00+00:00"]]},
    {"unit_id": "unit-1", "cells": [[10 ** 9, 2, "2026-11-14T10:00:00+00:00"]]},
    {"unit_id": "unit-1", "cells": [[1, 2, "noon"]]},
    {"unit_id": "unit-1", "cells": [[1, 2, "2026-11-14T10:00:00+00:00"]] * 5001},
    {"unit_id": "unit-1", "cells": [], "sent_at": "noon"},
])
def test_bad_coverage_is_refused(center, bad):
    assert center.client.post("/api/v1/coverage", json=bad).status_code == 400


def test_coverage_follows_the_token_rules(tmp_path):
    client = create_app(tmp_path / "c.db", token="boss", unit_tokens={"unit-1": "one"}).test_client()
    body = {"unit_id": "unit-1", "cells": [cell()]}
    assert client.post("/api/v1/coverage", json=body).status_code == 401
    assert client.post("/api/v1/coverage", json=body, headers={"Authorization": "Bearer one"}).status_code == 200
    assert client.post("/api/v1/coverage", json={**body, "unit_id": "unit-2"},
                       headers={"Authorization": "Bearer one"}).status_code == 403
    assert client.get("/api/v1/coverage", headers={"Authorization": "Bearer one"}).status_code == 403
    assert client.get("/api/v1/coverage", headers={"Authorization": "Bearer boss"}).status_code == 200


@pytest.mark.parametrize("hours", ["0", "200", "soon"])
def test_bad_coverage_windows_are_refused(center, hours):
    assert center.client.get(f"/api/v1/coverage?hours={hours}").status_code == 400


def test_coverage_from_a_wrong_clock_is_corrected(tmp_path):
    db = tmp_path / "c.db"
    store.setup(db)
    now = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
    unit_now = now + timedelta(hours=3)
    seen = (unit_now - timedelta(minutes=2)).isoformat(timespec="seconds")
    store.save_coverage(db, "unit-1", [[1, 2, seen]], sent_at=unit_now, now=now)
    assert store.coverage(db)[0]["last_seen"] == (now - timedelta(minutes=2)).isoformat(timespec="seconds")


def test_reports_say_how_much_ground_was_searched(tmp_path):
    db = tmp_path / "c.db"
    store.setup(db)
    store.save_status(db, {"unit_id": "unit-1", "latitude": here[0], "longitude": here[1], "queued": 0})
    store.save_coverage(db, "unit-1", [cell(-6432, c) for c in range(184000, 184100)])
    facts = reports.gather(db, minutes=30)
    # 100 squares of about 22 m by 22 m
    assert facts["searched_km2"] == pytest.approx(100 * geo.cell_area(-6432) / 1e6, abs=0.001)
    assert "square km searched" in reports.write_template(facts)


def test_the_map_page_has_the_searched_layer(center):
    page = center.client.get("/").get_data(as_text=True)
    assert 'id="show-searched"' in page and "/api/v1/coverage" in page
