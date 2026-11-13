import importlib.util
import os

import cv2
import numpy as np
import pytest

from contextlib import closing

from src.config import config
from src.logger import connect
from src.pipeline import thermal_pipeline

# the dashboard is a script, not a package, so it is loaded by file path
app_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")


def load_dashboard():
    """import the dashboard fresh, the same way starting the server does"""
    spec = importlib.util.spec_from_file_location("dashboard_app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log_one_detection():
    """run the real pipeline on a frame with one warm area"""
    pipe = thermal_pipeline()
    frame = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (260, 250), 255, -1)
    pipe.process_frames(thermal_frame=frame)
    return pipe


def test_empty_database_reports_nothing_measured():
    """with no detections the api shows empty values instead of made up ones"""
    client = load_dashboard().app.test_client()
    body = client.get("/api/telemetry").get_json()
    assert body["total"] == 0
    assert body["data"] == []
    assert body["avg_latency_ms"] is None
    assert body["detector"] is None


def test_telemetry_reports_pipeline_output():
    """the api returns exactly what the pipeline saved"""
    pipe = log_one_detection()
    client = load_dashboard().app.test_client()

    body = client.get("/api/telemetry").get_json()
    assert body["total"] == 1
    # nothing has been sent to a command center yet
    assert body["queued"] == 1 and body["refused"] == 0
    assert body["detector"] == "warm-spots"
    assert body["avg_latency_ms"] == pytest.approx(pipe.last_latency_ms, abs=0.01)
    assert body["data"][0]["confidence"] is None


def test_starting_dashboard_keeps_existing_detections():
    """restarting the dashboard must never wipe saved detections"""
    log_one_detection()
    load_dashboard()
    client = load_dashboard().app.test_client()
    assert client.get("/api/telemetry").get_json()["total"] == 1


def test_index_page_has_no_remote_assets():
    """the dashboard must work with no internet"""
    html = load_dashboard().app.test_client().get("/").get_data(as_text=True)
    assert "http://" not in html and "https://" not in html


def test_average_time_counts_every_frame_not_every_detection(monkeypatch):
    """the average used to come from detection rows, so frames with many hits counted many times
    and frames with none never counted"""
    monkeypatch.setattr(config, "status_every", 0.0)
    pipe = log_one_detection()
    for _ in range(4):
        pipe.process_frames(thermal_frame=np.zeros((480, 640), dtype=np.uint8))
    body = load_dashboard().app.test_client().get("/api/telemetry").get_json()
    assert body["total"] == 1
    assert body["avg_latency_ms"] == pytest.approx(sum(pipe.recent_ms) / 5, abs=0.01)


def test_refused_detections_show_on_the_dashboard():
    log_one_detection()
    with closing(connect()) as conn, conn:
        conn.execute("update detections set sync_error = 'bad box'")
    body = load_dashboard().app.test_client().get("/api/telemetry").get_json()
    assert body["refused"] == 1 and body["queued"] == 0
