import importlib.util
import os

import cv2
import numpy as np
import pytest

from src.pipeline import dual_spectral_pipeline

app_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")


def load_dashboard():
    """import the dashboard fresh, the same way starting the server does"""
    spec = importlib.util.spec_from_file_location("dashboard_app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log_one_detection():
    pipe = dual_spectral_pipeline()
    frame = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (260, 250), 255, -1)
    pipe.process_frames(thermal_frame=frame)
    return pipe


def test_empty_database_reports_nothing_measured():
    client = load_dashboard().app.test_client()
    body = client.get("/api/telemetry").get_json()
    assert body["total"] == 0
    assert body["data"] == []
    assert body["avg_latency_ms"] is None
    assert body["detector"] is None


def test_telemetry_reports_pipeline_output():
    pipe = log_one_detection()
    client = load_dashboard().app.test_client()

    body = client.get("/api/telemetry").get_json()
    assert body["total"] == 1
    assert body["detector"] == "classical-baseline"
    assert body["avg_latency_ms"] == pytest.approx(pipe.last_latency_ms, abs=0.01)
    assert body["data"][0]["confidence"] is None


def test_starting_dashboard_keeps_existing_detections():
    log_one_detection()
    load_dashboard()
    client = load_dashboard().app.test_client()
    assert client.get("/api/telemetry").get_json()["total"] == 1


def test_index_page_has_no_remote_assets():
    """the dashboard must work with no internet"""
    html = load_dashboard().app.test_client().get("/").get_data(as_text=True)
    assert "http://" not in html and "https://" not in html
