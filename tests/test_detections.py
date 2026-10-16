import sqlite3
from datetime import datetime

import cv2
import numpy as np
import pytest

from src.config import config
from src.logger import data_logger
from src.pipeline import dual_spectral_pipeline, detector_name


def thermal_with_target(dtype=np.uint8, peak=255):
    """cool frame with one warm rectangle"""
    frame = np.zeros((480, 640), dtype=dtype)
    cv2.rectangle(frame, (200, 150), (260, 250), int(peak), -1)
    return frame


def stored_rows(db):
    """every saved detection as a dict"""
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("select * from detections")]


@pytest.fixture
def pipeline():
    """fresh pipeline for each test"""
    return dual_spectral_pipeline()


def test_baseline_reports_no_confidence(pipeline):
    """no model means no confidence score"""
    hits = pipeline.process_frames(thermal_frame=thermal_with_target())
    assert len(hits) == 1
    assert hits[0]["confidence"] is None
    assert hits[0]["detector"] == detector_name
    assert hits[0]["thermal_peak"] == 255


def test_rgb_alone_is_not_treated_as_thermal(pipeline):
    """a bright object in the rgb frame is not heat"""
    bright_rgb = cv2.cvtColor(thermal_with_target(), cv2.COLOR_GRAY2BGR)
    assert pipeline.process_frames(rgb_frame=bright_rgb) == []


@pytest.mark.parametrize("frame", [
    thermal_with_target(np.uint16, 60000),     # raw 16 bit sensor data
    thermal_with_target(np.float32, 1.0),      # float data scaled 0 to 1
    thermal_with_target()[:, :, np.newaxis],   # single channel 3d array
])
def test_non_uint8_and_single_channel_thermal(pipeline, frame):
    """thermal cameras often give 16 bit or float frames instead of 8 bit"""
    assert len(pipeline.process_frames(thermal_frame=frame)) == 1


def test_latency_is_measured(pipeline):
    """every frame gets a real processing time"""
    pipeline.process_frames(thermal_frame=thermal_with_target())
    assert pipeline.last_latency_ms is not None and pipeline.last_latency_ms > 0


def test_detection_is_logged_with_configured_position(pipeline, isolated_db):
    """saved rows hold real values and nothing made up"""
    pipeline.process_frames(thermal_frame=thermal_with_target())

    rows = stored_rows(isolated_db)
    assert len(rows) == 1
    row = rows[0]
    # the unit reports its configured position with no random offset
    assert row["latitude"] == config.default_lat
    assert row["longitude"] == config.default_lon
    # full date and time so detections from different days stay apart
    datetime.fromisoformat(row["timestamp"])
    assert row["confidence"] is None
    assert row["detector"] == detector_name
    assert row["latency_ms"] == pytest.approx(pipeline.last_latency_ms)


def test_logger_upgrades_old_database(isolated_db):
    """databases made before the detector and latency columns still work"""
    with sqlite3.connect(isolated_db) as conn:
        conn.execute("""
            create table detections (
                id integer primary key autoincrement, timestamp text, latitude real,
                longitude real, confidence real, thermal_peak real, bbox text
            )
        """)
        conn.execute("insert into detections (timestamp) values ('12:00:00')")

    # creating the logger should add the missing columns before saving

    data_logger().log_events(
        [{"bbox": [0, 0, 1, 1], "confidence": None, "thermal_peak": 1, "detector": "x"}], 1.0)

    rows = stored_rows(isolated_db)
    assert len(rows) == 2
    assert rows[1]["detector"] == "x"
