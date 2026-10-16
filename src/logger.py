import sqlite3
from contextlib import closing
from datetime import datetime
from src.config import config

# one row per detection
schema = """
    create table if not exists detections (
        id integer primary key autoincrement,
        timestamp text,
        latitude real,
        longitude real,
        confidence real,
        thermal_peak real,
        bbox text,
        detector text,
        latency_ms real
    )
"""

# columns added later, so older databases get them on startup
new_columns = {"detector": "text", "latency_ms": "real"}


def connect():
    # open the shared database
    return sqlite3.connect(config.db_path)


class data_logger:
    def __init__(self):
        # create the table or upgrade an older one
        with closing(connect()) as conn, conn:
            conn.execute(schema)
            existing = {row[1] for row in conn.execute("pragma table_info(detections)")}
            for name, kind in new_columns.items():
                if name not in existing:
                    conn.execute(f"alter table detections add column {name} {kind}")

    def log_events(self, hits, latency_ms):
        # save all detections from one frame in one go
        ts = datetime.now().isoformat(timespec="seconds")
        rows = [
            (ts, config.default_lat, config.default_lon, hit["confidence"],
             hit["thermal_peak"], str(hit["bbox"]), hit["detector"], latency_ms)
            for hit in hits
        ]

        with closing(connect()) as conn, conn:
            conn.executemany("""
                insert into detections
                    (timestamp, latitude, longitude, confidence, thermal_peak, bbox, detector, latency_ms)
                values (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
