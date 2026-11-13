import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from src.config import config

# one row per detection
# uid stays the same everywhere, so the command center can tell a resent detection from a new one
# synced_at stays empty until the command center confirms it has the detection
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
        latency_ms real,
        uid text,
        unit_id text,
        synced_at text,
        sync_error text
    )
"""

# columns added later, so older databases get them on startup
new_columns = {
    "detector": "text",
    "latency_ms": "real",
    "uid": "text",
    "unit_id": "text",
    "synced_at": "text",
    "sync_error": "text",
}


def connect():
    # open the shared database
    # callers wrap it in closing so the file is released, which matters on windows
    # the pipeline and the sync process write at the same time, so wait for the other instead of failing
    return sqlite3.connect(config.db_path, timeout=10)


def now():
    # local time with its utc offset so logs from units in other time zones line up
    return datetime.now().astimezone().isoformat(timespec="seconds")


# saves detections to the local database so nothing is lost while offline
class data_logger:
    def __init__(self):
        # create the table or upgrade an older one
        # closing releases the file and the inner with commits the changes
        with closing(connect()) as conn, conn:
            # write ahead logging lets the dashboard and sync read while the pipeline writes
            conn.execute("pragma journal_mode=wal")
            conn.execute(schema)

            # add any column an older table is missing
            existing = {row[1] for row in conn.execute("pragma table_info(detections)")}
            for name, kind in new_columns.items():
                if name not in existing:
                    conn.execute(f"alter table detections add column {name} {kind}")

            # older rows get an id and this unit's name so they can be sent too
            for (row_id,) in conn.execute("select id from detections where uid is null").fetchall():
                conn.execute("update detections set uid = ? where id = ?", (uuid.uuid4().hex, row_id))
            conn.execute("update detections set unit_id = ? where unit_id is null", (config.unit_id,))
            conn.execute("create unique index if not exists detections_uid on detections (uid)")
            # finding unsent rows stays quick as the log grows
            conn.execute("create index if not exists detections_unsent on detections (synced_at, id)")

    def log_events(self, hits, latency_ms):
        # save all detections from one frame in one go
        # every detection in a frame shares the same time, position and processing time
        ts = now()
        rows = [
            (ts, config.default_lat, config.default_lon, hit["confidence"],
             hit["thermal_peak"], str(hit["bbox"]), hit["detector"], latency_ms,
             uuid.uuid4().hex, config.unit_id)
            for hit in hits
        ]

        with closing(connect()) as conn, conn:
            conn.executemany("""
                insert into detections
                    (timestamp, latitude, longitude, confidence, thermal_peak, bbox, detector, latency_ms, uid, unit_id)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
