import logging
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta
from src.config import check_unit_id, config

log = logging.getLogger(__name__)

# one row per detection
# uid stays the same everywhere, so the command center can tell a resent detection from a new one
# synced_at stays empty until the command center confirms it has the detection
# sync_error holds the reason when the command center refused it, such a row waits until it is retried by hand
# position_source says whether latitude and longitude are the unit's own position or worked out on the ground
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
        sync_error text,
        position_source text
    )
"""

# the unit's latest state in a single row, read by sync to tell the command center the unit is alive
status_schema = """
    create table if not exists unit_status (
        id integer primary key check (id = 1),
        updated_at text,
        latitude real,
        longitude real,
        detector text,
        frames integer,
        avg_latency_ms real
    )
"""

# the ground squares this unit's camera has covered, about 22 m across, see src/geo.py
# synced is 0 until the command center has the square's latest time
coverage_schema = """
    create table if not exists coverage (
        cell_row integer,
        cell_col integer,
        last_seen text,
        frames integer,
        synced integer default 0,
        primary key (cell_row, cell_col)
    )
"""

# columns added later, so older databases get them on startup
new_columns = {
    "detections": {
        "detector": "text",
        "latency_ms": "real",
        "uid": "text",
        "unit_id": "text",
        "synced_at": "text",
        "sync_error": "text",
        "position_source": "text",
    },
    "unit_status": {
        "avg_latency_ms": "real",
    },
}

# so the synced folder warning shows once
warned_about_folder = False


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
        # a name the command center would refuse is stopped here, before anything is logged under it
        check_unit_id(config.unit_id)

        # file syncing tools can lock or copy a database in the middle of a write
        global warned_about_folder
        if "onedrive" in str(config.db_path).lower() and not warned_about_folder:
            log.warning("the database at %s is inside a onedrive folder, set NEUSE_DB to a folder that is not synced",
                        config.db_path)
            warned_about_folder = True

        # create the tables or upgrade older ones
        # closing releases the file and the inner with commits the changes
        with closing(connect()) as conn, conn:
            # write ahead logging lets the dashboard and sync read while the pipeline writes
            conn.execute("pragma journal_mode=wal")
            conn.execute(schema)
            conn.execute(status_schema)
            conn.execute(coverage_schema)

            # add any column an older table is missing
            for table, columns in new_columns.items():
                existing = {row[1] for row in conn.execute(f"pragma table_info({table})")}
                for name, kind in columns.items():
                    if name not in existing:
                        conn.execute(f"alter table {table} add column {name} {kind}")

            # the first version marked a refused detection as sent, with the reason next to it
            # so it was never counted or tried again, mark those as refused instead so they show up
            # newer versions never set both, so this only touches rows from before the change
            conn.execute("update detections set synced_at = null where synced_at is not null and sync_error is not null")

            # older rows get an id and this unit's name so they can be sent too
            for (row_id,) in conn.execute("select id from detections where uid is null").fetchall():
                conn.execute("update detections set uid = ? where id = ?", (uuid.uuid4().hex, row_id))
            conn.execute("update detections set unit_id = ? where unit_id is null", (config.unit_id,))
            conn.execute("create unique index if not exists detections_uid on detections (uid)")
            # finding unsent rows stays quick as the log grows
            conn.execute("create index if not exists detections_unsent on detections (synced_at, id)")

    def log_events(self, hits, latency_ms, position=None, when=None):
        # save all detections from one frame in one go
        # every detection in a frame shares the same time and processing time
        # position is the unit's latitude and longitude when the frame was taken, the configured one if not given
        # a hit can carry its own position worked out on the ground, otherwise it is placed at the unit
        # when is the frame's time, given when the write was held back by a database error
        ts = when or now()
        unit_position = position or (config.default_lat, config.default_lon)
        rows = []
        for hit in hits:
            lat, lon = hit.get("position") or unit_position
            rows.append((ts, lat, lon, hit["confidence"], hit["thermal_peak"], str(hit["bbox"]), hit["detector"],
                         latency_ms, uuid.uuid4().hex, config.unit_id, hit.get("position_source", "unit")))

        with closing(connect()) as conn, conn:
            conn.executemany("""
                insert into detections
                    (timestamp, latitude, longitude, confidence, thermal_peak, bbox, detector, latency_ms, uid, unit_id,
                     position_source)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

    def note_status(self, position, detector, frames, avg_latency_ms=None):
        # overwrite the unit's single status row
        # avg_latency_ms is the mean time per frame over recent frames, with or without detections
        lat, lon = position or (config.default_lat, config.default_lon)
        with closing(connect()) as conn, conn:
            conn.execute("""
                insert into unit_status (id, updated_at, latitude, longitude, detector, frames, avg_latency_ms)
                values (1, ?, ?, ?, ?, ?, ?)
                on conflict (id) do update set updated_at = excluded.updated_at, latitude = excluded.latitude,
                    longitude = excluded.longitude, detector = excluded.detector, frames = excluded.frames,
                    avg_latency_ms = excluded.avg_latency_ms
            """, (now(), lat, lon, detector, frames, avg_latency_ms))


def note_coverage(cells):
    # saves ground squares the camera covered, cells maps (row, col) to (last seen time, frames)
    # a square seen again is marked unsent so its newer time reaches the command center
    with closing(connect()) as conn, conn:
        conn.executemany("""
            insert into coverage (cell_row, cell_col, last_seen, frames, synced) values (?, ?, ?, ?, 0)
            on conflict (cell_row, cell_col) do update set last_seen = excluded.last_seen,
                frames = coverage.frames + excluded.frames, synced = 0
        """, [(row, col, seen, frames) for (row, col), (seen, frames) in cells.items()])


def read_status():
    # the unit's latest status as a dict, or None before the pipeline has run
    with closing(connect()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "select updated_at, latitude, longitude, detector, frames, avg_latency_ms from unit_status").fetchone()
    return dict(row) if row else None


def prune(days):
    # removes detections the command center confirmed more than days ago and returns how many went
    # rows still waiting or refused are never removed
    if not days or days <= 0:
        return 0
    cutoff = (datetime.now().astimezone() - timedelta(days=days)).isoformat(timespec="seconds")
    with closing(connect()) as conn, conn:
        # compared as real moments, since the unit's utc offset can change with the seasons
        removed = conn.execute("""
            delete from detections
            where synced_at is not null and sync_error is null and julianday(synced_at) < julianday(?)
        """, (cutoff,)).rowcount
    return removed
