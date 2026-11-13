"""
storage for the command center
sqlite for now so it runs anywhere, kept in one place so it can move to huawei cloud rds later
"""
import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

schema = """
    create table if not exists detections (
        uid text primary key,
        unit_id text not null,
        timestamp text not null,
        timestamp_utc text not null,
        latitude real not null,
        longitude real not null,
        confidence real,
        thermal_peak real,
        bbox text,
        detector text,
        latency_ms real,
        received_at text not null
    )
"""

uid_pattern = re.compile(r"^[0-9a-f]{32}$")
unit_pattern = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def connect(path):
    return sqlite3.connect(path, timeout=10)


def setup(path):
    with closing(connect(path)) as conn, conn:
        conn.execute("pragma journal_mode=wal")
        conn.execute(schema)
        conn.execute("create index if not exists detections_unit on detections (unit_id, timestamp_utc)")


def number(value, low=-math.inf, high=math.inf):
    # true for a real finite number inside the range, json true and false do not count
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and low <= value <= high)


def problem(d):
    # returns why a detection cannot be stored, or None when it is fine
    if not isinstance(d, dict):
        return "not an object"
    if not isinstance(d.get("uid"), str) or not uid_pattern.match(d["uid"]):
        return "uid must be 32 lowercase hex characters"
    if not isinstance(d.get("unit_id"), str) or not unit_pattern.match(d["unit_id"]):
        return "unit_id must be 1 to 64 letters, digits, dots, dashes or underscores"
    try:
        if datetime.fromisoformat(d.get("timestamp")).tzinfo is None:
            return "timestamp needs a utc offset"
    except (TypeError, ValueError):
        return "timestamp must be an iso date and time"
    if not number(d.get("latitude"), -90, 90) or not number(d.get("longitude"), -180, 180):
        return "latitude and longitude must be valid coordinates"
    if d.get("confidence") is not None and not number(d["confidence"], 0, 1):
        return "confidence must be empty or between 0 and 1"
    for name in ("thermal_peak", "latency_ms"):
        if d.get(name) is not None and not number(d[name], 0):
            return f"{name} must be empty or a positive number"
    box = d.get("bbox")
    if box is not None and not (isinstance(box, list) and len(box) == 4 and all(number(v) for v in box)):
        return "bbox must be empty or four numbers"
    if d.get("detector") is not None and not (isinstance(d["detector"], str) and len(d["detector"]) <= 64):
        return "detector must be empty or a short name"
    return None


def save(path, detections):
    # stores each detection once and returns the ids taken and the ids refused with the reason
    # an id already stored counts as taken, so a unit that never heard back can safely send it again
    received = datetime.now(timezone.utc).isoformat(timespec="seconds")
    accepted, rejected, rows = [], {}, []
    for d in detections:
        why = problem(d)
        if why:
            # refuse only what can be named, an item with no usable uid is just dropped
            if isinstance(d, dict) and isinstance(d.get("uid"), str) and len(d["uid"]) <= 64:
                rejected[d["uid"]] = why
            continue
        accepted.append(d["uid"])
        # units can sit in different time zones, so times are also kept in utc for sorting
        utc = datetime.fromisoformat(d["timestamp"]).astimezone(timezone.utc).isoformat()
        rows.append((d["uid"], d["unit_id"], d["timestamp"], utc, d["latitude"], d["longitude"], d.get("confidence"),
                     d.get("thermal_peak"), json.dumps(d.get("bbox")), d.get("detector"), d.get("latency_ms"), received))

    with closing(connect(path)) as conn, conn:
        conn.executemany("""
            insert or ignore into detections
                (uid, unit_id, timestamp, timestamp_utc, latitude, longitude, confidence, thermal_peak, bbox, detector,
                 latency_ms, received_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    return accepted, rejected


def units(path):
    # one summary per field unit, newest activity first
    with closing(connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            select unit_id, count(*) as detections, max(timestamp_utc) as last_detection,
                   max(received_at) as last_received
            from detections group by unit_id order by last_received desc
        """).fetchall()
        summaries = []
        for row in rows:
            latest = conn.execute("""
                select latitude, longitude from detections where unit_id = ?
                order by timestamp_utc desc limit 1
            """, (row["unit_id"],)).fetchone()
            summaries.append({**dict(row), "latitude": latest["latitude"], "longitude": latest["longitude"]})
    return summaries


def count(path):
    with closing(connect(path)) as conn:
        return conn.execute("select count(*) from detections").fetchone()[0]
