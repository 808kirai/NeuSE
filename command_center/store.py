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

# the latest status note from each unit
units_schema = """
    create table if not exists units (
        unit_id text primary key,
        last_status_at text not null,
        queued integer,
        latitude real,
        longitude real,
        detector text,
        frames integer
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
        conn.execute(units_schema)
        conn.execute("create index if not exists detections_time on detections (timestamp_utc)")
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


def status_problem(note):
    # returns why a status note cannot be stored, or None when it is fine
    if not isinstance(note, dict):
        return "status must be an object"
    if not isinstance(note.get("unit_id"), str) or not unit_pattern.match(note["unit_id"]):
        return "unit_id must be 1 to 64 letters, digits, dots, dashes or underscores"
    if not number(note.get("latitude"), -90, 90) or not number(note.get("longitude"), -180, 180):
        return "latitude and longitude must be valid coordinates"
    for name in ("queued", "frames"):
        value = note.get(name)
        if value is not None and not (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
            return f"{name} must be empty or a whole number"
    if note.get("detector") is not None and not (isinstance(note["detector"], str) and len(note["detector"]) <= 64):
        return "detector must be empty or a short name"
    return None


def save_status(path, note):
    # keeps the newest status note per unit, timed by the command center's own clock
    received = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(connect(path)) as conn, conn:
        conn.execute("""
            insert into units (unit_id, last_status_at, queued, latitude, longitude, detector, frames)
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict (unit_id) do update set last_status_at = excluded.last_status_at, queued = excluded.queued,
                latitude = excluded.latitude, longitude = excluded.longitude, detector = excluded.detector,
                frames = excluded.frames
        """, (note["unit_id"], received, note.get("queued"), note["latitude"], note["longitude"],
              note.get("detector"), note.get("frames")))


def units(path):
    # one summary per field unit, from its detections and its latest status note
    with closing(connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        found = {row["unit_id"]: dict(row) for row in conn.execute("""
            select unit_id, count(*) as detections, max(timestamp_utc) as last_detection,
                   max(received_at) as last_received
            from detections group by unit_id
        """)}
        notes = {row["unit_id"]: dict(row) for row in conn.execute("select * from units")}

        summaries = []
        for unit_id in sorted(set(found) | set(notes)):
            seen = found.get(unit_id, {})
            note = notes.get(unit_id, {})
            summary = {
                "unit_id": unit_id,
                "detections": seen.get("detections", 0),
                "last_detection": seen.get("last_detection"),
                # the last time anything arrived from the unit
                "last_contact": max(t for t in (seen.get("last_received"), note.get("last_status_at")) if t),
                "queued": note.get("queued"),
                "detector": note.get("detector"),
                "frames": note.get("frames"),
                "latitude": note.get("latitude"),
                "longitude": note.get("longitude"),
            }
            # without a status note the unit is placed at its newest detection
            if summary["latitude"] is None:
                latest = conn.execute("""
                    select latitude, longitude from detections where unit_id = ?
                    order by timestamp_utc desc limit 1
                """, (unit_id,)).fetchone()
                summary["latitude"], summary["longitude"] = latest["latitude"], latest["longitude"]
            summaries.append(summary)
    # most recently heard from first
    return sorted(summaries, key=lambda u: u["last_contact"], reverse=True)


def recent(path, after=None, limit=1000):
    # detections in the order they arrived, each with a running number seq
    # with after, only those that arrived later, oldest first, so a page can follow along without missing any
    # without after, the latest ones, so a page that just opened starts with the current picture
    # units send their backlog newest first, so arrival order is used here rather than detection time
    with closing(connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        fields = """rowid as seq, uid, unit_id, timestamp, timestamp_utc, latitude, longitude, confidence,
                    thermal_peak, bbox, detector, received_at"""
        if after is None:
            rows = conn.execute(f"select {fields} from detections order by rowid desc limit ?", (limit,)).fetchall()
            rows = rows[::-1]
        else:
            rows = conn.execute(f"select {fields} from detections where rowid > ? order by rowid limit ?",
                                (after, limit)).fetchall()
    detections = []
    for row in rows:
        detection = dict(row)
        detection["bbox"] = json.loads(detection["bbox"]) if detection["bbox"] else None
        detections.append(detection)
    return detections


def count(path):
    with closing(connect(path)) as conn:
        return conn.execute("select count(*) from detections").fetchone()[0]
