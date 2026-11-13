"""
storage for the command center
sqlite for now so it runs anywhere, kept in one place so it can move to huawei cloud rds later
"""
import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

# seq counts detections in the order they arrived, the map follows it to catch up on backlogs
# timestamp is the time the unit gave, timestamp_utc is that time in utc after correcting a wrong unit clock
# clock_offset_s is the correction that was applied, empty when the unit sent no clock reading
# position_source says whether the position is the unit's own or worked out on the ground
schema = """
    create table if not exists detections (
        seq integer primary key autoincrement,
        uid text not null unique,
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
        received_at text not null,
        clock_offset_s real,
        position_source text
    )
"""

# the latest status note from each unit
# detector_idle_s is how long the unit's detector had been quiet when the note was sent
units_schema = """
    create table if not exists units (
        unit_id text primary key,
        last_status_at text not null,
        queued integer,
        latitude real,
        longitude real,
        detector text,
        frames integer,
        refused integer,
        detector_idle_s real
    )
"""

# columns added to the units table later, so older databases get them on startup
new_unit_columns = {"refused": "integer", "detector_idle_s": "real"}

uid_pattern = re.compile(r"^[0-9a-f]{32}$")
unit_pattern = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
position_sources = ("unit", "projected")

# unit clocks within this many seconds of the command center's are taken as right
clock_tolerance = 5.0
# largest whole number accepted for counts and sizes
max_whole = 10 ** 12


def connect(path):
    return sqlite3.connect(path, timeout=10)


def setup(path):
    with closing(connect(path)) as conn, conn:
        conn.execute("pragma journal_mode=wal")

        # the first version had no seq column and relied on sqlite's hidden row number
        # move such a table over, keeping the order its rows arrived in
        columns = [row[1] for row in conn.execute("pragma table_info(detections)")]
        if columns and "seq" not in columns:
            conn.execute("alter table detections rename to detections_old")
            conn.execute("drop index if exists detections_time")
            conn.execute("drop index if exists detections_unit")
            conn.execute(schema)
            conn.execute("""
                insert into detections (uid, unit_id, timestamp, timestamp_utc, latitude, longitude, confidence,
                                        thermal_peak, bbox, detector, latency_ms, received_at)
                select uid, unit_id, timestamp, timestamp_utc, latitude, longitude, confidence, thermal_peak, bbox,
                       detector, latency_ms, received_at
                from detections_old order by rowid
            """)
            conn.execute("drop table detections_old")

        conn.execute(schema)
        conn.execute(units_schema)
        existing = {row[1] for row in conn.execute("pragma table_info(units)")}
        for name, kind in new_unit_columns.items():
            if name not in existing:
                conn.execute(f"alter table units add column {name} {kind}")
        conn.execute("create index if not exists detections_time on detections (timestamp_utc)")
        conn.execute("create index if not exists detections_unit on detections (unit_id, timestamp_utc)")


def number(value, low=-math.inf, high=math.inf):
    # true for a real finite number inside the range, json true and false do not count
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    # a whole number with hundreds of digits cannot be turned into a float to check it
    if isinstance(value, int) and abs(value) > 10 ** 300:
        return False
    return math.isfinite(value) and low <= value <= high


def whole(value):
    # true for a count that fits in the database
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= max_whole


def parse_time(text):
    # an iso time with a utc offset as a moment in utc, or None when it is not one
    try:
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            return None
        return moment.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        # overflow covers times at the very start or end of the calendar
        return None


def problem(d):
    # returns why a detection cannot be stored, or None when it is fine
    if not isinstance(d, dict):
        return "not an object"
    if not isinstance(d.get("uid"), str) or not uid_pattern.match(d["uid"]):
        return "uid must be 32 lowercase hex characters"
    if not isinstance(d.get("unit_id"), str) or not unit_pattern.match(d["unit_id"]):
        return "unit_id must be 1 to 64 letters, digits, dots, dashes or underscores"
    if parse_time(d.get("timestamp")) is None:
        return "timestamp must be an iso date and time with a utc offset"
    if not number(d.get("latitude"), -90, 90) or not number(d.get("longitude"), -180, 180):
        return "latitude and longitude must be valid coordinates"
    if d.get("confidence") is not None and not number(d["confidence"], 0, 1):
        return "confidence must be empty or between 0 and 1"
    for name in ("thermal_peak", "latency_ms"):
        if d.get(name) is not None and not number(d[name], 0, 1e12):
            return f"{name} must be empty or a positive number"
    box = d.get("bbox")
    if box is not None and not (isinstance(box, list) and len(box) == 4 and all(number(v, -1e9, 1e9) for v in box)):
        return "bbox must be empty or four numbers"
    if d.get("detector") is not None and not (isinstance(d["detector"], str) and len(d["detector"]) <= 64):
        return "detector must be empty or a short name"
    if d.get("position_source") is not None and d["position_source"] not in position_sources:
        return "position_source must be empty, unit or projected"
    return None


def save(path, detections, sent_at=None, only_unit=None, now=None):
    # stores each detection once and returns the ids taken and the ids refused with the reason
    # an id already stored counts as taken, so a unit that never heard back can safely send it again
    # sent_at is the unit's clock when it sent the batch, as a moment in utc
    # the gap between it and this computer's clock corrects the times from a unit whose clock is wrong
    # only_unit limits the batch to one unit's name, for a token that belongs to that unit
    now = now or datetime.now(timezone.utc)
    received = now.isoformat(timespec="seconds")
    offset = (now - sent_at).total_seconds() if sent_at is not None else None

    accepted, rejected, rows = [], {}, []
    for d in detections:
        why = problem(d)
        if not why and only_unit is not None and d["unit_id"] != only_unit:
            why = f"this token only sends for {only_unit}"
        utc = None
        if not why:
            utc = parse_time(d["timestamp"])
            if offset is not None and abs(offset) > clock_tolerance:
                try:
                    # a detection cannot be from after the batch was sent
                    utc = min(utc + timedelta(seconds=offset), now)
                except OverflowError:
                    why = "timestamp is out of range"
        if why:
            # refuse only what can be named, an item with no usable uid is just dropped
            if isinstance(d, dict) and isinstance(d.get("uid"), str) and len(d["uid"]) <= 64:
                rejected[d["uid"]] = why
            continue
        accepted.append(d["uid"])
        rows.append((d["uid"], d["unit_id"], d["timestamp"], utc.isoformat(timespec="seconds"), d["latitude"],
                     d["longitude"], d.get("confidence"), d.get("thermal_peak"), json.dumps(d.get("bbox")),
                     d.get("detector"), d.get("latency_ms"), received, offset, d.get("position_source") or "unit"))

    with closing(connect(path)) as conn, conn:
        conn.executemany("""
            insert or ignore into detections
                (uid, unit_id, timestamp, timestamp_utc, latitude, longitude, confidence, thermal_peak, bbox, detector,
                 latency_ms, received_at, clock_offset_s, position_source)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    for name in ("queued", "frames", "refused"):
        if note.get(name) is not None and not whole(note[name]):
            return f"{name} must be empty or a whole number"
    if note.get("detector_idle_s") is not None and not number(note["detector_idle_s"], 0, 1e12):
        return "detector_idle_s must be empty or a positive number"
    if note.get("detector") is not None and not (isinstance(note["detector"], str) and len(note["detector"]) <= 64):
        return "detector must be empty or a short name"
    return None


def save_status(path, note):
    # keeps the newest status note per unit, timed by the command center's own clock
    received = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(connect(path)) as conn, conn:
        conn.execute("""
            insert into units (unit_id, last_status_at, queued, latitude, longitude, detector, frames, refused,
                               detector_idle_s)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict (unit_id) do update set last_status_at = excluded.last_status_at, queued = excluded.queued,
                latitude = excluded.latitude, longitude = excluded.longitude, detector = excluded.detector,
                frames = excluded.frames, refused = excluded.refused, detector_idle_s = excluded.detector_idle_s
        """, (note["unit_id"], received, note.get("queued"), note["latitude"], note["longitude"],
              note.get("detector"), note.get("frames"), note.get("refused"), note.get("detector_idle_s")))


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
                "last_status": note.get("last_status_at"),
                "queued": note.get("queued"),
                "refused": note.get("refused"),
                "detector": note.get("detector"),
                "frames": note.get("frames"),
                "detector_idle_s": note.get("detector_idle_s"),
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


def unit_state(unit, now, online_window, detecting_window):
    # whether a unit is in contact and whether its detector is still handling frames
    # a unit can keep syncing after its camera or detector has stopped, so the two are told apart
    contact = parse_time(unit["last_contact"])
    online = contact is not None and (now - contact).total_seconds() <= online_window
    idle = None
    status = parse_time(unit.get("last_status"))
    if unit.get("detector_idle_s") is not None and status is not None:
        # how quiet the detector was when the note was sent, plus the time since the note
        idle = unit["detector_idle_s"] + max(0.0, (now - status).total_seconds())
    return {
        "online": online,
        # None when the unit has never said, older units do not send it
        "detecting": None if idle is None else (online and idle <= detecting_window),
        "detector_idle_s": None if idle is None else round(idle),
    }


def recent(path, after=None, limit=1000):
    # detections in the order they arrived, each with its running number seq
    # with after, only those that arrived later, oldest first, so a page can follow along without missing any
    # without after, the latest ones, so a page that just opened starts with the current picture
    # units send their backlog newest first, so arrival order is used here rather than detection time
    with closing(connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        fields = """seq, uid, unit_id, timestamp, timestamp_utc, latitude, longitude, confidence,
                    thermal_peak, bbox, detector, received_at, clock_offset_s, position_source"""
        if after is None:
            rows = conn.execute(f"select {fields} from detections order by seq desc limit ?", (limit,)).fetchall()
            rows = rows[::-1]
        else:
            rows = conn.execute(f"select {fields} from detections where seq > ? order by seq limit ?",
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
