"""
storage for the command center
works with a local sqlite file, or a mysql database such as huawei cloud rds for mysql

every function takes db, which is either a path to an sqlite file
or an address like mysql://user:password@host:3306/database
characters such as @ or : in the password must be written with % codes, for example @ as %40
"""
import json
import math
import re
import sqlite3
import urllib.parse
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

# the same tables for mysql, which needs sizes on text that is indexed
# times are kept as iso text in utc, so they sort the same way in both databases
mysql_schema = """
    create table if not exists detections (
        seq bigint not null auto_increment primary key,
        uid char(32) not null unique,
        unit_id varchar(64) not null,
        timestamp varchar(40) not null,
        timestamp_utc varchar(32) not null,
        latitude double not null,
        longitude double not null,
        confidence double,
        thermal_peak double,
        bbox varchar(255),
        detector varchar(64),
        latency_ms double,
        received_at varchar(32) not null,
        clock_offset_s double,
        position_source varchar(16),
        index detections_time (timestamp_utc),
        index detections_unit (unit_id, timestamp_utc),
        index detections_received (received_at)
    ) engine=innodb default charset=utf8mb4
"""

mysql_units_schema = """
    create table if not exists units (
        unit_id varchar(64) not null primary key,
        last_status_at varchar(32) not null,
        queued bigint,
        latitude double,
        longitude double,
        detector varchar(64),
        frames bigint,
        refused bigint,
        detector_idle_s double
    ) engine=innodb default charset=utf8mb4
"""

# ground squares each unit's camera has covered, about 22 m across, see src/geo.py on the unit
coverage_schema = """
    create table if not exists coverage (
        unit_id text not null,
        cell_row integer not null,
        cell_col integer not null,
        last_seen text not null,
        primary key (unit_id, cell_row, cell_col)
    )
"""

mysql_coverage_schema = """
    create table if not exists coverage (
        unit_id varchar(64) not null,
        cell_row int not null,
        cell_col int not null,
        last_seen varchar(32) not null,
        primary key (unit_id, cell_row, cell_col),
        index coverage_seen (last_seen)
    ) engine=innodb default charset=utf8mb4
"""

# alerts raised for coordinators, kept whether or not a text was sent, see command_center/alerts.py
alerts_schema = """
    create table if not exists alerts (
        id integer primary key autoincrement,
        created_at text not null,
        latitude real not null,
        longitude real not null,
        sightings integer not null,
        units text not null,
        highest_confidence real,
        text text not null,
        sent_via text,
        delivered integer not null default 0,
        error text
    )
"""

mysql_alerts_schema = """
    create table if not exists alerts (
        id bigint not null auto_increment primary key,
        created_at varchar(32) not null,
        latitude double not null,
        longitude double not null,
        sightings int not null,
        units varchar(1024) not null,
        highest_confidence double,
        text varchar(600) not null,
        sent_via varchar(16),
        delivered tinyint not null default 0,
        error varchar(600),
        index alerts_time (created_at)
    ) engine=innodb default charset=utf8mb4
"""

# size of a coverage square in degrees, the same as src/geo.py on the unit
cell_degrees = 0.0002
# most squares one request may carry
max_cells = 5000

# columns added to the units table later, so older sqlite databases get them on startup
new_unit_columns = {"refused": "integer", "detector_idle_s": "real"}

uid_pattern = re.compile(r"^[0-9a-f]{32}$")
unit_pattern = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
position_sources = ("unit", "projected")

# unit clocks within this many seconds of the command center's are taken as right
clock_tolerance = 5.0
# largest whole number accepted for counts and sizes
max_whole = 10 ** 12


def is_mysql(db):
    return isinstance(db, str) and db.startswith("mysql://")


def connect(path):
    # a plain connection to an sqlite file
    return sqlite3.connect(path, timeout=10)


def mysql_settings(address):
    # reads mysql://user:password@host:port/database into connection settings
    parts = urllib.parse.urlsplit(address)
    database = parts.path.lstrip("/")
    if not parts.hostname or not parts.username or not database:
        raise ValueError("the database address needs the form mysql://user:password@host:3306/database")
    return {
        "host": parts.hostname,
        "port": parts.port or 3306,
        "user": urllib.parse.unquote(parts.username),
        "password": urllib.parse.unquote(parts.password or ""),
        "database": urllib.parse.unquote(database),
    }


class session:
    # one connection, committed when the block ends and rolled back if it fails
    # the sql is written with ? marks, and mysql gets them as %s
    def __init__(self, db):
        self.db = db
        self.mysql = is_mysql(db)

    def __enter__(self):
        if self.mysql:
            import pymysql
            self.conn = pymysql.connect(**mysql_settings(self.db), charset="utf8mb4", autocommit=False,
                                        connect_timeout=10, cursorclass=pymysql.cursors.DictCursor)
        else:
            self.conn = connect(self.db)
            self.conn.row_factory = sqlite3.Row
        return self

    def __exit__(self, kind, error, trace):
        try:
            if kind is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()

    def sql(self, text):
        return text.replace("?", "%s") if self.mysql else text

    def run(self, text, params=()):
        cursor = self.conn.cursor()
        cursor.execute(self.sql(text), params)
        return cursor

    def many(self, text, rows):
        if rows:
            self.conn.cursor().executemany(self.sql(text), rows)

    def rows(self, text, params=()):
        return [dict(row) for row in self.run(text, params).fetchall()]

    def one(self, text, params=()):
        found = self.rows(text, params)
        return found[0] if found else None


def setup(db):
    # creates the tables, and upgrades an older sqlite database
    if is_mysql(db):
        with session(db) as s:
            s.run(mysql_schema)
            s.run(mysql_units_schema)
            s.run(mysql_coverage_schema)
            s.run(mysql_alerts_schema)
            # alerts look sightings up by arrival, a database made before that index needs it added
            have = s.rows("""
                select count(*) as n from information_schema.statistics
                where table_schema = database() and table_name = 'detections' and index_name = 'detections_received'
            """)
            if not have[0]["n"]:
                s.run("create index detections_received on detections (received_at)")
        return

    with closing(connect(db)) as conn:
        conn.execute("pragma journal_mode=wal")

        # every change below happens in one transaction, so a stop part way leaves the database as it was
        # python's sqlite3 would otherwise commit each table change on its own
        conn.isolation_level = None
        conn.execute("begin immediate")
        try:
            tables = {row[0] for row in conn.execute("select name from sqlite_master where type = 'table'")}
            columns = [row[1] for row in conn.execute("pragma table_info(detections)")]

            # the first version had no seq column and relied on sqlite's hidden row number
            # move such a table over, keeping the order its rows arrived in
            if columns and "seq" not in columns:
                conn.execute("alter table detections rename to detections_old")
                conn.execute("drop index if exists detections_time")
                conn.execute("drop index if exists detections_unit")
                tables.add("detections_old")

            conn.execute(schema)
            # also picks up rows left behind by an earlier move that was stopped part way
            if "detections_old" in tables:
                conn.execute("""
                    insert or ignore into detections (uid, unit_id, timestamp, timestamp_utc, latitude, longitude,
                                                      confidence, thermal_peak, bbox, detector, latency_ms, received_at)
                    select uid, unit_id, timestamp, timestamp_utc, latitude, longitude, confidence, thermal_peak, bbox,
                           detector, latency_ms, received_at
                    from detections_old order by rowid
                """)
                conn.execute("drop table detections_old")

            conn.execute(units_schema)
            conn.execute(coverage_schema)
            conn.execute(alerts_schema)
            conn.execute("create index if not exists coverage_seen on coverage (last_seen)")
            existing = {row[1] for row in conn.execute("pragma table_info(units)")}
            for name, kind in new_unit_columns.items():
                if name not in existing:
                    conn.execute(f"alter table units add column {name} {kind}")
            conn.execute("create index if not exists detections_time on detections (timestamp_utc)")
            conn.execute("create index if not exists detections_unit on detections (unit_id, timestamp_utc)")
            conn.execute("create index if not exists detections_received on detections (received_at)")
            conn.execute("commit")
        except BaseException:
            conn.execute("rollback")
            raise


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
    if not isinstance(d.get("timestamp"), str) or len(d["timestamp"]) > 40 or parse_time(d["timestamp"]) is None:
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


def save(db, detections, sent_at=None, only_unit=None, now=None):
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

    with session(db) as s:
        # a uid already stored is skipped, which is what makes resending safe
        ignore = "insert ignore into" if s.mysql else "insert or ignore into"
        s.many(f"""
            {ignore} detections
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


def save_status(db, note):
    # keeps the newest status note per unit, timed by the command center's own clock
    received = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fields = ("last_status_at", "queued", "latitude", "longitude", "detector", "frames", "refused", "detector_idle_s")
    with session(db) as s:
        if s.mysql:
            update = "on duplicate key update " + ", ".join(f"{f} = values({f})" for f in fields)
        else:
            update = "on conflict (unit_id) do update set " + ", ".join(f"{f} = excluded.{f}" for f in fields)
        s.run(f"""
            insert into units (unit_id, {", ".join(fields)})
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            {update}
        """, (note["unit_id"], received, note.get("queued"), note["latitude"], note["longitude"],
              note.get("detector"), note.get("frames"), note.get("refused"), note.get("detector_idle_s")))


def units(db):
    # one summary per field unit, from its detections and its latest status note
    with session(db) as s:
        found = {row["unit_id"]: row for row in s.rows("""
            select unit_id, count(*) as detections, max(timestamp_utc) as last_detection,
                   max(received_at) as last_received
            from detections group by unit_id
        """)}
        notes = {row["unit_id"]: row for row in s.rows("select * from units")}

        summaries = []
        for unit_id in sorted(set(found) | set(notes)):
            seen = found.get(unit_id, {})
            note = notes.get(unit_id, {})
            summary = {
                "unit_id": unit_id,
                "detections": int(seen.get("detections", 0)),
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
                latest = s.one("""
                    select latitude, longitude from detections where unit_id = ?
                    order by timestamp_utc desc limit 1
                """, (unit_id,))
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


def recent(db, after=None, limit=1000):
    # detections in the order they arrived, each with its running number seq
    # with after, only those that arrived later, oldest first, so a page can follow along without missing any
    # without after, the latest ones, so a page that just opened starts with the current picture
    # units send their backlog newest first, so arrival order is used here rather than detection time
    fields = """seq, uid, unit_id, timestamp, timestamp_utc, latitude, longitude, confidence,
                thermal_peak, bbox, detector, received_at, clock_offset_s, position_source"""
    with session(db) as s:
        if after is None:
            rows = s.rows(f"select {fields} from detections order by seq desc limit ?", (limit,))[::-1]
        else:
            rows = s.rows(f"select {fields} from detections where seq > ? order by seq limit ?", (after, limit))
    for row in rows:
        row["bbox"] = json.loads(row["bbox"]) if row["bbox"] else None
    return rows


def sightings_between(db, start, end):
    # every detection seen from start to end, both iso times in utc, for situation reports
    with session(db) as s:
        return s.rows("""
            select unit_id, timestamp_utc, latitude, longitude, confidence, position_source
            from detections where timestamp_utc >= ? and timestamp_utc <= ?
        """, (start, end))


def sightings_arrived(db, since, seen_after):
    # detections that reached the command center since a time and were seen after another, for alerts
    # a unit that was out of range sends its backlog late, and those sightings still need an alert
    with session(db) as s:
        return s.rows("""
            select unit_id, timestamp_utc, latitude, longitude, confidence
            from detections where received_at >= ? and timestamp_utc >= ?
        """, (since, seen_after))


def coverage_problem(payload):
    # returns why a batch of covered squares cannot be stored, or None when it is fine
    if not isinstance(payload, dict):
        return "coverage must be an object"
    if not isinstance(payload.get("unit_id"), str) or not unit_pattern.match(payload["unit_id"]):
        return "unit_id must be 1 to 64 letters, digits, dots, dashes or underscores"
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) > max_cells:
        return f"cells must be a list of at most {max_cells}"
    # a square's row and column are latitude and longitude divided by the square size
    rows, cols = int(90 / cell_degrees), int(180 / cell_degrees)
    for cell in cells:
        if not (isinstance(cell, list) and len(cell) == 3 and whole_signed(cell[0], rows)
                and whole_signed(cell[1], cols) and isinstance(cell[2], str) and parse_time(cell[2]) is not None):
            return "each cell must be [row, column, last seen time with a utc offset]"
    return None


def whole_signed(value, limit):
    return isinstance(value, int) and not isinstance(value, bool) and -limit <= value <= limit


def corrected(moment, sent_at, now):
    # a time from the unit moved by the gap between its clock and this computer's, never later than now
    if sent_at is None:
        return moment
    offset = (now - sent_at).total_seconds()
    if abs(offset) <= clock_tolerance:
        return moment
    try:
        return min(moment + timedelta(seconds=offset), now)
    except OverflowError:
        return now


def save_coverage(db, unit_id, cells, sent_at=None, now=None):
    # keeps the latest time each unit covered each square, an older report never replaces a newer one
    # sent_at is the unit's clock when it sent them, used to correct a unit whose clock is wrong
    now = now or datetime.now(timezone.utc)
    rows = [(unit_id, row, col, corrected(parse_time(seen), sent_at, now).isoformat(timespec="seconds"))
            for row, col, seen in cells]
    with session(db) as s:
        if s.mysql:
            s.many("""
                insert into coverage (unit_id, cell_row, cell_col, last_seen) values (?, ?, ?, ?)
                on duplicate key update last_seen = greatest(last_seen, values(last_seen))
            """, rows)
        else:
            s.many("""
                insert into coverage (unit_id, cell_row, cell_col, last_seen) values (?, ?, ?, ?)
                on conflict (unit_id, cell_row, cell_col) do update set last_seen = max(last_seen, excluded.last_seen)
            """, rows)
    return len(rows)


def coverage(db, since=None):
    # covered squares across all units, each with the latest time any unit covered it and how many units did
    with session(db) as s:
        return s.rows("""
            select cell_row, cell_col, max(last_seen) as last_seen, count(*) as units
            from coverage where last_seen >= ?
            group by cell_row, cell_col
        """, (since or "",))


def save_alert(db, alert):
    with session(db) as s:
        s.run("""
            insert into alerts (created_at, latitude, longitude, sightings, units, highest_confidence, text, sent_via,
                                delivered, error)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (alert["created_at"], alert["latitude"], alert["longitude"], alert["sightings"],
              json.dumps(alert["units"]), alert["highest_confidence"], alert["text"], alert["sent_via"],
              alert["delivered"], alert["error"]))


def alerts(db, since=None, limit=200):
    # alerts raised since a time, newest first
    with session(db) as s:
        rows = s.rows("""
            select id, created_at, latitude, longitude, sightings, units, highest_confidence, text, sent_via,
                   delivered, error
            from alerts where created_at >= ? order by id desc limit ?
        """, (since or "", limit))
    for row in rows:
        row["units"] = json.loads(row["units"])
        row["delivered"] = bool(row["delivered"])
    return rows


def count(db):
    with session(db) as s:
        return int(s.one("select count(*) as n from detections")["n"])
