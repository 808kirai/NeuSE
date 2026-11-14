"""
command center that field units sync their detections to
units post detections and short status notes, and coordinators watch a live map of units and detections

who may do what
    NEUSE_CC_TOKEN is the coordinator token, it can read everything and also send
    NEUSE_CC_UNIT_TOKENS gives each unit its own token, like unit-1=abc,unit-2=def
    a unit token can only send, and only under its own unit's name, so a lost unit cannot read where people are
    with no tokens set the command center only answers the computer it runs on

the database is an sqlite file by default
set NEUSE_CC_DATABASE_URL=mysql://user:password@host:3306/database to use rds for mysql instead

usage
    python -m command_center.app
    NEUSE_CC_TOKEN=some-secret NEUSE_CC_HOST=0.0.0.0 python -m command_center.app
    see docs/DEPLOY.md to run it on huawei cloud
"""
import hmac
import json
import os
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from command_center import alerts, reports, store

project_root = Path(__file__).resolve().parent.parent

# most detections one request may carry
max_batch = 500
# most detections the map asks for at once
max_recent = 2000
# limits on the request body, before and after unzipping, so a bad upload cannot fill the memory
max_upload = 2 * 1024 * 1024
max_unzipped = 16 * 1024 * 1024
# a unit heard from within this many seconds counts as online
online_window = 60
# an online unit whose detector handled a frame within this many seconds counts as detecting
detecting_window = 30

# map background, the public openstreetmap tiles unless another source is set
default_tiles = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
default_tiles_credit = "© OpenStreetMap contributors"

local_addresses = ("127.0.0.1", "::1")


def unzip(body):
    # unzips a gzip body without letting it grow past max_unzipped
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    data = inflater.decompress(body, max_unzipped)
    if inflater.unconsumed_tail:
        raise ValueError("unzipped body too large")
    return data


def parse_unit_tokens(text):
    # reads unit-1=abc,unit-2=def into a dict of unit name to token
    tokens = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        unit, _, token = part.strip().partition("=")
        if not store.unit_pattern.match(unit) or not token:
            raise ValueError("NEUSE_CC_UNIT_TOKENS needs entries like unit-1=secret, separated by commas")
        tokens[unit] = token
    return tokens


def create_app(db_path=None, token=None, llm=None, unit_tokens=None, alert_settings=None, background_alerts=True):
    # builds the app, tests pass their own database, tokens and language model settings
    # background_alerts False checks for alerts during the request, which tests use so results are ready at once
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload
    # an rds for mysql address in NEUSE_CC_DATABASE_URL, or an sqlite file
    db = db_path or os.environ.get("NEUSE_CC_DATABASE_URL") or os.environ.get("NEUSE_CC_DB",
                                                                          project_root / "command_center.db")
    if not store.is_mysql(db):
        db = Path(db)
    secret = token if token is not None else os.environ.get("NEUSE_CC_TOKEN")
    per_unit = unit_tokens if unit_tokens is not None else parse_unit_tokens(os.environ.get("NEUSE_CC_UNIT_TOKENS"))
    # optional language model for situation reports, an openai style chat api such as huawei cloud's model service
    llm = llm if llm is not None else (os.environ.get("NEUSE_LLM_URL"), os.environ.get("NEUSE_LLM_KEY"),
                                       os.environ.get("NEUSE_LLM_MODEL"))
    # a setting left empty, as in the deployment's .env file, counts as not set
    tiles = os.environ.get("NEUSE_TILE_URL") or default_tiles
    tiles_credit = os.environ.get("NEUSE_TILE_CREDIT") or (default_tiles_credit if tiles == default_tiles else "")
    store.setup(db)
    # alerts for coordinators when an area gets several strong sightings, see command_center/alerts.py
    watch = alerts.watcher(db, alert_settings)
    if watch.problem:
        print(f"warning, {watch.problem}, alerts are still shown on the map", file=sys.stderr)

    def who():
        # returns True for the coordinator token, the unit's name for a unit token, or None when refused
        # detections show where people are, so nothing is given out without a reason to
        given = request.headers.get("Authorization", "").encode()
        if not secret and not per_unit:
            # no tokens set, so only this computer may use it
            # a request passed on by a proxy also arrives from this computer, so those are turned away too
            passed_on = request.headers.get("X-Forwarded-For") or request.headers.get("Forwarded")
            return True if request.remote_addr in local_addresses and not passed_on else None
        if secret and hmac.compare_digest(given, f"Bearer {secret}".encode()):
            return True
        # check every unit token so the time taken does not show which one matched
        matched = None
        for unit, unit_token in per_unit.items():
            if hmac.compare_digest(given, f"Bearer {unit_token}".encode()):
                matched = unit
        return matched

    def refuse():
        if not secret and not per_unit:
            return jsonify({"error": "no token is set, so this command center only answers the computer it runs on, "
                                     "set NEUSE_CC_TOKEN to use it over a network"}), 401
        return jsonify({"error": "missing or wrong token"}), 401

    def readers_only():
        # the reply to give when the caller may not read, or None when it may
        caller = who()
        if caller is None:
            if per_unit and not secret:
                # unit tokens only send, so with no coordinator token nobody could ever read
                return jsonify({"error": "no coordinator token is set, so the map and reports cannot be read, "
                                         "set NEUSE_CC_TOKEN as well as NEUSE_CC_UNIT_TOKENS"}), 401
            return refuse()
        if caller is not True:
            return jsonify({"error": "a unit token can send but not read"}), 403
        return None

    def read_body():
        # the request body as json, unzipped when the unit sent it compressed, or None when unreadable
        body = request.get_data()
        try:
            if request.headers.get("Content-Encoding") == "gzip":
                body = unzip(body)
            return json.loads(body)
        except (ValueError, zlib.error, RecursionError):
            # recursion covers json nested thousands of levels deep
            return None

    @app.get("/")
    def index():
        # the coordinator page, it asks for the token itself before loading any data
        return render_template("index.html", tiles=tiles, tiles_credit=tiles_credit)

    @app.get("/api/v1/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/v1/detections")
    def receive():
        caller = who()
        if caller is None:
            return refuse()
        payload = read_body()
        if payload is None:
            return jsonify({"error": "body must be json, optionally gzipped"}), 400
        detections = payload.get("detections") if isinstance(payload, dict) else None
        if not isinstance(detections, list):
            return jsonify({"error": "detections must be a list"}), 400
        if len(detections) > max_batch:
            return jsonify({"error": f"at most {max_batch} detections per request"}), 413

        # the unit's clock when it sent the batch, used to correct times from a unit whose clock is wrong
        sent_at = None
        if payload.get("sent_at") is not None:
            sent_at = store.parse_time(payload["sent_at"])
            if sent_at is None:
                return jsonify({"error": "sent_at must be an iso time with a utc offset"}), 400

        accepted, rejected = store.save(db, detections, sent_at, None if caller is True else caller)
        if accepted:
            watch.check_soon() if background_alerts else watch.check_now()
        return jsonify({"accepted": accepted, "rejected": rejected})

    @app.post("/api/v1/status")
    def status():
        # a unit saying it is alive, where it is and how much it still has to send
        caller = who()
        if caller is None:
            return refuse()
        note = read_body()
        if note is None:
            return jsonify({"error": "body must be json, optionally gzipped"}), 400
        problem = store.status_problem(note)
        if problem:
            return jsonify({"error": problem}), 400
        if caller is not True and note["unit_id"] != caller:
            return jsonify({"error": f"this token only sends for {caller}"}), 403
        store.save_status(db, note)
        return jsonify({"ok": True})

    @app.post("/api/v1/coverage")
    def receive_coverage():
        # ground squares a unit's camera has covered, so coordinators can see what has been searched
        caller = who()
        if caller is None:
            return refuse()
        payload = read_body()
        if payload is None:
            return jsonify({"error": "body must be json, optionally gzipped"}), 400
        problem = store.coverage_problem(payload)
        if problem:
            return jsonify({"error": problem}), 400
        if caller is not True and payload["unit_id"] != caller:
            return jsonify({"error": f"this token only sends for {caller}"}), 403
        sent_at = None
        if payload.get("sent_at") is not None:
            sent_at = store.parse_time(payload["sent_at"])
            if sent_at is None:
                return jsonify({"error": "sent_at must be an iso time with a utc offset"}), 400
        return jsonify({"saved": store.save_coverage(db, payload["unit_id"], payload["cells"], sent_at)})

    @app.get("/api/v1/coverage")
    def searched():
        # covered squares for the map, hours sets how far back to look, at most a week
        denied = readers_only()
        if denied:
            return denied
        try:
            hours = float(request.args.get("hours", 24))
        except ValueError:
            return jsonify({"error": "hours must be a number"}), 400
        if not 0 < hours <= 168:
            return jsonify({"error": "hours must be above 0 and at most 168"}), 400
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
        return jsonify({"cell_degrees": store.cell_degrees, "cells": store.coverage(db, since)})

    @app.get("/api/v1/alerts")
    def recent_alerts():
        # alerts for the map, hours sets how far back to look, at most a week
        denied = readers_only()
        if denied:
            return denied
        try:
            hours = float(request.args.get("hours", 24))
        except ValueError:
            return jsonify({"error": "hours must be a number"}), 400
        if not 0 < hours <= 168:
            return jsonify({"error": "hours must be above 0 and at most 168"}), 400
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
        return jsonify({"alerts": store.alerts(db, since), "texts": watch.sender is not None,
                        "problem": watch.problem})

    @app.get("/api/v1/units")
    def units():
        denied = readers_only()
        if denied:
            return denied
        now = datetime.now(timezone.utc)
        summaries = store.units(db)
        for unit in summaries:
            unit.update(store.unit_state(unit, now, online_window, detecting_window))
        return jsonify({"units": summaries, "detections": store.count(db)})

    @app.get("/api/v1/detections")
    def detections():
        # detections for the map, after takes the seq of the last one already seen
        denied = readers_only()
        if denied:
            return denied
        try:
            after = request.args.get("after")
            after = int(after) if after is not None else None
            limit = min(max(int(request.args.get("limit", 1000)), 1), max_recent)
        except ValueError:
            return jsonify({"error": "after and limit must be whole numbers"}), 400
        if after is not None and not -store.max_whole <= after <= store.max_whole:
            return jsonify({"error": "after is out of range"}), 400
        return jsonify({"detections": store.recent(db, after, limit)})

    @app.get("/api/v1/report")
    def situation_report():
        # a plain language summary of the last few minutes, with the facts it was written from
        denied = readers_only()
        if denied:
            return denied
        try:
            minutes = int(request.args.get("minutes", 30))
        except ValueError:
            return jsonify({"error": "minutes must be a whole number"}), 400
        if not 1 <= minutes <= 1440:
            return jsonify({"error": "minutes must be between 1 and 1440"}), 400
        return jsonify(reports.report(db, minutes, llm, online_window=online_window,
                                      detecting_window=detecting_window))

    return app


def main():
    host = os.environ.get("NEUSE_CC_HOST", "127.0.0.1")
    port = int(os.environ.get("NEUSE_CC_PORT", "8000"))
    has_tokens = os.environ.get("NEUSE_CC_TOKEN") or os.environ.get("NEUSE_CC_UNIT_TOKENS")
    # opening it to a network without a token would only turn every caller away, so say so up front
    if host not in ("127.0.0.1", "localhost", "::1") and not has_tokens:
        sys.exit("set NEUSE_CC_TOKEN before opening the command center to a network, "
                 "without it only this computer is answered")
    # unit tokens can only send, so without a coordinator token the map and reports are closed to everyone
    if os.environ.get("NEUSE_CC_UNIT_TOKENS") and not os.environ.get("NEUSE_CC_TOKEN"):
        print("warning, NEUSE_CC_UNIT_TOKENS is set but NEUSE_CC_TOKEN is not, so nobody can open the map or reports",
              file=sys.stderr)

    app = create_app()
    # waitress is a real web server, flask's own is only meant for development
    # put it behind https before sending tokens over a network, see the readme
    from waitress import serve
    print(f"command center on http://{host}:{port}")
    serve(app, host=host, port=port, threads=8)


if __name__ == "__main__":
    main()
