"""
command center that field units sync their detections to
units post detections and short status notes, and coordinators watch a live map of units and detections

who may do what
    NEUSE_CC_TOKEN is the coordinator token, it can read everything and also send
    NEUSE_CC_UNIT_TOKENS gives each unit its own token, like unit-1=abc,unit-2=def
    a unit token can only send, and only under its own unit's name, so a lost unit cannot read where people are
    with no tokens set the command center only answers the computer it runs on

security, see docs/SAFETY.md
    every page and reply carries headers that stop it being framed, sniffed or cached, and the map page
    a content security policy that only runs its own scripts
    a caller that gives wrong tokens again and again is turned away for a while
    every api request is logged with who made it, without the token
    NEUSE_CC_KEEP_DAYS removes sightings, searched ground and alerts older than that many days
    behind a proxy such as caddy, set NEUSE_CC_TRUST_PROXY=1 so the caller's own address is used

the database is an sqlite file by default
set NEUSE_CC_DATABASE_URL=mysql://user:password@host:3306/database to use rds for mysql instead

usage
    python -m command_center.app
    NEUSE_CC_TOKEN=some-secret NEUSE_CC_HOST=0.0.0.0 python -m command_center.app
    see docs/DEPLOY.md to run it on huawei cloud
"""
import hmac
import json
import logging
import os
import secrets
import sys
import threading
import time
import urllib.parse
import zlib
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

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

# a caller with this many refused tokens within the window is turned away until the window passes
most_failures = 20
failure_window = 300
# shortest token accepted when the command center opens to a network
min_token_length = 16

log = logging.getLogger("neuse.command_center")


class failures:
    # counts refused tokens per caller address, so guessing tokens is slow
    def __init__(self, most=most_failures, window=failure_window, clock=time.monotonic):
        self.most, self.window, self.clock = most, window, clock
        self.seen = {}
        self.lock = threading.Lock()

    def _recent(self, address):
        times = self.seen.get(address)
        if times is None:
            return None
        while times and self.clock() - times[0] > self.window:
            times.popleft()
        if not times:
            del self.seen[address]
            return None
        return times

    def blocked(self, address):
        with self.lock:
            times = self._recent(address)
            return times is not None and len(times) >= self.most

    def failed(self, address):
        with self.lock:
            # forget callers whose window has passed, so many addresses cannot fill the memory
            if len(self.seen) > 10000:
                for old in list(self.seen):
                    self._recent(old)
            self.seen.setdefault(address, deque(maxlen=self.most)).append(self.clock())


def loggable(value, limit=200):
    # text from a request made safe for one log line, so a crafted path cannot start a fake line
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    return "".join(c if c.isprintable() else f"\\x{ord(c):02x}" for c in text)[:limit]


def token_problem(name, token):
    # why a token is too weak to open the command center to a network, or None when it will do
    if len(token) < min_token_length:
        return f"{name} is shorter than {min_token_length} characters"
    if "change-me" in token.lower():
        return f"{name} still has the example value from .env.example"
    return None


def tile_origin(tiles):
    # where map tiles come from, for the content security policy, {s} subdomains become a wildcard
    parts = urllib.parse.urlsplit(tiles.replace("{s}", "tiles-sub"))
    if parts.scheme not in ("http", "https") or not parts.hostname or "{" in parts.netloc:
        return "https:"
    host = parts.netloc.replace("tiles-sub", "*")
    return f"{parts.scheme}://{host}"


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


def create_app(db_path=None, token=None, llm=None, unit_tokens=None, alert_settings=None, background_alerts=True,
               keep_days=None, trust_proxy=None, refused=None):
    # builds the app, tests pass their own database, tokens and language model settings
    # background_alerts False checks for alerts during the request, which tests use so results are ready at once
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload
    # behind caddy every request comes from the proxy, which passes on the caller's own address
    trust_proxy = trust_proxy if trust_proxy is not None else os.environ.get("NEUSE_CC_TRUST_PROXY") == "1"
    if trust_proxy:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    refused = refused or failures()
    # days to keep sightings, searched ground and alerts, 0 or unset keeps them
    keep_days = keep_days if keep_days is not None else float(os.environ.get("NEUSE_CC_KEEP_DAYS") or 0)
    pruned = {"at": float("-inf")}
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
    tiles_from = tile_origin(tiles)
    store.setup(db)
    # alerts for coordinators when an area gets several strong sightings, see command_center/alerts.py
    watch = alerts.watcher(db, alert_settings)
    if watch.problem:
        print(f"warning, {watch.problem}, alerts are still shown on the map", file=sys.stderr)

    def who():
        g.caller = caller = identify()
        return caller

    def identify():
        # returns True for the coordinator token, the unit's name for a unit token, or None when refused
        # detections show where people are, so nothing is given out without a reason to
        given = request.headers.get("Authorization", "").encode()
        if not secret and not per_unit:
            # no tokens set, so only this computer may use it
            # a request passed on by a proxy also arrives from this computer, so those are turned away too
            passed_on = request.headers.get("X-Forwarded-For") or request.headers.get("Forwarded")
            return True if request.remote_addr in local_addresses and not passed_on else None
        if not given:
            return None
        if secret and hmac.compare_digest(given, f"Bearer {secret}".encode()):
            return True
        # check every unit token so the time taken does not show which one matched
        matched = None
        for unit, unit_token in per_unit.items():
            if hmac.compare_digest(given, f"Bearer {unit_token}".encode()):
                matched = unit
        return matched

    def refuse():
        # each refused token counts against the caller, see failures
        refused.failed(request.remote_addr or "unknown")
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

    def prune_now():
        # removes old records at most once an hour, so a long running command center keeps only what it needs
        if keep_days <= 0 or time.monotonic() - pruned["at"] < 3600:
            return
        pruned["at"] = time.monotonic()
        before = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat(timespec="seconds")
        try:
            removed = store.prune(db, before)
            if removed:
                log.info("removed %d records older than %g days", removed, keep_days)
        except Exception as err:  # keeping data too long must never stop the command center
            log.warning("could not remove old records, %s", err)

    @app.before_request
    def guard():
        # turns away a caller that keeps giving wrong tokens, and notes the time for the access log
        g.started = time.perf_counter()
        g.nonce = secrets.token_urlsafe(16)
        if request.path.startswith("/api/") and refused.blocked(request.remote_addr or "unknown"):
            return jsonify({"error": "too many wrong tokens from this address, try again in a few minutes"}), 429
        prune_now()
        return None

    @app.after_request
    def harden(response):
        # headers that stop the page being framed by another site, files being read as another type,
        # page addresses leaking to other sites, and replies holding where people are being cached
        # other sites get only this server's own address, which openstreetmap's tile servers require
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        if response.mimetype == "text/html":
            # only the page's own scripts run, so text from a unit could never run as code even if it slipped in
            response.headers["Content-Security-Policy"] = (
                f"default-src 'self'; script-src 'self' 'nonce-{g.nonce}'; style-src 'self' 'unsafe-inline'; "
                f"img-src 'self' data: {tiles_from}; connect-src 'self'; object-src 'none'; base-uri 'none'; "
                "form-action 'self'; frame-ancestors 'none'")
        if request.path.startswith("/api/") and request.path != "/api/v1/health":
            caller = getattr(g, "caller", None)
            name = "coordinator" if caller is True else caller or "refused"
            log.info("%s %s %s %s from %s in %.0f ms", loggable(request.method), loggable(request.path),
                     response.status_code, loggable(name), loggable(request.remote_addr),
                     (time.perf_counter() - g.started) * 1000)
        return response

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
        return render_template("index.html", tiles=tiles, tiles_credit=tiles_credit, nonce=g.nonce)

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
    # a short or example token on a network can be guessed, so the command center will not start with one
    if host not in ("127.0.0.1", "localhost", "::1"):
        named = [("NEUSE_CC_TOKEN", os.environ.get("NEUSE_CC_TOKEN") or "")] if os.environ.get("NEUSE_CC_TOKEN") else []
        named += [(f"the token for {unit}", value)
                  for unit, value in parse_unit_tokens(os.environ.get("NEUSE_CC_UNIT_TOKENS")).items()]
        for name, value in named:
            problem = token_problem(name, value)
            if problem:
                sys.exit(f"{problem}, make one with  python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
    if os.environ.get("NEUSE_CC_UNIT_TOKENS") and not os.environ.get("NEUSE_CC_TOKEN"):
        print("warning, NEUSE_CC_UNIT_TOKENS is set but NEUSE_CC_TOKEN is not, so nobody can open the map or reports",
              file=sys.stderr)

    # the access log and warnings go to the console, where docker keeps them
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = create_app()
    # waitress is a real web server, flask's own is only meant for development
    # put it behind https before sending tokens over a network, see the readme
    from waitress import serve
    print(f"command center on http://{host}:{port}")
    serve(app, host=host, port=port, threads=8)


if __name__ == "__main__":
    main()
