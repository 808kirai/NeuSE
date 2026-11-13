"""
command center that field units sync their detections to
units post detections and short status notes, and coordinators watch a live map of units and detections

usage
    python -m command_center.app
    NEUSE_CC_TOKEN=some-secret python -m command_center.app
"""
import hmac
import json
import os
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from command_center import reports, store

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


def unzip(body):
    # unzips a gzip body without letting it grow past max_unzipped
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    data = inflater.decompress(body, max_unzipped)
    if inflater.unconsumed_tail:
        raise ValueError("unzipped body too large")
    return data


def utc(text):
    # reads an iso time with a utc offset and gives it back in utc, or None when it cannot
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        return None
    return moment.astimezone(timezone.utc)


def create_app(db_path=None, token=None, llm=None):
    # builds the app, tests pass their own database, token and language model settings
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload
    db = Path(db_path or os.environ.get("NEUSE_CC_DB", project_root / "command_center.db"))
    secret = token if token is not None else os.environ.get("NEUSE_CC_TOKEN")
    # optional language model for situation reports, an openai style chat api such as huawei cloud's model service
    llm = llm if llm is not None else (os.environ.get("NEUSE_LLM_URL"), os.environ.get("NEUSE_LLM_KEY"),
                                       os.environ.get("NEUSE_LLM_MODEL"))
    store.setup(db)

    def allowed():
        # with a token set, every api call must carry it, since detections show where people are
        if not secret:
            return True
        given = request.headers.get("Authorization", "")
        return hmac.compare_digest(given.encode(), f"Bearer {secret}".encode())

    def read_body():
        # the request body as json, unzipped when the unit sent it compressed, or None when unreadable
        body = request.get_data()
        try:
            if request.headers.get("Content-Encoding") == "gzip":
                body = unzip(body)
            return json.loads(body)
        except (ValueError, zlib.error):
            return None

    refused = ({"error": "missing or wrong token"}, 401)

    @app.get("/")
    def index():
        # the coordinator page, it asks for the token itself before loading any data
        return render_template("index.html", online_window=online_window)

    @app.get("/api/v1/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/v1/detections")
    def receive():
        if not allowed():
            return refused
        payload = read_body()
        if payload is None:
            return jsonify({"error": "body must be json, optionally gzipped"}), 400
        detections = payload.get("detections") if isinstance(payload, dict) else None
        if not isinstance(detections, list):
            return jsonify({"error": "detections must be a list"}), 400
        if len(detections) > max_batch:
            return jsonify({"error": f"at most {max_batch} detections per request"}), 413

        accepted, rejected = store.save(db, detections)
        return jsonify({"accepted": accepted, "rejected": rejected})

    @app.post("/api/v1/status")
    def status():
        # a unit saying it is alive, where it is and how much it still has to send
        if not allowed():
            return refused
        note = read_body()
        if note is None:
            return jsonify({"error": "body must be json, optionally gzipped"}), 400
        problem = store.status_problem(note)
        if problem:
            return jsonify({"error": problem}), 400
        store.save_status(db, note)
        return jsonify({"ok": True})

    @app.get("/api/v1/units")
    def units():
        if not allowed():
            return refused
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=online_window)
        summaries = store.units(db)
        for unit in summaries:
            contact = utc(unit["last_contact"])
            unit["online"] = contact is not None and contact >= cutoff
        return jsonify({"units": summaries, "detections": store.count(db)})

    @app.get("/api/v1/detections")
    def detections():
        # detections for the map, after takes the seq of the last one already seen
        if not allowed():
            return refused
        try:
            after = request.args.get("after")
            after = int(after) if after is not None else None
            limit = min(max(int(request.args.get("limit", 1000)), 1), max_recent)
        except ValueError:
            return jsonify({"error": "after and limit must be whole numbers"}), 400
        return jsonify({"detections": store.recent(db, after, limit)})

    @app.get("/api/v1/report")
    def situation_report():
        # a plain language summary of the last few minutes, with the facts it was written from
        if not allowed():
            return refused
        try:
            minutes = int(request.args.get("minutes", 30))
        except ValueError:
            return jsonify({"error": "minutes must be a whole number"}), 400
        if not 1 <= minutes <= 1440:
            return jsonify({"error": "minutes must be between 1 and 1440"}), 400
        return jsonify(reports.report(db, minutes, llm))

    return app


if __name__ == "__main__":
    # local only by default, set NEUSE_CC_HOST=0.0.0.0 on the cloud server
    create_app().run(
        host=os.environ.get("NEUSE_CC_HOST", "127.0.0.1"),
        port=int(os.environ.get("NEUSE_CC_PORT", "8000")),
    )
