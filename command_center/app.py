"""
command center api that field units sync their detections to
this is the receiving end of store and forward, the map and reports for coordinators come later

usage
    python -m command_center.app
    NEUSE_CC_TOKEN=some-secret python -m command_center.app
"""
import hmac
import json
import os
import zlib
from pathlib import Path

from flask import Flask, jsonify, request

from command_center import store

project_root = Path(__file__).resolve().parent.parent

# most detections one request may carry
max_batch = 500
# limits on the request body, before and after unzipping, so a bad upload cannot fill the memory
max_upload = 2 * 1024 * 1024
max_unzipped = 16 * 1024 * 1024


def unzip(body):
    # unzips a gzip body without letting it grow past max_unzipped
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    data = inflater.decompress(body, max_unzipped)
    if inflater.unconsumed_tail:
        raise ValueError("unzipped body too large")
    return data


def create_app(db_path=None, token=None):
    # builds the app, tests pass their own database and token
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload
    db = Path(db_path or os.environ.get("NEUSE_CC_DB", project_root / "command_center.db"))
    secret = token if token is not None else os.environ.get("NEUSE_CC_TOKEN")
    store.setup(db)

    def allowed():
        # with a token set, every api call must carry it, since detections show where people are
        if not secret:
            return True
        given = request.headers.get("Authorization", "")
        return hmac.compare_digest(given.encode(), f"Bearer {secret}".encode())

    @app.get("/api/v1/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/v1/detections")
    def receive():
        if not allowed():
            return jsonify({"error": "missing or wrong token"}), 401
        body = request.get_data()
        try:
            if request.headers.get("Content-Encoding") == "gzip":
                body = unzip(body)
            payload = json.loads(body)
        except (ValueError, zlib.error):
            return jsonify({"error": "body must be json, optionally gzipped"}), 400

        detections = payload.get("detections") if isinstance(payload, dict) else None
        if not isinstance(detections, list):
            return jsonify({"error": "detections must be a list"}), 400
        if len(detections) > max_batch:
            return jsonify({"error": f"at most {max_batch} detections per request"}), 413

        accepted, rejected = store.save(db, detections)
        return jsonify({"accepted": accepted, "rejected": rejected})

    @app.get("/api/v1/units")
    def units():
        if not allowed():
            return jsonify({"error": "missing or wrong token"}), 401
        return jsonify({"units": store.units(db), "detections": store.count(db)})

    return app


if __name__ == "__main__":
    # local only by default, set NEUSE_CC_HOST=0.0.0.0 on the cloud server
    create_app().run(
        host=os.environ.get("NEUSE_CC_HOST", "127.0.0.1"),
        port=int(os.environ.get("NEUSE_CC_PORT", "8000")),
    )
