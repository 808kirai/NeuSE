import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

# put the project root on the path so src can be imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify
from src.logger import data_logger, connect, read_status

# local web dashboard that shows what the pipeline has logged
app = Flask(__name__)

# make sure the table exists without touching saved detections
data_logger()


def fetch_telemetry(limit=15):
    # newest detections, the total count, how many still wait to reach the command center
    # how many it refused, and the unit's latest status row
    # limit sets how many rows the table on the page shows
    try:
        with closing(connect()) as conn:
            # rows come back as dicts so they turn straight into json
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                select id, timestamp, latitude, longitude, confidence, thermal_peak, bbox, detector, latency_ms
                from detections
                order by id desc limit ?
            """, (limit,)).fetchall()
            total = conn.execute("select count(*) from detections").fetchone()[0]
            queued = conn.execute(
                "select count(*) from detections where synced_at is null and sync_error is null").fetchone()[0]
            refused = conn.execute(
                "select count(*) from detections where synced_at is null and sync_error is not null").fetchone()[0]
        status = read_status()
    except sqlite3.Error as err:
        # a locked or broken database shows as empty instead of crashing the page
        print(f"could not read detections {err}")
        return [], 0, 0, 0, None

    return [dict(r) for r in rows], total, queued, refused, status


@app.route("/")
def index():
    # serve the dashboard page
    return render_template("index.html")


@app.route("/api/telemetry")
def telemetry():
    # data the dashboard page asks for every two seconds
    logs, total, queued, refused, status = fetch_telemetry()
    status = status or {}
    average = status.get("avg_latency_ms")
    return jsonify({
        "status": "online",
        # the detector the pipeline last said it was running, or the one named on the newest row
        "detector": status.get("detector") or (logs[0]["detector"] if logs else None),
        "total": total,
        # detections the command center has not confirmed yet, they stay safe here until it does
        "queued": queued,
        # detections the command center turned away, they need someone to look at them
        "refused": refused,
        "count": len(logs),
        # mean time per frame over the pipeline's last 100 frames, with or without detections
        # averaging over detections instead would count busy frames many times and empty frames never
        "avg_latency_ms": round(average, 2) if average is not None else None,
        "data": logs
    })


if __name__ == "__main__":
    # local only by default
    # set NEUSE_HOST=0.0.0.0 to open it to the network
    # set NEUSE_DEBUG=1 only on a trusted dev machine
    app.run(
        host=os.environ.get("NEUSE_HOST", "127.0.0.1"),
        port=int(os.environ.get("NEUSE_PORT", "5000")),
        debug=os.environ.get("NEUSE_DEBUG") == "1",
    )
