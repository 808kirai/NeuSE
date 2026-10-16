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
from src.logger import data_logger, connect

# local web dashboard that shows what the pipeline has logged
app = Flask(__name__)

# make sure the table exists without touching saved detections
data_logger()


def fetch_telemetry(limit=15):
    # newest detections plus the total count
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
    except sqlite3.Error as err:
        # a locked or broken database shows as empty instead of crashing the page
        print(f"could not read detections {err}")
        return [], 0

    return [dict(r) for r in rows], total


@app.route("/")
def index():
    # serve the dashboard page
    return render_template("index.html")


@app.route("/api/telemetry")
def telemetry():
    # data the dashboard page asks for every two seconds
    logs, total = fetch_telemetry()
    latencies = [r["latency_ms"] for r in logs if r["latency_ms"] is not None]
    return jsonify({
        "status": "online",
        # the newest row shows which detector is running
        "detector": logs[0]["detector"] if logs else None,
        "total": total,
        "count": len(logs),
        # average measured time per frame across recent detections
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
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
