import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, jsonify
from src.config import config

app = Flask(__name__)

def fetch_recent_logs():
    # pull latest 15 detections from sqlite
    try:
        with sqlite3.connect(config.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                select id, timestamp, latitude, longitude, confidence, thermal_peak, bbox 
                from detections 
                order by id desc limit 15
            """)
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []

@app.route("/")
def index():
    # render telemetry dashboard UI
    return render_template("index.html")

@app.route("/api/telemetry")
def telemetry():
    # JSON endpoint for dashboard frontend
    logs = fetch_recent_logs()
    return jsonify({
        "status": "online",
        "device": config.target_chip,
        "count": len(logs),
        "data": logs
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)