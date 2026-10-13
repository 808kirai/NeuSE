import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Make sure the project root is on the Python path so local modules can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Flask app setup and project config.
from flask import Flask, render_template, jsonify
from src.config import config

app = Flask(__name__)

# Keep the SQLite database in the project root so it works no matter where the app starts.
DB_PATH = PROJECT_ROOT / "neuse_local.db"


def init_db():
    """Create the detections table and load a few sample readings for the dashboard."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            # Reset the table so the app always starts with a clean, known dataset.
            cur.execute("DROP TABLE IF EXISTS detections")
            cur.execute("""
                CREATE TABLE detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    latitude REAL,
                    longitude REAL,
                    confidence REAL,
                    thermal_peak REAL,
                    bbox TEXT
                )
            """)

            # Seed the app with recent sample telemetry to test the UI immediately.
            sample_data = [
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), -1.29210, 36.82190, 0.945, 41.2, '[45, 60, 120, 180]'),
                ((datetime.now() - timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S'), -1.29450, 36.82500, 0.891, 39.8, '[12, 40, 95, 150]'),
                ((datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'), -1.29000, 36.81900, 0.968, 42.5, '[200, 150, 280, 310]')
            ]
            cur.executemany("""
                INSERT INTO detections (timestamp, latitude, longitude, confidence, thermal_peak, bbox)
                VALUES (?, ?, ?, ?, ?, ?)
            """, sample_data)
            conn.commit()
    except Exception as e:
        print(f"Database initialization error: {e}")

# Build the local SQLite data store as soon as the app loads.
init_db()


def fetch_recent_logs():
    """Fetch the newest detections in a format the frontend can render."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT id, timestamp, latitude, longitude, confidence, thermal_peak, bbox 
                FROM detections 
                ORDER BY id DESC LIMIT 15
            """)
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"Fetch logs error: {e}")
        return []


@app.route("/")
def index():
    # Serve the main dashboard page.
    return render_template("index.html")


@app.route("/api/telemetry")
def telemetry():
    # Return the most recent detections to the frontend as JSON.
    logs = fetch_recent_logs()
    return jsonify({
        "status": "online",
        "device": config.target_chip,
        "count": len(logs),
        "data": logs
    })


if __name__ == "__main__":
    # Run the app locally for development and live dashboard testing.
    app.run(host="0.0.0.0", port=5000, debug=True)