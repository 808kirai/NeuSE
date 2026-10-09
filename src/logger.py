import sqlite3
import random
from datetime import datetime
from src.config import config


class data_logger:
    def __init__(self):
        # create the local detections table
        with sqlite3.connect(config.db_file) as conn:
            conn.execute("""
                create table if not exists detections (
                    id integer primary key autoincrement,
                    timestamp text,
                    latitude real,
                    longitude real,
                    confidence real,
                    thermal_peak real,
                    bbox text
                )
            """)

    def log_event(self, conf, peak_heat, bbox):
        # save the detection and its location
        ts = datetime.now().strftime("%H:%M:%S")
        lat = -1.286389 + random.uniform(-0.0005, 0.0005)
        lon = 36.817223 + random.uniform(-0.0005, 0.0005)

        with sqlite3.connect(config.db_file) as conn:
            conn.execute("""
                insert into detections (timestamp, latitude, longitude, confidence, thermal_peak, bbox)
                values (?, ?, ?, ?, ?, ?)
            """, (ts, lat, lon, conf, peak_heat, str(bbox)))