import sqlite3
import time
from src.config import config

# offline logger writing coordinate metadata directly to the disk
class data_logger:
    def __init__(self, db_path=config.db_file):
        self.db_path = db_path
        self.setup_db()

    def setup_db(self):
        # create table if first time running 
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                create table if not exists detections (
                    id integer primary key autoincrement,
                    timestamp real,
                    latitude real,
                    longitude real,
                    confidence real,
                    thermal_peak integer,
                    bbox text
                )
            """)
            conn.commit()

    def log_event(self, conf, thermal_peak, bbox, lat=config.default_lat, lon=config.default_lon):
        # dump detection payload into local db
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                insert into detections (timestamp, latitude, longitude, confidence, thermal_peak, bbox)
                values (?, ?, ?, ?, ?, ?)
            """, (time.time(), lat, lon, conf, thermal_peak, str(bbox)))
            conn.commit()