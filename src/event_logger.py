# Event logger for tracking bag ownership events
import sqlite3
import os
import cv2
from datetime import datetime

# Resolve paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "logs", "events.db")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "logs", "snapshots")


class EventLogger:
    def __init__(self, db_path=DB_PATH, snapshot_dir=SNAPSHOT_DIR):
        self.db_path = db_path
        self.snapshot_dir = snapshot_dir

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)

        self._init_db()

        # Track which bag_ids we've already logged as unattended,
        # so we don't spam duplicate entries every frame
        self.logged_bag_ids = set()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bag_id INTEGER,
                class_name TEXT,
                duration_seconds REAL,
                timestamp TEXT,
                snapshot_path TEXT,
                alert_message TEXT
            )
        """)
        conn.commit()
        conn.close()

    def should_log(self, bag_id, state):
        """
        Returns True only the first time a bag transitions to 'unattended'.
        Prevents duplicate log entries every frame it stays unattended.
        """
        if state == "unattended" and bag_id not in self.logged_bag_ids:
            self.logged_bag_ids.add(bag_id)
            return True
        return False

    def reset_bag(self, bag_id):
        """Call this when a bag goes back to 'attended' so it can be re-logged if orphaned again later."""
        self.logged_bag_ids.discard(bag_id)

    def log_event(self, bag_id, class_name, duration_seconds, frame=None, snapshot_path=None, alert_message=""):
        """
        Saves a snapshot image and inserts an event record into the database.
        Returns (event_id, snapshot_path).
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if snapshot_path is None:
            snapshot_filename = f"bag{bag_id}_{timestamp}.jpg"
            snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)

        if frame is not None and not os.path.exists(snapshot_path):
            cv2.imwrite(snapshot_path, frame)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO events (bag_id, class_name, duration_seconds, timestamp, snapshot_path, alert_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (bag_id, class_name, duration_seconds, timestamp, snapshot_path, alert_message))
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print(f"[LOGGED] Bag #{bag_id} ({class_name}) unattended for {int(duration_seconds)}s — saved to {snapshot_path}")
        return event_id, snapshot_path

    def update_alert_message(self, event_id, alert_message):
        """Updates an event record with Gemma 4's enriched alert description."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE events SET alert_message = ? WHERE id = ?", (alert_message, event_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB ERROR] Failed to update alert message: {e}")