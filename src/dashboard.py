import os
import sqlite3
import threading
import time
import logging
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from detector import DetectionEngine
from notifier import send_alert_notification

# Project root paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "logs", "events.db")
SNAP_DIR = os.path.join(BASE_DIR, "logs", "snapshots")

# Ensure required folders exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

# Silence standard Werkzeug logging to keep console output clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

template_dir = os.path.join(os.path.dirname(__file__), "templates")
if not os.path.exists(template_dir):
    template_dir = os.path.dirname(__file__)

app = Flask(__name__, template_folder=template_dir)

# Thread-safe frame and telemetry cache
_frame = None
_lock = threading.Lock()
_last_frame_time = 0
_external_dashboard_url = None
_telemetry = {
    "bags": [],
    "person_count": 0,
    "bag_count": 0,
    "fps": 0.0,
    "camera_active": False,
    "last_updated": 0
}

# Detection Engine instance
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        source = os.environ.get("CAMERA_SOURCE", "0")
        threshold = int(os.environ.get("ALERT_THRESHOLD", "15"))
        _detector = DetectionEngine(source=source, threshold=threshold)
    return _detector


def set_frame(frame):
    """
    Called with each annotated frame (BGR numpy array).
    Encodes frame as JPEG for MJPEG browser streaming.
    """
    global _frame, _last_frame_time
    if frame is None:
        return
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return
    jpg_bytes = buf.tobytes()
    with _lock:
        _frame = jpg_bytes
        _last_frame_time = time.time()

    if _external_dashboard_url:
        try:
            import requests
            requests.post(f"{_external_dashboard_url}/api/push_frame", data=jpg_bytes, timeout=0.08)
        except Exception:
            pass


def update_telemetry(data):
    """
    Called to update live tracking state for dashboard cards.
    """
    global _telemetry
    with _lock:
        _telemetry = {
            "bags": data.get("bags", []),
            "person_count": data.get("person_count", 0),
            "bag_count": data.get("bag_count", 0),
            "fps": data.get("fps", 0.0),
            "camera_active": True,
            "last_updated": time.time()
        }

    if _external_dashboard_url:
        try:
            import requests
            requests.post(f"{_external_dashboard_url}/api/push_telemetry", json=data, timeout=0.08)
        except Exception:
            pass


def _create_standby_frame():
    """Generates an aesthetic standby card when camera is waiting to start."""
    canvas = np.zeros((540, 960, 3), dtype=np.uint8)
    canvas[:] = (20, 24, 33)

    cv2.rectangle(canvas, (20, 20), (940, 520), (45, 55, 75), 2)
    cv2.circle(canvas, (480, 210), 40, (38, 45, 60), -1)
    cv2.circle(canvas, (480, 210), 40, (90, 100, 125), 2)
    cv2.putText(canvas, "!", (473, 222), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (245, 165, 36), 3)

    cv2.putText(
        canvas,
        "CAMERA STREAM STANDBY",
        (300, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (230, 233, 239),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Click 'Start Camera' or launch main.py to activate feed.",
        (240, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (138, 147, 166),
        1,
        cv2.LINE_AA
    )
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        canvas,
        f"System Time: {ts}",
        (380, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 110, 130),
        1,
        cv2.LINE_AA
    )

    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def _mjpeg():
    """Generator yielding multipart MJPEG frames to browser clients."""
    while True:
        with _lock:
            data = _frame
            last_time = _last_frame_time

        # If no frame fed in last 4 seconds, show standby visual
        if data is None or (time.time() - last_time > 4.0):
            data = _create_standby_frame()
            time.sleep(0.5)
        else:
            time.sleep(0.033)  # ~30 FPS

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" +
            data + b"\r\n"
        )


def _query(sql, args=()):
    """Executes SQLite query and returns dictionary rows."""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    resp = Response(_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/alerts")
def alerts():
    """Returns past logged alerts with normalized fields for the frontend."""
    limit = request.args.get("limit", 50, type=int)
    class_filter = request.args.get("class", None)

    try:
        if class_filter:
            rows = _query(
                """
                SELECT id, bag_id, class_name, duration_seconds, timestamp, snapshot_path, alert_message
                FROM events
                WHERE class_name = ?
                ORDER BY id DESC LIMIT ?
                """,
                (class_filter, limit)
            )
        else:
            rows = _query(
                """
                SELECT id, bag_id, class_name, duration_seconds, timestamp, snapshot_path, alert_message
                FROM events
                ORDER BY id DESC LIMIT ?
                """,
                (limit,)
            )
    except sqlite3.Error as e:
        print(f"[DB ERROR] {e}")
        rows = []

    for r in rows:
        r["track_id"] = r.get("bag_id")
        r["event_type"] = "unattended"
        r["alert_text"] = r.get("alert_message") or "(No alert message available)"

        snap_path = r.get("snapshot_path")
        if snap_path:
            filename = os.path.basename(snap_path.replace("\\", "/"))
            r["snapshot"] = filename
            r["snapshot_exists"] = os.path.exists(os.path.join(SNAP_DIR, filename))
        else:
            r["snapshot"] = None
            r["snapshot_exists"] = False

        raw_ts = r.get("timestamp", "")
        try:
            if "_" in raw_ts:
                dt_part, tm_part = raw_ts.split("_", 1)
                r["formatted_time"] = f"{dt_part} {tm_part.replace('-', ':')}"
            else:
                r["formatted_time"] = raw_ts
        except Exception:
            r["formatted_time"] = raw_ts

    return jsonify(rows)


@app.route("/stats")
def stats():
    """Returns aggregate dashboard KPI stats."""
    total = 0
    unique_bags = 0
    by_class = {}
    last_timestamp = ""

    try:
        total_row = _query("SELECT COUNT(*) AS n FROM events")
        total = total_row[0]["n"] if total_row else 0

        unique_row = _query("SELECT COUNT(DISTINCT bag_id) AS n FROM events")
        unique_bags = unique_row[0]["n"] if unique_row else 0

        class_rows = _query("SELECT class_name, COUNT(*) as cnt FROM events GROUP BY class_name")
        for cr in class_rows:
            by_class[cr["class_name"]] = cr["cnt"]

        latest_row = _query("SELECT timestamp FROM events ORDER BY id DESC LIMIT 1")
        if latest_row:
            last_timestamp = latest_row[0]["timestamp"]

    except sqlite3.Error as e:
        print(f"[DB STATS ERROR] {e}")

    with _lock:
        is_camera_active = (time.time() - _last_frame_time) < 4.0 and _frame is not None
        telemetry_copy = dict(_telemetry)
        telemetry_copy["camera_active"] = is_camera_active

    return jsonify({
        "total": total,
        "unattended": total,
        "unique_bags": unique_bags,
        "by_class": by_class,
        "last_timestamp": last_timestamp,
        "camera_active": is_camera_active,
        "telemetry": telemetry_copy
    })


@app.route("/telemetry")
def telemetry():
    """Returns live telemetry of active objects currently tracked."""
    with _lock:
        is_camera_active = (time.time() - _last_frame_time) < 4.0 and _frame is not None
        data = dict(_telemetry)
        data["camera_active"] = is_camera_active
    return jsonify(data)


@app.route("/snapshots/<path:name>")
def snapshot(name):
    """Serves snapshot image file safely."""
    clean_name = os.path.basename(name)
    file_path = os.path.join(SNAP_DIR, clean_name)
    if not os.path.exists(file_path):
        return ("Snapshot not found", 404)
    return send_from_directory(SNAP_DIR, clean_name)


@app.route("/api/test_alert", methods=["POST", "GET"])
def test_alert():
    """Creates a sample unattended event for testing and demo purposes."""
    bag_id = int(time.time()) % 1000
    class_name = "backpack"
    duration = 15.2
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_filename = f"bag{bag_id}_{timestamp}.jpg"
    snapshot_path = os.path.join(SNAP_DIR, snapshot_filename)

    test_img = np.zeros((360, 640, 3), dtype=np.uint8)
    test_img[:] = (30, 35, 45)
    cv2.rectangle(test_img, (180, 80), (460, 280), (0, 0, 255), 2)
    cv2.putText(
        test_img,
        f"TEST ALERT: {class_name} #{bag_id}",
        (190, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )
    cv2.imwrite(snapshot_path, test_img)

    alert_message = (
        f"Simulated Alert: An unattended {class_name} (ID {bag_id}) was detected "
        f"left without an owner for {int(duration)} seconds."
    )

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO events (bag_id, class_name, duration_seconds, timestamp, snapshot_path, alert_message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (bag_id, class_name, duration, timestamp, snapshot_path, alert_message)
        )
        conn.commit()
        conn.close()

        # Trigger desktop toast notification
        send_alert_notification(
            bag_id=bag_id,
            class_name=class_name,
            alert_text=alert_message,
            snapshot_path=snapshot_path
        )

        return jsonify({"status": "success", "message": "Test alert created", "bag_id": bag_id})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/camera/start", methods=["POST", "GET"])
def camera_start():
    """Starts or resumes the built-in detection engine."""
    det = get_detector()
    if not det.running:
        det.start(frame_callback=set_frame, telemetry_callback=update_telemetry)
    return jsonify({"status": "running", "source": det.source})


@app.route("/api/camera/stop", methods=["POST", "GET"])
def camera_stop():
    """Pauses the built-in detection engine."""
    det = get_detector()
    det.stop()
    return jsonify({"status": "stopped"})


@app.route("/api/camera/status", methods=["GET"])
def camera_status():
    det = get_detector()
    return jsonify({"running": det.running, "source": str(det.source)})


@app.route("/api/push_frame", methods=["POST"])
def push_frame():
    """Receives JPEG frames when an external pipeline runs."""
    global _frame, _last_frame_time
    data = request.get_data()
    if data:
        with _lock:
            _frame = data
            _last_frame_time = time.time()
    return ("", 204)


@app.route("/api/push_telemetry", methods=["POST"])
def push_telemetry():
    """Receives telemetry state when an external pipeline runs."""
    global _telemetry
    data = request.get_json(silent=True)
    if data:
        with _lock:
            _telemetry = {
                "bags": data.get("bags", []),
                "person_count": data.get("person_count", 0),
                "bag_count": data.get("bag_count", 0),
                "fps": data.get("fps", 0.0),
                "camera_active": True,
                "last_updated": time.time()
            }
    return ("", 204)


def is_port_in_use(port, host="127.0.0.1"):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_dashboard(host="127.0.0.1", port=5000, autostart_camera=True):
    """
    Starts the Flask dashboard in a background daemon thread.
    Can be called directly from main.py before the video loop starts.
    """
    global _external_dashboard_url

    if is_port_in_use(port, host):
        print(f"\n[DASHBOARD] An active dashboard server is already running on http://{host}:{port}.")
        print(f"[DASHBOARD] Forwarding camera stream and telemetry directly to it.\n")
        _external_dashboard_url = f"http://{host}:{port}"
        return

    if autostart_camera:
        det = get_detector()
        det.start(frame_callback=set_frame, telemetry_callback=update_telemetry)

    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False),
        daemon=True,
    )
    t.start()
    print(f"\n=======================================================")
    print(f"  DASHBOARD ONLINE: http://{host}:{port}")
    print(f"  Live video stream & alerts monitoring ready.")
    print(f"=======================================================\n")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting dashboard on http://127.0.0.1:{port}...")
    det = get_detector()
    det.start(frame_callback=set_frame, telemetry_callback=update_telemetry)
    app.run(host="127.0.0.1", port=port, threaded=True)
