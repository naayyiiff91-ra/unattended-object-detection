# Modular Detection Engine for Unattended Object Detection
import time
import os
import threading
from datetime import datetime
import cv2
from ultralytics import YOLO

from event_logger import EventLogger
from logic import OwnershipTracker
from alert_llm import generate_alert
from notifier import send_alert_notification

# COCO class IDs we care about
TARGET_CLASSES = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase"
}

BAG_CLASS_IDS = {24, 26, 28}
PERSON_CLASS_ID = 0


class DetectionEngine:
    """
    Manages camera capture, YOLO tracking, ownership proximity logic,
    and event logging in a dedicated thread.
    """

    def __init__(self, source=0, threshold=15, proximity=150, model_path="yolov8s.pt"):
        self.source = source
        self.threshold = threshold
        self.proximity = proximity
        self.model_path = model_path
        self.running = False
        self.thread = None
        self.cap = None
        self.frame_callback = None
        self.telemetry_callback = None

        # Resolve model path
        if not os.path.exists(self.model_path):
            base_model = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.model_path)
            if os.path.exists(base_model):
                self.model_path = base_model

        self.ownership_tracker = OwnershipTracker(
            proximity_threshold=self.proximity,
            unattended_threshold_seconds=self.threshold
        )
        self.event_logger = EventLogger()

    def start(self, frame_callback=None, telemetry_callback=None):
        if self.running:
            return
        self.frame_callback = frame_callback
        self.telemetry_callback = telemetry_callback
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _run_loop(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap or not self.cap.isOpened():
            print(f"[DETECTOR ERROR] Could not open video source '{self.source}'")
            self.running = False
            return

        print(f"[DETECTOR] Loading YOLO model: {self.model_path}...")
        model = YOLO(self.model_path)
        print(f"[DETECTOR] Detection engine active on source: {self.source}")

        frame_count = 0
        fps = 0.0
        start_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if not isinstance(src, int):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue
                else:
                    time.sleep(0.05)
                    continue

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            frame = cv2.resize(frame, (960, 540))

            results = model.track(
                frame,
                classes=list(TARGET_CLASSES.keys()),
                persist=True,
                tracker="botsort.yaml",
                conf=0.15,
                imgsz=320,
                verbose=False
            )

            person_boxes = []
            bag_boxes = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = box
                    if class_id == PERSON_CLASS_ID:
                        person_boxes.append((int(track_id), (x1, y1, x2, y2)))
                    elif class_id in BAG_CLASS_IDS:
                        class_name = TARGET_CLASSES[class_id]
                        bag_boxes.append((int(track_id), (x1, y1, x2, y2), class_name))

            bag_statuses = self.ownership_tracker.update(person_boxes, bag_boxes)
            annotated_frame = results[0].plot()

            # Process unattended events
            for bag_id, info in bag_statuses.items():
                state = info["state"]
                if self.event_logger.should_log(bag_id, state):
                    duration_sec = info["seconds_orphaned"]
                    class_name = info["class_name"]

                    # 1. Save high-resolution annotated snapshot
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    snap_name = f"bag{bag_id}_{ts}.jpg"
                    snap_path = os.path.join(self.event_logger.snapshot_dir, snap_name)
                    cv2.imwrite(snap_path, annotated_frame)

                    # 2. Instant fast alert message (notified within milliseconds)
                    instant_alert = (
                        f"Security Alert: An unattended {class_name} (ID {bag_id}) was detected "
                        f"left without an owner for {int(duration_sec)} seconds."
                    )
                    print(f"\n[ALERT DETECTED] {instant_alert}")

                    # 3. Insert immediately into SQLite so dashboard updates right now
                    event_id, _ = self.event_logger.log_event(
                        bag_id=bag_id,
                        class_name=class_name,
                        duration_seconds=duration_sec,
                        snapshot_path=snap_path,
                        alert_message=instant_alert
                    )

                    # 4. Dispatch native Windows 10/11 desktop toast notification immediately
                    send_alert_notification(
                        bag_id=bag_id,
                        class_name=class_name,
                        alert_text=instant_alert,
                        snapshot_path=snap_path
                    )

                    # 5. Background thread: Ask Gemma 4 multimodal vision to inspect snapshot & enrich record
                    def _enrich_with_gemma(eid=event_id, bid=bag_id, cname=class_name, dur=duration_sec, spath=snap_path):
                        try:
                            gemma_alert = generate_alert(
                                bag_id=bid,
                                class_name=cname,
                                duration_seconds=dur,
                                snapshot_path=spath
                            )
                            if gemma_alert and gemma_alert != instant_alert:
                                print(f"[GEMMA VLM ENRICHED] Bag #{bid}: {gemma_alert}")
                                self.event_logger.update_alert_message(eid, gemma_alert)
                        except Exception as e:
                            print(f"[GEMMA VLM BACKGROUND WARNING] {e}")

                    threading.Thread(target=_enrich_with_gemma, daemon=True).start()

                if state == "attended":
                    self.event_logger.reset_bag(bag_id)

            # Draw labels
            for bag_id, info in bag_statuses.items():
                x1, y1, x2, y2 = info["box"]
                state = info["state"]
                seconds = info["seconds_orphaned"]

                if state == "attended":
                    color = (0, 255, 0)
                    label = f"{info['class_name']} #{bag_id}: ATTENDED"
                elif state == "orphaned":
                    color = (0, 255, 255)
                    label = f"{info['class_name']} #{bag_id}: ORPHANED ({int(seconds)}s)"
                else:
                    color = (0, 0, 255)
                    label = f"{info['class_name']} #{bag_id}: UNATTENDED ({int(seconds)}s)"

                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # Frame callback
            if self.frame_callback:
                self.frame_callback(annotated_frame)

            # Telemetry callback
            if self.telemetry_callback:
                active_bags_data = [
                    {
                        "bag_id": b_id,
                        "class_name": b_info["class_name"],
                        "state": b_info["state"],
                        "seconds_orphaned": round(b_info["seconds_orphaned"], 1),
                        "threshold": self.threshold,
                    }
                    for b_id, b_info in bag_statuses.items()
                ]
                self.telemetry_callback({
                    "bags": active_bags_data,
                    "person_count": len(person_boxes),
                    "bag_count": len(bag_boxes),
                    "fps": round(fps, 1),
                })

            time.sleep(0.01)

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

from event_logger import EventLogger
from logic import OwnershipTracker
from alert_llm import generate_alert

# COCO class IDs we care about
TARGET_CLASSES = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase"
}

BAG_CLASS_IDS = {24, 26, 28}
PERSON_CLASS_ID = 0


class DetectionEngine:
    """
    Manages camera capture, YOLO tracking, ownership proximity logic,
    and event logging in a dedicated thread.
    """

    def __init__(self, source=0, threshold=15, proximity=150, model_path="yolov8s.pt"):
        self.source = source
        self.threshold = threshold
        self.proximity = proximity
        self.model_path = model_path
        self.running = False
        self.thread = None
        self.cap = None
        self.frame_callback = None
        self.telemetry_callback = None

        # Resolve model path
        if not os.path.exists(self.model_path):
            base_model = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.model_path)
            if os.path.exists(base_model):
                self.model_path = base_model

        self.ownership_tracker = OwnershipTracker(
            proximity_threshold=self.proximity,
            unattended_threshold_seconds=self.threshold
        )
        self.event_logger = EventLogger()

    def start(self, frame_callback=None, telemetry_callback=None):
        if self.running:
            return
        self.frame_callback = frame_callback
        self.telemetry_callback = telemetry_callback
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _run_loop(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap or not self.cap.isOpened():
            print(f"[DETECTOR ERROR] Could not open video source '{self.source}'")
            self.running = False
            return

        print(f"[DETECTOR] Loading YOLO model: {self.model_path}...")
        model = YOLO(self.model_path)
        print(f"[DETECTOR] Detection engine active on source: {self.source}")

        frame_count = 0
        fps = 0.0
        start_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if not isinstance(src, int):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue
                else:
                    time.sleep(0.05)
                    continue

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            frame = cv2.resize(frame, (960, 540))

            results = model.track(
                frame,
                classes=list(TARGET_CLASSES.keys()),
                persist=True,
                tracker="botsort.yaml",
                conf=0.15,
                imgsz=320,
                verbose=False
            )

            person_boxes = []
            bag_boxes = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = box
                    if class_id == PERSON_CLASS_ID:
                        person_boxes.append((int(track_id), (x1, y1, x2, y2)))
                    elif class_id in BAG_CLASS_IDS:
                        class_name = TARGET_CLASSES[class_id]
                        bag_boxes.append((int(track_id), (x1, y1, x2, y2), class_name))

            bag_statuses = self.ownership_tracker.update(person_boxes, bag_boxes)
            annotated_frame = results[0].plot()

            # Process unattended events
            for bag_id, info in bag_statuses.items():
                state = info["state"]
                if self.event_logger.should_log(bag_id, state):
                    alert_message = generate_alert(
                        bag_id=bag_id,
                        class_name=info["class_name"],
                        duration_seconds=info["seconds_orphaned"]
                    )
                    print(f"[ALERT] {alert_message}")
                    self.event_logger.log_event(
                        bag_id=bag_id,
                        class_name=info["class_name"],
                        duration_seconds=info["seconds_orphaned"],
                        frame=annotated_frame,
                        alert_message=alert_message
                    )
                if state == "attended":
                    self.event_logger.reset_bag(bag_id)

            # Draw labels
            for bag_id, info in bag_statuses.items():
                x1, y1, x2, y2 = info["box"]
                state = info["state"]
                seconds = info["seconds_orphaned"]

                if state == "attended":
                    color = (0, 255, 0)
                    label = f"{info['class_name']} #{bag_id}: ATTENDED"
                elif state == "orphaned":
                    color = (0, 255, 255)
                    label = f"{info['class_name']} #{bag_id}: ORPHANED ({int(seconds)}s)"
                else:
                    color = (0, 0, 255)
                    label = f"{info['class_name']} #{bag_id}: UNATTENDED ({int(seconds)}s)"

                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # Frame callback
            if self.frame_callback:
                self.frame_callback(annotated_frame)

            # Telemetry callback
            if self.telemetry_callback:
                active_bags_data = [
                    {
                        "bag_id": b_id,
                        "class_name": b_info["class_name"],
                        "state": b_info["state"],
                        "seconds_orphaned": round(b_info["seconds_orphaned"], 1),
                        "threshold": self.threshold,
                    }
                    for b_id, b_info in bag_statuses.items()
                ]
                self.telemetry_callback({
                    "bags": active_bags_data,
                    "person_count": len(person_boxes),
                    "bag_count": len(bag_boxes),
                    "fps": round(fps, 1),
                })

            time.sleep(0.01)

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

