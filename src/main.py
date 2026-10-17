# Unattended Object Detection Script
import argparse
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
from dashboard import start_dashboard, set_frame, update_telemetry

# COCO class IDs we care about for this project
TARGET_CLASSES = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase"
}

BAG_CLASS_IDS = {24, 26, 28}
PERSON_CLASS_ID = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Unattended Object Detection with Real-Time Web Dashboard")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. '0') or path to video file"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web dashboard (default: 5000)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=15,
        help="Seconds of separation before object is flagged unattended (default: 15s)"
    )
    parser.add_argument(
        "--proximity",
        type=int,
        default=150,
        help="Pixel distance within which an object is considered attended (default: 150px)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="YOLO model weight file (default: yolov8s.pt or yolov8n.pt)"
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable local OpenCV GUI window (view stream on web dashboard only)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine video source (camera index vs video file path)
    if args.source.isdigit():
        video_src = int(args.source)
        cap = cv2.VideoCapture(video_src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_src)
    else:
        video_src = args.source
        cap = cv2.VideoCapture(video_src)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{args.source}'.")
        print("Tips:")
        print("  - If using a webcam, check if another application is using it.")
        print("  - You can also test with a video file: python src/main.py --source path/to/video.mp4")
        return

    # Check model path
    model_path = args.model
    if not os.path.exists(model_path):
        base_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
        if os.path.exists(base_model_path):
            model_path = base_model_path

    print(f"Loading YOLO model: {model_path}...")
    model = YOLO(model_path)

    # Initialize tracker and event logger
    ownership_tracker = OwnershipTracker(
        proximity_threshold=args.proximity,
        unattended_threshold_seconds=args.threshold
    )
    event_logger = EventLogger()

    # Start Web Dashboard background server
    start_dashboard(port=args.port)

    print(f"Monitoring started. Unattended threshold: {args.threshold}s.")
    print("Dashboard: http://127.0.0.1:" + str(args.port))
    if not args.no_window:
        print("Local display: Press 'q' in OpenCV window to quit.")
        cv2.namedWindow("Unattended Object Detection", cv2.WINDOW_NORMAL)

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            # If video file reached end, loop or break
            if not isinstance(video_src, int):
                print("End of video file reached.")
            else:
                print("Error: Failed to grab frame from camera.")
            break

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

        # Extract person boxes and bag boxes separately with their tracking IDs
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

        # Update ownership/proximity state for all currently tracked bags
        bag_statuses = ownership_tracker.update(person_boxes, bag_boxes)

        # Draw the base YOLO annotations (boxes + labels)
        annotated_frame = results[0].plot()

        # Log unattended events (snapshot + DB record + instant toast + LLM alert)
        for bag_id, info in bag_statuses.items():
            state = info["state"]

            if event_logger.should_log(bag_id, state):
                duration_sec = info["seconds_orphaned"]
                class_name = info["class_name"]

                # 1. Save high-resolution annotated snapshot
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                snap_name = f"bag{bag_id}_{ts}.jpg"
                snap_path = os.path.join(event_logger.snapshot_dir, snap_name)
                cv2.imwrite(snap_path, annotated_frame)

                # 2. Instant alert message (notified within milliseconds)
                instant_alert = (
                    f"Security Alert: An unattended {class_name} (ID {bag_id}) was detected "
                    f"left without an owner for {int(duration_sec)} seconds."
                )
                print(f"\n[ALERT DETECTED] {instant_alert}")

                # 3. Insert immediately into SQLite database so dashboard updates right now
                event_id, _ = event_logger.log_event(
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
                            event_logger.update_alert_message(eid, gemma_alert)
                    except Exception as e:
                        print(f"[GEMMA VLM BACKGROUND WARNING] {e}")

                threading.Thread(target=_enrich_with_gemma, daemon=True).start()

            if state == "attended":
                event_logger.reset_bag(bag_id)

        # Overlay ownership state on top of each bag's box
        for bag_id, info in bag_statuses.items():
            x1, y1, x2, y2 = info["box"]
            state = info["state"]
            seconds = info["seconds_orphaned"]

            # Color code: green = attended, yellow = orphaned, red = unattended
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

        # Stream annotated frame to web dashboard
        set_frame(annotated_frame)

        # Update real-time telemetry for dashboard cards
        active_bags_data = [
            {
                "bag_id": b_id,
                "class_name": b_info["class_name"],
                "state": b_info["state"],
                "seconds_orphaned": round(b_info["seconds_orphaned"], 1),
                "threshold": args.threshold,
            }
            for b_id, b_info in bag_statuses.items()
        ]

        update_telemetry({
            "bags": active_bags_data,
            "person_count": len(person_boxes),
            "bag_count": len(bag_boxes),
            "fps": round(fps, 1),
        })

        if not args.no_window:
            cv2.imshow("Unattended Object Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.001)

    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()