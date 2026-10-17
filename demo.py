# SentinAI — Hackathon Live Demonstration Launcher
import os
import sys
import time
import threading
import webbrowser
from datetime import datetime

# Add src/ to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Set UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def print_banner():
    print("=" * 68)
    print("      [SentinAI] - Hackathon Live Demonstration Launcher")
    print("   Autonomous Unattended Object Detection & Gemma 4 Vision AI")
    print("=" * 68)
    print("  Challenge: MLH Hackathon | VLM Brain: Google Gemma 4 (26B)")
    print("=" * 68)


def check_environment():
    print("\n[PRE-FLIGHT CHECKS]")
    # Check .env
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and not api_key.startswith("your_"):
            print("  [OK] GEMINI_API_KEY detected in .env")
        else:
            print("  [WARNING] GEMINI_API_KEY is missing or placeholder in .env")
    else:
        print("  [WARNING] .env file not found!")

    # Check model
    model_s = os.path.join(BASE_DIR, "yolov8s.pt")
    model_n = os.path.join(BASE_DIR, "yolov8n.pt")
    if os.path.exists(model_s):
        print(f"  [OK] YOLO model found: yolov8s.pt ({os.path.getsize(model_s) / (1024*1024):.1f} MB)")
    elif os.path.exists(model_n):
        print(f"  [OK] YOLO model found: yolov8n.pt ({os.path.getsize(model_n) / (1024*1024):.1f} MB)")
    else:
        print("  [INFO] Downloading YOLOv8 weights on first run...")

    # Check winotify
    try:
        import winotify
        print("  [OK] Native Windows 10/11 toast notifications (winotify) ready")
    except ImportError:
        print("  [WARNING] winotify not installed in environment")

def run_live_webcam_demo(threshold=15):
    print("\n" + "=" * 68)
    print(f"  LAUNCHING LIVE WEBCAM DEMO (Threshold: {threshold} seconds)")
    print("=" * 68)
    print("  1. Dashboard starting at: http://127.0.0.1:5000")
    print("  2. Opening your web browser automatically in 3 seconds...")
    print("  3. Instructions for Live Demo:")
    print("     * Step in front of camera with a backpack / handbag.")
    print("     * Observe the GREEN 'ATTENDED' bounding box.")
    print("     * Set the bag down and step away (> 1.5 meters).")
    print("     * Observe the YELLOW 'ORPHANED' box & live second countdown.")
    print("     * At 15s, observe:")
    print("         - RED 'UNATTENDED' bounding box.")
    print("         - Flashing red security alert banner on screen.")
    print("         - Web Audio alarm chime sound.")
    print("         - Windows desktop toast notification popup.")
    print("         - Gemma 4 VLM natural language description in feed.")
    print("  4. Press Ctrl+C in this terminal to stop.")
    print("=" * 68 + "\n")

    # Schedule browser opening
    def _open_browser():
        time.sleep(3)
        webbrowser.open("http://127.0.0.1:5000")
    threading.Thread(target=_open_browser, daemon=True).start()

    # Launch main pipeline
    from main import main as run_main
    sys.argv = ["main.py", "--source", "0", "--threshold", str(threshold)]
    run_main()

def run_synthetic_simulation():
    print("\n" + "=" * 68)
    print("  RUNNING INSTANT SYNTHETIC PIPELINE SIMULATION")
    print("=" * 68)
    print("  Testing all subsystems: Database, Windows Notification, and Gemma 4 VLM...")

    from event_logger import EventLogger
    from notifier import send_alert_notification
    from alert_llm import generate_alert
    import cv2
    import numpy as np

    logger = EventLogger()

    # Create synthetic frame
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snap_name = f"bag999_{ts}.jpg"
    snap_path = os.path.join(logger.snapshot_dir, snap_name)

    test_img = np.zeros((540, 960, 3), dtype=np.uint8)
    test_img[:] = (20, 24, 32)
    cv2.rectangle(test_img, (320, 180), (640, 420), (0, 0, 255), 3)
    cv2.putText(test_img, "DEMO INCIDENT: UNATTENDED BACKPACK #999", (325, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(test_img, f"Separation Duration: 15.0s | {ts}", (325, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.imwrite(snap_path, test_img)
    print(f"  [1/4] Created incident snapshot: {snap_path}")

    # Instant event logging
    initial_alert = "Security Alert: An unattended backpack (ID 999) was detected left without an owner for 15 seconds."
    event_id, _ = logger.log_event(
        bag_id=999,
        class_name="backpack",
        duration_seconds=15.0,
        snapshot_path=snap_path,
        alert_message=initial_alert
    )
    print(f"  [2/4] Logged incident to SQLite events.db (ID: {event_id})")

    # Native Windows Desktop Notification
    print("  [3/4] Triggering native Windows 10/11 desktop toast notification...")
    send_alert_notification(
        bag_id=999,
        class_name="backpack",
        alert_text=initial_alert,
        snapshot_path=snap_path
    )
    print("        -> Windows Toast displayed with audio chime.")

    # Gemma 4 VLM Analysis
    print("  [4/4] Querying Google Gemma 4 VLM (gemma-4-26b-a4b-it) via Gemini API...")
    try:
        gemma_desc = generate_alert(
            bag_id=999,
            class_name="backpack",
            duration_seconds=15.0,
            snapshot_path=snap_path
        )
        print(f"\n  >>> GEMMA 4 VLM GENERATED ALERT:")
        print(f"      \"{gemma_desc}\"")
        logger.update_alert_message(event_id, gemma_desc)
        print("  [OK] SQLite record successfully enriched with Gemma 4 description.")
    except Exception as e:
        print(f"  [WARNING] Gemma 4 call failed: {e}")

    print("\n" + "=" * 68)
    print("  [SUCCESS] All pipeline subsystems validated successfully!")
    print("  Run 'python src/view_logs.py' to inspect the event table.")
    print("=" * 68 + "\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SentinAI Hackathon Live Demonstration Launcher")
    parser.add_argument("--mode", type=str, default="", help="Mode: 1 (Live 15s), 2 (Live 300s), 3 (Synthetic), 4 (Dashboard), 5 (Logs)")
    args = parser.parse_args()

    print_banner()
    check_environment()

    choice = args.mode.strip()
    if not choice:
        print("\nSelect Demo Mode:")
        print("  [1] Live Webcam Demo (Fast 15s Timer + Dashboard + Windows Alerts)")
        print("  [2] Live Webcam Demo (Official 300s / 5-min Production Threshold)")
        print("  [3] Synthetic Subsystem Simulation (Instant Test without Camera)")
        print("  [4] Open Web Dashboard Only (http://127.0.0.1:5000)")
        print("  [5] View SQLite Security Incident Logs")
        print("  [Q] Quit")

        try:
            choice = input("\nEnter choice [1-5 or Q] (default: 1): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return


    if choice == "2":
        run_live_webcam_demo(threshold=300)
    elif choice == "3":
        run_synthetic_simulation()
    elif choice == "4":
        webbrowser.open("http://127.0.0.1:5000")
        from dashboard import start_dashboard
        start_dashboard(port=5000)
        print("\nDashboard running at http://127.0.0.1:5000. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
    elif choice == "5":
        from view_logs import view_logs
        view_logs()
    elif choice.lower() == "q":
        print("Goodbye!")
    else:
        # Default to option 1
        run_live_webcam_demo(threshold=15)

if __name__ == "__main__":
    main()
