# SentinAI — Autonomous VLM-Powered Unattended Object Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8s-green.svg)](https://ultralytics.com/)
[![Google Gemma 4](https://img.shields.io/badge/AI%20Brain-Google%20Gemma%204%20(26B)-orange.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, real-time edge-to-AI security surveillance pipeline that detects abandoned luggage in high-traffic public spaces (airports, train stations, shopping malls), tracks spatial-temporal owner proximity, dispatches native desktop toast notifications with zero latency, and leverages **Google Gemma 4 Multimodal VLM** (`gemma-4-26b-a4b-it`) to visually analyze and describe unattended objects in natural language.

---

## 🌟 Key Features

* **Real-Time Edge Detection & Tracking**:
  * YOLOv8s with BoT-SORT multi-object tracking for continuous identity persistence across frames.
  * Tracks Person (class `0`), Backpack (class `24`), Handbag (class `26`), and Suitcase (class `28`).
* **Spatial-Temporal Proximity Engine**:
  * Edge-to-edge Euclidean bounding box distance computation.
  * 3-State Machine: `ATTENDED` (green) $\rightarrow$ `ORPHANED` (yellow) $\rightarrow$ `UNATTENDED` (red).
  * **3.0-Second Anti-Flicker Grace Period**: Prevents momentary YOLO tracker dropouts or lighting shifts from resetting the countdown timer.
* **Instant Non-Blocking Alert Dispatch (0 ms Delay)**:
  * Camera feed never freezes; continues running at a smooth 30 FPS.
  * Native **Windows 10/11 Desktop Toast Notifications** via `winotify` with sound and snapshot preview.
  * Real-time Web Audio alarm chime and pulsing red screen banner.
* **Multimodal VLM Intelligence (Google Gemma 4)**:
  * Captures high-resolution snapshot JPEGs and feeds them to `gemma-4-26b-a4b-it` via the Gemini API.
  * Produces natural-language security reports describing the object's color, branding, and immediate environment.
* **Security Operations Center (SOC) Dashboard**:
  * Live MJPEG video stream with bounding boxes, active tracking matrix, and live countdown timers.
  * Click-to-enlarge Lightbox zoom modal for incident snapshots.
  * True borderless full-screen kiosk mode (`F11` or on-screen buttons).
  * One-click demo runner and simulation tools.

---

## 🏗️ System Architecture

```
[ Camera / CCTV Feed ]
         │
         ▼
[ YOLOv8s + BoT-SORT Tracking ]
  • Person (Class 0)
  • Backpack (Class 24), Handbag (Class 26), Suitcase (Class 28)
         │
         ▼
[ Spatial-Temporal Proximity Engine ]
  • Euclidean edge distance (<=150px = Attended)
  • 3-State Machine: ATTENDED ➔ ORPHANED ➔ UNATTENDED
  • 3.0s Tracker Grace Period (Eliminates frame-drop flicker)
         │
         ├─── [ Instant Non-Blocking Dispatch (0ms Latency) ]
         │         ├─► SQLite Database (logs/events.db)
         │         ├─► High-Resolution Snapshot (logs/snapshots/)
         │         ├─► Windows 10/11 Desktop Toast Alert (winotify + sound)
         │         └─► Web SOC Dashboard (Chime sound + Pulsing Red Banner)
         │
         └─── [ Asynchronous AI Brain (Google Gemini API) ]
                   └─► Gemma 4 (gemma-4-26b-a4b-it) Multimodal Vision
                         • Analyzes snapshot JPEG + prompt
                         • Enriches SQLite record with natural-language description
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-username/unattended-object-detection.git
cd unattended-object-detection

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

Open `.env` and enter your free **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey):

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Verify Gemma 4 VLM Connection

```bash
python test_gemma.py
```

---

## 🎮 Running the System

### Option A: Interactive Demo Launcher (Recommended)
Double-click `run_demo.bat` or run:

```bash
python demo.py
```

* **Option 1**: Fast 15-second webcam demonstration (automatically opens browser).
* **Option 2**: Official 300-second (5-minute) production threshold.
* **Option 3**: Synthetic pipeline test (tests database, notification, and Gemma 4 without camera).
* **Option 4**: Standalone dashboard.
* **Option 5**: View SQLite incident log table.

### Option B: Direct Command Line

```bash
# Run with webcam
python main.py

# Run with recorded test video file
python main.py --source path/to/video.mp4

# Run with custom threshold (e.g. 300s production rule)
python main.py --threshold 300

# Run in headless mode (view live feed exclusively on web dashboard)
python main.py --no-window
```

Open **`http://127.0.0.1:5000`** in any browser. Press **`F11`** for full-screen kiosk monitoring!

---

## 📁 Repository Structure

```text
unattended-object-detection/
├── .env                       # API keys (GEMINI_API_KEY) - NEVER COMMIT
├── .env.example               # Sanitized environment template
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT Open Source License
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
├── test_gemma.py              # Verification script for Gemma 4 API
├── main.py                    # Root entrypoint wrapper
├── demo.py                    # Interactive demo runner
├── run_demo.bat               # 1-click Windows batch launcher
├── logs/
│   ├── events.db              # SQLite persistent incident database
│   └── snapshots/             # Timestamped incident snapshots (.jpg)
└── src/
    ├── alert_llm.py           # Gemma 4 multimodal VLM alert generator
    ├── dashboard.py           # Flask web server, telemetry & MJPEG stream
    ├── detector.py            # Modular DetectionEngine (YOLO + camera loop)
    ├── event_logger.py        # SQLite database logger and snapshot writer
    ├── logic.py               # Edge distance calculation & 3-state machine
    ├── main.py                # Main surveillance loop with CLI args
    ├── notifier.py            # Native Windows 10/11 toast notification dispatcher
    ├── view_logs.py           # Terminal table viewer for logged events
    └── templates/
        └── index.html         # Modern dark-mode SOC dashboard UI
```

---

## ⏱️ Separation Thresholds

* **Testing / Hackathon Demonstration**: Set to **`15 seconds`** for quick live evaluation.
* **Real-World Airport / Transit Standard**: Set to **`300 seconds (5 minutes)`**:
  ```bash
  python main.py --threshold 300
  ```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
