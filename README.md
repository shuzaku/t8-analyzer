# Tekken 8 Video Analyzer

Analyzes Tekken 8 match videos to automatically detect on-screen UI events (Counter Hit, Heat activations, Rage Arts, wall splats, etc.), attributes each event to **P1 (left)** or **P2 (right)**, and exports per-player statistics as JSON, CSV, and charts.

---

## Features

- **OCR-based detection** (EasyOCR, no model training required — works out of the box)
- **Optional YOLO** model support via `--yolo` for higher accuracy with a trained model
- **Optional Roboflow** model support via `--roboflow`
- Per-player event attribution by horizontal screen position
- Cooldown logic to prevent double-counting the same popup
- `SUPERSEDES` rules to suppress partial-read sub-events (e.g. "HEAT" when "HEAT BURST" fired)
- JSON, CSV, and matplotlib chart export
- Optional **MongoDB** persistence
- **YouTube URL** input — download and analyze directly from a URL

---

## Detected Events

| Canonical Name  | Description                              |
|-----------------|------------------------------------------|
| COUNTER HIT     | Hit that interrupts the opponent         |
| COUNTER         | Standalone counter notification          |
| PUNISH          | Punish hit on a whiffed or unsafe move   |
| HEAT            | Heat system activation                   |
| HEAT BURST      | Burst attack while in Heat               |
| HEAT SMASH      | Smash attack while in Heat               |
| HEAT ENGAGER    | Move that activates Heat on hit          |
| RAGE ART        | Rage Art super move                      |
| RAGE DRIVE      | Rage Drive powered attack                |
| WALL SPLAT      | Opponent is slammed into the wall        |
| WALL BREAK      | Wall is destroyed, stage transition      |
| FLOOR BREAK     | Floor breaks, stage transition           |
| TORNADO         | Tornado launcher state                   |
| POWER CRUSH     | Armour move absorbing an attack          |
| THROW           | Throw landed                             |
| REVERSAL        | Defensive reversal move                  |
| JUST FRAME      | Tight-window just-frame input            |

> **Note:** OCR accuracy depends on video quality and resolution. 1080p60 footage gives the best results. Tune `_ELEMENT_HUE_RANGES` in `ui_detector.py` if you're seeing false positives.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended for speed):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Copy `.env.example` to `.env`

```bash
cp .env.example .env
```

Fill in `MONGO_URI` if you want MongoDB storage, or leave it blank to skip.

### 3. Run the analyzer

```bash
# Local video file
python main.py path/to/match.mp4

# YouTube URL
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Force CPU only
python main.py match.mp4 --cpu

# Custom output directory
python main.py match.mp4 --output-dir results/my_match

# Higher precision (check more frames)
python main.py match.mp4 --frame-skip 6

# Skip chart generation
python main.py match.mp4 --no-visualizations
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `video_path` | *(required)* | Local file path or YouTube URL |
| `--output-dir` | `output` | Folder for JSON, CSV, and charts |
| `--frame-skip` | `15` | Process every Nth frame (~4 checks/sec at 60 fps) |
| `--confidence` | `0.5` | OCR confidence threshold (0–1) |
| `--cooldown` | `2.0` | Seconds to suppress the same event after detection |
| `--cpu` | off | Force CPU mode (disables GPU OCR) |
| `--full-frame` | off | Scan full frame instead of center hit zone |
| `--no-visualizations` | off | Skip chart generation |
| `--roboflow MODEL_ID` | — | Use a Roboflow model (requires `ROBOFLOW_API_KEY` in `.env`) |
| `--yolo WEIGHTS` | — | Use a local YOLOv8 weights file |
| `--no-mongo` | off | Skip MongoDB save even if `MONGO_URI` is set |
| `--youtube-download-dir DIR` | temp | Where to save downloaded YouTube videos |
| `--keep-youtube-download` | off | Don't delete temp YouTube download after analysis |

---

## Output

All files are written to `--output-dir` (default: `output/`).

| File | Contents |
|------|----------|
| `stats.json` | Full analysis: summary, video info, per-player stats, all detections |
| `detections.csv` | Flat event log: `frame, time_seconds, timestamp, player, element` |
| `event_counts.png` | Grouped bar chart — P1 vs P2 event counts |
| `timeline.png` | Scatter plot of events over time |
| `events_per_minute.png` | Normalized event frequency bar chart |

### `stats.json` schema

```json
{
  "summary": {
    "total_events": 42,
    "video_duration_seconds": 123.4,
    "video_duration_timestamp": "2:03.400",
    "events_per_minute": 20.42,
    "unique_event_types": 5,
    "player_stats": {
      "P1": { "COUNTER HIT": 8, "HEAT": 3 },
      "P2": { "COUNTER HIT": 5, "WALL SPLAT": 2 }
    }
  },
  "video_info": { "fps": 60.0, "resolution": [1920, 1080], "..." : "..." },
  "player_stats": { "P1": {}, "P2": {} },
  "frame_detections": [
    { "frame": 300, "time_seconds": 5.0, "timestamp": "0:05.000", "player": "P1", "element": "COUNTER HIT" }
  ]
}
```

---

## MongoDB

Set `MONGO_URI` in `.env`. Each run inserts one document with the full analysis payload plus metadata:

```
analyzed_at    (UTC timestamp)
video_path     (resolved absolute path)
detector       ("ocr" | "roboflow" | "yolo")
detector_detail (model id / weights path, or null)
... + all stats.json fields
```

Default database: `tekken8_analyzer`, collection: `analyses`. Override with `MONGO_DATABASE` / `MONGO_COLLECTION` in `.env`.

---

## Roboflow Model

```bash
# Add to .env:
ROBOFLOW_API_KEY=your_key_here

# Run with Roboflow model:
python main.py match.mp4 --roboflow "workspace/tekken8-ui/1"
```

> Roboflow inference has a NumPy 2.x conflict with EasyOCR. Use a separate virtualenv if you need both.

---

## Tuning for Better Results

**Frame skip:** Lower values catch more events but are slower. `--frame-skip 6` gives ~10 checks per second at 60 fps and is a good balance for competitive footage.

**Confidence:** Lower `--confidence` catches more text but increases false positives. Start at `0.5` and adjust.

**Hue validation:** `_ELEMENT_HUE_RANGES` in `ui_detector.py` filters detections by the banner's background colour. Adjust HSV ranges if your footage has different colour grading or you're using SDR/HDR captures.

**Scan zone:** By default only the center vertical band (excluding top HUD ~15% and bottom health bars ~10%) is scanned. Use `--full-frame` to scan the entire frame if popups appear outside this zone.

---

## Project Structure

```
Tekken8-Analyzer/
├── main.py              # CLI entry point and pipeline orchestration
├── ui_detector.py       # EasyOCR-based Tekken 8 UI text detection
├── video_processor.py   # Frame loop, ROI, player attribution, cooldowns
├── stats_generator.py   # Summary stats, DataFrame, JSON/CSV, charts
├── mongo_store.py       # Optional MongoDB persistence
├── youtube_input.py     # YouTube URL detection and yt-dlp download
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── output/              # Generated output (gitignored)
```
# t8-analyzer
