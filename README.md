<div align="right">
  <a href="README.fr.md">🇫🇷 Lire en français</a>
</div>

<div align="center">
  <h1>✌️ MemePose</h1>
  <p><strong>Real-time gesture-triggered meme overlay using your webcam</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/MediaPipe-0.10%2B-green?logo=google&logoColor=white" alt="MediaPipe">
    <img src="https://img.shields.io/badge/OpenCV-4.8%2B-red?logo=opencv&logoColor=white" alt="OpenCV">
    <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
  </p>
</div>

---

## What is MemePose?

MemePose detects hand gestures in real time through your webcam and overlays a meme image on screen when a specific gesture is recognized.

Flash the **peace sign ✌️** → a meme pops up with the text **"HAMSTER DETECTED!"**

The hand skeleton (21 landmarks) is always displayed to give visual feedback on what the AI is tracking.

---

## Demo

> *Flash a ✌️ at your camera — instant meme.*

| No gesture | Peace sign detected |
|---|---|
| Hand skeleton tracking | Meme overlay + banner |

---

## Features

- **Real-time hand tracking** via MediaPipe Hand Landmarker (Tasks API)
- **Mirror mode** — movements feel natural on screen
- **Alpha-blended PNG overlay** — transparent memes with no background artifacts
- **Anti-flicker** — meme stays visible for a few frames after the gesture ends
- **Plug-and-play assets** — drop any PNG into `assets/` and rename it

---

## Requirements

- Python 3.10+
- A webcam

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/MemePose.git
cd MemePose

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> The MediaPipe hand landmark model (~3 MB) is downloaded automatically on first run.

---

## Usage

```bash
python main.py
```

| Key | Action |
|-----|--------|
| `Q` or `Esc` | Quit |

---

## Project Structure

```
MemePose/
├── main.py                   # Application entry point
├── generate_placeholder.py   # Utility: generates a placeholder meme PNG
├── requirements.txt
├── .gitignore
└── assets/
    └── hamster.png           # Meme PNG (BGRA with transparency)
```

---

## Adding Your Own Memes

1. Prepare a **PNG file with a transparent background** (BGRA / RGBA).
2. Drop it into `assets/`.
3. Update `MEME_FILENAME` in [main.py](main.py):

```python
MEME_FILENAME = "your_meme.png"
```

No real meme yet? Run the placeholder generator:

```bash
python generate_placeholder.py
```

---

## How the Gesture Detection Works

MediaPipe returns 21 normalized landmarks (x, y, z) for each detected hand.

The **peace sign ✌️** is validated when:
- Index fingertip (point 8) is **above** its second knuckle (point 6)
- Middle fingertip (point 12) is **above** its second knuckle (point 10)
- Ring fingertip (point 16) is **below** its second knuckle (point 14)
- Pinky fingertip (point 20) is **below** its second knuckle (point 18)

*"Above" = smaller Y value in image coordinates (origin is top-left).*

---

## Configuration

All tunable parameters are at the top of [main.py](main.py). If detections feel off, start with the face EAR thresholds.

### General

| Constant | Default | Description |
|---|---|---|
| `CONFIRM_FRAMES` | `5` | Consecutive frames a pose must be held before triggering — raise if too sensitive, lower if sluggish |
| `DISPLAY_HOLD_FRAMES` | `15` | Frames the meme stays visible after the gesture ends |
| `MIN_HAND_CONFIDENCE` | `0.7` | Minimum confidence to detect a hand |
| `MIN_TRACK_CONFIDENCE` | `0.6` | Minimum confidence to keep tracking a hand |
| `MIN_FACE_CONFIDENCE` | `0.5` | Minimum confidence to detect a face |

### Face expressions — EAR (Eye Aspect Ratio)

EAR measures how open an eye is: `vertical height / horizontal width`. A fully closed eye is near `0.0`; a wide-open eye is typically `0.25–0.35`. **These values vary a lot between people** — someone with naturally large eyes will have a higher resting EAR than average.

> **How to calibrate:** add a `print(ear_l, ear_r)` call inside `is_squinting()` and watch the console while you hold each expression. Use those values to set your thresholds.

| Constant | Default | What it controls |
|---|---|---|
| `SQUINT_EAR_MIN` | `0.12` | Lower bound of squint — below this the eye is considered closed (wink zone) |
| `SQUINT_EAR_MAX` | `0.22` | Upper bound of squint — above this the eye is considered normal/open |
| `WINK_EAR_CLOSED` | `0.10` | One eye must be below this to count as closed for a wink |
| `WINK_EAR_OPEN` | `0.20` | The other eye must be above this to confirm it is open during a wink |
| `WIDE_EAR_MIN` | `0.48` | Both eyes must exceed this to trigger "wide eyes" — **raise this if it triggers at rest** |

**Example — naturally large eyes:** if your resting EAR is around `0.38`, set `WIDE_EAR_MIN = 0.52` so only a deliberate surprised look triggers it.

**Example — small/narrow eyes:** if squinting never triggers, lower `SQUINT_EAR_MAX` from `0.22` to `0.18`.

### Motion gestures

| Constant | Default | Description |
|---|---|---|
| `WAVE_MIN_DELTA` | `0.10` | Minimum vertical displacement each hand must travel for a wave |
| `SCUBA_MIN_X_RANGE` | `0.20` | Minimum horizontal travel of the waving hand for scuba |
| `SCUBA_PINCH_MAX_DIST` | `0.09` | Maximum 3D distance between thumb and index to count as a pinch |

---

## License

MIT — see [LICENSE](LICENSE) for details.
