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

All tunable parameters are at the top of [main.py](main.py):

| Constant | Default | Description |
|---|---|---|
| `MIN_DETECTION_CONFIDENCE` | `0.7` | Minimum confidence to detect a hand |
| `MIN_TRACKING_CONFIDENCE` | `0.6` | Minimum confidence to keep tracking |
| `DISPLAY_HOLD_FRAMES` | `15` | Frames to keep the meme visible after the gesture ends |
| `MEME_FILENAME` | `hamster.png` | PNG file to overlay |

---

## License

MIT — see [LICENSE](LICENSE) for details.
