import cv2
import mediapipe as mp
import numpy as np
import os
import random
import sys
import time
import urllib.request
from collections import deque
from PIL import Image

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    from mediapipe.python.solutions import face_mesh as _fm
    _FACE_CONNECTIONS: frozenset = (
        _fm.FACEMESH_FACE_OVAL
        | _fm.FACEMESH_LEFT_EYE
        | _fm.FACEMESH_RIGHT_EYE
        | _fm.FACEMESH_LEFT_EYEBROW
        | _fm.FACEMESH_RIGHT_EYEBROW
        | _fm.FACEMESH_LIPS
    )
except Exception:
    _FACE_CONNECTIONS = frozenset()

# ---------------------------------------------------------------------------
# Paths & model URLs
# ---------------------------------------------------------------------------
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

HAND_MODEL_PATH = os.path.join(ASSETS_DIR, "hand_landmarker.task")
HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_PATH = os.path.join(ASSETS_DIR, "face_landmarker.task")
FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
MIN_HAND_CONFIDENCE  = 0.7
MIN_TRACK_CONFIDENCE = 0.6
MIN_FACE_CONFIDENCE  = 0.5
DISPLAY_HOLD_FRAMES  = 15

MEME_POPUP_SIZE = (750, 750)
POPUP_WINDOW    = "MemePose — Gesture"

WAVE_HISTORY_LEN     = 20
WAVE_MIN_DELTA       = 0.10
WAVE_COOLDOWN_FRAMES = 45

SCUBA_HISTORY_LEN     = 25
SCUBA_MIN_X_RANGE     = 0.20   # horizontal travel of the waving hand (normalised)
SCUBA_PINCH_MAX_DIST  = 0.09   # 3-D distance thumb–index for a pinch (includes depth z)
SCUBA_COOLDOWN_FRAMES = 45

CONFIRM_FRAMES = 5  # consecutive frames a pose must be held before triggering

SQUINT_EAR_MIN = 0.12
SQUINT_EAR_MAX = 0.22

WINK_EAR_CLOSED = 0.10   # eye considered closed
WINK_EAR_OPEN   = 0.20   # other eye considered open

WIDE_EAR_MIN = 0.40      # both eyes wide open above this

# ---------------------------------------------------------------------------
# Hand landmark indices
# ---------------------------------------------------------------------------
TIP_THUMB  = 4;  IP_THUMB   = 3;  MCP_THUMB  = 2
TIP_INDEX  = 8;  PIP_INDEX  = 6
TIP_MIDDLE = 12; PIP_MIDDLE = 10
TIP_RING   = 16; PIP_RING   = 14
TIP_PINKY  = 20; PIP_PINKY  = 18

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Face landmark indices for EAR (left eye / right eye, 6 points each)
L_EYE = (263, 387, 385, 362, 380, 373)
R_EYE = (33,  160, 158, 133, 153, 144)


# ===========================================================================
# Gesture detection — single-hand static poses  →  GESTURES (PNG + GIF pool)
# ===========================================================================

def _up(lms: list, tip: int, pip: int) -> bool:
    return lms[tip].y < lms[pip].y


def is_peace_sign(lms: list) -> bool:
    # Index + middle clearly up (tip above MCP, not just PIP)
    index_up  = lms[TIP_INDEX].y  < lms[5].y   # MCP index  = 5
    middle_up = lms[TIP_MIDDLE].y < lms[9].y   # MCP middle = 9
    # Ring + pinky clearly curled (tip below their MCP)
    ring_down  = lms[TIP_RING].y  > lms[13].y  # MCP ring   = 13
    pinky_down = lms[TIP_PINKY].y > lms[17].y  # MCP pinky  = 17
    # Not at forehead (would be salute territory)
    not_salute = lms[0].y > 0.30
    return index_up and middle_up and ring_down and pinky_down and not_salute


def is_thumbs_up(lms: list) -> bool:
    # Thumb fully extended upward through all three joints
    thumb_up = lms[TIP_THUMB].y < lms[IP_THUMB].y < lms[MCP_THUMB].y
    # All four fingers curled past their MCP (stronger than just PIP)
    fingers_curled = (
        lms[TIP_INDEX].y  > lms[5].y
        and lms[TIP_MIDDLE].y > lms[9].y
        and lms[TIP_RING].y   > lms[13].y
        and lms[TIP_PINKY].y  > lms[17].y
    )
    return thumb_up and fingers_curled


# (label, detect fn, assets subfolder)  —  supports PNG and GIF
GESTURES: list[tuple] = [
    ("Peace Sign", is_peace_sign, "peace_sign"),
    ("Thumbs Up",  is_thumbs_up,  "thumbs_up"),
]


# ===========================================================================
# Gesture detection — single-hand static poses  →  POSE_GIF_GESTURES
# (same pool system as GESTURES, kept as a separate registry for clarity)
# ===========================================================================

def is_salute(lms: list) -> bool:
    """Flat hand held HORIZONTALLY at forehead level (fingers point sideways, not upward)."""
    if lms[0].y >= 0.38:
        return False
    # Fingers must be roughly horizontal: index tip at similar height as wrist
    dy_index = abs(lms[TIP_INDEX].y - lms[0].y)
    if dy_index > 0.12:          # fingers pointing too vertical → not a salute
        return False
    # Fingers must be extended laterally (not just bent forward)
    dx_index = abs(lms[TIP_INDEX].x - lms[0].x)
    if dx_index < 0.10:          # fingers not extended enough sideways
        return False
    return True


POSE_GIF_GESTURES: list[tuple] = [
    ("Salute", is_salute, "salute"),
]


# ===========================================================================
# Gesture detection — two-hand motion  →  wave
# ===========================================================================

def check_wave(history: deque) -> bool:
    valid = [h for h in history if h is not None]
    if len(valid) < int(WAVE_HISTORY_LEN * 0.8):
        return False
    d0, d1 = valid[-1][0] - valid[0][0], valid[-1][1] - valid[0][1]
    return (
        (d0 > WAVE_MIN_DELTA and d1 < -WAVE_MIN_DELTA) or
        (d0 < -WAVE_MIN_DELTA and d1 > WAVE_MIN_DELTA)
    )


# ===========================================================================
# Gesture detection — two-hand combo  →  scuba dance
# One hand pinches near nose, the other waves horizontally.
# ===========================================================================

def is_pinching_nose(lms: list) -> bool:
    # 3-D distance: works even when the hand faces the camera (z handles depth foreshortening)
    dx = lms[TIP_THUMB].x - lms[TIP_INDEX].x
    dy = lms[TIP_THUMB].y - lms[TIP_INDEX].y
    dz = lms[TIP_THUMB].z - lms[TIP_INDEX].z
    dist_3d = (dx * dx + dy * dy + dz * dz) ** 0.5
    return dist_3d < SCUBA_PINCH_MAX_DIST and lms[0].y < 0.75


def check_scuba_wave(history: deque) -> bool:
    valid = [x for x in history if x is not None]
    if len(valid) < int(SCUBA_HISTORY_LEN * 0.7):
        return False
    return max(valid) - min(valid) > SCUBA_MIN_X_RANGE


# ===========================================================================
# Gesture detection — face expressions  →  FACE_GESTURES
# ===========================================================================

def _ear(lms: list, p: tuple) -> float:
    v = (abs(lms[p[1]].y - lms[p[5]].y) + abs(lms[p[2]].y - lms[p[4]].y)) / 2
    h = abs(lms[p[0]].x - lms[p[3]].x)
    return v / h if h > 0 else 0.0


def is_squinting(lms: list) -> bool:
    ear_l = _ear(lms, L_EYE)
    ear_r = _ear(lms, R_EYE)
    return (
        SQUINT_EAR_MIN < ear_l < SQUINT_EAR_MAX
        and SQUINT_EAR_MIN < ear_r < SQUINT_EAR_MAX
    )


def is_winking(lms: list) -> bool:
    ear_l = _ear(lms, L_EYE)
    ear_r = _ear(lms, R_EYE)
    l_closed = ear_l < WINK_EAR_CLOSED
    r_closed = ear_r < WINK_EAR_CLOSED
    l_open   = ear_l > WINK_EAR_OPEN
    r_open   = ear_r > WINK_EAR_OPEN
    return (l_closed and r_open) or (r_closed and l_open)


def is_wide_eyes(lms: list) -> bool:
    ear_l = _ear(lms, L_EYE)
    ear_r = _ear(lms, R_EYE)
    return ear_l > WIDE_EAR_MIN and ear_r > WIDE_EAR_MIN


FACE_GESTURES: list[tuple] = [
    ("Squint",     is_squinting, "squint"),
    ("Wink",       is_winking,   "wink"),
    ("Wide Eyes",  is_wide_eyes, "wide_eyes"),
]


# ===========================================================================
# Asset loading  —  unified format
#
# Every asset in every pool is:  list[tuple[np.ndarray, int]]
#   • Static PNG  →  [(bgr_frame, -1)]          duration -1 = no animation
#   • Animated GIF →  [(f0, ms0), (f1, ms1), …]
#
# This means ALL folders accept both PNG and GIF files.
# ===========================================================================

def download_model(path: str, url: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {os.path.basename(path)}...")

    def _prog(n, bs, total):
        pct = min(100, int(n * bs * 100 / total)) if total > 0 else 0
        print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, path, reporthook=_prog)
    print("\nDone.")


def _load_raw(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[ERROR] Cannot read: {path}")
        return None
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = 255
    return img


def _overlay(bg: np.ndarray, ov: np.ndarray, x: int, y: int) -> np.ndarray:
    hb, wb = bg.shape[:2]
    ho, wo = ov.shape[:2]
    x1b, y1b = max(x, 0), max(y, 0)
    x2b, y2b = min(x + wo, wb), min(y + ho, hb)
    x1o, y1o = x1b - x, y1b - y
    x2o, y2o = x1o + (x2b - x1b), y1o + (y2b - y1b)
    if x2b <= x1b or y2b <= y1b:
        return bg
    roi_bg = bg[y1b:y2b, x1b:x2b]
    roi_ov = ov[y1o:y2o, x1o:x2o]
    alpha  = roi_ov[:, :, 3:4].astype(np.float32) / 255.0
    out    = bg.copy()
    out[y1b:y2b, x1b:x2b] = (alpha * roi_ov[:, :, :3] + (1 - alpha) * roi_bg).astype(np.uint8)
    return out


def _png_to_asset(path: str) -> list[tuple] | None:
    raw = _load_raw(path)
    if raw is None:
        return None
    large = cv2.resize(raw, MEME_POPUP_SIZE, interpolation=cv2.INTER_LANCZOS4)
    bg    = np.full((MEME_POPUP_SIZE[1], MEME_POPUP_SIZE[0], 3), 18, dtype=np.uint8)
    return [(_overlay(bg, large, 0, 0), -1)]   # -1 = static


def _gif_to_asset(path: str) -> list[tuple] | None:
    frames = []
    try:
        gif = Image.open(path)
        while True:
            bgr = cv2.cvtColor(np.array(gif.copy().convert("RGB")), cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr, MEME_POPUP_SIZE, interpolation=cv2.INTER_LANCZOS4)
            frames.append((bgr, max(gif.info.get("duration", 100), 20)))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames if frames else None


def load_asset_pool(folder: str) -> list[list[tuple]]:
    """Load every PNG and GIF from assets/<folder>/ into the unified asset format."""
    folder_path = os.path.join(ASSETS_DIR, folder)
    if not os.path.isdir(folder_path):
        print(f"[WARNING] Missing folder: assets/{folder}/")
        return []
    pool = []
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        lo    = fname.lower()
        if lo.endswith(".png"):
            asset = _png_to_asset(fpath)
            tag   = "PNG"
        elif lo.endswith(".gif"):
            asset = _gif_to_asset(fpath)
            tag   = f"GIF {len(asset)} frames" if asset else "GIF"
        else:
            continue
        if asset:
            pool.append(asset)
            print(f"  Loaded [{tag}]: {folder}/{fname}")
    if not pool:
        print(f"[WARNING] No PNG/GIF in assets/{folder}/")
    return pool


def get_frame(asset: list[tuple], t0: float) -> np.ndarray:
    """Return the correct display frame for a unified asset at the current time."""
    if len(asset) == 1 or asset[0][1] < 0:
        return asset[0][0]                     # static
    total = sum(d for _, d in asset)
    if total == 0:
        return asset[0][0]
    elapsed = ((time.time() - t0) * 1000) % total
    acc = 0
    for img, dur in asset:
        acc += dur
        if elapsed < acc:
            return img
    return asset[-1][0]


# ===========================================================================
# Drawing helpers
# ===========================================================================

def draw_hand_skeleton(frame: np.ndarray, lms: list, w: int, h: int) -> None:
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 200, 0), 2, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(frame, (px, py), 5, (255, 255, 255), -1)
        cv2.circle(frame, (px, py), 5, (0, 140, 0), 1)


def draw_face_landmarks(frame: np.ndarray, lms: list, w: int, h: int) -> None:
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for s, e in _FACE_CONNECTIONS:
        if s < len(pts) and e < len(pts):
            cv2.line(frame, pts[s], pts[e], (0, 190, 140), 1, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(frame, (px, py), 1, (180, 220, 255), -1)


def draw_label(frame: np.ndarray, text: str, w: int, h: int) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 1.1, 2
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    tx, ty = (w - tw) // 2, h - 30
    cv2.putText(frame, text, (tx + 2, ty + 2), font, scale, (0, 0, 0),    thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx,     ty),     font, scale, (0, 220, 255), thick,     cv2.LINE_AA)


_LOAD_WIN = "MemePose"
_LOAD_W, _LOAD_H = 620, 340


def _draw_loading(step: int, total: int, message: str) -> None:
    img = np.full((_LOAD_H, _LOAD_W, 3), 14, dtype=np.uint8)

    # Title
    font  = cv2.FONT_HERSHEY_DUPLEX
    title = "MemePose"
    (tw, th), _ = cv2.getTextSize(title, font, 2.2, 3)
    cv2.putText(img, title, ((_LOAD_W - tw) // 2, 110),
                font, 2.2, (0, 220, 255), 3, cv2.LINE_AA)

    # Subtitle dots animation (fake — uses step as frame index)
    dots = "." * ((step % 3) + 1)
    sub  = f"Loading{dots}"
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, sub, ((_LOAD_W - sw) // 2, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    # Progress bar
    bx, by, bw, bh = 60, 190, _LOAD_W - 120, 16
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (35, 35, 35), -1)
    filled = int(bw * min(step, total) / total) if total > 0 else 0
    if filled > 0:
        # gradient-ish: draw two overlapping rects
        cv2.rectangle(img, (bx, by), (bx + filled, by + bh), (0, 140, 100), -1)
        cv2.rectangle(img, (bx, by), (bx + filled, by + bh // 2), (0, 200, 150), -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60, 60, 60), 1)

    # Percentage
    pct = f"{int(min(step, total) * 100 / total)}%" if total > 0 else ""
    cv2.putText(img, pct, (bx + bw + 10, by + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    # Step message
    (mw, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, message, ((_LOAD_W - mw) // 2, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.imshow(_LOAD_WIN, img)
    cv2.waitKey(1)


def draw_mode_select(frame: np.ndarray) -> None:
    h, w   = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    title = "MEMEPOSEN"
    font  = cv2.FONT_HERSHEY_DUPLEX
    (tw, _), _ = cv2.getTextSize(title, font, 2.0, 3)
    cv2.putText(frame, title, ((w - tw) // 2, h // 2 - 90),
                font, 2.0, (0, 220, 255), 3, cv2.LINE_AA)

    for i, (key, label, desc) in enumerate([
        ("G", "Gesture Mode", "Hand signs & GIFs"),
        ("F", "Face Mode",    "Facial expressions & GIFs"),
    ]):
        bx, bw, bh = (w - 340) // 2, 340, 70
        by = h // 2 - 10 + i * (bh + 16)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 200, 200), 2)
        cv2.putText(frame, f"[{key}]  {label}", (bx + 18, by + 30),
                    font, 0.7, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (bx + 18, by + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, "[Q] Quit", ((w - 90) // 2, h // 2 + 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, "[M] back to this menu (in-session)", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)


# ===========================================================================
# Main
# ===========================================================================

def _gif_st(registry: list[tuple]) -> dict:
    return {lbl: {"until": 0.0, "asset": None, "t0": 0.0} for lbl, _, _ in registry}


def main() -> None:
    # Total loading steps: 2 models + 4 asset groups + 1 init
    TOTAL_STEPS = 7
    step = 0

    cv2.namedWindow(_LOAD_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_LOAD_WIN, _LOAD_W, _LOAD_H)

    for path, url, label in [
        (HAND_MODEL_PATH, HAND_MODEL_URL, "Downloading hand model..."),
        (FACE_MODEL_PATH, FACE_MODEL_URL, "Downloading face model..."),
    ]:
        _draw_loading(step, TOTAL_STEPS, label if not os.path.exists(path) else label.replace("Downloading", "Checking"))
        if not os.path.exists(path):
            download_model(path, url)
        step += 1
        _draw_loading(step, TOTAL_STEPS, "")

    _draw_loading(step, TOTAL_STEPS, "Loading gesture assets...")
    gesture_pools  = {lbl: load_asset_pool(f"gestures/{fld}") for lbl, _, fld in GESTURES}
    pose_gif_pools = {lbl: load_asset_pool(f"gestures/{fld}") for lbl, _, fld in POSE_GIF_GESTURES}
    step += 1
    _draw_loading(step, TOTAL_STEPS, "Loading wave & scuba assets...")
    wave_pool  = load_asset_pool("gestures/wave")
    scuba_pool = load_asset_pool("gestures/scuba")
    step += 1
    _draw_loading(step, TOTAL_STEPS, "Loading expression assets...")
    face_gif_pools = {lbl: load_asset_pool(f"expressions/{fld}") for lbl, _, fld in FACE_GESTURES}
    step += 1
    _draw_loading(step, TOTAL_STEPS, "Initializing AI models...")

    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=MIN_HAND_CONFIDENCE,
        min_tracking_confidence=MIN_TRACK_CONFIDENCE,
    )
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=MIN_FACE_CONFIDENCE,
        min_face_presence_confidence=MIN_FACE_CONFIDENCE,
        min_tracking_confidence=MIN_FACE_CONFIDENCE,
    )

    step += 1
    _draw_loading(step, TOTAL_STEPS, "Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    step += 1
    _draw_loading(step, TOTAL_STEPS, "Ready!")
    cv2.waitKey(400)

    mode        = None
    popup_open  = False
    start_time  = time.time()   # fixed at launch — never reset (MediaPipe needs monotonic timestamps)

    # --- Gesture Mode state ---
    static_hold  = 0;  static_label = None;  static_asset = None;  static_t0 = 0.0
    wave_hist    = deque(maxlen=WAVE_HISTORY_LEN)
    wave_until   = 0.0;  wave_cd = 0;  wave_asset = None;  wave_t0 = 0.0
    scuba_hist   = deque(maxlen=SCUBA_HISTORY_LEN)
    scuba_until  = 0.0;  scuba_cd = 0;  scuba_asset = None;  scuba_t0 = 0.0
    pose_gif_st  = _gif_st(POSE_GIF_GESTURES)
    # Confirmation counters: gesture must be held CONFIRM_FRAMES in a row before triggering
    g_confirm    = {lbl: 0 for lbl, _, _ in GESTURES}
    p_confirm    = {lbl: 0 for lbl, _, _ in POSE_GIF_GESTURES}
    f_confirm    = {lbl: 0 for lbl, _, _ in FACE_GESTURES}

    # --- Face Mode state ---
    face_gif_st  = _gif_st(FACE_GESTURES)

    with (
        mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lmk,
        mp_vision.FaceLandmarker.create_from_options(face_opts) as face_lmk,
    ):
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # ----------------------------------------------------------------
            # MODE SELECTION SCREEN
            # ----------------------------------------------------------------
            if mode is None:
                draw_mode_select(frame)
                cv2.imshow("MemePose", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('g'):
                    mode = 'gestures'
                    static_hold = 0; static_label = None; static_asset = None
                    wave_hist.clear();  wave_until  = 0.0;  wave_cd  = 0
                    scuba_hist.clear(); scuba_until = 0.0;  scuba_cd = 0
                    pose_gif_st = _gif_st(POSE_GIF_GESTURES)
                    g_confirm = {lbl: 0 for lbl, _, _ in GESTURES}
                    p_confirm = {lbl: 0 for lbl, _, _ in POSE_GIF_GESTURES}
                elif key == ord('f'):
                    mode = 'face'
                    face_gif_st = _gif_st(FACE_GESTURES)
                    f_confirm = {lbl: 0 for lbl, _, _ in FACE_GESTURES}
                elif key in (27, ord('q')):
                    break
                continue

            # ----------------------------------------------------------------
            # SHARED: MediaPipe image
            # ----------------------------------------------------------------
            ts    = int((time.time() - start_time) * 1000)
            mp_im = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            popup_img  = None
            label_text = None

            # ----------------------------------------------------------------
            # GESTURE MODE
            # ----------------------------------------------------------------
            if mode == 'gestures':
                res = hand_lmk.detect_for_video(mp_im, ts)

                # Collect which gestures are detected this frame (any hand)
                frame_g = set(); frame_p = set(); pinch_idx = None

                for i, lms in enumerate(res.hand_landmarks):
                    draw_hand_skeleton(frame, lms, w, h)
                    for lbl, fn, _ in GESTURES:
                        if fn(lms):
                            frame_g.add(lbl)
                    for lbl, fn, _ in POSE_GIF_GESTURES:
                        if fn(lms):
                            frame_p.add(lbl)
                    if pinch_idx is None and is_pinching_nose(lms):
                        pinch_idx = i

                # Update confirmation counters and resolve confirmed gesture
                detected_label = None
                for lbl, _, _ in GESTURES:
                    if lbl in frame_g:
                        g_confirm[lbl] = g_confirm.get(lbl, 0) + 1
                    else:
                        g_confirm[lbl] = 0
                    if g_confirm[lbl] >= CONFIRM_FRAMES and detected_label is None:
                        detected_label = lbl

                for lbl, _, _ in POSE_GIF_GESTURES:
                    if lbl in frame_p:
                        p_confirm[lbl] = p_confirm.get(lbl, 0) + 1
                    else:
                        p_confirm[lbl] = 0
                    if p_confirm[lbl] >= CONFIRM_FRAMES:
                        now   = time.time()
                        state = pose_gif_st[lbl]
                        if now >= state["until"]:
                            pool = pose_gif_pools.get(lbl, [])
                            if pool:
                                state["asset"] = random.choice(pool)
                                state["t0"]    = now
                        state["until"] = now + 1.0

                # --- Static hold logic ---
                if detected_label:
                    if static_hold == 0 or detected_label != static_label:
                        pool = gesture_pools.get(detected_label, [])
                        if pool:
                            static_asset = random.choice(pool)
                            static_t0    = time.time()
                    static_hold  = DISPLAY_HOLD_FRAMES
                    static_label = detected_label
                elif static_hold > 0:
                    static_hold -= 1

                # --- Wave (vertical seesaw) ---
                lms_list = res.hand_landmarks
                if len(lms_list) == 2:
                    wr = sorted([(l[0].x, l[0].y) for l in lms_list], key=lambda p: p[0])
                    wave_hist.append((wr[0][1], wr[1][1]))
                else:
                    wave_hist.append(None)

                if wave_cd > 0:
                    wave_cd -= 1
                if wave_cd == 0 and check_wave(wave_hist) and wave_pool:
                    now = time.time()
                    if now >= wave_until:
                        wave_asset = random.choice(wave_pool)
                        wave_t0    = now
                        wave_cd    = WAVE_COOLDOWN_FRAMES
                    wave_until = now + 1.0
                    wave_hist.clear()

                # --- Scuba (pinch nose + horizontal hand wave) ---
                wave_x = None
                if pinch_idx is not None:
                    for i, lms in enumerate(res.hand_landmarks):
                        if i != pinch_idx:
                            wave_x = lms[0].x

                scuba_hist.append(wave_x)

                if scuba_cd > 0:
                    scuba_cd -= 1
                if (scuba_cd == 0 and pinch_idx is not None
                        and check_scuba_wave(scuba_hist) and scuba_pool):
                    now = time.time()
                    if now >= scuba_until:
                        scuba_asset = random.choice(scuba_pool)
                        scuba_t0    = now
                        scuba_cd    = SCUBA_COOLDOWN_FRAMES
                    scuba_until = now + 1.0
                    scuba_hist.clear()

                # --- Resolve display (priority: scuba > wave > pose GIF > static) ---
                now = time.time()
                active_pose = max(
                    ((l, s) for l, s in pose_gif_st.items()
                     if now < s["until"] and s["asset"]),
                    key=lambda x: x[1]["until"], default=None,
                )

                if now < scuba_until and scuba_asset:
                    popup_img  = get_frame(scuba_asset, scuba_t0)
                    label_text = "Scuba"
                elif now < wave_until and wave_asset:
                    popup_img  = get_frame(wave_asset, wave_t0)
                    label_text = "Wave"
                elif active_pose:
                    lbl, state = active_pose
                    popup_img  = get_frame(state["asset"], state["t0"])
                    label_text = lbl
                elif static_hold > 0 and static_asset:
                    popup_img  = get_frame(static_asset, static_t0)
                    label_text = static_label

                if not label_text:
                    hints = ([l for l, _, _ in GESTURES + POSE_GIF_GESTURES]
                             + ["Wave", "Scuba"])
                    cv2.putText(frame, "  |  ".join(hints), (10, h - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (160, 160, 160), 1, cv2.LINE_AA)

            # ----------------------------------------------------------------
            # FACE MODE
            # ----------------------------------------------------------------
            elif mode == 'face':
                res = face_lmk.detect_for_video(mp_im, ts)

                frame_f = set()
                for face_lms in res.face_landmarks:
                    draw_face_landmarks(frame, face_lms, w, h)
                    for lbl, fn, _ in FACE_GESTURES:
                        if fn(face_lms):
                            frame_f.add(lbl)

                for lbl, _, _ in FACE_GESTURES:
                    if lbl in frame_f:
                        f_confirm[lbl] = f_confirm.get(lbl, 0) + 1
                    else:
                        f_confirm[lbl] = 0
                    if f_confirm[lbl] >= CONFIRM_FRAMES:
                        now   = time.time()
                        state = face_gif_st[lbl]
                        if now >= state["until"]:
                            pool = face_gif_pools.get(lbl, [])
                            if pool:
                                state["asset"] = random.choice(pool)
                                state["t0"]    = now
                        state["until"] = now + 1.0

                now         = time.time()
                active_face = max(
                    ((l, s) for l, s in face_gif_st.items()
                     if now < s["until"] and s["asset"]),
                    key=lambda x: x[1]["until"], default=None,
                )
                if active_face:
                    lbl, state = active_face
                    popup_img  = get_frame(state["asset"], state["t0"])
                    label_text = lbl

                if not label_text:
                    hints = [l for l, _, _ in FACE_GESTURES]
                    cv2.putText(frame, "  |  ".join(hints), (10, h - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (160, 160, 160), 1, cv2.LINE_AA)

            # ----------------------------------------------------------------
            # SHARED: popup + label + mode tag
            # ----------------------------------------------------------------
            if label_text:
                draw_label(frame, label_text, w, h)

            if popup_img is not None:
                if not popup_open:
                    cv2.namedWindow(POPUP_WINDOW, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(POPUP_WINDOW, *MEME_POPUP_SIZE)
                    popup_open = True
                cv2.imshow(POPUP_WINDOW, popup_img)
            elif popup_open:
                cv2.destroyWindow(POPUP_WINDOW)
                popup_open = False

            tag = "GESTURES" if mode == 'gestures' else "FACE"
            cv2.putText(frame, f"{tag}  [M] menu", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

            cv2.imshow("MemePose", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('m'):
                mode = None
                if popup_open:
                    cv2.destroyWindow(POPUP_WINDOW)
                    popup_open = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
