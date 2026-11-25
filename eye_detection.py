import os
import queue
import threading
import time
from types import SimpleNamespace

import cv2
import joblib
import mediapipe as mp
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

import morse_config
from blink_detector import BlinkDetector, BlinkSettings, estimate_gaze
from calibration_profile import load_profile
from communication_engine import (
    ALPHABET,
    QUICK,
    CommunicationEngine,
    EngineSettings,
)
from message_editor import MessageEditor
from settings import load_settings, save_settings
from ui_renderer import (
    CANVAS_H,
    CANVAS_W,
    UIInfo,
    fit_to_window,
    render,
    window_to_canvas,
)


# ============================================================
# BlinkTap - Assistive Communication System
#
# Morse-code communication aid for CONSCIOUS paralysis patients who
# can intentionally control eye blinks and/or a tap input.  This is an
# assistive communication tool, not a medical diagnosis, treatment or
# patient-monitoring system.
#
# File layout:
#   eye_detection.py        this file: camera loop, MediaPipe, TTS, menu
#   blink_detector.py       blink / closure state machine (gaze aware)
#   communication_engine.py Morse decoding, modes, confirm, editing
#   morse_config.py         A-Z Morse table, control commands, quick messages
#   quick_messages.json     custom messages + patterns (edit in app: E key)
#   settings.py/.json       dot/dash ranges, letter/word timeouts
#   message_editor.py       caregiver editor for the custom messages
#   ui_renderer.py          on-screen interface
#   calibration.py          patient calibration + ML sample collection
#   calibration_profile.py  patient thresholds (calibration_profile.json)
# ============================================================


# ============================================================
# CONFIGURATION  (original BlinkTap values; a calibration profile,
# if present, overrides the blink thresholds)
# ============================================================

APP_NAME = "BlinkTap"

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(HERE, "models", "face_landmarker.task")
ML_MODEL_PATH = os.path.join(HERE, "blink_model.pkl")

# Blink detection.  Dot/dash ranges, letter and word timeouts are NOT here:
# they live in settings.json (see settings.py) and can be changed in the app
# or by calibration.py.
CLOSED_THRESHOLD = 0.21
OPEN_THRESHOLD = 0.235
CLOSED_CONFIRM_FRAMES = 2
OPEN_CONFIRM_FRAMES = 5
MIN_BLINK_DURATION = 0.06
MAX_BLINK_DURATION = 4.0
BLINK_COOLDOWN = 0.12
EAR_HISTORY_SIZE = 3

# Eye goodbye (deliberate continuous eye-closure hold)
GOODBYE_HOLD_DURATION = 3.50

# Temporary face / geometry failure tolerance
MAX_FACE_LOST_FRAMES = 12
MAX_GEOMETRY_BAD_FRAMES = 12

# Head orientation
MAX_EYE_WIDTH_RATIO = 4.0
MAX_NOSE_OFFSET_RATIO = 0.45

# Tap
TAP_MIN_DURATION = 0.06
TAP_MAX_DURATION = 5.0
TAP_LONG_THRESHOLD = 0.30
TAP_GOODBYE_DURATION = 3.00

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Confirmation prompt
CONFIRM_TIMEOUT = 20.0
QUICK_REQUIRES_CONFIRMATION = True

ML_FEATURE_NAMES = [
    "left_ear",
    "right_ear",
    "average_ear",
    "min_ear",
    "ear_difference",
    "eye_width_ratio",
    "nose_offset_ratio",
    "blink_duration",
]

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Presets selectable from the command line:  python eye_detection.py --preset responsive
# "responsive" reproduces the settings of the old eye_detection_responsive.py
# (shorter eye goodbye, longer tap goodbye, stricter head angle, no ML).
PRESETS = {
    "responsive": dict(
        GOODBYE_HOLD_DURATION=1.50,
        TAP_GOODBYE_DURATION=4.50,
        TAP_LONG_THRESHOLD=0.40,
        MAX_EYE_WIDTH_RATIO=3.5,
        MAX_NOSE_OFFSET_RATIO=0.35,
        USE_ML=False,
    ),
}


def build_config(preset=None):
    """Collect the module constants into one namespace (plus preset)."""
    cfg = {
        name: value
        for name, value in globals().items()
        if name.isupper() and not name.startswith("_")
    }
    cfg.setdefault("USE_ML", True)
    if preset:
        if preset not in PRESETS:
            raise SystemExit(
                f"Unknown preset {preset!r}. Available: {', '.join(PRESETS)}"
            )
        cfg.update(PRESETS[preset])
    return SimpleNamespace(**cfg)


# ============================================================
# TTS  (unchanged: persistent worker, fresh engine per request)
# ============================================================

tts_queue = queue.Queue()
tts_thread = None
tts_thread_lock = threading.Lock()
tts_shutdown_requested = False


def _tts_worker():
    """
    Dedicated TTS worker.

    A fresh pyttsx3 engine is created for every request.
    This is intentional for Windows/SAPI reliability.
    """

    pythoncom = None

    try:
        import pythoncom

        pythoncom.CoInitialize()

    except Exception:
        pythoncom = None

    print("TTS worker started.")

    try:

        while True:

            text = tts_queue.get()

            try:

                if text is None:
                    break

                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.setProperty("volume", 1.0)
                engine.say(str(text))
                engine.runAndWait()

                try:
                    engine.stop()
                except Exception:
                    pass

                print(f"Speaker: {str(text)!r}")

            except Exception as e:

                print("TTS error:", repr(e))

            finally:

                tts_queue.task_done()

    finally:

        if pythoncom is not None:

            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

        print("TTS worker stopped.")


def initialize_tts():

    global tts_thread

    with tts_thread_lock:

        if tts_thread is not None and tts_thread.is_alive():
            return

        tts_thread = threading.Thread(
            target=_tts_worker,
            name="BlinkTap-TTS",
            daemon=True,
        )
        tts_thread.start()


def notify_sound(kind):
    """Short beep: one low beep = letter finished, two higher = word finished."""

    def _play():
        try:
            import winsound

            if kind == "LETTER":
                winsound.Beep(880, 120)
            elif kind == "PAUSE":
                winsound.Beep(500, 350)          # one low beep: stopped
            elif kind == "RESUME":
                winsound.Beep(900, 120)          # two rising beeps: running
                time.sleep(0.05)
                winsound.Beep(1300, 160)
            else:
                winsound.Beep(1320, 120)
                time.sleep(0.05)
                winsound.Beep(1320, 120)
        except Exception:
            # no winsound (macOS/Linux): terminal bell, may be silent
            print("\a", end="", flush=True)

    threading.Thread(target=_play, daemon=True).start()


def speak_text(text):

    text = str(text).strip()

    if not text:
        return

    initialize_tts()
    tts_queue.put(text)


def shutdown_tts():

    global tts_shutdown_requested

    if tts_shutdown_requested:
        return

    tts_shutdown_requested = True

    try:
        tts_queue.put(None)
    except Exception:
        pass


# ============================================================
# BASIC GEOMETRY (unchanged)
# ============================================================

def distance_2d(p1, p2):

    return np.linalg.norm(
        np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
    )


def calculate_ear(points):

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = distance_2d(p2, p6)
    vertical_2 = distance_2d(p3, p5)
    horizontal = distance_2d(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def get_eye_width(points):

    return distance_2d(points[0], points[3])


def eye_width_ratio_of(left_points, right_points):

    smaller = min(get_eye_width(left_points), get_eye_width(right_points))
    larger = max(get_eye_width(left_points), get_eye_width(right_points))

    if smaller < 1e-6:
        return 0.0

    return float(larger / smaller)


def nose_offset_ratio_of(face_landmarks):

    left_corner = face_landmarks[33]
    right_corner = face_landmarks[263]
    nose = face_landmarks[1]

    eye_mid_x = (left_corner.x + right_corner.x) / 2.0
    eye_span = abs(right_corner.x - left_corner.x)

    if eye_span < 1e-6:
        return None

    return float(abs(nose.x - eye_mid_x) / eye_span)


def head_orientation_ok(face_landmarks, left_points, right_points, cfg):

    left_width = get_eye_width(left_points)
    right_width = get_eye_width(right_points)

    smaller = min(left_width, right_width)
    larger = max(left_width, right_width)

    if smaller < 1e-6:
        return False

    if larger / smaller > cfg.MAX_EYE_WIDTH_RATIO:
        return False

    nose_offset = nose_offset_ratio_of(face_landmarks)

    if nose_offset is None:
        return False

    return nose_offset <= cfg.MAX_NOSE_OFFSET_RATIO


# ============================================================
# BLINK ML MODEL  (advisory only; duration decides dot/dash)
# ============================================================

def load_ml_model(cfg):
    """Returns a predict(features) callable, or None (never raises)."""

    if not cfg.USE_ML:
        print("Blink ML advisory classifier: off for this preset.")
        return None

    if not os.path.exists(ML_MODEL_PATH):
        print("WARNING: blink_model.pkl not found - continuing without the "
              "advisory ML classifier (duration-based blink classification "
              "is unaffected).")
        return None

    try:
        blink_model = joblib.load(ML_MODEL_PATH)
    except Exception as e:
        print("WARNING: could not load blink_model.pkl:", repr(e))
        return None

    print("Blink ML model loaded successfully.")

    if (
        hasattr(blink_model, "n_features_in_")
        and blink_model.n_features_in_ != len(ML_FEATURE_NAMES)
    ):
        print(
            f"WARNING: Blink ML advisory classifier disabled. Model expects "
            f"{blink_model.n_features_in_} features, BlinkTap provides "
            f"{len(ML_FEATURE_NAMES)}."
        )
        return None

    if hasattr(blink_model, "feature_names_in_") and set(
        blink_model.feature_names_in_
    ) != set(ML_FEATURE_NAMES):
        print("WARNING: Blink ML advisory classifier disabled. Model feature "
              "names do not match BlinkTap's expected features.")
        return None

    def predict(features):

        try:

            if pd is not None and hasattr(blink_model, "feature_names_in_"):

                columns = list(blink_model.feature_names_in_)
                row = dict(zip(ML_FEATURE_NAMES, features))
                sample = pd.DataFrame(
                    [[row[column] for column in columns]], columns=columns
                )
                return blink_model.predict(sample)[0]

            return blink_model.predict([features])[0]

        except Exception as e:

            print("ML prediction warning:", repr(e))
            return None

    return predict


def ml_features_from_snapshot(snapshot, duration):

    if snapshot is None:
        return None

    return [
        snapshot["left_ear"],
        snapshot["right_ear"],
        snapshot["average_ear"],
        snapshot["min_ear"],
        snapshot["ear_difference"],
        snapshot["eye_width_ratio"],
        snapshot["nose_offset_ratio"],
        duration,
    ]


# ============================================================
# MEDIAPIPE
# ============================================================

def create_landmarker():
    """Face landmarker with blendshapes (eye-blink / gaze scores)."""

    if not os.path.exists(MODEL_PATH):
        print("ERROR: face_landmarker.task not found!")
        print("Expected:", MODEL_PATH)
        raise SystemExit(1)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    common = dict(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        min_tracking_confidence=0.25,
    )

    try:
        return (
            FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(output_face_blendshapes=True, **common)
            ),
            True,
        )
    except Exception as e:
        print("WARNING: blendshapes unavailable (", repr(e), ") - using EAR "
              "only for blink detection.")
        return (
            FaceLandmarker.create_from_options(FaceLandmarkerOptions(**common)),
            False,
        )


def blendshape_dict(result):

    try:
        if result.face_blendshapes:
            return {
                c.category_name: float(c.score)
                for c in result.face_blendshapes[0]
            }
    except Exception:
        pass

    return None


# ============================================================
# MODE MENU (EYE / TAP / BOTH)
# ============================================================

MENU_WINDOW_NAME = "BlinkTap - Choose Communication Method"

MODE_BUTTONS = {
    "EYE": (120, 230, 780, 310),
    "TAP": (120, 335, 780, 415),
    "BOTH": (120, 440, 780, 520),
}


def point_in_rect(x, y, rect):

    x1, y1, x2, y2 = rect

    return x1 <= x <= x2 and y1 <= y <= y2


def choose_communication_mode(profile_calibrated=False):

    selected = {"mode": None}

    def menu_callback(event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for mode, rect in MODE_BUTTONS.items():

            if point_in_rect(x, y, rect):
                selected["mode"] = mode

    cv2.namedWindow(MENU_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MENU_WINDOW_NAME, 900, 650)
    cv2.setMouseCallback(MENU_WINDOW_NAME, menu_callback)

    labels = {
        "EYE": "1  -  EYE BLINKS ONLY",
        "TAP": "2  -  TAP / TOUCH ONLY",
        "BOTH": "3  -  EYE + TAP (BOTH)",
    }

    while selected["mode"] is None:

        menu = np.zeros((620, 900, 3), dtype=np.uint8)

        cv2.putText(menu, "BlinkTap", (320, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.20, (255, 255, 255), 3)
        cv2.putText(menu, "Assistive communication for conscious paralysis patients",
                    (115, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)
        cv2.putText(menu, "Choose the input the patient can control reliably",
                    (210, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)
        cv2.putText(menu, "Short blink / tap = dot (.)     Long blink / press = dash (-)",
                    (175, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 215, 255), 1)

        for mode, rect in MODE_BUTTONS.items():

            x1, y1, x2, y2 = rect
            cv2.rectangle(menu, (x1, y1), (x2, y2), (255, 255, 255), 2)

            text = labels[mode]
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0]
            tx = x1 + (x2 - x1 - text_size[0]) // 2
            ty = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(menu, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

        if profile_calibrated:
            note, color = "Patient calibration profile found and will be used", (110, 230, 120)
        else:
            note, color = ("No calibration profile yet - run calibration.py once "
                           "for best blink accuracy", (40, 150, 255))
        cv2.putText(menu, note, (130, 560),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        cv2.putText(menu, "SELECT ONE OPTION TO CONTINUE   (Q = quit)", (245, 595),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        cv2.imshow(MENU_WINDOW_NAME, menu)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("1"):
            selected["mode"] = "EYE"
        elif key == ord("2"):
            selected["mode"] = "TAP"
        elif key == ord("3"):
            selected["mode"] = "BOTH"
        elif key in (ord("q"), ord("Q")):
            cv2.destroyWindow(MENU_WINDOW_NAME)
            return None

    cv2.destroyWindow(MENU_WINDOW_NAME)

    return selected["mode"]


# ============================================================
# EYE BACKEND (webcam + MediaPipe) - only exists in EYE / BOTH mode
# ============================================================

class EyeBackend:
    """
    Owns the webcam, the MediaPipe landmarker and the blink detector.

    Nothing here is created until start() is called, and stop() releases the
    camera and closes MediaPipe, so Tapping mode never touches the webcam.
    The factories can be replaced in tests.
    """

    def __init__(self, cfg, profile, open_camera=None, make_landmarker=None):
        self.cfg = cfg
        self.profile = profile
        self.open_camera = open_camera or self._default_open_camera
        self.make_landmarker = make_landmarker or create_landmarker
        self.camera = None
        self.landmarker = None
        self.blendshapes_ok = False
        self.detector = None
        self.ml_predict = None
        self.ml_loaded = False

    @property
    def active(self):
        return self.camera is not None

    def _default_open_camera(self):
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            camera.release()
            return None
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.CAMERA_HEIGHT)
        return camera

    def _build_detector(self):
        cfg, profile = self.cfg, self.profile
        cal = bool(profile.get("calibrated"))
        return BlinkDetector(BlinkSettings(
            closed_threshold=profile["closed_threshold"] if cal else cfg.CLOSED_THRESHOLD,
            open_threshold=profile["open_threshold"] if cal else cfg.OPEN_THRESHOLD,
            closed_confirm_frames=cfg.CLOSED_CONFIRM_FRAMES,
            open_confirm_frames=cfg.OPEN_CONFIRM_FRAMES,
            min_blink_duration=cfg.MIN_BLINK_DURATION,
            max_blink_duration=cfg.MAX_BLINK_DURATION,
            cooldown=cfg.BLINK_COOLDOWN,
            ear_history_size=cfg.EAR_HISTORY_SIZE,
            goodbye_hold=cfg.GOODBYE_HOLD_DURATION,
            max_face_lost_frames=cfg.MAX_FACE_LOST_FRAMES,
            max_geometry_bad_frames=cfg.MAX_GEOMETRY_BAD_FRAMES,
            use_blink_scores=self.blendshapes_ok,
            blink_score_threshold=profile["blink_score_threshold"],
            initial_baseline=profile["open_ear"],
        ))

    def start(self):
        """Open the webcam and initialise MediaPipe.  True on success."""
        if self.active:
            return True

        camera = self.open_camera()
        if camera is None:
            print("ERROR: Could not open webcam.")
            return False

        try:
            self.landmarker, self.blendshapes_ok = self.make_landmarker()
        except SystemExit:
            camera.release()
            raise
        except Exception as e:
            print("ERROR: could not start MediaPipe:", repr(e))
            camera.release()
            return False

        if not self.ml_loaded:
            self.ml_predict = load_ml_model(self.cfg)
            self.ml_loaded = True

        self.camera = camera
        self.detector = self._build_detector()
        print("Webcam and eye detection started.")
        return True

    def stop(self):
        """Release the webcam and close MediaPipe (safe to call twice)."""
        was_active = self.active or self.landmarker is not None
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
        self.camera = None
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
        self.landmarker = None
        self.detector = None
        if was_active:
            print("Webcam released and eye detection stopped.")

    def process(self, engine, info):
        """
        Read one frame, run MediaPipe + the blink detector, feed blinks into
        the engine and fill in the UI info.  Returns (ok, goodbye).
        ok is False if the camera stopped delivering frames.
        """
        cfg, detector = self.cfg, self.detector

        success, frame = self.camera.read()
        if not success:
            print("ERROR: Could not read webcam frame.")
            return False, False

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        info.camera_frame = frame
        goodbye = False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self.landmarker.detect(mp_image)
        except Exception as e:
            # A transient MediaPipe failure must not end the session.
            detector.reset()
            engine.last_action_status = f"Eye detection error: {type(e).__name__}"
            info.face_status = "EYE DETECTION ERROR - RETRYING"
            result = None

        now = time.time()

        if result is not None and result.face_landmarks:

            face_landmarks = result.face_landmarks[0]

            left_points = [
                (int(face_landmarks[i].x * width), int(face_landmarks[i].y * height))
                for i in LEFT_EYE
            ]
            right_points = [
                (int(face_landmarks[i].x * width), int(face_landmarks[i].y * height))
                for i in RIGHT_EYE
            ]

            if head_orientation_ok(face_landmarks, left_points, right_points, cfg):

                info.face_status = "OK"

                shapes = blendshape_dict(result)
                blink_left = shapes.get("eyeBlinkLeft") if shapes else None
                blink_right = shapes.get("eyeBlinkRight") if shapes else None
                nose_offset = nose_offset_ratio_of(face_landmarks)

                events = detector.process(
                    now,
                    calculate_ear(left_points),
                    calculate_ear(right_points),
                    blink_left,
                    blink_right,
                    extras={
                        "eye_width_ratio": eye_width_ratio_of(left_points, right_points),
                        "nose_offset_ratio": nose_offset if nose_offset is not None else 0.0,
                    },
                )

                info.gaze = estimate_gaze(face_landmarks, shapes)

                for event in events:

                    if event["type"] == "closure":

                        engine.register_eye_signal(
                            event["duration"],
                            ml_features_from_snapshot(event["snapshot"], event["duration"]),
                            closure=event["closure"],
                        )

                    elif event["type"] == "goodbye":

                        engine.last_type = "GOODBYE"
                        engine.last_blink = "BYE"
                        engine.last_action_status = "Goodbye gesture detected"
                        print("Goodbye gesture detected.")
                        goodbye = True

            else:

                detector.frame_without_eyes(True, False)
                info.face_status = "HEAD ANGLE TOO LARGE - PAUSED"

        elif result is not None:

            detector.frame_without_eyes(False)
            info.face_status = "FACE NOT DETECTED - LOOK AT CAMERA"

        info.eyes_closed = detector.eyes_closed
        info.closure_elapsed = detector.closure_elapsed
        info.left_ear = detector.left_ear
        info.right_ear = detector.right_ear
        info.blink_score = detector.blink_score

        return True, goodbye


# ============================================================
# MAIN
# ============================================================

WINDOW_NAME = "BlinkTap - Assistive Communication"


class TapState:
    """Mouse press = tap (same behaviour as before: tap anywhere)."""

    def __init__(self):
        self.pressed = False
        self.start = None
        self.goodbye_requested = False
        self.mode = None


def main(preset=None):

    cfg = build_config(preset)
    profile = load_profile()
    profile_calibrated = bool(profile.get("calibrated"))
    timing, timing_notes = load_settings()

    if preset:
        print(f"Preset: {preset}")

    if profile_calibrated:
        print("Using patient calibration profile "
              f"({profile.get('created', 'unknown date')}).")
        for note in profile.get("notes", []):
            print("  note:", note)
    else:
        print("No calibration profile - using default thresholds. Run "
              "calibration.py to personalise them.")

    for note in timing_notes:
        print("Settings:", note)

    initialize_tts()

    engine = CommunicationEngine(
        speak_fn=speak_text,
        notify_fn=notify_sound,
        echo_fn=speak_text,
    )

    def save_messages(entries):
        valid, warnings = morse_config.save_quick_messages(entries)
        engine.set_quick_messages(valid)
        morse_config.reload_quick_messages()

    editor = MessageEditor(morse_config.QUICK_MESSAGE_LIST, save_messages)
    editor_prev_pause = {"value": False}

    info = UIInfo()
    info.calibrated = profile_calibrated
    info.goodbye_hold = cfg.GOODBYE_HOLD_DURATION
    info.tap_long_threshold = cfg.TAP_LONG_THRESHOLD
    info.tap_goodbye = cfg.TAP_GOODBYE_DURATION
    info.editor = editor
    info.warnings = list(morse_config.QUICK_MESSAGE_WARNINGS)

    tap = TapState()
    backend = EyeBackend(cfg, profile)
    window = {"w": CANVAS_W, "h": CANVAS_H}

    # ---------------- communication method ----------------
    # The menu runs BEFORE any webcam or MediaPipe is touched.

    tap.mode = choose_communication_mode(profile_calibrated)

    if tap.mode is None:

        print("No communication mode selected. BlinkTap exiting.")
        shutdown_tts()
        raise SystemExit(0)

    def apply_mode(new_mode):
        """Start/stop the webcam to match the mode.  True on success."""

        if new_mode in ("EYE", "BOTH"):

            if not backend.start():
                return False

        else:

            backend.stop()          # tapping mode: webcam OFF
            info.camera_frame = None
            info.eyes_closed = False
            info.closure_elapsed = 0.0
            info.face_status = "OK"

        tap.mode = new_mode
        return True

    if not apply_mode(tap.mode):

        shutdown_tts()
        raise SystemExit(1)

    print()
    print("BlinkTap communication settings:")
    print(f"Dot  = {timing['dot_min']:.2f}-{timing['dot_max']:.2f}s eyes closed")
    print(f"Dash = {timing['dash_min']:.2f}-{timing['dash_max']:.2f}s eyes closed")
    print(f"Letter timeout = {timing['letter_timeout']:.1f}s, "
          f"word timeout = {timing['word_timeout']:.1f}s")
    print(f"Eye goodbye = {cfg.GOODBYE_HOLD_DURATION:.1f}s hold")
    print(f"Tap goodbye = {cfg.TAP_GOODBYE_DURATION:.1f}s hold")
    print("Selected communication mode:", tap.mode)
    print()

    # ---------------- mouse: taps (TAP/BOTH) or message clicks (EYE) ----------------

    def mouse_callback(event, x, y, flags, param):

        if tap.mode in ("TAP", "BOTH"):

            if event == cv2.EVENT_LBUTTONDOWN:

                if tap.pressed:
                    return

                tap.pressed = True
                tap.start = time.time()

            elif event == cv2.EVENT_LBUTTONUP:

                if tap.pressed and tap.start is not None:

                    duration = time.time() - tap.start

                    if duration >= cfg.TAP_GOODBYE_DURATION:

                        tap.goodbye_requested = True
                        engine.last_action_status = "Tap goodbye gesture detected"
                        print("Tap goodbye gesture detected.")

                    else:

                        engine.register_tap_signal(duration)

                tap.pressed = False
                tap.start = None

            return

        # EYE mode: the mouse belongs to the caregiver - click a custom
        # message to select it (then confirm with Enter or a long blink).
        if (
            event == cv2.EVENT_LBUTTONDOWN
            and engine.mode == QUICK
            and not editor.active
            and not engine.confirming
        ):
            point = window_to_canvas(x, y, window["w"], window["h"])
            if point is None:
                return
            for x1, y1, x2, y2, index in info.hit_rects:
                if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                    engine.select_quick(index)
                    return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1180, 716)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    def adjust(which, delta):
        engine.adjust_timeout(which, delta)
        try:
            save_settings(engine.s.timing_dict())
        except Exception as e:
            engine.last_action_status += f" (not saved: {e})"

    # ============================================================
    # MAIN LOOP
    # ============================================================

    try:

        while True:

            info.input_method = tap.mode
            goodbye = False

            # ---- eye processing: only when the webcam is active ----
            if backend.active:

                ok, goodbye = backend.process(engine, info)

                if not ok:
                    break

            # ---- tap state for the feedback bar ----
            info.tap_pressed = tap.pressed
            info.tap_elapsed = (
                time.time() - tap.start
                if tap.pressed and tap.start is not None
                else 0.0
            )

            # ---- letter / word timeouts (never mid-blink or mid-press) ----
            engine.tick(input_active=info.eyes_closed or tap.pressed)

            # ---- display ----
            canvas = render(engine, info)

            try:
                _, _, window["w"], window["h"] = cv2.getWindowImageRect(WINDOW_NAME)
            except cv2.error:
                window["w"], window["h"] = CANVAS_W, CANVAS_H

            cv2.imshow(WINDOW_NAME, fit_to_window(canvas, window["w"], window["h"]))

            # ---- keyboard ----
            key = cv2.waitKey(1 if backend.active else 15) & 0xFF

            if goodbye or tap.goodbye_requested:
                break

            if key == 255:
                continue

            # The message editor owns the keyboard while it is open.
            if editor.active:

                editor.handle_key(key)

                if not editor.active:
                    engine.paused = editor_prev_pause["value"]

                continue

            if key in (ord("q"), ord("Q")):
                break

            elif key in (ord("m"), ord("M")):

                tap.pressed = False
                tap.start = None

                new_mode = choose_communication_mode(profile_calibrated)

                if new_mode is None:
                    break

                if apply_mode(new_mode):

                    engine.paused = False
                    engine.clear_all(status=f"Input changed to {new_mode}")
                    print("Communication mode changed to:", new_mode)

                else:

                    # webcam could not be opened: fall back to tapping
                    apply_mode("TAP")
                    engine.clear_all(status="Webcam unavailable - tapping mode")

            elif key == ord("1"):
                engine.set_mode(ALPHABET)

            elif key == ord("2"):
                engine.set_mode(QUICK)

            elif key in (ord("n"), ord("N")):
                engine.quick_move(1)

            elif key in (ord("b"), ord("B")):
                engine.quick_move(-1)

            elif key in (ord("e"), ord("E")):
                editor_prev_pause["value"] = engine.paused
                engine.paused = True
                editor.open(list(engine.quick_messages.items()))

            elif key in (8, 127):
                engine.backspace()

            elif key in (ord("c"), ord("C")):
                engine.clear_draft()
                print("Morse sequence and message cleared.")

            elif key in (ord("s"), ord("S")):
                engine.space_key()

            elif key in (ord("t"), ord("T"), 13):
                engine.speak_key()

            elif key in (ord("r"), ord("R")):
                engine.repeat_last()

            elif key in (ord("p"), ord("P")):
                engine.toggle_pause()

            elif key in (ord("l"), ord("L")):
                engine.echo_enabled = not engine.echo_enabled
                engine.last_action_status = (
                    "Reading letters/words aloud: ON" if engine.echo_enabled
                    else "Reading letters/words aloud: OFF (beeps only)"
                )

            elif key == ord("["):
                adjust("letter_timeout", -0.5)

            elif key == ord("]"):
                adjust("letter_timeout", 0.5)

            elif key == ord(";"):
                adjust("word_timeout", -0.5)

            elif key == ord("'"):
                adjust("word_timeout", 0.5)

            elif key == 27:
                engine.cancel_confirm()

    # ============================================================
    # CLEAN SHUTDOWN
    # ============================================================

    finally:

        print()
        print("Shutting down BlinkTap...")

        backend.stop()
        shutdown_tts()
        cv2.destroyAllWindows()

        print()
        print("================================================")
        print("BlinkTap stopped.")
        print("Remaining Morse:", engine.current_morse)
        print("Final Message:", engine.decoded_message)
        print("================================================")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="BlinkTap assistive communication")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                        help="settings preset (responsive = old "
                             "eye_detection_responsive.py behaviour)")
    args = parser.parse_args()

    main(preset=args.preset)
