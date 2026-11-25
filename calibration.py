import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from blink_detector import BlinkDetector, BlinkSettings
from settings import load_settings, save_settings
from calibration_profile import (
    PROFILE_FILE,
    compute_profile,
    load_profile,
    merge_profile,
    save_profile,
)


# ============================================================
# BlinkTap - Patient Calibration + ML Data Collection
# ============================================================
#
# One calibration session does two things:
#
#   1. Collects samples for the advisory ML classifier
#      (blink_data.csv, unchanged format -> train_model.py still works)
#
#   2. Builds a PATIENT PROFILE (calibration_profile.json) that
#      eye_detection.py loads automatically:
#        * open-eye EAR and closed-eye EAR
#        * EAR while looking around the screen (so looking at the
#          chart is never mistaken for a blink)
#        * eye-blink score threshold
#        * short / long blink duration threshold
#        * minimum deliberate-blink duration (natural blinks ignored)
#
# Steps (the program advances by itself when a step is complete):
#
#   O  OPEN   - look at the camera, eyes open
#   G  LOOK   - eyes open, look at left / right / top / bottom of screen
#   C  CLOSED - close both eyes
#   B  SHORT  - deliberate short blinks (dots)
#   L  LONG   - deliberate long blinks (dashes)
#
# Calibration only needs to be repeated if the camera position, lighting
# or the patient's blinking changes.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(HERE, "models", "face_landmarker.task")

DATA_FILE = os.path.join(HERE, "blink_data.csv")

CAMERA_INDEX = 0

# Same basic thresholds BlinkTap uses before a profile exists.
CLOSED_THRESHOLD = 0.21
OPEN_THRESHOLD = 0.235

CLOSED_CONFIRM_FRAMES = 2
OPEN_CONFIRM_FRAMES = 5

MIN_BLINK_DURATION = 0.06
MAX_BLINK_DURATION = 4.0

LONG_BLINK_THRESHOLD = 0.40

EAR_HISTORY_SIZE = 3

# Number of automatic samples/events to collect.
OPEN_TARGET = 150
GAZE_TARGET = 200
CLOSED_TARGET = 100
# Few practice blinks are enough - the patient should not be tired out.
SHORT_BLINK_TARGET = 15
LONG_BLINK_TARGET = 15

# Accepted durations while collecting each blink type.  Deliberately wider
# than the default 0.40 s boundary: the profile learns the patient's own.
SHORT_STEP_MIN = 0.08
SHORT_STEP_MAX = 1.00
LONG_STEP_MIN = 0.30

# Frames with an eye-blink score at/above this are not "eyes open".
OPEN_SCORE_MAX = 0.25
CLOSED_SCORE_MIN = 0.50

STEP_ORDER = ["OPEN", "GAZE", "CLOSED", "SHORT_BLINK", "LONG_BLINK"]

STEP_HELP = {
    "OPEN": ("STEP 1/5  OPEN EYES",
             "Look straight at the camera with eyes open. Do not blink."),
    "GAZE": ("STEP 2/5  LOOK AROUND",
             "Eyes OPEN: slowly look at the left, right, top and bottom of "
             "the screen. Do not blink."),
    "CLOSED": ("STEP 3/5  CLOSE EYES",
               "Close both eyes gently and keep them closed until this "
               "step finishes."),
    "SHORT_BLINK": ("STEP 4/5  SHORT BLINKS (dots)",
                    "Make a comfortable, deliberate short blink, then rest "
                    "a moment. No need to hurry. Repeat."),
    "LONG_BLINK": ("STEP 5/5  LONG BLINKS (dashes)",
                   "Close both eyes slowly for a comfortable, clearly longer "
                   "time, then open. Make it clearly longer than a dot."),
}


# ============================================================
# MEDIAPIPE
# ============================================================

if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: face_landmarker.task not found!")
    print()
    print("Expected:")
    print(MODEL_PATH)
    print()

    raise SystemExit(1)


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_common = dict(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.35,
    min_face_presence_confidence=0.35,
    min_tracking_confidence=0.25,
)

try:
    landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(output_face_blendshapes=True, **_common)
    )
    BLENDSHAPES = True
except Exception as e:
    print("WARNING: blendshapes unavailable:", repr(e))
    landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(**_common)
    )
    BLENDSHAPES = False


# ============================================================
# EYE LANDMARKS / EAR
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


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


def calculate_geometry(face_landmarks, left_points, right_points):

    left_width = get_eye_width(left_points)
    right_width = get_eye_width(right_points)

    if min(left_width, right_width) < 1e-6:
        eye_width_ratio = 99.0
    else:
        eye_width_ratio = max(left_width, right_width) / min(left_width, right_width)

    left_corner = face_landmarks[33]
    right_corner = face_landmarks[263]
    nose = face_landmarks[1]

    eye_mid_x = (left_corner.x + right_corner.x) / 2.0
    eye_span = abs(right_corner.x - left_corner.x)

    if eye_span < 1e-6:
        nose_offset_ratio = 99.0
    else:
        nose_offset_ratio = abs(nose.x - eye_mid_x) / eye_span

    return eye_width_ratio, nose_offset_ratio


# ============================================================
# CSV SETUP (unchanged format)
# ============================================================

CSV_FIELDS = [
    "label",
    "left_ear",
    "right_ear",
    "average_ear",
    "min_ear",
    "ear_difference",
    "eye_width_ratio",
    "nose_offset_ratio",
    "blink_duration",
]

file_exists = os.path.exists(DATA_FILE)

csv_file = open(DATA_FILE, "a", newline="", encoding="utf-8")

csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)

if not file_exists:
    csv_writer.writeheader()
    csv_file.flush()


def save_sample(label, left_ear, right_ear, average_ear, min_ear,
                eye_width_ratio, nose_offset_ratio, blink_duration):

    csv_writer.writerow({
        "label": label,
        "left_ear": round(left_ear, 6),
        "right_ear": round(right_ear, 6),
        "average_ear": round(average_ear, 6),
        "min_ear": round(min_ear, 6),
        "ear_difference": round(abs(left_ear - right_ear), 6),
        "eye_width_ratio": round(eye_width_ratio, 6),
        "nose_offset_ratio": round(nose_offset_ratio, 6),
        "blink_duration": round(blink_duration, 6),
    })

    csv_file.flush()


# ============================================================
# SESSION DATA (for the patient profile)
# ============================================================

counts = {
    "OPEN": 0,
    "GAZE": 0,
    "CLOSED": 0,
    "SHORT_BLINK": 0,
    "LONG_BLINK": 0,
}

TARGETS = {
    "OPEN": OPEN_TARGET,
    "GAZE": GAZE_TARGET,
    "CLOSED": CLOSED_TARGET,
    "SHORT_BLINK": SHORT_BLINK_TARGET,
    "LONG_BLINK": LONG_BLINK_TARGET,
}

open_ears = []
gaze_ears = []
closed_ears = []
short_durations = []
long_durations = []
open_scores = []
closed_scores = []

calibration_mode = "OPEN"

# Same detector the application uses, so durations are measured identically.
detector = BlinkDetector(
    BlinkSettings(
        closed_threshold=CLOSED_THRESHOLD,
        open_threshold=OPEN_THRESHOLD,
        closed_confirm_frames=CLOSED_CONFIRM_FRAMES,
        open_confirm_frames=OPEN_CONFIRM_FRAMES,
        min_blink_duration=MIN_BLINK_DURATION,
        max_blink_duration=MAX_BLINK_DURATION,
        ear_history_size=EAR_HISTORY_SIZE,
        use_blink_scores=BLENDSHAPES,
    )
)

# Per-blink accumulation (mean / min EAR over the closure, as before)
blink_left_ears = []
blink_right_ears = []
blink_average_ears = []
blink_min_ear = 1.0

last_message = ""


def step_done(mode):

    return counts[mode] >= TARGETS[mode]


def advance_if_done():
    """Move to the next step automatically when one is complete."""

    global calibration_mode, last_message

    if not step_done(calibration_mode):
        return

    index = STEP_ORDER.index(calibration_mode)

    if index + 1 < len(STEP_ORDER):

        calibration_mode = STEP_ORDER[index + 1]
        last_message = f"Step complete. Next: {STEP_HELP[calibration_mode][0]}"
        print()
        print(last_message)
        print(STEP_HELP[calibration_mode][1])


# ============================================================
# DISPLAY
# ============================================================

def put_text(frame, text, position, scale=0.65, color=(255, 255, 255), thickness=2):

    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def wrap(text, width_chars=62):

    words, lines, line = text.split(), [], ""

    for word in words:

        if len(line) + len(word) + 1 > width_chars and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()

    lines.append(line)

    return lines


def draw_instructions(frame, left_ear, right_ear, average_ear, blink_score):

    height, width, _ = frame.shape

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (width - 10, 270), (20, 25, 35), -1)
    frame[:] = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

    title, help_text = STEP_HELP[calibration_mode]

    put_text(frame, "BlinkTap - Patient Calibration", (25, 38), 0.65, (0, 255, 255), 2)
    put_text(frame, title, (25, 72), 0.8, (0, 255, 0), 2)

    for i, line in enumerate(wrap(help_text)):
        put_text(frame, line, (25, 104 + i * 26), 0.62, (255, 255, 255), 1)

    score = "n/a" if blink_score is None else f"{blink_score:.2f}"

    put_text(frame,
             f"EAR L {left_ear:.3f}  R {right_ear:.3f}  avg {average_ear:.3f}  "
             f"blink score {score}",
             (25, 178), 0.5)

    # progress for the current step
    done, target = counts[calibration_mode], TARGETS[calibration_mode]
    bar_x1, bar_x2, bar_y = 25, width - 25, 200
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 20), (60, 60, 70), -1)
    fill = int((bar_x2 - bar_x1) * min(done / target, 1.0))
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + fill, bar_y + 20), (0, 200, 0), -1)
    put_text(frame, f"{done}/{target}", (bar_x2 - 80, bar_y + 16), 0.5, (255, 255, 255), 1)

    summary = "   ".join(
        f"{('LOOK' if m == 'GAZE' else m.replace('_BLINK', '')[:5])}:{counts[m]}"
        for m in STEP_ORDER
    )
    put_text(frame, summary, (25, 244), 0.45, (200, 200, 200), 1)

    put_text(frame, "Keys: O Open  G Look  C Closed  B Short  L Long   Q = finish & save",
             (25, height - 18), 0.5, (255, 255, 255), 1)

    if last_message:
        put_text(frame, last_message[:80], (25, height - 48), 0.5, (0, 200, 255), 1)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():

    print()
    print("ERROR: Could not open webcam.")
    print()

    csv_file.close()
    landmarker.close()

    raise SystemExit(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WINDOW_NAME = "BlinkTap - Patient Calibration"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 700)


# ============================================================
# START MESSAGE
# ============================================================

print()
print("================================================")
print("       BlinkTap - Patient Calibration")
print("================================================")
print()
print("Follow the on-screen steps. Each step advances automatically.")
print("Keep your face visible to the camera, at eye level if possible.")
print()
print("Sample data file :", DATA_FILE)
print("Patient profile  :", PROFILE_FILE)
print()
print(STEP_HELP[calibration_mode][0])
print(STEP_HELP[calibration_mode][1])
print()
print("Press Q when finished to save the profile.")
print("================================================")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        success, frame = camera.read()

        if not success:

            print("ERROR: Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        left_ear = right_ear = average_ear = 0.0
        blink_score = None
        eye_width_ratio = 99.0
        nose_offset_ratio = 99.0

        now = time.time()

        if result.face_landmarks:

            face_landmarks = result.face_landmarks[0]

            left_points = [
                (int(face_landmarks[i].x * width), int(face_landmarks[i].y * height))
                for i in LEFT_EYE
            ]
            right_points = [
                (int(face_landmarks[i].x * width), int(face_landmarks[i].y * height))
                for i in RIGHT_EYE
            ]

            eye_width_ratio, nose_offset_ratio = calculate_geometry(
                face_landmarks, left_points, right_points
            )

            blink_left = blink_right = None

            if BLENDSHAPES and result.face_blendshapes:
                shapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
                blink_left = shapes.get("eyeBlinkLeft")
                blink_right = shapes.get("eyeBlinkRight")

            events = detector.process(
                now,
                calculate_ear(left_points),
                calculate_ear(right_points),
                blink_left,
                blink_right,
            )

            left_ear, right_ear = detector.left_ear, detector.right_ear
            average_ear = detector.average_ear
            blink_score = detector.blink_score

            have_scores = blink_score is not None

            # Same gating as the original code when no blink score exists.
            if have_scores:
                eyes_open_now = blink_score < OPEN_SCORE_MAX
                eyes_closed_now = blink_score >= CLOSED_SCORE_MIN
            else:
                eyes_open_now = left_ear > OPEN_THRESHOLD and right_ear > OPEN_THRESHOLD
                eyes_closed_now = left_ear < CLOSED_THRESHOLD and right_ear < CLOSED_THRESHOLD

            # =================================================
            # OPEN SAMPLES
            # =================================================

            if (
                calibration_mode == "OPEN"
                and eyes_open_now
                and not detector.eyes_closed
                and counts["OPEN"] < OPEN_TARGET
            ):

                save_sample("OPEN", left_ear, right_ear, average_ear, average_ear,
                            eye_width_ratio, nose_offset_ratio, 0.0)
                counts["OPEN"] += 1
                open_ears.append(average_ear)
                if have_scores:
                    open_scores.append(blink_score)

            # =================================================
            # GAZE (look-around, eyes open) SAMPLES
            # not written to the CSV; used only for the profile
            # =================================================

            if (
                calibration_mode == "GAZE"
                and eyes_open_now
                and not detector.eyes_closed
                and counts["GAZE"] < GAZE_TARGET
            ):

                counts["GAZE"] += 1
                gaze_ears.append(average_ear)
                if have_scores:
                    open_scores.append(blink_score)

            # =================================================
            # CLOSED SAMPLES
            # =================================================

            if (
                calibration_mode == "CLOSED"
                and eyes_closed_now
                and counts["CLOSED"] < CLOSED_TARGET
            ):

                save_sample("CLOSED", left_ear, right_ear, average_ear, average_ear,
                            eye_width_ratio, nose_offset_ratio, 0.0)
                counts["CLOSED"] += 1
                closed_ears.append(average_ear)
                if have_scores:
                    closed_scores.append(blink_score)

            # =================================================
            # BLINK EVENTS (same detector as the application)
            # =================================================

            if detector.eyes_closed:

                blink_left_ears.append(left_ear)
                blink_right_ears.append(right_ear)
                blink_average_ears.append(average_ear)
                blink_min_ear = min(blink_min_ear, average_ear)

            for event in events:

                if event["type"] != "closure":
                    continue

                blink_duration = event["duration"]      # legacy measure (ML)
                blink_closure = event["closure"]         # true lid-closure time

                if blink_left_ears:
                    event_left_ear = float(np.mean(blink_left_ears))
                    event_right_ear = float(np.mean(blink_right_ears))
                    event_average_ear = float(np.mean(blink_average_ears))
                    event_min_ear = min(blink_min_ear, event_average_ear)
                else:
                    event_left_ear, event_right_ear = left_ear, right_ear
                    event_average_ear = event_min_ear = average_ear

                # The step the patient is in decides the label, so the
                # profile learns THEIR short/long durations rather than
                # assuming the default 0.40 s boundary.
                if (
                    SHORT_STEP_MIN <= blink_closure < SHORT_STEP_MAX
                    and calibration_mode == "SHORT_BLINK"
                    and counts["SHORT_BLINK"] < SHORT_BLINK_TARGET
                ):

                    save_sample("SHORT_BLINK", event_left_ear, event_right_ear,
                                event_average_ear, event_min_ear, eye_width_ratio,
                                nose_offset_ratio, blink_duration)
                    counts["SHORT_BLINK"] += 1
                    short_durations.append(blink_closure)

                    print(f"Short blink collected {counts['SHORT_BLINK']}/"
                          f"{SHORT_BLINK_TARGET} duration={blink_duration:.3f}s")

                elif (
                    LONG_STEP_MIN <= blink_closure <= MAX_BLINK_DURATION
                    and calibration_mode == "LONG_BLINK"
                    and counts["LONG_BLINK"] < LONG_BLINK_TARGET
                ):

                    save_sample("LONG_BLINK", event_left_ear, event_right_ear,
                                event_average_ear, event_min_ear, eye_width_ratio,
                                nose_offset_ratio, blink_duration)
                    counts["LONG_BLINK"] += 1
                    long_durations.append(blink_closure)

                    print(f"Long blink collected {counts['LONG_BLINK']}/"
                          f"{LONG_BLINK_TARGET} duration={blink_duration:.3f}s")

                elif calibration_mode in ("SHORT_BLINK", "LONG_BLINK"):

                    last_message = (
                        f"Blink {blink_closure:.2f}s not used - "
                        + ("make it SHORTER" if calibration_mode == "SHORT_BLINK"
                           else "keep eyes closed LONGER")
                    )

                blink_left_ears.clear()
                blink_right_ears.clear()
                blink_average_ears.clear()
                blink_min_ear = 1.0

            advance_if_done()

        else:

            put_text(frame, "FACE NOT DETECTED", (25, height - 80), 0.8, (0, 0, 255), 2)
            detector.frame_without_eyes(False)

        # ====================================================
        # DISPLAY
        # ====================================================

        draw_instructions(frame, left_ear, right_ear, average_ear, blink_score)

        if detector.eyes_closed:

            put_text(frame, f"Current closure: {detector.closure_elapsed:.2f}s",
                     (25, 300), 0.62, (0, 165, 255), 2)

        if all(step_done(m) for m in STEP_ORDER):

            put_text(frame, "ALL STEPS COMPLETE - press Q to save",
                     (25, 335), 0.7, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        for letter, mode in (("o", "OPEN"), ("g", "GAZE"), ("c", "CLOSED"),
                             ("b", "SHORT_BLINK"), ("l", "LONG_BLINK")):

            if key in (ord(letter), ord(letter.upper())):

                calibration_mode = mode
                last_message = ""
                print()
                print(STEP_HELP[mode][0])
                print(STEP_HELP[mode][1])


finally:

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()
    csv_file.close()
    landmarker.close()
    cv2.destroyAllWindows()

    print()
    print("================================================")
    print("Calibration finished.")
    print("================================================")
    print()
    print(f"OPEN samples:         {counts['OPEN']}")
    print(f"LOOK-AROUND samples:  {counts['GAZE']}")
    print(f"CLOSED samples:       {counts['CLOSED']}")
    print(f"SHORT BLINK samples:  {counts['SHORT_BLINK']}")
    print(f"LONG BLINK samples:   {counts['LONG_BLINK']}")
    print()
    print("Data file:")
    print(DATA_FILE)
    print()

    # ------------------------------------------------------------
    # Patient profile
    # ------------------------------------------------------------

    have_ear = len(open_ears) >= 10 and len(closed_ears) >= 10
    have_durations = len(short_durations) >= 5 and len(long_durations) >= 5
    have_scores = len(open_scores) >= 10 and len(closed_scores) >= 10

    if have_ear or have_durations or have_scores:

        profile, notes = compute_profile(
            open_ears,
            closed_ears,
            short_durations,
            long_durations,
            gaze_ears,
            closed_scores,
            open_scores,
        )

        profile = merge_profile(
            load_profile(), profile, have_ear, have_durations, have_scores
        )

        save_profile(profile)

        print("Patient profile saved:")
        print(PROFILE_FILE)
        print()
        print(f"  closed threshold      : {profile['closed_threshold']}")
        print(f"  open threshold        : {profile['open_threshold']}")
        print(f"  blink score threshold : {profile['blink_score_threshold']}")

        if profile.get("timings"):

            timings = load_settings()[0]
            timings.update(profile["timings"])
            save_settings(timings)
            t = profile["timings"]
            print()
            print("Comfortable blink ranges estimated and saved to settings.json:")
            print(f"  dot  : {t['dot_min']} - {t['dot_max']} s eyes closed")
            print(f"  dash : {t['dash_min']} - {t['dash_max']} s eyes closed")
            print("  (letter / word timeouts unchanged - adjust them in the app "
                  "with [ ] and ; ')")

        for note in notes:
            print("  note:", note)

    else:

        print("Not enough samples to build a patient profile "
              "(need open + closed samples and/or short + long blinks).")
        print("Existing profile (if any) was left unchanged.")

    print()
    print("Next steps:")
    print("  - python eye_detection.py          (uses the profile automatically)")
    print("  - python train_model.py            (optional: retrain advisory ML)")
    print("================================================")
