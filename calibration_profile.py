"""
BlinkTap - patient calibration profile.

calibration.py collects samples (open eyes, closed eyes, eyes open while
looking around the screen, short blinks, long blinks) and uses
compute_profile() to turn them into patient-specific thresholds, saved to
calibration_profile.json.  eye_detection.py loads that file at start-up so
the patient does not have to calibrate every time.

Nothing here needs a camera, so it is unit-testable.
"""

import json
import os
import time

import numpy as np

from settings import validate_settings

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_FILE = os.path.join(HERE, "calibration_profile.json")

# Original BlinkTap values.  Used when there is no profile (or a profile is
# missing a value), so behaviour without calibration matches the old app.
DEFAULT_PROFILE = {
    "open_ear": 0.33,               # typical open-eye EAR
    "closed_ear": 0.06,             # typical closed-eye EAR
    "closed_threshold": 0.21,       # below this = eye closed
    "open_threshold": 0.235,        # above this = eye open
    "blink_score_threshold": 0.40,  # eyeBlink blendshape needed for "closed"
    "gaze_open_ear_min": None,      # lowest open-eye EAR seen while looking around
    "timings": None,                # dot/dash ranges estimated by calibration
    "calibrated": False,
    "created": None,
    "notes": [],
}

# Sanity limits so a bad calibration can never produce an unusable profile.
LIMITS = {
    "closed_threshold": (0.08, 0.26),
    "blink_score_threshold": (0.25, 0.75),
}


def _clip(name, value):
    lo, hi = LIMITS[name]
    return float(min(max(value, lo), hi))


def _pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def compute_profile(
    open_ears,
    closed_ears,
    short_durations=None,
    long_durations=None,
    gaze_open_ears=None,
    closed_blink_scores=None,
    open_blink_scores=None,
):
    """
    Build a profile dict from calibration samples.

    Every argument is a list of floats.  Missing / too-small groups simply
    leave the corresponding default in place and add a note explaining why.

    Returns (profile, notes).
    """
    profile = dict(DEFAULT_PROFILE)
    notes = []

    open_ears = [v for v in (open_ears or []) if np.isfinite(v)]
    closed_ears = [v for v in (closed_ears or []) if np.isfinite(v)]
    short_durations = [v for v in (short_durations or []) if np.isfinite(v)]
    long_durations = [v for v in (long_durations or []) if np.isfinite(v)]
    gaze_open_ears = [v for v in (gaze_open_ears or []) if np.isfinite(v)]

    # ---------------- EAR thresholds ----------------
    if len(open_ears) >= 10:
        profile["open_ear"] = round(float(np.median(open_ears)), 4)
    else:
        notes.append("Not enough open-eye samples; default open EAR kept.")

    if len(closed_ears) >= 10:
        profile["closed_ear"] = round(float(np.median(closed_ears)), 4)
    else:
        notes.append("Not enough closed-eye samples; default closed EAR kept.")

    if len(gaze_open_ears) >= 10:
        profile["gaze_open_ear_min"] = round(_pct(gaze_open_ears, 3), 4)
    elif open_ears:
        notes.append(
            "No look-around (gaze) samples: thresholds were not checked "
            "against eye movement. Re-run calibration and press G."
        )

    if len(open_ears) >= 10 and len(closed_ears) >= 10:
        # Lowest EAR the patient shows with eyes open (including looking
        # around the chart), highest EAR they show with eyes closed.
        open_low = _pct(open_ears + gaze_open_ears, 3)
        closed_high = _pct(closed_ears, 97)
        gap = open_low - closed_high

        if gap >= 0.04:
            thr = closed_high + 0.5 * gap
            hysteresis = max(0.02, 0.12 * gap)
            profile["closed_threshold"] = round(_clip("closed_threshold", thr), 4)
            profile["open_threshold"] = round(
                profile["closed_threshold"] + hysteresis, 4
            )
        else:
            notes.append(
                "WARNING: open-eye EAR (including looking around) overlaps "
                "closed-eye EAR. Default thresholds kept; the app will rely "
                "more on the eye-blink score. Improve lighting / camera "
                "position (camera at eye level) and recalibrate."
            )

    # ---------------- Blink score (blendshape) threshold ----------------
    closed_scores = [v for v in (closed_blink_scores or []) if np.isfinite(v)]
    open_scores = [v for v in (open_blink_scores or []) if np.isfinite(v)]
    if len(closed_scores) >= 10 and len(open_scores) >= 10:
        low_closed = _pct(closed_scores, 10)
        high_open = _pct(open_scores, 97)
        if low_closed > high_open:
            profile["blink_score_threshold"] = round(
                _clip("blink_score_threshold", (low_closed + high_open) / 2), 3
            )
        else:
            notes.append(
                "WARNING: eye-blink score overlaps between open and closed "
                "eyes; default score threshold kept."
            )

    # ---------------- Dot / dash ranges ----------------
    # short_durations / long_durations are TRUE lid-closure times of the
    # patient's practice blinks.  Ranges are placed around what the patient
    # actually did, with a guaranteed gap between dot and dash.
    if len(short_durations) >= 5 and len(long_durations) >= 5:
        s5, s95 = _pct(short_durations, 5), _pct(short_durations, 95)
        l5, l95 = _pct(long_durations, 5), _pct(long_durations, 95)

        dot_min = max(0.08, 0.8 * s5)
        dot_max = s95 * 1.15
        dash_min = 0.85 * l5
        dash_max = l95 * 1.25

        if dash_min < dot_max + 0.10:
            if s95 < l5:
                mid = (s95 + l5) / 2
            else:
                mid = (np.median(short_durations) + np.median(long_durations)) / 2
            dot_max, dash_min = mid - 0.05, mid + 0.05
            notes.append(
                "WARNING: the patient's short and long blinks overlap or are "
                "too close together to separate reliably. Make long blinks "
                "clearly longer than short ones and recalibrate."
            )

        timings, _ = validate_settings({
            "dot_min": dot_min, "dot_max": dot_max,
            "dash_min": dash_min, "dash_max": dash_max,
        })
        profile["timings"] = {
            k: timings[k] for k in ("dot_min", "dot_max", "dash_min", "dash_max")
        }
        if timings["dot_min"] < 0.15:
            notes.append(
                "The patient's deliberate short blinks are about as brief as "
                "natural blinks, so natural blinks cannot be filtered out by "
                "duration. Ask the patient to make dots as FIRM, slightly "
                "slower blinks and recalibrate for better rejection."
            )
    else:
        notes.append(
            "Not enough short/long blink samples; default dot/dash ranges "
            "kept."
        )

    profile["calibrated"] = True
    profile["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    profile["notes"] = notes
    return profile, notes


def merge_profile(previous, new, have_ear, have_durations, have_scores):
    """
    Combine a new (possibly partial) calibration with the saved one so that
    re-running only some steps does not throw away the others.
    """
    merged = dict(new)
    if previous.get("calibrated"):
        groups = {
            "ear": (have_ear, ("open_ear", "closed_ear", "closed_threshold",
                               "open_threshold", "gaze_open_ear_min")),
            "dur": (have_durations, ("timings",)),
            "score": (have_scores, ("blink_score_threshold",)),
        }
        for have, fields in groups.values():
            if not have:
                for field in fields:
                    merged[field] = previous[field]
    return merged


def save_profile(profile, path=PROFILE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    return path


def load_profile(path=PROFILE_FILE):
    """
    Load the saved profile merged over the defaults.  Never raises: on any
    problem the defaults are returned (calibrated=False).
    """
    profile = dict(DEFAULT_PROFILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in DEFAULT_PROFILE:
                if key in data and data[key] is not None:
                    profile[key] = data[key]
            # Re-apply limits in case the file was edited by hand.
            for name in LIMITS:
                profile[name] = _clip(name, profile[name])
            if profile["open_threshold"] <= profile["closed_threshold"]:
                profile["open_threshold"] = profile["closed_threshold"] + 0.025
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Calibration profile unreadable ({e}); using defaults.")
        profile = dict(DEFAULT_PROFILE)
    return profile
