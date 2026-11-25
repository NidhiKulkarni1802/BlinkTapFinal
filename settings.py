"""
BlinkTap - user timing settings (settings.json).

Starting values are NOT medical standards; every patient is different.  They
can be changed

  * in the app (letter / word timeout keys),
  * by editing settings.json,
  * automatically by calibration.py, which estimates comfortable dot and
    dash ranges from the patient's own practice blinks.

All durations are the time the eyelids are actually closed, in seconds.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(HERE, "settings.json")

DEFAULT_SETTINGS = {
    "dot_min": 0.15,         # shorter than this = too quick / natural blink
    "dot_max": 0.6,
    "dash_min": 0.8,
    "dash_max": 1.5,         # longer than this = too long, ignored
    "letter_timeout": 2.5,   # pause that finishes a letter
    "word_timeout": 6.0,     # pause that ends a word (space)
}

MIN_GAP = 0.05               # required safety gap between ranges


def validate_settings(values):
    """
    Return (clean_settings, notes).  Guarantees
        0 < dot_min < dot_max < dash_min < dash_max
    with no overlap, and word_timeout > letter_timeout.
    """
    notes = []
    s = dict(DEFAULT_SETTINGS)
    for key in DEFAULT_SETTINGS:
        try:
            if key in values and values[key] is not None:
                s[key] = float(values[key])
        except (TypeError, ValueError):
            notes.append(f"Ignored invalid value for {key}.")

    if s["dot_min"] <= 0.05:
        s["dot_min"] = 0.05
    if s["dot_max"] < s["dot_min"] + MIN_GAP:
        s["dot_max"] = s["dot_min"] + MIN_GAP
        notes.append("dot_max raised so the dot range is not empty.")
    if s["dash_min"] < s["dot_max"] + MIN_GAP:
        s["dash_min"] = s["dot_max"] + MIN_GAP
        notes.append("dash_min raised so dots and dashes cannot overlap.")
    if s["dash_max"] < s["dash_min"] + MIN_GAP:
        s["dash_max"] = s["dash_min"] + MIN_GAP
        notes.append("dash_max raised so the dash range is not empty.")

    s["letter_timeout"] = min(max(s["letter_timeout"], 0.5), 15.0)
    s["word_timeout"] = min(max(s["word_timeout"], 1.0), 30.0)
    if s["word_timeout"] <= s["letter_timeout"]:
        s["word_timeout"] = s["letter_timeout"] + 1.0
        notes.append("word_timeout raised to be longer than letter_timeout.")

    for key in s:
        s[key] = round(s[key], 3)
    return s, notes


def load_settings(path=SETTINGS_FILE):
    """Never raises: any problem falls back to the defaults."""
    values = {}
    notes = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            values = data
    except FileNotFoundError:
        pass
    except Exception as e:
        notes.append(f"settings.json unreadable ({e}); using defaults.")
    s, more = validate_settings(values)
    return s, notes + more


def save_settings(settings, path=SETTINGS_FILE):
    clean, _ = validate_settings(settings)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    return clean


def classify_duration(seconds, s):
    """
    Classify one blink length.  Returns one of
        "dot", "dash", "too_short", "ambiguous", "too_long"
    The ranges never overlap, so every duration has exactly one answer.
    """
    if seconds < s["dot_min"]:
        return "too_short"
    if seconds <= s["dot_max"]:
        return "dot"
    if seconds < s["dash_min"]:
        return "ambiguous"
    if seconds <= s["dash_max"]:
        return "dash"
    return "too_long"
