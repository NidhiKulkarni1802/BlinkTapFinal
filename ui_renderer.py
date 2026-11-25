"""
BlinkTap - on-screen interface (OpenCV drawing only, no camera or state of
its own).  render() turns the engine state + live eye/tap information into one
image that eye_detection.py shows in its window.

Layout (1280 x 780, scaled to the window):

  +-------------------------------------------------------------+
  | header: title | input method | MODE                         |
  +-----------------+----------+--------------------------------+
  | MORSE SEQUENCE  | LETTER   | MESSAGE  (or confirm prompt)   |   upper
  | + timeout bar   |          |                                |   section
  +-----------------+----------+--------------------------------+
  | camera preview  |  A-Z Morse chart + digits (Alphabet mode) |
  | live input bar  |  or Custom Messages list / editor         |
  +-----------------+-------------------------------------------+
  | status                         | control commands           |
  | footer: keyboard shortcuts and current timings              |
  +-------------------------------------------------------------+
"""

import math
import time

import cv2
import numpy as np

import morse_config
from communication_engine import ALPHABET, QUICK
from settings import classify_duration

CANVAS_W = 1280
CANVAS_H = 780

# BGR colours - high contrast on a dark background
BG = (30, 18, 11)                # deep navy
BG_TOP = (44, 28, 18)            # lighter navy at the top (gradient)
PANEL = (58, 38, 26)             # slate card
PANEL_EDGE = (112, 82, 60)
SHADOW = (20, 11, 6)
CELL = (76, 52, 38)              # chart cells
CELL2 = (88, 62, 46)
CELL_DIM = (46, 31, 22)
CELL_DIM_EDGE = (70, 50, 38)
TRACK = (28, 17, 10)             # empty progress bars
WHITE = (255, 255, 255)
GREY = (195, 176, 162)
DIM = (140, 120, 104)
DOT_COLOR = (60, 205, 255)       # amber
DASH_COLOR = (255, 200, 90)      # sky blue
GREEN = (130, 230, 140)
RED = (90, 90, 255)
ORANGE = (50, 155, 255)
ALPHA_COLOR = (235, 150, 55)     # blue mode colour
QUICK_COLOR = (40, 150, 250)     # orange mode colour
HIGHLIGHT = (125, 232, 140)

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA


# ----------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------

_BG_CACHE = {}


def _background():
    """Vertical navy gradient (built once)."""
    if "img" not in _BG_CACHE:
        t = np.linspace(0.0, 1.0, CANVAS_H, dtype=np.float32)[:, None, None]
        top = np.array(BG_TOP, dtype=np.float32)
        bot = np.array(BG, dtype=np.float32)
        col = top * (1 - t) + bot * t
        _BG_CACHE["img"] = np.repeat(col, CANVAS_W, axis=1).astype(np.uint8)
    return _BG_CACHE["img"]


def text(img, s, org, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, str(s), org, FONT, scale, color, thick, AA)


def text_width(s, scale, thick=1):
    return cv2.getTextSize(str(s), FONT, scale, thick)[0][0]


def fit_text(s, max_w, scale, thick=1):
    """Shorten s with '...' so it fits max_w pixels."""
    s = str(s)
    if text_width(s, scale, thick) <= max_w:
        return s
    while len(s) > 3 and text_width(s + "...", scale, thick) > max_w:
        s = s[:-1]
    return s + "..."


def rrect(img, p1, p2, color, radius=10, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                       (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, AA)


def panel(img, p1, p2, edge=PANEL_EDGE, fill=PANEL, thickness=2):
    rrect(img, (p1[0] + 3, p1[1] + 4), (p2[0] + 3, p2[1] + 4), SHADOW, 12, -1)
    rrect(img, p1, p2, fill, 12, -1)
    rrect(img, p1, p2, edge, 12, thickness)


def pattern_width(pattern, dot_r, dash_w, gap):
    w = 0
    for i, ch in enumerate(pattern):
        w += (2 * dot_r) if ch == "." else dash_w
        if i < len(pattern) - 1:
            w += gap
    return w


def draw_pattern(img, pattern, x, cy, dot_r=7, dash_w=26, dash_h=None,
                 gap=9, dot_color=DOT_COLOR, dash_color=DASH_COLOR):
    """Dots are filled circles, dashes are long rounded bars."""
    dash_h = dash_h or max(6, int(dot_r * 1.5))
    for ch in pattern:
        if ch == ".":
            cv2.circle(img, (x + dot_r, cy), dot_r, dot_color, -1, AA)
            x += 2 * dot_r + gap
        else:
            rrect(img, (x, cy - dash_h // 2), (x + dash_w, cy + dash_h // 2),
                  dash_color, dash_h // 2, -1)
            x += dash_w + gap
    return x


def wrap_text(s, max_w, scale, thick=2):
    words = s.split(" ")
    lines, line = [], ""
    for word in words:
        trial = word if not line else line + " " + word
        if text_width(trial, scale, thick) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


# ----------------------------------------------------------------------
# state handed to render()
# ----------------------------------------------------------------------

class UIInfo:
    """Live, non-engine information the renderer needs."""

    def __init__(self):
        self.input_method = "EYE"          # EYE / TAP / BOTH
        self.camera_frame = None           # BGR image or None (tap mode)
        self.face_status = "OK"            # OK / NO FACE / HEAD ANGLE / ERROR
        self.eyes_closed = False
        self.closure_elapsed = 0.0
        self.left_ear = 0.0
        self.right_ear = 0.0
        self.blink_score = None
        self.gaze = "CENTER"
        self.goodbye_hold = 3.5
        self.tap_pressed = False
        self.tap_elapsed = 0.0
        self.tap_long_threshold = 0.30
        self.tap_goodbye = 3.0
        self.calibrated = False
        self.warnings = []
        self.editor = None                 # MessageEditor or None
        self.hit_rects = []                # output: (x1, y1, x2, y2, index)


# ----------------------------------------------------------------------
# layout constants
# ----------------------------------------------------------------------

FLASH_SECONDS = 1.2
UP_Y1, UP_Y2 = 58, 238
LOW_Y1, LOW_Y2 = 246, 684
LEFT_X1, LEFT_X2 = 12, 412
RIGHT_X1, RIGHT_X2 = 424, CANVAS_W - 12

ZONE_COLORS = {
    "too_short": ((70, 70, 78), (150, 150, 160)),
    "dot": ((40, 90, 105), DOT_COLOR),
    "ambiguous": ((70, 70, 78), (150, 150, 160)),
    "dash": ((105, 80, 40), DASH_COLOR),
    "too_long": ((60, 60, 120), RED),
}


# ----------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------

def _draw_header(img, engine, info):
    # eye logo
    cv2.ellipse(img, (36, 30), (20, 11), 0, 0, 360, DASH_COLOR, 3, AA)
    cv2.circle(img, (36, 30), 7, DOT_COLOR, -1, AA)
    cv2.circle(img, (36, 30), 3, BG, -1, AA)
    text(img, "BlinkTap", (66, 39), 1.0, WHITE, 2)
    if not engine.paused:
        text(img, "Assistive communication for conscious paralysis patients",
             (236, 37), 0.52, GREY, 1)

    if engine.mode == ALPHABET:
        label, color = "ALPHABET MODE  (International Morse)", ALPHA_COLOR
    else:
        label, color = "CUSTOM MESSAGES MODE", QUICK_COLOR
    w = text_width(label, 0.68, 2) + 36
    x2 = CANVAS_W - 16
    rrect(img, (x2 - w, 10), (x2, 48), color, 18, -1)
    text(img, label, (x2 - w + 18, 36), 0.68, (15, 15, 15), 2)

    if engine.paused:
        rrect(img, (x2 - w - 440, 10), (x2 - w - 210, 48), RED, 18, -1)
        text(img, "PAUSED", (x2 - w - 392, 36), 0.7, (15, 15, 15), 2)

    method = {"EYE": "Input: EYE BLINKS", "TAP": "Input: TAP",
              "BOTH": "Input: EYE + TAP"}.get(info.input_method, info.input_method)
    text(img, method, (x2 - w - 190, 24), 0.5, WHITE, 1)
    if info.input_method != "TAP":
        if info.calibrated:
            text(img, "Calibrated", (x2 - w - 190, 44), 0.5, GREEN, 1)
        else:
            text(img, "Not calibrated", (x2 - w - 190, 44), 0.5, ORANGE, 1)
    else:
        text(img, "Webcam OFF", (x2 - w - 190, 44), 0.5, GREEN, 1)


def _draw_sequence_box(img, engine, x1, y1, x2, y2):
    panel(img, (x1, y1), (x2, y2))
    text(img, "MORSE SEQUENCE", (x1 + 14, y1 + 26), 0.55, GREY, 1)

    seq = engine.current_morse
    if seq:
        width = pattern_width(seq, 16, 54, 16)
        k = min(1.0, (x2 - x1 - 32) / max(width, 1))
        draw_pattern(img, seq, x1 + 16, y1 + 66, max(8, int(16 * k)),
                     max(24, int(54 * k)), max(12, int(24 * k)), max(8, int(16 * k)))
        prev = engine.preview()
        if prev:
            kind, val = prev
            label = {"LETTER": "= ", "QUICK": "= ", "COMMAND": "command: ",
                     "CONFIRM": "= "}[kind]
            text(img, fit_text(label + str(val), x2 - x1 - 30, 0.8, 2),
                 (x1 + 16, y1 + 112), 0.8, HIGHLIGHT, 2)
        else:
            text(img, "no match yet", (x1 + 16, y1 + 112), 0.6, GREY, 1)
    elif engine.confirming:
        text(img, "Waiting for YES / NO", (x1 + 16, y1 + 70), 0.7, WHITE, 1)
    else:
        text(img, "Blink short (.) or long (-)", (x1 + 16, y1 + 70), 0.7, GREY, 1)

    # "letter / word finished" cue (also beeps, see eye_detection.py)
    if engine.flash_kind and time.time() - engine.flash_time < FLASH_SECONDS:
        if engine.flash_kind == "LETTER":
            msg, color = "LETTER DONE - next letter", HIGHLIGHT
        else:
            msg, color = "WORD DONE - next word", DASH_COLOR
        rrect(img, (x1 + 14, y2 - 50), (x2 - 14, y2 - 12), color, 10, -1)
        text(img, msg, (x1 + 26, y2 - 22), 0.8, (15, 15, 15), 2)
        return

    # timeout bar: letter first, then word
    lp = engine.letter_progress()
    wp = engine.word_progress()
    label, frac, color = "", 0.0, GREY
    if lp is not None:
        remaining = (1 - lp) * engine.s.letter_timeout
        label = f"Letter ends in {remaining:.1f}s - blink to add more"
        frac, color = lp, DOT_COLOR
    elif wp is not None:
        remaining = (1 - wp) * engine.s.word_timeout
        label = f"Word ends in {remaining:.1f}s - next letter continues the word"
        frac, color = wp, DASH_COLOR
    else:
        label = f"Pause {engine.s.letter_timeout:.1f}s = letter done, {engine.s.word_timeout:.1f}s = word done"
    by1, by2 = y2 - 34, y2 - 16
    text(img, fit_text(label, x2 - x1 - 28, 0.48), (x1 + 14, by1 - 6), 0.48,
         GREY if frac == 0 else WHITE, 1)
    rrect(img, (x1 + 14, by1), (x2 - 14, by2), TRACK, 6, -1)
    if frac > 0:
        rrect(img, (x1 + 14, by1), (x1 + 14 + max(8, int((x2 - x1 - 28) * frac)), by2),
              color, 6, -1)
    rrect(img, (x1 + 14, by1), (x2 - 14, by2), PANEL_EDGE, 6, 1)


def _draw_letter_box(img, engine, x1, y1, x2, y2):
    panel(img, (x1, y1), (x2, y2))
    text(img, "LETTER", (x1 + 14, y1 + 26), 0.55, GREY, 1)
    letter = engine.last_letter if engine.mode == ALPHABET else ""
    if letter:
        scale = 4.6
        w = text_width(letter, scale, 8)
        text(img, letter, (x1 + (x2 - x1 - w) // 2, y1 + 140), scale, WHITE, 8)
    else:
        w = text_width("-", 2.5, 4)
        text(img, "-", (x1 + (x2 - x1 - w) // 2, y1 + 120), 2.5, DIM, 4)


def _draw_message_box(img, engine, x1, y1, x2, y2):
    if engine.paused:
        panel(img, (x1, y1), (x2, y2), RED, (40, 34, 66), 3)
        text(img, "PAUSED - the system is not listening", (x1 + 16, y1 + 34), 0.8, RED, 2)
        text(img, "To start again, blink this code:", (x1 + 16, y1 + 78), 0.65, WHITE, 1)
        start = morse_config.CONTROL_COMMANDS["START"]
        draw_pattern(img, start, x1 + 20, y1 + 108, 11, 38, 16, 12)
        text(img, "(or press Enter / P on the keyboard)", (x1 + 20, y1 + 142), 0.55, GREY, 1)
        if engine.decoded_message:
            text(img, fit_text("Your message so far: " + engine.decoded_message.strip(),
                               x2 - x1 - 30, 0.5), (x1 + 14, y2 - 10), 0.5, DIM, 1)
        return
    if engine.confirming:
        clearing = engine.confirm_source == "CLEAR"
        panel(img, (x1, y1), (x2, y2), ORANGE if clearing else HIGHLIGHT,
              (30, 40, 62) if clearing else (36, 62, 40), 3)
        text(img, "CLEAR EVERYTHING?" if clearing else "SPEAK THIS MESSAGE?",
             (x1 + 16, y1 + 30), 0.75, ORANGE if clearing else HIGHLIGHT, 2)
        lines = wrap_text(engine.confirm_text, x2 - x1 - 40, 1.1, 2)
        for i, line in enumerate(lines[:2]):
            text(img, line, (x1 + 16, y1 + 76 + i * 42), 1.1, WHITE, 2)
        by = y2 - 28
        draw_pattern(img, morse_config.CONFIRM_YES, x1 + 16, by, 9, 34, 12, 10)
        text(img, "= YES, clear it (or press Enter)" if clearing else "= YES, speak (or press Enter)",
             (x1 + 70, by + 8), 0.55, GREEN, 2)
        draw_pattern(img, morse_config.CONFIRM_NO, x1 + 400, by, 9, 34, 12, 10)
        text(img, "= NO, keep it" if clearing else "= NO, cancel", (x1 + 460, by + 8), 0.55, ORANGE, 2)
        if engine.confirm_started is not None:
            left = max(0, engine.s.confirm_timeout - (time.time() - engine.confirm_started))
            text(img, f"{left:0.0f}s", (x2 - 56, y1 + 30), 0.6, GREY, 1)
        return

    panel(img, (x1, y1), (x2, y2))
    if engine.mode == ALPHABET:
        text(img, "MESSAGE", (x1 + 14, y1 + 26), 0.55, GREY, 1)
        msg = engine.decoded_message
        if msg:
            shown = None
            for scale, thick in ((1.9, 4), (1.4, 3), (1.0, 2)):
                lines = wrap_text(msg + "_", x2 - x1 - 36, scale, thick)
                if len(lines) <= (2 if scale > 1.0 else 3):
                    shown = (lines, scale, thick)
                    break
            lines, scale, thick = shown or (wrap_text(msg + "_", x2 - x1 - 36, 1.0, 2)[-3:], 1.0, 2)
            lh = int(34 * scale) + 12
            for i, line in enumerate(lines):
                text(img, line, (x1 + 16, y1 + 40 + int(34 * scale) + i * lh), scale, WHITE, thick)
        else:
            text(img, "Spell your message using the chart", (x1 + 16, y1 + 100), 0.8, GREY, 1)
    else:
        text(img, "CUSTOM MESSAGES", (x1 + 14, y1 + 26), 0.55, GREY, 1)
        text(img, "Blink the pattern shown next to a message,",
             (x1 + 16, y1 + 70), 0.75, WHITE, 1)
        text(img, "or caregiver: N/B choose, Enter select.",
             (x1 + 16, y1 + 102), 0.65, GREY, 1)
        text(img, "You will be asked to confirm before it is spoken.",
             (x1 + 16, y1 + 130), 0.55, DIM, 1)
    if engine.last_spoken:
        text(img, fit_text("Last spoken: " + engine.last_spoken, x2 - x1 - 30, 0.5),
             (x1 + 14, y2 - 12), 0.5, DIM, 1)


def _draw_camera(img, info, x1, y1, x2, y2):
    panel(img, (x1, y1), (x2, y2))
    aw, ah = x2 - x1 - 8, y2 - y1 - 8
    cam = info.camera_frame
    if cam is not None and info.input_method != "TAP":
        ph = ah
        pw = int(ph * cam.shape[1] / cam.shape[0])
        if pw > aw:
            pw = aw
            ph = int(pw * cam.shape[0] / cam.shape[1])
        px, py = x1 + 4 + (aw - pw) // 2, y1 + 4 + (ah - ph) // 2
        img[py:py + ph, px:px + pw] = cv2.resize(cam, (pw, ph), interpolation=cv2.INTER_AREA)
        strip_y = py + ph - 22
        region = img[strip_y:py + ph, px:px + pw]
        img[strip_y:py + ph, px:px + pw] = (region * 0.25).astype(np.uint8)
        score = "-" if info.blink_score is None else f"{info.blink_score:.2f}"
        text(img, f"EAR L {info.left_ear:.2f} R {info.right_ear:.2f} blink {score}",
             (px + 8, py + ph - 6), 0.42, WHITE, 1)
        if info.face_status != "OK":
            rrect(img, (px + 6, py + 6), (px + pw - 6, py + 38), ORANGE, 8, -1)
            text(img, fit_text(info.face_status, pw - 24, 0.55, 2), (px + 14, py + 29),
                 0.55, (10, 10, 10), 2)
    elif info.input_method == "TAP":
        cx = (x1 + x2) // 2
        for i, (msg, sc, col, th) in enumerate((
                ("TAP MODE", 1.0, WHITE, 2),
                ("Webcam is OFF", 0.65, GREEN, 1),
                ("Tap / click anywhere in this window", 0.5, GREY, 1),
                ("short tap = dot     long press = dash", 0.5, GREY, 1))):
            text(img, msg, (cx - text_width(msg, sc, th) // 2, y1 + 70 + i * 36), sc, col, th)
    else:
        text(img, "Starting webcam...", (x1 + 80, (y1 + y2) // 2), 0.7, GREY, 1)


def _draw_input_feedback(img, engine, info, x1, y1, x2, y2):
    """Live bars showing dot / dash zones while eyes are closed / press held."""
    panel(img, (x1, y1), (x2, y2))
    text(img, "LIVE INPUT", (x1 + 14, y1 + 22), 0.5, GREY, 1)
    s = engine.s

    rows = []
    if info.input_method in ("EYE", "BOTH"):
        rows.append("EYE")
    if info.input_method in ("TAP", "BOTH"):
        rows.append("TAP")

    top = y1 + 30
    row_h = (y2 - top - 4) // max(1, len(rows))
    for i, name in enumerate(rows):
        ry = top + i * row_h
        bx1, bx2 = x1 + 14, x2 - 14
        by1, by2 = ry + 24, ry + 24 + 26
        active = info.eyes_closed if name == "EYE" else info.tap_pressed
        elapsed = info.closure_elapsed if name == "EYE" else info.tap_elapsed

        if name == "EYE":
            span = s.dash_max * 1.25
            zones = [(0, s.dot_min, "too_short"), (s.dot_min, s.dot_max, "dot"),
                     (s.dot_max, s.dash_min, "ambiguous"),
                     (s.dash_min, s.dash_max, "dash"), (s.dash_max, span, "too_long")]
            kind = classify_duration(elapsed, s.timing_dict()) if active else None
            if active and elapsed >= info.goodbye_hold:
                state, color = "GOODBYE hold...", RED
            elif kind == "dot":
                state, color = "SHORT  ( . )  - open your eyes now", DOT_COLOR
            elif kind == "dash":
                state, color = "LONG  ( - )  - open your eyes now", DASH_COLOR
            elif kind == "ambiguous":
                state, color = "between dot and dash - keep going or relax", GREY
            elif kind == "too_long":
                state, color = "too long - will be ignored", RED
            elif kind == "too_short":
                state, color = "closing...", GREY
            else:
                state, color = f"looking {info.gaze.lower()} - not a blink", GREY
        else:
            thr = info.tap_long_threshold
            span = thr * 2.0
            zones = [(0, thr, "dot"), (thr, span, "dash")]
            if active:
                state = "LONG  ( - )" if elapsed >= thr else "SHORT  ( . )"
                color = DASH_COLOR if elapsed >= thr else DOT_COLOR
                if elapsed >= info.tap_goodbye:
                    state, color = "GOODBYE hold...", RED
            else:
                state, color = "waiting...", GREY

        text(img, name, (bx1, ry + 17), 0.55, WHITE, 2)
        text(img, fit_text(state, bx2 - bx1 - 50, 0.5, 1), (bx1 + 46, ry + 17), 0.5, color, 1)

        rrect(img, (bx1, by1), (bx2, by2), TRACK, 6, -1)
        bar_w = bx2 - bx1
        for z1, z2, kind_z in zones:
            zx1 = bx1 + int(bar_w * z1 / span)
            zx2 = bx1 + int(bar_w * min(z2, span) / span)
            cv2.rectangle(img, (zx1, by1 + 2), (max(zx1 + 1, zx2 - 1), by2 - 2),
                          ZONE_COLORS[kind_z][0], -1)
        if active:
            kind_now = classify_duration(elapsed, s.timing_dict()) if name == "EYE" else (
                "dash" if elapsed >= info.tap_long_threshold else "dot")
            fx = bx1 + int(bar_w * min(elapsed, span) / span)
            cv2.rectangle(img, (bx1 + 2, by1 + 6), (max(bx1 + 4, fx), by2 - 6),
                          ZONE_COLORS[kind_now][1], -1)
        rrect(img, (bx1, by1), (bx2, by2), PANEL_EDGE, 6, 1)
        # zone captions
        if name == "EYE":
            for z1, z2, kind_z in zones:
                if kind_z in ("dot", "dash"):
                    cx = bx1 + int(bar_w * (z1 + z2) / 2 / span)
                    draw_pattern(img, "." if kind_z == "dot" else "-", cx - (5 if kind_z == "dot" else 10),
                                 by2 + 12, 4, 20, 7, 4)


def _prefix_state(pattern, prefix):
    """0 = no prefix typed, 1 = still possible, 2 = exact match, -1 = excluded"""
    if not prefix:
        return 0
    if pattern == prefix:
        return 2
    if pattern.startswith(prefix):
        return 1
    return -1


CMD_H = 54


def _draw_command_strip(img, engine, x1, y1, x2, y2):
    """Row of control commands (erase, clear all, custom messages, back...)."""
    cmds = list(morse_config.CONTROL_COMMANDS.items())
    cell = (x2 - x1) / len(cmds)
    cur = engine.current_morse
    for i, (name, pattern) in enumerate(cmds):
        cx1 = int(x1 + i * cell)
        cx2 = int(cx1 + cell - 5)
        st = _prefix_state(pattern, cur) if not engine.confirming else 0
        fill, edge = CELL, PANEL_EDGE
        dc, dsc = DOT_COLOR, DASH_COLOR
        if st == -1:
            fill, dc, dsc = CELL_DIM, (60, 100, 115), (115, 95, 55)
        elif st == 1:
            edge = HIGHLIGHT
        elif st == 2:
            fill, edge = (45, 90, 50), HIGHLIGHT
        rrect(img, (cx1, y1), (cx2, y2), fill, 7, -1)
        rrect(img, (cx1, y1), (cx2, y2), edge, 7, 1 if st < 1 else 2)
        label = morse_config.CONTROL_LABELS[name]
        text(img, fit_text(label, cx2 - cx1 - 10, 0.46, 1), (cx1 + 8, y1 + 17), 0.46,
             WHITE if st != -1 else DIM, 1)
        pw = pattern_width(pattern, 4, 13, 5)
        draw_pattern(img, pattern, cx1 + max(8, int((cx2 - cx1 - pw) / 2)), y1 + 37,
                     dot_r=4, dash_w=13, dash_h=6, gap=5, dot_color=dc, dash_color=dsc)


def _draw_chart(img, engine, x1, y1, x2, y2):
    panel(img, (x1, y1), (x2, y2), ALPHA_COLOR)
    text(img, "MORSE CODE CHART", (x1 + 14, y1 + 26), 0.7, WHITE, 2)
    legend_x = x2 - 330
    draw_pattern(img, ".", legend_x, y1 + 20, 7, 26, gap=6)
    text(img, "short blink", (legend_x + 22, y1 + 26), 0.55, DOT_COLOR, 1)
    draw_pattern(img, "-", legend_x + 140, y1 + 20, 7, 26, gap=6)
    text(img, "long blink", (legend_x + 176, y1 + 26), 0.55, DASH_COLOR, 1)

    letters = sorted(morse_config.ALPHABET_MORSE.items())
    cols = 7
    rows = math.ceil(len(letters) / cols)
    gx1, gy1 = x1 + 10, y1 + 40
    gap = 6
    digits_h = 62
    cw = (x2 - x1 - 20 - gap * (cols - 1)) / cols
    ch = (y2 - y1 - 50 - digits_h - CMD_H - 22 - gap * (rows - 1)) / rows

    prefix = engine.current_morse if (engine.mode == ALPHABET and not engine.confirming) else ""

    for idx, (letter, code) in enumerate(letters):
        r, c = divmod(idx, cols)
        cx1 = int(gx1 + c * (cw + gap))
        cy1 = int(gy1 + r * (ch + gap))
        cx2, cy2 = int(cx1 + cw), int(cy1 + ch)

        st = _prefix_state(code, prefix)
        fill, edge, letter_color = CELL2, PANEL_EDGE, WHITE
        dot_c, dash_c, thick_edge = DOT_COLOR, DASH_COLOR, 1
        if st == -1:
            fill, edge, letter_color = CELL_DIM, CELL_DIM_EDGE, DIM
            dot_c, dash_c = (60, 100, 115), (115, 95, 55)
        elif st == 1:
            edge, thick_edge = HIGHLIGHT, 2
        elif st == 2:
            fill, edge, thick_edge = (45, 90, 50), HIGHLIGHT, 3

        rrect(img, (cx1, cy1), (cx2, cy2), fill, 8, -1)
        rrect(img, (cx1, cy1), (cx2, cy2), edge, 8, thick_edge)
        text(img, letter, (cx1 + 10, cy1 + int(ch * 0.55)), 1.25, letter_color, 3)
        text(img, code, (cx2 - text_width(code, 0.55) - 8, cy1 + 22), 0.55,
             DIM if st == -1 else (225, 212, 200), 1)
        pw = pattern_width(code, 6, 22, 8)
        draw_pattern(img, code, cx1 + int((cw - pw) / 2), cy1 + int(ch * 0.80),
                     dot_r=6, dash_w=22, dash_h=9, gap=8,
                     dot_color=dot_c, dash_color=dash_c)

    # numbers strip (standard International Morse 0-9)
    dy1 = y2 - digits_h - CMD_H - 10
    text(img, "NUMBERS", (x1 + 10, dy1 - 4), 0.45, GREY, 1)
    cell = (x2 - x1 - 20) / 10
    for i, (digit, code) in enumerate(sorted(morse_config.DIGIT_MORSE.items())):
        st = _prefix_state(code, prefix)
        dx = int(x1 + 10 + i * cell)
        dx2 = int(dx + cell - 5)
        fill, edge = CELL2, PANEL_EDGE
        color = WHITE
        dc, dsc = DOT_COLOR, DASH_COLOR
        if st == -1:
            fill, edge, color = CELL_DIM, CELL_DIM_EDGE, DIM
            dc, dsc = (60, 100, 115), (115, 95, 55)
        elif st == 1:
            edge = HIGHLIGHT
        elif st == 2:
            fill, edge = (45, 90, 50), HIGHLIGHT
        rrect(img, (dx, dy1), (dx2, dy1 + digits_h), fill, 7, -1)
        rrect(img, (dx, dy1), (dx2, dy1 + digits_h), edge, 7, 1 if st < 1 else 2)
        text(img, digit, (dx + 8, dy1 + 30), 1.0, color, 2)
        pw = pattern_width(code, 3, 10, 4)
        draw_pattern(img, code, dx + max(4, int((dx2 - dx - pw) / 2)), dy1 + 49,
                     dot_r=3, dash_w=10, dash_h=5, gap=4, dot_color=dc, dash_color=dsc)

    _draw_command_strip(img, engine, x1 + 10, y2 - CMD_H - 4, x2 - 6, y2 - 6)


def _draw_quick(img, engine, info, x1, y1, x2, y2):
    panel(img, (x1, y1), (x2, y2), QUICK_COLOR)
    text(img, "CUSTOM MESSAGES", (x1 + 14, y1 + 26), 0.7, WHITE, 2)
    text(img, "blink the pattern  |  caregiver: N/B choose, Enter select, E edit list",
         (x1 + 250, y1 + 26), 0.48, GREY, 1)

    items = list(engine.quick_messages.items())
    n = len(items)
    cols = 1 if n <= 7 else (2 if n <= 16 else 3)
    rows = max(1, math.ceil(n / cols))
    gx1, gy1 = x1 + 10, y1 + 40
    gap = 6
    cw = (x2 - x1 - 20 - gap * (cols - 1)) / cols
    ch = min(72.0, (y2 - y1 - 50 - CMD_H - 8 - gap * (rows - 1)) / rows)
    scale = (0.8 if cols == 1 else 0.68) if cols < 3 else 0.5
    small = cols >= 3
    g_dot, g_dash, g_h, g_gap = (5, 18, 7, 7) if small else (7, 26, 9, 11)
    prefix = engine.current_morse if (engine.mode == QUICK and not engine.confirming) else ""
    glyph_w = max((pattern_width(p, g_dot, g_dash, g_gap) for p, _ in items), default=26)

    for idx, (pattern, message) in enumerate(items):
        c, r = divmod(idx, rows)
        cx1 = int(gx1 + c * (cw + gap))
        cy1 = int(gy1 + r * (ch + gap))
        cx2, cy2 = int(cx1 + cw), int(cy1 + ch)
        info.hit_rects.append((cx1, cy1, cx2, cy2, idx))

        st = _prefix_state(pattern, prefix)
        fill, edge, tcolor = CELL2, PANEL_EDGE, WHITE
        dot_c, dash_c, te = DOT_COLOR, DASH_COLOR, 1
        if st == -1:
            fill, edge, tcolor = CELL_DIM, CELL_DIM_EDGE, DIM
            dot_c, dash_c = (60, 100, 115), (115, 95, 55)
        elif st == 1:
            edge, te = HIGHLIGHT, 2
        elif st == 2:
            fill, edge, te = (45, 90, 50), HIGHLIGHT, 3
        if idx == engine.quick_cursor and not prefix:
            edge, te = WHITE, 3          # caregiver selection cursor

        rrect(img, (cx1, cy1), (cx2, cy2), fill, 8, -1)
        rrect(img, (cx1, cy1), (cx2, cy2), edge, 8, te)
        mid = (cy1 + cy2) // 2
        draw_pattern(img, pattern, cx1 + 10, mid, g_dot, g_dash, g_h, g_gap, dot_c, dash_c)
        tx = cx1 + 10 + glyph_w + 12
        avail = cx2 - tx - 6
        if small and text_width(message, scale, 1) > avail:
            lines = wrap_text(message, avail, 0.42, 1)[:2]
            for li, line in enumerate(lines):
                text(img, line, (tx, mid - 2 + li * 15 - (7 if len(lines) > 1 else -5)), 0.42, tcolor, 1)
        else:
            text(img, fit_text(message, avail, scale, 1 if small else 2),
                 (tx, mid + (6 if small else 8)), scale, tcolor, 1 if small else 2)

    _draw_command_strip(img, engine, x1 + 10, y2 - CMD_H - 4, x2 - 6, y2 - 6)


def _draw_editor(img, editor, x1, y1, x2, y2):
    from message_editor import DELETE, LIST, PATTERN, TEXT
    panel(img, (x1, y1), (x2, y2), WHITE, (40, 30, 22), 3)
    text(img, "EDIT CUSTOM MESSAGES  (caregiver)", (x1 + 14, y1 + 28), 0.75, WHITE, 2)

    if editor.state in (LIST, DELETE):
        visible = 7
        n = len(editor.entries)
        start = max(0, min(editor.cursor - visible // 2, n - visible))
        for row, idx in enumerate(range(start, min(n, start + visible))):
            pattern, message = editor.entries[idx]
            ry = y1 + 46 + row * 46
            sel = idx == editor.cursor
            rrect(img, (x1 + 10, ry), (x2 - 10, ry + 40), (110, 78, 52) if sel else CELL, 8, -1)
            if sel:
                rrect(img, (x1 + 10, ry), (x2 - 10, ry + 40), WHITE, 8, 2)
            draw_pattern(img, pattern, x1 + 24, ry + 20, 6, 22, 9, 8)
            text(img, fit_text(message, x2 - x1 - 190, 0.7, 2), (x1 + 170, ry + 28), 0.7, WHITE, 2)
        if n == 0:
            text(img, "No messages yet - press A to add one", (x1 + 24, y1 + 90), 0.7, GREY, 1)
        if editor.state == DELETE and n:
            msg = editor.entries[editor.cursor][1]
            rrect(img, (x1 + 40, y1 + 150), (x2 - 40, y1 + 250), (30, 30, 90), 12, -1)
            rrect(img, (x1 + 40, y1 + 150), (x2 - 40, y1 + 250), RED, 12, 3)
            text(img, fit_text(f"Delete: {msg}", x2 - x1 - 110, 0.8, 2), (x1 + 60, y1 + 195), 0.8, WHITE, 2)
            text(img, "Y = yes, delete      any other key = keep", (x1 + 60, y1 + 230), 0.65, WHITE, 1)
        help_txt = "N next   B back   A add   E edit   D delete   Esc / Q close"
    elif editor.state == TEXT:
        text(img, "Type the message the patient should be able to say:", (x1 + 20, y1 + 90), 0.7, GREY, 1)
        rrect(img, (x1 + 20, y1 + 110), (x2 - 20, y1 + 170), TRACK, 8, -1)
        rrect(img, (x1 + 20, y1 + 110), (x2 - 20, y1 + 170), WHITE, 8, 2)
        text(img, fit_text(editor.text_buf + "_", x2 - x1 - 70, 0.9, 2), (x1 + 34, y1 + 150), 0.9, WHITE, 2)
        help_txt = "Enter = next step   Backspace = delete letter   Esc = cancel"
    else:  # PATTERN
        text(img, fit_text("Message: " + editor.text_buf, x2 - x1 - 40, 0.7, 1), (x1 + 20, y1 + 90), 0.7, GREY, 1)
        text(img, "Type the blink pattern:   .  = short     -  = long", (x1 + 20, y1 + 135), 0.7, WHITE, 1)
        rrect(img, (x1 + 20, y1 + 155), (x2 - 20, y1 + 225), TRACK, 8, -1)
        rrect(img, (x1 + 20, y1 + 155), (x2 - 20, y1 + 225), WHITE, 8, 2)
        draw_pattern(img, editor.pattern_buf, x1 + 36, y1 + 190, 12, 46, 20, 14)
        text(img, editor.pattern_buf, (x2 - 180, y1 + 200), 0.9, GREY, 2)
        help_txt = "Enter = save   Backspace = undo   Esc = cancel"

    if editor.error:
        text(img, fit_text(editor.error, x2 - x1 - 40, 0.65, 2), (x1 + 20, y2 - 46), 0.65, ORANGE, 2)
    elif editor.info:
        text(img, fit_text(editor.info, x2 - x1 - 40, 0.65, 2), (x1 + 20, y2 - 46), 0.65, GREEN, 2)
    text(img, help_txt, (x1 + 20, y2 - 16), 0.55, GREY, 1)


def _draw_status_strip(img, engine, x1, y1, x2, y2):
    panel(img, (x1, y1), (x2, y2), fill=CELL_DIM, thickness=1)
    text(img, fit_text(engine.last_action_status, x2 - x1 - 28, 0.58, 1), (x1 + 14, y1 + 25),
         0.58, GREEN, 1)


def _draw_footer(img, engine, info):
    s = engine.s
    text(img, "Keys: 1 Alphabet | 2 Custom messages | N/B choose message | Enter or T = Confirm & Speak | "
              "R Repeat | Backspace Delete | C Clear All | S Space | Q Quit",
         (16, CANVAS_H - 30), 0.45, GREY, 1)
    text(img, f"E Edit messages | P Pause | L Read aloud on/off | M Input method | [ ] letter timeout | ; ' word timeout  ||  "
              f"Dot {s.dot_min:.1f}-{s.dot_max:.1f}s  Dash {s.dash_min:.1f}-{s.dash_max:.1f}s  "
              f"Letter {s.letter_timeout:.1f}s  Word {s.word_timeout:.1f}s",
         (16, CANVAS_H - 10), 0.45, GREY, 1)
    if info.warnings:
        text(img, fit_text(info.warnings[0], 400, 0.42), (CANVAS_W - 416, CANVAS_H - 10),
             0.42, ORANGE, 1)
    elif engine.ignored_natural:
        text(img, f"too-quick / natural blinks ignored: {engine.ignored_natural}",
             (CANVAS_W - 416, CANVAS_H - 10), 0.42, DIM, 1)


# ----------------------------------------------------------------------
# public
# ----------------------------------------------------------------------

def render(engine, info):
    img = np.empty((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    img[:] = _background()
    info.hit_rects = []

    _draw_header(img, engine, info)

    # upper section: sequence, current letter, whole message
    _draw_sequence_box(img, engine, 12, UP_Y1, 400, UP_Y2)
    _draw_letter_box(img, engine, 412, UP_Y1, 560, UP_Y2)
    _draw_message_box(img, engine, 572, UP_Y1, CANVAS_W - 12, UP_Y2)

    # lower section
    _draw_camera(img, info, LEFT_X1, LOW_Y1, LEFT_X2, LOW_Y1 + 224)
    _draw_input_feedback(img, engine, info, LEFT_X1, LOW_Y1 + 232, LEFT_X2, LOW_Y2)

    if info.editor is not None and info.editor.active:
        _draw_editor(img, info.editor, RIGHT_X1, LOW_Y1, RIGHT_X2, LOW_Y2)
    elif engine.mode == ALPHABET:
        _draw_chart(img, engine, RIGHT_X1, LOW_Y1, RIGHT_X2, LOW_Y2)
    else:
        _draw_quick(img, engine, info, RIGHT_X1, LOW_Y1, RIGHT_X2, LOW_Y2)

    _draw_status_strip(img, engine, 12, 690, CANVAS_W - 12, 726)
    _draw_footer(img, engine, info)
    return img


def _fit_geometry(win_w, win_h):
    scale = min(win_w / CANVAS_W, win_h / CANVAS_H)
    new_w, new_h = max(1, int(CANVAS_W * scale)), max(1, int(CANVAS_H * scale))
    return scale, (win_w - new_w) // 2, (win_h - new_h) // 2, new_w, new_h


def window_to_canvas(x, y, win_w, win_h):
    """Map a mouse position in the window to canvas coordinates (or None)."""
    if win_w < 50 or win_h < 50:
        return x, y
    scale, ox, oy, new_w, new_h = _fit_geometry(win_w, win_h)
    if not (ox <= x < ox + new_w and oy <= y < oy + new_h):
        return None
    return int((x - ox) / scale), int((y - oy) / scale)


def fit_to_window(img, win_w, win_h):
    """Scale to the window keeping the aspect ratio (letterbox)."""
    if win_w < 100 or win_h < 100:
        return img
    scale, ox, oy, new_w, new_h = _fit_geometry(win_w, win_h)
    resized = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.empty((win_h, win_w, 3), dtype=np.uint8)
    canvas[:] = BG
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas
