"""
BlinkTap - blink detector (camera independent).

This is the original BlinkTap closure state machine (EAR thresholds with
hysteresis, confirm-frame counts, cooldown, duration measured from the first
closed candidate frame, face-loss tolerance, goodbye hold) moved out of the
main loop so it can be shared with calibration.py and unit-tested without a
camera.  Duration is measured exactly as before, so existing thresholds and
calibration data stay valid.

What is new (to stop normal eye movement being read as a blink):

* Both eyes must be closed (unchanged) AND, when MediaPipe's eye-blink
  blendshape scores are available, both scores must agree.  Looking down or
  sideways narrows the eye (lower EAR) but does not raise the blink score the
  way real eyelid closure does.
* The closed threshold for each eye is capped relative to a slowly adapting
  "open" baseline, so a persistently lower EAR when looking at the bottom of
  the chart does not look like closure.
* A blink-score "clearly open" reading can end a closure, so the detector
  cannot get stuck in the ambiguous EAR band after a gaze change.

If blendshapes are unavailable the detector falls back to EAR only.
"""

from collections import deque

import numpy as np


class BlinkSettings:
    """Tunable values.  Defaults equal the original BlinkTap constants."""

    def __init__(self, **overrides):
        self.closed_threshold = 0.21
        self.open_threshold = 0.235
        self.closed_confirm_frames = 2
        self.open_confirm_frames = 5
        self.min_blink_duration = 0.06
        self.max_blink_duration = 4.0
        self.cooldown = 0.12
        self.ear_history_size = 3
        self.goodbye_hold = 3.50
        self.max_face_lost_frames = 12
        self.max_geometry_bad_frames = 12

        # --- new, gaze related ---
        self.use_blink_scores = True
        self.blink_score_threshold = 0.40   # both eyes must reach this
        self.blink_score_open = 0.25        # both below this = clearly open
        self.relative_close_ratio = 0.65    # closed if EAR < baseline * ratio
        self.baseline_alpha = 0.05          # baseline adaptation per frame
        self.initial_baseline = 0.30

        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown BlinkSettings field: {key}")
            setattr(self, key, value)

    @classmethod
    def from_profile(cls, profile, **overrides):
        values = dict(
            closed_threshold=profile["closed_threshold"],
            open_threshold=profile["open_threshold"],
            blink_score_threshold=profile["blink_score_threshold"],
            initial_baseline=profile["open_ear"],
        )
        values.update(overrides)
        return cls(**values)


class BlinkDetector:

    def __init__(self, settings=None):
        self.s = settings or BlinkSettings()
        self.left_hist = deque(maxlen=self.s.ear_history_size)
        self.right_hist = deque(maxlen=self.s.ear_history_size)
        self.left_score_hist = deque(maxlen=self.s.ear_history_size)
        self.right_score_hist = deque(maxlen=self.s.ear_history_size)
        self.last_blink_time = 0.0
        self.reset()
        self.left_baseline = self.s.initial_baseline
        self.right_baseline = self.s.initial_baseline

    # ------------------------------------------------------------
    # state
    # ------------------------------------------------------------

    def reset(self):
        """Cancel any gesture in progress (same as old reset_blink_state)."""
        self.eyes_closed = False
        self.blink_start_time = None
        self.candidate_start_time = None
        self.snapshot = None
        self.closed_frames = 0
        self.open_frames = 0
        self.first_open_time = None
        self.face_lost_frames = 0
        self.geometry_bad_frames = 0
        self.left_hist.clear()
        self.right_hist.clear()
        self.left_score_hist.clear()
        self.right_score_hist.clear()
        # live values for the UI
        self.left_ear = 0.0
        self.right_ear = 0.0
        self.blink_score = None
        self.closure_elapsed = 0.0

    def reset_baseline(self):
        self.left_baseline = self.s.initial_baseline
        self.right_baseline = self.s.initial_baseline

    @property
    def average_ear(self):
        return (self.left_ear + self.right_ear) / 2.0

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------

    @staticmethod
    def _smooth(history, value):
        if not np.isfinite(value):
            return history[-1] if history else 0.0
        history.append(float(value))
        return float(sum(history) / len(history))

    def thresholds(self):
        """Effective (closed, open) thresholds per eye, gaze compensated."""
        s = self.s
        hyst = max(0.0, s.open_threshold - s.closed_threshold)
        out = []
        for base in (self.left_baseline, self.right_baseline):
            closed = min(s.closed_threshold, base * s.relative_close_ratio)
            out.append((closed, closed + hyst))
        return out

    # ------------------------------------------------------------
    # per-frame input
    # ------------------------------------------------------------

    def frame_without_eyes(self, face_found, geometry_ok=True):
        """
        Call when there are no usable eye measurements this frame (face
        missing, or head turned too far).  Mirrors the old tolerance logic.
        """
        s = self.s
        if not face_found:
            self.face_lost_frames += 1
            if self.face_lost_frames > s.max_face_lost_frames:
                self.reset()
        elif not geometry_ok:
            self.geometry_bad_frames += 1
            if (
                not self.eyes_closed
                and self.geometry_bad_frames > s.max_geometry_bad_frames
            ):
                self.reset()
        return []

    def process(
        self,
        now,
        left_ear,
        right_ear,
        blink_left=None,
        blink_right=None,
        extras=None,
    ):
        """
        Feed one frame with valid eye geometry.

        left_ear/right_ear   raw EAR values
        blink_left/right     MediaPipe eyeBlink blendshape scores (0..1) or None
        extras               dict copied into the closure snapshot (used for
                             the advisory ML model)

        Returns a list of events:
          {"type": "closure", "duration": float, "closure": float,
           "snapshot": dict|None}
          {"type": "goodbye"}
        """
        s = self.s
        events = []

        self.face_lost_frames = 0
        self.geometry_bad_frames = 0

        left = self._smooth(self.left_hist, left_ear)
        right = self._smooth(self.right_hist, right_ear)
        self.left_ear, self.right_ear = left, right

        use_scores = (
            s.use_blink_scores
            and blink_left is not None
            and blink_right is not None
        )
        if use_scores:
            sl = self._smooth(self.left_score_hist, blink_left)
            sr = self._smooth(self.right_score_hist, blink_right)
            self.blink_score = min(sl, sr)
        else:
            self.blink_score = None

        (cl, ol), (cr, orr) = self.thresholds()

        ear_closed = left < cl and right < cr
        ear_open = left > ol and right > orr

        if use_scores:
            score_closed = self.blink_score >= s.blink_score_threshold
            score_open = max(sl, sr) < s.blink_score_open
        else:
            score_closed = True
            score_open = False

        # Closed needs EAR *and* blink score to agree.  This is what keeps
        # a downward / sideways glance (narrower eye, no real lid closure)
        # from being treated as a blink.
        both_closed = ear_closed and score_closed
        # Open: EAR clearly open, or the blink score says both eyes are
        # clearly open (e.g. looking down: EAR is low but lids are not shut).
        both_open = ear_open or score_open

        # ---- adapt the "open" baseline only while genuinely open ----
        if (
            not self.eyes_closed
            and self.closed_frames == 0
            and (ear_open or score_open)
            and (not use_scores or self.blink_score < s.blink_score_threshold)
        ):
            a = s.baseline_alpha
            self.left_baseline += a * (left - self.left_baseline)
            self.right_baseline += a * (right - self.right_baseline)

        # ---- candidate tracking (identical to original) ----
        if both_closed:
            self.open_frames = 0
            self.first_open_time = None
            if self.candidate_start_time is None:
                self.candidate_start_time = now
            self.closed_frames += 1
        elif both_open:
            self.closed_frames = 0
            if self.open_frames == 0:
                self.first_open_time = now
            self.open_frames += 1
        else:
            if not self.eyes_closed:
                self.candidate_start_time = None
            self.closed_frames = 0
            self.open_frames = 0

        # ---- OPEN -> CLOSED ----
        if (
            not self.eyes_closed
            and self.closed_frames >= s.closed_confirm_frames
            and now - self.last_blink_time >= s.cooldown
        ):
            self.eyes_closed = True
            self.blink_start_time = (
                self.candidate_start_time
                if self.candidate_start_time is not None
                else now
            )
            self.closed_frames = 0
            self.open_frames = 0
            snap = {
                "left_ear": left,
                "right_ear": right,
                "average_ear": (left + right) / 2.0,
                "min_ear": min(left, right),
                "ear_difference": abs(left - right),
                "eye_width_ratio": 0.0,
                "nose_offset_ratio": 0.0,
            }
            if extras:
                snap.update(extras)
            self.snapshot = snap

        # ---- CLOSED -> OPEN ----
        elif (
            self.eyes_closed
            and self.open_frames >= s.open_confirm_frames
            and self.blink_start_time is not None
        ):
            self.eyes_closed = False
            duration = now - self.blink_start_time
            # "duration" (kept exactly as the original app measured it) ends
            # when the eyes have been confirmed open; "closure" ends at the
            # first open frame, i.e. how long the lids were really shut.
            closure = (
                (self.first_open_time - self.blink_start_time)
                if self.first_open_time is not None
                else duration
            )
            snapshot = self.snapshot
            self.blink_start_time = None
            self.candidate_start_time = None
            self.open_frames = 0
            self.closed_frames = 0
            self.snapshot = None
            if s.min_blink_duration <= duration <= s.max_blink_duration:
                self.last_blink_time = now
            events.append(
                {
                    "type": "closure",
                    "duration": duration,
                    "closure": closure,
                    "snapshot": snapshot,
                }
            )

        # ---- goodbye hold ----
        if self.eyes_closed and self.blink_start_time is not None:
            self.closure_elapsed = now - self.blink_start_time
            if self.closure_elapsed >= s.goodbye_hold:
                events.append({"type": "goodbye"})
        else:
            self.closure_elapsed = 0.0

        return events


# ----------------------------------------------------------------------
# Informational gaze estimate (never used to decide blinks)
# ----------------------------------------------------------------------

def estimate_gaze(landmarks, blendshapes=None):
    """
    Return a label such as "CENTER", "LEFT", "DOWN-RIGHT" describing where
    the eyes are pointing in the (mirrored) preview.  Purely informational:
    shown to the patient/caregiver so it is obvious that looking around the
    chart is not counted as a blink.
    """
    horizontal = ""
    vertical = ""
    try:
        if len(landmarks) > 477:
            ratios = []
            for corner_a, corner_b, iris in ((33, 133, 468), (362, 263, 473)):
                xa, xb = landmarks[corner_a].x, landmarks[corner_b].x
                span = xb - xa
                if abs(span) > 1e-6:
                    ratios.append((landmarks[iris].x - xa) / span)
            if ratios:
                # Outer->inner direction differs per eye; fold to image-left
                # (0) .. image-right (1)
                r_first = ratios[0]
                r_second = 1.0 - ratios[1] if len(ratios) > 1 else r_first
                h = (r_first + r_second) / 2.0
                if h < 0.40:
                    horizontal = "LEFT"
                elif h > 0.60:
                    horizontal = "RIGHT"
    except Exception:
        pass

    if blendshapes:
        try:
            down = (blendshapes.get("eyeLookDownLeft", 0.0)
                    + blendshapes.get("eyeLookDownRight", 0.0)) / 2.0
            up = (blendshapes.get("eyeLookUpLeft", 0.0)
                  + blendshapes.get("eyeLookUpRight", 0.0)) / 2.0
            if down > 0.45:
                vertical = "DOWN"
            elif up > 0.45:
                vertical = "UP"
        except Exception:
            pass

    label = "-".join(part for part in (vertical, horizontal) if part)
    return label or "CENTER"
