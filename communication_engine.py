"""
BlinkTap - communication engine (camera / screen independent).

Holds everything that used to be global state and helper functions in
eye_detection.py: the current Morse sequence, the decoded message, letter /
word gap timing, backspace / clear / space, and speaking - now with

* two separate modes (ALPHABET and QUICK) so the same blink pattern is never
  decoded by the wrong table,
* control commands (send, erase, clear, switch mode) usable with blinks/taps,
* a confirmation step: nothing is spoken until the patient (or caregiver)
  confirms.

Blinks and taps both enter through register_eye_signal / register_tap_signal,
exactly as before.
"""

import time

import morse_config
from settings import DEFAULT_SETTINGS, classify_duration

ALPHABET = "ALPHABET"
QUICK = "QUICK"


class EngineSettings:
    """
    Timing values.  Blink ranges are the time the eyelids are closed
    (seconds) and never overlap; see settings.py.
    """

    def __init__(self, **overrides):
        self.dot_min = DEFAULT_SETTINGS["dot_min"]
        self.dot_max = DEFAULT_SETTINGS["dot_max"]
        self.dash_min = DEFAULT_SETTINGS["dash_min"]
        self.dash_max = DEFAULT_SETTINGS["dash_max"]
        self.letter_timeout = DEFAULT_SETTINGS["letter_timeout"]
        self.word_timeout = DEFAULT_SETTINGS["word_timeout"]
        self.min_blink_duration = 0.06     # detector noise floor
        self.max_blink_duration = 4.0
        self.tap_long_threshold = 0.30
        self.tap_min_duration = 0.06
        self.tap_max_duration = 5.0
        self.confirm_timeout = 20.0
        self.quick_requires_confirmation = True
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown EngineSettings field: {key}")
            setattr(self, key, value)

    def timing_dict(self):
        return {k: getattr(self, k) for k in DEFAULT_SETTINGS}


class CommunicationEngine:

    def __init__(self, speak_fn=None, settings=None, quick_messages=None,
                 ml_predict=None, notify_fn=None, echo_fn=None):
        self.s = settings or EngineSettings()
        # notify_fn(kind) is called with "LETTER" or "WORD" when a letter /
        # word has just ended (used for a sound cue).
        self.notify_fn = notify_fn
        # echo_fn(text): reads a finished letter / word aloud (optional)
        self.echo_fn = echo_fn
        self.echo_enabled = True
        self.speak_fn = speak_fn or (lambda text: None)
        self.ml_predict = ml_predict
        self.quick_messages = (
            quick_messages
            if quick_messages is not None
            else morse_config.QUICK_MESSAGES
        )
        self.mode = ALPHABET
        self.ignored_natural = 0
        self.last_spoken = ""
        self.paused = False
        self.quick_cursor = 0
        self.clear_all(status="Ready")

    # ------------------------------------------------------------
    # state reset
    # ------------------------------------------------------------

    def clear_all(self, status="Cleared"):
        self.current_morse = ""
        self.decoded_message = ""
        self.last_signal_time = None
        self.letter_decoded = False
        self.word_space_added = False

        self.confirming = False
        self.confirm_text = ""
        self.confirm_source = None
        self.confirm_started = None

        self.last_letter = ""
        self.flash_kind = None        # "LETTER" / "WORD": shown on screen briefly
        self.flash_time = 0.0
        self.last_blink = "None"
        self.last_type = "None"
        self.last_duration = 0.0
        self.last_tap_duration = 0.0
        self.last_input_source = "None"
        self.last_ml_prediction = "None"
        self.last_action_status = status

    def _touch(self, now=None):
        self.last_signal_time = time.time() if now is None else now
        self.letter_decoded = False
        self.word_space_added = False

    # ------------------------------------------------------------
    # modes
    # ------------------------------------------------------------

    def set_mode(self, mode):
        if mode not in (ALPHABET, QUICK):
            raise ValueError(mode)
        self.mode = mode
        self.current_morse = ""
        self.last_signal_time = None
        self.letter_decoded = False
        self.word_space_added = False
        self._cancel_confirm(announce=False)
        name = "Alphabet" if mode == ALPHABET else "Custom Messages"
        self.last_action_status = f"{name} mode"
        print(f"Mode: {name}")

    def toggle_mode(self):
        self.set_mode(QUICK if self.mode == ALPHABET else ALPHABET)

    # ------------------------------------------------------------
    # input registration (eye / tap)
    # ------------------------------------------------------------

    def _classify_ml(self, ml_features):
        prediction = None
        if ml_features is not None and self.ml_predict is not None:
            prediction = self.ml_predict(ml_features)
            if prediction is not None:
                prediction = str(prediction).upper()
        self.last_ml_prediction = prediction if prediction else "None"
        return prediction

    def register_eye_signal(self, duration, ml_features=None, now=None,
                            closure=None):
        """
        duration: blink length as BlinkTap has always measured it (kept for
                  the ML model).  closure: true time the lids were shut.
                  Classification uses the closure when given.
        Returns "." / "-" if registered, otherwise None.
        """
        s = self.s

        if not (s.min_blink_duration <= duration <= s.max_blink_duration):
            print(f"Ignored closure: {duration:.3f}s")
            return None

        shut = duration if closure is None else closure
        kind = classify_duration(shut, self.s.timing_dict())

        if kind == "too_short":
            self.ignored_natural += 1
            self.last_action_status = (
                f"Ignored {shut:.2f}s - too quick (natural blink)"
            )
            print(f"Ignored natural blink: {shut:.3f}s")
            return None
        if kind == "ambiguous":
            self.last_action_status = (
                f"Ignored {shut:.2f}s - between dot and dash, try again"
            )
            print(f"Ambiguous blink ignored: {shut:.3f}s")
            return None
        if kind == "too_long":
            self.last_action_status = (
                f"Ignored {shut:.2f}s - too long, try a shorter dash"
            )
            print(f"Too long blink ignored: {shut:.3f}s")
            return None

        self.last_duration = shut
        self.last_input_source = "EYE"

        ml_prediction = self._classify_ml(ml_features)

        if kind == "dash":
            signal, signal_type = "-", "LONG BLINK"
        else:
            signal, signal_type = ".", "SHORT BLINK"

        self._add_signal(signal, signal_type, now)

        if self.paused:
            self.last_action_status = "PAUSED - blink the START code to resume"
            return signal

        if ml_prediction in ("SHORT_BLINK", "LONG_BLINK"):
            expected = "LONG_BLINK" if signal == "-" else "SHORT_BLINK"
            if ml_prediction == expected:
                self.last_action_status = f"{signal_type}: {signal} (ML agrees)"
            else:
                self.last_action_status = (
                    f"{signal_type}: {signal} (ML={ml_prediction}, "
                    "duration used)"
                )
        else:
            self.last_action_status = f"{signal_type}: {signal}"

        print(f"BLINK {shut:.3f}s -> {signal_type} {signal} | "
              f"sequence: {self.current_morse}")
        return signal

    def register_tap_signal(self, duration, now=None):
        s = self.s
        if not (s.tap_min_duration <= duration <= s.tap_max_duration):
            print(f"Ignored tap: {duration:.3f}s")
            return None

        self.last_tap_duration = duration
        self.last_input_source = "TAP"

        if duration >= s.tap_long_threshold:
            signal, signal_type = "-", "LONG TAP"
        else:
            signal, signal_type = ".", "SHORT TAP"

        self._add_signal(signal, signal_type, now)
        self.last_action_status = (
            "PAUSED - tap the START code to resume" if self.paused
            else f"{signal_type}: {signal}"
        )
        print(f"TAP {duration:.3f}s -> {signal_type} {signal} | "
              f"sequence: {self.current_morse}")
        return signal

    def _add_signal(self, signal, signal_type, now):
        self.current_morse += signal
        self.last_blink = signal
        self.last_type = signal_type
        self._touch(now)

    # ------------------------------------------------------------
    # timing: letter / word boundaries, confirmation timeout
    # ------------------------------------------------------------

    def tick(self, now=None, input_active=False):
        """
        Call every frame.  input_active = eyes currently closed / button
        currently pressed: a letter is never finalised in the middle of a
        blink or press (a long dash can take longer than the timeout).
        """
        now = time.time() if now is None else now

        if (
            self.confirming
            and self.confirm_started is not None
            and now - self.confirm_started >= self.s.confirm_timeout
        ):
            self._cancel_confirm(announce=False)
            self.last_action_status = "Confirmation timed out - nothing spoken"

        if self.last_signal_time is None or input_active:
            return

        gap = now - self.last_signal_time

        if self.paused:
            # Only the START code is listened for while paused.
            if gap >= self.s.letter_timeout and self.current_morse:
                seq, self.current_morse = self.current_morse, ""
                self.last_signal_time = None
                if seq == morse_config.CONTROL_COMMANDS["START"]:
                    self.set_paused(False)
                else:
                    self.last_action_status = (
                        "PAUSED - only the START code works "
                        f"({morse_config.CONTROL_COMMANDS['START']})"
                    )
            return

        if gap >= self.s.letter_timeout and self.current_morse and not self.letter_decoded:
            self.resolve_sequence(now)
            if not self.paused:          # STOP has its own beep
                self._announce("LETTER", set_status=self.letter_decoded)

        if (
            gap >= self.s.word_timeout
            and self.letter_decoded
            and not self.word_space_added
        ):
            self.add_word_space()
            self._announce("WORD")

    def _announce(self, kind, set_status=True):
        """Tell the user a letter/word has ended and they may continue."""
        self.flash_kind = kind
        self.flash_time = time.time()
        # Keep error / command messages ("Unknown pattern", "No quick
        # message", ...) visible; only overwrite for a normally decoded letter.
        if kind == "LETTER" and set_status:
            self.last_action_status = "Letter finished - you can start the next letter"
        elif kind == "WORD":
            self.last_action_status = "Word finished - you can start the next word"
        echoed = False
        if self.echo_enabled and self.echo_fn is not None:
            if kind == "LETTER" and set_status and self.mode == ALPHABET:
                echoed = self._echo(self.last_letter)
            elif kind == "WORD":
                words = self.decoded_message.split()
                echoed = self._echo(words[-1] if words else "")
        if not echoed:
            self._notify(kind)

    def _echo(self, text):
        text = str(text or "").strip()
        if not text:
            return False
        try:
            self.echo_fn(text)
            return True
        except Exception:
            return False

    def _notify(self, kind):
        if self.notify_fn is not None:
            try:
                self.notify_fn(kind)
            except Exception:
                pass

    def letter_progress(self, now=None):
        """0..1 progress of the letter timeout (None when nothing pending)."""
        now = time.time() if now is None else now
        if self.last_signal_time is None or not self.current_morse:
            return None
        return min(1.0, (now - self.last_signal_time) / self.s.letter_timeout)

    def word_progress(self, now=None):
        """0..1 progress of the word timeout after a letter was decoded."""
        now = time.time() if now is None else now
        if (self.last_signal_time is None or not self.letter_decoded
                or self.word_space_added or not self.decoded_message):
            return None
        return min(1.0, (now - self.last_signal_time) / self.s.word_timeout)

    def adjust_timeout(self, which, delta):
        """which: 'letter_timeout' | 'word_timeout'.  Returns new value."""
        s = self.s
        value = getattr(s, which) + delta
        if which == "letter_timeout":
            value = min(max(value, 0.5), 15.0)
            s.letter_timeout = round(value, 2)
            if s.word_timeout <= s.letter_timeout:
                s.word_timeout = round(s.letter_timeout + 1.0, 2)
        else:
            value = min(max(value, s.letter_timeout + 0.5), 30.0)
            s.word_timeout = round(value, 2)
        self.last_action_status = (
            f"Letter timeout {s.letter_timeout:.1f}s, "
            f"word timeout {s.word_timeout:.1f}s"
        )
        return getattr(s, which)

    # ------------------------------------------------------------
    # sequence resolution
    # ------------------------------------------------------------

    def preview(self):
        """
        What the sequence currently being entered WOULD mean, as
        (kind, text) or None.  kind is LETTER / QUICK / COMMAND / CONFIRM.
        Used by the UI; never changes state.
        """
        seq = self.current_morse
        if not seq:
            return None
        if self.paused:
            if seq == morse_config.CONTROL_COMMANDS["START"]:
                return ("COMMAND", morse_config.CONTROL_LABELS["START"])
            return None
        if self.confirming:
            if seq == morse_config.CONFIRM_YES:
                return ("CONFIRM", "Speak")
            if seq == morse_config.CONFIRM_NO:
                return ("CONFIRM", "Cancel")
            return None
        for name, pattern in morse_config.CONTROL_COMMANDS.items():
            if seq == pattern:
                return ("COMMAND", morse_config.CONTROL_LABELS[name])
        if self.mode == ALPHABET:
            ch = morse_config.MORSE_CODE.get(seq)
            return ("LETTER", ch) if ch else None
        msg = self.quick_messages.get(seq)
        return ("QUICK", msg) if msg else None

    def resolve_sequence(self, now=None):
        """Interpret current_morse (called at a letter gap)."""
        seq = self.current_morse
        if not seq:
            return False
        self.current_morse = ""
        self.letter_decoded = False  # may be set True below for letters

        # 1. inside the confirmation prompt
        if self.confirming:
            if seq == morse_config.CONFIRM_YES:
                self.confirm_speak()
            elif seq == morse_config.CONFIRM_NO:
                self._cancel_confirm()
            else:
                self.last_action_status = (
                    f"Not understood ({seq}). Long blink = speak, "
                    "two short = cancel"
                )
            return True

        # 2. control commands (both modes)
        for name, pattern in morse_config.CONTROL_COMMANDS.items():
            if seq == pattern:
                self._run_command(name)
                return True

        # 3. alphabet mode: standard Morse
        if self.mode == ALPHABET:
            ch = morse_config.MORSE_CODE.get(seq)
            if ch is None:
                # Unknown pattern: do not put garbage in the message.
                self.last_action_status = f"Unknown pattern {seq} - try again"
                print(f"Unknown Morse pattern: {seq}")
                return True
            self.decoded_message += ch
            self.last_letter = ch
            self.letter_decoded = True
            self.last_action_status = f"Decoded {seq} -> {ch}"
            print(f"Decoded letter: {seq} -> {ch} | Message: {self.decoded_message}")
            return True

        # 4. quick mode: project-defined shortcuts only
        msg = self.quick_messages.get(seq)
        if msg is None:
            self.last_action_status = f"No quick message for {seq}"
            print(f"No quick message for pattern {seq}")
            return True
        self._offer_quick(msg)
        return True

    def _offer_quick(self, msg):
        if self.s.quick_requires_confirmation:
            self._start_confirm(msg, "QUICK")
        else:
            self._speak(msg)

    def _run_command(self, name):
        if name == "SEND":
            self.request_confirm()
        elif name == "ERASE":
            self.erase_last()
        elif name == "CLEAR":
            self.request_clear()
        elif name == "MESSAGES":
            if self.mode == QUICK:
                self.last_action_status = "Already on custom messages"
            else:
                self.set_mode(QUICK)
        elif name == "STOP":
            self.set_paused(True)
        elif name == "START":
            self.last_action_status = "Already running"
        elif name == "BACK":
            if self.mode == ALPHABET:
                self.last_action_status = "Already on the letters page"
            else:
                self.set_mode(ALPHABET)

    # ------------------------------------------------------------
    # editing
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # custom (quick) message list: highlight + select without Morse
    # ------------------------------------------------------------

    def set_quick_messages(self, entries):
        """Replace the message list (list of (pattern, message))."""
        self.quick_messages = dict(entries)
        n = len(self.quick_messages)
        self.quick_cursor = min(self.quick_cursor, max(0, n - 1))

    def quick_move(self, delta):
        n = len(self.quick_messages)
        if n:
            self.quick_cursor = (self.quick_cursor + delta) % n

    def select_quick(self, index=None):
        """Pick a message from the list (caregiver key / click)."""
        items = list(self.quick_messages.items())
        if not items:
            self.last_action_status = "No custom messages defined"
            return False
        if index is not None:
            self.quick_cursor = min(max(index, 0), len(items) - 1)
        self.current_morse = ""
        pattern, message = items[self.quick_cursor]
        self._offer_quick(message)
        return True

    def decode_pending(self):
        """Decode the in-progress sequence now (S / T keys do this)."""
        if self.current_morse:
            self.resolve_sequence()

    def add_word_space(self):
        if not self.decoded_message:
            return False
        if not self.decoded_message.endswith(" "):
            self.decoded_message += " "
        self.word_space_added = True
        self.last_action_status = "Word space added"
        return True

    def space_key(self):
        """Keyboard S: finish the letter in progress, then add a space."""
        if self.mode == ALPHABET:
            if self.current_morse:
                self.decode_pending()
            self.last_signal_time = None
            self.letter_decoded = False
            self.word_space_added = False
            self.add_word_space()

    def backspace(self):
        """Remove the last dot/dash, else the last character."""
        if self.confirming:
            self._cancel_confirm()
            return
        if self.current_morse:
            removed = self.current_morse[-1]
            self.current_morse = self.current_morse[:-1]
            self.last_action_status = f"Removed {removed}"
        elif self.decoded_message:
            removed = self.decoded_message[-1]
            self.decoded_message = self.decoded_message[:-1]
            self.last_action_status = f"Removed {removed!r}"
        else:
            self.last_action_status = "Nothing to remove"
        self.last_signal_time = None
        self.letter_decoded = False
        self.word_space_added = False

    def erase_last(self):
        """ERASE command: delete the last decoded character."""
        # A word space is added automatically after a pause; erase should
        # remove the last real character, not just that space.
        self.decoded_message = self.decoded_message.rstrip(" ")
        if self.decoded_message:
            removed = self.decoded_message[-1]
            self.decoded_message = self.decoded_message[:-1]
            self.last_action_status = f"Erased {removed!r}"
        else:
            self.last_action_status = "Nothing to erase"
        self.letter_decoded = False
        self.word_space_added = False
        self.last_signal_time = None

    def clear_draft(self):
        """CLEAR command / C key: restart input."""
        self.current_morse = ""
        self.decoded_message = ""
        self.last_letter = ""
        self.last_signal_time = None
        self.letter_decoded = False
        self.word_space_added = False
        self._cancel_confirm(announce=False)
        self.last_action_status = "Cleared - start again"

    # ------------------------------------------------------------
    # confirmation + speaking
    # ------------------------------------------------------------

    def request_confirm(self):
        """SEND command / T key: ask before speaking the draft."""
        if self.current_morse:
            self.decode_pending()
            if self.confirming:   # a quick message got offered
                return
        if self.mode == QUICK:
            self.last_action_status = "Nothing to send - blink a quick message"
            return
        text = self.decoded_message.strip()
        if not text:
            self.last_action_status = "Nothing to speak yet"
            return
        self._start_confirm(text, "DRAFT")

    def request_clear(self):
        """CLEAR command: never wipes the draft without asking first."""
        self.current_morse = ""
        if not self.decoded_message:
            self.last_action_status = "Nothing to clear"
            return
        self._start_confirm(self.decoded_message.strip() or "(spaces)", "CLEAR")

    def _start_confirm(self, text, source):
        self.confirming = True
        self.confirm_text = text
        self.confirm_source = source
        self.confirm_started = time.time()
        self.last_signal_time = None
        if source == "CLEAR":
            self.last_action_status = (
                "Clear EVERYTHING?  Long blink = YES clear, two short = NO keep"
            )
        else:
            self.last_action_status = (
                f"Speak '{text}'?  Long blink = YES, two short = NO"
            )

    def _cancel_confirm(self, announce=True):
        was = self.confirming
        was_clear = self.confirm_source == "CLEAR"
        self.confirming = False
        self.confirm_text = ""
        self.confirm_source = None
        self.confirm_started = None
        if was and announce:
            self.last_action_status = (
                "Cancelled - message kept" if was_clear else "Cancelled - nothing spoken"
            )

    def cancel_confirm(self):
        """Esc key / caregiver: abandon the confirmation prompt."""
        if self.confirming:
            self._cancel_confirm()

    def confirm_speak(self):
        """Patient/caregiver confirmed: speak now."""
        if not self.confirming or not self.confirm_text:
            self.last_action_status = "Nothing to confirm"
            return
        text, source = self.confirm_text, self.confirm_source
        self._cancel_confirm(announce=False)
        if source == "CLEAR":
            self.clear_draft()
            return
        self._speak(text)

    def _speak(self, text):
        self.last_spoken = text
        self.last_action_status = f"Speaking: {text}"
        print(f"Speak: {text!r}")
        self.speak_fn(text)

    def speak_key(self):
        """Keyboard T / Enter (caregiver): ask, then confirm on 2nd press."""
        if self.paused:
            self.set_paused(False)       # Enter / T also START again
        elif self.confirming:
            self.confirm_speak()
        elif self.mode == QUICK and not self.current_morse:
            self.select_quick()          # speak the highlighted message
        else:
            self.request_confirm()

    def set_paused(self, paused):
        """Pause / resume.  While paused only the START code is heard."""
        self.paused = bool(paused)
        self.current_morse = ""
        self.last_signal_time = None
        self.letter_decoded = False
        self.last_action_status = (
            "PAUSED - blink the START code "
            f"({morse_config.CONTROL_COMMANDS['START']}) or press Enter / P to resume"
            if self.paused else "Resumed - you can blink again"
        )
        self._notify("PAUSE" if self.paused else "RESUME")

    def toggle_pause(self):
        """Caregiver key P: pause / resume the patient's input."""
        self.set_paused(not self.paused)

    def repeat_last(self):
        if self.last_spoken:
            self._speak(self.last_spoken)
        else:
            self.last_action_status = "Nothing spoken yet"
