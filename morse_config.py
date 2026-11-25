"""
BlinkTap - Morse and Quick Message configuration (single source of truth).

Two completely separate code tables live here:

1. ALPHABET_MORSE   - standard International Morse code for A-Z.
                      Never edit these.  MORSE_CODE (used by the decoder) is
                      this table plus the digits / punctuation that the
                      original BlinkTap decoder already supported.

2. QUICK_MESSAGES   - project-defined "Morse-style" shortcuts loaded from
                      quick_messages.json.  These are NOT International Morse.
                      They are only decoded in Quick Messages mode.

Control commands (send / erase / clear / switch mode) are long, standard
procedural signals (5+ elements) that cannot be confused with any letter or
digit.  They work in both modes.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK_MESSAGES_FILE = os.path.join(HERE, "quick_messages.json")

# ------------------------------------------------------------
# 1. Standard International Morse alphabet (A-Z).  DO NOT CHANGE.
# ------------------------------------------------------------

ALPHABET_MORSE = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
}

# Standard International Morse digits.
DIGIT_MORSE = {
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}

# Extra symbols the original BlinkTap decoder already understood.
# Kept so nothing that used to work stops working.
_EXTRA_SYMBOLS = {code: digit for digit, code in DIGIT_MORSE.items()}
_EXTRA_SYMBOLS.update({
    ".-.-.-": ".", "--..--": ",", "..--..": "?", "-.-.--": "!",
    "-....-.": "-", ".----.": "'", ".-..-.": '"', "-..-.": "/",
    ".--.-.": "@",
})

# pattern -> character (the table the Alphabet-mode decoder uses)
MORSE_CODE = {code: letter for letter, code in ALPHABET_MORSE.items()}
MORSE_CODE.update(_EXTRA_SYMBOLS)

# ------------------------------------------------------------
# 2. Control commands (work in BOTH modes)
#    All are >= 5 elements so they cannot be a letter or digit.
#    Change them here; the UI legend updates automatically.
# ------------------------------------------------------------

CONTROL_COMMANDS = {
    "SEND":     ".-.-.",    # standard "end of message" (AR): ask to speak draft
    "ERASE":    "...-.",    # delete ONE character (5 elements, not a letter/number)
    "CLEAR":    "-.-.-",    # delete EVERYTHING (5 elements, not a letter/number);
                            # the app always asks "Clear everything?" first
    "MESSAGES": "-...-",    # standard "break" (BT): open the custom messages sheet
    "BACK":     "..-..",    # return from the messages sheet to the letters page
    "STOP":     "--.--",    # pause: ignore everything except the START code
    "START":    "-..--",    # resume after STOP (the only code heard while paused)
}

CONTROL_LABELS = {
    "SEND":     "Send / Speak",
    "ERASE":    "Erase 1",
    "CLEAR":    "Clear all",
    "MESSAGES": "Messages",
    "BACK":     "Letters",
    "STOP":     "STOP (pause)",
    "START":    "START (resume)",
}

# Inside the "Speak this message?" confirmation prompt only:
CONFIRM_YES = "-"    # one long blink / long tap  -> speak
CONFIRM_NO = ".."    # two short blinks / taps    -> cancel

# ------------------------------------------------------------
# Quick message loading
# ------------------------------------------------------------

DEFAULT_QUICK_MESSAGES = [
    (".", "Yes, please."),
    ("-", "No, thank you."),
    ("...", "I need help."),
    (".-.", "I am in pain."),
    (".--", "I need water."),
    ("-.", "I need food."),
    ("---", "I need the toilet."),
    ("-..", "Please change my position."),
    ("..--", "I am too hot."),
    ("..-.", "I am too cold."),
    ("..-", "I am uncomfortable."),
    ("-.-.", "I need my medication."),
    (".--.", "Please call the nurse."),
    (".---", "Please call my family."),
    ("...-", "I am having trouble breathing."),
    ("..", "Please wait."),
    (".-", "I need to rest."),
    ("--.", "Thank you."),
    ("-.-", "Please repeat that."),
    ("--", "I am okay."),
]


def validate_quick_messages(entries):
    """
    Validate a list of (pattern, message) pairs.

    Returns (valid_list, warnings).  Invalid or conflicting entries are
    skipped with a warning instead of crashing the application.
    """
    valid = []
    warnings = []
    seen = set()
    control_patterns = set(CONTROL_COMMANDS.values())
    reserved = {CONFIRM_YES, CONFIRM_NO}  # only matter inside the prompt

    for item in entries:
        try:
            pattern, message = item
        except (TypeError, ValueError):
            warnings.append(f"Ignored malformed quick message entry: {item!r}")
            continue

        pattern = str(pattern).strip()
        message = str(message).strip()

        if not pattern or any(ch not in ".-" for ch in pattern):
            warnings.append(
                f"Ignored {message!r}: pattern {pattern!r} must contain only "
                "'.' and '-'."
            )
            continue
        if not message:
            warnings.append(f"Ignored empty message for pattern {pattern!r}.")
            continue
        if pattern in seen:
            warnings.append(
                f"Ignored {message!r}: pattern {pattern!r} is already used."
            )
            continue
        if pattern in control_patterns:
            warnings.append(
                f"Ignored {message!r}: pattern {pattern!r} is a control "
                "command."
            )
            continue

        seen.add(pattern)
        valid.append((pattern, message))

    # Informational only: patterns that equal the confirm/cancel gestures are
    # fine because confirmation is a separate state, so no warning needed.
    del reserved
    return valid, warnings


def load_quick_messages(path=QUICK_MESSAGES_FILE):
    """
    Load quick messages from JSON.  Falls back to the built-in defaults if
    the file is missing or unreadable.  Returns (list_of_(pattern,message),
    warnings).
    """
    warnings = []
    entries = None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("messages", []) if isinstance(data, dict) else data
        entries = []
        for row in raw:
            if isinstance(row, dict):
                entries.append((row.get("pattern", ""), row.get("message", "")))
            else:
                entries.append(row)
    except FileNotFoundError:
        warnings.append(
            f"{os.path.basename(path)} not found - using built-in quick "
            "messages."
        )
    except Exception as e:  # malformed JSON etc.
        warnings.append(
            f"Could not read {os.path.basename(path)} ({e}) - using built-in "
            "quick messages."
        )

    if entries is None:
        entries = list(DEFAULT_QUICK_MESSAGES)

    valid, more = validate_quick_messages(entries)
    warnings.extend(more)

    if not valid:
        valid, _ = validate_quick_messages(DEFAULT_QUICK_MESSAGES)
        warnings.append("No valid quick messages found - using defaults.")

    return valid, warnings


def save_quick_messages(entries, path=QUICK_MESSAGES_FILE):
    """Write (pattern, message) pairs back to quick_messages.json."""
    valid, warnings = validate_quick_messages(entries)
    data = {
        "_comment": (
            "Custom messages. Press E in the app to add, edit or delete "
            "them (saved automatically), or edit this file by hand. "
            "Patterns use '.' for a short blink/tap and '-' for a long "
            "blink/tap. They are project-defined shortcuts, NOT "
            "International Morse, and are only used in Custom Messages "
            "mode. Each pattern must be unique and must not equal a "
            "control command (see morse_config.py)."
        ),
        "messages": [{"pattern": p, "message": m} for p, m in valid],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return valid, warnings


def validate_control_commands():
    """Control commands must never be decodable as a letter/digit/symbol."""
    problems = []
    seen = {}
    for name, pattern in CONTROL_COMMANDS.items():
        if pattern in MORSE_CODE:
            problems.append(
                f"Control command {name} ({pattern}) collides with the "
                f"character {MORSE_CODE[pattern]!r}."
            )
        if pattern in seen:
            problems.append(
                f"Control commands {name} and {seen[pattern]} share {pattern}."
            )
        seen[pattern] = name
    return problems


# ------------------------------------------------------------
# Public, ready-to-use objects
# ------------------------------------------------------------

QUICK_MESSAGE_LIST, QUICK_MESSAGE_WARNINGS = load_quick_messages()
QUICK_MESSAGES = dict(QUICK_MESSAGE_LIST)  # pattern -> message

_problems = validate_control_commands()
if _problems:  # developer error, fail loudly
    raise ValueError("; ".join(_problems))

for _w in QUICK_MESSAGE_WARNINGS:
    print("Quick messages:", _w)


def reload_quick_messages():
    """Re-read quick_messages.json (used by tests / future live reload)."""
    global QUICK_MESSAGE_LIST, QUICK_MESSAGE_WARNINGS, QUICK_MESSAGES
    QUICK_MESSAGE_LIST, QUICK_MESSAGE_WARNINGS = load_quick_messages()
    QUICK_MESSAGES = dict(QUICK_MESSAGE_LIST)
    return QUICK_MESSAGE_LIST
