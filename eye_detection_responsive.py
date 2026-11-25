"""
BlinkTap - "responsive" variant.

This file used to be a second, older copy of the whole application with its
own Morse table, TTS code and detection loop.  To avoid keeping two diverging
copies, it now starts the main application (eye_detection.py) with the
"responsive" preset, which keeps this variant's settings:

    * eye goodbye hold 1.5 s, tap goodbye hold 4.5 s
    * tap long-press threshold 0.40 s
    * stricter head-angle limits
    * no advisory ML classifier

All new features (Morse chart, quick messages, confirmation, gaze-aware blink
detection) are therefore available here too.

Run:  python eye_detection_responsive.py
"""

from eye_detection import main

if __name__ == "__main__":
    main(preset="responsive")
