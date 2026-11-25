# BlinkTap – Assistive Communication System

BlinkTap is a **Morse-code communication aid for conscious paralysis patients** who have difficulty speaking or moving normally but can **intentionally control their eye blinks and/or a tap input**. It is an assistive communication tool – **not** a medical diagnosis, treatment or patient-monitoring system. Use it only if the patient can reliably operate the chosen input; abilities differ between patients.

Short blink / short tap = **dot (.)**  ·  Long blink / long press = **dash (-)**

## Run
```
pip install -r requirements.txt
python calibration.py        # once per patient / camera position (recommended, eye mode)
python eye_detection.py      # start communicating
python -m pytest tests       # run the automated tests
```
`python eye_detection_responsive.py` starts the same app with the old "responsive" settings (1.5 s eye goodbye, 4.5 s tap goodbye, stricter head angle, no ML).

## Eye mode and Tapping mode are independent
Pick **Eye**, **Tap** or **Both** in the start menu (the menu appears *before* anything touches the camera).

* **Eye / Both** – the webcam is opened and MediaPipe is initialised.
* **Tap** – **no webcam, no MediaPipe, no eye processing at all.** Tap = press the mouse/touch anywhere in the window (short = dot, long press = dash).
* Press **M** to change method at any time. Leaving Eye mode releases the webcam and closes MediaPipe; entering it starts them again. If the webcam cannot be opened the app falls back to Tap mode instead of crashing.

## Timing (all configurable)
Starting values – **not medical standards**; every patient is different. Durations are the time the eyelids are actually closed.

| Setting | Start value | Meaning |
|---|---|---|
| Dot | 0.2 – 0.6 s | short blink |
| Dash | 0.8 – 1.5 s | long blink |
| Letter timeout | 2.5 s | pause that finishes a letter (screen flashes **LETTER DONE**, one beep) |
| Word timeout | 6 s | pause (counted from the last blink) that ends the word and adds a space (screen flashes **WORD DONE**, two beeps) |

* Dot and dash ranges **never overlap**. A blink *between* them (0.6–0.8 s) is ignored with a message ("between dot and dash, try again") instead of being guessed; shorter than the dot range = "too quick" (treated as a natural blink and ignored); longer than the dash range = "too long", ignored. The live bar on screen shows these zones while the eyes are closed.
* **Letter logic:** every dot/dash is added to the current sequence and the letter timeout restarts. The letter is only decoded when the timeout expires with no further blink; a space is added once when the word timeout expires. A letter is never finalised while the eyes are closed or a press is held (a slow dash cannot split a letter). A countdown bar shows the time left.
* **Change them:** in the app press `[` / `]` (letter timeout −/+ 0.5 s) and `;` / `'` (word timeout −/+ 0.5 s) – saved to `settings.json`; or edit `settings.json` (created on first change/calibration; invalid or overlapping values are repaired automatically; delete the file for defaults).
* **Calibration** (`calibration.py`) asks for ~15 comfortable short and ~15 comfortable long practice blinks (patients are not asked to be fast or precise) and estimates the patient's own dot/dash ranges with a guaranteed gap, writing them to `settings.json`. It also records open/closed/look-around eye data (see below). Re-running only some steps keeps the other saved values. It warns when the patient's short and long blinks are too similar.
* **Natural blinks:** blinks shorter than the dot range are ignored. A slow patient can raise the ranges (or calibrate) so natural blinks fall outside. See *Limitations*.

## Screen layout
**Upper section:** current Morse sequence (large dots/bars, with what it means and the timeout bar) · the **current letter** (very large) · the **complete message** (auto-sizing text) – or the confirmation prompt.
**Lower left:** camera preview (or "Webcam OFF" in Tap mode) and live input bars.
**Lower right:** the A–Z Morse chart plus digits 0–9 (Alphabet mode, letters that still match the sequence stay lit) or the **Custom Messages** list (Custom mode) – or the message editor.
**Bottom:** status line, control-command legend, caregiver keys and current timings. The 1280×780 layout scales to the window (starts at 1180×716, fits laptop screens).

## Custom Messages
Press **2** (or blink the *Custom messages* command `-...-`; blink `..-..` to go *Back to letters*) to open **Custom Messages mode**. Each message is shown with its blink pattern (e.g. `...` = "I need help.").
Select a message by any of:
* blinking/tapping its pattern, **or**
* caregiver: **N / B** move the highlight, **Enter** selects, **or** (Eye mode) click the message with the mouse.

The selected message is shown large with *"Speak this message?"*. **One long blink (-) / Enter = YES, speak**; **two short (..) / Esc = NO**. Nothing is spoken before confirmation.

Included examples: Yes, No, I need help, I need water, Please call someone, I am uncomfortable, I need to change my position, Thank you, Please wait, I am okay, I am in pain, I need rest.

**Add / edit / delete (caregiver or user):** press **E** → *N/B* choose · **A** add · **E** edit · **D** delete (asks Y to confirm) · **Esc/Q** close. Adding asks for the text, then the pattern typed with `.` and `-`. Duplicate patterns and control-command patterns are rejected. Changes are saved immediately to `quick_messages.json` (which can also be edited by hand). While the editor is open, blink/tap input is paused.
Custom patterns are project-defined shortcuts – they are **not** International Morse and are only decoded in Custom mode.

## Alphabet mode
Standard International Morse for **A–Z and 0–9** (plus the punctuation the original app already decoded). Unknown patterns are not inserted; a message is shown instead.

## Controls (existing features preserved)
| Key | Action |
|---|---|
| **Enter / T** | **Confirm and Speak** (first press asks, second speaks; in Custom mode first press selects the highlighted message) |
| **Backspace** | **Delete** last dot/dash, then last character |
| **C** | **Clear All** |
| **S** | **Space** (decodes the letter in progress first) |
| **Q** | **Quit** |
| R | repeat last spoken message |
| P | pause / resume input (patient resting) |
| 1 / 2 | Alphabet / Custom messages mode |
| N / B, E | choose message / edit messages |
| M | change input method |
| `[ ]` `;` `'` | letter / word timeouts |
| Esc | cancel confirmation |

Blink/tap control commands (work in both modes; defined in `morse_config.py`): **Send** `.-.-.` · **Erase 1 letter** `...-.` · **Clear all** `-.-.-` · **Custom messages** `-...-` · **Back to letters** `..-..` · **STOP (pause)** `--.--` · **START (resume)** `-..--`. While stopped the system ignores everything except the START code, so natural blinks while resting cannot type anything; the screen turns red and shows the START code. Keyboard: **P** pauses/resumes, and **Enter** (or **T**) also starts again while paused. Finished letters and words are read aloud (key **L** switches this on/off); **Send** reads the whole message after you confirm. All of them are also drawn on screen under the number chart. **Clear all** always asks "Clear everything?" first (long blink = yes, two short = no / wait to cancel), so a slip cannot wipe the message. Numbers 0-9 use standard Morse (e.g. `.----` = 1) and are shown in the NUMBERS row of the chart. The custom messages sheet ships with 24 frequently needed messages (needs, pain, breathing/emergency, family, comfort); edit them with **E** or in `quick_messages.json`.
Goodbye gestures are unchanged (hold eyes closed 3.5 s / hold press 3 s). Text-to-speech is unchanged (`pyttsx3`).

## Eye-blink detection and gaze
The original EAR detection is kept, with safeguards so looking around the chart is not read as a blink (`blink_detector.py`): both eyes must close; MediaPipe's eye-blink scores must agree with EAR; the closed threshold adapts to a slowly-updated open baseline; the score can end a closure so the detector cannot get stuck after a gaze change; confirm-frame counts, smoothing and cooldown stop one blink being counted repeatedly. If blink scores are unavailable it falls back to EAR only.

**Duration measurement:** the original code measured blink duration up to the moment the eyes were *confirmed open again* (about 0.13–0.17 s longer than the real closure). Dot/dash classification now uses the **true closure time** so the 0.2–0.6 / 0.8–1.5 s settings mean what they say. The original value is still passed to the advisory ML model, so `blink_model.pkl` and `blink_data.csv` are unchanged and still valid.

## Files
`eye_detection.py` menu, webcam/MediaPipe backend, TTS, main loop · `blink_detector.py` · `communication_engine.py` decoding, timeouts, modes, confirmation · `settings.py` (+ `settings.json`) timings · `morse_config.py` tables, control commands · `quick_messages.json` custom messages · `message_editor.py` · `ui_renderer.py` · `calibration.py`, `calibration_profile.py` · `train_model.py`, `camera_test.py`, `blink_model.pkl` unchanged · `tests/`.
`code.txt` / `code2.txt` are old text copies of earlier versions and were not touched.

## Behaviour changes vs. the original
* Letters finalise after a 2.5 s pause and spaces after a 6 s pause, never mid-blink. When a letter or word ends the app shows a green LETTER DONE / blue WORD DONE banner and beeps (Windows `winsound`; elsewhere only the terminal bell, which may be silent). An existing `settings.json` overrides these defaults - delete it or edit it.
* Unknown Morse patterns are no longer inserted as `?`.
* `T` asks for confirmation first; the spoken draft is then cleared (`R` repeats).
* A missing `blink_model.pkl` is a warning, not a fatal error.
* Model/profile files are found relative to the script.

## Limitations
* A natural blink can be hard to tell from a deliberate short one. Duration filtering only helps if the patient's dots are firmer/slower than natural blinks; in the supplied `blink_data.csv` the short blinks are about as brief as natural ones, so with the default 0.2 s minimum some of that patient's dots would be ignored until recalibrated with the new (true-closure) measure. Use `P` to pause while resting; erase/clear commands fix stray entries.
* The ≈0.13–0.17 s open-confirmation delay (5 frames) is still part of when a blink is *registered*, which delays feedback slightly.
* Webcam accuracy depends on lighting, camera height and head position.
* Automated tests use scripted/fake cameras and scripted signals; they do not replace trying the app with a real person.
