"""
BlinkTap - caregiver editor for the Custom Messages list.

Opened with the E key in the app.  Pure keyboard logic (no drawing, no files): the
interface draws .state and friends, and on_save(entries) persists the list.

  LIST     n / b  move down / up     a  add     e  edit     d  delete
           Esc or Q  close
  TEXT     type the message, Enter = next, Esc = cancel
  PATTERN  type . and - (short / long), Enter = save, Esc = cancel
  DELETE   y = really delete, any other key = keep it
"""

import morse_config

LIST, TEXT, PATTERN, DELETE = "LIST", "TEXT", "PATTERN", "DELETE"

KEY_ENTER = (13, 10)
KEY_BACKSPACE = (8, 127)
KEY_ESC = 27


class MessageEditor:

    MAX_TEXT = 60
    MAX_PATTERN = 8

    def __init__(self, entries, on_save):
        self.entries = [list(e) for e in entries]   # [pattern, message]
        self.on_save = on_save
        self.active = False
        self.state = LIST
        self.cursor = 0
        self.text_buf = ""
        self.pattern_buf = ""
        self.editing_index = None
        self.error = ""
        self.info = ""

    # ------------------------------------------------------------

    def open(self, entries=None):
        if entries is not None:
            self.entries = [list(e) for e in entries]
        self.active = True
        self.state = LIST
        self.cursor = min(self.cursor, max(0, len(self.entries) - 1))
        self.error = self.info = ""

    def close(self):
        self.active = False
        self.state = LIST
        self.error = self.info = ""

    @property
    def as_pairs(self):
        return [(p, m) for p, m in self.entries]

    def _save(self):
        try:
            self.on_save(self.as_pairs)
            return True
        except Exception as e:  # disk full, read-only folder ...
            self.error = f"Could not save: {e}"
            return False

    # ------------------------------------------------------------

    def handle_key(self, key):
        """Returns True if the key was used by the editor."""
        if not self.active:
            return False
        self.error = ""
        self.info = ""
        if self.state == LIST:
            self._key_list(key)
        elif self.state == TEXT:
            self._key_text(key)
        elif self.state == PATTERN:
            self._key_pattern(key)
        elif self.state == DELETE:
            self._key_delete(key)
        return True

    def _key_list(self, key):
        ch = chr(key).lower() if 0 <= key < 256 else ""
        n = len(self.entries)
        if key == KEY_ESC or ch == "q":
            self.close()
        elif ch == "n" and n:
            self.cursor = (self.cursor + 1) % n
        elif ch == "b" and n:
            self.cursor = (self.cursor - 1) % n
        elif ch == "a":
            self.editing_index = None
            self.text_buf = ""
            self.pattern_buf = ""
            self.state = TEXT
        elif ch == "d" and n:
            self.state = DELETE
        elif ch == "e" and n:
            self._start_edit()

    def _start_edit(self):
        self.editing_index = self.cursor
        self.pattern_buf, self.text_buf = self.entries[self.cursor]
        self.state = TEXT

    def _key_text(self, key):
        if key == KEY_ESC:
            self.state = LIST
        elif key in KEY_ENTER:
            if not self.text_buf.strip():
                self.error = "Type the message first."
            else:
                self.state = PATTERN
        elif key in KEY_BACKSPACE:
            self.text_buf = self.text_buf[:-1]
        elif 32 <= key <= 126 and len(self.text_buf) < self.MAX_TEXT:
            self.text_buf += chr(key)

    def _key_pattern(self, key):
        if key == KEY_ESC:
            self.state = LIST
        elif key in KEY_BACKSPACE:
            self.pattern_buf = self.pattern_buf[:-1]
        elif key in (ord("."), ord("-")) and len(self.pattern_buf) < self.MAX_PATTERN:
            self.pattern_buf += chr(key)
        elif key in KEY_ENTER:
            self._commit()

    def _commit(self):
        pattern = self.pattern_buf.strip()
        message = self.text_buf.strip()
        if not pattern:
            self.error = "Type a pattern using . (short) and - (long)."
            return
        if pattern in morse_config.CONTROL_COMMANDS.values():
            self.error = "That pattern is a control command - choose another."
            return
        for i, (p, m) in enumerate(self.entries):
            if p == pattern and i != self.editing_index:
                self.error = f"Pattern already used by: {m}"
                return

        previous = [list(e) for e in self.entries]
        if self.editing_index is None:
            self.entries.append([pattern, message])
            self.cursor = len(self.entries) - 1
        else:
            self.entries[self.editing_index] = [pattern, message]
            self.cursor = self.editing_index

        if self._save():
            self.info = "Saved"
            self.state = LIST
        else:
            self.entries = previous

    def _key_delete(self, key):
        if key in (ord("y"), ord("Y")):
            previous = [list(e) for e in self.entries]
            removed = self.entries.pop(self.cursor)
            if self._save():
                self.info = f"Deleted: {removed[1]}"
                self.cursor = min(self.cursor, max(0, len(self.entries) - 1))
            else:
                self.entries = previous
        self.state = LIST
