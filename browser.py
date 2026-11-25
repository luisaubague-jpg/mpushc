"""
MPushC sample browser: walk folders on the Mac / external SSD from the Push,
hear each sample as you land on it.

seq.py owns the buttons and knobs and calls into this class.
"""

import os
import threading
import time
import numpy as np
import soundfile as sf

AUDIO_EXT = (".wav", ".aif", ".aiff", ".mp3", ".flac")
PREVIEW_SECONDS = 20          # long loops: only the first 20 s are loaded for preview
SETTLE = 0.15                 # wait this long on an item before previewing it
ROWS = 6                      # list rows on screen
FAVS = "*favorites"           # virtual folders (self.path values)
SEARCH = "*search"
MAX_RESULTS = 500
SEARCH_SECONDS = 15           # give up walking after this long (huge drives)


class Browser:
    def __init__(self, roots, sr, on_preview, on_stop, favorites=None, on_favorites=None, search_roots=None):
        # roots: list of (label, path) · favorites: a list of paths kept by the app
        self.roots = [(label, path) for label, path in roots]
        self.favorites = favorites if favorites is not None else []
        self.search_roots = search_roots or [p for _, p in self.roots]
        self.on_favorites = on_favorites or (lambda: None)
        self.query, self.results, self.searching = "", [], False
        self.search_from = None   # (path, index, history) to go back to
        self.sr = sr
        self.on_preview, self.on_stop = on_preview, on_stop
        self.active = False
        self.path = None          # None = the list of roots
        self.items = []           # (name, full path, is_folder)
        self.index = 0
        self.history = []         # index we had in each parent folder
        self.moved_at = 0.0
        self.previewed = None     # path of the last previewed file
        self.preview = None       # dict: path, audio, peaks, seconds
        self.message = ""
        self._load_list()

    # ----------------------------------------------------------- listing
    def _load_list(self):
        self.message = ""
        if self.path is None:
            self.items = [("FAVORITES", FAVS, True)] + [(label, path, True) for label, path in self.roots]
            missing = [label for label, path in self.roots if not os.path.isdir(path)]
            if missing:
                self.message = "not connected: " + ", ".join(missing)
            return
        if self.path == FAVS:
            favs = [(os.path.basename(p.rstrip(os.sep)) + "   · " + self.short(os.path.dirname(p)), p,
                     os.path.isdir(p)) for p in self.favorites]
            self.items = sorted(favs, key=lambda x: (not x[2], x[0].lower()))
            if not self.items:
                self.message = "no favorites yet: FAV on a sample or folder"
            return
        if self.path == SEARCH:
            self.items = list(self.results)
            if self.searching:
                self.message = f"searching for \"{self.query}\"..."
            elif not self.items:
                self.message = f"nothing found for \"{self.query}\""
            return
        folders, files = [], []
        try:
            with os.scandir(self.path) as it:
                for e in it:
                    if e.name.startswith("."):
                        continue
                    if e.is_dir():
                        folders.append((e.name, e.path, True))
                    elif e.name.lower().endswith(AUDIO_EXT):
                        files.append((e.name, e.path, False))
        except OSError as err:
            self.message = f"can't open folder ({err.strerror})"
        key = lambda x: x[0].lower()
        self.items = sorted(folders, key=key) + sorted(files, key=key)
        if not self.items and not self.message:
            self.message = "no samples or folders here"

    # ----------------------------------------------------------- navigation
    def toggle(self):
        self.active = not self.active
        if not self.active:
            self.on_stop()

    def move(self, d):
        if not self.items:
            return
        new = max(0, min(len(self.items) - 1, self.index + d))
        if new != self.index:
            self.index = new
            self.moved_at = time.monotonic()
            self.on_stop()

    def current(self):
        return self.items[self.index] if self.items else None

    def enter(self):
        item = self.current()
        if item is None:
            return
        name, path, is_folder = item
        if is_folder:
            if path != FAVS and not os.path.isdir(path):
                self.message = f"{name}: not connected"
                return
            self.history.append(self.index)
            if self.path == FAVS:
                self._came_from = {path: FAVS}
            self.path, self.index = path, 0
            self.moved_at = time.monotonic()
            self._load_list()
        else:
            self.previewed = None           # right arrow on a file = hear it again
            self.moved_at = 0.0

    def back(self):
        if self.path is None:
            return
        if self.path == SEARCH and self.search_from is not None:   # back to where you searched
            self.path, idx, self.history = self.search_from
            self.search_from = None
            self._load_list()
            self.index = min(idx, max(0, len(self.items) - 1))
            self.moved_at = time.monotonic()
            self.on_stop()
            return
        root_paths = [p for _, p in self.roots]
        if self.path in (FAVS, SEARCH) or self.path in root_paths:
            self.path = None
        elif self.history_root() == FAVS:
            self.path = FAVS                   # opened from FAVORITES: back to the list
        else:
            self.path = os.path.dirname(self.path)
        self._load_list()
        self.index = min(self.history.pop() if self.history else 0, max(0, len(self.items) - 1))
        self.moved_at = time.monotonic()
        self.on_stop()

    def history_root(self):
        return getattr(self, "_came_from", {}).get(self.path)

    def short(self, path):
        for label, root in self.roots:
            if path.startswith(root):
                rel = os.path.relpath(path, root)
                return label if rel == "." else f"{label} / {rel.replace(os.sep, ' / ')}"
        return path

    # ----------------------------------------------------------- favorites + search
    def is_favorite(self, path):
        return path in self.favorites

    def toggle_favorite(self):
        """FAV on the highlighted sample or folder. Returns the new state (None = nothing)."""
        item = self.current()
        if item is None or item[1] in (FAVS, SEARCH) or (self.path is None):
            return None
        path = item[1]
        if path in self.favorites:
            self.favorites.remove(path)
            on = False
        else:
            self.favorites.append(path)
            on = True
        self.on_favorites()
        if self.path == FAVS:
            idx = self.index
            self._load_list()
            self.index = min(idx, max(0, len(self.items) - 1))
        return on

    def search(self, text):
        """Files whose name (or folder) holds every word, under the folder you're in
        (or the whole library from the top / favorites). Runs in the background."""
        words = text.lower().split()
        if not words:
            return
        here = self.path if self.path not in (None, FAVS, SEARCH) else None
        tops = [here] if here else [p for p in self.search_roots if os.path.isdir(p)]
        if self.path != SEARCH:
            self.search_from = (self.path, self.index, list(self.history))
        self.query, self.results, self.searching = text, [], True
        self.path, self.index, self.history = SEARCH, 0, []
        self._load_list()
        self.on_stop()
        threading.Thread(target=self._search, args=(tops, words, text), daemon=True).start()

    def _search(self, tops, words, text):
        found, t0 = [], time.monotonic()
        for top in tops:
            for folder, dirs, files in os.walk(top):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                hay_dir = os.path.relpath(folder, top).lower()
                for f in sorted(files):
                    if f.startswith(".") or not f.lower().endswith(AUDIO_EXT):
                        continue
                    hay = hay_dir + "/" + f.lower()
                    if all(w in hay for w in words):
                        full = os.path.join(folder, f)
                        found.append((f + "   · " + self.short(folder), full, False))
                if len(found) >= MAX_RESULTS or time.monotonic() - t0 > SEARCH_SECONDS:
                    break
            if len(found) >= MAX_RESULTS or time.monotonic() - t0 > SEARCH_SECONDS:
                break
        if self.query != text:                     # a newer search took over
            return
        self.results, self.searching = found[:MAX_RESULTS], False
        if self.path == SEARCH:
            self._load_list()
            self.index, self.moved_at = 0, time.monotonic()
            if len(found) >= MAX_RESULTS:
                self.message = f"first {MAX_RESULTS} results: search for more words"

    def where(self):
        if self.path == FAVS:
            return "Favorites"
        if self.path == SEARCH:
            n = "" if self.searching else f" ({len(self.results)})"
            return f'Search: "{self.query}"{n}'
        if self.path is None:
            return "Library"
        for label, root in self.roots:
            if self.path.startswith(root):
                rel = os.path.relpath(self.path, root)
                return label if rel == "." else f"{label} / {rel.replace(os.sep, ' / ')}"
        return self.path

    # ----------------------------------------------------------- preview
    def tick(self):
        """Called from the main loop: preview a file once you rest on it."""
        item = self.current()
        if not self.active or item is None or item[2]:
            return
        if item[1] == self.previewed or time.monotonic() - self.moved_at < SETTLE:
            return
        self.previewed = item[1]
        threading.Thread(target=self._load_preview, args=(item[1],), daemon=True).start()

    def _load_preview(self, path):
        try:
            info = sf.info(path)
            frames = min(info.frames, int(PREVIEW_SECONDS * info.samplerate))
            data, sr = sf.read(path, frames=frames, dtype="float32", always_2d=True)
        except Exception as err:                       # unreadable / odd format
            self.message = f"can't read this file ({type(err).__name__})"
            return
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        data = data[:, :2]
        if sr != self.sr:
            n = int(len(data) * self.sr / sr)
            x_old, x_new = np.linspace(0, 1, len(data)), np.linspace(0, 1, n)
            data = np.column_stack([np.interp(x_new, x_old, data[:, c]) for c in range(2)]).astype(np.float32)
        if path != self.previewed:                     # you already moved on
            return
        mono = np.abs(data).max(axis=1)
        p = np.array([c.max() if len(c) else 0.0 for c in np.array_split(mono, 360)])
        self.preview = {"path": path, "audio": data, "peaks": p / p.max() if p.max() > 0 else p,
                        "seconds": info.frames / info.samplerate}
        self.message = ""
        self.on_preview(data)

    # ----------------------------------------------------------- drawing
    def draw(self, d, W, H, fonts, colours):
        small, tiny = fonts
        ink, quiet, accent, wave = colours
        top = 24
        d.text((12, 10), self.where()[-70:], font=small, fill=accent, anchor="lm")
        if self.items:
            d.text((560, 10), f"{self.index + 1} / {len(self.items)}", font=tiny, fill=quiet, anchor="lm")
        # list
        row_h = (H - top - 18) / ROWS
        first = max(0, min(self.index - ROWS // 2, len(self.items) - ROWS))
        for r in range(ROWS):
            i = first + r
            if i >= len(self.items):
                break
            name, _, is_folder = self.items[i]
            y = top + r * row_h
            if i == self.index:
                d.rectangle([8, y + 1, 548, y + row_h - 1], fill=(45, 45, 58))
            label = (name + "  /") if is_folder else name
            d.text((16, y + row_h / 2), label[:56], font=small,
                   fill=ink if i == self.index else (quiet if is_folder else (170, 170, 180)), anchor="lm")
        # hints
        d.text((12, H - 9), "knob 1: scroll   knob 2: fast   < back   > open / replay   Browse: close",
               font=tiny, fill=quiet, anchor="lm")
        # right side: preview of the highlighted sample
        x0 = 570
        item = self.current()
        if self.message:
            d.text((x0, 40), self.message[:45], font=tiny, fill=(230, 120, 90), anchor="lm")
        if item and not item[2] and self.preview and self.preview["path"] == item[1]:
            pv = self.preview
            mid, half = 92, 40
            for i, p in enumerate(pv["peaks"]):
                h = max(1, int(p * half))
                d.line([(x0 + i, mid - h), (x0 + i, mid + h)], fill=wave)
            d.text((x0, 146), f"{pv['seconds']:.2f} s", font=tiny, fill=quiet, anchor="lm")
        elif item and item[2]:
            d.text((x0, 92), "folder: > to open", font=small, fill=quiet, anchor="lm")
