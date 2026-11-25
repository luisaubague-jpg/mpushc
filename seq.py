"""
MPushC - kit + step sequencer (1-8 bars), with knob pages.

Pads
  * Bottom-left 4x4: your kit (kit/ folder). Hit a pad to play it AND select it.
  * Top two rows: 16 steps of the bar shown (top row 1-8, next row 9-16).
Page buttons (the row right under the 8 knobs) choose what the 8 knobs do:
  * 1 PAD: volume, pitch, pan, swing, humanize of the selected pad
  * 2-5 FILTER / EQ / COMP / SAMPLE: coming later (dark for now)
  * 7 PATTERN: length, swing, humanize, BPM      * 8 SETUP: click volume, velocity curve
  The top of the screen labels every knob on the current page.
Other controls
  * Play: start / stop     * Tempo encoder: BPM     * Master encoder: volume
  * Swing encoder: swing of the selected pad
  * Hold Mute or Solo + hit a kit pad: mute / solo it (sequenced playback)
  * Session / Note: grid / waveform view     * Double Loop: double the pattern
  * Page < / >: bar on the step pads   * Layout: follow playhead   * Up/Down: zoom
  * Undo: undo   * Shift + Undo: redo   * Hold Delete + kit pad: clear its steps
    (again, on a pad with no steps: remove the sample). Everything musical is undoable.
  * Buttons under the screen: patterns 1-8 (Left/Right arrows: 1-8 / 9-16). While playing,
    a new pattern starts at the end of the loop. Duplicate: copy to the next empty slot;
    hold Duplicate + slot: copy there; hold Delete + slot: clear it.
  * Metronome: click on/off   * Tap Tempo: tap 2+ times to set the BPM
  * Hold a step pad: knob 1 = its velocity, knob 2 = nudge early/late
  * Quantize: pull the selected pad's hits halfway to the grid (Shift: whole pattern)
  * Record: record pad hits into the pattern (quantize: SETUP page, off by default)
  * Repeat: hold a pad to repeat it; the 8 buttons right of the pads pick the rate
  * Select: the buttons under the screen switch between PATTERNS and BANKS (A-H, 16 pads each)
  * Octave up/down: select the pad above/below on the grid, in any bank (Shift: 8 at a time)
  * SAMPLE page = chop the selected pad: SLICES knob (equal), Record (tap to chop),
    MARKER/MOVE/FINE knobs, Up/Down zoom the waveform, Left/Right scroll it, pads play slices, New (twice) = slices onto the next empty bank
  * Convert: resample the pattern onto the first empty pad (SAMPLE page or hold a pad: that pad)
    · Shift+Convert: export stems
  * Browse: sample browser (knob 1 scroll, knob 2 fast, < back, > open, hear on landing)
Everything autosaves to project.json and reloads at start-up.
Ctrl+C to quit.
"""

import gc
import glob
try:
    import native_fx                    # our own effects (needs scipy)
except ImportError:
    native_fx = None
    print("Our own effects are off: .venv/bin/python -m pip install scipy")
import sys
import os
import math
import random
import threading
import time
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw

import push2
from browser import Browser
from play import load_sample, velocity_to_gain, SR, BLOCK, MAX_VOICES
from render import load_font, BG, INK, QUIET, WAVE, PLAYED

W, H = push2.WIDTH, push2.HEIGHT

# ---------------------------------------------------------------------------
# Kit
# ---------------------------------------------------------------------------
# Pads are identified by a pad id 0-127: 8 banks (A-H) x 16 pads. The 16 kit
# pads on the Push (bottom-left 4x4) show the current bank.
NUM_BANKS, BANK_NAMES = 8, "ABCDEFGH"
KIT_NOTES = [push2.pad_note(i // 4, i % 4) for i in range(16)]  # MIDI notes of the kit pads
KIT_ALL = list(range(NUM_BANKS * 16))                            # every pad id
bank = {"cur": 0}


def note_to_pid(note):
    """The kit pad you hit -> pad id in the current bank (None if not a kit pad)."""
    return bank["cur"] * 16 + KIT_NOTES.index(note) if note in KIT_NOTES else None


def pad_name(pid):
    return f"{BANK_NAMES[pid // 16]}{pid % 16 + 1}"
pad_samples = {}                                   # note -> (name, audio), loaded pads only
pad_files = {}                                     # note -> file path the sample came from
kit_notes = []                                     # loaded pads, in pad order


def refresh_kit_notes():
    kit_notes[:] = [n for n in KIT_ALL if n in pad_samples]

# ---------------------------------------------------------------------------
# Tuning (knob 1). Samples are assumed to be C3 until pitch detection exists.
# ---------------------------------------------------------------------------
ROOT, TUNE_RANGE, TICKS_PER_SEMITONE = 60, 24, 4
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
tune = {n: 0 for n in KIT_ALL}
pitched = {}


def note_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 2}"


SINC_HALF, SINC_OVER = 16, 512                     # 32 taps, table resolution
_sinc_tables = {}


def _sinc_table(cutoff):
    t = _sinc_tables.get(cutoff)
    if t is None:
        x = np.arange(-SINC_HALF * SINC_OVER, SINC_HALF * SINC_OVER + 1) / SINC_OVER
        t = (cutoff * np.sinc(cutoff * x) * np.kaiser(len(x), 8.6)).astype(np.float32)
        _sinc_tables[cutoff] = t
    return t


def repitch(audio, semitones, block=32768):
    """Classic (tape-style) repitch: speed and pitch move together. Band-limited
    windowed-sinc resampling, so no aliasing going up and no dulling going down."""
    if semitones == 0:
        return audio
    rate = 2 ** (semitones / 12)
    n_out = max(1, int(len(audio) / rate))
    table = _sinc_table(round(min(1.0, 1.0 / rate) * 0.97, 4))
    src = np.pad(np.asarray(audio, np.float32), ((SINC_HALF + 1, SINC_HALF + 2), (0, 0)))
    k = np.arange(-SINC_HALF + 1, SINC_HALF + 1)
    out = np.empty((n_out, audio.shape[1]), np.float32)
    for s in range(0, n_out, block):
        pos = np.arange(s, min(n_out, s + block)) * rate
        base = np.floor(pos).astype(np.int64)
        idx = np.rint((k[None, :] - (pos - base)[:, None]) * SINC_OVER).astype(np.int64) + SINC_HALF * SINC_OVER
        out[s:s + len(pos)] = np.einsum("nk,nkc->nc", table[idx], src[base[:, None] + k[None, :] + SINC_HALF + 1])
        time.sleep(0)                                  # let the audio thread in
    return out


WAVE_X, WAVE_W, WAVE_MID, WAVE_HALF = 220, 720, 84, 62


def peaks(audio, width=WAVE_W):
    mono = np.abs(audio).max(axis=1)
    p = np.array([c.max() if len(c) else 0.0 for c in np.array_split(mono, width)])
    return p / p.max() if p.max() > 0 else p


wave_peaks = {}


pad_region = {}                                    # pid -> (start, end) in the file, for chops


def set_pad_sample(note, name, audio, path, region=None):
    """Put a sample on a kit pad (pitch back to the sample's original).
    region = (start, end): the pad plays only that part (a chop)."""
    pad_samples[note] = (name, audio)
    pad_files[note] = path
    if region:
        pad_region[note] = region
    else:
        pad_region.pop(note, None)
    tune[note] = 0
    pitched[note] = audio
    pitched_shift[note] = ("classic", 0, 1.0) if padp[note].get("style", "classic") == "classic" else None
    wave_peaks[note] = peaks(audio)
    refresh_kit_notes()


def load_kit_folder():
    """First run: the WAVs in kit/ fill the pads in file-name order."""
    for i, path in enumerate(sorted(glob.glob(os.path.join(APP_DIR, "kit", "*.wav")))[:16]):
        set_pad_sample(KIT_ALL[i], os.path.basename(path), load_sample(path), path)


pitched_shift = {}                                 # pad -> semitones its `pitched` audio was made with


def pad_shift(note):
    """Total repitch of a pad in semitones: PITCH + FINE (cents)."""
    return tune[note] + padp[note].get("fine", 0) / 100


def modern_speed(note):
    """MODERN pads with STRETCH on play at the project tempo (follows BPM changes)."""
    f = padp[note].get("stretch")
    bpm = pad_bpm(note)
    if padp[note].get("style") != "modern" or not f or not bpm:
        return 1.0
    return clock["bpm"] / (bpm * f)


def stretch_audio(audio, speed, semis):
    """Time-stretch (speed: 2 = twice as fast) + pitch shift, independently. Must be
    one process() call; it releases the GIL, so a background thread keeps audio smooth."""
    import python_stretch as ps
    st = ps.Signalsmith.Stretch()
    st.preset(2, SR)
    st.timeFactor = speed
    if semis:
        st.setTransposeSemitones(semis)
    x = np.ascontiguousarray(audio.T, dtype=np.float32)
    out = st.process(x)                                # one pass: chunking smears it
    return np.ascontiguousarray(out.T, dtype=np.float32)


def render_key(note):
    return (padp[note].get("style", "classic"), round(pad_shift(note), 3), round(modern_speed(note), 5))


retune_jobs = {}                                   # pad -> key being rendered in the background


def retune(note):
    if note not in pad_samples:
        return
    key = render_key(note)
    if pitched_shift.get(note) == key:
        return
    style, shift, speed = key
    src = pad_samples[note][1]
    if shift == 0 and (style != "modern" or speed == 1.0) or (style != "modern" and len(src) < SR):
        retune_jobs.pop(note, None)                    # a slower render still running is now stale
        pitched[note] = repitch(src, shift)
        wave_peaks[note] = peaks(pitched[note])
        pitched_shift[note] = key
        return
    if retune_jobs.get(note) == key:
        return
    retune_jobs[note] = key
    if note not in pitched:
        pitched[note] = src
        wave_peaks[note] = peaks(src)

    def work():
        try:
            audio = stretch_audio(src, speed, shift) if style == "modern" else repitch(src, shift)
        except ImportError:
            toast("MODERN needs: python -m pip install python-stretch")
            retune_jobs.pop(note, None)
            return
        if retune_jobs.get(note) == key and pad_samples.get(note, (None, None))[1] is src:
            pitched[note] = audio
            wave_peaks[note] = peaks(audio)
            pitched_shift[note] = key
            retune_jobs.pop(note, None)
    threading.Thread(target=work, daemon=True).start()


def set_tune(note, semitones):
    semitones = max(-TUNE_RANGE, min(TUNE_RANGE, semitones))
    tune[note] = semitones
    retune(note)

# ---------------------------------------------------------------------------
# Per-pad settings. swing / humanize = None means "follow the pattern".
# ---------------------------------------------------------------------------
# mode: "one-shot" plays to the end, "gate" stops when you let go of the pad.
# choke: 0 = off, 1-8 = pads in the same group cut each other off.
padp = {n: {"volume": 100, "pan": 0, "swing": None, "humanize": None, "velocity": 116,
            "mode": "one-shot", "choke": 0, "mono_sum": False, "colour": 0,
            "root": 60, "fine": 0, "stretch": None, "style": "classic",
            "loop_bpm": None, "bus": None, "inst": None, "ip": None, "fx": None, "vary": 0} for n in KIT_ALL}
# vary: 0-100 cents. Each hit plays one of a few slightly detuned copies (Settings · PAD).
# style: "classic" = speed and pitch move together (SP-404 / MPC60 repitch);
#        "modern"  = time-stretch: tempo and pitch are independent (Signalsmith Stretch).
# root: the note the sample plays naturally (keyboard keys then play real pitches).
# fine: extra tuning in cents (-50..+50), e.g. from FIT (match a loop to the BPM).
# Pad colours: one knob, 0-127, like the Push / SP-404 MK2: blue at 0 sweeping
# through cyan, green, yellow, orange, red, pink and purple, then white at 127.
# The colour wheel is built from these anchor points: (value, name, LED RGB,
# screen RGB). LEDs show colour differently from the screen, so each has its own.
COLOUR_WHEEL = [
    (0, "BLUE", (0, 40, 255), (52, 120, 230)),
    (16, "CYAN", (0, 170, 255), (60, 190, 230)),
    (40, "GREEN", (0, 255, 20), (70, 200, 90)),
    (64, "YELLOW", (255, 150, 0), (240, 205, 40)),
    (78, "ORANGE", (255, 45, 0), (240, 125, 30)),
    (86, "RED", (255, 0, 0), (232, 64, 58)),
    (95, "PINK", (255, 0, 70), (240, 90, 150)),
    (104, "PURPLE", (110, 0, 255), (150, 90, 230)),
    (126, "VIOLET", (40, 0, 255), (110, 85, 235)),
    (127, "WHITE", (200, 200, 200), (225, 225, 230)),
]
COLOUR_PRESETS = [
    ("BLUE", (0, 40, 255)), ("CYAN", (0, 170, 255)), ("GREEN", (0, 255, 20)),
    ("YELLOW", (255, 150, 0)), ("ORANGE", (255, 45, 0)), ("RED", (255, 0, 0)),
    ("PINK", (255, 0, 70)), ("PURPLE", (110, 0, 255)), ("BROWN", (110, 18, 0)),
    ("WHITE", (200, 200, 200)),
]
PRESET_RGB = dict(COLOUR_PRESETS)
DIM_LEVEL = 0.45                                   # unselected pads: 45% of the colour
PAD_LED_FIRST = 30                                 # palette entries 30-61: bright + dim per kit pad
pad_flash = {}                                     # pad -> time it was last triggered (as heard)
last_voice = {}                                    # pad -> its most recent voice (for playheads)
palette_sent = {}                                  # palette entry -> rgb already sent to the Push


def wheel_rgb(v, screen=False):
    """Colour knob value 0-127 -> RGB (for the LED, or the screen), blending between anchors."""
    v = clamp(int(v), 0, 127)
    k = 3 if screen else 2
    for a, b in zip(COLOUR_WHEEL, COLOUR_WHEEL[1:]):
        (v0, c0), (v1, c1) = (a[0], a[k]), (b[0], b[k])
        if v0 <= v <= v1:
            t = 0 if v1 == v0 else (v - v0) / (v1 - v0)
            return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))
    return COLOUR_WHEEL[-1][k]


def wheel_name(v):
    """Name of the nearest anchor, for the screen."""
    return min(COLOUR_WHEEL, key=lambda a: abs(a[0] - v))[1]


def colour_value(pid):
    """The pad's colour as a knob value 0-127 (older projects stored names or RGB)."""
    c = padp[pid]["colour"]
    if isinstance(c, int):
        return c
    rgb = PRESET_RGB.get(c, PRESET_RGB["BLUE"]) if isinstance(c, str) else tuple(c)
    value = min(range(128), key=lambda v: sum((a - b) ** 2 for a, b in zip(wheel_rgb(v), rgb)))
    padp[pid]["colour"] = value                       # convert once
    return value


def pad_rgb(pid):
    return wheel_rgb(colour_value(pid))


def pad_screen_colour(pid):
    return wheel_rgb(colour_value(pid), screen=True)


def pad_led(pid, bright):
    """Palette entry for a kit pad (each of the 16 kit pads has its own pair)."""
    return PAD_LED_FIRST + 2 * (pid % 16) + (0 if bright else 1)


# ---------------------------------------------------------------------------
# Mute groups (pads numbered 1-16 from bottom-left)
# ---------------------------------------------------------------------------
CHOKE_GROUPS = 8
FADE = 96                                          # 2 ms
padp[2]["choke"] = padp[3]["choke"] = 1            # default: A3 closed hat + A4 open hat

# ---------------------------------------------------------------------------
# Voices. pos < 0 means "starts that many samples into the current block",
# which lets the sequencer place hits on the exact sample.
# ---------------------------------------------------------------------------
class Voice:
    __slots__ = ("audio", "pos", "gain", "note", "fade_left", "lr", "stop")

    def __init__(self, audio, gain, note, offset=0, pan=0):
        self.audio, self.pos, self.gain, self.note, self.fade_left = audio, -offset, gain, note, None
        self.stop = None                                  # play this many samples, then fade (note length)
        p = pan / 50                                      # -1 (left) .. +1 (right)
        self.lr = np.array([min(1.0, 1 - p), min(1.0, 1 + p)], dtype=np.float32) * gain


def pad_lr(note, velocity_gain):
    """Left/right gain of a pad right now: volume and pan apply live, even to a
    sound that's already playing."""
    pp = padp[note]
    p = pp["pan"] / 50
    return np.array([min(1.0, 1 - p), min(1.0, 1 + p)], dtype=np.float32) * (velocity_gain * pp["volume"] / 100)


voices = []


# ---------------------------------------------------------------------------
# Instruments on pads (Add Device). A synth pad has no sample: its notes are
# SynthVoices that render their own audio each block.
# ---------------------------------------------------------------------------
class InstNote:
    """What note_audio() returns for an instrument pad: just the note to play."""
    __slots__ = ("n",)

    def __init__(self, n):
        self.n = n


class SynthVoice:
    __slots__ = ("synth", "pos", "gain", "note", "fade_left", "stop", "events", "released", "audio", "midi",
                 "want")

    def __init__(self, synth, pid, n, gain, offset):
        self.synth, self.note, self.gain = synth, pid, gain
        self.pos, self.fade_left, self.stop, self.released = -offset, None, None, False
        self.events = [(-offset, n, gain)]              # (voice time, note, gain): notes to start
        self.audio = INST_PLACEHOLDER                   # so code that peeks at .audio stays happy
        self.midi = self.want = n                       # want: the latest note asked for

    def render(self, n, rel_at):
        """n samples from now; rel_at = where in them the key is let go (None = held)."""
        out = np.zeros((n, 2) if getattr(self.synth, "stereo", False) else n, np.float32)
        cuts = sorted({0, n} | {max(0, min(n, t - self.pos)) for t, _, _ in self.events if t - self.pos < n}
                      | ({rel_at} if rel_at is not None else set()))
        for a, b in zip(cuts, cuts[1:]):
            for ev in [e for e in self.events if e[0] - self.pos <= a]:
                self.events.remove(ev)
                self.synth.note_on(ev[1], ev[2])
                self.midi = ev[1]
            if rel_at is not None and a >= rel_at and not self.released:
                self.synth.note_off()
                self.released = True
            out[a:b] = self.synth.render(b - a)
        return out

    def retrigger(self, n, gain, offset, stop):
        """Mono: a new note on the same synth (glides from the one playing)."""
        self.events.append((self.pos + offset, n, gain))
        self.want = n
        self.released, self.fade_left = False, None
        self.stop = None if stop is None else self.pos + offset + stop


INST_PLACEHOLDER = np.zeros((SR // 2, 2), np.float32)
INST_TYPES = ["BASS", "KEYS"]
KEYS_POLY = 4                                           # KEYS: notes at once per pad
MAX_INSTRUMENTS = 3
BASS_DEFAULTS = {"wave": "SAW", "detune": 7, "sub": 40, "cutoff": 400, "emphasis": 30, "contour": 55,
                 "decay": 30, "glide": 0, "drive": 20, "release": 10}
INST_KNOBS = {"BASS": [("WAVE", "wave", "choice", ["SAW", "TRI", "SQUARE", "PULSE"]),
                       ("DETUNE", "detune", 0, 50), ("SUB", "sub", 0, 100), ("CUTOFF", "cutoff", "log", (30, 8000)),
                       ("EMPHASIS", "emphasis", 0, 100), ("CONTOUR", "contour", 0, 100),
                       ("DECAY", "decay", 0, 100), ("GLIDE", "glide", 0, 100)],
              "KEYS": [("PROGRAM", "program", "choice", native_fx.KEYS_PROGRAMS), ("OCTAVE", "octave", -2, 2),
                       ("ATTACK", "attack", 0, 100), ("RELEASE", "release", 0, 100),
                       ("TONE", "tone", 0, 100), ("VELOCITY", "velo", 0, 100),
                       ("CHORD", "chord", "choice", ["OFF", "5TH", "TRIAD", "SUS2", "SUS4", "7TH", "ADD9"]),
                       ("CHORUS", "chorus", "choice", ["OFF", "I", "II", "I+II"])]}
KEYS_DEFAULTS = {"program": "PIANO", "octave": 0, "attack": 0, "release": 30, "tone": 85, "velo": 50,
                 "chord": "OFF", "chorus": "OFF"}
CHORD_STEPS = {"TRIAD": (0, 2, 4), "7TH": (0, 2, 4, 6),
               "ADD9": (0, 2, 4, 8)}                   # scale steps above the note you play
choruses = {}                                           # pad -> JunoChorus (audio thread)
strip_in = {}                                           # pad -> [this block's sound, where it goes]


def pad_strip_on(note):
    """Does this pad's sound go through its own strip (KEYS CHORUS or pad FX)?"""
    pp = padp[note]
    if pp.get("inst") == "KEYS" and (pp.get("ip") or {}).get("chorus", "OFF") != "OFF":
        return True
    return pad_engine is not None and bool(pad_engine.chains.get(note))


def chord_offsets(n, kind):
    """Semitones to add to note n for a CHORD built from the project's KEY / SCALE:
    each key plays the chord of its own scale step (C in C major = C major, D = D minor).
    A note outside the scale borrows the shape of the scale step just below it."""
    if kind in (None, "OFF"):
        return (0,)
    if kind == "5TH":
        return (0, 7, 12)                               # power chord: root, fifth, octave
    if kind in ("SUS2", "SUS4"):                        # sus chords keep a clean fifth
        return (0, 2, 7) if kind == "SUS2" else (0, 5, 7)
    sc = SCALES.get(setup["key_scale"], SCALES["MAJOR"])
    if len(sc) == 12:                                   # chromatic: no scale, use major
        sc = SCALES["MAJOR"]
    rel = (n - setup["key_root"]) % 12
    deg = max(i for i, v in enumerate(sc) if v <= rel)
    base = sc[deg]
    tone = lambda k: sc[(deg + k) % len(sc)] + 12 * ((deg + k) // len(sc)) - base
    return tuple(tone(k) for k in CHORD_STEPS[kind])
INST_DEFAULTS = {"BASS": BASS_DEFAULTS, "KEYS": KEYS_DEFAULTS}
INST_LABELS = {"BASS": "MODEL D STYLE BASS", "KEYS": "ROMPLER + FM KEYS"}


def keys_preload(pid):
    """Loads a KEYS pad's samples (main thread, never the audio thread)."""
    if padp[pid].get("inst") == "KEYS":
        prog = padp[pid]["ip"].get("program")
        if prog in native_fx.SAMPLED and prog not in native_fx._banks:
            native_fx.keys_bank(prog)


def create_instrument(pid, kind, undo=True):
    """Put a synth on a pad: no sample, a placeholder so the rest of MPushC sees a pad."""
    if undo:
        checkpoint(f"add {kind.lower()} synth")
    n = 1 + sum(1 for q in pad_samples if q != pid and is_inst(q) and padp[q]["inst"] == kind)
    set_pad_sample(pid, f"{kind} {n}", INST_PLACEHOLDER, "")
    padp[pid].update(inst=kind, ip=dict(INST_DEFAULTS[kind]), mode="gate", root=60, fine=0, stretch=None,
                     style="classic", choke=0)
    detected[pid] = {"busy": False, "src": id(INST_PLACEHOLDER), "root": 60.0, "bpm": None, "key": None}
    keys_preload(pid)
    mark_dirty()


def switch_instrument(pid):
    """Shift + Add Device on a synth pad: BASS <-> KEYS (steps and pad settings stay)."""
    kind = INST_TYPES[(INST_TYPES.index(padp[pid]["inst"]) + 1) % len(INST_TYPES)]
    checkpoint(f"synth to {kind.lower()}")
    for v in voices:
        if isinstance(v, SynthVoice) and v.note == pid:
            v.fade_left = 0
    n = 1 + sum(1 for q in pad_samples if q != pid and is_inst(q) and padp[q]["inst"] == kind)
    set_pad_sample(pid, f"{kind} {n}", INST_PLACEHOLDER, "")
    padp[pid].update(inst=kind, ip=dict(INST_DEFAULTS[kind]))
    keys_preload(pid)
    ui["page"], ui["open"], ui["settings"] = "INST", True, None
    toast(f"pad {pad_name(pid)} is now a {kind} synth")
    mark_dirty()


def follow_selection():
    """Main loop: picking a synth pad shows its synth page instead of PAD / SAMPLE;
    picking a sample pad from a synth page goes back to PAD. Only when the selection
    changes, so the PAD page button still works for a synth's volume, pan and BUS."""
    sp = ui["selected"]
    if sp == ui.get("last_sel"):
        return
    ui["last_sel"] = sp
    if sp is None or not ui["open"] or ui.get("settings"):
        return
    if is_inst(sp) and ui["page"] in ("PAD", "SAMPLE"):
        ui["before_inst"], ui["page"] = ui["page"], "INST"
    elif not is_inst(sp) and ui["page"] == "INST":
        ui["page"] = ui.get("before_inst", "PAD")             # back to PAD or SAMPLE, as before


def add_device_on(pid):
    """Hold Add Device + press a pad."""
    ui["add_used"] = True
    if is_inst(pid):
        ui["selected"] = pid
        ui["page"], ui["open"], ui["settings"] = "INST", True, None
        return
    if pid in pad_samples:
        toast(f"pad {pad_name(pid)} has a sample: pick an empty pad (or Delete it first)")
        return
    if sum(1 for q in pad_samples if is_inst(q)) >= MAX_INSTRUMENTS:
        toast(f"{MAX_INSTRUMENTS} synths per project: delete one first (Delete + pad)")
        return
    create_instrument(pid, "BASS")
    ui["selected"] = pid
    ui["page"], ui["open"], ui["settings"] = "INST", True, None
    toast(f"BASS synth on pad {pad_name(pid)}  ·  Shift + Add Device: KEYS")


def add_device():
    """Add Device: the selected synth pad opens its page; otherwise a new BASS synth goes
    on the first empty pad of this bank."""
    sp = ui["selected"]
    if sp is not None and is_inst(sp):
        ui["page"], ui["open"], ui["settings"] = "INST", True, None
        return
    if sum(1 for q in pad_samples if is_inst(q)) >= MAX_INSTRUMENTS:
        toast(f"{MAX_INSTRUMENTS} synths per project: delete one first (Delete + pad)")
        return
    empty = [q for q in range(bank["cur"] * 16, bank["cur"] * 16 + 16) if q not in pad_samples] or \
            [q for q in range(NUM_BANKS * 16) if q not in pad_samples]
    if not empty:
        toast("no empty pad left")
        return
    pid = empty[0]
    create_instrument(pid, "BASS")
    bank["cur"], ui["selected"] = pid // 16, pid
    ui["page"], ui["open"], ui["settings"] = "INST", True, None
    toast(f"BASS synth on pad {pad_name(pid)}  ·  Shift + Add Device: KEYS")


def inst_knobs():
    sp = ui["selected"]
    if sp is None or not is_inst(sp):
        return [None] * 8
    key = ("inst", sp, padp[sp]["inst"])
    if _inst_knobs.get("key") != key:
        knobs = []
        for label, k, kind, rng in INST_KNOBS[padp[sp]["inst"]]:
            knobs.append(inst_param(sp, label, k, kind, rng))
        _inst_knobs.update(key=key, knobs=(knobs + [None] * 8)[:8])
    return _inst_knobs["knobs"]


_inst_knobs = {}


def inst_param(pid, label, k, kind, rng):
    def get():
        ip = padp[pid].get("ip") or {}
        return ip.get(k)

    def put(d):
        ip = padp[pid].get("ip")
        if not ip:
            return
        v = ip[k]
        if kind == "choice":
            v = step_choice(rng, v, d)
        elif kind == "log":
            nv = int(round(v * 1.04 ** d))
            v = int(clamp(nv if nv != v else v + d, rng[0], rng[1]))
        else:
            v = int(clamp(v + d, kind, rng))
        ip[k] = v
        if k == "program":
            keys_preload(pid)

    def show(v):
        if v is None:
            return "-"
        if k == "cutoff":
            return f"{v / 1000:.1f} kHz" if v >= 1000 else f"{v} Hz"
        if k == "detune":
            return f"{v} ct"
        if k == "octave":
            return f"{v:+d}" if v else "0"
        return str(v) if kind == "choice" else f"{v}%"
    return Param(label, get, put, per=4 if kind == "choice" or k == "octave" else 1,
                 show=show).auto_as(lambda: f"ip:{pid}:{k}" if k != "program" else None)   # samples load on change


def is_inst(pid):
    return padp[pid].get("inst") is not None


def make_synth(pid):
    if padp[pid]["inst"] == "KEYS":
        return native_fx.KeysNote(padp[pid]["ip"])
    return native_fx.MonoBass(padp[pid]["ip"])


def trigger(note, gain, offset=0, audio=None, stop=None):
    if is_inst(note):
        # keyboard notes are real pitches; a pad tap or a drum step plays the key's root (C2-B2)
        n = audio.n if isinstance(audio, InstNote) else 48 + setup["key_root"]
        n += tune[note]                                     # PAD page PITCH transposes the synth
        keys = padp[note]["inst"] == "KEYS"
        if keys:
            n += 12 * padp[note]["ip"].get("octave", 0) + (0 if isinstance(audio, InstNote) else 12)
            mine = [v for v in voices if isinstance(v, SynthVoice) and v.note == note
                    and not v.synth.done and not v.synth.fast]
            if len(mine) >= KEYS_POLY:                  # polyphonic: the oldest note makes room
                mine[0].synth.kill()
                mine[0].released = True
        for v in ([] if keys else voices):              # mono: the synth that's sounding takes it
            if isinstance(v, SynthVoice) and v.note == note and not v.synth.done:
                v.retrigger(n, gain, offset, stop)
                last_voice[note] = v
                pad_flash[note] = time.monotonic() + offset / SR + audio_state["latency"]
                return v
        v = SynthVoice(make_synth(note), note, n, gain, offset)
        if keys:
            v.synth.chord = chord_offsets(n, padp[note]["ip"].get("chord"))
        v.stop = None if stop is None else stop
        pad_flash[note] = time.monotonic() + offset / SR + audio_state["latency"]
        last_voice[note] = v
        voices.append(v)
        return v
    return trigger_sample(note, gain, offset, audio)


def inst_render(pid, n, length):
    """A synth note as audio (for resample / export): held `length` samples, then its
    release tail. No glide between notes here."""
    key = ("inst", pid, n, length, setup["key_root"], setup["key_scale"], tuple(sorted(padp[pid]["ip"].items())))
    if key in note_cache:
        return note_cache[key]
    syn = make_synth(pid)
    keys = padp[pid]["inst"] == "KEYS"
    if keys:
        n += 12 * padp[pid]["ip"].get("octave", 0)
        syn.chord = chord_offsets(n, padp[pid]["ip"].get("chord"))
    syn.note_on(n, 1.0)
    parts = [syn.render(max(1, length))]
    syn.note_off()
    for _ in range(200):                                # up to ~4 s of tail
        if syn.done:
            break
        parts.append(syn.render(1024))
    mono = np.concatenate(parts)
    audio = (mono if mono.ndim == 2 else np.stack([mono, mono], axis=1)).astype(np.float32)
    if keys and padp[pid]["ip"].get("chorus", "OFF") != "OFF":
        audio = native_fx.JunoChorus().process(np.ascontiguousarray(audio.T), padp[pid]["ip"]["chorus"]).T
    if len(note_cache) > 256:
        note_cache.clear()
    note_cache[key] = audio
    return audio


VARY_SPREAD = (-1.0, -0.5, 0.0, 0.5, 1.0)          # the copies, as shares of VARY (0 = as tuned)
vary_cache = {}                                    # pad -> ((source id, cents), [audio per copy])
vary_jobs = {}                                     # pad -> key being made in the background


def vary_pick(note):
    """A pad's audio for one hit: a random detuned copy when VARY is on and ready."""
    c = vary_cache.get(note)
    base = pitched[note]
    if c is None or c[0][0] != id(base) or c[0][1] != padp[note].get("vary", 0):
        return base
    return random.choice(c[1])


def vary_tick():
    """Main loop: make the detuned copies (in a thread) for pads with VARY on."""
    for pid in list(pad_samples):
        cents = padp[pid].get("vary", 0)
        if not cents or is_inst(pid) or pid not in pitched:
            vary_cache.pop(pid, None)
            continue
        base = pitched[pid]
        key = (id(base), cents)
        if (vary_cache.get(pid) or ((None,),))[0] == key or vary_jobs.get(pid) == key:
            continue
        vary_jobs[pid] = key

        def work(pid=pid, base=base, key=key, cents=cents):
            try:
                copies = [base if f == 0 else repitch(base, f * cents / 100) for f in VARY_SPREAD]
            except Exception:
                copies = None
            if copies and vary_jobs.get(pid) == key:
                vary_cache[pid] = (key, copies)
            if vary_jobs.get(pid) == key:
                vary_jobs.pop(pid, None)
        threading.Thread(target=work, daemon=True).start()


def trigger_sample(note, gain, offset=0, audio=None):
    g = padp[note]["choke"]
    if g:
        for v in voices:
            if v.note is not None and padp[v.note]["choke"] == g and v.fade_left is None:
                v.fade_left = FADE + offset
    pp = padp[note]
    v = Voice(vary_pick(note) if audio is None else audio, gain, note, offset)   # gain = velocity only
    pad_flash[note] = time.monotonic() + offset / SR + audio_state["latency"]
    last_voice[note] = v
    if len(voices) >= MAX_VOICES:
        voices.pop(0)
    voices.append(v)
    return v

# ---------------------------------------------------------------------------
# Clock, metronome, sequencer. Counted in 16th-note ticks inside the audio
# callback, so every step lands on the exact sample.
# ---------------------------------------------------------------------------
STEPS = 16                    # steps per bar = one page of step pads
MAX_BARS = 8
MAX_STEPS = STEPS * MAX_BARS  # 128
# A step is None (off) or (velocity 1-127, offset). The offset is how far off
# the grid the hit sits, as a fraction of a step: -0.5 (early) .. +0.5 (late).
DEFAULT_VEL = 116                      # = the level programmed steps always had
MAX_NUDGE = 0.5


def step_event(ev):
    """A step's (first) hit as (velocity, nudge). Old projects stored True: a
    default, on-grid step."""
    if ev is True:
        return (DEFAULT_VEL, 0.0)
    return ev[0] if isinstance(ev[0], tuple) else ev


def step_hits(ev):
    """Every hit in a step. A step usually holds one (velocity, nudge); finer grids
    (1/32, triplets) can put several in one 16th: then it's a tuple of them."""
    if not ev:
        return []
    if ev is True:
        return [(DEFAULT_VEL, 0.0)]
    return list(ev) if isinstance(ev[0], tuple) else [ev]


def hit_span(steps):
    """Width of one hit on screen, in steps: 1, or the closest spacing on finer grids."""
    pos = sorted(i + n for i, ev in enumerate(steps) for _, n in step_hits(ev))
    gaps = [b - a for a, b in zip(pos, pos[1:]) if b - a > 0.05]
    return min([1.0] + gaps)


def make_step(hits):
    hits = sorted(hits, key=lambda h: h[1])
    return None if not hits else (hits[0] if len(hits) == 1 else tuple(hits))
PLAY_BUTTON, NOTE_BUTTON, SESSION_BUTTON = 85, 50, 51
ACCENT_BUTTON, SCALE_BUTTON = 57, 58
PAGE_LEFT, PAGE_RIGHT, DOUBLE_BUTTON = 62, 63, 117
TICKS_PER_BPM = 2
FOLLOW_BUTTON = 31                   # "Layout" button: follow playhead on/off
ZOOM_IN, ZOOM_OUT = 46, 47           # up / down arrow buttons: screen zoom
BROWSE_BUTTON = 111                  # "Browse": open / close the sample browser
ARROW_LEFT, ARROW_RIGHT = 44, 45     # browser: back / open
BROWSE_TICKS = 6                     # knob 1: knob ticks per item (higher = slower)
BROWSE_FAST_TICKS = 4                # knob 2: knob ticks per 10 items
LIBRARY = [
    ("SSD Samples", "/Volumes/SanDiskSSD/1 Music/Samples"),
    ("MPushC kit", os.path.abspath("kit")),
]
SEARCH_ROOTS = [path for _, path in LIBRARY]   # "search everything" = the sample library only
LIBRARY += [                                   # the rest of the computer, to browse anywhere
    ("Home", os.path.expanduser("~")),
    ("Drives", "/Volumes"),
    ("Computer", "/"),
]
ZOOM_LEVELS = [1, 2, 4, 8]           # bars shown on screen at once
seq = {"bars": 1, "page": 0,         # page = the bar shown on the step pads
       "follow": False, "zoom": 1}
TEMPO_KNOB = 14               # far-left encoder: BPM
VOLUME_KNOBS = (79,)          # Master encoder (far right)
SWING_KNOB = 15               # Swing encoder (second from left)
MUTE_BUTTON, SOLO_BUTTON = 60, 61
DELETE_BUTTON, UNDO_BUTTON, SHIFT_BUTTON = 118, 119, 49
DUPLICATE_BUTTON = 88
METRONOME_BUTTON, TAP_TEMPO_BUTTON = 9, 3
RECORD_BUTTON, REPEAT_BUTTON = 86, 56
QUANTIZE_BUTTON = 116
SETUP_BUTTON = 30                       # settings for what you're doing (PAD / PATTERN / GENERAL)
HELP_BUTTON = 59                        # User: help for the current screen
ADD_TRACK_BUTTON = 53                   # Projects
STOP_CLIP_BUTTON = 29                   # silence everything now (the sequencer keeps going)
FIXED_LENGTH_BUTTON = 90                # hold + knob 1: pattern length
HELP_BUTTONS = (59,)                    # (the Push sends User on BOTH its ports: see toggle_help)
# The AudioFuse already works in ~10 ms chunks (512 samples): asking for 128-sample
# blocks only meant 4 rushed callbacks per chunk, each fighting the screen for
# Python's lock. One 512 block per chunk = same latency, 4x fewer fights.
OUT_BLOCK = 512
ADD_DEVICE_BUTTON = 52                  # Add Device = synths on pads
MIX_BUTTON = 112                        # Mix = the bus mixer
DEVICE_BUTTON = 110                     # Device = the FX buses
MASTER_BUTTON = 28                      # Master = the master bus page (comp + limiter)
CLIP_BUTTON = 113                       # Clip = SONG mode (the arrangement)
CONVERT_BUTTON = 35                     # Convert = resample the pattern (or a pad) onto a new pad
SELECT_BUTTON = 48
OCTAVE_UP, OCTAVE_DOWN = 55, 54        # step through loaded pads (any bank) on the grid
NUDGE_PER_TICK = 0.01                  # knob 2 while holding a step: 1% of a step per click
# the 8 buttons right of the pads, bottom to top: 1/4, 1/4t, 1/8, 1/8t, 1/16, 1/16t, 1/32, 1/32t
RATE_BUTTONS = list(range(36, 44))
RATES = [("1/4", 4), ("1/4T", 8 / 3), ("1/8", 2), ("1/8T", 4 / 3),
         ("1/16", 1), ("1/16T", 2 / 3), ("1/32", 0.5), ("1/32T", 1 / 3)]     # length in 16th steps
LOWER_BUTTONS = list(range(20, 28))          # the 8 buttons under the screen
NUM_SLOTS, SLOTS_PER_PAGE = 16, 8
SWING_MIN, SWING_MAX = 50, 75 # MPC-style: 50% = straight, ~66% = triplet shuffle
HUMANIZE_MAX_MS = 15          # at 100%: hits land up to 15 ms late...
HUMANIZE_MAX_GAIN = 0.3       # ...and up to 30% softer
groove = {"swing": 50, "humanize": 0}
mutes, solos = set(), set()   # kit notes
VOLUME_STEP = 0.01            # 1% per knob tick
setup = {"click": 60, "metronome": True, "rec_quant": 0, "rec_grid": "1/16", "count_in": 1,
         "key_root": 0, "key_scale": "MINOR", "stem_q": "FAST",
         "smp_input": "1-2", "smp_gain": 0, "smp_thresh": -40, "smp_length": 0,
         "curve": "lin", "gain": 67, "threshold": 1, "buffer": 128}

# Pad response, modelled on the SP-404 MK2's TRIGGER page:
#   CURVE  lin = natural; exp = hard hits change more; log = soft hits change more; fix = always full
#   GAIN   0-100 sensitivity: higher = less force needed for full volume
#   THRESH 1-100 minimum hit strength; softer hits are ignored (stops ghost triggers)
CURVES = ["lin", "exp", "log", "fix"]


def pad_level(v):
    """Pad velocity (0-127) -> 0..1 before the curve, using GAIN."""
    full = 127 - setup["gain"] / 100 * 87          # gain 0: needs 127 · gain 100: full at 40
    return max(0.0, min(1.0, v / full))


def velo_gain(v):
    x = pad_level(v)
    c = setup["curve"]
    if c == "fix":
        return 1.0
    if c == "exp":
        return x ** 2
    if c == "log":
        return x ** 0.5
    return x
master = {"volume": 0.8}
# Master bus (Master button): SSL-style bus compressor, then a brick-wall limiter.
# Steps like the SSL's switches. RATIO OFF = compressor bypassed (the default).
MASTER_DEFAULTS = {"ratio": 0, "thresh": -10, "attack": 3, "release": 0.3, "makeup": 0, "mix": 100,
                   "drive": 0}
RATIO_CHOICES = [0, 2, 4, 10]
ATTACK_CHOICES = [0.1, 0.3, 1, 3, 10, 30]           # ms
RELEASE_CHOICES = [0.1, 0.3, 0.6, 1.2]              # s
CEILING_DB = -0.3
master_fx = dict(MASTER_DEFAULTS)
meters = {"gr": 0.0, "lim": 0.0, "out": [0.0, 0.0], "in": 0.0}  # dB of gain reduction, output peaks


class MasterBus:
    """Runs in the audio thread. Knobs change master_fx; apply() copies them over."""

    def __init__(self):
        import pedalboard as pb
        self.comp = pb.Compressor()
        self.lim = pb.BrickwallLimiter(ceiling_db=CEILING_DB, release_ms=80, lookahead_ms=1.0)
        self.applied = None
        self.apply()

    def apply(self):
        m = master_fx
        key = tuple(sorted(m.items()))
        if key == self.applied:
            return
        self.applied = key
        self.comp.threshold_db = float(m["thresh"])
        self.comp.ratio = float(m["ratio"] or 1)
        self.comp.attack_ms = float(m["attack"])
        self.comp.release_ms = float(m["release"] * 1000)

    def process(self, x):
        """x: (frames, 2) pads only. Returns the processed block."""
        self.apply()
        m = master_fx
        meters["in"] = max(float(np.abs(x).max()), meters["in"] * 0.7)
        y = np.ascontiguousarray(x.T)
        if m["ratio"]:
            c = self.comp(y, SR, reset=False)
            rin, rout = float(np.sqrt(np.mean(y * y))), float(np.sqrt(np.mean(c * c)))
            gr = 20 * math.log10(max(rout, 1e-9) / rin) if rin > 1e-4 else 0.0
            meters["gr"] = min(0.0, gr) if gr < meters["gr"] else 0.6 * meters["gr"] + 0.4 * min(0.0, gr)
            c = c * (10 ** (m["makeup"] / 20))
            w = m["mix"] / 100
            y = c if w >= 1 else c * w + y * (1 - w)
        else:
            meters["gr"] = 0.0
        if m["drive"]:
            y = y * (10 ** (m["drive"] / 20))
        pre = float(np.abs(y).max())
        n = y.shape[1]
        y = self.lim(np.ascontiguousarray(y, dtype=np.float32), SR, reset=False)
        if y.shape[1] < n:                          # first block: the 1 ms lookahead fills up
            y = np.pad(y, ((0, 0), (n - y.shape[1], 0)))
        post = float(np.abs(y).max())
        lim = 20 * math.log10(post / pre) if pre > 10 ** (CEILING_DB / 20) else 0.0
        meters["lim"] = min(lim, 0.7 * meters["lim"])
        h, w = master_hist, master_hist_w[0]                # for the scrolling view on screen
        h[0, w] = 20 * math.log10(max(float(np.abs(x).max()), 1e-6))
        h[1, w] = 20 * math.log10(max(post, 1e-6))
        h[2, w] = meters["gr"]
        h[3, w] = lim
        master_hist_w[0] = (w + 1) % h.shape[1]
        return y.T


master_bus = None                                   # made in main() (needs pedalboard)
master_hist = np.zeros((4, 400), np.float32)        # in dB, out dB, comp GR, limiter GR per block
master_hist_w = [0]


def render_master(audio, loop=True):
    """The master bus applied to a whole render (resample = as you hear it)."""
    try:
        import pedalboard as pb
    except ImportError:
        return audio
    m = master_fx
    y = np.ascontiguousarray(audio.T, dtype=np.float32)
    if m["ratio"]:
        comp = pb.Compressor(threshold_db=m["thresh"], ratio=m["ratio"], attack_ms=m["attack"],
                             release_ms=m["release"] * 1000)
        if loop:
            comp(y, SR, reset=False)                # one pass to settle, like a playing loop
        c = comp(y, SR, reset=False) * (10 ** (m["makeup"] / 20))
        w = m["mix"] / 100
        y = c * w + y * (1 - w)
    y = y * (10 ** (m["drive"] / 20))
    lim = pb.BrickwallLimiter(ceiling_db=CEILING_DB, release_ms=80, lookahead_ms=1.0)
    shift = int(SR * 0.001)                         # undo the 1 ms lookahead delay
    if loop:
        lim(y, SR, reset=False)
        y = np.roll(lim(y, SR, reset=False), -shift, axis=1)
    else:
        y = np.concatenate([y, np.zeros((2, shift), np.float32)], axis=1)
        y = lim(np.ascontiguousarray(y), SR, reset=False)[:, shift:]
    return np.ascontiguousarray(y.T, dtype=np.float32)
BPM_MIN, BPM_MAX = 40, 200


def make_click(freq, ms=35):
    t = np.arange(int(SR * ms / 1000)) / SR
    c = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 90)
    return np.column_stack([c, c]).astype(np.float32)


CLICK_ACCENT, CLICK = make_click(1600), make_click(1000)
clock = {"playing": False, "bpm": 90, "sample": 0, "next_tick": 0.0, "tick": -1}
pattern = {n: [None] * MAX_STEPS for n in KIT_ALL}
# Melodic notes (keyboard modes), per pad: (position in steps, MIDI note, velocity,
# length in steps). Any position, any length, chords allowed. Drum steps above
# stay as they are (they play the pad as tuned = C3).
melody = {}
NOTE_MIN_LEN = 0.25


def pattern_length():
    return seq["bars"] * STEPS


def current_step():
    """Playing step across the whole pattern (0..127), or -1 when stopped."""
    return clock["tick"] % pattern_length() if clock["playing"] and clock["tick"] >= 0 else -1


def set_bars(bars):
    seq["bars"] = max(1, min(MAX_BARS, bars))
    seq["page"] = min(seq["page"], seq["bars"] - 1)


def double_pattern():
    """Copy the pattern onto the end of itself: 1 -> 2 -> 4 -> 8 bars."""
    n = pattern_length()
    if seq["bars"] * 2 > MAX_BARS:
        return
    for steps in pattern.values():
        steps[n:2 * n] = steps[:n]
    for pid, evs in melody.items():
        melody[pid] = evs + [(e[0] + n, e[1], e[2], e[3]) for e in evs]
    set_bars(seq["bars"] * 2)


def audible(note):
    """Solo wins: if any pad is soloed, only soloed pads play. Otherwise mutes apply."""
    if solos:
        return note in solos
    return note not in mutes


# ---------------------------------------------------------------------------
# Pattern slots: 16 patterns (2 pages of 8 on the buttons under the screen).
# The pattern you're on lives in `pattern` / `seq["bars"]` / `groove`;
# switching copies it into its slot and the new one out of its slot.
# While playing, a switch waits for the end of the current loop.
# ---------------------------------------------------------------------------
slots = [None] * NUM_SLOTS
pat = {"cur": 0, "queued": None, "page": 0, "row": "patterns"}   # row: what the lower buttons do


def capture():
    return {"bars": seq["bars"], "swing": groove["swing"], "humanize": groove["humanize"],
            "steps": {n: list(p) for n, p in pattern.items() if any(p)},
            "notes": {n: list(e) for n, e in melody.items() if e},
            "auto": copy_lanes(auto_lanes)}


def copy_slot(sl):
    return None if sl is None else {**sl, "steps": {n: list(p) for n, p in sl["steps"].items()},
                                    "notes": {n: list(e) for n, e in sl.get("notes", {}).items()},
                                    "auto": copy_lanes(sl.get("auto"))}


def apply_slot(sl):
    for n in pattern:
        pattern[n][:] = sl["steps"].get(n, [None] * MAX_STEPS)
    set_bars(sl["bars"])
    groove.update(swing=sl["swing"], humanize=sl["humanize"])
    melody.clear()
    melody.update({n: list(e) for n, e in sl.get("notes", {}).items()})
    auto_lanes.clear()
    auto_lanes.update(copy_lanes(sl.get("auto")))


def store_current():
    slots[pat["cur"]] = capture()


def switch_pattern(i):
    """Go to slot i now (an empty slot becomes a new, empty pattern)."""
    store_current()
    if slots[i] is None:
        slots[i] = {"bars": seq["bars"], "swing": groove["swing"], "humanize": groove["humanize"], "steps": {}}
    apply_slot(slots[i])
    pat["cur"], pat["queued"] = i, None
    seq["page"] = 0


def run_clock(frames):
    """Start this block's clicks and sequenced hits as offset voices."""
    if not clock["playing"]:
        return
    spt = SR * 60.0 / clock["bpm"] / 4             # samples per 16th
    start, end = clock["sample"], clock["sample"] + frames
    while clock["next_tick"] < end:
        offset = int(clock["next_tick"]) - start
        if clock["tick"] < -1:                       # count-in: clicks only, then bar 1
            clock["tick"] += 1
            if clock["tick"] % 4 == 0:
                voices.append(Voice(CLICK_ACCENT if clock["tick"] % STEPS == 0 else CLICK,
                                    max(setup["click"], 40) / 100, None, offset))
            clock["next_tick"] += spt
            continue
        if song["pos"] is not None and clock["tick"] >= 0 and (clock["tick"] + 1) % pattern_length() == 0 \
                and pat["queued"] is None:
            cur = pat["cur"]
            if not song_advance():                   # the song's over (END = STOP)
                clock["playing"] = False
                break
            if pat["cur"] != cur:
                clock["tick"] = -1
        if pat["queued"] is not None and (clock["tick"] + 1) % pattern_length() == 0:
            switch_pattern(pat["queued"])           # loop ended: change pattern
            clock["tick"] = -1
        clock["tick"] += 1
        step = clock["tick"] % pattern_length()
        if step % 4 == 0:
            if setup["click"] and setup["metronome"]:
                voices.append(Voice(CLICK_ACCENT if step % STEPS == 0 else CLICK,
                                    setup["click"] / 100, None, offset))
        nxt = (step + 1) % pattern_length()
        for note, steps in pattern.items():
            if note not in pad_samples or not audible(note):
                continue
            # this step's on-time / late hits, plus the NEXT step's early hits
            for st, base in ((step, 0.0), (nxt, 1.0)):
                hits_ = step_hits(steps[st])
                if hits_ and not chance_roll(note, st):
                    continue                                 # CHANCE: this step sits out this time
                for vel, nudge in hits_:
                    if (base == 0.0 and nudge < 0) or (base == 1.0 and nudge >= 0):
                        continue
                    if (note, st, nudge) in rec["fresh"]:
                        rec["fresh"].discard((note, st, nudge))   # heard live already; plays from next loop
                        continue
                    sw = padp[note]["swing"] if padp[note]["swing"] is not None else groove["swing"]
                    hu = padp[note]["humanize"] if padp[note]["humanize"] is not None else groove["humanize"]
                    # swing: every second 16th lands late (50% = straight)
                    swing_delay = (2 * sw / 100 - 1) * spt if st % 2 == 1 else 0.0
                    h = hu / 100
                    late = random.uniform(0, h * HUMANIZE_MAX_MS / 1000 * SR) if h else 0.0
                    gain = vel / 127 * (1 - random.uniform(0, h * HUMANIZE_MAX_GAIN))
                    trigger(note, gain, offset + max(0, int((base + nudge) * spt + swing_delay + late)),
                            stop=int(spt) if is_inst(note) else None)
        for pid, evs in melody.items():                 # keyboard notes starting in this step
            if pid not in pad_samples or not audible(pid):
                continue
            for ev in evs:
                pos, n, vel, length = ev
                if int(pos) != step:
                    continue
                if (pid, ev) in fresh_notes:
                    fresh_notes.discard((pid, ev))      # just recorded: you heard it live
                    continue
                q = int(round(pos)) % pattern_length()
                sw = padp[pid]["swing"] if padp[pid]["swing"] is not None else groove["swing"]
                hu = (padp[pid]["humanize"] if padp[pid]["humanize"] is not None else groove["humanize"]) / 100
                swing_delay = (2 * sw / 100 - 1) * spt if q % 2 == 1 else 0.0
                late = random.uniform(0, hu * HUMANIZE_MAX_MS / 1000 * SR) if hu else 0.0
                gain = vel / 127 * (1 - random.uniform(0, hu * HUMANIZE_MAX_GAIN))
                inst = is_inst(pid)
                v = trigger(pid, gain, offset + int((pos - step) * spt + swing_delay + late),
                            audio=note_audio(pid, n), stop=int(length * spt) if inst else None)
                if padp[pid]["mode"] == "gate" and not inst:
                    v.stop = int(length * spt)
        clock["next_tick"] += spt
    clock["sample"] = end


# ---------------------------------------------------------------------------
# Live recording + note repeat
# ---------------------------------------------------------------------------
rec = {"on": False, "fresh": set()}   # fresh: (note, step) just recorded, not yet reached
repeat = {"on": False, "rate": 4, "held": {}}     # held: note -> {"next": sample or None, "gain": g}
audio_state = {"cpu": 0.0, "latency": 0.0, "cb_time": 0.0, "cb_clock": 0, "cb_tick": -1,
               "cb_next": 0.0, "abs": 0}
REPEAT_TOLERANCE = 0.03                    # s: a press this late still counts as "on the beat"


def samples_per_step():
    return SR * 60.0 / clock["bpm"] / 4


def tick_at(clock_sample):
    """Position in steps (fractional) of a moment on the playing clock."""
    spt = samples_per_step()
    return audio_state["cb_tick"] + (clock_sample - (audio_state["cb_next"] - spt)) / spt


def now_on_clock():
    """Where the clock is right now, as heard (corrected for output latency)."""
    elapsed = time.monotonic() - audio_state["cb_time"]
    return audio_state["cb_clock"] + (elapsed - audio_state["latency"]) * SR


def grid_steps():
    """REC GRID (Settings page) as a length in 16th steps."""
    return dict(RATES).get(setup.get("rec_grid", "1/16"), 1)


def quantize_pos(raw, pid, grid=None, strength=None):
    """Where a hit played at `raw` (steps, as heard) gets recorded: swing taken
    out (playback adds it back), pulled toward the REC GRID by REC QUANT."""
    g = grid or grid_steps()
    q = setup["rec_quant"] / 100 if strength is None else strength
    sw = padp[pid]["swing"] if padp[pid]["swing"] is not None else groove["swing"]
    straight = raw - ((2 * sw / 100 - 1) if round(raw) % 2 == 1 else 0.0)
    gp = round(straight / g) * g
    return gp + (straight - gp) * (1 - q)


def place_hit(note, pos, vel, tol):
    """Put a hit at straight position `pos` (steps). A hit already within `tol`
    of it is replaced (playing over it again); others in that 16th stay."""
    L = pattern_length()
    r = round(pos)
    step = r % L
    nudge = round(max(-MAX_NUDGE, min(MAX_NUDGE, pos - r)), 3)
    hits = [h for h in step_hits(pattern[note][step]) if abs(h[1] - nudge) >= tol]
    pattern[note][step] = make_step(hits + [(max(1, min(127, int(vel))), nudge)])
    return r, step, nudge


def record_at(note, pos, vel, grid=None, strength=None):
    """Write a hit played at step position `pos` (fractional, as heard)."""
    if round(pos) < 0:
        return                                   # during the count-in: too early to keep
    g = grid or grid_steps()
    r, step, nudge = place_hit(note, quantize_pos(pos, note, g, strength), vel, min(0.5, g / 2) - 1e-6)
    scheduled_at = r - 1 if nudge < 0 else r       # the tick that will schedule this hit
    if scheduled_at > clock["tick"]:
        rec["fresh"].add((note, step, nudge))    # you just played it: don't play it twice this pass
    mark_dirty()


def start_repeat(note, gain):
    """Hold a pad with Repeat on: hits locked to the grid (or free-running when stopped)."""
    st = {"next": None, "gain": gain, "grid": clock["playing"]}
    if clock["playing"]:
        rate = RATES[repeat["rate"]][1]
        spt = samples_per_step()
        now = now_on_clock()
        pos = tick_at(now)
        k = math.ceil((pos - REPEAT_TOLERANCE * SR / spt) / rate)
        t = (audio_state["cb_next"] - spt) + (k * rate - audio_state["cb_tick"]) * spt
        st["next"] = max(t, now)
    repeat["held"][note] = st


def run_repeats(frames):
    if not repeat["held"]:
        return
    rate_samples = RATES[repeat["rate"]][1] * samples_per_step()
    for note, st in list(repeat["held"].items()):
        if note not in pad_samples:
            continue
        grid = st["grid"] and clock["playing"]
        start = clock["sample"] if grid else audio_state["abs"]
        if st["next"] is None:
            st["next"] = start                      # free-running: first hit right away
        while st["next"] < start + frames:
            offset = max(0, int(st["next"] - start))
            trigger(note, st["gain"], offset)
            if rec["on"] and grid:
                record_at(note, tick_at(st["next"]), st["gain"] * 127,
                          grid=RATES[repeat["rate"]][1], strength=1.0)
            st["next"] += rate_samples


diag = {"phase": "start", "prev": 0.0, "slow": [], "count": 0}   # dropout detective


def audio_callback(out, frames, time_info, status):
    if audio_state.get("bounce_tid"):               # an export is rendering: the live output rests
        out[:] = 0
        return
    t0 = time.perf_counter()
    gap = (t0 - diag["prev"]) * 1000 if diag["prev"] else 0.0
    diag["prev"] = t0
    phase_at_start = diag["phase"]
    try:
        _audio_callback(out, frames, time_info, status)
    finally:
        took = (time.perf_counter() - t0) * 1000
        budget = frames / SR * 1000
        if took > 0.8 * budget or (status and status.output_underflow):
            diag["count"] += 1
            if len(diag["slow"]) < 200:
                diag["slow"].append((time.monotonic(), round(took, 2), round(gap, 2),
                                     round(diag.get("master_ms", 0), 2), len(voices),
                                     phase_at_start, diag["phase"], bool(status and status.output_underflow),
                                     clock["playing"]))


def _audio_callback(out, frames, time_info, status):
    if status and status.output_underflow:
        audio_state["xrun_t"] = time.monotonic()      # a dropout: shown where CPU is
        audio_state["xruns"] = audio_state.get("xruns", 0) + 1
    out[:] = 0
    audio_state.update(cb_time=time.monotonic(), cb_clock=clock["sample"],
                       cb_tick=clock["tick"], cb_next=clock["next_tick"])
    run_repeats(frames)
    run_clock(frames)
    audio_state["abs"] += frames
    aux = np.zeros_like(out)                       # click + previews: not through the master bus
    if bus_engine is not None:
        bus_engine.audio_apply()
    if pad_engine is not None:
        pad_engine.audio_apply()
    chains = bus_engine.chains if bus_engine is not None else [None] * NUM_BUSES
    bus_in = [np.zeros_like(out) for _ in chains]  # every bus has a fader now, FX or not
    for v in list(voices):
        dst = max(0, -v.pos)                       # where in this block it starts
        if dst >= frames:                          # starts in a later block (swing)
            v.pos += frames
            continue
        if isinstance(v, SynthVoice):
            mix_synth_voice(v, out, bus_in, frames)
            continue
        src = max(0, v.pos)
        if v.stop is not None and v.fade_left is None and v.pos + frames > v.stop:
            v.fade_left = FADE + max(0, v.stop - v.pos)     # note length reached: fade out
        chunk = v.audio[src:src + frames - dst]
        strip = v.note is not None and pad_strip_on(v.note)
        if v.note is not None:                     # a pad: MONO, volume and pan apply live
            if padp[v.note]["mono_sum"]:
                chunk = chunk.mean(axis=1, keepdims=True)
            # a pad with FX: volume and pan come after its strip (below), velocity now
            chunk = chunk * v.gain if strip else chunk * pad_lr(v.note, v.gain)
        else:
            chunk = chunk * v.lr
        if v.fade_left is not None:
            n = len(chunk)
            ramp = np.clip((v.fade_left - dst - np.arange(n)) / FADE, 0, 1).astype(np.float32)
            chunk = chunk * ramp[:, None]
            v.fade_left -= frames
        if v.note is None:
            dest = aux
        else:
            b = padp[v.note].get("bus")
            if b is not None and buses[b].get("mute"):
                dest = None                            # muted bus: plays on, silently
            elif bounce_skip(v.note):
                dest = None                            # a stem: the other pads play on, silently
            else:
                dest = bus_in[b] if b is not None and bus_in[b] is not None else out
        if dest is not None and strip:
            dest = strip_in.setdefault(v.note, [np.zeros_like(out), dest])[0]
        if dest is not None:
            dest[dst:dst + len(chunk)] += chunk
        v.pos += frames
        if v.pos >= len(v.audio) or (v.fade_left is not None and v.fade_left <= 0):
            try:
                voices.remove(v)
            except ValueError:
                pass
    for pid, (buf, dest) in list(strip_in.items()):   # pads with a strip: chorus > FILTER > EQ > COMP
        y = np.ascontiguousarray(buf.T)
        mode = (padp[pid].get("ip") or {}).get("chorus", "OFF") if padp[pid].get("inst") == "KEYS" else "OFF"
        if mode != "OFF":
            if pid not in choruses:
                choruses[pid] = native_fx.JunoChorus()
            y = choruses[pid].process(y, mode)
        ch = pad_engine.chains.get(pid) if pad_engine is not None else None
        if ch:
            y = run_stages(ch, np.ascontiguousarray(y, dtype=np.float32))
        dest += y.T * pad_lr(pid, 1.0)
    strip_in.clear()
    for b, ch in enumerate(chains):                # buses run even when silent: tails ring on
        y = bus_in[b]
        if ch is not None:
            y = run_stages(ch, np.ascontiguousarray(y.T, dtype=np.float32)).T
        y = y * bus_lr(b)                          # the mixer: fader + pan
        bus_meters[b] = max(float(np.abs(y).max()), bus_meters[b] * 0.7)
        if not buses[b].get("mute"):
            out += y
    if audio_state.get("bounce_tid"):              # rendering an export: no click / previews, no clip
        if audio_state.get("bounce_master"):
            if master_bus is not None:
                out[:] = master_bus.process(out)
            out *= master["volume"]
        return
    if master_bus is not None:
        tm = time.perf_counter()
        out[:] = master_bus.process(out)
        diag["master_ms"] = (time.perf_counter() - tm) * 1000
    out += aux
    out *= master["volume"]
    meters["out"] = [max(float(np.abs(out[:, c]).max()), meters["out"][c] * 0.7) for c in (0, 1)]
    np.clip(out, -1.0, 1.0, out=out)

def mix_synth_voice(v, out, bus_in, frames):
    """A synth note into the mix: gate ends (note length, key up, choke, Stop) release it."""
    rel = None
    if v.fade_left is not None:
        rel = 0
    elif v.stop is not None and v.pos + frames > v.stop:
        rel = max(0, v.stop - v.pos)
    mono = v.render(frames, rel if not v.released else None)
    strip = pad_strip_on(v.note)
    if strip:
        chunk = mono if mono.ndim == 2 else np.repeat(mono[:, None], 2, axis=1)
    else:
        chunk = (mono if mono.ndim == 2 else mono[:, None]) * pad_lr(v.note, 1.0)
    b = padp[v.note].get("bus")
    if (b is not None and buses[b].get("mute")) or bounce_skip(v.note):
        pass
    else:
        dest = bus_in[b] if b is not None else out
        if strip:                                       # CHORUS / pad FX: the pad's notes together
            dest = strip_in.setdefault(v.note, [np.zeros_like(out), dest])[0]
        dest[:len(chunk)] += chunk
    v.pos += frames
    if v.synth.done and not v.events:
        try:
            voices.remove(v)
        except ValueError:
            pass

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
ui = {"selected": None, "voice": None, "view": "seq", "page": "PAD", "open": True}
held = set()
held_buttons = set()
ticks = {"bpm": 0, "swing": 0, "browse": 0, "browse_fast": 0}
midi = None


def step_of(note):
    """Top row = steps 1-8, row below = steps 9-16. None if not a step pad."""
    row, col = push2.pad_position(note)
    if row == 7:
        return col
    if row == 6:
        return 8 + col
    return None


def knob_steps(name, value, per):
    ticks[name] += value if value < 64 else value - 128
    moves = 0
    while abs(ticks[name]) >= per:
        d = 1 if ticks[name] > 0 else -1
        ticks[name] -= d * per
        moves += d
    return moves


# ---------------------------------------------------------------------------
# Saving: everything is written to project.json automatically, about a
# second after any change, and loaded again at start-up. Steps and tuning
# are stored by sample file name, so reordering the kit folder is safe.
# ---------------------------------------------------------------------------
import json

PROJECT_FILE = "project.json"                 # inside the open project's folder (we chdir there)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
native_fx.KEYS_DIR = os.path.join(APP_DIR, "keys")
PROJECTS_DIR = os.path.join(APP_DIR, "projects")
NUM_PROJECTS = 16
STATE_FILE = os.path.join(APP_DIR, "mpushc_state.json")    # which project was open last
SHOTS_DIR = os.path.join(APP_DIR, "screenshots")
save_state = {"dirty_since": None}


def mark_dirty():
    if save_state["dirty_since"] is None:
        save_state["dirty_since"] = time.monotonic()


SAMPLES_DIR = "samples"                       # samples used by this project live here


def pad_number(note):
    return KIT_ALL.index(note) + 1


def copy_into_project(path):
    """Copy a sample into samples/ (once) and return the new path."""
    import shutil
    if not path:
        return path                                         # an instrument pad: no file
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    if os.path.abspath(os.path.dirname(path)) == os.path.abspath(SAMPLES_DIR):
        return path
    base, ext = os.path.splitext(os.path.basename(path))
    dest, k = os.path.join(SAMPLES_DIR, base + ext), 2
    while os.path.exists(dest) and os.path.getsize(dest) != os.path.getsize(path):
        dest, k = os.path.join(SAMPLES_DIR, f"{base} ({k}){ext}"), k + 1   # same name, other sound
    if not os.path.exists(dest):
        shutil.copy2(path, dest)
    return dest


def encode_step(i, ev):
    hits = step_hits(ev)
    if len(hits) > 1:
        return [i, [[v, round(n, 3)] for v, n in hits]]
    vel, nudge = hits[0]
    return i if (vel, nudge) == (DEFAULT_VEL, 0.0) else [i, vel, round(nudge, 3)]


def decode_steps(items):
    out = [None] * MAX_STEPS
    for it in items:
        if isinstance(it, int):
            if 0 <= it < MAX_STEPS:
                out[it] = (DEFAULT_VEL, 0.0)
        elif len(it) == 2:                          # several hits in one 16th
            i, hits = it
            if 0 <= i < MAX_STEPS:
                out[i] = make_step([(int(v), float(n)) for v, n in hits])
        else:
            i, vel, nudge = it
            if 0 <= i < MAX_STEPS:
                out[i] = (int(vel), float(nudge))
    return out


source_cache = {}                       # file -> audio, so 16 chops load the file once


def save_project():
    for n in list(pad_files):                    # samples from kit/ (first run) get copied too
        pad_files[n] = copy_into_project(pad_files[n])
    store_current()
    data = {
        "version": 5,
        "current_pattern": pat["cur"],
        "name": proj.get("name", ""),
        "song": {"rows": [[r[0] + 1, r[1]] for r in song["rows"]], "mode": song["mode"], "end": song["end"]},
        "patterns": [None if sl is None else {
            "bars": sl["bars"], "swing": sl["swing"], "humanize": sl["humanize"],
            "steps": {str(pad_number(n)): [encode_step(i, ev) for i, ev in enumerate(st) if ev]
                      for n, st in sl["steps"].items()},
            "notes": {str(pad_number(n)): [[round(e[0], 4), e[1], e[2], round(e[3], 4)] for e in evs]
                      for n, evs in sl.get("notes", {}).items() if evs},
            "auto": {k: [[i, v] for i, v in enumerate(lane) if v is not None]
                     for k, lane in sl.get("auto", {}).items() if any(v is not None for v in lane)},
        } for sl in slots],
        "bpm": clock["bpm"],
        "bars": seq["bars"],
        "follow": seq["follow"],
        "zoom": seq["zoom"],
        "swing": groove["swing"],
        "humanize": groove["humanize"],
        "muted": [pad_number(n) for n in mutes],
        "soloed": [pad_number(n) for n in solos],
        "volume": master["volume"],
        "master_fx": master_fx,
        "buses": buses,
        "setup": setup,
        "page": ui["page"],
        "pads": {
            str(pad_number(n)): {"file": pad_files[n], "name": pad_samples[n][0], "tune": tune[n], **padp[n],
                                 **({"start": pad_region[n][0], "end": pad_region[n][1]} if n in pad_region else {})}
            for n in kit_notes
        },
    }
    tmp = PROJECT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, PROJECT_FILE)            # never leaves a half-written file
    save_state["dirty_since"] = None


def autosave():
    t = save_state["dirty_since"]
    if t is not None and time.monotonic() - t > 1.0:
        save_project()


def apply_pad(note, pad):
    for key in padp[note]:
        if key in pad:
            padp[note][key] = pad[key]
    padp[note]["fx"] = load_pad_fx(pad.get("fx"))
    set_tune(note, pad.get("tune", 0))
    pattern[note][:] = decode_steps(pad.get("steps", []))


def load_project(first_run=False):
    if not os.path.exists(PROJECT_FILE):
        if first_run:
            load_kit_folder()
            print("No saved project yet: starting from the kit folder.")
        return
    with open(PROJECT_FILE) as f:
        data = json.load(f)
    proj["name"] = data.get("name", "")
    clock["bpm"] = data.get("bpm", clock["bpm"])
    master["volume"] = data.get("volume", master["volume"])
    master_fx.clear()
    master_fx.update(MASTER_DEFAULTS)
    master_fx.update({k: v for k, v in data.get("master_fx", {}).items() if k in MASTER_DEFAULTS})
    load_buses(data.get("buses"))
    set_bars(data.get("bars", 1))
    seq["follow"] = data.get("follow", False)
    seq["zoom"] = data.get("zoom", 1) if data.get("zoom", 1) in ZOOM_LEVELS else 1
    groove["swing"] = data.get("swing", 50)
    groove["humanize"] = data.get("humanize", 0)
    setup.update({k: v for k, v in data.get("setup", {}).items() if k in setup})
    if setup["curve"] not in CURVES:
        setup["curve"] = "lin"
    setup.pop("velocity", None)
    if data.get("page") in PAGES and PAGES[data["page"]].get("ready"):
        ui["page"] = data["page"]

    if data.get("version", 1) >= 4:
        # pads are numbered 1-16, each with its own copied sample file
        for num, pad in data.get("pads", {}).items():
            note = int(num) - 1
            path = pad.get("file", "")
            if pad.get("inst"):
                create_instrument(note, pad["inst"], undo=False)
                apply_pad(note, pad)
                padp[note]["ip"] = {**INST_DEFAULTS[pad["inst"]], **(padp[note]["ip"] or {})}
                keys_preload(note)
                continue
            if not os.path.exists(path):
                print(f"  pad {num}: missing sample {path}")
                continue
            if path not in source_cache:
                source_cache[path] = load_sample(path)
            full = source_cache[path]
            if "start" in pad:
                region = (pad["start"], pad["end"])
                set_pad_sample(note, pad.get("name", os.path.basename(path)), full[region[0]:region[1]], path, region)
            else:
                set_pad_sample(note, pad.get("name", os.path.basename(path)), full, path)
            apply_pad(note, pad)
        mutes.update(x - 1 for x in data.get("muted", []) if x - 1 in pad_samples)
        solos.update(x - 1 for x in data.get("soloed", []) if x - 1 in pad_samples)
    else:
        # older projects: kit/ folder, pads matched by file name
        load_kit_folder()
        by_name = {pad_samples[n][0]: n for n in pad_samples}
        for name, pad in data.get("pads", {}).items():
            if name in by_name:
                apply_pad(by_name[name], pad)
        mutes.update(by_name[x] for x in data.get("muted", []) if x in by_name)
        solos.update(by_name[x] for x in data.get("soloed", []) if x in by_name)
        mark_dirty()                          # re-save in the new format right away
    if data.get("version", 1) >= 5:
        for i, sl in enumerate(data.get("patterns", [])[:NUM_SLOTS]):
            if sl is None:
                continue
            steps = {}
            for num, idx in sl.get("steps", {}).items():
                steps[int(num) - 1] = decode_steps(idx)
            slots[i] = {"bars": sl.get("bars", 1), "swing": sl.get("swing", 50),
                        "humanize": sl.get("humanize", 0), "steps": steps,
                        "notes": {int(num) - 1: [tuple(e) for e in evs]
                                  for num, evs in sl.get("notes", {}).items()},
                        "auto": {k: load_lane(pts) for k, pts in (sl.get("auto") or {}).items()}}
        pat["cur"] = data.get("current_pattern", 0)
        sg = data.get("song") or {}
        song.update(rows=[[int(r[0]) - 1, int(r[1])] for r in sg.get("rows", [])
                          if 1 <= int(r[0]) <= NUM_SLOTS],
                    mode=sg.get("mode", "SONG"), end=sg.get("end", "LOOP"), sel=0, page=0, pos=None, rep=0)
        if slots[pat["cur"]] is None:
            slots[pat["cur"]] = capture()
        apply_slot(slots[pat["cur"]])
        pat["page"] = pat["cur"] // SLOTS_PER_PAGE
    else:
        slots[0] = capture()                 # older projects: the one pattern becomes P1
    print(f"Loaded {PROJECT_FILE}: {clock['bpm']} BPM, pattern {pat['cur'] + 1}, "
          f"{seq['bars']} bar(s), {len(kit_notes)} pads.")


# ---------------------------------------------------------------------------
# Knob pages. Each page gives the 8 knobs above the screen a job.
# A Param knows its label, how to show its value and how to change it.
# ---------------------------------------------------------------------------
KNOB_FIRST = 71                          # knobs 1-8 above the screen: CC 71-78
UPPER_BUTTONS = list(range(102, 110))    # the 8 buttons right under those knobs


class Param:
    def __init__(self, label, get, put, per=1, show=str, needs_pad=False):
        self.label, self.get, self.put, self.per, self.show = label, get, put, per, show
        self.needs_pad, self.ticks = needs_pad, 0
        self.target = None              # automation: () -> the setting's key, or None

    def active(self):
        return not self.needs_pad or ui["selected"] is not None

    def text(self):
        return self.show(self.get()) if self.active() else "-"

    def auto_as(self, target):
        """Automatable: target() = the setting's key (None = nothing to automate now)."""
        def t():
            try:
                return target() if self.active() else None
            except (KeyError, TypeError):
                return None
        self.target = t
        return self

    def turn(self, value):
        if not self.active():
            return
        self.ticks += value if value < 64 else value - 128
        while abs(self.ticks) >= self.per:
            d = 1 if self.ticks > 0 else -1
            self.ticks -= d * self.per
            self.put(d)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sel():
    return ui["selected"]


def step_choice(options, current, d):
    """Move one step through a list of options (e.g. 'pattern', 50, 51 ...)."""
    i = options.index(current) if current in options else 0
    return options[clamp(i + d, 0, len(options) - 1)]


GRID_CHOICES = ["1/8", "1/8T", "1/16", "1/16T", "1/32", "1/32T"]
SWING_CHOICES = [None] + list(range(SWING_MIN, SWING_MAX + 1))
HUMAN_CHOICES = [None] + list(range(0, 101))


def put_pad(key, choices):
    def put(d):
        padp[sel()][key] = step_choice(choices, padp[sel()][key], d)
    return put


def pan_text(p):
    return "C" if p == 0 else (f"L{-p}" if p < 0 else f"R{p}")


PAGES = {
    "PAD": {"ready": True, "colour": (255, 140, 40), "knobs": [
        Param("VOLUME", lambda: padp[sel()]["volume"],
              lambda d: padp[sel()].update(volume=clamp(padp[sel()]["volume"] + d, 0, 100)),
              show=lambda v: f"{v}%", needs_pad=True).auto_as(lambda: f"pad:{sel()}:volume"),
        Param("PITCH", lambda: tune[sel()], lambda d: set_tune(sel(), tune[sel()] + d),
              per=TICKS_PER_SEMITONE, show=lambda t: f"{note_name(ROOT + t)} {t:+d}", needs_pad=True),
        Param("PAN", lambda: padp[sel()]["pan"],
              lambda d: padp[sel()].update(pan=clamp(padp[sel()]["pan"] + d, -50, 50)),
              show=pan_text, needs_pad=True).auto_as(lambda: f"pad:{sel()}:pan"),
        Param("CHANNELS", lambda: padp[sel()]["mono_sum"], lambda d: padp[sel()].update(mono_sum=d > 0),
              per=4, show=lambda m: "MONO" if m else "STEREO", needs_pad=True),
        Param("SWING", lambda: padp[sel()]["swing"], put_pad("swing", SWING_CHOICES), per=2,
              show=lambda v: "pattern" if v is None else f"{v}%", needs_pad=True),
        Param("HUMAN", lambda: padp[sel()]["humanize"], put_pad("humanize", HUMAN_CHOICES),
              show=lambda v: "pattern" if v is None else f"{v}%", needs_pad=True),
        Param("BUS", lambda: padp[sel()].get("bus"), put_pad("bus", [None, 0, 1, 2, 3]), per=4,
              show=lambda b: "MASTER" if b is None else f"BUS {b + 1}", needs_pad=True),
        Param("CHOKE", lambda: padp[sel()]["choke"],
              lambda d: padp[sel()].update(choke=clamp(padp[sel()]["choke"] + d, 0, CHOKE_GROUPS)), per=4,
              show=lambda g: "OFF" if g == 0 else str(g), needs_pad=True)]},
    "FILTER": {"ready": True, "colour": (255, 150, 60), "knobs": [None] * 8},      # the pad's strip
    "EQ": {"ready": True, "colour": (90, 200, 255), "knobs": [None] * 8},
    "COMP": {"ready": True, "colour": (240, 90, 110), "knobs": [None] * 8},
    "SAMPLE": {"ready": True, "colour": (232, 64, 58), "knobs": [
        Param("START", lambda: trim_bounds(sel())[0], lambda d: trim_move("start", d * trim_step()),
              show=lambda v: f"{v / SR:.3f} s", needs_pad=True),
        Param("END", lambda: trim_bounds(sel())[1], lambda d: trim_move("end", d * trim_step()),
              show=lambda v: f"{v / SR:.3f} s", needs_pad=True),
        Param("TRIM", lambda: "TURN >" if sel() in pad_region else "—", lambda d: trim_commit(sel(), d),
              per=6, needs_pad=True),
        Param("BPM", lambda: pad_bpm(sel()), lambda d: set_pad_bpm(sel(), d), per=2,
              show=lambda b: "—" if not b else f"{b:g}", needs_pad=True),
        Param("STYLE", lambda: padp[sel()].get("style", "classic").upper(), lambda d: set_style(sel(), d),
              per=4, needs_pad=True),
        Param("MODE", lambda: padp[sel()]["mode"],
              lambda d: padp[sel()].update(mode="gate" if d > 0 else "one-shot"), per=4,
              show=lambda m: "GATE" if m == "gate" else "1-SHOT", needs_pad=True),
        Param("STRETCH", lambda: stretch_text(sel()), lambda d: stretch_to_bpm(sel(), d), per=6,
              needs_pad=True),
        Param("TUNE", lambda: padp[sel()].get("fine", 0), lambda d: set_fine(sel(), d),
              show=lambda c: f"{c:+d} ct", needs_pad=True)]},
    "PATTERN": {"ready": True, "colour": (190, 110, 255), "knobs": [
        Param("BARS", lambda: seq["bars"], lambda d: set_bars(seq["bars"] + d), per=6),
        Param("SWING", lambda: groove["swing"],
              lambda d: groove.update(swing=clamp(groove["swing"] + d, SWING_MIN, SWING_MAX)),
              per=2, show=lambda v: f"{v}%"),
        Param("HUMAN", lambda: groove["humanize"],
              lambda d: groove.update(humanize=clamp(groove["humanize"] + d, 0, 100)),
              show=lambda v: f"{v}%"),
        Param("BPM", lambda: clock["bpm"],
              lambda d: clock.update(bpm=clamp(clock["bpm"] + d, BPM_MIN, BPM_MAX)), per=TICKS_PER_BPM),
        None, None, None, None]},
    "DEVICE": {"ready": True, "colour": (60, 190, 200), "knobs": [None] * 8},
    "INST": {"ready": True, "colour": (250, 170, 60), "knobs": [None] * 8},
    "MIX": {"ready": True, "colour": (235, 235, 240), "knobs": [None] * 8},      # filled in below
    "SONG": {"ready": True, "colour": (120, 220, 140), "knobs": [None] * 8},
    "MASTER": {"ready": True, "colour": (235, 235, 240), "knobs": [
        Param("THRESH", lambda: master_fx["thresh"],
              lambda d: master_fx.update(thresh=clamp(master_fx["thresh"] + d, -30, 0)),
              show=lambda v: f"{v} dB").auto_as(lambda: "master:thresh"),
        Param("RATIO", lambda: master_fx["ratio"],
              lambda d: master_fx.update(ratio=step_choice(RATIO_CHOICES, master_fx["ratio"], d)), per=4,
              show=lambda r: "OFF" if not r else f"{r}:1").auto_as(lambda: "master:ratio"),
        Param("ATTACK", lambda: master_fx["attack"],
              lambda d: master_fx.update(attack=step_choice(ATTACK_CHOICES, master_fx["attack"], d)), per=4,
              show=lambda a: f"{a:g} ms").auto_as(lambda: "master:attack"),
        Param("RELEASE", lambda: master_fx["release"],
              lambda d: master_fx.update(release=step_choice(RELEASE_CHOICES, master_fx["release"], d)),
              per=4, show=lambda r: f"{r:g} s").auto_as(lambda: "master:release"),
        Param("MAKEUP", lambda: master_fx["makeup"],
              lambda d: master_fx.update(makeup=clamp(master_fx["makeup"] + d, 0, 15)), per=2,
              show=lambda v: f"+{v} dB").auto_as(lambda: "master:makeup"),
        Param("MIX", lambda: master_fx["mix"],
              lambda d: master_fx.update(mix=clamp(master_fx["mix"] + 2 * d, 0, 100)),
              show=lambda v: f"{v}%").auto_as(lambda: "master:mix"),
        Param("LIMIT", lambda: master_fx["drive"],
              lambda d: master_fx.update(drive=clamp(master_fx["drive"] + d, 0, 12)), per=2,
              show=lambda v: f"+{v} dB").auto_as(lambda: "master:drive"),
        Param("VOLUME", lambda: round(master["volume"] * 100),
              lambda d: master.update(volume=round(clamp(master["volume"] + d * VOLUME_STEP, 0.0, 1.0), 2)),
              show=lambda v: f"{v}%").auto_as(lambda: "mastervol")]},
    "SETUP": {"ready": True, "colour": (200, 200, 210), "knobs": [
        Param("CLICK", lambda: setup["click"],
              lambda d: setup.update(click=clamp(setup["click"] + 2 * d, 0, 100)),
              show=lambda v: "off" if v == 0 else f"{v}%"),
        Param("KEY", lambda: setup["key_root"],
              lambda d: setup.update(key_root=(setup["key_root"] + d) % 12), per=3,
              show=lambda r: NOTE_NAMES_12[r]),
        Param("SCALE", lambda: setup["key_scale"],
              lambda d: setup.update(key_scale=step_choice(SCALE_NAMES, setup["key_scale"], d)), per=3),
        None,
        Param("COUNT-IN", lambda: setup["count_in"],
              lambda d: setup.update(count_in=clamp(setup["count_in"] + d, 0, 2)), per=6,
              show=lambda b: "off" if b == 0 else f"{b} bar"),
        Param("REC QUANT", lambda: setup["rec_quant"],
              lambda d: setup.update(rec_quant=step_choice([0, 25, 50, 75, 100], setup["rec_quant"], d)),
              per=6, show=lambda v: "off" if v == 0 else f"{v}%"),
        Param("REC GRID", lambda: setup["rec_grid"],
              lambda d: setup.update(rec_grid=step_choice(GRID_CHOICES, setup["rec_grid"], d)), per=6),
        None]},
}
PAGE_ORDER = ["PAD", "FILTER", "EQ", "COMP", "SAMPLE", None, "PATTERN", "SETUP"]
PAGE_LABELS = {"SETUP": "SETTINGS"}                 # what the screen calls a page


# ---------------------------------------------------------------------------
# Sample browser (browser.py). Previews play through the normal mixer.
# ---------------------------------------------------------------------------
def play_preview(audio):
    stop_preview()
    v = Voice(audio, 0.8, None)
    voices.append(v)
    ui["preview"] = v


def stop_preview():
    v = ui.get("preview")
    if v is not None and v.fade_left is None:
        v.fade_left = FADE
    ui["preview"] = None


favorites = []                                       # favorite samples / folders (all projects)
browser = Browser(LIBRARY, SR, play_preview, stop_preview, favorites=favorites, search_roots=SEARCH_ROOTS,
                  on_favorites=lambda: save_app_state())


def load_to_pad(note, path):
    """Copy the sample into the project, load it, put it on the pad.
    The pad keeps its steps, volume, pan, swing and humanize."""
    browser.message = f"loading onto pad {pad_name(note)}..."
    try:
        dest = copy_into_project(path)
        audio = load_sample(dest)
    except Exception as err:
        browser.message = f"couldn't load ({type(err).__name__})"
        return
    stop_preview()
    checkpoint(f"load pad {pad_name(note)}")
    set_pad_sample(note, os.path.basename(dest), audio, dest)
    ui["selected"] = note
    auto_detect(note, apply_root=True)
    mark_dirty()
    browser.message = f"loaded onto pad {pad_name(note)}"


def browser_control(c, v):
    """Buttons and knobs while the browser is open. Returns True if handled."""
    if c == KNOB_FIRST:
        browser.move(knob_steps("browse", v, BROWSE_TICKS))
    elif c == KNOB_FIRST + 1:
        browser.move(10 * knob_steps("browse_fast", v, BROWSE_FAST_TICKS))
    elif c == ARROW_LEFT and v > 0:
        if ui.get("pending"):
            ui["pending"] = None              # cancel the replace question
        else:
            browser.back()
    elif c == ARROW_RIGHT and v > 0:
        browser.enter()
    elif c == ZOOM_IN and v > 0:          # up arrow
        browser.move(-1)
    elif c == ZOOM_OUT and v > 0:         # down arrow
        browser.move(1)
    elif c in UPPER_BUTTONS and UPPER_BUTTONS.index(c) == 6:
        if v > 0:                         # FAV: the highlighted sample or folder
            on = browser.toggle_favorite()
            if on is not None:
                item = browser.current()
                toast(("added to" if on else "removed from") + " favorites"
                      + (f": {os.path.basename(item[1].rstrip(os.sep))}" if on and item else ""))
    elif c in UPPER_BUTTONS and UPPER_BUTTONS.index(c) == 7:
        if v > 0:                         # SEARCH: typed on the computer
            ui["pending"] = None
            where_ = "this folder" if browser.path not in (None, "*favorites", "*search") else "the sample library"
            ask_on_computer("text", f"Search samples in {where_} (words in the file or folder name)",
                            browser.query, lambda t: t and t.strip() and browser.search(t.strip()))
    elif KNOB_FIRST <= c < KNOB_FIRST + 8 or c in UPPER_BUTTONS:
        pass                              # knob pages are paused while browsing
    else:
        return False
    return True


# ---------------------------------------------------------------------------
# Undo / redo. Before every change we keep a snapshot of the music (steps,
# pads, samples, tuning, groove, mutes, bars, BPM). Turning a knob counts as
# one change until you leave it alone for a moment.
# ---------------------------------------------------------------------------
UNDO_LIMIT = 100
history = {"undo": [], "redo": [], "last_key": None, "last_time": 0.0}


def snapshot():
    return {
        "pattern": {n: list(p) for n, p in pattern.items()},
        "melody": {n: list(e) for n, e in melody.items()},
        "padp": {n: dict(p, ip=dict(p["ip"]) if p.get("ip") else None, fx=copy_pad_fx(p.get("fx")))
                 for n, p in padp.items()},
        "tune": dict(tune), "samples": dict(pad_samples), "files": dict(pad_files),
        "regions": dict(pad_region),
        "mutes": set(mutes), "solos": set(solos), "groove": dict(groove),
        "bars": seq["bars"], "bpm": clock["bpm"], "setup": dict(setup), "master_fx": dict(master_fx),
        "buses": copy_buses(),
        "selected": ui["selected"],
        "slots": [copy_slot(sl) for sl in slots[:pat["cur"]]] + [capture()]
                 + [copy_slot(sl) for sl in slots[pat["cur"] + 1:]],
        "cur": pat["cur"],
        "song": {"rows": [list(r) for r in song["rows"]], "mode": song["mode"], "end": song["end"]},
    }


def restore(snap):
    for n, p in snap["pattern"].items():
        pattern[n][:] = p
    melody.clear()
    melody.update({n: list(e) for n, e in snap.get("melody", {}).items()})
    for n, p in snap["padp"].items():
        padp[n].clear()
        padp[n].update(p)
        padp[n]["fx"] = copy_pad_fx(p.get("fx"))
        padp[n]["ip"] = dict(p["ip"]) if p.get("ip") else None
    old_samples, old_tune = dict(pad_samples), dict(tune)
    pad_samples.clear(); pad_samples.update(snap["samples"])
    pad_files.clear(); pad_files.update(snap["files"])
    pad_region.clear(); pad_region.update(snap["regions"])
    tune.update(snap["tune"])
    for n in KIT_ALL:
        if n not in pad_samples:
            pitched.pop(n, None)
            wave_peaks.pop(n, None)
        elif old_samples.get(n) is not pad_samples[n] or pitched_shift.get(n) != pad_shift(n):
            pitched_shift[n] = None
            retune(n)
    refresh_kit_notes()
    mutes.clear(); mutes.update(snap["mutes"])
    solos.clear(); solos.update(snap["solos"])
    groove.update(snap["groove"])
    setup.update(snap["setup"])
    master_fx.update(snap.get("master_fx", {}))
    if "buses" in snap:
        load_buses(snap["buses"])
    set_bars(snap["bars"])
    clock["bpm"] = snap["bpm"]
    ui["selected"] = snap["selected"] if snap["selected"] in pad_samples else None
    slots[:] = [copy_slot(sl) for sl in snap["slots"]]
    pat["cur"], pat["queued"] = snap["cur"], None
    auto_lanes.clear()
    auto_lanes.update(copy_lanes((slots[pat["cur"]] or {}).get("auto")))
    if "song" in snap:
        song["rows"] = [list(r) for r in snap["song"]["rows"]]
        song.update(mode=snap["song"]["mode"], end=snap["song"]["end"])
        song["sel"] = max(0, min(song["sel"], len(song["rows"]) - 1))
        if song["pos"] is not None and song["pos"] >= len(song["rows"]):
            song["pos"] = None
    pat["page"] = pat["cur"] // SLOTS_PER_PAGE


def checkpoint(label, key=None):
    """Call BEFORE a change. key groups a run of knob turns into one undo."""
    now = time.monotonic()
    if key is not None and key == history["last_key"] and now - history["last_time"] < 1.5:
        history["last_time"] = now
        return
    history["undo"].append((label, snapshot()))
    del history["undo"][:-UNDO_LIMIT]
    history["redo"].clear()
    history["last_key"], history["last_time"] = key, now


def toast(text):
    ui["toast"] = (text, time.monotonic())


def undo():
    if not history["undo"]:
        toast("nothing to undo")
        return
    label, snap = history["undo"].pop()
    history["redo"].append((label, snapshot()))
    restore(snap)
    history["last_key"] = None
    toast(f"undo: {label}")
    mark_dirty()


def redo():
    if not history["redo"]:
        toast("nothing to redo")
        return
    label, snap = history["redo"].pop()
    history["undo"].append((label, snapshot()))
    restore(snap)
    history["last_key"] = None
    toast(f"redo: {label}")
    mark_dirty()


def pad_label(note):
    return os.path.splitext(pad_samples[note][0])[0][:14] if note in pad_samples else f"pad {pad_name(note)}"


def delete_pad(note):
    """Delete + pad: clear its steps. If it has none, remove the sample."""
    if any(pattern[note]) or melody.get(note):
        checkpoint(f"clear {pad_label(note)}")
        pattern[note][:] = [None] * MAX_STEPS
        melody.pop(note, None)
        toast(f"cleared steps: {pad_label(note)}")
    else:
        checkpoint(f"remove {pad_label(note)}")
        name = pad_label(note)
        for d in (pad_samples, pad_files, pitched, wave_peaks, pad_region):
            d.pop(note, None)
        padp[note].update(inst=None, ip=None)
        mutes.discard(note)
        solos.discard(note)
        refresh_kit_notes()
        if ui["selected"] == note:
            ui["selected"] = None
        toast(f"removed sample: {name}")
    mark_dirty()


taps = []


def tap_tempo():
    """Tap 2+ times in time: BPM = the average gap of the last taps."""
    now = time.monotonic()
    if taps and now - taps[-1] > 2.0:
        taps.clear()                                # long pause: start over
    taps.append(now)
    del taps[:-5]
    if len(taps) >= 2:
        gaps = [b - a for a, b in zip(taps, taps[1:])]
        bpm = clamp(round(60 / (sum(gaps) / len(gaps))), BPM_MIN, BPM_MAX)
        checkpoint("tap tempo", key="tap")
        clock["bpm"] = bpm
        toast(f"tap tempo: {bpm} BPM")
        mark_dirty()
    else:
        toast("tap tempo: keep tapping")


held_steps = {}            # step pad note -> {"pad", "s", "existed", "edited"}


def edit_held_steps(knob, amount):
    """While step pads are held: knob 1 = velocity, knob 2 = nudge (early/late);
    Shift + knob 1 = CHANCE (how often the step plays)."""
    if knob == 0 and SHIFT_BUTTON in held_buttons:
        checkpoint("step chance", key=("stepchance", tuple(sorted(held_steps))))
        for info in held_steps.values():
            info["edited"] = True
            if pattern[info["pad"]][info["s"]]:
                set_step_chance(info["pad"], info["s"],
                                clamp(step_chance(info["pad"], info["s"]) + 5 * amount, 0, 100))
        mark_dirty()
        return
    checkpoint("edit step", key=("stepedit", knob, tuple(sorted(held_steps))))
    for info in held_steps.values():
        info["edited"] = True
        ev = pattern[info["pad"]][info["s"]]
        if not ev:
            continue
        new = []
        for vel, nudge in step_hits(ev):                # several hits: all move together
            if knob == 0:
                vel = clamp(vel + amount, 1, 127)
            else:
                nudge = round(max(-MAX_NUDGE, min(MAX_NUDGE, nudge + amount * NUDGE_PER_TICK)), 3)
            new.append((vel, nudge))
        pattern[info["pad"]][info["s"]] = make_step(new)
    mark_dirty()


def quantize(whole_pattern=False):
    """Pull hits halfway to the REC GRID each press (press again to tighten more)."""
    notes = list(pattern) if whole_pattern else ([ui["selected"]] if ui["selected"] is not None else [])
    if not notes:
        toast("select a pad first (or Shift + Quantize for the whole pattern)")
        return
    checkpoint("quantize")
    moved = 0
    g, L = grid_steps(), pattern_length()
    for n in notes:
        hits = []                                       # (straight position, velocity)
        for i, ev in enumerate(pattern[n]):
            for vel, nudge in step_hits(ev):
                pos = i + nudge
                gp = round(pos / g) * g
                if abs(pos - gp) > 1e-6:
                    pos = gp if abs(pos - gp) < 0.02 else gp + (pos - gp) / 2
                    moved += 1
                hits.append((pos, vel, i < L))
        pattern[n][:] = [None] * MAX_STEPS
        for pos, vel, inside in hits:
            r = round(pos)
            step = r % L if inside else min(r, MAX_STEPS - 1)
            nudge = round(max(-MAX_NUDGE, min(MAX_NUDGE, pos - r)), 3)
            have = step_hits(pattern[n][step])
            if any(abs(h[1] - nudge) < 0.01 for h in have):   # landed on the same spot: keep one
                continue
            pattern[n][step] = make_step(have + [(vel, nudge)])
    toast(("quantized pattern" if whole_pattern else f"quantized {pad_label(notes[0])}")
          + ("" if moved else "  ·  already on the grid"))
    mark_dirty()


def select_neighbour(d):
    """Select the pad above/below on the grid without playing it, in any bank,
    so its steps show on the step pads."""
    if not kit_notes:
        return
    i = kit_notes.index(ui["selected"]) if ui["selected"] in kit_notes else (-1 if d > 0 else len(kit_notes))
    ui["selected"] = kit_notes[max(0, min(len(kit_notes) - 1, i + d))]


def start_playing(count_in=0):
    """count_in = bars of clicks before bar 1 (used when Record starts playback)."""
    clock.update(sample=0, next_tick=0.0, tick=-1 - count_in * STEPS)
    song["pos"] = None
    if song_on():
        song_start()
    clock["playing"] = True


def duplicate_pattern(target):
    """Copy the current pattern into `target` (or the next empty slot) and jump there.
    Same length, so playback carries on seamlessly."""
    if target is None:
        empty = [i for i in range(NUM_SLOTS) if slots[i] is None and i != pat["cur"]]
        if not empty:
            toast("no empty pattern slot")
            return
        target = empty[0]
    if target == pat["cur"]:
        return
    checkpoint(f"duplicate P{pat['cur'] + 1} to P{target + 1}")
    store_current()
    slots[target] = copy_slot(slots[pat["cur"]])
    pat["cur"], pat["queued"] = target, None
    pat["page"] = target // SLOTS_PER_PAGE
    toast(f"duplicated to P{target + 1}")
    mark_dirty()


def pattern_button(i):
    if DUPLICATE_BUTTON in held_buttons:            # hold Duplicate + slot
        ui["dup_used"] = True
        duplicate_pattern(i)
    elif ui.get("confirm_clear") == i:            # second press: yes, clear it
        ui["confirm_clear"] = None
        checkpoint(f"clear P{i + 1}")
        if i == pat["cur"]:
            for p in pattern.values():
                p[:] = [None] * MAX_STEPS
            melody.clear()
            auto_lanes.clear()
        else:
            slots[i] = None
        toast(f"cleared P{i + 1}")
        mark_dirty()
    elif DELETE_BUTTON in held_buttons:             # hold Delete + slot: ask first
        sl = capture() if i == pat["cur"] else slots[i]
        if sl is None or not sl["steps"]:
            toast(f"P{i + 1} is already empty")
        else:
            ui["confirm_clear"] = i
    elif i == pat["cur"]:
        pat["queued"] = None                       # cancel a queued change
    elif clock["playing"]:
        pat["queued"] = i                          # change at the end of this loop
    else:
        checkpoint(f"go to P{i + 1}")
        switch_pattern(i)
        mark_dirty()


# ---------------------------------------------------------------------------
# Automation (Automate button). Each pattern keeps lanes: setting key -> one value
# per 16th step (None = no point). Armed + playing, a knob you touch or turn writes
# its value into every step the playhead passes; let go and the lane plays back,
# moving the setting (and the knob's value) by itself. Between two points the value
# glides; with gaps it holds the last point. Delete + touch a knob: clear its lane.
# Keys: "pad:<pad>:volume|pan", "padfx:<pad>:<FILTER|EQ|COMP>:<setting>",
# "ip:<pad>:<setting>" (synths), "bus:<bus>:<slot>:<effect>:<setting>",
# "busmix:<bus>:vol|pan", "master:<setting>", "mastervol".
# ---------------------------------------------------------------------------
AUTOMATE_BUTTON = 89
auto_lanes = {}                          # the current pattern's lanes (like `pattern`)
auto = {"arm": False, "touched": set(), "turned": {}, "writing": set()}


def step_chance(pid, s):
    """A step's chance of playing (0-100). Stored per pattern as an automation-style
    lane "chance:<pad>" (saved, copied, undone with the pattern); never played back."""
    lane = auto_lanes.get(f"chance:{pid}")
    v = lane[s] if lane and 0 <= s < len(lane) else None
    return 100 if v is None else v


def set_step_chance(pid, s, v):
    key = f"chance:{pid}"
    if v >= 100:
        lane = auto_lanes.get(key)
        if lane:
            lane[s] = None
            if all(x is None for x in lane):
                auto_lanes.pop(key)
    else:
        auto_lanes.setdefault(key, [None] * MAX_STEPS)[s] = int(v)


def chance_roll(pid, s):
    c = step_chance(pid, s)
    return c >= 100 or random.random() * 100 < c


def real_lanes():
    """Automation lanes (not the CHANCE ones)."""
    return [k for k in auto_lanes if not k.startswith("chance:")]


def auto_resolve(key):
    """(get, set) for a lane's setting, or None if it's gone (pad emptied, effect changed)."""
    try:
        part = key.split(":")
        kind = part[0]
        if kind == "pad":
            dd, f = padp[int(part[1])], part[2]
        elif kind == "padfx":
            fx = padp[int(part[1])].get("fx")
            if not fx:
                return None
            dd, f = fx[part[2]]["p"], part[3]
        elif kind == "ip":
            dd, f = padp[int(part[1])].get("ip"), part[2]
        elif kind == "bus":
            sl = buses[int(part[1])]["slots"][int(part[2])]
            if not sl or sl["type"] != part[3]:
                return None
            dd, f = sl["p"], part[4]
        elif kind == "busmix":
            dd, f = buses[int(part[1])], part[2]
        elif kind == "master":
            dd, f = master_fx, part[1]
        elif kind == "mastervol":
            dd, f = master, "volume"
        else:
            return None
        if dd is None or f not in dd:
            return None
        return (lambda: dd[f]), (lambda v: dd.__setitem__(f, v))
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def knob_target(i):
    ks = page_knobs() if ui["open"] and not ui.get("settings") else []
    param = ks[i] if i < len(ks) else None
    t = getattr(param, "target", None) if param is not None else None
    return t() if t else None


def lock_value(x):
    """A step lock (set by holding a step) is stored as {"v": value}: no glide in or out."""
    return x["v"] if isinstance(x, dict) else x


def lane_value(lane, step, frac, L):
    """The lane's value at step + frac (0..1): glide to the next point if it's the very
    next step, otherwise hold the last point (looking back round the loop)."""
    v = None
    for k in range(L):
        v = lane[(step - k) % L]
        if v is not None:
            break
    if v is None:
        return None
    nxt = lane[(step + 1) % L]
    if isinstance(v, dict) or isinstance(nxt, dict):
        return lock_value(v)
    if lane[step] is not None and nxt is not None and isinstance(v, (int, float)) \
            and not isinstance(v, bool) and isinstance(nxt, (int, float)) and not isinstance(nxt, bool):
        x = v + (nxt - v) * frac
        return int(round(x)) if isinstance(v, int) and isinstance(nxt, int) else x
    return v


def auto_tick():
    """Main loop: record the knobs you hold (armed) and play the other lanes back."""
    if audio_state.get("bounce_tid"):
        return
    if not clock["playing"] or clock["tick"] < 0:
        auto["writing"] = set()
        return
    L = pattern_length()
    spt = SR * 60.0 / clock["bpm"] / 4
    step = clock["tick"] % L
    frac = max(0.0, min(1.0, 1 - (clock["next_tick"] - clock["sample"]) / spt))
    now = time.monotonic()
    writing = set()
    knobs = auto["touched"] | {i for i, t in auto["turned"].items() if now - t < 0.3}
    if held_steps:                            # holding a step: knobs edit that step, not the playhead
        knobs = set()
    if not auto["arm"]:                       # not armed: the knob you hold wins, nothing is written
        writing = {k for k in (knob_target(i) for i in knobs) if k}
    else:
        for i in knobs:
            key = knob_target(i)
            r = auto_resolve(key) if key else None
            if r is None:
                continue
            lane = auto_lanes.setdefault(key, [None] * MAX_STEPS)
            v = r[0]()
            if lane[step] != v:
                if key not in auto["writing"]:
                    checkpoint("automation", key=("auto", key))
                lane[step] = v
                mark_dirty()
            writing.add(key)
    auto["writing"] = writing
    pstep, pfrac = auto_pos(L, spt, SR * 0.03)       # a frame ahead: hits get their step's value
    auto_play(pstep, pfrac, L, skip=writing)


def auto_pos(L, spt, ahead):
    """(step, fraction) of the playhead `ahead` samples from now."""
    pos = clock["tick"] + 1 - (clock["next_tick"] - clock["sample"] - ahead) / spt
    st = math.floor(pos)
    return st % L, max(0.0, min(1.0, pos - st))


def auto_play(step, frac, L, skip=()):
    """Set every automated setting to its lane's value here. True if anything moved."""
    moved = False
    for key, lane in list(auto_lanes.items()):
        if key in skip:
            continue
        v = lane_value(lane, step, frac, L)
        r = auto_resolve(key) if v is not None else None
        if r is not None and r[0]() != v:
            r[1](v)
            moved = True
    return moved


HOLD_TO_EDIT = 0.35          # holding a lit step longer than this = editing it: letting go keeps it


def held_long(info):
    return time.monotonic() - info.get("t", time.monotonic()) > HOLD_TO_EDIT


def auto_touch(i, down):
    """A knob touched (Push encoders sense your finger): record / clear its lane."""
    if down:
        for info in list(held_steps.values()) + list(held_note_steps.values()):
            info["edited"] = True                 # reaching for a knob while holding a step = editing
        auto["touched"].add(i)
        if DELETE_BUTTON in held_buttons:
            auto_clear(i)
    else:
        auto["touched"].discard(i)


def auto_clear(i):
    key = knob_target(i)
    if key and key in auto_lanes:
        checkpoint("clear automation")
        auto_lanes.pop(key)
        toast(f"automation cleared: {page_knobs()[i].label} (P{pat['cur'] + 1})")
        mark_dirty()
    elif key:
        toast("no automation on that knob in this pattern")


def auto_has(param):
    t = getattr(param, "target", None)
    key = t() if t else None
    return bool(key and key in auto_lanes and any(v is not None for v in auto_lanes[key]))


def load_lane(points):
    lane = [None] * MAX_STEPS
    for i, v in points:
        if 0 <= int(i) < MAX_STEPS:
            lane[int(i)] = v
    return lane


def copy_lanes(lanes):
    return {k: list(v) for k, v in (lanes or {}).items()}


# ---------------------------------------------------------------------------
# Bounce: exports are rendered by the real engine itself, faster than real time
# (the live output rests meanwhile). So a render is exactly what you hear:
# automation, synths, pad strips, buses, choke, swing, humanize (same random
# rolls for the mix and every stem), the master. Settings are put back after.
# ---------------------------------------------------------------------------
BOUNCE_BLOCK = 512
SONG_TAIL_MAX = 10.0                     # seconds of tail rendered after a song ends


class _NoStatus:
    output_underflow = False


def bounce_skip(note):
    only = audio_state.get("bounce_only")
    return only is not None and note != only


def bounce_progress(text):
    """Keep the Push screen alive with a progress line while rendering."""
    disp = ui.get("display")
    if disp is None or time.monotonic() - ui.get("bounce_drawn", 0) < 0.25:
        return
    ui["bounce_drawn"] = time.monotonic()
    ui["toast"] = (text, time.monotonic())
    try:
        disp.show(frame())
    except Exception:
        pass


def bounce(kind, only=None, master_on=True, label="rendering"):
    """kind 'loop': the current pattern as a seamless loop (rendered twice, the second
    pass kept, so tails from the end ring into the start like when it loops).
    kind 'song': the song from row 1 to its end plus its tail. Returns (audio, T) with
    T = where the music itself ends (song) or the loop length."""
    global voices, bus_engine, pad_engine, master_bus
    store_current()
    snap = snapshot()
    saved = {"voices": voices, "bus": bus_engine, "pad": pad_engine, "master": master_bus,
             "clock": dict(clock), "song": dict(song), "page": seq["page"], "rnd": random.getstate(),
             "choruses": dict(choruses), "queued": pat["queued"], "volume": master["volume"]}
    audio_state.update(bounce_tid=threading.get_ident(), bounce_only=only, bounce_master=master_on)
    try:
        voices = []
        repeat["held"].clear()
        choruses.clear()
        strip_in.clear()
        bus_engine, pad_engine = BusEngine(), PadEngine()
        try:
            master_bus = MasterBus() if master_on else None
        except ImportError:
            master_bus = None
        random.seed(20261124)                     # the same rolls for the mix and each stem
        pat["queued"] = None
        if kind == "song":
            song.update(mode="SONG", end="STOP")
            start_playing()
            song["pos"], song["rep"] = 0, 0
            if song["rows"][0][0] != pat["cur"]:
                switch_pattern(song["rows"][0][0])
        else:
            song["mode"] = "PATTERN"
            start_playing()
        spt = SR * 60.0 / clock["bpm"] / 4
        loop_len = int(round(pattern_length() * spt))
        bus_engine.sync()
        pad_engine.sync()
        chunks, total, T = [], 0, None
        limit = SR * 60 * 30                      # 30 minutes: something went wrong
        while total < limit:
            if clock["playing"] and clock["tick"] >= 0:
                L = pattern_length()
                if auto_play(*auto_pos(L, spt, BOUNCE_BLOCK), L):   # where this block ends
                    bus_engine.sync()
                    pad_engine.sync()
            out = np.zeros((BOUNCE_BLOCK, 2), np.float32)
            _audio_callback(out, BOUNCE_BLOCK, None, _NoStatus)
            chunks.append(out)
            total += BOUNCE_BLOCK
            if kind == "loop":
                if total >= 2 * loop_len:
                    break
                bounce_progress(f"{label}  {min(99, int(total / (2 * loop_len) * 100))}%")
            else:
                if T is None and not clock["playing"]:
                    T = clock["sample"]           # the last loop of the last row just ended
                if T is not None:
                    tail = np.abs(np.concatenate(chunks[-int(SR / BOUNCE_BLOCK):])).max() \
                        if len(chunks) * BOUNCE_BLOCK > SR else 1.0
                    if total - T > SR and tail < 1e-4 or total - T >= SONG_TAIL_MAX * SR:
                        break
                bounce_progress(f"{label}  {total / SR:.0f} s")
        audio = np.concatenate(chunks) if chunks else np.zeros((0, 2), np.float32)
        if kind == "loop":
            return audio[loop_len:2 * loop_len], loop_len
        return audio, (T if T is not None else total)
    finally:
        voices = saved["voices"]
        bus_engine, pad_engine, master_bus = saved["bus"], saved["pad"], saved["master"]
        restore(snap)                              # automation moved settings: put them back
        clock.update(saved["clock"])
        song.update(saved["song"])
        seq["page"] = saved["page"]
        pat["queued"] = saved["queued"]
        master["volume"] = saved["volume"]
        choruses.clear()
        choruses.update(saved["choruses"])
        strip_in.clear()
        random.setstate(saved["rnd"])
        audio_state.update(bounce_tid=None, bounce_only=None, bounce_master=False)
        ui["toast"] = None
        if bus_engine is not None:                 # live engines: push the restored settings again
            bus_engine.pkeys.clear()
        if pad_engine is not None:
            pad_engine.pkeys.clear()


def song_pads():
    hits, _ = song_hits()
    return sorted({h[1] for h in hits})


# ---------------------------------------------------------------------------
# Display calibration (Shift + Setup). The Push's screen and its button LEDs don't
# show colours the same way, and this screen tints dark greys. Here you match them
# by eye, once; the result is kept in display_calibration.json (all projects).
#   GREY (lower button 1): knobs 1-3 = red / green / blue in the darks (gamma),
#     knobs 5-7 = red / green / blue in the brights (gain), until the ramp is grey.
#   COLOURS (press a page button): knobs 1-3 = red / green / blue of that page's tab
#     on screen, until it matches the button's light. Every tab is shown lit.
#   Delete: back to the default for what you're editing · Shift + Setup: save, leave.
# The grey correction is applied to the whole frame just before it goes to the Push
# (screenshots stay uncorrected).
# ---------------------------------------------------------------------------
CAL_FILE = os.path.join(APP_DIR, "display_calibration.json")
calib = {"gamma": [100, 100, 100], "gain": [100, 100, 100], "tabs": {}}
cal_ui = {"on": False, "step": "GREY", "slot": 0}
_cal_lut = {"key": None, "lut": None}


def load_calibration():
    try:
        with open(CAL_FILE) as f:
            data = json.load(f)
        calib["gamma"] = [int(x) for x in data.get("gamma", calib["gamma"])][:3]
        calib["gain"] = [int(x) for x in data.get("gain", calib["gain"])][:3]
        calib["tabs"] = {k: tuple(int(c) for c in v) for k, v in data.get("tabs", {}).items()}
    except (OSError, ValueError, TypeError):
        pass
    apply_card_colours()


def apply_card_colours():
    """The PAD and PATTERN status cards use the calibrated PAD / PATTERN tab colours."""
    global RED, BLUE
    if "PAD" in calib["tabs"]:
        RED = calib["tabs"]["PAD"]
    if "PATTERN" in calib["tabs"]:
        BLUE = calib["tabs"]["PATTERN"]


def save_calibration():
    try:
        with open(CAL_FILE, "w") as f:
            json.dump({"gamma": calib["gamma"], "gain": calib["gain"],
                       "tabs": {k: list(v) for k, v in calib["tabs"].items()}}, f, indent=1)
    except OSError:
        pass


def display_correct(img):
    """The calibrated frame for the Push: per-channel darks (gamma) and brights (gain)."""
    key = (tuple(calib["gamma"]), tuple(calib["gain"]))
    if key == ((100, 100, 100), (100, 100, 100)):
        return img
    if _cal_lut["key"] != key:
        x = np.arange(256) / 255.0
        _cal_lut["lut"] = [np.clip(np.round(255 * calib["gain"][c] / 100 * x ** (calib["gamma"][c] / 100)),
                                   0, 255).astype(np.uint8) for c in range(3)]
        _cal_lut["key"] = key
    lut = _cal_lut["lut"]
    return np.stack([lut[c][img[:, :, c]] for c in range(3)], axis=2)


def tab_colour(name):
    return calib["tabs"].get(name) or PAGES[name].get("colour", NAV)


def cal_slots():
    return [n for n in PAGE_ORDER if n is not None]


def cal_toggle():
    cal_ui["on"] = not cal_ui["on"]
    if not cal_ui["on"]:
        save_calibration()
        apply_card_colours()
        toast("display calibration saved")
    else:
        ui["settings"] = None
        cal_ui["step"] = "GREY"


def cal_knob(i, v):
    d = v if v < 64 else v - 128
    if cal_ui["step"] == "GREY":
        if i in (0, 1, 2):
            calib["gamma"][i] = int(clamp(calib["gamma"][i] + d, 50, 200))
        elif i in (4, 5, 6):
            calib["gain"][i - 4] = int(clamp(calib["gain"][i - 4] + d, 50, 100))
    elif i in (0, 1, 2):
        name = cal_ui["slot"]
        c = list(tab_colour(name))
        c[i] = int(clamp(c[i] + 2 * d, 0, 255))
        calib["tabs"][name] = tuple(c)


def cal_button(c):
    """Buttons while calibrating. True = handled."""
    if c in UPPER_BUTTONS:
        name = PAGE_ORDER[UPPER_BUTTONS.index(c)]
        if name is not None:
            cal_ui.update(step="COLOURS", slot=name)
        return True
    if c in LOWER_BUTTONS:
        if LOWER_BUTTONS.index(c) == 0:
            cal_ui["step"] = "GREY"
        return True
    if c == DELETE_BUTTON:
        if cal_ui["step"] == "GREY":
            calib["gamma"], calib["gain"] = [100, 100, 100], [100, 100, 100]
        else:
            calib["tabs"].pop(cal_ui["slot"], None)
        toast("back to the default")
        return True
    return False


def draw_calibration(d):
    # every tab lit in its colour: compare each with the button above it
    for slot, name in enumerate(PAGE_ORDER):
        x0, x1 = slot * SLOT_W + 3, slot * SLOT_W + SLOT_W - 3
        if name is None:
            d.rectangle([x0, 1, x1, 16], fill=EMPTY)
            continue
        col = tab_colour(name)
        sel = cal_ui["step"] == "COLOURS" and cal_ui["slot"] == name
        d.rectangle([x0, 1, x1, 16], fill=col, outline=SEL_OUTLINE if sel else None)
        d.text(((x0 + x1) / 2, 9), PAGE_LABELS.get(name, name), font=F_TAB, anchor="mm",
               fill=BG if sum(col) / 3 > 170 else TXT)
    if cal_ui["step"] == "GREY":
        for k, lab in enumerate(("DARK R", "DARK G", "DARK B")):
            param_cell(d, k, lab, f"{calib['gamma'][k] / 100:.2f}", "", fill=((255, 90, 90), MINT, CYAN)[k])
        for k, lab in enumerate(("BRIGHT R", "BRIGHT G", "BRIGHT B")):
            param_cell(d, k + 4, lab, str(calib["gain"][k]), "%", fill=((255, 90, 90), MINT, CYAN)[k])
        empty_cell(d, 3)
        empty_cell(d, 7)
        levels = [8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 224, 248]
        w = (948 - 12) / len(levels)
        for k, lv in enumerate(levels):
            d.rectangle([12 + k * w, 62, 12 + (k + 1) * w - 2, 118], fill=(lv, lv, lv))
            d.text((12 + k * w + w / 2, 124), str(lv), font=F_MONO_S, fill=GREY, anchor="mm")
        d.text((12, 138), "GREY: turn knobs 1-3 (darks) and 5-7 (brights) until every bar looks neutral grey   ·   "
               "page button: match its colour   ·   Delete: reset   ·   Shift + Setup: save", font=F_BODY, fill=GREY,
               anchor="lm")
    else:
        name = cal_ui["slot"]
        col = tab_colour(name)
        for k, lab in enumerate(("RED", "GREEN", "BLUE")):
            param_cell(d, k, lab, str(col[k]), "", fill=((255, 90, 90), MINT, CYAN)[k])
        for k in range(3, 8):
            empty_cell(d, k)
        slot = PAGE_ORDER.index(name)
        d.rectangle([slot * SLOT_W + 3, 60, slot * SLOT_W + SLOT_W - 3, 140], fill=col)    # under its button
        led = LED_COLOURS.get(name, PAGES[name].get("colour"))
        tx = 12 if slot >= 4 else 500
        d.text((tx, 74), f"{PAGE_LABELS.get(name, name)}: match this block to the button's light", font=F_BODY,
               fill=TXT, anchor="lm")
        d.text((tx, 94), f"screen {col[0]}, {col[1]}, {col[2]}   ·   LED {led[0]}, {led[1]}, {led[2]}",
               font=F_MONO, fill=GREY, anchor="lm")
        d.text((tx, 114), "knobs 1-3: red, green, blue   ·   Delete: default", font=F_BODY, fill=GREY, anchor="lm")
        d.text((tx, 134), "lower button 1: grey balance   ·   Shift + Setup: save", font=F_BODY, fill=GREY,
               anchor="lm")


# ---------------------------------------------------------------------------
# SONG mode (Clip button): an ordered list of rows, each = a pattern slot played
# N times. With PLAY on SONG, Play runs the list (from the selected row when the
# SONG page is open, else from row 1); each pattern change happens at the end of
# its loop, like a queued pattern. Convert / Shift + Convert export the song.
# ---------------------------------------------------------------------------
MAX_SONG_ROWS = 64
song = {"rows": [], "sel": 0, "page": 0, "mode": "SONG", "end": "LOOP", "pos": None, "rep": 0}


def song_open():
    return ui["page"] == "SONG" and ui["open"]


def song_on():
    return song["mode"] == "SONG" and bool(song["rows"])


def slot_bars(i):
    if i == pat["cur"]:
        return seq["bars"]
    return slots[i]["bars"] if slots[i] is not None else None


def song_bars():
    return sum((slot_bars(p) or 1) * r for p, r in song["rows"])


def song_start():
    """Play pressed with the song on: start at the selected row (SONG page) or row 1."""
    r = min(song["sel"], len(song["rows"]) - 1) if song_open() else 0
    song["pos"], song["rep"] = r, 0
    p = song["rows"][r][0]
    if p != pat["cur"]:
        switch_pattern(p)
    pat["queued"] = None
    pat["page"] = pat["cur"] // SLOTS_PER_PAGE


def song_advance():
    """End of a loop while the song plays: repeat, next row, or the end.
    Returns False when the song is over (END = STOP)."""
    rows = song["rows"]
    if not rows or song["pos"] is None:
        song["pos"] = None
        return True
    song["rep"] += 1
    if song["pos"] < len(rows) and song["rep"] < rows[song["pos"]][1]:
        return True
    song["pos"], song["rep"] = song["pos"] + 1, 0
    if song["pos"] >= len(rows):
        if song["end"] == "STOP":
            song["pos"] = None
            return False
        song["pos"] = 0
    p = rows[song["pos"]][0]
    if p != pat["cur"]:
        switch_pattern(p)
    pat["page"] = pat["cur"] // SLOTS_PER_PAGE
    return True


def song_lower_button(k):
    i = song["page"] * 8 + k
    rows = song["rows"]
    if i > len(rows):
        return
    if DELETE_BUTTON in held_buttons:
        if i < len(rows):
            checkpoint(f"delete song row {i + 1}")
            rows.pop(i)
            song["sel"] = max(0, min(song["sel"], len(rows) - 1))
            if song["pos"] is not None and song["pos"] >= len(rows):
                song["pos"] = None if not rows else song["pos"]
            toast(f"row {i + 1} deleted")
            mark_dirty()
        return
    if DUPLICATE_BUTTON in held_buttons:
        ui["dup_used"] = True
        if i < len(rows) and len(rows) < MAX_SONG_ROWS:
            checkpoint(f"duplicate song row {i + 1}")
            rows.insert(i + 1, list(rows[i]))
            song["sel"] = i + 1
            toast(f"row {i + 1} copied to row {i + 2}")
            mark_dirty()
        return
    if i == len(rows):                                   # the first empty one: a new row
        if len(rows) >= MAX_SONG_ROWS:
            toast(f"{MAX_SONG_ROWS} rows max")
            return
        checkpoint("add song row")
        p = rows[-1][0] if rows else pat["cur"]
        rows.append([p, 1])
        song["mode"] = "SONG"
        toast(f"row {i + 1}: P{p + 1} x1  ·  knobs 2-3: pattern, repeats")
        mark_dirty()
    song["sel"] = i


def song_knobs():
    def row():
        return song["rows"][song["sel"]] if song["rows"] and song["sel"] < len(song["rows"]) else None

    def put_sel(d):
        song["sel"] = int(clamp(song["sel"] + d, 0, max(0, len(song["rows"]) - 1)))
        song["page"] = song["sel"] // 8

    def put_pat(d):
        r = row()
        if r:
            r[0] = int(clamp(r[0] + d, 0, NUM_SLOTS - 1))

    def put_rep(d):
        r = row()
        if r:
            r[1] = int(clamp(r[1] + d, 1, 64))
    return [Param("ROW", lambda: song["sel"] + 1 if song["rows"] else None, put_sel, per=3,
                  show=lambda v: "-" if v is None else str(v)),
            Param("PATTERN", lambda: row()[0] + 1 if row() else None, put_pat, per=3,
                  show=lambda v: "-" if v is None else f"P{v}"),
            Param("REPEATS", lambda: row()[1] if row() else None, put_rep, per=2,
                  show=lambda v: "-" if v is None else f"x{v}"),
            None, None, None,
            Param("END", lambda: song["end"], lambda d: song.update(end="STOP" if d > 0 else "LOOP"), per=4),
            Param("PLAY", lambda: song["mode"],
                  lambda d: song.update(mode="SONG" if d > 0 else "PATTERN"), per=4)]


def draw_song(d):
    """SONG page: rows as compact arrangement blocks over the lower buttons (8 a page):
    selected = white outline, playing = green with its progress, next = cyan tag.
    Under them, the whole song as a timeline (block widths = bars)."""
    top = strip_height() + 2
    rows = song["rows"]
    playing = clock["playing"] and song["pos"] is not None
    nxt = None
    if playing and rows:
        p_, r_ = song["pos"], song["rep"]
        if p_ < len(rows) and r_ + 1 >= rows[p_][1]:
            nxt = p_ + 1 if p_ + 1 < len(rows) else (0 if song["end"] == "LOOP" else None)
    y0, y1 = top, 112
    for k in range(8):
        i = song["page"] * 8 + k
        x0, x1 = k * SLOT_W + 3, (k + 1) * SLOT_W - 3
        if i < len(rows):
            p, reps = rows[i]
            bars = slot_bars(p)
            now = playing and i == song["pos"]
            sel = i == song["sel"]
            d.rounded_rectangle([x0, y0, x1, y1], 2, fill=(16, 40, 32) if now else PANEL,
                                outline=SEL_OUTLINE if sel else None, width=2 if sel else 1)
            d.rectangle([x0 + 1, y0 + 1, x0 + 3, y1 - 1], fill=MINT if now else (BLUE if bars else DIM))
            d.text((x0 + 10, y0 + 8), f"ROW {i + 1}", font=F_NAV, fill=MINT if now else GREY, anchor="lm")
            if now:
                d.polygon([(x1 - 14, y0 + 4), (x1 - 14, y0 + 12), (x1 - 7, y0 + 8)], fill=MINT)
            elif i == nxt:
                d.text((x1 - 7, y0 + 8), "NEXT", font=F_NAV, fill=CYAN, anchor="rm")
            d.text((x0 + 10, y0 + 31), f"P{p + 1}", font=F_VALUE, fill=TXT if bars else DIM, anchor="ls")
            d.text((x0 + 12 + F_VALUE.getlength(f"P{p + 1}"), y0 + 31), f"×{reps}", font=F_UNIT, fill=GREY,
                   anchor="ls")
            d.text((x0 + 10, y0 + 42), "EMPTY" if bars is None else f"{bars * reps} BAR" + ("" if bars * reps == 1 else "S"),
                   font=F_MONO_S, fill=GREY if bars else DIM, anchor="lm")
            if now:
                L = pattern_length()
                frac = (song["rep"] + max(0, clock["tick"]) % L / L) / reps
                d.rectangle([x0 + 10, y1 - 5, x1 - 8, y1 - 4], fill=(32, 64, 52))
                d.rectangle([x0 + 10, y1 - 5, x0 + 10 + (x1 - x0 - 18) * min(1.0, frac), y1 - 4], fill=MINT)
                if reps > 1:
                    d.text((x1 - 7, y0 + 42), f"{song['rep'] + 1}/{reps}", font=F_MONO_S, fill=TXT, anchor="rm")
        elif i == len(rows):
            d.rounded_rectangle([x0, y0, x1, y1], 2, outline=BORDER)
            d.text(((x0 + x1) / 2, (y0 + y1) / 2 - 6), "+ ADD ROW", font=F_NAV, fill=GREY, anchor="mm")
            d.text(((x0 + x1) / 2, (y0 + y1) / 2 + 8), "PRESS BELOW", font=F_MONO_S, fill=DIM, anchor="mm")
        else:
            d.rounded_rectangle([x0, y0, x1, y1], 2, fill=EMPTY)
    # the whole song as a timeline
    total = max(1, song_bars())
    tx0, tx1, ty = 6, 954, 118
    d.rectangle([tx0, ty, tx1, ty + 7], fill=EMPTY)
    x = tx0
    for i, (p, reps) in enumerate(rows):
        w = (tx1 - tx0) * (slot_bars(p) or 1) * reps / total
        now = playing and i == song["pos"]
        d.rectangle([x, ty, max(x + 1, x + w - 2), ty + 7],
                    fill=MINT if now else (SEL_OUTLINE if i == song["sel"] else (BLUE if slot_bars(p) else DIM)))
        x += w
    secs = song_bars() * 16 * 60.0 / clock["bpm"] / 4
    pages = max(1, len(rows) // 8 + 1)
    d.text((6, 134), f"{len(rows)} ROW" + ("" if len(rows) == 1 else "S") + f"  ·  {song_bars()} BARS  ·  "
           f"{int(secs // 60)}:{int(secs % 60):02d} @ {clock['bpm']} BPM  ·  END {song['end']}  ·  PLAY {song['mode']}"
           + (f"  ·  PAGE {song['page'] + 1}/{pages}" if pages > 1 else ""), font=F_MONO,
           fill=MINT if song_on() else GREY, anchor="lm")
    d.text((6, 150), "DELETE + ROW: REMOVE  ·  DUPLICATE + ROW: COPY AFTER  ·  CONVERT: SONG WAV  ·  "
                     "SHIFT + CONVERT: STEMS", font=F_NAV, fill=DIM, anchor="lm")


def song_hits():
    store_current()
    cur, page = pat["cur"], seq["page"]
    hits, t = [], 0
    spt = SR * 60.0 / clock["bpm"] / 4
    try:
        for p, reps in song["rows"]:
            sl = slots[p]
            if sl is None:
                t += int(round(16 * spt)) * reps          # an empty pattern: one bar of silence
                continue
            apply_slot(sl)
            for _ in range(reps):
                h, L = pattern_hits()
                hits += [(pos + t, n, g, a, st) for pos, n, g, a, st in h]
                t += L
    finally:
        apply_slot(slots[cur])
        seq["page"] = page
    return hits, t


def render_song(only=None, prepared=None):
    hits, T = prepared or song_hits()
    if not hits:
        return None
    longest = max(len(h[3]) for h in hits)
    L = T + min(SR * 10, max(SR * 3, longest))            # room for the last tails
    events = sorted(hits, key=lambda e: e[0])
    out = np.zeros((L, 2), np.float32)
    per_bus, per_pad = {}, {}
    for i, (pos, note, gain, audio, stop) in enumerate(events):
        if only is not None and note != only:
            continue
        end = pos + len(audio)
        if stop is not None:
            end = min(end, pos + stop + FADE)
        group = padp[note]["choke"]
        if group:
            for later, n2, *_ in events[i + 1:]:
                if padp[n2]["choke"] == group and later >= pos:
                    end = min(end, later + FADE)
                    break
        a, b = max(pos, 0), min(end, L)
        if a >= b or bus_muted(note):
            continue
        chunk = audio[a - pos:b - pos]
        if padp[note]["mono_sum"]:
            chunk = chunk.mean(axis=1, keepdims=True)
        strip = bool(pad_fx_active(note))
        chunk = chunk * gain if strip else chunk * pad_lr(note, gain)
        if end < pos + len(audio):
            fade = np.clip((end - np.arange(a, b)) / FADE, 0, 1).astype(np.float32)
            chunk = chunk * fade[:, None]
        if strip:
            per_pad.setdefault(note, np.zeros((L, 2), np.float32))[a:b] += chunk
            continue
        bb = pad_bus(note)
        if bb is not None:
            per_bus.setdefault(bb, np.zeros((L, 2), np.float32))[a:b] += chunk
        else:
            out[a:b] += chunk
    mix_pad_strips(per_pad, per_bus, out, L, wrap=False)
    if per_bus:
        out += render_buses(per_bus, L, wrap=False)
    return out


def song_end(audio, T):
    """Where the song's tail has died away (never before the last bar ends)."""
    loud = np.nonzero(np.abs(audio[T:]).max(axis=1) > 10 ** (-80 / 20))[0]
    return T + (int(loud[-1]) + SR // 10 if len(loud) else 0)


def song_ready():
    if not song["rows"]:
        toast("no song yet: press a lower button to add a row")
        return False
    if clock["playing"]:
        toast("stop playback first, then Convert")
        return False
    return True


def write_stems(folder, name, pads, stems, mix, scale):
    import soundfile as sf
    for p in pads:
        nm = clean_name(f"{name} - {pad_name(p)} {os.path.splitext(pad_samples[p][0])[0]}")
        sf.write(os.path.join(folder, nm + ".wav"), stems[p] * scale, SR, subtype="PCM_24")
    sf.write(os.path.join(folder, clean_name(f"{name} - MIX") + ".wav"), mix * scale, SR, subtype="PCM_24")


def export_song(name=None):
    """Convert on the SONG page: the whole song, as you hear it, as one WAV."""
    import soundfile as sf
    if not song_ready():
        return
    name = name or f"{project_label()} SONG {clock['bpm']} BPM"
    if not song_pads():
        toast("nothing to export: the song's patterns are empty")
        return
    mix, T = bounce("song", master_on=True, label="song")
    mix = mix[:song_end(mix, T)]
    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix *= 0.99 / peak
    os.makedirs(export_base(), exist_ok=True)
    path = free_path(os.path.join(export_base(), name + ".wav"))
    sf.write(path, mix, SR, subtype="PCM_24")
    secs = len(mix) / SR
    toast(f"song ({int(secs // 60)}:{int(secs % 60):02d}) saved to {short_path(path)}"
          + ("  (level lowered)" if peak > 0.99 else ""))
    print(f"Song saved: {os.path.abspath(path)}")


def export_song_stems(name=None):
    """Shift + Convert on the SONG page: one WAV per pad + the mix, full song length,
    all starting at bar 1, so they line up in any DAW."""
    if not song_ready():
        return
    name = name or f"{project_label()} SONG {clock['bpm']} BPM"
    pads = song_pads()
    if not pads:
        toast("nothing to export: the song's patterns are empty")
        return
    mix, T = bounce("song", master_on=False, label="song mix")
    n = song_end(mix, T)
    stems = {}
    for k, p in enumerate(pads):
        a, _ = bounce("song", only=p, master_on=False, label=f"stem {k + 1}/{len(pads)}")
        stems[p] = np.pad(a, ((0, max(0, n - len(a))), (0, 0)))[:n]
    mix = mix[:n]
    peak = float(np.abs(mix).max())
    scale = 0.99 / peak if peak > 0.99 else 1.0
    folder = free_path(os.path.join(export_base(), name))
    os.makedirs(folder, exist_ok=True)
    write_stems(folder, name, pads, stems, mix, scale)
    toast(f"song: {len(pads)} stems + mix saved to {short_path(folder)}/" + ("  (level lowered)" if scale < 1 else ""))
    print(f"Song stems saved in {os.path.abspath(folder)}")


# ---------------------------------------------------------------------------
# Chopping (SAMPLE page). Markers are positions in the selected pad's audio;
# slice k runs from marker k to the next marker (or the end).
#   SLICES knob: equal slices · Record: tap pads in time to place markers
#   MARKER / MOVE / FINE: pick a marker and move it · pads: hear slice 1-16
#   New: put the slices on the next empty bank (press twice)
# ---------------------------------------------------------------------------
NEW_BUTTON = 87
MAX_SLICES = 16
chop = {"pid": None, "markers": [0], "cur": 0, "tap": False, "voice": None, "confirm": False,
        "play_from": 0,                      # where in the sample the preview voice started
        "zoom": 1,                           # Up/Down arrows on the SAMPLE page
        "view": None,                        # Left/Right scrolled view start; None = follow the marker
        "auto": 0}                           # AUTO knob: transient sensitivity, 0 = off
CHOP_ZOOMS = [1, 2, 4, 8, 16, 32, 64, 128]


def chop_audio():
    return pad_samples[chop["pid"]][1] if chop["pid"] in pad_samples else np.zeros((1, 2), np.float32)


def hear(audio):
    """Chop previews follow the pad's STEREO/MONO setting."""
    if chop["pid"] in pad_samples and padp[chop["pid"]]["mono_sum"]:
        return np.repeat(audio.mean(axis=1, keepdims=True), 2, axis=1).astype(np.float32)
    return audio


def chop_state():
    """Start fresh whenever a different pad is selected."""
    if chop["pid"] != ui["selected"]:
        if chop["pid"] is not None:                     # remember this pad's markers
            chop_memory[chop["pid"]] = (chop["markers"], chop["cur"], chop["zoom"])
        if chop["tap"]:
            stop_preview()
        markers, cur, zoom = chop_memory.get(ui["selected"], ([0], 0, 1))
        L = len(pad_samples[ui["selected"]][1]) if ui["selected"] in pad_samples else 1
        markers = [m for m in markers if m < L] or [0]          # the sample may have changed
        cur = min(cur, len(markers) - 1)
        chop.update(pid=ui["selected"], markers=list(markers), cur=cur, zoom=zoom, tap=False, auto=0,
                    confirm=False, view=None)
    return chop


chop_memory = {}                       # pad -> markers you set, kept while MPushC runs


def chop_equal(n):
    chop_state()
    n = clamp(n, 1, MAX_SLICES)
    L = len(chop_audio())
    chop.update(markers=[i * L // n for i in range(n)], cur=0, confirm=False, view=None, auto=0)


# ---- Auto: find the hits (transients) in the sample -----------------------
ONSET_HOP, ONSET_WIN = 256, 1024          # 5 ms steps, 21 ms window
MIN_GAP = SR // 20                        # hits closer than 50 ms count as one
onset_cache = {}                          # id(audio) -> [(position, strength)], strongest first


def find_onsets(audio):
    """Spectral flux: how much new energy appears from one 5 ms frame to the next.
    Peaks in it are hits. Returns (position, strength 0..1), strongest first."""
    key = (id(audio), len(audio))
    if key in onset_cache:
        return onset_cache[key]
    mono = audio.mean(axis=1)
    n = 1 + max(0, len(mono) - ONSET_WIN) // ONSET_HOP
    if n < 3:
        onset_cache[key] = []
        return []
    idx = np.arange(ONSET_WIN)[None, :] + ONSET_HOP * np.arange(n)[:, None]
    frames = mono[idx] * np.hanning(ONSET_WIN).astype(np.float32)
    mag = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1)))
    LAG = 3                                  # compare with 15 ms earlier: the whole attack counts
    rise = np.maximum(0, mag[LAG:] - mag[:-LAG])
    hz = np.fft.rfftfreq(ONSET_WIN, 1 / SR)
    # Look for hits in 3 bands separately, so a kick (all low end) counts as much as a
    # snare or a hat (lots of bright bins). Hats weigh a bit less: kicks and snares
    # are usually the cuts you want.
    novelty = np.zeros(n)
    for lo_hz, hi_hz, weight in ((0, 200, 1.0), (200, 3000, 1.0), (3000, SR / 2, 0.7)):
        # average rise per bin: a band with few bins (the lows) counts as much as a wide
        # one, and a band holding only noise stays small
        band = np.concatenate([np.zeros(LAG), rise[:, (hz >= lo_hz) & (hz < hi_hz)].mean(axis=1)])
        local = np.convolve(band, np.ones(40) / 40, mode="same")   # ~0.2 s around each frame
        novelty = np.maximum(novelty, weight * np.maximum(0, band - local))
    if novelty.max() <= 0:
        onset_cache[key] = []
        return []
    novelty /= novelty.max()
    gap = max(1, MIN_GAP // ONSET_HOP)
    peaks_ = []
    for i in range(1, n - 1):
        lo, hi = max(0, i - gap), min(n, i + gap + 1)
        if novelty[i] > 0.02 and novelty[i] == novelty[lo:hi].max():
            peaks_.append((i, float(novelty[i])))
    out = []
    for i, strength in peaks_:
        # the flux only says roughly where (a frame is 21 ms long), so look at the
        # waveform itself: the hit starts where it first gets loud in that area.
        # Cut at the quietest sample just before it: no click, attack kept.
        centre = i * ONSET_HOP + ONSET_WIN // 2
        a, b = max(0, centre - ONSET_WIN), min(len(mono), centre + ONSET_WIN)
        seg = np.abs(mono[a:b])
        if len(seg) == 0 or seg.max() <= 0:
            continue
        before = np.median(seg[:ONSET_HOP]) if len(seg) > ONSET_HOP else 0.0
        rise = before + 0.25 * (seg.max() - before)
        start = a + int(np.argmax(seg >= rise))
        c = max(0, start - SR // 500)                            # 2 ms back
        pos = c + int(np.argmin(np.abs(mono[c:start + 1]))) if start > c else start
        out.append((pos, strength))
    out.sort(key=lambda x: -x[1])
    onset_cache[key] = out
    return out


def chop_auto(sens):
    """AUTO knob: more % = more (weaker) hits become slices. Up to 16 slices."""
    chop_state()
    sens = int(clamp(sens, 0, 100))
    chop.update(auto=sens, confirm=False, view=None, cur=0)
    if sens == 0:
        chop["markers"] = [0]
        return
    need = 1 - sens / 100                       # 5% = only the very strongest hits
    chosen, left_out = [0], 0                   # strongest hits first, up to 16 slices
    for p, st in find_onsets(chop_audio()):
        if st < need:
            break
        if all(abs(p - q) >= MIN_GAP for q in chosen):
            if len(chosen) < MAX_SLICES:
                chosen.append(p)
            else:
                left_out += 1
    if left_out:
        toast(f"16 slices max: {left_out} weaker hit{'s' if left_out > 1 else ''} left out")
    markers = sorted(chosen)
    chop["markers"] = markers


def chop_move(amount):
    chop_state()
    m, i = chop["markers"], chop["cur"]
    lo = m[i - 1] + SR // 100 if i > 0 else 0                     # keep slices at least 10 ms
    hi = m[i + 1] - SR // 100 if i + 1 < len(m) else len(chop_audio()) - SR // 100
    m[i] = int(clamp(m[i] + amount, lo, max(lo, hi)))
    chop.update(confirm=False, view=None)
    play_slice(i)


def slices():
    m, L = chop["markers"], len(chop_audio())
    return [(m[k], m[k + 1] if k + 1 < len(m) else L) for k in range(len(m))]


def play_slice(k):
    if k < len(chop["markers"]):
        a, b = slices()[k]
        stop_preview()
        play_preview(hear(chop_audio()[a:b]))
        chop["voice"], chop["play_from"] = ui["preview"], a


def chop_tap_toggle():
    """Record on the SAMPLE page: the sample plays; every pad tap starts a new slice."""
    chop_state()
    if chop["tap"]:
        chop["tap"] = False
        stop_preview()
        return
    chop.update(markers=[0], cur=0, tap=True, confirm=False, view=None, auto=0)
    stop_preview()
    play_preview(hear(chop_audio()))
    chop["voice"], chop["play_from"] = ui["preview"], 0


def chop_tap():
    v = chop["voice"]
    if not chop["tap"] or v is None:
        return
    pos = v.pos + (time.monotonic() - audio_state["cb_time"] - audio_state["latency"]) * SR
    pos = int(clamp(pos, 0, len(chop_audio()) - 1))
    if (pos - chop["markers"][-1] > SR // 50 and pos < len(chop_audio()) - SR // 50
            and len(chop["markers"]) < MAX_SLICES):
        chop["markers"].append(pos)
        chop["cur"] = len(chop["markers"]) - 1


def chop_apply():
    """New (press twice): slices onto pads 1-n of the first empty bank."""
    chop_state()
    if chop["pid"] is None:
        return
    empty = [b for b in range(NUM_BANKS) if not any(b * 16 <= p < b * 16 + 16 for p in pad_samples)]
    if not empty:
        toast("no empty bank left")
        return
    if not chop["confirm"]:
        chop["confirm"] = True
        toast(f"press New again: {len(chop['markers'])} slices to bank {BANK_NAMES[empty[0]]}")
        return
    target = empty[0]
    src = chop["pid"]
    name, audio = pad_samples[src]
    base = os.path.splitext(name)[0][:10]
    offset = pad_region.get(src, (0, 0))[0]
    used = {padp[p]["choke"] for p in pad_samples}
    group = next((g for g in range(1, CHOKE_GROUPS + 1) if g not in used), CHOKE_GROUPS)
    checkpoint(f"chop {base}")
    for k, (a, b) in enumerate(slices()):
        pid = target * 16 + k
        set_pad_sample(pid, f"{base}.{k + 1}", audio[a:b], pad_files[src], (offset + a, offset + b))
        padp[pid].update(volume=padp[src]["volume"], pan=padp[src]["pan"], mode=padp[src]["mode"],
                         mono_sum=padp[src]["mono_sum"], colour=padp[src]["colour"], choke=group,
                         vary=padp[src].get("vary", 0))
    bank["cur"] = target
    pat["row"] = "banks"
    ui["selected"] = target * 16
    chop["confirm"] = False
    toast(f"{len(chop['markers'])} slices on bank {BANK_NAMES[target]} · choke group {group}")
    mark_dirty()


# ---------------------------------------------------------------------------
# Sampling from the AudioFuse inputs. Shift + Record opens the SAMPLING screen.
#   Record: start (or wait for signal, with THRESH) / stop.  Hit an empty kit
#   pad to choose where the sample goes. Loaded pads still play, so you can
#   sample along with your beat. Shift + Record or a page button: leave.
# Knobs: INPUT (1-2, 3-4 or one mono input), GAIN, THRESH (auto start when the
# signal gets this loud, OFF = start right away), LENGTH (FREE or 1-8 bars at
# the current BPM: stops by itself).
# Monitoring: use the AudioFuse's own direct monitoring (no delay).
# ---------------------------------------------------------------------------
SMP_INPUTS = ["1-2", "3-4", "1", "2", "3", "4"]
SMP_COLS = {"1-2": (0, 1), "3-4": (2, 3), "1": (0, 0), "2": (1, 1), "3": (2, 2), "4": (3, 3)}
SMP_THRESH = ["OFF"] + list(range(-60, -5, 2))       # dB
SMP_LENGTHS = [0, 1, 2, 4, 8]                        # bars, 0 = free
SMP_MAX_SECONDS = 300
SMP_PREROLL = SR // 100                              # keep 10 ms before the signal starts
smp = {"open": False, "stream": None, "channels": 0, "state": "idle",   # idle/waiting/recording/done
       "blocks": [], "count": 0, "target": None, "peak": np.zeros(2), "clip": 0.0,
       "last": None, "error": ""}


def smp_target_default():
    here = [p for p in range(bank["cur"] * 16, bank["cur"] * 16 + 16) if p not in pad_samples]
    anywhere = [p for p in range(NUM_BANKS * 16) if p not in pad_samples]
    return (here or anywhere or [None])[0]


def smp_length_samples():
    bars = setup["smp_length"]
    return int(bars * 16 * samples_per_step()) if bars else SMP_MAX_SECONDS * SR


def input_callback(indata, frames, time_info, status):
    cols = SMP_COLS[setup["smp_input"]]
    if max(cols) >= indata.shape[1]:
        cols = (0, min(1, indata.shape[1] - 1))
    block = indata[:, cols] * (10 ** (setup["smp_gain"] / 20))
    peak = np.abs(block).max(axis=0)
    smp["peak"] = np.maximum(smp["peak"], peak)
    if peak.max() >= 0.999:
        smp["clip"] = time.monotonic()
    state = smp["state"]
    if state == "waiting":
        th = setup["smp_thresh"]
        loud = np.nonzero(np.abs(block).max(axis=1) >= 10 ** (th / 20))[0]
        if len(loud):
            i = loud[0]
            before = smp["last"] if smp["last"] is not None else np.zeros((0, 2), np.float32)
            pre = np.concatenate([before, block[:i]])[-SMP_PREROLL:]
            smp["blocks"] = [pre.copy(), block[i:].copy()]
            smp["count"] = len(pre) + frames - i
            smp["state"] = "recording"
    elif state == "recording":
        smp["blocks"].append(block.copy())
        smp["count"] += frames
        if smp["count"] >= smp_length_samples():
            smp["state"] = "done"
    last = smp["last"] if smp["last"] is not None else np.zeros((0, 2), np.float32)
    smp["last"] = np.concatenate([last, block])[-SMP_PREROLL:]      # the last 10 ms, for the pre-roll


def sampling_open():
    smp.update(open=True, state="idle", blocks=[], count=0, target=smp_target_default(), error="")
    browser.active and browser.toggle()
    try:
        info = sd.query_devices("AudioFuse", "input")
        smp["channels"] = min(4, int(info["max_input_channels"]))
        smp["stream"] = sd.InputStream(samplerate=SR, channels=smp["channels"], device="AudioFuse",
                                       blocksize=BLOCK, dtype="float32", callback=input_callback)
        smp["stream"].start()
    except Exception as err:                               # no input / busy device
        smp["stream"] = None
        smp["error"] = f"no input: {err}"[:60]


def sampling_close():
    if smp["state"] == "recording":
        smp["state"] = "done"
        sampling_tick()                                    # keep what was recorded
    smp.update(open=False, state="idle", blocks=[])
    st = smp["stream"]
    smp["stream"] = None
    if st is not None:
        try:
            st.abort()
            st.close()
        except Exception:
            pass


def sampling_record():
    """Record button on the SAMPLING screen."""
    if smp["stream"] is None:
        toast("no audio input")
        return
    if smp["target"] is None:
        toast("no empty pad left")
        return
    if smp["state"] == "idle":
        smp.update(blocks=[], count=0)
        smp["state"] = "waiting" if setup["smp_thresh"] != "OFF" else "recording"
    elif smp["state"] == "waiting":
        smp["state"] = "idle"                              # never started: cancel
    elif smp["state"] == "recording":
        smp["state"] = "done"


def sampling_tick():
    """Main loop: turn a finished recording into a pad."""
    if smp["state"] != "done":
        return
    smp["state"] = "idle"
    blocks, smp["blocks"] = smp["blocks"], []
    if not blocks:
        return
    audio = np.concatenate(blocks).astype(np.float32)[:smp_length_samples() + SMP_PREROLL]
    if len(audio) < SR // 20:
        toast("too short: nothing saved")
        return
    n = min(96, len(audio) // 4)                           # 2 ms fades: no clicks at the edges
    ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    import soundfile as sf
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    k = 1
    while os.path.exists(os.path.join(SAMPLES_DIR, f"sample {k:02d}.wav")):
        k += 1
    path = os.path.join(SAMPLES_DIR, f"sample {k:02d}.wav")
    sf.write(path, audio, SR, subtype="PCM_24")
    pid = smp["target"]
    checkpoint(f"sample to {pad_name(pid)}")
    set_pad_sample(pid, f"SMP {k:02d}", audio, path)
    auto_detect(pid, apply_root=True)
    padp[pid].update(volume=100, pan=0, swing=None, humanize=None, mode="one-shot", choke=0, mono_sum=False,
                     fx=None, vary=0)
    ui["selected"] = pid
    bank["cur"] = pid // 16
    smp["target"] = smp_target_default()                   # ready for the next one
    toast(f"{len(audio) / SR:.1f} s sampled to pad {pad_name(pid)}")
    mark_dirty()


SMP_KNOBS = [
    Param("INPUT", lambda: setup["smp_input"],
          lambda d: setup.update(smp_input=step_choice(SMP_INPUTS if smp["channels"] >= 4 else ["1-2", "1", "2"],
                                                       setup["smp_input"], d)), per=4),
    Param("GAIN", lambda: setup["smp_gain"], lambda d: setup.update(smp_gain=clamp(setup["smp_gain"] + d, -12, 24)),
          per=2, show=lambda g: f"{g:+d} dB"),
    Param("THRESH", lambda: setup["smp_thresh"], lambda d: setup.update(smp_thresh=step_choice(SMP_THRESH, setup["smp_thresh"], d)),
          per=2, show=lambda t: "OFF" if t == "OFF" else f"{t} dB"),
    Param("LENGTH", lambda: setup["smp_length"], lambda d: setup.update(smp_length=step_choice(SMP_LENGTHS, setup["smp_length"], d)),
          per=4, show=lambda b: "FREE" if b == 0 else f"{b} BAR{'S' if b > 1 else ''}"),
    None, None, None, None]


def sampling_control(c, v):
    """Buttons and knobs while the SAMPLING screen is open. True = handled."""
    if KNOB_FIRST <= c < KNOB_FIRST + 8:
        param = SMP_KNOBS[c - KNOB_FIRST]
        if param is not None:
            param.turn(v)
            mark_dirty()
        return True
    if c == RECORD_BUTTON:
        if v > 0:
            sampling_record()
        return True
    if c in UPPER_BUTTONS and v > 0:
        sampling_close()                                    # a page button leaves sampling
        return False                                        # ...and opens that page
    return False


# ---------------------------------------------------------------------------
# Keyboard modes for the selected pad. Note = CHROMATIC (every note), Scale =
# SCALE (only notes in the key). Bottom 6 rows = keyboard in fourths (like the
# Push), top 2 rows = the pad's steps. Octave Up/Down = octave. Session = back
# to the kit. Accent = 16 LEVELS: the kit 4x4 plays the selected pad at 16
# velocities. Key (root + scale) is per project: Setup > SCALE.
# The pad's own pitch (C3 = as tuned on the PAD page) sits on the C3 key.
# ---------------------------------------------------------------------------
SCALES = {
    "MAJOR": [0, 2, 4, 5, 7, 9, 11], "MINOR": [0, 2, 3, 5, 7, 8, 10],
    "DORIAN": [0, 2, 3, 5, 7, 9, 10], "PHRYGIAN": [0, 1, 3, 5, 7, 8, 10],
    "MIXOLYDIAN": [0, 2, 4, 5, 7, 9, 10], "HARM MINOR": [0, 2, 3, 5, 7, 8, 11],
    "MAJ PENTA": [0, 2, 4, 7, 9], "MIN PENTA": [0, 3, 5, 7, 10],
    "BLUES": [0, 3, 5, 6, 7, 10], "CHROMATIC": list(range(12)),
}
SCALE_NAMES = list(SCALES)
NOTE_NAMES_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KB_ROWS = 6                                   # keyboard rows (the top 2 stay steps)
LEVELS = [round((k + 1) * 127 / 16) for k in range(16)]   # 16 levels: 8 .. 127
kb = {"mode": None, "octave": 3, "levels": False}         # mode: None / "chromatic" / "scale"
note_cache = {}                               # (pad, audio id, semitones) -> repitched audio


def in_key(n):
    return (n - setup["key_root"]) % 12 in SCALES[setup["key_scale"]]


def kb_note(row, col):
    """MIDI note of a keyboard pad (row 0 = bottom)."""
    base = 12 * (kb["octave"] + 2) + setup["key_root"]
    if kb["mode"] == "chromatic":
        return base + row * 5 + col                   # fourths, like the Push
    sc = SCALES[setup["key_scale"]]
    degree = row * 3 + col                            # in key: each row 3 scale steps up
    return base + 12 * (degree // len(sc)) + sc[degree % len(sc)]


def kb_pad_note(midi_pad):
    row, col = push2.pad_position(midi_pad)
    return kb_note(row, col) if row < KB_ROWS else None


def note_audio(pid, n):
    """The pad's sample played at MIDI note n (C3 = 60 = the pad as tuned)."""
    if is_inst(pid):
        return InstNote(n)
    semis = round(clamp(pad_shift(pid) + n - padp[pid]["root"], -48, 48), 2)
    src = pad_samples[pid][1]
    style, speed = padp[pid].get("style", "classic"), round(modern_speed(pid), 5)
    key = (pid, id(src), semis, style, speed)
    if key not in note_cache:
        if len(note_cache) > 256:
            note_cache.clear()
        if style == "modern":
            try:
                note_cache[key] = stretch_audio(src, speed, semis) if (semis or speed != 1.0) else src
            except ImportError:
                note_cache[key] = repitch(src, semis)
        else:
            note_cache[key] = repitch(src, semis)
    return note_cache[key]


def set_kb_mode(mode):
    kb["mode"] = None if kb["mode"] == mode else mode
    kb["levels"] = False
    ui["settings"] = None
    if kb["mode"] and ui["selected"] not in pad_samples:
        toast("select a pad first")
        kb["mode"] = None


fresh_notes = set()                      # (pad, note event) just recorded, not to be replayed this pass
rec_notes = {}                           # keyboard pad -> (pad, event, raw start) while held


def add_note(pid, ev):
    melody.setdefault(pid, []).append(ev)
    note_audio(pid, ev[1])               # prepare the pitched audio now, not in the audio thread


def replace_note(pid, old, new):
    evs = melody.get(pid, [])
    if old in evs:
        evs[evs.index(old)] = new
    if old in fresh_notes:
        fresh_notes.discard((pid, old))


def record_note_on(pad_midi, pid, n, vel):
    raw = tick_at(now_on_clock())
    if round(raw) < 0:
        return                                           # well inside the count-in: too early
    # a hair before bar 1 (end of the count-in) counts as an early bar 1, like drum hits
    L = pattern_length()
    pos = round(quantize_pos(raw, pid) % L, 4)
    ev = (pos, n, max(1, min(127, int(vel))), 1.0)
    add_note(pid, ev)
    if int(pos) > audio_state["cb_tick"] % L:              # its step hasn't played yet this pass
        fresh_notes.add((pid, ev))
    rec_notes[pad_midi] = (pid, ev, raw)
    mark_dirty()


def record_note_off(pad_midi):
    item = rec_notes.pop(pad_midi, None)
    if item is None:
        return
    pid, ev, raw = item
    length = tick_at(now_on_clock()) - raw
    length = round(max(NOTE_MIN_LEN, min(length, pattern_length())), 3)
    new = (ev[0], ev[1], ev[2], length)
    was_fresh = (pid, ev) in fresh_notes
    replace_note(pid, ev, new)
    if was_fresh:
        fresh_notes.add((pid, new))


def notes_at_step(pid, s):
    return [e for e in melody.get(pid, []) if int(round(e[0])) % pattern_length() == s]


def edit_held_notes(knob, amount):
    """Holding a top-row step in a keyboard mode: 1 velocity, 2 nudge, 3 note, 4 length."""
    checkpoint("edit notes", key=("noteedit", knob, tuple(sorted(held_note_steps))))
    for info in held_note_steps.values():
        info["edited"] = True
        pid, s = info["pad"], info["s"]
        for ev in notes_at_step(pid, s):
            pos, n, vel, length = ev
            if knob == 0:
                vel = clamp(vel + amount, 1, 127)
            elif knob == 1:
                pos = round(clamp(pos + amount * NUDGE_PER_TICK, s - MAX_NUDGE, s + MAX_NUDGE) % pattern_length(), 4)
            elif knob == 2:
                n = clamp(n + amount, 0, 127)
                if kb["mode"] == "scale":                    # stay in key
                    while not in_key(n) and 0 < n < 127:
                        n += 1 if amount > 0 else -1
            elif knob == 3:
                length = round(clamp(length + amount * 0.25, NOTE_MIN_LEN, pattern_length()), 3)
            replace_note(pid, ev, (pos, n, vel, length))
            note_audio(pid, n)
    mark_dirty()


held_note_steps = {}                    # top-row pad -> {pad, s, existed, edited} (keyboard modes)
live_voices = {}                         # pad note -> voice you are playing by hand
live_midi = {}                           # pad note -> MIDI note it played (mono synths)


def gate_release(note, pid):
    """Letting go of a pad: GATE pads stop (short fade), one-shots keep playing."""
    v = live_voices.pop(note, None)
    n = live_midi.pop(note, None)
    if isinstance(v, SynthVoice) and n is not None and v.want != n:
        return                                              # mono: a newer key is playing now
    if v is not None and v.note is not None and padp[v.note]["mode"] == "gate" and v.fade_left is None:
        v.fade_left = FADE
    chopping = chop_view()
    if (chopping and not chop["tap"] and padp[ui["selected"]]["mode"] == "gate"
            and chop.get("held") == note):
        stop_preview()                       # chop view: slices follow the pad's mode


# ---------------------------------------------------------------------------
# Resample (Convert): the current pattern, as you hear it, becomes a sample on
# the first empty pad. Rendered offline (exact, faster than real time) as a
# seamless loop: tails ringing past the end continue at the start, the way they
# do when the pattern loops. Mutes/solos, volume, pan, MONO, choke, swing,
# humanize and nudges are all included; the metronome is not.
# ---------------------------------------------------------------------------
def pattern_hits():
    """Every hit of the current pattern: (position in the loop, pad, velocity gain).
    Humanize is rolled here once, so stems and the mix get the very same hits."""
    spt = SR * 60.0 / clock["bpm"] / 4
    steps_n = pattern_length()
    L = int(round(steps_n * spt))
    hits = []                                       # (position in the loop, pad, velocity gain)
    for note, steps in pattern.items():
        if note not in pad_samples or not audible(note):
            continue
        sw = padp[note]["swing"] if padp[note]["swing"] is not None else groove["swing"]
        hu = (padp[note]["humanize"] if padp[note]["humanize"] is not None else groove["humanize"]) / 100
        for st in range(steps_n):
            if steps[st] and not chance_roll(note, st):
                continue
            for vel, nudge in step_hits(steps[st]):
                swing_delay = (2 * sw / 100 - 1) * spt if st % 2 == 1 else 0.0
                late = random.uniform(0, hu * HUMANIZE_MAX_MS / 1000 * SR) if hu else 0.0
                gain = vel / 127 * (1 - random.uniform(0, hu * HUMANIZE_MAX_GAIN))
                aud = inst_render(note, 60, int(spt)) if is_inst(note) else vary_pick(note)
                hits.append((int((st + nudge) * spt + swing_delay + late), note, gain, aud, None))
    for pid, evs in melody.items():                 # keyboard notes
        if pid not in pad_samples or not audible(pid):
            continue
        sw = padp[pid]["swing"] if padp[pid]["swing"] is not None else groove["swing"]
        hu = (padp[pid]["humanize"] if padp[pid]["humanize"] is not None else groove["humanize"]) / 100
        for pos, n, vel, length in evs:
            q = int(round(pos)) % steps_n
            swing_delay = (2 * sw / 100 - 1) * spt if q % 2 == 1 else 0.0
            late = random.uniform(0, hu * HUMANIZE_MAX_MS / 1000 * SR) if hu else 0.0
            gain = vel / 127 * (1 - random.uniform(0, hu * HUMANIZE_MAX_GAIN))
            stop = int(length * spt) if padp[pid]["mode"] == "gate" else None
            aud = note_audio(pid, n)
            if isinstance(aud, InstNote):                   # a synth note: render it (with its tail)
                aud, stop = inst_render(pid, n, int(length * spt)), None
            hits.append((int(int(pos) * spt) + int((pos - int(pos)) * spt + swing_delay + late),
                         pid, gain, aud, stop))
    return hits, L


def render_pattern(only=None, prepared=None):
    """The pattern as one seamless loop. only = a pad: just that pad (a stem); other
    pads still play silently so their choke groups cut it exactly as in the mix."""
    hits, L = prepared or pattern_hits()
    if not hits:
        return None
    longest = max(len(h[3]) for h in hits)
    loops = 1 + math.ceil(longest / L)              # earlier loops whose tails reach this one
    # stable sort by time: hits at the same moment keep the engine's order (drums, then notes)
    events = sorted(((pos + k * L, n, g, a, st) for k in range(-loops, 2) for pos, n, g, a, st in hits),
                    key=lambda e: e[0])
    out = np.zeros((L, 2), np.float32)
    per_bus, per_pad = {}, {}
    for i, (pos, note, gain, audio, stop) in enumerate(events):
        if only is not None and note != only:
            continue
        end = pos + len(audio)
        if stop is not None:                        # note length (GATE pads): then a 2 ms fade
            end = min(end, pos + stop + FADE)
        group = padp[note]["choke"]
        if group:                                   # cut by the next hit in the same choke group
            for later, n2, *_ in events[i + 1:]:
                if padp[n2]["choke"] == group and later >= pos:
                    end = min(end, later + FADE)
                    break
        a, b = max(pos, 0), min(end, L)             # the part that falls inside this loop
        if a >= b:
            continue
        chunk = audio[a - pos:b - pos]
        if padp[note]["mono_sum"]:
            chunk = chunk.mean(axis=1, keepdims=True)
        strip = bool(pad_fx_active(note))
        chunk = chunk * gain if strip else chunk * pad_lr(note, gain)
        if end < pos + len(audio):                  # choked: same 2 ms fade as live
            fade = np.clip((end - np.arange(a, b)) / FADE, 0, 1).astype(np.float32)
            chunk = chunk * fade[:, None]
        if bus_muted(note):
            continue
        if strip:                                   # pad FX: the pad's sound together, below
            per_pad.setdefault(note, np.zeros((L, 2), np.float32))[a:b] += chunk
            continue
        bb = pad_bus(note)
        if bb is not None:
            per_bus.setdefault(bb, np.zeros((L, 2), np.float32))[a:b] += chunk
        else:
            out[a:b] += chunk
    mix_pad_strips(per_pad, per_bus, out, L, wrap=True)
    if per_bus:
        out += render_buses(per_bus, L)
    return out


def mix_pad_strips(per_pad, per_bus, out, L, wrap):
    """Offline: pads with FX through their strip, then volume / pan, then their bus."""
    for note, buf in per_pad.items():
        y = render_pad_strip(note, buf, wrap=wrap) * pad_lr(note, 1.0)
        bb = pad_bus(note)
        if bb is not None:
            per_bus.setdefault(bb, np.zeros((L, 2), np.float32))[:] += y
        else:
            out += y


def resample_target():
    """Pad to resample: a kit pad held down, or the selected pad on the SAMPLE page.
    None = resample the pattern."""
    if not kb["mode"]:
        for n in held:
            pid = note_to_pid(n)
            if pid in pad_samples:
                return pid
    return ui["selected"] if sample_page() else None


def render_pad(pid):
    """The pad as you hear it: trim, PITCH/TUNE, stretch (classic or modern),
    volume, pan and CHANNELS printed into the audio."""
    audio = pitched[pid]
    if padp[pid]["mono_sum"]:
        audio = np.repeat(audio.mean(axis=1, keepdims=True), 2, axis=1)
    audio = render_pad_strip(pid, audio, wrap=False)        # its FILTER > EQ > COMP
    return (audio * pad_lr(pid, 1.0)).astype(np.float32)


def resample_pad(pid):
    import soundfile as sf
    if is_inst(pid):
        toast("a synth pad: resample the pattern instead (Convert)")
        return
    empty = [p for p in range(bank["cur"] * 16, bank["cur"] * 16 + 16) if p not in pad_samples] or \
            [p for p in range(NUM_BANKS * 16) if p not in pad_samples]
    if not empty:
        toast("no empty pad left")
        return
    if pid in retune_jobs or pitched_shift.get(pid) != render_key(pid):
        toast("still rendering the new pitch / stretch: try again in a second")
        return
    audio = render_pad(pid)
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 0.99:
        audio *= 0.99 / peak
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    k = 1
    while os.path.exists(os.path.join(SAMPLES_DIR, f"resample {k:02d}.wav")):
        k += 1
    path = os.path.join(SAMPLES_DIR, f"resample {k:02d}.wav")
    sf.write(path, audio, SR, subtype="PCM_24")
    new = empty[0]
    checkpoint(f"resample {pad_label(pid)} to {pad_label(new)}")
    # what it was, moved by the pitch / stretch that is now printed in
    shift = pad_shift(pid)
    bpm = pad_bpm(pid)
    old = detected.get(pid, {})
    src_len = len(pad_samples[pid][1])
    set_pad_sample(new, f"RS {k:02d} {pad_name(pid)}", audio, path)
    keep = {k_: padp[pid][k_] for k_ in ("mode", "colour")}
    padp[new].update(volume=100, pan=0, mono_sum=False, choke=0, fine=0, stretch=None, style="classic",
                     swing=padp[pid]["swing"], humanize=padp[pid]["humanize"],
                     root=int(clamp(round(padp[pid]["root"] + shift), 0, 127)), **keep)
    padp[new]["loop_bpm"] = round(bpm * src_len / max(1, len(audio)), 2) if bpm else None
    key = old.get("key")
    if key:
        key = ((key[0] + int(round(shift))) % 12,) + tuple(key[1:])
    root = old.get("root")
    detected[new] = {"busy": False, "src": id(audio), "bpm": padp[new]["loop_bpm"], "key": key,
                     "root": root + shift if root is not None else None}
    bank["cur"] = new // 16
    ui["selected"] = new
    note = " (level lowered to avoid clipping)" if peak > 0.99 else ""
    toast(f"pad {pad_name(pid)} resampled to pad {pad_name(new)}{note}")
    mark_dirty()


def resample():
    import soundfile as sf
    pid = resample_target()
    if pid is not None:
        return resample_pad(pid)
    empty = [p for p in range(bank["cur"] * 16, bank["cur"] * 16 + 16) if p not in pad_samples] or \
            [p for p in range(NUM_BANKS * 16) if p not in pad_samples]
    if not empty:
        toast("no empty pad left")
        return
    if not pattern_hits()[0]:
        toast("nothing to resample: the pattern is empty")
        return
    audio, _ = bounce("loop", master_on=True, label="resampling")   # as you hear it
    peak = float(np.abs(audio).max())
    if peak > 0.99:                                 # would clip: bring it down
        audio *= 0.99 / peak
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    k = 1
    while os.path.exists(os.path.join(SAMPLES_DIR, f"resample {k:02d}.wav")):
        k += 1
    path = os.path.join(SAMPLES_DIR, f"resample {k:02d}.wav")
    sf.write(path, audio, SR, subtype="PCM_24")
    pid = empty[0]
    checkpoint(f"resample to {pad_label(pid)}")
    set_pad_sample(pid, f"RS {k:02d} P{pat['cur'] + 1}", audio, path)
    padp[pid].update(volume=100, pan=0, swing=None, humanize=None, mode="one-shot", choke=0, mono_sum=False,
                     fx=None, vary=0)
    bank["cur"] = pid // 16
    ui["selected"] = pid
    note = " (level lowered to avoid clipping)" if peak > 0.99 else ""
    toast(f"pattern {pat['cur'] + 1} resampled to pad {pad_name(pid)}{note}")
    mark_dirty()


# ---------------------------------------------------------------------------
# Move / copy a pad: hold Duplicate, press the pad, then the destination pad, then
# MOVE or COPY (upper buttons). MOVE takes the pad's steps along, so the beat still
# sounds the same; onto a loaded pad it swaps the two. COPY = sample + settings
# only, replacing what the destination had (its steps stay).
# ---------------------------------------------------------------------------
PAD_DICTS = lambda: (pad_samples, pad_files, pad_region, pitched, pitched_shift, wave_peaks,
                     detected, chop_memory)


def swap_entry(dct, a, b):
    va, vb = dct.pop(a, None), dct.pop(b, None)
    if va is not None:
        dct[b] = va
    if vb is not None:
        dct[a] = vb


def move_pad(a, b):
    checkpoint(f"move {pad_label(a)} to {pad_name(b)}")
    for dct in PAD_DICTS():
        swap_entry(dct, a, b)
    tune[a], tune[b] = tune[b], tune[a]
    pa, pb = dict(padp[a]), dict(padp[b])
    padp[a].clear(); padp[a].update(pb)
    padp[b].clear(); padp[b].update(pa)
    pattern[a][:], pattern[b][:] = list(pattern[b]), list(pattern[a])
    swap_entry(melody, a, b)
    swap_entry(auto_lanes, f"chance:{a}", f"chance:{b}")
    for sl in slots:
        if sl is not None:
            swap_entry(sl.setdefault("auto", {}), f"chance:{a}", f"chance:{b}")
            swap_entry(sl["steps"], a, b)
            swap_entry(sl.setdefault("notes", {}), a, b)
    for st in (mutes, solos):
        ia, ib = a in st, b in st
        st.discard(a); st.discard(b)
        if ia: st.add(b)
        if ib: st.add(a)
    if stem_job.get("pid") in (a, b):
        stem_job["pid"] = b if stem_job["pid"] == a else a
    after_pad_change(a, b)


def copy_pad(a, b):
    checkpoint(f"copy {pad_label(a)} to {pad_name(b)}")
    for dct in PAD_DICTS():
        if a in dct:
            dct[b] = dict(dct[a]) if isinstance(dct[a], dict) else (
                list(dct[a]) if isinstance(dct[a], list) else dct[a])
        else:
            dct.pop(b, None)
    tune[b] = tune[a]
    padp[b].clear(); padp[b].update(dict(padp[a]))
    padp[b]["fx"] = copy_pad_fx(padp[a].get("fx"))
    padp[b]["ip"] = dict(padp[a]["ip"]) if padp[a].get("ip") else None
    after_pad_change(a, b)


def after_pad_change(a, b):
    for p in (a, b):
        retune_jobs.pop(p, None)
        last_voice.pop(p, None)
        if p in pad_samples:
            retune(p)
    note_cache.clear()
    refresh_kit_notes()
    bank["cur"] = b // 16
    ui["selected"] = b
    mark_dirty()


def pad_move_press(pid):
    """A kit pad pressed while Duplicate is held."""
    ui["dup_used"] = True
    src = ui.get("dup_src")
    if src is None:
        if pid not in pad_samples:
            toast(f"pad {pad_name(pid)} is empty: hold Duplicate, press a loaded pad")
            return
        ui["dup_src"] = pid
        toast(f"{pad_label(pid)}: now press where it goes")
    elif pid != src:
        ui["dup_src"] = None
        ui["pad_move"] = (src, pid)
        ui["toast"] = None


def pad_move_choice(i):
    """Upper button i while the MOVE / COPY question is up."""
    a, b = ui.pop("pad_move")
    if a not in pad_samples:
        return
    if i == 0:
        swap = b in pad_samples
        move_pad(a, b)
        toast(f"swapped pads {pad_name(a)} and {pad_name(b)}" if swap else f"moved to pad {pad_name(b)}")
    elif i == 1:
        copy_pad(a, b)
        toast(f"copied to pad {pad_name(b)}")


def draw_pad_move(d):
    a, b = ui["pad_move"]
    d.rectangle([0, 0, W, 20], fill=BLACK)
    for slot, (label, col) in enumerate((("MOVE", TXT), ("COPY", TXT))):
        caps(d, (slot * SLOT_W + 10, 2), label, col)
        d.rectangle([slot * SLOT_W + 10, 14, (slot + 1) * SLOT_W - 6, 15], fill=RED)
    caps(d, (7 * SLOT_W + 10, 2), "CANCEL", GREY)
    d.rounded_rectangle([140, 26, 820, 150], 8, fill=CARD, outline=RED, width=2)
    caps(d, (156, 36), f"PAD {pad_name(a)}  >  PAD {pad_name(b)}", RED)
    d.text((156, 62), os.path.splitext(pad_samples[a][0])[0][:40], font=F_BODY, fill=TXT)
    if b in pad_samples:
        d.text((156, 88), f"MOVE swaps it with {os.path.splitext(pad_samples[b][0])[0][:28]} (steps too)",
               font=F_BODY, fill=GREY)
        d.text((156, 110), f"COPY replaces {pad_name(b)}'s sample + settings (its steps stay)",
               font=F_BODY, fill=GREY)
    else:
        d.text((156, 88), "MOVE takes its steps along   ·   COPY: sample + settings only",
               font=F_BODY, fill=GREY)
    caps(d, (156, 136), "UNDO TAKES IT BACK", GREY)


EXPORT_DIR = "exports"
app_state = {"export_dir": None,                    # where exports go (all projects); None = the project's exports/
             "pad_bright": 100, "pad_unlit": 45}  # pad LEDs: brightness, and unselected pads (% of it)


# ---------------------------------------------------------------------------
# Names and the export folder are typed / picked on the computer: a small macOS
# dialog comes up (osascript), the Push keeps playing meanwhile. Its answer is
# handled by the main loop (main_jobs).
# ---------------------------------------------------------------------------
main_jobs = []


def run_main_jobs():
    while main_jobs:
        job = main_jobs.pop(0)
        try:
            job()
        except Exception:
            log_error("main job")


def clean_name(t):
    return " ".join("".join(ch for ch in (t or "") if ch not in '/\\:*?"<>|\n\r\t').split())[:80]


def ask_on_computer(kind, prompt, default, then):
    """kind 'text': type a name · 'folder': pick a folder. then(answer) runs in the main
    loop; answer None = cancelled. Not a Mac (no osascript): the default, no question."""
    import shutil
    import subprocess
    if ui.get("asking"):
        toast("answer the question on the computer first")
        return
    if shutil.which("osascript") is None:
        then(default)
        return
    q = lambda t: '"' + str(t).replace("\\", "\\\\").replace('"', '\\"') + '"'
    if kind == "text":
        lines = ["activate", f"text returned of (display dialog {q(prompt)} default answer {q(default)} "
                             f'with title "MPushC" buttons {{"Cancel", "OK"}} default button "OK")']
    else:
        where = f" default location (POSIX file {q(default)})" if default and os.path.isdir(default) else ""
        lines = ["activate", f"POSIX path of (choose folder with prompt {q(prompt)}{where})"]
    ui["asking"] = True
    toast("on the computer: " + prompt.lower())

    def run():
        try:
            r = subprocess.run(["osascript"] + [a for ln in lines for a in ("-e", ln)],
                               capture_output=True, text=True, timeout=900)
            ans = r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            ans = None
        main_jobs.append(lambda: (ui.update(asking=False), then(ans)))
    threading.Thread(target=run, daemon=True).start()


def project_label():
    return proj.get("name") or f"Project {proj['cur']:02d}"


def export_base():
    """Chosen folder: <folder>/<project name>/ · otherwise the project's own exports/."""
    d = app_state.get("export_dir")
    if d and os.path.isdir(d):
        return os.path.join(d, clean_name(project_label()) or f"Project {proj['cur']:02d}")
    return EXPORT_DIR


def free_path(path):
    """path, or 'path 2', 'path 3'... if it's taken (works for files and folders)."""
    root, ext = os.path.splitext(path) if path.endswith(".wav") else (path, "")
    k, out = 2, path
    while os.path.exists(out):
        out, k = f"{root} {k}{ext}", k + 1
    return out


def short_path(path):
    """A short 'saved to ...' text for the toast."""
    d = app_state.get("export_dir")
    if d and os.path.abspath(path).startswith(os.path.abspath(d)):
        return os.path.basename(os.path.normpath(d)) + "/" + os.path.relpath(path, d)
    return os.path.relpath(path)


def request_export(kind):
    """Convert / Shift + Convert: ask for the export's name on the computer, then export."""
    if kind in ("song", "song_stems") and not song_ready():
        return
    if kind == "stems" and not pattern_hits()[0]:
        toast("nothing to export: the pattern is empty")
        return
    what = {"stems": f"P{pat['cur'] + 1}", "song": "SONG", "song_stems": "SONG"}[kind]
    default = f"{project_label()} {what} {clock['bpm']} BPM"
    run = {"stems": export_stems, "song": export_song, "song_stems": export_song_stems}[kind]

    def then(name):
        name = clean_name(name)
        if not name:
            toast("export cancelled")
            return
        if kind != "stems" and clock["playing"]:
            toast("stop playback first, then Convert")
            return
        run(name)
    ask_on_computer("text", "Name this export (stems and mix get this name)", default, then)


def name_project():
    def then(name):
        if name is None:
            toast("cancelled")
            return
        proj["name"] = clean_name(name)
        save_project()
        proj["info"] = {n: project_info(n) for n in range(1, NUM_PROJECTS + 1)}
        toast(f"project {proj['cur']:02d}: {project_label()}")
    ask_on_computer("text", f"Name project {proj['cur']:02d}", project_label(), then)


def choose_export_folder():
    def then(path):
        if not path:
            toast("cancelled")
            return
        app_state["export_dir"] = path.rstrip("/")
        save_app_state()
        toast(f"exports go to {os.path.basename(app_state['export_dir'])}/{clean_name(project_label())}/")
    ask_on_computer("folder", "Where should MPushC save exports?",
                    app_state.get("export_dir") or os.path.expanduser("~/Music"), then)


def export_stems(name=None):
    """Shift + Convert: one WAV per pad in the pattern + the full mix, all the same
    length and seamless loops, so they line up and tile in any DAW."""
    import soundfile as sf
    name = name or f"{project_label()} P{pat['cur'] + 1} {clock['bpm']} BPM"
    hits = pattern_hits()[0]
    if not hits:
        toast("nothing to export: the pattern is empty")
        return
    pads = sorted({h[1] for h in hits})
    mix, _ = bounce("loop", master_on=False, label="mix")
    stems = {}
    for k, n in enumerate(pads):
        stems[n], _ = bounce("loop", only=n, master_on=False, label=f"stem {k + 1}/{len(pads)}")
    peak = float(np.abs(mix).max())
    scale = 0.99 / peak if peak > 0.99 else 1.0      # same scale for all: stems still add up to the mix
    folder = free_path(os.path.join(export_base(), name))
    os.makedirs(folder, exist_ok=True)
    write_stems(folder, name, pads, stems, mix, scale)
    toast(f"{len(pads)} stems + mix saved to {short_path(folder)}/" + ("  (level lowered)" if scale < 1 else ""))
    print(f"Stems saved in {os.path.abspath(folder)}")


# ---------------------------------------------------------------------------
# Detection (Setup > PAD > DETECT): root note of one-shots, BPM and key of loops.
# Plain numpy, runs in the background when you ask for it.
# ---------------------------------------------------------------------------


def detect_root(audio):
    """Pitch of a one-shot (YIN on its steadiest part). Returns MIDI note as a float, or None."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    # 16 kHz is plenty for pitch and 3x faster
    x = mono[: SR * 2][::3].astype(np.float64)
    sr = SR / 3
    start = int(0.03 * sr)                              # skip the attack
    W, H = 2048, 256
    if len(x) < start + W + 512:
        return None
    tau_min, tau_max = int(sr / 1500), int(sr / 35)
    f0s, weights = [], []
    for s in range(start, len(x) - W - tau_max, H):
        fr = x[s:s + W + tau_max]
        e = np.sum(fr[:W] ** 2)
        if e < 1e-6:
            continue
        # YIN difference function via FFT
        n = len(fr)
        F = np.fft.rfft(fr, 2 * n)
        acf = np.fft.irfft(F * np.conj(np.fft.rfft(fr[:W], 2 * n)))[:tau_max + 1]
        sq = np.cumsum(np.concatenate([[0], fr ** 2]))
        energy_tau = sq[np.arange(tau_max + 1) + W] - sq[np.arange(tau_max + 1)]
        d = sq[W] + energy_tau - 2 * acf
        d[0] = 0
        cm = d[1:] * np.arange(1, tau_max + 1) / np.maximum(np.cumsum(d[1:]), 1e-12)
        cm = np.concatenate([[1], cm])
        cand = np.nonzero(cm[tau_min:] < 0.2)[0]
        if not len(cand):
            continue
        t = cand[0] + tau_min
        while t + 1 <= tau_max and cm[t + 1] < cm[t]:
            t += 1                                      # walk down to the dip
        if 1 <= t < tau_max:                            # parabolic refinement
            a, b, c = cm[t - 1], cm[t], cm[t + 1]
            den = a - 2 * b + c
            t = t + (0.5 * (a - c) / den if den else 0)
        f0s.append(sr / t)
        weights.append(e * (1 - min(cm[int(round(t))], 1)))
        if len(f0s) > 120:
            break
    if len(f0s) < 3:
        return None
    m = 69 + 12 * np.log2(np.array(f0s) / 440)
    order = np.argsort(m)
    w = np.array(weights)[order]
    cum = np.cumsum(w) / w.sum()
    return float(m[order][np.searchsorted(cum, 0.5)])   # weighted median


def onset_envelope(audio, hop=256, win=1024):
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    n = 1 + max(0, len(mono) - win) // hop
    if n < 8:
        return None
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    mag = np.log1p(100 * np.abs(np.fft.rfft(mono[idx] * np.hanning(win), axis=1)))
    flux = np.maximum(0, mag[2:] - mag[:-2]).mean(axis=1)
    flux = np.concatenate([[0, 0], flux])
    flux -= np.convolve(flux, np.ones(32) / 32, mode="same")
    return np.maximum(flux, 0)


def detect_bpm(audio, lo=70, hi=140):
    """Tempo of a loop. Autocorrelation of the onset envelope, then snapped to a whole
    number of beats in the loop when that agrees. Returns BPM or None."""
    env = onset_envelope(audio)
    if env is None or len(env) < 200:
        return None
    hop_s = 256 / SR
    env = env - env.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(env, 2 * len(env))) ** 2)[: len(env)]

    def score_of(bpm):
        score = 0
        for k, wgt in ((1, 1.0), (2, 0.5), (4, 0.25), (0.5, 0.5)):   # beat, 2 beats, bar, 8th
            lag = 60 / bpm * k / hop_s
            i = int(lag)
            if i + 1 >= len(ac):
                continue
            score += wgt * (ac[i] + (ac[i + 1] - ac[i]) * (lag - i))
        return score

    best, best_score = None, -1e30
    for bpm in np.arange(lo, hi, 0.25):
        score = score_of(bpm)
        if score > best_score:
            best, best_score = bpm, score
    if best is None:
        return None
    dur = len(audio) / SR
    # Loops are almost always whole bars: among the tempos that make the loop exactly
    # 1, 2, 3, 4... bars long, take the best-scoring one if it's nearly as convincing.
    bar_fits = [60 * 4 * n / dur for n in range(1, 65) if lo <= 60 * 4 * n / dur < hi]
    if bar_fits and best_score > 0:
        fit = max(bar_fits, key=score_of)
        if score_of(fit) >= 0.6 * best_score:
            return round(fit, 2)
    beats = dur * best / 60                             # a loop: a whole number of beats?
    for b in sorted({round(beats), 2 ** round(np.log2(max(beats, 1)))}, key=lambda v: abs(v - beats)):
        if b >= 2:
            exact = 60 * b / dur
            if abs(exact - best) / best < 0.04 and lo <= exact < hi:
                return round(exact, 2)
    return round(float(best), 1)


# Key profiles (Temperley / Krumhansl-style, minor and major)
MAJOR_P = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_P = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def chroma(audio):
    """How much of each pitch class there is. Each note's strength is its fundamental
    plus its harmonics (harmonic summation), so a note's own overtones (its 5th
    from the 3rd harmonic) don't get counted as a separate note."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    x = mono[: SR * 30][::2]
    sr = SR / 2
    W, H = 8192, 4096
    if len(x) < W:
        x = np.pad(x, (0, W - len(x)))
    n = 1 + (len(x) - W) // H
    idx = np.arange(W)[None, :] + H * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * np.hanning(W), axis=1))
    spec = mag.sum(axis=0)                                  # average spectrum
    hz = np.fft.rfftfreq(W, 1 / sr)
    notes = np.arange(36, 97)                               # C1 .. C6 candidates
    f0 = 440 * 2 ** ((notes - 69) / 12)
    sal = np.zeros(len(notes))
    for h in range(1, 7):
        fh = f0 * h
        ok = fh < sr / 2 - 100
        # strongest bin within +/- a quarter tone of the harmonic
        lo = np.searchsorted(hz, fh * 2 ** (-0.5 / 12))
        hi = np.searchsorted(hz, fh * 2 ** (0.5 / 12)) + 1
        peak = np.array([spec[a:b].max() if ok[k] and b > a else 0 for k, (a, b) in enumerate(zip(lo, hi))])
        sal += peak * 0.8 ** (h - 1)
    # a note is only "there" if its fundamental is: keep the salience of real peaks
    lo = np.searchsorted(hz, f0 * 2 ** (-0.5 / 12)); hi = np.searchsorted(hz, f0 * 2 ** (0.5 / 12)) + 1
    fund = np.array([spec[a:b].max() if b > a else 0 for a, b in zip(lo, hi)])
    # remove what lower notes' overtones put on top of higher "notes" (octave, 5th, 3rd...)
    for i in range(len(notes)):
        for h, up in ((2, 12), (3, 19), (4, 24), (5, 28), (6, 31)):
            j = i + up
            if j < len(notes):
                fund[j] = max(0.0, fund[j] - fund[i] / h ** 0.8)
    sal = np.sqrt(sal * fund)
    c = np.zeros(12)
    np.add.at(c, notes % 12, sal)
    return c


def detect_key(audio):
    """Returns (root 0-11, 'MAJOR' or 'MINOR', confidence 0..1) or None."""
    c = chroma(audio)
    if c.sum() <= 0:
        return None
    # loops usually start on the home chord: the first quarter counts a bit extra.
    # That's what splits a minor key from its relative major (same notes).
    head = audio[: max(len(audio) // 4, SR // 2)]
    ch = chroma(head)
    c = c / c.sum() + 0.5 * (ch / ch.sum() if ch.sum() > 0 else 0)
    if c.sum() <= 0:
        return None
    scores = []
    for r in range(12):
        for name, prof in (("MAJOR", MAJOR_P), ("MINOR", MINOR_P)):
            scores.append((np.corrcoef(c, np.roll(prof, r))[0, 1], r, name))
    scores.sort(reverse=True)
    (s1, r, name), (s2, *_rest) = scores[0], scores[1]
    return r, name, float(max(0.0, min(1.0, (s1 - s2) * 5 + s1 / 2)))


detected = {}                                       # pad -> {"busy", "root", "bpm", "key"}


def name_hints(name):
    """Sample packs usually put tempo and key in the name: 'CSV_90_soul_stack_Am.wav',
    'Keys 85bpm F#min'. Returns (bpm or None, (root, scale) or None)."""
    import re
    base = os.path.splitext(os.path.basename(name))[0]
    bpm = None
    m = re.search(r"(?<!\d)(\d{2,3}(?:\.\d)?)\s*bpm", base, re.I) or \
        re.search(r"(?:^|[_\-\s.])(\d{2,3})(?=$|[_\-\s.])", base)
    if m and 55 <= float(m.group(1)) <= 200:
        bpm = float(m.group(1))
    key = None
    tokens = re.split(r"[_\-\s.()\[\]]+", base)
    for tok in reversed(tokens):                        # keys are usually near the end
        k = re.fullmatch(r"([A-Ga-g])([#b♯♭]?)(m|min|minor|maj|major|M)?", tok)
        if not k or (len(tok) == 1 and tok.islower()):
            continue
        root = "C D EF G A B".index(k.group(1).upper())
        root = (root + (1 if k.group(2) in "#♯" and k.group(2) else -1 if k.group(2) in "b♭" and k.group(2) else 0)) % 12
        q = k.group(3) or ""
        scale = "MINOR" if q in ("m", "min", "minor") else "MAJOR"
        if not q and len(tok) == 1:
            continue                                    # a lone capital letter is too risky
        key = (root, scale)
        break
    return bpm, key


def run_detect(pid, apply_root=False):
    audio = pad_samples[pid][1]
    res = {"busy": False, "root": None, "bpm": None, "key": None, "src": id(audio)}
    try:
        if len(audio) > 2 * SR:                         # a loop: BPM + key
            hint_bpm, hint_key = name_hints(pad_files.get(pid, "") or pad_samples[pid][0])
            # (a trimmed part of a loop keeps the loop's tempo and key, so the name still holds)
            res["bpm"] = hint_bpm or detect_bpm(audio)
            res["key"] = (hint_key + (1.0,)) if hint_key else detect_key(audio)
        else:                                           # a one-shot: its note
            res["root"] = detect_root(audio)
    except Exception as err:                            # never take the app down
        print("detect failed:", err)
    detected[pid] = res
    if apply_root and res["root"] is not None:
        ui["detect_apply"] = (pid, int(round(res["root"])))   # main loop sets the pad's ROOT


def auto_detect(pid, apply_root=False):
    """Analyse a pad in the background. apply_root: a freshly loaded sample gets its
    ROOT set (part of loading it). Pads already in a project only show what was found,
    so melodies you recorded keep their pitch."""
    if pid not in pad_samples:
        return
    r = detected.get(pid, {})
    if r.get("busy") or (r.get("src") == id(pad_samples[pid][1]) and not apply_root):
        return
    detected[pid] = {"busy": True, "src": id(pad_samples[pid][1])}
    threading.Thread(target=run_detect, args=(pid, apply_root), daemon=True).start()


def detect_tick():
    """Main loop: set a new sample's ROOT; analyse the selected pad if not done yet."""
    item = ui.pop("detect_apply", None)
    if item is not None:
        pid, root = item
        if pid in pad_samples:
            padp[pid]["root"] = int(clamp(root, 0, 127))
            mark_dirty()
    sel_ = ui["selected"]
    if sel_ in pad_samples and detected.get(sel_, {}).get("src") != id(pad_samples[sel_][1]):
        auto_detect(sel_)
    again = ui.get("refit_after_detect")
    if again is not None and not detected.get(again, {}).get("busy") and detected.get(again):
        ui["refit_after_detect"] = None
        refit(again)
        toast(detect_summary(again) or "nothing detected")


def pad_bpm(pid):
    """The pad's tempo: what you set by hand (BPM knob) or what was detected."""
    return padp[pid].get("loop_bpm") or detected.get(pid, {}).get("bpm")


def set_pad_bpm(pid, d):
    """BPM knob (SAMPLE page): correct the loop's tempo by hand (Shift = fine)."""
    bpm = pad_bpm(pid) or float(clock["bpm"])
    step = 0.05 if SHIFT_BUTTON in held_buttons else 0.5
    bpm = round(clamp(bpm + d * step, 40, 240), 2)
    padp[pid]["loop_bpm"] = bpm
    refit(pid)


def refit(pid):
    """A stretched loop whose tempo changed: fit it again."""
    f = padp[pid].get("stretch")
    if f is None:
        return
    if padp[pid].get("style") == "modern":
        retune(pid)
    else:
        classic_fit(pid, f)


def redetect(pid, d):
    """Settings · PAD > DETECT: analyse again (forgets a tempo set by hand)."""
    if d <= 0 or pid not in pad_samples:
        return
    padp[pid]["loop_bpm"] = None
    detected.pop(pid, None)
    ui["refit_after_detect"] = pid
    auto_detect(pid)
    toast("analysing again…")


def detect_summary(pid):
    """One line for the screen: '92.5 BPM · A MINOR' or 'ROOT A1 -12 CT'."""
    r = detected.get(pid, {})
    if r.get("busy"):
        return "ANALYSING…"
    parts = []
    if pad_bpm(pid):
        parts.append(f"{pad_bpm(pid):g} BPM" + (" (SET)" if padp[pid].get("loop_bpm") and
                                                padp[pid]["loop_bpm"] != r.get("bpm") else ""))
    if r.get("key"):
        parts.append(f"{NOTE_NAMES_12[r['key'][0]]} {r['key'][1]}")
    if r.get("root") is not None:
        cents = round((r["root"] - round(r["root"])) * 100)
        parts.append(f"ROOT {note_name(int(round(r['root'])))}" + (f" {cents:+d} CT" if abs(cents) >= 5 else ""))
    return "  ·  ".join(parts)


def det_text(pid, field):
    r = detected.get(pid, {})
    if r.get("busy"):
        return "…"
    v = r.get(field)
    if v is None:
        return "—"
    if field == "bpm":
        return f"{v:g}"
    if field == "key":
        return f"{NOTE_NAMES_12[v[0]]}{'' if v[1] == 'MAJOR' else 'm'}"          # A = A major, Am = A minor
    return v


STRETCH_CHOICES = [None, 1, 0.5, 2]                 # OFF, as is, half time, double time


tempo_watch = {"bpm": None, "since": 0.0}


def follow_tempo():
    """MODERN stretched loops follow the project BPM (re-rendered once it settles)."""
    if tempo_watch["bpm"] != clock["bpm"]:
        tempo_watch.update(bpm=clock["bpm"], since=time.monotonic())
        return
    if time.monotonic() - tempo_watch["since"] < 0.4:
        return
    for pid in list(pad_samples):
        if padp[pid].get("style") == "modern" and padp[pid].get("stretch"):
            retune(pid)


def stretch_text(pid):
    f = padp[pid].get("stretch")
    bpm = pad_bpm(pid)
    if f is None:
        return "OFF" if bpm else "—"
    label = {0.5: "÷2", 1: "x1", 2: "x2"}[f]
    return f"{label} as {bpm * f:g}" if bpm else label


def stretch_to_bpm(pid, d):
    """Turn through OFF / x1 / ÷2 / x2: the loop is repitched so that, played as if it
    were half / the same / double its tempo, it fits the project BPM (classic: pitch
    moves too). OFF puts PITCH and TUNE back to 0."""
    bpm = pad_bpm(pid)
    if not bpm:
        return
    f = step_choice(STRETCH_CHOICES, padp[pid].get("stretch"), 1 if d > 0 else -1)
    padp[pid]["loop_bpm"] = bpm                            # remembered, so it works after reopening
    if padp[pid].get("style") == "modern":                 # tempo only: pitch stays where it is
        padp[pid]["stretch"] = f
        retune(pid)
        toast("stretch off" if f is None else
              f"as {bpm * f:g} BPM → {clock['bpm']} BPM, pitch unchanged")
        mark_dirty()
        return
    if f is None:
        padp[pid].update(stretch=None, fine=0)
        set_tune(pid, 0)
        toast("stretch off")
        mark_dirty()
        return
    classic_fit(pid, f)


def classic_fit(pid, f):
    bpm = pad_bpm(pid)
    if not bpm:
        return
    semis = 12 * math.log2(clock["bpm"] / (bpm * f))
    whole = int(round(semis))
    if abs(whole) > TUNE_RANGE:
        toast(f"as {bpm * f:g} BPM it's too far from {clock['bpm']} BPM")
        return
    padp[pid].update(stretch=f, fine=int(round((semis - whole) * 100)))
    set_tune(pid, whole)
    toast(f"as {bpm * f:g} BPM → {clock['bpm']} BPM: {whole:+d} st {padp[pid]['fine']:+d} ct")
    mark_dirty()


def set_style(pid, d):
    """STYLE: CLASSIC (speed + pitch together) or MODERN (independent). A stretched
    loop is re-fitted for the new style."""
    style = "modern" if d > 0 else "classic"
    if padp[pid].get("style") == style:
        return
    padp[pid]["style"] = style
    f = padp[pid].get("stretch")
    if f is not None:
        if style == "modern":                            # the classic fit lived in PITCH/TUNE: undo it
            padp[pid]["fine"] = 0
            tune[pid] = 0
        else:
            classic_fit(pid, f)                          # classic: the fit goes into PITCH / TUNE
    note_cache.clear()
    retune(pid)


def set_fine(pid, d):
    padp[pid]["fine"] = int(clamp(padp[pid].get("fine", 0) + d, -50, 50))
    retune(pid)


# ---------------------------------------------------------------------------
# Projects: 16 slots, SP-404 style (projects/01 .. projects/16, each with its own
# project.json, samples/ and exports/). Shift + Select: PROJECTS screen, the kit
# pads are the 16 slots. Hit a pad: open it (an empty slot = a new project).
# Delete + pad twice: delete it (moved to projects/_deleted, never erased).
# Shift + User: screenshot of the Push screen (screenshots/).
# ---------------------------------------------------------------------------
proj = {"open": False, "cur": 1, "info": {}, "confirm_delete": None, "name": ""}
BLANK = {}                                      # the empty music state, taken at start-up


def project_dir(n):
    return os.path.join(PROJECTS_DIR, f"{n:02d}")


def project_info(n):
    """Short summary of a slot for the screen (None = empty)."""
    path = os.path.join(project_dir(n), PROJECT_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {"bpm": "?", "pads": 0, "date": "?"}
    when = time.strftime("%d %b", time.localtime(os.path.getmtime(path)))
    return {"bpm": data.get("bpm", "?"), "pads": len(data.get("pads", {})), "date": when,
            "name": data.get("name", "")}


def migrate_old_layout():
    """Before projects existed, the project lived in the app folder: move it to slot 01."""
    if os.path.isdir(PROJECTS_DIR):
        return False
    os.makedirs(project_dir(1), exist_ok=True)
    moved = False
    for name in (PROJECT_FILE, SAMPLES_DIR, EXPORT_DIR):
        src = os.path.join(APP_DIR, name)
        if os.path.exists(src):
            os.rename(src, os.path.join(project_dir(1), name))
            moved = True
    return moved or None


def open_initial_project():
    first_install = migrate_old_layout() is None and not os.path.exists(os.path.join(project_dir(1), PROJECT_FILE))
    n = 1
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
        n = int(st.get("project", 1))
        app_state["export_dir"] = st.get("export_dir")
        favorites[:] = [f for f in st.get("favorites", []) if isinstance(f, str)]
        for k_, lo_ in (("pad_bright", 10), ("pad_unlit", 0)):
            app_state[k_] = clamp(int(st.get(k_, app_state[k_])), lo_, 100)
    except Exception:
        pass
    BLANK.update(snapshot())
    enter_project(n, first_run=first_install)


def enter_project(n, first_run=False):
    os.makedirs(project_dir(n), exist_ok=True)
    os.chdir(project_dir(n))
    proj["cur"], proj["name"] = n, ""
    load_project(first_run=first_run)
    if ui["selected"] not in pad_samples and kit_notes:
        ui["selected"] = kit_notes[0]                # a pad is always ready to edit
    save_app_state()
    print(f"Project {n:02d}" + (f": {proj['name']}" if proj["name"] else ""))


def save_app_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"project": proj["cur"], "export_dir": app_state.get("export_dir"),
                       "favorites": favorites, "pad_bright": app_state["pad_bright"],
                       "pad_unlit": app_state["pad_unlit"]}, f)
    except OSError:
        pass


def switch_project(n):
    if n == proj["cur"]:
        proj["open"] = False
        return
    clock["playing"] = False
    rec["on"] = False
    voices.clear()
    smp["open"] and sampling_close()
    save_project()
    keep = dict(setup)                               # pad response, click... carry over
    restore(BLANK)
    setup.update(keep)
    for d in (detected, chop_memory, note_cache, source_cache, live_voices, last_voice, pad_flash,
              held_steps, held_note_steps, rec_notes):
        d.clear()
    fresh_notes.clear()
    history["undo"].clear(); history["redo"].clear()
    kb.update(mode=None, levels=False)
    chop.update(pid=None, markers=[0], cur=0, tap=False)
    ui.update(selected=None, settings=None, help=False)
    bank["cur"] = 0
    enter_project(n)
    proj["open"] = False
    save_state["dirty_since"] = None
    info = project_info(n)
    toast(f"project {n:02d}" + ("" if info else ": new"))


def delete_project(n):
    if n == proj["cur"]:
        toast("open another project first")
        return
    if project_info(n) is None:
        return
    if proj["confirm_delete"] != n:
        proj["confirm_delete"] = n
        toast(f"Delete + pad again: delete project {n:02d}")
        return
    proj["confirm_delete"] = None
    trash = os.path.join(PROJECTS_DIR, "_deleted")
    os.makedirs(trash, exist_ok=True)
    os.rename(project_dir(n), os.path.join(trash, f"{n:02d} {time.strftime('%Y-%m-%d %H.%M.%S')}"))
    proj["info"][n] = None
    toast(f"project {n:02d} deleted (kept in projects/_deleted)")


def open_projects():
    proj["open"] = not proj["open"]
    proj["confirm_delete"] = None
    if proj["open"]:
        save_project()
        proj["info"] = {n: project_info(n) for n in range(1, NUM_PROJECTS + 1)}


def silence_all():
    """Stop Clip: every sound fades out (2 ms); the sequencer keeps playing."""
    for vc in list(voices):
        if vc.pos < 0:
            voices.remove(vc)
        elif vc.fade_left is None:
            vc.fade_left = FADE
    repeat["held"].clear()
    toast("silenced")


# ---------------------------------------------------------------------------
# Stem separation (Settings · PAD > STEMS): the selected pad's sample is split
# into drums / bass / other / vocals by stems.py (demucs-mlx) in a separate
# process, then the 4 stems land on empty pads of the bank, in time with the
# original (same PITCH / TUNE / STRETCH). Undoable.
# ---------------------------------------------------------------------------
STEM_ORDER = [("drums", 0), ("bass", 40), ("other", 64), ("vocals", 95)]   # name, colour
stem_job = {"pid": None, "proc": None, "start": 0.0, "dir": None, "log": [], "mode": "fast",
            "choice": "OFF", "go_at": None}


def stems_knob(pid, d):
    """STEMS: turn right = separate (with the QUALITY next to it); turn left = cancel."""
    if d > 0:
        if stem_job["proc"] is None:
            stems_start(pid, setup["stem_q"].lower())
        return
    proc = stem_job["proc"]
    if proc is not None:
        proc.kill()
        stem_job["proc"] = None
        toast("stem separation cancelled")


def stems_start(pid, mode="fast"):
    if pid not in pad_samples:
        return
    if stem_job["proc"] is not None:
        toast("already separating: wait for it to finish")
        return
    import soundfile as sf
    import subprocess
    base = os.path.splitext(pad_samples[pid][0])[0][:24]
    k = 1
    while os.path.exists(os.path.join(SAMPLES_DIR, "stems", f"{base} {k:02d}")):
        k += 1
    out = os.path.join(SAMPLES_DIR, "stems", f"{base} {k:02d}")
    os.makedirs(out, exist_ok=True)
    src = os.path.join(out, "source.wav")
    sf.write(src, pad_samples[pid][1], SR, subtype="FLOAT")    # the pad's audio (a chop = just its part)
    # demucs-mlx lives in the original .venv (Python 3.11); MPushC itself may run on the
    # free-threaded .venv-ft, so the stems worker uses the old one when it's there.
    old = os.path.join(APP_DIR, ".venv", "bin", "python")
    py = old if os.path.exists(old) else sys.executable
    proc = subprocess.Popen([py, os.path.join(APP_DIR, "stems.py"), src, out, mode],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stem_job.update(pid=pid, proc=proc, start=time.monotonic(), dir=out, log=[], mode=mode)
    threading.Thread(target=lambda: stem_job["log"].extend(proc.stdout), daemon=True).start()
    toast(f"separating {base} ({mode})… keep playing, it runs in the background")


def consecutive_empty(n, near):
    """n empty pads in a row: in `near`'s bank (after it first), else the next banks."""
    b0 = near // 16 if near is not None else bank["cur"]
    for k in range(NUM_BANKS):
        b = (b0 + k) % NUM_BANKS
        starts = list(range(b * 16, b * 16 + 17 - n))
        if k == 0 and near is not None:                 # after the source pad first
            starts = [p for p in starts if p > near] + [p for p in starts if p <= near]
        for p in starts:
            if all(q not in pad_samples for q in range(p, p + n)):
                return list(range(p, p + n))
    return []


def stems_tick():
    """Main loop: start a settled FAST / BEST choice; when the worker is done, put the stems on pads."""
    proc = stem_job["proc"]
    if proc is None or proc.poll() is None:
        return
    stem_job["proc"] = None
    time.sleep(0.05)                                            # let the log thread finish reading
    log = [l.strip() for l in stem_job["log"]]
    if not log or log[-1] != "DONE":
        err = next((l for l in reversed(log) if l.startswith("ERROR")), log[-1] if log else "no output")
        toast(err[:70])
        print("stems.py said:", *log[-5:], sep="\n  ")
        return
    src_pid, out = stem_job["pid"], stem_job["dir"]
    free = consecutive_empty(4, src_pid)
    if not free:
        toast("stems saved in samples/stems, but no bank has 4 empty pads in a row")
        return
    checkpoint("stems")
    base = os.path.splitext(pad_samples[src_pid][0])[0][:12] if src_pid in pad_samples else "stem"
    keep = {k: padp[src_pid][k] for k in ("volume", "pan", "mode", "mono_sum", "fine", "stretch", "root")} \
        if src_pid in pad_samples else {}
    for (name, colour), pid in zip(STEM_ORDER, free):
        path = os.path.join(out, f"{name}.wav")
        if not os.path.exists(path):
            continue
        set_pad_sample(pid, f"{base} {name.upper()}", load_sample(path), path)
        padp[pid].update(keep, colour=colour, choke=0)
        set_tune(pid, tune.get(src_pid, 0))
        retune(pid)
        detected[pid] = dict(detected.get(src_pid, {}), src=id(pad_samples[pid][1]))
    bank["cur"] = free[0] // 16
    ui["selected"] = free[0]
    secs = time.monotonic() - stem_job["start"]
    toast(f"4 stems on pads {pad_name(free[0])}-{pad_name(free[3])}  ({secs:.0f} s)")
    mark_dirty()


def stems_status(pid):
    if stem_job["proc"] is not None:
        if stem_job["pid"] != pid:
            return "BUSY"
        return f"… {time.monotonic() - stem_job['start']:.0f}s"
    return "TURN >"


def take_screenshot():
    ui["shot"] = True


def save_screenshot(rgb):
    os.makedirs(SHOTS_DIR, exist_ok=True)
    name = time.strftime("shot %Y-%m-%d %H.%M.%S.png")
    Image.fromarray(rgb).save(os.path.join(SHOTS_DIR, name))
    toast("screenshot saved")
    print("Screenshot:", os.path.join(SHOTS_DIR, name))


def draw_projects(d):
    caps(d, (10, 2), "PROJECTS", TXT)
    d.rectangle([10, 14, SLOT_W - 6, 15], fill=TXT)
    used = sum(1 for i in proj["info"].values() if i)
    caps(d, (SLOT_W + 10, 2), f"OPEN: {proj['cur']:02d}  ·  {used} OF {NUM_PROJECTS} USED", GREY)
    for slot, label in ((6, "NAME"), (7, "EXPORT TO")):
        caps(d, (slot * SLOT_W + 10, 2), label, TXT)
        d.rectangle([slot * SLOT_W + 10, 14, (slot + 1) * SLOT_W - 6, 15], fill=BLUE)
    dest = app_state.get("export_dir")
    caps(d, (3 * SLOT_W + 10, 2), "EXPORTS: " + (os.path.basename(dest) + "/" if dest and os.path.isdir(dest)
                                                 else "INSIDE EACH PROJECT"), GREY)
    for k in range(NUM_PROJECTS):
        n = k + 1
        col, row = k % 4, k // 4                      # same layout as the pads: 1 bottom-left
        x = 10 + col * 236
        y = 128 - row * 32
        info = proj["info"].get(n)
        cur = n == proj["cur"]
        card(d, [x, y - 6, x + 228, y + 22], TXT if cur else (BLUE if info else DIM))
        caps(d, (x + 10, y - 1), f"{n:02d}", TXT if cur else (BLUE if info else DIM))
        if info:
            label = (f"{info['name'][:20]} · {info['bpm']} BPM · {info['date']}" if info.get("name")
                     else f"{info['bpm']} BPM · {info['pads']} pads · {info['date']}")
            d.text((x + 36, y + 8), label, font=F_ROW, fill=TXT if cur else GREY, anchor="lm")
        else:
            d.text((x + 36, y + 8), "empty", font=F_ROW, fill=DIM, anchor="lm")
        if proj["confirm_delete"] == n:
            d.text((x + 220, y + 8), "DELETE?", font=F_ROW, fill=RED, anchor="rm")


# ---------------------------------------------------------------------------
# Trim (SAMPLE page + Setup = SETTINGS · SAMPLE): move the start and end of the
# pad's sample. Non-destructive: the pad plays a region of its file, so you can
# always open it back up (RESET = whole file). Chops, stems, detection and
# resample all work on the trimmed part.
# ---------------------------------------------------------------------------
trim_ui = {"edit": "start", "voice": None, "from": 0}   # end touched last; preview voice + where it starts


def sample_page():
    return ui["page"] == "SAMPLE" and ui["open"] and ui["selected"] in pad_samples


def chop_view():
    """SAMPLE page + Setup: slicing (the plain SAMPLE page is trimming)."""
    return sample_page() and ui.get("settings") == "SAMPLE"


def source_audio(pid):
    """The pad's whole file (a trimmed / chopped pad only holds part of it)."""
    if pid not in pad_region:
        return pad_samples[pid][1]
    path = pad_files[pid]
    if path not in source_cache:
        source_cache[path] = load_sample(path)
    return source_cache[path]


def trim_bounds(pid):
    full = source_audio(pid)
    return pad_region.get(pid, (0, len(full)))


def set_trim(pid, start, end):
    full = source_audio(pid)
    start = int(clamp(start, 0, len(full) - SR // 100))
    end = int(clamp(end, start + SR // 100, len(full)))           # at least 10 ms
    name = pad_samples[pid][0]
    pad_samples[pid] = (name, full[start:end])
    if start == 0 and end == len(full):
        pad_region.pop(pid, None)
    else:
        pad_region[pid] = (start, end)
    pitched_shift[pid] = None
    retune(pid)
    chop_memory.pop(pid, None)
    if chop["pid"] == pid:
        chop["pid"] = None                                            # chop view starts fresh


def trim_step():
    """Coarse step: 1/400 of what's on screen."""
    full = len(source_audio(ui["selected"]))
    return max(1, full // chop["zoom"] // 400)


def trim_move(which, amount):
    pid = ui["selected"]
    s, e = trim_bounds(pid)
    trim_ui["edit"] = which
    if which == "start":
        set_trim(pid, s + amount, e)
    else:
        set_trim(pid, s, e + amount)
    play_trim_edge(pid, which)


def play_trim_edge(pid, which):
    """Hear what you're trimming: from the start, or the last half second before the end."""
    a = pad_samples[pid][1]
    s, e = trim_bounds(pid)
    stop_preview()
    if which == "start":
        play_preview(a[:SR])
        trim_ui["from"] = s
    else:
        tail = max(0, len(a) - SR // 2)
        play_preview(a[tail:])
        trim_ui["from"] = s + tail
    trim_ui["voice"] = ui["preview"]


def trim_commit(pid, d):
    """TRIM: make it final. The kept part becomes a new file on the pad (the original
    file stays on disk, so Undo brings the untrimmed sample back)."""
    if d <= 0 or pid not in pad_region:
        return
    import soundfile as sf
    audio = np.array(pad_samples[pid][1], dtype=np.float32)
    name = pad_samples[pid][0]
    base = os.path.splitext(name)[0][:28]
    k = 1
    while os.path.exists(os.path.join(SAMPLES_DIR, f"{base} trim {k:02d}.wav")):
        k += 1
    path = os.path.join(SAMPLES_DIR, f"{base} trim {k:02d}.wav")
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    sf.write(path, audio, SR, subtype="PCM_24")
    keep_tune = tune[pid]
    set_pad_sample(pid, name, audio, path)
    set_tune(pid, keep_tune)
    pitched_shift[pid] = None
    retune(pid)
    chop_memory.pop(pid, None)
    if chop["pid"] == pid:
        chop["pid"] = None
    chop["zoom"] = 1
    toast(f"trimmed to {len(audio) / SR:.3f} s")
    mark_dirty()


def trim_playhead(pid):
    """Where in the whole file the sound is right now (trim preview or the pad itself)."""
    since = max(0.0, time.monotonic() - audio_state["cb_time"] - audio_state["latency"])
    v = trim_ui["voice"]
    if v is not None and v is ui.get("preview") and v in voices and v.pos >= 0:
        return trim_ui["from"] + min(len(v.audio), v.pos + since * SR)
    v = last_voice.get(pid)
    if v is not None and v in voices and v.pos >= 0 and len(v.audio):
        s, e = trim_bounds(pid)
        return s + min(1.0, (v.pos + since * SR) / len(v.audio)) * (e - s)
    return None


def draw_trim(d):
    """SETTINGS · SAMPLE: the whole file, the kept part bright, start / end markers."""
    top = strip_height() + 2
    card(d, [222, top, 954, 145])
    pid = ui["selected"]
    full = source_audio(pid)
    L = len(full)
    s, e = trim_bounds(pid)
    colour = pad_screen_colour(pid)
    info = detect_summary(pid)
    d.text((232, top + 8), f"START {s / SR:.3f}s   END {e / SR:.3f}s   LENGTH {(e - s) / SR:.3f}s"
           + (f"   {info}" if info else ""), font=F_MONO, fill=GREY, anchor="lm")
    d.text((944, top + 8), (f"ZOOM ×{chop['zoom']}   " if chop["zoom"] > 1 else "") + f"{L / SR:.2f}s",
           font=F_MONO, fill=GREY, anchor="rm")
    x0, width = 232, 712
    y0, y1 = top + 18, 139
    mid, half = (y0 + y1) // 2, (y1 - y0) // 2 - 2
    span = max(1, L // chop["zoom"])
    centre = s if trim_ui["edit"] == "start" else e
    vs = int(clamp(centre - span // 2, 0, max(0, L - span)))
    xs = lambda pos: x0 + (pos - vs) / span * width
    for k in range(1, 8):                                            # quiet grid
        d.line([(x0 + k * width / 8, y0), (x0 + k * width / 8, y1)], fill=GRID)
    d.line([(x0, mid), (x0 + width, mid)], fill=GRID_MAJOR)
    a_, b_ = max(x0, xs(s)), min(x0 + width, xs(e))
    if b_ > a_:
        d.rectangle([a_, y0, b_, y1], fill=(32, 32, 32))              # the kept part, lifted a little
    pk = peaks(full[vs:vs + span], width)
    out_col = (64, 64, 64)
    for i in range(width):
        pos = vs + i * span / width
        inside = s <= pos < e
        h = max(1, int(pk[i] * half))
        d.line([(x0 + i, mid - h), (x0 + i, mid + h)], fill=colour if inside else out_col)
    head = trim_playhead(pid)
    if head is not None and vs <= head <= vs + span:                 # playhead
        d.line([(xs(head), y0 + 13), (xs(head), y1)], fill=TXT, width=1)
    for label, pos in (("S", s), ("E", e)):
        if vs <= pos <= vs + span:
            x = xs(pos)
            hot = (label == "S") == (trim_ui["edit"] == "start")
            c_ = SEL_OUTLINE if hot else GREY
            d.line([(x, y0), (x, y1)], fill=c_, width=2 if hot else 1)
            fx = x if label == "S" else x - 13
            d.rectangle([fx, y0, fx + 13, y0 + 12], fill=c_)
            d.text((fx + 7, y0 + 6), label, font=F_CHIP, fill=BG, anchor="mm")



# ---------------------------------------------------------------------------
# Setup button: a short settings menu for the current mode. PAD / SAMPLE page:
# the selected pad's colour (RGB). PATTERN page: pattern settings (none yet).
# Otherwise: GENERAL (the SETUP page). Setup again or a page button: close.
# ---------------------------------------------------------------------------
def copy_colour_to_bank(pid, d):
    """Turn right: every loaded pad in this bank gets this colour."""
    if d <= 0:
        return
    for p in range(pid // 16 * 16, pid // 16 * 16 + 16):
        if p in pad_samples:
            padp[p]["colour"] = colour_value(pid)
    toast(f"colour copied to bank {BANK_NAMES[pid // 16]}")


SETTINGS = {
    "PAD": [
        Param("COLOUR", lambda: colour_value(sel()),
              lambda d: padp[sel()].update(colour=clamp(colour_value(sel()) + d, 0, 127)),
              show=lambda v: f"{v} {wheel_name(v)}", needs_pad=True),
        Param("ROOT", lambda: padp[sel()]["root"],
              lambda d: padp[sel()].update(root=int(clamp(padp[sel()]["root"] + d, 0, 127))), per=3,
              show=note_name, needs_pad=True),
        Param("DETECT", lambda: "…" if detected.get(sel(), {}).get("busy") else "AGAIN >",
              lambda d: redetect(sel(), d), per=6, needs_pad=True),
        Param("STEP VELO", lambda: padp[sel()]["velocity"],
              lambda d: padp[sel()].update(velocity=clamp(padp[sel()]["velocity"] + d, 1, 127)),
              needs_pad=True),
        Param("VARY", lambda: padp[sel()].get("vary", 0),                # random pitch per hit (percs, hats)
              lambda d: padp[sel()].update(vary=clamp(padp[sel()].get("vary", 0) + 5 * d, 0, 100)),
              show=lambda v: "OFF" if v == 0 else f"±{v} ct", needs_pad=True),
        None, None,
        Param("TO BANK", lambda: "TURN >", lambda d: copy_colour_to_bank(sel(), d), per=8, needs_pad=True)],
    "PATTERN": [None] * 8,
    "SAMPLE": [
        Param("SLICES", lambda: len(chop_state()["markers"]), lambda d: chop_equal(len(chop["markers"]) + d),
              per=4, needs_pad=True),
        Param("MARKER", lambda: chop_state()["cur"] + 1,
              lambda d: chop.update(cur=clamp(chop["cur"] + d, 0, len(chop["markers"]) - 1), view=None),
              per=4, needs_pad=True),
        Param("MOVE", lambda: chop_state()["markers"][chop["cur"]],      # finer as you zoom in
              lambda d: chop_move(d * max(1, len(chop_audio()) // chop["zoom"] // 400)),
              show=lambda pos: f"{pos / SR:.3f} s", needs_pad=True),
        Param("AUTO", lambda: chop_state()["auto"], lambda d: chop_auto(chop["auto"] + d * 5), per=2,
              show=lambda a: "OFF" if a == 0 else f"{a}%", needs_pad=True),
        Param("STEMS", lambda: stems_status(sel()), lambda d: stems_knob(sel(), d), per=6, needs_pad=True),
        Param("QUALITY", lambda: setup["stem_q"],
              lambda d: setup.update(stem_q="BEST" if d > 0 else "FAST"), per=4),
        None, None],
    "GENERAL": [
        Param("CURVE", lambda: setup["curve"],
              lambda d: setup.update(curve=step_choice(CURVES, setup["curve"], d)), per=6,
              show=lambda c: c.upper()),
        Param("GAIN", lambda: setup["gain"],
              lambda d: setup.update(gain=clamp(setup["gain"] + d, 0, 100))),
        Param("THRESH", lambda: setup["threshold"],
              lambda d: setup.update(threshold=clamp(setup["threshold"] + d, 1, 100))),
        None, None,
        Param("BUFFER", lambda: setup.get("buffer", 128),         # knob 6
              lambda d: setup.update(buffer=step_choice([128, 256, 512], setup.get("buffer", 128), d)), per=6,
              show=lambda n: f"{n} {n / SR * 1000:.1f}ms"),
        Param("PAD LEDS", lambda: app_state["pad_bright"],                # knobs 7-8: this Push, all projects
              lambda d: set_pad_led("pad_bright", d, 10), show=lambda v: f"{v}%"),
        Param("UNLIT", lambda: app_state["pad_unlit"],
              lambda d: set_pad_led("pad_unlit", d, 0), show=lambda v: f"{v}%")],
    "SCALE": [
        Param("ROOT", lambda: setup["key_root"],
              lambda d: setup.update(key_root=(setup["key_root"] + d) % 12), per=3,
              show=lambda r: NOTE_NAMES_12[r]),
        Param("SCALE", lambda: setup["key_scale"],
              lambda d: setup.update(key_scale=step_choice(SCALE_NAMES, setup["key_scale"], d)), per=3),
        None, None, None, None, None, None],
}


HELP = {
    "kit": ["KIT · PATTERN", "Kit pads: play / select   ·   top 2 rows: steps of the selected pad",
            "Hold a step: knob 1 velocity (Shift: CHANCE), knob 2 nudge   ·   Delete + pad: clear its steps",
            "Note / Scale: keyboard   ·   Accent: 16 levels   ·   Select: patterns / banks",
            "Hold Duplicate + a pad, then another pad: MOVE or COPY it there",
            "Convert: resample the pattern (hold a pad + Convert: that pad)   ·   Shift + Convert: stems",
            "Shift+Record: sampling  ·  Add Track: projects  ·  Stop Clip: silence  ·  Fixed Length + knob 1: bars"],
    "trim": ["SAMPLE", "START / END: choose the part to keep (Up/Down zoom: the closer, the finer they move)",
             "TRIM: turn right to cut the sample to that part (Undo brings the whole sample back)",
             "BPM: the loop's tempo (fix it by hand if detection got it wrong; Shift = fine)",
             "STYLE: CLASSIC (speed + pitch together) / MODERN (time-stretch: tempo and pitch apart)",
             "MODE: 1-shot / gate   ·   STRETCH: OFF / x1 / ÷2 / x2 fits a loop to the BPM   ·   TUNE: cents",
             "Convert: resample this pad as you hear it onto an empty pad   ·   Setup: slicing + stems"],
    "chop": ["SAMPLE · SLICES (Setup)", "SLICES: equal slices   ·   AUTO: slices on the hits   ·   Record: tap to chop",
             "MARKER / MOVE: move a marker (Up/Down zoom = finer)   ·   Left/Right: scroll",
             "Top 2 rows: play slices   ·   New twice: slices onto the next empty bank",
             "STEMS: turn right = split into 4 pads (left = cancel)   ·   QUALITY: FAST or BEST (cleaner, ~5x slower)"],
    "keys": ["KEYBOARD", "Bottom 6 rows: notes (root in the pad colour)   ·   Octave Up/Down: octave",
             "Record + play: notes   ·   top 2 rows: steps with notes (tap: add the last note)",
             "Hold a step: velocity, nudge, note, length, or press keys for a chord",
             "Delete + key: remove that note   ·   Setup: key + scale   ·   Session: back to the kit"],
    "pad": ["PAD PAGE", "VOLUME, PITCH, PAN, CHANNELS (stereo / mono) of the selected pad",
            "SWING, HUMAN: this pad's groove ('pattern' = follow the pattern)",
            "STEP VELO: velocity of new steps   ·   CHOKE: pads in a group cut each other",
            "PITCH: CLASSIC pads change speed too; MODERN (SAMPLE page, STYLE) = pitch only, stays in time",
            "Setup: pad settings (colour, root note, copy colour to the bank)"],
    "pattern": ["PATTERN PAGE", "BARS: pattern length (1-8)   ·   SWING, HUMAN: the pattern's groove   ·   BPM",
                "Select: lower buttons = patterns (Left/Right: 1-8 / 9-16) or banks",
                "Duplicate: copy to the next empty slot (hold + slot: copy there)   ·   Delete + slot twice: clear",
                "Double Loop: double the length   ·   Quantize: tighten the selected pad (Shift: all)"],
    "device": ["FX BUSES (Device button)", "PAD page > BUS: send a pad to BUS 1-4 (MASTER = no bus)",
               "Buttons under the screen: 1-4 = the bus's effect slots, 5-8 = BUS 1-4",
               "Knob 1: the slot's effect (to change a set-up one, hold its slot button)   ·   knobs 2-6: settings",
               "Shift + slot: on/off   ·   Delete + slot: clear   ·   hold slot + Left/Right: move it",
               "Hold Mute + a bus button (5-8): mute / unmute that whole bus",
               "Signal: slot 1 > 2 > 3 > 4 > master   ·   Device again: back"],
    "inst": ["SYNTH (Add Device)", "Add Device: BASS synth on the first empty pad · hold Add Device + pad: exactly there (max 3)",
             "Select a synth pad + Add Device: its knobs   ·   Note / Scale: play it   ·   steps work too",
             "WAVE, DETUNE (2nd oscillator), SUB (one octave down), CUTOFF, EMPHASIS (resonance)",
             "CONTOUR: how far the filter opens on each note   ·   DECAY: how fast it closes   ·   GLIDE",
             "KEYS (Shift + Add Device switches BASS / KEYS): PROGRAM piano, strings, brass, organ, e.piano, bell",
             "KEYS: OCTAVE, ATTACK, RELEASE, TONE, VELOCITY (how much harder playing = louder), 4 notes at once",
             "KEYS CHORD: 5TH, TRIAD, SUS2, SUS4, 7TH, ADD9 from the KEY / SCALE (Settings)   ·   CHORUS: Juno I / II / I+II",
             "Shift + Layout: the screen shows the sequencer / piano roll, the knobs stay on the synth",
             "Blue step pads = its notes: hold one + turn a knob = that step's own value (automation)",
             "PAD page: volume, pan, BUS as usual   ·   Delete + pad (no steps): remove the synth"],
    "automate": ["AUTOMATION (Automate button)", "Automate: arm (blinks red while playing)   ·   again: stop recording",
                 "Armed + playing: touch or turn a knob = its moves are recorded into this pattern",
                 "Let go: the knob plays back by itself (a red dot on its value = automated)",
                 "Delete + touch (or turn) a knob: clear its automation in this pattern",
                 "Hold step pad(s) + turn knob 3-8: a value for those steps only (red dot = set)",
                 "Synths: blue step pads = notes (hold one + knobs) · keyboard view: hold a top step + knob 5-8",
                 "Shift + Layout: synth / FX pages show the sequencer, the knobs stay on the page",
                 "Pad volume / pan, pad FX, synths, bus FX, mixer, master · per pattern, also in song mode"],
    "calibrate": ["DISPLAY CALIBRATION (Shift + Setup)", "Lower button 1: GREY · turn knobs 1-3 (darks) and 5-7 (brights) until the bars are neutral grey",
                  "Page button: COLOURS · knobs 1-3 = red, green, blue of that tab until it matches the button's light",
                  "Delete: back to the default   ·   Shift + Setup: save and leave (kept for every project)"],
    "padfx": ["PAD FX (FILTER / EQ / COMP tabs)", "Every pad has its own strip: FILTER > EQ > COMP, then its volume, pan and BUS",
              "Knob 1: ON / OFF   ·   turning any other knob switches the module on",
              "FILTER: MODE, CUTOFF, RESO, DRIVE   ·   EQ: LOW / MID / HIGH gain + frequency, MID Q",
              "COMP: THRESH, RATIO, ATTACK, RELEASE, MIX   ·   the strip lights up top right",
              "Pick another pad to edit its strip   ·   resample and exports include it"],
    "song": ["SONG (Clip button)", "Lower buttons: song rows (Left/Right: more rows)   ·   the blinking one adds a row",
             "Knob 1 ROW · 2 PATTERN · 3 REPEATS · 7 END (LOOP / STOP) · 8 PLAY (SONG / PATTERN)",
             "PLAY = SONG: Play runs the song (from the selected row while this page is open, else from row 1)",
             "Delete + row: remove it   ·   Duplicate + row: copy it after   ·   patterns change at the end of their loop",
             "Convert: whole song as a WAV in exports/   ·   Shift + Convert: song stems + mix (stop playback first)"],
    "mix": ["MIXER (Mix button)", "Knobs 1-4: BUS 1-4 volume, or pan (knob 8 VOL/PAN switches)   ·   knob 5: MASTER",
            "Buttons under the screen 1-4: mute / unmute BUS 1-4",
            "Meters show each bus after its effects and fader; far right: the master output",
            "Pads go to a bus on the PAD page (BUS knob)   ·   Mix again: back"],
    "master": ["MASTER (Master button)", "SSL-style bus compressor on the whole mix, then a limiter (never clips)",
               "THRESH, RATIO (OFF = no compression), ATTACK, RELEASE: like the SSL's switches",
               "MAKEUP: gain after the compressor   ·   MIX: blend with the dry mix (parallel)",
               "LIMIT: push the mix into the limiter (louder)   ·   VOLUME: listening level",
               "Metronome and previews skip the master bus   ·   Convert resamples through it"],
    "settings": ["SETTINGS PAGE", "CLICK: metronome level   ·   KEY, SCALE: the project's key",
                 "COUNT-IN: bars before recording   ·   REC QUANT: how much recording snaps to the grid",
                 "REC GRID: that grid (1/8 to 1/32, triplets T) · also used by Quantize · Repeat records at its rate",
                 "Setup: general settings (pad response: curve, gain, threshold)"],
    "projects": ["PROJECTS", "16 projects, one per kit pad (01 = bottom-left)   ·   white = open, blue = used",
                 "Hit a pad: open it (everything is saved first)   ·   empty pad: start a new project",
                 "Delete + pad twice: delete (kept in projects/_deleted)   ·   Add Track: close",
                 "Upper button 7 NAME: name this project · 8 EXPORT TO: pick the export folder (on the computer)",
                 "Exports go to <that folder>/<project name>/ · each export asks its name on the computer",
                 "Shift + User: screenshot of this screen (saved in screenshots/)"],
    "menu": ["SETTINGS MENU", "Settings for what you're doing: the pad, the pattern, the keyboard or general",

             "Turn right on TO BANK to copy the colour to every pad in the bank",
             "Settings page + Setup = General · knob 6 BUFFER: 128 / 256 / 512 - smaller = faster pads",
             "Pad · knob 5 VARY: each hit a slightly different pitch (± cents) - percs, hats",
             "General · knob 7 PAD LEDS: pad brightness · knob 8 UNLIT: unselected pads (all projects)",
             "Setup again or a page button: close"],
    "levels": ["16 LEVELS", "The kit pads play the selected pad at 16 velocities, soft to hard",
               "Record and Note Repeat work as usual   ·   Accent or Session: back to the kit"],
    "sampling": ["SAMPLING", "Record: start / stop (with THRESH it waits for sound)",
                 "Hit an empty pad: sample there   ·   LENGTH: stop after 1-8 bars",
                 "Listen through the AudioFuse's direct monitoring   ·   Shift + Record: leave"],
    "browser": ["BROWSER", "Knob 1: scroll   ·   knob 2: scroll fast   ·   > open / play again   ·   < back",
                "Hit a kit pad: load the sample there (hit again to replace)   ·   Browse: close",
                "Upper button 7 FAV: add / remove the highlighted sample or folder (FAVORITES at the top)",
                "Upper button 8 SEARCH: type words on the computer; searches this folder (or the sample library from the top)",
                "Home / Drives / Computer at the top: browse any folder on the Mac",
                "Search results / favorites: < goes back to where you were"],
}


def toggle_help():
    """The User button arrives twice (Live port + User port): the second copy, a few
    ms later, would switch help straight back off. Ignore it. Shift + User: screenshot."""
    now = time.monotonic()
    if now - ui.get("help_t", 0) < 0.15:
        return
    ui["help_t"] = now
    if SHIFT_BUTTON in held_buttons:
        take_screenshot()
        return
    if not ui.get("help"):
        ui["help"], ui["help_page"] = True, 0
    elif ui["help_page"] + 1 < len(help_pages()):
        ui["help_page"] += 1                               # User again: next help page
    else:
        ui["help"] = False


def help_pages():
    ctx = help_context()
    return [ctx, "kit"] if ctx in ("pad", "pattern", "settings", "menu") else [ctx]


def help_context():
    if cal_ui["on"]:
        return "calibrate"
    if proj["open"]:
        return "projects"
    if auto["arm"]:
        return "automate"
    if smp["open"]:
        return "sampling"
    if browser.active:
        return "browser"
    if kb["levels"]:
        return "levels"
    if kb["mode"]:
        return "keys"
    if chop_view():
        return "chop"
    if ui.get("settings"):
        return "menu"
    if chop_view():
        return "chop"
    if sample_page():
        return "trim"
    if ui["open"] and ui["page"] in PAD_FX:
        return "padfx"
    if ui["open"] and ui["page"] in ("PAD", "PATTERN", "MASTER", "DEVICE", "MIX", "INST", "SONG"):
        return ui["page"].lower()
    if ui["open"] and ui["page"] == "SETUP":
        return "settings"
    return "kit"


def draw_help(d):
    pages = help_pages()
    i = min(ui.get("help_page", 0), len(pages) - 1)
    lines = HELP[pages[i]]
    d.rounded_rectangle([140, 26, 820, 150], 8, fill=(18, 18, 22), outline=GREY)
    caps(d, (156, 34), "HELP · " + lines[0], TXT)
    gap = 22 if len(lines) <= 5 else 18                    # 5 lines still fit
    for row, line in enumerate(lines[1:]):
        d.text((156, 52 + row * gap), line, font=F_BODY, fill=(205, 205, 212))
    more = i + 1 < len(pages)
    caps(d, (804, 34), (f"{i + 1}/{len(pages)} · USER: NEXT" if more else
                        (f"{i + 1}/{len(pages)} · " if len(pages) > 1 else "") + "USER: CLOSE"), GREY, anchor="ra")


def settings_context():
    if kb["mode"]:
        return "SCALE"
    if ui["page"] == "SAMPLE" and ui["open"] and ui["selected"] in pad_samples:
        return "SAMPLE"
    if ui["page"] in ("PAD", "SAMPLE") or not ui["open"]:
        return "PAD"
    if ui["page"] == "PATTERN":
        return "PATTERN"
    return "GENERAL"


def settings_button():
    if ui.get("settings"):
        ui["settings"] = None
        return
    ui["settings"] = settings_context()


def on_message(msg):
    if msg.type in ("note_on", "note_off") and msg.note < 8:      # knob 1-8 touched / let go
        auto_touch(msg.note, msg.type == "note_on" and msg.velocity > 0)
        return
    if msg.type in ("note_on", "note_off") and push2.is_pad(msg.note):
        down = msg.type == "note_on" and msg.velocity > 0
        pid = note_to_pid(msg.note)
        if not down:
            held.discard(msg.note)
            gate_release(msg.note, pid)
            record_note_off(msg.note)
            info = held_note_steps.pop(msg.note, None)
            if info and info["existed"] and not info["edited"] and not held_long(info):
                checkpoint(f"note step {info['s'] + 1}")
                melody[info["pad"]] = [e for e in melody.get(info["pad"], [])
                                       if e not in notes_at_step(info["pad"], info["s"])]
                mark_dirty()
            if pid is not None:
                repeat["held"].pop(pid, None)
            info = held_steps.pop(msg.note, None)
            if info and info["existed"] and not info["edited"] and not held_long(info):
                checkpoint(f"step {info['s'] + 1} {pad_label(info['pad'])}")
                pattern[info["pad"]][info["s"]] = None
                set_step_chance(info["pad"], info["s"], 100)
                mark_dirty()
            return
        held.add(msg.note)
        if proj["open"]:
            if msg.note in KIT_NOTES:
                n = KIT_NOTES.index(msg.note) + 1
                if DELETE_BUTTON in held_buttons:
                    delete_project(n)
                else:
                    switch_project(n)
            return
        if ADD_DEVICE_BUTTON in held_buttons and pid is not None and not kb["mode"] and not kb["levels"] \
                and not browser.active and not smp["open"]:
            add_device_on(pid)
            return
        if DUPLICATE_BUTTON in held_buttons and pid is not None and not kb["mode"] and not kb["levels"] \
                and not browser.active and not smp["open"]:
            pad_move_press(pid)
            return
        sel_pad = ui["selected"]
        if kb["mode"] and sel_pad in pad_samples and not browser.active and not smp["open"]:
            n = kb_pad_note(msg.note)
            if n is not None:                               # keyboard: play the pad at this note
                if DELETE_BUTTON in held_buttons:            # Delete + key: remove that note
                    gone = [e for e in melody.get(sel_pad, []) if e[1] == n]
                    if gone:
                        checkpoint(f"delete {note_name(n)}")
                        melody[sel_pad] = [e for e in melody[sel_pad] if e[1] != n]
                        toast(f"deleted every {note_name(n)} of {pad_label(sel_pad)}")
                        mark_dirty()
                    return
                if held_note_steps:                          # holding steps + key: add / remove that note
                    checkpoint("chord note", key=("chord", tuple(sorted(held_note_steps)), n))
                    for info in held_note_steps.values():
                        info["edited"] = True
                        auto_note = info.pop("auto_note", None)
                        if auto_note is not None and auto_note in melody.get(sel_pad, []):
                            # the note the tap just added becomes the key you play
                            replace_note(sel_pad, auto_note, (auto_note[0], n, auto_note[2], auto_note[3]))
                            continue
                        evs = notes_at_step(sel_pad, info["s"])
                        same = [e for e in evs if e[1] == n]
                        if same:
                            melody[sel_pad] = [e for e in melody[sel_pad] if e not in same]
                        else:
                            length = evs[0][3] if evs else 1.0
                            add_note(sel_pad, (float(info["s"]), n, padp[sel_pad]["velocity"], length))
                    kb["last_note"] = n
                    mark_dirty()
                if msg.velocity >= setup["threshold"]:
                    ui["last_velocity"] = msg.velocity
                    gain = velo_gain(msg.velocity)
                    v = trigger(sel_pad, gain, audio=note_audio(sel_pad, n))
                    live_voices[msg.note] = v
                    live_midi[msg.note] = v.want if isinstance(v, SynthVoice) else n   # after OCTAVE / PITCH
                    kb["last_note"] = n
                    if rec["on"] and clock["playing"] and not held_note_steps:
                        record_note_on(msg.note, sel_pad, n, gain * 127)
                return
            s_top = step_of(msg.note)
            if s_top is not None:                            # top rows: this pad's notes per step
                s = seq["page"] * STEPS + s_top
                evs = notes_at_step(sel_pad, s)
                last = kb.get("last_note", ROOT)
                # tap: adds the last note you played (also onto a step that has other notes:
                # that builds a chord). Tap a step that already has that note: clears the step.
                existed = any(e[1] == last for e in evs)
                auto_note = None
                if not existed:
                    checkpoint(f"note step {s + 1}")
                    length = evs[0][3] if evs else 1.0
                    auto_note = (float(s), last, padp[sel_pad]["velocity"], length)
                    add_note(sel_pad, auto_note)
                    mark_dirty()
                held_note_steps[msg.note] = {"pad": sel_pad, "s": s, "existed": existed, "edited": False,
                                             "t": time.monotonic(),
                                             "auto_note": auto_note}
                return
        if kb["levels"] and sel_pad in pad_samples and msg.note in KIT_NOTES and not browser.active:
            level = LEVELS[KIT_NOTES.index(msg.note)]        # 16 levels: fixed velocity per pad
            gain = level / 127
            if repeat["on"]:
                start_repeat(sel_pad, gain)
            else:
                live_voices[msg.note] = trigger(sel_pad, gain)
                if rec["on"] and clock["playing"]:
                    record_at(sel_pad, tick_at(now_on_clock()), level)
            return
        if smp["open"] and pid is not None and pid not in pad_samples:
            if smp["state"] in ("idle", "waiting"):
                smp["target"] = pid                         # sampling: the sample goes here
            return
        item = browser.current() if browser.active else None
        chopping = chop_view() and not browser.active
        if chopping:
            chop_state()
        if chopping and chop["tap"] and (pid is not None or step_of(msg.note) is not None):
            chop_tap()                                      # tapping: any pad starts a new slice
        elif chopping and step_of(msg.note) is not None:
            k = step_of(msg.note)                       # top two rows = slices 1-16: hear it, jump to it
            if k < len(chop["markers"]):
                chop.update(cur=k, view=None)
            play_slice(k)
            chop["held"] = msg.note
        elif item is not None and not item[2] and pid is not None:
            # browser open on a sample + hit a kit pad = load it there.
            # Replacing a sample asks first: hit the same pad again to confirm.
            pending = ui.get("pending")
            if pid not in pad_samples or (pending and pending == (pid, item[1])):
                ui["pending"] = None
                threading.Thread(target=load_to_pad, args=(pid, item[1]), daemon=True).start()
            else:
                ui["pending"] = (pid, item[1])
        elif pid in pad_samples and DELETE_BUTTON in held_buttons:
            delete_pad(pid)                       # hold Delete + kit pad
        elif pid in pad_samples and (MUTE_BUTTON in held_buttons or SOLO_BUTTON in held_buttons):
            # hold Mute or Solo + hit a kit pad = toggle it (no sound)
            target = mutes if MUTE_BUTTON in held_buttons else solos
            checkpoint(("mute " if target is mutes else "solo ") + pad_label(pid))
            target.symmetric_difference_update({pid})
            mark_dirty()
        elif pid in pad_samples and msg.velocity < setup["threshold"]:
            pass                                        # below THRESH: ignored
        elif pid in pad_samples:
            ui["selected"] = pid
            ui["last_velocity"] = msg.velocity
            gain = velo_gain(msg.velocity)
            if repeat["on"]:
                start_repeat(pid, gain)            # the repeat engine plays it
            else:
                ui["voice"] = trigger(pid, gain)
                live_voices[msg.note] = ui["voice"]
                if rec["on"] and clock["playing"]:
                    record_at(pid, tick_at(now_on_clock()), gain * 127)
        elif step_of(msg.note) is not None and ui["selected"] is not None:
            # Press an empty step: it's added. Press a lit step: it's removed when you
            # let go, unless you turned a knob while holding it (then you were editing).
            s = seq["page"] * STEPS + step_of(msg.note)
            sel_pad = ui["selected"]
            steps = pattern[sel_pad]
            existed = bool(steps[s])
            if not existed and notes_at_step(sel_pad, s):
                # keyboard notes here (a synth line): holding = edit this step, nothing is added
                held_steps[msg.note] = {"pad": sel_pad, "s": s, "existed": False, "edited": False, "notes": True}
                return
            if not existed:
                checkpoint(f"step {s + 1} {pad_label(sel_pad)}")
                steps[s] = (padp[sel_pad]["velocity"], 0.0)
                set_step_chance(sel_pad, s, 100)
                mark_dirty()
            held_steps[msg.note] = {"pad": sel_pad, "s": s, "existed": existed, "edited": False,
                                    "t": time.monotonic()}
    elif msg.type == "polytouch" and note_to_pid(msg.note) in repeat["held"]:
        if msg.value > 0:
            repeat["held"][note_to_pid(msg.note)]["gain"] = velo_gain(msg.value)   # press harder = louder
    elif msg.type == "aftertouch" and repeat["held"]:
        if msg.value > 0:
            for st in repeat["held"].values():
                st["gain"] = velo_gain(msg.value)
    elif msg.type == "control_change":
        c, v = msg.control, msg.value
        q = ui.get("confirm_clear")
        if q is not None and v > 0 and c not in (DELETE_BUTTON, LOWER_BUTTONS[q % SLOTS_PER_PAGE]):
            ui["confirm_clear"] = None                  # any other button cancels the question
        if ui.get("pad_move") and c in UPPER_BUTTONS:
            if v > 0:
                i = UPPER_BUTTONS.index(c)
                pad_move_choice(i) if i in (0, 1) else ui.pop("pad_move")
            return
        if ui.get("pad_move") and c == DUPLICATE_BUTTON and v > 0:
            ui.pop("pad_move")                           # start over
        if proj["open"] and c in UPPER_BUTTONS:
            if v > 0:
                i = UPPER_BUTTONS.index(c)
                if i == 6:
                    name_project()
                elif i == 7:
                    choose_export_folder()
            return
        if c == BROWSE_BUTTON and v > 0:
            browser.toggle()
            ui["pending"] = None
        elif c in HELP_BUTTONS and browser.active:
            if v > 0:
                toggle_help()
        elif browser.active and browser_control(c, v):
            pass
        elif c in (MUTE_BUTTON, SOLO_BUTTON, DELETE_BUTTON, SHIFT_BUTTON):
            (held_buttons.add if v > 0 else held_buttons.discard)(c)
            if cal_ui["on"] and c == DELETE_BUTTON and v > 0:
                cal_button(c)                               # calibrating: reset what you're editing
        elif c in HELP_BUTTONS:
            if v > 0:
                toggle_help()
        elif c == RECORD_BUTTON and v > 0 and SHIFT_BUTTON in held_buttons:
            sampling_close() if smp["open"] else sampling_open()
        elif c == ADD_TRACK_BUTTON and v > 0:
            open_projects()
        elif c == STOP_CLIP_BUTTON and v > 0:
            silence_all()
        elif c == FIXED_LENGTH_BUTTON:
            (held_buttons.add if v > 0 else held_buttons.discard)(c)
        elif FIXED_LENGTH_BUTTON in held_buttons and KNOB_FIRST <= c < KNOB_FIRST + 8:
            param = PAGES["PATTERN"]["knobs"][0]                  # BARS, from anywhere
            checkpoint("bars", key=("bars",))
            param.turn(v)
            toast(f"pattern length: {seq['bars']} bar{'s' if seq['bars'] > 1 else ''}")
            mark_dirty()
        elif smp["open"] and sampling_control(c, v):
            pass
        elif c == SETUP_BUTTON and v > 0 and SHIFT_BUTTON in held_buttons:
            cal_toggle()                                    # Shift + Setup: display calibration
        elif cal_ui["on"] and KNOB_FIRST <= c < KNOB_FIRST + 8:
            cal_knob(c - KNOB_FIRST, v)
        elif cal_ui["on"] and v > 0 and cal_button(c):
            pass
        elif c == SETUP_BUTTON and v > 0:
            settings_button()
        elif ui.get("settings") and KNOB_FIRST <= c < KNOB_FIRST + 8:
            param = SETTINGS[ui["settings"]][c - KNOB_FIRST]
            if param is not None and param.active():
                if param.label != "STEMS":                     # the stems themselves are one undo step
                    checkpoint(param.label.lower(), key=("settings", c, ui["selected"]))
                param.turn(v)
                mark_dirty()
        elif c == METRONOME_BUTTON and v > 0:
            setup["metronome"] = not setup["metronome"]
            toast("metronome on" if setup["metronome"] else "metronome off")
            mark_dirty()
        elif c == TAP_TEMPO_BUTTON:
            (held_buttons.add if v > 0 else held_buttons.discard)(c)
            if v > 0:
                tap_tempo()
        elif c == DUPLICATE_BUTTON:
            if v > 0:
                held_buttons.add(c)
                ui["dup_used"] = False
                ui["dup_src"] = None
            else:
                held_buttons.discard(c)
                ui["dup_src"] = None
                if not ui.get("dup_used"):
                    duplicate_pattern(None)       # Duplicate on its own: next empty slot
        elif c in (OCTAVE_UP, OCTAVE_DOWN) and v > 0 and kb["mode"]:
            kb["octave"] = int(clamp(kb["octave"] + (1 if c == OCTAVE_UP else -1), 0, 7))
            toast(f"octave {kb['octave']}")
        elif c in (OCTAVE_UP, OCTAVE_DOWN) and v > 0:
            jump = 8 if SHIFT_BUTTON in held_buttons else 1
            select_neighbour(-jump if c == OCTAVE_UP else jump)
        elif c == SELECT_BUTTON and v > 0 and SHIFT_BUTTON in held_buttons:
            open_projects()
        elif c == SELECT_BUTTON and v > 0:
            pat["row"] = "banks" if pat["row"] == "patterns" else "patterns"
        elif c in LOWER_BUTTONS and song_open() and not ui.get("settings"):
            if v > 0:
                song_lower_button(LOWER_BUTTONS.index(c))
        elif c in (ARROW_LEFT, ARROW_RIGHT) and v > 0 and song_open():
            song["page"] = int(clamp(song["page"] + (-1 if c == ARROW_LEFT else 1), 0,
                                     min(len(song["rows"]) // 8, MAX_SONG_ROWS // 8 - 1)))
        elif c in LOWER_BUTTONS and mix_open():
            k = LOWER_BUTTONS.index(c)
            if v > 0 and k < NUM_BUSES:                         # lower buttons 1-4: mute a bus
                checkpoint("bus mute")
                buses[k]["mute"] = not buses[k].get("mute", False)
                toast(f"BUS {k + 1} {'muted' if buses[k]['mute'] else 'on'}")
                mark_dirty()
        elif c in LOWER_BUTTONS and device_open():
            if v > 0:
                held_buttons.add(c)
                device_lower_button(LOWER_BUTTONS.index(c))
            else:
                held_buttons.discard(c)
        elif c in (ARROW_LEFT, ARROW_RIGHT) and v > 0 and device_open() and \
                LOWER_BUTTONS[dev["slot"]] in held_buttons:
            device_move_slot(-1 if c == ARROW_LEFT else 1)
        elif c in LOWER_BUTTONS and v > 0 and pat["row"] == "banks":
            bank["cur"] = LOWER_BUTTONS.index(c)
            toast(f"bank {BANK_NAMES[bank['cur']]}")
        elif c in LOWER_BUTTONS and v > 0:
            pattern_button(pat["page"] * SLOTS_PER_PAGE + LOWER_BUTTONS.index(c))
        elif c in (ARROW_LEFT, ARROW_RIGHT) and v > 0 and chop_view():
            chop_scroll(-1 if c == ARROW_LEFT else 1)      # chop view: move through the waveform
        elif c in (ARROW_LEFT, ARROW_RIGHT) and v > 0:
            pat["page"] = 0 if c == ARROW_LEFT else 1      # patterns 1-8 / 9-16
        elif c == UNDO_BUTTON and v > 0:
            redo() if SHIFT_BUTTON in held_buttons else undo()
        elif held_note_steps and KNOB_FIRST <= c < KNOB_FIRST + 4:
            edit_held_notes(c - KNOB_FIRST, v if v < 64 else v - 128)
        elif held_note_steps and KNOB_FIRST + 4 <= c < KNOB_FIRST + 8:
            step_automation(c - KNOB_FIRST, v)                 # hold a note step + knob 5-8
        elif held_steps and c in (KNOB_FIRST, KNOB_FIRST + 1) and \
                not all(i.get("notes") for i in held_steps.values()):
            edit_held_steps(c - KNOB_FIRST, v if v < 64 else v - 128)
        elif held_steps and c in (KNOB_FIRST, KNOB_FIRST + 1):
            step_automation(c - KNOB_FIRST, v)                 # a synth note step: all 8 knobs
        elif held_steps and KNOB_FIRST + 2 <= c < KNOB_FIRST + 8:
            step_automation(c - KNOB_FIRST, v)                 # hold a step + knob 3-8: that step only
        elif c == QUANTIZE_BUTTON:
            (held_buttons.add if v > 0 else held_buttons.discard)(c)
            if v > 0:
                quantize(whole_pattern=SHIFT_BUTTON in held_buttons)
        elif KNOB_FIRST <= c < KNOB_FIRST + 8 and DELETE_BUTTON in held_buttons and not ui.get("settings"):
            if not auto.get("cleared_t") or time.monotonic() - auto["cleared_t"] > 0.5:
                auto["cleared_t"] = time.monotonic()
                auto_clear(c - KNOB_FIRST)                     # Delete + turn: clear its automation
        elif KNOB_FIRST <= c < KNOB_FIRST + 8:
            auto["turned"][c - KNOB_FIRST] = time.monotonic()
            param = page_knobs()[c - KNOB_FIRST]
            if param is not None and ui["open"] and param.active():
                checkpoint(param.label.lower(), key=(ui["page"], c, ui["selected"]))
                param.turn(v)
                mark_dirty()
        elif c in UPPER_BUTTONS and v > 0:
            ui["settings"] = None                              # a page button closes Setup
            name = PAGE_ORDER[UPPER_BUTTONS.index(c)]
            if name in PAGES and PAGES[name]["ready"]:
                if name == ui["page"]:
                    ui["open"] = not ui["open"]       # same button again: show / hide the page
                else:
                    ui["page"], ui["open"] = name, True
        elif c == SWING_KNOB:
            # Swing encoder = swing of the selected pad (pattern swing if none selected)
            param = PAGES["PAD"]["knobs"][3] if ui["selected"] is not None else PAGES["PATTERN"]["knobs"][1]
            checkpoint("swing", key=("swing", ui["selected"]))
            param.turn(v)
            mark_dirty()
        elif c == PLAY_BUTTON and v > 0:
            if clock["playing"]:
                clock["playing"] = False
                song["pos"] = None
                rec["on"] = False                          # stopping ends the recording
                for vc in list(voices):                    # and everything stops now (2 ms fade),
                    if vc is ui.get("preview"):            # except a browser / chop preview
                        continue
                    if vc.pos < 0:
                        voices.remove(vc)                  # scheduled but not started yet
                    elif vc.fade_left is None:
                        vc.fade_left = FADE
            else:
                start_playing()
            repeat["held"].clear()
        elif c == RECORD_BUTTON and v > 0 and chop_view():
            chop_tap_toggle()
        elif c == ADD_DEVICE_BUTTON:
            # tap: synth on the first empty pad (or open/close the selected synth's page);
            # hold + press a pad: put it exactly there
            if v > 0:
                held_buttons.add(c)
                ui["add_used"] = False
            else:
                held_buttons.discard(c)
                sp = ui["selected"]
                if not ui.get("add_used") and SHIFT_BUTTON in held_buttons and sp is not None and is_inst(sp):
                    switch_instrument(sp)
                elif not ui.get("add_used"):
                    if ui["page"] == "INST" and ui["open"]:
                        ui["page"] = "PAD"                           # Add Device again: back
                    else:
                        add_device()
        elif c == AUTOMATE_BUTTON and v > 0:
            auto["arm"] = not auto["arm"]
            toast("automation: ARMED  ·  touch or turn a knob while playing to record it" if auto["arm"]
                  else "automation: playing back (not recording)")
        elif c == CLIP_BUTTON and v > 0:
            ui["settings"] = None
            if song_open():
                ui["page"] = ui.get("before_song", "PAD")
            else:
                if ui["page"] != "SONG":
                    ui["before_song"] = ui["page"]
                ui["page"], ui["open"] = "SONG", True
                song["page"] = song["sel"] // 8
        elif c == MIX_BUTTON and v > 0:
            ui["settings"] = None
            if mix_open():
                ui["page"] = ui.get("before_mix", "PAD")
            else:
                if ui["page"] != "MIX":
                    ui["before_mix"] = ui["page"]
                ui["page"], ui["open"] = "MIX", True
        elif c == DEVICE_BUTTON and v > 0:
            ui["settings"] = None
            if device_open():
                ui["page"] = ui.get("before_device", "PAD")
            else:
                if ui["page"] != "DEVICE":
                    ui["before_device"] = ui["page"]
                sp = ui["selected"]
                if sp is not None and padp[sp].get("bus") is not None:
                    dev["bus"] = padp[sp]["bus"]                     # open the selected pad's bus
                ui["page"], ui["open"] = "DEVICE", True
        elif c == MASTER_BUTTON and v > 0:
            ui["settings"] = None
            if ui["page"] == "MASTER" and ui["open"]:
                ui["page"] = ui.get("before_master", "PAD")         # Master again: back
            else:
                if ui["page"] != "MASTER":
                    ui["before_master"] = ui["page"]
                ui["page"], ui["open"] = "MASTER", True
        elif c == CONVERT_BUTTON and v > 0:
            if song_open():
                request_export("song_stems" if SHIFT_BUTTON in held_buttons else "song")
            else:
                request_export("stems") if SHIFT_BUTTON in held_buttons else main_jobs.append(resample)
        elif c == NEW_BUTTON and v > 0:
            if chop_view():
                chop_apply()
        elif c == RECORD_BUTTON and v > 0:
            if rec["on"]:
                rec["on"] = False
                toast("recording off")
            else:
                checkpoint("record")                       # one undo per recording pass
                history["last_key"] = None
                rec["on"] = True
                if not clock["playing"]:
                    start_playing(setup["count_in"])
                toast("recording" + ("  ·  quantize off" if setup["rec_quant"] == 0
                                     else f"  ·  quantize {setup['rec_quant']}% to {setup['rec_grid']}"))
        elif c == REPEAT_BUTTON and v > 0:
            repeat["on"] = not repeat["on"]
            repeat["held"].clear()
            toast(f"note repeat {RATES[repeat['rate']][0]}" if repeat["on"] else "note repeat off")
        elif c in RATE_BUTTONS and v > 0:
            repeat["rate"] = RATE_BUTTONS.index(c)
            toast(f"repeat rate {RATES[repeat['rate']][0]}")
        elif c == PAGE_LEFT and v > 0:
            seq["follow"] = False                 # choosing a bar by hand stops following
            seq["page"] = max(0, seq["page"] - 1)
        elif c == PAGE_RIGHT and v > 0:
            seq["follow"] = False
            seq["page"] = min(seq["bars"] - 1, seq["page"] + 1)
        elif c == FOLLOW_BUTTON and v > 0 and SHIFT_BUTTON in held_buttons:
            ui["seq_view"] = not ui.get("seq_view")      # synth / FX pages: show the steps instead
            toast("screen: sequencer (knobs stay on this page)" if ui["seq_view"]
                  else "screen: this page's view")
        elif c == FOLLOW_BUTTON and v > 0:
            seq["follow"] = not seq["follow"]
            mark_dirty()
        elif c in (ZOOM_IN, ZOOM_OUT) and v > 0 and sample_page() and not browser.active:
            chop_zoom(1 if c == ZOOM_IN else -1)
        elif c in (ZOOM_IN, ZOOM_OUT) and v > 0:
            i = ZOOM_LEVELS.index(seq["zoom"]) + (1 if c == ZOOM_OUT else -1)
            seq["zoom"] = ZOOM_LEVELS[max(0, min(len(ZOOM_LEVELS) - 1, i))]
            mark_dirty()
        elif c == DOUBLE_BUTTON and v > 0:
            checkpoint("double loop")
            double_pattern()
            mark_dirty()
        elif c == SESSION_BUTTON and v > 0:
            proj["open"] = False
            kb.update(mode=None, levels=False)
            if ui.get("settings") == "SCALE":
                ui["settings"] = None
            # Session = the kit. Shift + Session: switch the screen between grid and waveform
            if SHIFT_BUTTON in held_buttons:
                ui["view"] = "sample" if ui["view"] == "seq" else "seq"
            else:
                ui["view"] = "seq"
        elif c == NOTE_BUTTON and v > 0:
            set_kb_mode("chromatic")
        elif c == SCALE_BUTTON and v > 0:
            set_kb_mode("scale")
        elif c == ACCENT_BUTTON and v > 0:
            if ui["selected"] not in pad_samples:
                toast("select a pad first")
            else:
                kb["levels"] = not kb["levels"]
                kb["mode"] = None
                toast("16 levels: " + pad_label(ui["selected"]) if kb["levels"] else "16 levels off")
        elif c == TEMPO_KNOB:
            checkpoint("tempo", key="bpm")
            clock["bpm"] = max(BPM_MIN, min(BPM_MAX, clock["bpm"] + knob_steps("bpm", v, TICKS_PER_BPM)))
            mark_dirty()
        elif c in VOLUME_KNOBS:
            step = v if v < 64 else v - 128
            master["volume"] = round(max(0.0, min(1.0, master["volume"] + step * VOLUME_STEP)), 2)
            mark_dirty()

# ---------------------------------------------------------------------------
# Pad + button lights: computed every frame, only changes are sent
# ---------------------------------------------------------------------------
sent = {}
# Most Push 2 buttons only have white LEDs: brightness is all we can set.
WHITE_ON, WHITE_DIM = 127, 16


def desired_lights():
    want = {}
    sel = ui["selected"]
    play_step = current_step()
    for note in range(push2.PAD_FIRST, push2.PAD_LAST + 1):
        color = push2.OFF
        pending = ui.get("pending")
        pid = note_to_pid(note)
        if pid is not None and pending and pending[0] == pid:
            color = push2.RED if int(time.monotonic() * 4) % 2 else push2.WHITE   # waiting for yes
        elif pid is not None and pid in pad_samples:
            if note in held:
                color = push2.WHITE
            elif not audible(pid):
                color = push2.DARK_GRAY                # muted, or silenced by a solo
            else:
                flashing = 0 <= time.monotonic() - pad_flash.get(pid, -9) < 0.09
                color = push2.WHITE if flashing else pad_led(pid, bright=pid == sel)
        elif pid is not None and browser.active:
            color = push2.DARK_GRAY                    # empty pad: a place to drop a sample
        else:
            s = step_of(note)
            if s is not None and sel is not None:
                g = seq["page"] * STEPS + s             # step in the whole pattern
                if g == play_step:
                    color = push2.WHITE
                elif pattern[sel][g]:
                    color = push2.GREEN
                elif notes_at_step(sel, g):
                    color = push2.BLUE                 # keyboard notes (synths): hold to edit the step
                elif s % 4 == 0:
                    color = push2.DARK_GRAY            # beat markers
        want[("note", note)] = color
    want[("cc", PLAY_BUTTON)] = push2.GREEN if clock["playing"] else push2.OFF
    want[("cc", PAGE_LEFT)] = 127 if seq["page"] > 0 else 0
    want[("cc", PAGE_RIGHT)] = 127 if seq["page"] < seq["bars"] - 1 else 0
    want[("cc", DOUBLE_BUTTON)] = 127 if seq["bars"] * 2 <= MAX_BARS else 0
    want[("cc", FOLLOW_BUTTON)] = WHITE_ON if seq["follow"] else WHITE_DIM
    want[("cc", ZOOM_IN)] = 127 if seq["zoom"] > ZOOM_LEVELS[0] else 0
    want[("cc", ZOOM_OUT)] = 127 if seq["zoom"] < ZOOM_LEVELS[-1] else 0
    want[("cc", MUTE_BUTTON)] = WHITE_ON if (mutes or MUTE_BUTTON in held_buttons) else WHITE_DIM
    want[("cc", SOLO_BUTTON)] = WHITE_ON if (solos or SOLO_BUTTON in held_buttons) else WHITE_DIM
    for i, cc in enumerate(UPPER_BUTTONS):
        name = PAGE_ORDER[i]
        if name in PAGE_LEDS:
            bright, dim = PAGE_LEDS[name]
            want[("cc", cc)] = bright if (cal_ui["on"] or (name == ui["page"] and ui["open"])) else dim
        else:
            want[("cc", cc)] = 0                      # page not built yet
    kit_mode = not kb["mode"] and not kb["levels"]
    want[("cc", SESSION_BUTTON)] = WHITE_ON if kit_mode else WHITE_DIM
    want[("cc", NOTE_BUTTON)] = WHITE_ON if kb["mode"] == "chromatic" else WHITE_DIM
    want[("cc", SCALE_BUTTON)] = WHITE_ON if kb["mode"] == "scale" else WHITE_DIM
    want[("cc", ACCENT_BUTTON)] = WHITE_ON if kb["levels"] else WHITE_DIM
    want[("cc", BROWSE_BUTTON)] = WHITE_ON if browser.active else WHITE_DIM
    want[("cc", UNDO_BUTTON)] = WHITE_ON if history["undo"] else WHITE_DIM
    want[("cc", DELETE_BUTTON)] = WHITE_ON if DELETE_BUTTON in held_buttons else WHITE_DIM
    want[("cc", SHIFT_BUTTON)] = WHITE_ON if SHIFT_BUTTON in held_buttons else WHITE_DIM
    want[("cc", DUPLICATE_BUTTON)] = WHITE_ON if DUPLICATE_BUTTON in held_buttons else WHITE_DIM
    want[("cc", METRONOME_BUTTON)] = WHITE_ON if setup["metronome"] else WHITE_DIM
    red_on, red_dim = PAGE_LEDS.get("PAD", (push2.RED, push2.RED))
    want[("cc", RECORD_BUTTON)] = red_on if rec["on"] else red_dim
    want[("cc", REPEAT_BUTTON)] = WHITE_ON if repeat["on"] else WHITE_DIM
    want[("cc", QUANTIZE_BUTTON)] = WHITE_ON if QUANTIZE_BUTTON in held_buttons else WHITE_DIM
    want[("cc", CONVERT_BUTTON)] = WHITE_ON if CONVERT_BUTTON in held_buttons else WHITE_DIM
    want[("cc", SETUP_BUTTON)] = WHITE_ON if ui.get("settings") else WHITE_DIM
    want[("cc", ADD_TRACK_BUTTON)] = WHITE_ON if proj["open"] else WHITE_DIM
    want[("cc", STOP_CLIP_BUTTON)] = WHITE_DIM
    want[("cc", FIXED_LENGTH_BUTTON)] = WHITE_ON if FIXED_LENGTH_BUTTON in held_buttons else WHITE_DIM
    want[("cc", HELP_BUTTON)] = WHITE_ON if ui.get("help") else WHITE_DIM
    chopping_view = chop_view()
    want[("cc", NEW_BUTTON)] = (WHITE_ON if chop["confirm"] else WHITE_DIM) if chopping_view else 0
    if chopping_view and not browser.active:
        want[("cc", ZOOM_IN)] = 127 if chop["zoom"] < CHOP_ZOOMS[-1] else 0
        want[("cc", ZOOM_OUT)] = 127 if chop["zoom"] > 1 else 0
        red_on, red_dim = PAGE_LEDS.get("PAD", (push2.RED, push2.RED))
        want[("cc", RECORD_BUTTON)] = red_on if chop["tap"] else red_dim
        n = len(chop_state()["markers"])
        for note in range(push2.PAD_FIRST, push2.PAD_LAST + 1):
            k = step_of(note)
            if k is not None:                             # top two rows = slices
                want[("note", note)] = (push2.WHITE if note in held else
                                        (push2.RED if k == chop["cur"] else push2.BLUE) if k < n else push2.OFF)
            elif note_to_pid(note) == ui["selected"] and note not in held:
                want[("note", note)] = push2.RED          # the pad being chopped
    if smp["open"]:
        blink = int(time.monotonic() * 4) % 2
        st = smp["state"]
        red_on, red_dim = PAGE_LEDS.get("PAD", (push2.RED, push2.RED))
        want[("cc", RECORD_BUTTON)] = red_on if st == "recording" else (
            (red_on if blink else red_dim) if st == "waiting" else red_dim)
        for note in KIT_NOTES:
            pid = note_to_pid(note)
            if pid == smp["target"]:
                want[("note", note)] = push2.RED if (blink or st == "recording") else push2.OFF
            elif pid not in pad_samples:
                want[("note", note)] = push2.DARK_GRAY        # empty: tap to sample here
    sel_pad = ui["selected"]
    if kb["mode"] and sel_pad in pad_samples and not browser.active and not smp["open"]:
        for note in range(push2.PAD_FIRST, push2.PAD_LAST + 1):
            n = kb_pad_note(note)
            if n is None:
                continue
            if note in held:
                want[("note", note)] = push2.WHITE
            elif (n - setup["key_root"]) % 12 == 0:
                want[("note", note)] = SEL_LED_BRIGHT          # root: full pad colour
            elif in_key(n):
                want[("note", note)] = push2.LIGHT_GRAY        # in the key: white
            else:
                want[("note", note)] = push2.OFF
        play_step = current_step()
        for note in range(push2.PAD_FIRST, push2.PAD_LAST + 1):   # top rows: steps with notes
            st = step_of(note)
            if st is None:
                continue
            g = seq["page"] * STEPS + st
            if note in held:
                want[("note", note)] = push2.WHITE
            elif g == play_step:
                want[("note", note)] = push2.WHITE
            elif notes_at_step(sel_pad, g):
                want[("note", note)] = SEL_LED_BRIGHT
            else:
                want[("note", note)] = push2.DARK_GRAY if st % 4 == 0 else push2.OFF
    if kb["levels"] and sel_pad in pad_samples and not browser.active and not smp["open"]:
        for k, note in enumerate(KIT_NOTES):
            want[("note", note)] = push2.WHITE if note in held else (SEL_LED_BRIGHT if k >= 8 else SEL_LED_DIM)
    if proj["open"]:
        blink = int(time.monotonic() * 4) % 2
        for k, cc in enumerate(UPPER_BUTTONS):
            want[("cc", cc)] = push2.WHITE if k >= 6 else 0
        for note in range(push2.PAD_FIRST, push2.PAD_LAST + 1):
            want[("note", note)] = push2.OFF
        for k, note in enumerate(KIT_NOTES):
            n = k + 1
            if proj["confirm_delete"] == n:
                want[("note", note)] = push2.RED if blink else push2.OFF
            elif n == proj["cur"]:
                want[("note", note)] = push2.WHITE
            elif proj["info"].get(n):
                want[("note", note)] = push2.BLUE
            else:
                want[("note", note)] = push2.DARK_GRAY
    for k, cc in enumerate(RATE_BUTTONS):
        want[("cc", cc)] = (push2.WHITE if k == repeat["rate"] else push2.DARK_GRAY) if repeat["on"] else 0
    want[("cc", TAP_TEMPO_BUTTON)] = WHITE_ON if TAP_TEMPO_BUTTON in held_buttons else WHITE_DIM
    blink = int(time.monotonic() * 4) % 2
    bright, dim = PAGE_LEDS.get("PATTERN", (push2.BLUE, push2.BLUE))
    want[("cc", SELECT_BUTTON)] = WHITE_ON if pat["row"] == "banks" else WHITE_DIM
    on_first = bool(kit_notes) and ui["selected"] == kit_notes[0]
    on_last = bool(kit_notes) and ui["selected"] == kit_notes[-1]
    want[("cc", OCTAVE_UP)] = WHITE_DIM if (not kit_notes or on_first) else WHITE_ON
    want[("cc", OCTAVE_DOWN)] = WHITE_DIM if (not kit_notes or on_last) else WHITE_ON
    for k, cc in enumerate(LOWER_BUTTONS):
        if pat["row"] == "banks":
            red_on, red_dim = PAGE_LEDS.get("PAD", (push2.RED, push2.RED))
            used = any(k * 16 <= p < k * 16 + 16 for p in pad_samples)
            want[("cc", cc)] = red_on if k == bank["cur"] else (red_dim if used else 0)
            continue
        i = pat["page"] * SLOTS_PER_PAGE + k
        if i == ui.get("confirm_clear"):
            red = PAGE_LEDS.get("PAD", (push2.RED, push2.RED))
            want[("cc", cc)] = red[0] if blink else red[1]
        elif i == pat["queued"]:
            want[("cc", cc)] = bright if blink else dim
        elif i == pat["cur"]:
            want[("cc", cc)] = bright
        elif slots[i] is not None:
            want[("cc", cc)] = dim
        else:
            want[("cc", cc)] = 0
    if not browser.active:
        want[("cc", ARROW_LEFT)] = WHITE_ON if pat["page"] == 0 else WHITE_DIM
        want[("cc", ARROW_RIGHT)] = WHITE_ON if pat["page"] == 1 else WHITE_DIM
    if browser.active:
        want[("cc", ARROW_LEFT)] = 127 if browser.path is not None else 0
        want[("cc", ARROW_RIGHT)] = 127 if browser.items else 0
    if chopping_view and not browser.active:
        vs, ve = chop_window()
        want[("cc", ARROW_LEFT)] = 127 if vs > 0 else 0
        want[("cc", ARROW_RIGHT)] = 127 if ve < len(chop_audio()) else 0
    if browser.active:
        want[("cc", ZOOM_IN)] = want[("cc", ZOOM_OUT)] = 127
        for k, cc in enumerate(UPPER_BUTTONS):
            want[("cc", cc)] = push2.WHITE if k >= 6 else 0
    want[("cc", MASTER_BUTTON)] = WHITE_ON if (ui["page"] == "MASTER" and ui["open"]) else WHITE_DIM
    want[("cc", DEVICE_BUTTON)] = WHITE_ON if device_open() else WHITE_DIM
    want[("cc", MIX_BUTTON)] = WHITE_ON if mix_open() else WHITE_DIM
    want[("cc", AUTOMATE_BUTTON)] = (push2.RED if blink or not clock["playing"] else push2.OFF) \
        if auto["arm"] else (WHITE_DIM if not real_lanes() else push2.WHITE)
    want[("cc", CLIP_BUTTON)] = WHITE_ON if song_open() else (WHITE_DIM if not song_on() else 60)
    if song_open() and not ui.get("settings"):
        for k, cc in enumerate(LOWER_BUTTONS):
            i = song["page"] * 8 + k
            if i < len(song["rows"]):
                now = clock["playing"] and song["pos"] == i
                want[("cc", cc)] = push2.GREEN if now else (push2.WHITE if i == song["sel"] else push2.DARK_GRAY)
            else:
                want[("cc", cc)] = push2.DARK_GRAY if (i == len(song["rows"]) and blink) else 0
        want[("cc", ARROW_LEFT)] = WHITE_ON if song["page"] > 0 else WHITE_DIM
        want[("cc", ARROW_RIGHT)] = WHITE_ON if song["page"] < len(song["rows"]) // 8 else WHITE_DIM
    want[("cc", ADD_DEVICE_BUTTON)] = WHITE_ON if (ui["page"] == "INST" and ui["open"]) else WHITE_DIM
    if mix_open():
        for k, cc in enumerate(LOWER_BUTTONS):
            want[("cc", cc)] = (push2.RED if buses[k].get("mute") else push2.WHITE) if k < NUM_BUSES else 0
    if device_open():
        for k, cc in enumerate(LOWER_BUTTONS):
            if k < NUM_SLOTS:
                s_ = buses[dev["bus"]]["slots"][k]
                want[("cc", cc)] = push2.WHITE if k == dev["slot"] else (push2.DARK_GRAY if s_ else 0)
            else:
                bb = k - NUM_SLOTS
                if buses[bb].get("mute"):
                    want[("cc", cc)] = push2.RED
                else:
                    want[("cc", cc)] = push2.WHITE if bb == dev["bus"] else push2.DARK_GRAY
    if ui.get("pad_move"):
        for i, cc in enumerate(UPPER_BUTTONS):
            want[("cc", cc)] = 127 if i in (0, 1, 7) else 0
    return want


SEL_LED_BRIGHT, SEL_LED_DIM = 62, 63           # the selected pad's colour (keyboard / levels)


def set_pad_led(key, d, lo):
    app_state[key] = clamp(app_state[key] + 2 * d, lo, 100)
    app_state["led_save"] = time.monotonic() + 1.0          # saved once you stop turning


def sync_pad_palette():
    """Keep the Push's palette entries for the 16 kit pads in step with their colours."""
    changed = False
    sel_ = ui["selected"] if ui["selected"] is not None else 0
    for k in range(17):
        pid = bank["cur"] * 16 + k if k < 16 else sel_
        lit = app_state["pad_bright"] / 100
        unlit = lit * app_state["pad_unlit"] / 100
        r, g, b = pad_rgb(pid)
        for entry, rgb in ((PAD_LED_FIRST + 2 * k, tuple(int(x * lit) for x in (r, g, b))),
                           (PAD_LED_FIRST + 2 * k + 1, tuple(int(x * unlit) for x in (r, g, b)))):
            if palette_sent.get(entry) != rgb:
                rr, gg, bb = rgb
                midi.send(midi.mido.Message("sysex", data=[0x00, 0x21, 0x1D, 0x01, 0x01, 0x03, entry,
                          rr & 127, rr >> 7, gg & 127, gg >> 7, bb & 127, bb >> 7, 0, 0]))
                palette_sent[entry] = rgb
                changed = True
    if changed:
        midi.send(midi.mido.Message("sysex", data=[0x00, 0x21, 0x1D, 0x01, 0x01, 0x05]))


def update_lights():
    if app_state.get("led_save") and time.monotonic() > app_state["led_save"]:
        app_state["led_save"] = None
        save_app_state()
    sync_pad_palette()
    for key, value in desired_lights().items():
        if sent.get(key) != value:
            kind, num = key
            if kind == "note":
                midi.pad_color(num, value)
            else:
                midi.send(midi.mido.Message("control_change", control=num, value=value))
            sent[key] = value

# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Look: "Pro Sans" with MPC60 colours. Dark grey cards float on black.
# Every card: small grey caps label on top, big bright value under it.
# Red = pad, blue = pattern, white = master.
# ---------------------------------------------------------------------------
from PIL import ImageFont

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def ui_font(name, size):
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return load_font(size)                 # fonts folder missing: fall back


# ---------------------------------------------------------------------------
# Visual Design System v2 ("Hardware Console", 2026-11-24). All screen colours and
# fonts live here; the draw_... functions only use these names.
# Fonts (SIL OFL, in fonts/): Barlow Semi Condensed (values, labels, tabs),
# Barlow (names, sentences), IBM Plex Mono (technical metadata).
# ---------------------------------------------------------------------------
def first_font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(os.path.join(FONT_DIR, n), size)
        except OSError:
            continue
    return load_font(size)


_SC_BOLD = ("BarlowSemiCondensed-Bold.ttf", "InterDisplay-SemiBold.otf")
_SC_SEMI = ("BarlowSemiCondensed-SemiBold.ttf", "Inter-SemiBold.otf")
_SC_MED = ("BarlowSemiCondensed-Medium.ttf", "Inter-Medium.otf")
_B_SEMI = ("Barlow-SemiBold.ttf", "InterDisplay-Bold.otf")
_B_MED = ("Barlow-Medium.ttf", "Inter-Medium.otf")
_MONO = ("IBMPlexMono-Medium.ttf", "Inter-Medium.otf")
F_LABEL = first_font(_SC_MED, 11)          # caps labels
F_TAB = first_font(_SC_SEMI, 13)           # page tabs
F_VALUE = first_font(_SC_BOLD, 22)         # parameter values
F_VALUE_S = first_font(_SC_BOLD, 18)       # ... when a long value would not fit
F_UNIT = first_font(_SC_MED, 13)
F_NAME = first_font(_B_SEMI, 28)           # sample / program names
F_SMALL = first_font(_SC_SEMI, 17)
F_ROW = first_font(_SC_MED, 13)
F_BODY = first_font(_B_MED, 14)            # sentences
F_MONO = first_font(_MONO, 11)             # technical metadata
F_MONO_S = first_font(_MONO, 10)

# Neutral greys, every channel a multiple of 8: the Push screen gets RGB565 (5-6-5 bits),
# so these stay exactly grey there. (The spec's blue-greys looked plainly blue on the
# hardware, 2026-11-24.)
BG = (8, 8, 8)                     # base background
PANEL = (24, 24, 24)               # panels, parameter blocks
RAISED = (40, 40, 40)              # raised controls, inactive tabs
INACTIVE = (56, 56, 56)            # inactive controls
BORDER = (64, 64, 64)              # borders
TXT = (240, 240, 232)              # primary text (#F1EFE7, warm white)
GREY = (160, 160, 160)             # secondary text
DIM = (96, 96, 96)                 # disabled text
SEL_OUTLINE = (248, 248, 240)      # selected outline
EMPTY = (16, 16, 16)               # unused encoder / tab columns
NAV = (240, 240, 232)              # off-white: navigation, actions, selection (was magenta)
CYAN = (0, 212, 232)               # #00D4E8 timing / information
MINT = (0, 230, 154)               # #00E69A playback / positive active
AMBER = (255, 157, 40)             # #FF9D28 processing / synthesis
VIOLET = (166, 106, 255)           # #A66AFF pitch / modulation
WARN = (255, 64, 92)               # #FF405C recording / warnings
GRID, GRID_MAJOR, AXIS = (32, 32, 32), (48, 48, 48), (72, 72, 72)   # graph lines
BLACK, CARD, CARD2 = BG, PANEL, RAISED     # older names, same tokens
OFF_STEP, OFF_BEAT, SILENT = (32, 32, 32), (40, 40, 40), (64, 64, 64)
# status cards: PAD red and PATTERN blue as Luis matched them to the button lights (2026-11-24,
# Shift + Setup); a later calibration of the PAD / PATTERN tabs updates them too
RED, BLUE, MASTER_C = (255, 74, 54), (0, 124, 255), (235, 235, 240)
# values whose meaning gets a colour (the rest stay primary text)
VALUE_TINTS = {**{k: CYAN for k in ("SWING", "HUMAN", "BPM", "TIME", "ATTACK", "RELEASE", "DECAY", "GLIDE",
                                    "NUDGE", "START", "END", "BUFFER", "REPEATS")},
               **{k: VIOLET for k in ("PITCH", "TUNE", "ROOT", "NOTE", "OCTAVE", "KEY", "DETUNE", "VARY")}}
for _name, _c in (("PAD", RED), ("SAMPLE", RED), ("PATTERN", BLUE), ("SETUP", MASTER_C)):
    PAGES[_name]["colour"] = _c
TRACK = 0.4                                    # letter spacing of caps labels

# The Push's LEDs show colour differently from the screen (green comes out much
# stronger), so the buttons get their own values, picked by eye with led_test.py.
LED_COLOURS = {"PAD": (255, 0, 0), "SAMPLE": (255, 0, 0), "PATTERN": (0, 40, 255), "SETUP": (200, 200, 200)}

SLOT_W = W // 8
PAGE_LEDS = {}                        # page -> (bright palette index, dim palette index)


def setup_button_colours():
    """Teach the Push a bright + dim colour per page (palette entries 1-14), then apply."""
    idx = 1
    for name in PAGE_ORDER:
        if name is None or not PAGES[name]["ready"]:
            continue
        r, g, b = LED_COLOURS.get(name, PAGES[name]["colour"])
        for (rr, gg, bb) in ((r, g, b), (r // 6, g // 6, b // 6)):
            data = [0x00, 0x21, 0x1D, 0x01, 0x01, 0x03, idx,
                    rr & 127, rr >> 7, gg & 127, gg >> 7, bb & 127, bb >> 7, 0, 0]
            midi.send(midi.mido.Message("sysex", data=data))
            idx += 1
        PAGE_LEDS[name] = (idx - 2, idx - 1)
    midi.send(midi.mido.Message("sysex", data=[0x00, 0x21, 0x1D, 0x01, 0x01, 0x05]))


def caps(d, xy, s, fill, anchor="la"):
    """Small tracked caps label."""
    s = s.upper()
    if anchor[0] in "rm":
        width = sum(F_LABEL.getlength(c) + TRACK for c in s) - TRACK
        xy = (xy[0] - (width if anchor[0] == "r" else width / 2), xy[1])
    x, y = xy
    start = x
    for ch in s:
        d.text((x, y), ch, font=F_LABEL, fill=fill)
        x += F_LABEL.getlength(ch) + TRACK
    return start if anchor[0] == "r" else x       # right-aligned: where the text starts


def card(d, box, accent=None, side="left"):
    """A panel. accent = a 3 px status bar on the left (no coloured top borders in v2)."""
    d.rounded_rectangle(box, 3, fill=CARD)
    if accent and side == "left":
        x0, y0, x1, y1 = box
        d.rectangle([x0, y0 + 1, x0 + 2, y1 - 1], fill=accent)


def fit_font(text, width, fonts=None):
    """The biggest value font the text fits in."""
    for f in fonts or (F_VALUE, F_VALUE_S, F_SMALL):
        if f.getlength(text) <= width:
            return f
    return (fonts or (F_VALUE, F_VALUE_S, F_SMALL))[-1]


def param_cell(d, slot, label, value, unit="", fill=TXT, mark=None, span=1, label_fill=GREY, bg=None):
    """One Hardware Console parameter block under encoder `slot` (span = columns)."""
    x0 = slot * SLOT_W + 3
    x1 = x0 + span * SLOT_W - 6
    d.rounded_rectangle([x0, 20, x1, 54], 2, fill=bg or PANEL)
    cx = (x0 + x1) / 2
    caps(d, (cx, 21), label, label_fill, anchor="ma")
    action = value.endswith(">") or unit == ">"
    if action:                                          # an action, not a value: a button-like block
        value = value.rstrip(" >")
        unit = ""
        d.rounded_rectangle([x0, 20, x1, 54], 2, fill=RAISED, outline=BORDER)
        caps(d, (cx, 21), label, label_fill, anchor="ma")
    uw = F_UNIT.getlength(unit) + 2 if unit else 0
    f = fit_font(value, x1 - x0 - 10 - uw - (14 if action else 0))
    vw = f.getlength(value)
    vx = cx - (vw + uw) / 2
    if action:
        vx -= 6
        d.text((vx + vw + 5, 50), "›", font=F_VALUE, fill=NAV, anchor="ls")
    d.text((vx, 51), value, font=f, fill=fill, anchor="ls")
    if unit:
        d.text((vx + vw + 2, 51), unit, font=F_UNIT, fill=GREY, anchor="ls")
    if mark:
        d.ellipse([x1 - 9, 23, x1 - 4, 28], fill=mark)


def empty_cell(d, slot):
    x0 = slot * SLOT_W + 3
    d.rounded_rectangle([x0, 20, x0 + SLOT_W - 6, 54], 2, fill=EMPTY)


def value_tint(param):
    return VALUE_TINTS.get(param.label, TXT)


def split_value(s):
    """'80%' -> ('80', '%'),  'D3 +2' -> ('D3', '+2'),  'pattern' -> ('pattern', '')."""
    if s.endswith("%"):
        return s[:-1], "%"
    if " " in s:
        v, u = s.split(" ", 1)
        return v, u
    return s, ""


def value_unit(d, xy, value, unit, fill=TXT, font=F_VALUE, unit_dy=5):
    d.text(xy, value, font=font, fill=fill)
    end = xy[0] + font.getlength(value)
    if unit:
        d.text((end + 2, xy[1] + unit_dy), unit, font=F_UNIT, fill=GREY)


def strip_open():
    return ui["open"] or bool(held_steps) or bool(held_note_steps) or bool(ui.get("settings"))


def strip_height():
    return 56 if strip_open() else 18


def mix_bus_param(b):
    """Knobs 1-4: a bus's VOLUME or PAN, whichever knob 8 (VOL/PAN) is set to."""
    def get():
        return buses[b].get("pan", 0) if ui.get("mix_mode") == "PAN" else buses[b].get("vol", 0)

    def put(d):
        if ui.get("mix_mode") == "PAN":
            buses[b]["pan"] = int(clamp(buses[b].get("pan", 0) + d, -50, 50))
        else:
            buses[b]["vol"] = int(clamp(buses[b].get("vol", 0) + d, -60, 6))

    def show(v):
        if ui.get("mix_mode") == "PAN":
            return pan_text(v)
        return "-inf" if v <= -60 else f"{v:+d} dB"
    return Param(f"BUS {b + 1}", get, put, show=show).auto_as(
        lambda: f"busmix:{b}:{'pan' if ui.get('mix_mode') == 'PAN' else 'vol'}")


def mix_knobs():
    master_vol = Param("MASTER", lambda: round(master["volume"] * 100),
                       lambda d: master.update(volume=round(clamp(master["volume"] + d * VOLUME_STEP, 0.0, 1.0), 2)),
                       show=lambda v: f"{v}%").auto_as(lambda: "mastervol")
    mode = Param("VOL/PAN", lambda: ui.get("mix_mode", "VOL"),
                 lambda d: ui.update(mix_mode="PAN" if d > 0 else "VOL"), per=4)
    return [mix_bus_param(b) for b in range(NUM_BUSES)] + [master_vol, None, None, mode]


def mix_open():
    return ui["page"] == "MIX" and ui["open"]


def level_meter(d, x, y0, y1, levels, lo=-48.0, hi=6.0, dim=False, w=5):
    """A stereo peak meter in segments: green, then yellow above -6 dB, red over 0."""
    my = lambda db: y1 - (y1 - y0) * max(0.0, min(1.0, (db - lo) / (hi - lo)))
    for c_, lv in enumerate(levels):
        mx = x + c_ * (w + 2)
        d.rectangle([mx, y0, mx + w, y1], fill=(24, 24, 24))
        db_ = 20 * math.log10(max(lv, 1e-6))
        top_ = my(db_)
        yy = y1
        while yy > top_ and yy > y0:                       # 3 px segments, 1 px gaps
            seg_db = lo + (y1 - yy) / (y1 - y0) * (hi - lo)
            c = (96, 96, 96) if dim else (MINT if seg_db < -6 else (AMBER if seg_db < 0 else WARN))
            ya = max(top_, yy - 3)
            if ya <= yy - 1:
                d.rectangle([mx, ya, mx + w, yy - 1], fill=c)
            yy -= 4
    d.line([(x - 2, my(0)), (x + len(levels) * (w + 2), my(0))], fill=GREY)
    return my


def draw_mixer(d):
    """Mix page: five strips side by side - BUS 1-4 (each straight to the master, not
    into each other) and MASTER: meter + fader mark, level, pan, and the effect chain."""
    top = strip_height() + 2
    lo, hi = -48.0, 6.0
    y0, y1 = top + 18, 141
    W5 = 960 / 5
    mode = ui.get("mix_mode", "VOL")
    for b in range(NUM_BUSES + 1):
        x = int(b * W5) + 3
        xr = int((b + 1) * W5) - 3
        if b < NUM_BUSES:
            col, bus = BUS_COLOURS[b], buses[b]
            muted = bus.get("mute", False)
            name = f"BUS {b + 1}"
            vol, pan = bus.get("vol", 0), bus.get("pan", 0)
            lv = [bus_meters[b]] * 2
            n_pads = sum(1 for nn in kit_notes if padp[nn].get("bus") == b)
            sub = f"{n_pads} PAD" + ("" if n_pads == 1 else "S")
            lines = [SHORT_NAMES.get(s_["type"], s_["type"]) + ("" if s_["on"] else " · OFF")
                     for s_ in bus["slots"] if s_] or ["NO FX"]
            vol_txt, unit = ("-inf", "") if vol <= -60 else (f"{vol:+d}", "dB")
        else:
            col, muted, name = MASTER_C, False, "MASTER"
            vol_txt, unit, pan = f"{round(master['volume'] * 100)}", "%", None
            lv = meters["out"]
            sub = ""
            m = master_fx
            lines = [("BUS COMP " + (f"{m['ratio']}:1" if m["ratio"] else "OFF")), f"LIMIT {CEILING_DB} dB"]
        card(d, [x, top, xr, 145], WARN if muted else col)
        d.text((x + 10, top + 8), name, font=F_NAV, fill=WARN if muted else col, anchor="lm")
        if muted:
            d.text((x + 14 + F_NAV.getlength(name), top + 8), "MUTED", font=F_MONO_S, fill=WARN, anchor="lm")
        elif sub:
            d.text((xr - 8, top + 8), sub, font=F_MONO_S, fill=GREY, anchor="rm")
        my = level_meter(d, x + 10, y0, y1, lv, lo, hi, dim=muted)
        if b < NUM_BUSES:                                       # fader mark
            fy = my(vol if vol > -60 else lo)
            d.rectangle([x + 7, fy - 1, x + 26, fy + 1], fill=TXT)
        tx = x + 36
        active = mode == "VOL" or b == NUM_BUSES
        d.text((tx, top + 30), vol_txt, font=F_VALUE, fill=(DIM if muted else TXT) if active else GREY, anchor="ls")
        if unit:
            d.text((tx + F_VALUE.getlength(vol_txt) + 2, top + 30), unit, font=F_UNIT, fill=GREY, anchor="ls")
        if pan is not None:                                      # pan: a centred bar
            px0, px1, py = tx, xr - 10, top + 40
            pc = (px0 + px1) / 2
            d.rectangle([px0, py - 2, px1, py + 2], fill=(32, 32, 32))
            pv = pc + (px1 - px0) / 2 * pan / 50
            d.rectangle([min(pc, pv), py - 2, max(pc, pv), py + 2], fill=col if mode == "PAN" else GREY)
            d.line([(pc, py - 4), (pc, py + 4)], fill=GREY)
            d.text((px1, top + 30), pan_text(pan), font=F_NAV, fill=TXT if mode == "PAN" else GREY, anchor="rs")
        for i, nm in enumerate(lines[:4]):
            d.text((tx, top + 54 + i * 12), nm, font=F_NAV, fill=DIM if nm == "NO FX" else GREY, anchor="lm")


def page_knobs():
    if ui["page"] in PAD_FX:
        return pad_fx_knobs(ui["page"])
    if ui["page"] == "SONG":
        if PAGES["SONG"]["knobs"][0] is None:
            PAGES["SONG"]["knobs"] = song_knobs()
        return PAGES["SONG"]["knobs"]
    if ui["page"] == "MIX" and PAGES["MIX"]["knobs"][0] is None:      # made once, on first use
        PAGES["MIX"]["knobs"] = mix_knobs()
    if ui["page"] == "INST":
        return inst_knobs()
    return device_knobs() if ui["page"] == "DEVICE" else PAGES[ui["page"]]["knobs"]


def draw_tabs(d):
    for slot, name in enumerate(PAGE_ORDER):
        if name is None:
            continue
        x = slot * SLOT_W + 10
        page = PAGES[name]
        is_open = name == ui["page"] and ui["open"]
        caps(d, (x, 2), PAGE_LABELS.get(name, name), TXT if is_open else (GREY if page["ready"] else DIM))
        if is_open:
            d.rectangle([x, 14, x + SLOT_W - 16, 15], fill=page["colour"])


def draw_tabs_row(d):
    """Page tabs over the upper buttons: solid blocks, the open page magenta."""
    for slot, name in enumerate(PAGE_ORDER):
        x0 = slot * SLOT_W + 3
        x1 = x0 + SLOT_W - 6
        if name is None:
            d.rectangle([x0, 1, x1, 16], fill=EMPTY)
            continue
        page = PAGES[name]
        is_open = name == ui["page"] and ui["open"]
        col = tab_colour(name)                              # the page's colour = its button's LED
        light = sum(col) / 3 > 170
        d.rounded_rectangle([x0, 1, x1, 16], 2, fill=col if is_open else RAISED)
        d.text(((x0 + x1) / 2, 9), PAGE_LABELS.get(name, name), font=F_TAB, anchor="mm",
               fill=(BG if light else TXT) if is_open else (GREY if page["ready"] else DIM))


def group_colour():
    """The open page's colour: its tab (or the bus on the Device page)."""
    page = ui["page"]
    if page == "DEVICE":
        return BUS_COLOURS[dev["bus"]]
    return tab_colour(page) if page in PAGES else NAV


def group_bg(col):
    return shade(col, PANEL, 0.13)


def draw_group(d, slots, col):
    """Chrome-style tab group: the open page's tab, a band under it and a frame round
    the knobs that belong to it, all in the page's colour."""
    if not slots:
        return
    gx0, gx1 = min(slots) * SLOT_W + 1, (max(slots) + 1) * SLOT_W - 1
    page = ui["page"]
    if page in PAGE_ORDER:
        t = PAGE_ORDER.index(page)
        tx0, tx1 = t * SLOT_W + 3, (t + 1) * SLOT_W - 3
        d.rectangle([tx0, 15, tx1, 18], fill=col)                 # the tab runs into the group
        d.rectangle([min(tx0, gx0), 17, max(tx1, gx1), 18], fill=col)
    else:
        d.rectangle([gx0, 17, gx1, 18], fill=col)
    d.line([(gx0, 18), (gx0, 56)], fill=col)
    d.line([(gx1, 18), (gx1, 56)], fill=col)
    d.line([(gx0, 56), (gx1, 56)], fill=col)


def draw_strip(d):
    draw_tabs_row(d)
    if held_steps:
        draw_step_editor(d)
        return
    if held_note_steps:
        draw_note_editor(d)
        return
    if ui.get("settings"):
        draw_settings(d)
        return
    if not ui["open"]:
        return
    col = group_colour()
    bg = group_bg(col)
    used = []
    for slot, param in enumerate(page_knobs()):
        if param is None:
            empty_cell(d, slot)
            continue
        used.append(slot)
        v, u = split_value(param.text())
        mark = None
        if auto_has(param):                                  # automated in this pattern
            mark = WARN if (param.target() in auto["writing"] and auto["arm"]) else (240, 120, 130)
        param_cell(d, slot, param.label, v, u, fill=value_tint(param) if param.active() else DIM, mark=mark, bg=bg)
    draw_group(d, used, col)



def draw_settings(d):
    ctx = ui["settings"]
    knobs = SETTINGS[ctx]
    sel_ = ui["selected"]
    colour = pad_screen_colour(sel_) if ctx == "PAD" and sel_ is not None else BLUE
    gcol = group_colour()
    used = []
    for slot, param in enumerate(knobs):
        if param is None:
            continue
        used.append(slot)
        v, u = split_value(param.text())
        param_cell(d, slot, param.label, v, u, fill=value_tint(param) if param.active() else DIM, bg=group_bg(gcol))
    draw_group(d, used, gcol)
    def title(slot, span, sub, sub_fill=GREY):
        x0 = slot * SLOT_W + 3
        d.rounded_rectangle([x0, 20, x0 + span * SLOT_W - 6, 54], 2, fill=PANEL)
        d.text((x0 + 8, 29), f"SETTINGS · {ctx}", font=F_NAV, fill=NAV, anchor="lm")
        d.text((x0 + 8, 44), sub, font=F_BODY, fill=sub_fill, anchor="lm")
        return x0
    if ctx == "SAMPLE":
        title(6, 2, "slices · stems")
        return
    if ctx == "GENERAL":
        draw_curve(d, 3 * SLOT_W + 3, "SETTINGS · GENERAL")  # pad response graph next to its knobs
        return
    if ctx == "PAD" and sel_ is not None:
        x0 = title(5, 2, fit_text(f"{pad_name(sel_)}  {pad_label(sel_)}", F_BODY, 2 * SLOT_W - 66), TXT)
        sx1 = x0 + 2 * SLOT_W - 12                          # the pad's colour
        d.rectangle([sx1 - 44, 23, sx1, 51], fill=pad_screen_colour(sel_), outline=SEL_OUTLINE)
    elif ctx == "PATTERN":
        title(4, 3, "nothing here yet")
    elif ctx == "SCALE":
        title(4, 3, f"key of the project: {NOTE_NAMES_12[setup['key_root']]} {setup['key_scale'].lower()}", TXT)
    else:
        title(4, 3, "")


def draw_curve(d, x, heading=None):
    """Pad response graph: hit strength (right) -> volume (up). Dot = your last hit."""
    d.rounded_rectangle([x, 20, x + 2 * SLOT_W - 6, 54], 2, fill=PANEL)
    d.text((x + 8, 29), heading or "PAD RESPONSE", font=F_NAV, fill=NAV if heading else GREY, anchor="lm")
    gx0, gy0, gw, gh = x + 112, 24, 112, 26
    d.rectangle([gx0, gy0, gx0 + gw, gy0 + gh], fill=(16, 16, 16), outline=GRID_MAJOR)
    for k in (1, 2, 3):
        d.line([(gx0 + k * gw / 4, gy0), (gx0 + k * gw / 4, gy0 + gh)], fill=GRID)
    pts = [(gx0 + i, gy0 + gh - velo_gain(i / gw * 127) * gh) for i in range(gw + 1)]
    d.polygon([(gx0, gy0 + gh)] + pts + [(gx0 + gw, gy0 + gh)], fill=(0, 64, 72))
    d.line(pts, fill=CYAN, width=2)
    th = gx0 + setup["threshold"] / 127 * gw
    d.line([(th, gy0), (th, gy0 + gh)], fill=WARN)
    last = ui.get("last_velocity")
    if last:
        d.text((x + 8, 44), f"LAST HIT {last}", font=F_MONO_S, fill=GREY, anchor="lm")
        lx = gx0 + last / 127 * gw
        ly = gy0 + gh - velo_gain(last) * gh
        d.ellipse([lx - 3, ly - 3, lx + 3, ly + 3], fill=TXT)


def draw_note_editor(d):
    """Holding a top-row step in a keyboard mode: 1 velocity, 2 nudge, 3 note, 4 length."""
    info = next(iter(held_note_steps.values()))
    evs = notes_at_step(info["pad"], info["s"])
    colour = pad_screen_colour(info["pad"])
    if not evs:
        return
    pos, n, vel, length = evs[0]
    nudge = pos - round(pos)
    chord = f" +{len(evs) - 1}" if len(evs) > 1 else ""
    for slot, (label, value, unit) in enumerate([
            ("VELOCITY", str(vel), ""), ("NUDGE", f"{'+' if nudge > 0 else ''}{round(nudge * 100)}", "%"),
            ("NOTE", note_name(n), chord), ("LENGTH", f"{length:g}", "STEPS" if length != 1 else "STEP")]):
        param_cell(d, slot, label, value, unit, fill=(CYAN, CYAN, VIOLET, CYAN)[slot] if slot else TXT)
    if any(step_knobs()):                          # knobs 5-8: this step's automation
        draw_step_auto(d, info)
        return
    x = 4 * SLOT_W + 6
    card(d, [x, 19, x + 3 * SLOT_W - 14, 52])
    caps(d, (x + 8, 24), f"STEP {info['s'] + 1}", colour)
    d.text((x + 8, 33), "hold + play a key: change the note", font=F_BODY, fill=GREY)


def step_knobs():
    """While a step is held: the open page's knobs that can be automated on it -
    3-8 for a kit step (1-2 = velocity, nudge), 5-8 for a note step (1-4 = the note)."""
    if not ui["open"] or ui.get("settings"):
        return [None] * 8
    notes_only = held_steps and all(i.get("notes") for i in held_steps.values())
    first = 4 if held_note_steps and not held_steps else (0 if notes_only else 2)
    ks = list(page_knobs()) + [None] * 8
    return [k if (i >= first and k is not None and getattr(k, "target", None)) else None
            for i, k in enumerate(ks[:8])]


def held_any_steps():
    return list(held_steps.values()) + list(held_note_steps.values())


def step_automation(i, value):
    """Hold step pad(s) + turn knob 3-8: set that setting for those steps only."""
    param = step_knobs()[i]
    key = param.target() if param is not None else None
    r = auto_resolve(key) if key else None
    if r is None:
        toast("that knob can't be automated" if param is None or key is None else "nothing to automate here")
        return
    checkpoint("step automation", key=("stepauto", key, tuple(sorted(held_steps)) + tuple(sorted(held_note_steps))))
    fresh = key not in auto_lanes or all(x is None for x in auto_lanes[key])
    lane = auto_lanes.setdefault(key, [None] * MAX_STEPS)
    live = r[0]()
    if fresh:                                     # the other steps keep today's value
        lane[:pattern_length()] = [live] * pattern_length()
    for info in held_any_steps():
        info["edited"] = True
        s_ = info["s"]
        r[1](lock_value(lane[s_]) if lane[s_] is not None else live)
        param.ticks = 0
        param.turn(value if param.per == 1 else (param.per if (value if value < 64 else value - 128) > 0 else 128 - param.per))
        lane[s_] = {"v": r[0]()}
    r[1](live)                                    # the knob itself stays put; playback does the rest
    mark_dirty()


def draw_step_editor(d):
    """Holding a step pad: knob 1 = velocity, knob 2 = nudge, knobs 3-8 = the page's
    settings on this step only (automation; red dot = this step has its own value)."""
    info = next(iter(held_steps.values()))
    ev = pattern[info["pad"]][info["s"]]
    if not ev:
        if info.get("notes"):                      # a synth's note step (kit view)
            evs = notes_at_step(info["pad"], info["s"])
            if any(step_knobs()[:2]):                  # knobs 1-2 are settings too: the step in the tabs
                d.rectangle([6 * SLOT_W, 0, W, 17], fill=BLACK)
                caps(d, (6 * SLOT_W + 10, 2), f"STEP {info['s'] + 1} · " +
                     " ".join(note_name(e[1]) for e in evs[:3]), RED)
                draw_step_auto(d, info)
                return
            x = 6
            card(d, [x, 19, x + 2 * SLOT_W - 14, 52], pad_screen_colour(info["pad"]), side="top")
            caps(d, (x + 8, 24), f"STEP {info['s'] + 1}  ·  {pad_label(info['pad'])}", RED)
            d.text((x + 8, 33), " ".join(note_name(e[1]) for e in evs[:6]) or "-", font=F_BODY, fill=TXT)
            if not draw_step_auto(d, info):
                hint_step_auto(d, info)
        return
    vel, nudge = step_event(ev)
    ms = nudge * samples_per_step() / SR * 1000
    many = f" +{len(held_steps) - 1}" if len(held_steps) > 1 else ""
    ch = step_chance(info["pad"], info["s"])
    if SHIFT_BUTTON in held_buttons:                     # Shift + knob 1 = CHANCE
        first_cell = (f"STEP {info['s'] + 1}{many} CHANCE", str(ch), "%")
    else:
        first_cell = (f"STEP {info['s'] + 1}{many} VELO", str(vel), "" if ch >= 100 else f"· {ch}%")
    for slot, (label, value, unit) in enumerate([
            first_cell,
            ("NUDGE", f"{'+' if nudge > 0 else ''}{round(nudge * 100)}", f"% {ms:+.0f}ms")]):
        param_cell(d, slot, label, value, unit, fill=CYAN if slot else TXT, label_fill=NAV if slot == 0 else GREY)
    if not draw_step_auto(d, info):
        hint_step_auto(d, info)


def hint_step_auto(d, info):
    x = 2 * SLOT_W + 6
    card(d, [x, 19, x + 3 * SLOT_W - 14, 52])
    caps(d, (x + 8, 24), pad_label(info["pad"]), GREY)
    d.text((x + 8, 33), "open a synth / FX page: its knobs set this step", font=F_ROW, fill=DIM)


def draw_step_auto(d, info):
    """The held step's own values for the automatable knobs. False = none on this page."""
    drawn = False
    for i, param in enumerate(step_knobs()):
        if param is None:
            continue
        drawn = True
        key = param.target()
        lane = auto_lanes.get(key) if key else None
        here = lock_value(lane[info["s"]]) if lane and isinstance(lane[info["s"]], dict) else None
        try:
            v_, u_ = split_value(param.show(here if here is not None else param.get()))
        except Exception:
            v_, u_ = "-", ""
        param_cell(d, i, param.label, v_, u_, fill=TXT if here is not None else DIM,
                   mark=WARN if here is not None else None)
    return drawn

F_NAME_S = first_font(_B_SEMI, 21)


def fit_text(text, font, width):
    """Cut text (with …) to fit `width` pixels."""
    if font.getlength(text) <= width:
        return text
    while text and font.getlength(text + "…") > width:
        text = text[:-1]
    return text + "…"


def draw_left(d):
    """The context rail: PAD (red), PATTERN (blue), MASTER (white) status cards."""
    sel = ui["selected"]
    top = strip_height() + 2
    opened = strip_open()
    # pad card
    card(d, [6, top, 214, 95], RED)
    if sel is None or sel not in pad_samples:
        d.text((14, top + 9), "PAD", font=F_TAB, fill=RED, anchor="lm")
        name, colour = "MPushC", TXT
    else:
        d.text((14, top + 9), f"PAD {pad_name(sel)}", font=F_TAB, fill=RED, anchor="lm")
        meta = f"{len(pitched[sel]) / SR:.2f}s {note_name(padp[sel]['root'] + tune[sel])}"
        d.text((207, top + 9), meta, font=F_MONO, fill=GREY, anchor="rm")
        name, colour = os.path.splitext(pad_samples[sel][0])[0], pad_screen_colour(sel)
        if not opened:                                   # room for a second line
            b_ = padp[sel].get("bus")
            info = ("SYNTH · " if is_inst(sel) else "") + ("MASTER" if b_ is None else f"BUS {b_ + 1}") + \
                f" · {padp[sel]['mode'].upper()} · VOL {padp[sel]['volume']}%"
            d.text((14, top + 24), info, font=F_MONO_S, fill=DIM, anchor="lm")
    f = F_NAME_S if opened else F_NAME
    d.text((14, 89 if not opened else top + 25), fit_text(name, f, 194), font=f, fill=colour,
           anchor="ls" if not opened else "lm")
    # pattern card
    card(d, [6, 99, 214, 120], BLUE)
    d.text((14, 110), f"P{pat['cur'] + 1}", font=F_TAB, fill=BLUE, anchor="lm")
    bpm = str(clock["bpm"])
    d.text((40, 110), bpm, font=F_SMALL, fill=TXT if clock["playing"] else GREY, anchor="lm")
    d.text((42 + F_SMALL.getlength(bpm), 111), "BPM", font=F_LABEL, fill=DIM, anchor="lm")
    playing_bar = current_step() // STEPS if current_step() >= 0 else -1
    for i in range(seq["bars"]):
        x = 104 + i * 9
        d.rectangle([x, 105, x + 6, 115], fill=BLUE if i == playing_bar else None,
                    outline=TXT if i == seq["page"] else BORDER)
    tags = (f"M{len(mutes)} " if mutes else "") + (f"S{len(solos)}" if solos else "")
    x = 207
    if clock["playing"] and clock["tick"] < -1:
        beats_left = min(4 * max(1, setup["count_in"]), (-1 - clock["tick"]) // 4 + 1)
        caps(d, (207, 105), f"COUNT-IN {beats_left}", WARN, anchor="ra")
    elif rec["on"]:
        x = caps(d, (x, 105), "REC", WARN, anchor="ra")
        d.ellipse([x - 10, 106, x - 4, 112], fill=WARN)
        x -= 18
    if repeat["on"]:
        caps(d, (x, 105), RATES[repeat["rate"]][0], CYAN, anchor="ra")
    elif tags and not rec["on"]:
        caps(d, (207, 105), tags.strip(), BLUE, anchor="ra")
    # master card
    card(d, [6, 124, 214, 145], MASTER_C)
    caps(d, (14, 129), "MASTER", GREY)
    vol = str(round(master["volume"] * 100))
    d.text((72, 135), vol, font=F_SMALL, fill=TXT, anchor="lm")
    d.text((74 + F_SMALL.getlength(vol), 136), "%", font=F_LABEL, fill=DIM, anchor="lm")
    cpu = audio_state["cpu"]
    if time.monotonic() - audio_state.get("xrun_t", -99) < 3:
        d.text((207, 135), f"DROPOUT {audio_state['xruns']}", font=F_MONO, fill=WARN, anchor="rm")
    else:
        d.text((207, 135), f"CPU {cpu:.0f}%", font=F_MONO, fill=WARN if cpu > 70 else GREY, anchor="rm")

F_NAV = first_font(_SC_SEMI, 12)


def nav_cell(d, k, text, colour, active=False, sub=None, sub_colour=None):
    """Bottom navigation: the label for lower button k (+ an optional quiet suffix)."""
    x = k * SLOT_W + 10
    d.text((x, 151), text, font=F_NAV, fill=colour, anchor="lm")
    if sub:
        d.text((x + F_NAV.getlength(text) + 5, 151), sub, font=F_NAV, fill=sub_colour or DIM, anchor="lm")
    if active:
        d.rectangle([k * SLOT_W + 4, 158, (k + 1) * SLOT_W - 4, 159], fill=colour)


def draw_bank_row(d):
    """Bottom edge in BANKS mode: banks A-H in red (pad colour)."""
    for k in range(NUM_BANKS):
        count = sum(1 for p in pad_samples if k * 16 <= p < k * 16 + 16)
        cur = k == bank["cur"]
        nav_cell(d, k, f"BANK {BANK_NAMES[k]}", RED if cur else (GREY if count else DIM), active=cur,
                 sub=(f"{count} PAD" + ("S" if count > 1 else "")) if count else None)


def draw_pattern_row(d):
    """Bottom edge: the 8 pattern slots, right above their buttons."""
    if pat["row"] == "banks":
        draw_bank_row(d)
        return
    blink = int(time.monotonic() * 4) % 2
    for k in range(SLOTS_PER_PAGE):
        i = pat["page"] * SLOTS_PER_PAGE + k
        if i == pat["cur"]:
            colour = BLUE
        elif i == pat["queued"]:
            colour = BLUE if blink else GREY
        elif slots[i] is not None:
            colour = GREY
        else:
            colour = DIM
        sl = capture() if i == pat["cur"] else slots[i]
        sub = (f"{sl['bars']} BAR" + ("S" if sl["bars"] > 1 else "")) if sl is not None and sl["steps"] else None
        nav_cell(d, k, f"P{i + 1}", colour, active=i == pat["cur"], sub=sub)
    d.text((952, 151), "1-8" if pat["page"] == 0 else "9-16", font=F_NAV, fill=GREY, anchor="rm")


def apply_follow():
    """When following, the step pads (and so the screen) jump to the playing bar."""
    if seq["follow"] and clock["playing"] and current_step() >= 0:
        seq["page"] = current_step() // STEPS


def screen_window():
    """Which bars the screen shows: first bar and how many."""
    shown = min(seq["zoom"], seq["bars"])
    first = (seq["page"] // shown) * shown
    first = max(0, min(first, seq["bars"] - shown))
    return first, shown


ROW_H = 12


def max_rows():
    return max(1, int((145 - (strip_height() + 2) - 16) // ROW_H))


def visible_rows():
    """As many pad rows as fit, scrolled so the selected pad is always on screen."""
    n = max_rows()
    if len(kit_notes) <= n:
        return list(kit_notes)
    i = kit_notes.index(ui["selected"]) if ui["selected"] in kit_notes else 0
    start = max(0, min(i - n // 2, len(kit_notes) - n))
    return kit_notes[start:start + n]


def shade(colour, base, k):
    """Mix colour into base: k = 1 full colour, k = 0 just base."""
    return tuple(int(b + (c - b) * k) for c, b in zip(colour, base))


def roll_notes():
    """The notes shown as piano-roll rows, lowest first (scale mode: only notes in key)."""
    n_rows = max_rows()
    if kb["mode"] == "chromatic":
        base = kb_note(0, 0)
        return [base + i for i in range(n_rows)]
    sc = SCALES[setup["key_scale"]]
    base = 12 * (kb["octave"] + 2) + setup["key_root"]
    return [base + 12 * (i // len(sc)) + sc[i % len(sc)] for i in range(n_rows)]


TIME_LINES = [(16, (88, 88, 88)), (8, (64, 64, 64)), (4, (56, 56, 56)), (2, (48, 48, 48)), (1, (40, 40, 40))]


def draw_time_lines(d, x0, cw, cols, first, y0, y1):
    """Bar lines, then finer divisions (half bar, beat, 1/8, 1/16) while they're
    at least 9 px apart: zoomed out you see bars and halves, zoomed in down to 16ths."""
    for every, colour in reversed(TIME_LINES):         # faintest first, bars on top
        if every * cw < 9:
            continue
        for sidx in range(0, cols):
            g = first + sidx
            if sidx and g % every == 0:                 # brighter levels are drawn over fainter ones
                x = int(x0 + sidx * cw) - 1
                d.line([(x, y0 - 2), (x, y1)], fill=colour)


def draw_roll(d):
    """Keyboard modes: a piano roll of the selected pad (highest note on top)."""
    top = strip_height() + 2
    card(d, [222, top, 954, 145])
    sel_ = ui["selected"]
    colour = pad_screen_colour(sel_)
    notes = list(reversed(roll_notes()))
    x0 = 326
    first_bar, shown = screen_window()
    cols = shown * STEPS
    cw = (946 - x0) / cols
    first = first_bar * STEPS
    head_y, row0 = top + 3, top + 16
    bottom = row0 + len(notes) * ROW_H
    mode = "CHROMATIC" if kb["mode"] == "chromatic" else f"{NOTE_NAMES_12[setup['key_root']]} {setup['key_scale']}"
    d.text((230, head_y + 5), mode, font=F_NAV, fill=colour, anchor="lm")
    for b in range(shown):
        d.text((x0 + b * STEPS * cw + 1, head_y + 5), str(first_bar + b + 1), font=F_MONO_S,
               fill=TXT if first_bar + b == seq["page"] else DIM, anchor="lm")
    sounding = {kb_pad_note(n) for n in held}
    steps = pattern[sel_]
    lane_of = {}
    for r, n in enumerate(notes):
        y = row0 + r * ROW_H
        is_root = (n - setup["key_root"]) % 12 == 0
        lane = (32, 32, 32) if in_key(n) else (16, 16, 16)
        if is_root:
            lane = (48, 48, 48)
        lane_of[n] = lane
        d.rectangle([x0, y + 1, 946, y + ROW_H - 3], fill=lane)
        label = f"{NOTE_NAMES_12[n % 12]}{n // 12 - 2}"
        if n in sounding:                               # the key you hold: filled + white mark
            d.rectangle([224, y - 1, 320, y + ROW_H - 2], fill=colour)
            d.rectangle([224, y - 1, 225, y + ROW_H - 2], fill=SEL_OUTLINE)
            d.text((234, y + ROW_H / 2 - 1), label, font=F_LANE, anchor="lm", fill=BG)
        else:
            if is_root:
                d.rectangle([224, y - 1, 320, y + ROW_H - 2], fill=RAISED)
            d.text((234, y + ROW_H / 2 - 1), label, font=F_LANE, anchor="lm",
                   fill=TXT if is_root else (GREY if in_key(n) else DIM))
    draw_time_lines(d, x0, cw, cols, first, row0, bottom)

    def note_bar(xa, xb, y, vel, lane):
        fill = shade(colour, lane, 0.45 + 0.55 * vel / 127)
        d.rectangle([xa, y + 1, xb, y + ROW_H - 3], fill=fill)
        d.line([(xa, y + 1), (xb, y + 1)], fill=shade((255, 255, 255), fill, 0.35))   # a lit top edge
    for r, n in enumerate(notes):
        if n != padp[sel_]["root"] + tune[sel_]:          # drum steps: the pad as tuned
            continue
        y = row0 + r * ROW_H
        span = hit_span(steps)
        for sidx in range(cols):                        # drum steps play the pad as tuned = C3
            for vel, nudge in step_hits(steps[first + sidx]):
                x = x0 + (sidx + nudge + (pattern_length() if first + sidx + nudge < 0 else 0)) * cw
                note_bar(x + 1, x + max(cw * span - 2, 2), y, vel, lane_of[n])
    row_of = {n: r for r, n in enumerate(notes)}
    for pos, n, vel, length in melody.get(sel_, []):   # keyboard notes: bars from start to length
        if n not in row_of:
            continue
        start = pos - first
        if start + length < 0 or start >= cols:
            continue
        xa = x0 + max(0, start) * cw
        xb = x0 + min(cols, start + length) * cw
        note_bar(xa + 1, max(xa + 2, xb - 1), row0 + row_of[n] * ROW_H, vel, lane_of[n])
    play = current_step() - first
    if 0 <= play < cols and clock["playing"]:
        draw_playhead(d, x0 + play * cw, row0 - 1, bottom)
    if padp[sel_]["root"] + tune[sel_] not in notes and any(steps[first:first + cols]):
        d.text((944, head_y + 5), f"STEPS PLAY AT {note_name(padp[sel_]['root'] + tune[sel_])}",
               font=F_NAV, fill=GREY, anchor="rm")


F_LANE = first_font(_SC_MED, 12)
F_CHIP = first_font(_SC_BOLD, 10)


def lane_label(d, y, note, selected, silent, text):
    """Left of a lane: the pad chip (pad colour) and its name; selected = raised + white mark."""
    pc = pad_screen_colour(note)
    if selected:
        d.rectangle([224, y - 1, 946, y + ROW_H - 2], fill=RAISED)
        d.rectangle([224, y - 1, 225, y + ROW_H - 2], fill=SEL_OUTLINE)
    d.rounded_rectangle([229, y, 249, y + ROW_H - 3], 2, fill=DIM if silent else (pc if selected else shade(pc, PANEL, 0.55)))
    d.text((239, y + ROW_H / 2 - 1), pad_name(note), font=F_CHIP, fill=BG, anchor="mm")
    d.text((254, y + ROW_H / 2 - 1), fit_text(text, F_LANE, 66), font=F_LANE, anchor="lm",
           fill=DIM if silent else (TXT if selected else GREY))


def draw_playhead(d, x, y0, y1):
    d.line([(x, y0), (x, y1)], fill=TXT, width=1)
    d.polygon([(x - 3, y0 - 4), (x + 3, y0 - 4), (x, y0)], fill=TXT)


def draw_grid(d):
    top = strip_height() + 2
    card(d, [222, top, 954, 145])
    rows = visible_rows()
    if not rows:
        d.text((240, (top + 156) // 2), "No samples yet: press Browse and hit a pad to load one",
               font=F_BODY, fill=GREY, anchor="lm")
        return
    x0 = 326
    first_bar, shown = screen_window()
    cols = shown * STEPS
    cw = (946 - x0) / cols
    first = first_bar * STEPS
    wide = cw >= 12
    head_y, row0 = top + 3, top + 16
    bottom = row0 + len(rows) * ROW_H
    # header: beat numbers (1 bar) or bar numbers (zoomed out); bar lines
    if shown == 1:
        for b in range(4):
            d.text((x0 + b * 4 * cw + 1, head_y + 5), f"{first_bar + 1}.{b + 1}", font=F_MONO_S,
                   fill=TXT if b == 0 else DIM, anchor="lm")
    else:
        bx = x0 + (seq["page"] - first_bar) * STEPS * cw          # bar on the step pads
        d.rectangle([bx, head_y + 11, bx + STEPS * cw - 2, head_y + 12], fill=(200, 200, 192))
        for b in range(shown):
            bar = first_bar + b
            d.text((x0 + b * STEPS * cw + 1, head_y + 5), str(bar + 1), font=F_MONO_S,
                   fill=TXT if bar == seq["page"] else DIM, anchor="lm")
    for r, note in enumerate(rows):
        y = row0 + r * ROW_H
        selected = note == ui["selected"]
        silent = not audible(note)
        tag = "S " if note in solos else ("M " if note in mutes else "")
        lane_label(d, y, note, selected, silent, tag + os.path.splitext(pad_samples[note][0])[0])
        pc = pad_screen_colour(note)
        on_colour = SILENT if silent else (pc if selected else shade(pc, (200, 200, 210), 0.75))
        base = RAISED if selected else PANEL
        steps = pattern[note]
        span = hit_span(steps)
        for s in range(cols):
            ev = steps[first + s]
            off_fill = OFF_BEAT if s % 4 == 0 else OFF_STEP
            if wide:
                x = x0 + s * cw + (2 if s % 4 == 0 else 1)
                d.rectangle([x, y + 1, x + cw - 4, y + ROW_H - 3], fill=off_fill)
            hits = step_hits(ev)
            for vel, nudge in hits:
                fill = shade(on_colour, off_fill, 0.35 + 0.65 * vel / 127)   # brighter = harder
                at = s + nudge + (pattern_length() if first + s + nudge < 0 else 0)   # wraps: end of the loop
                x = x0 + at * cw + (2 if wide and s % 4 == 0 else (1 if wide else 0))
                w = cw - 4 if wide else max(cw - 1, 1)
                if span < 1:                            # 1/32 or triplets: thinner blocks
                    w = max(cw * span - (3 if wide else 1), 1)
                if wide:
                    held_here = any(i["pad"] == note and i["s"] == first + s for i in held_steps.values())
                    d.rectangle([x, y + 1, x + w, y + ROW_H - 3], fill=fill,
                                outline=SEL_OUTLINE if held_here else None)
                    ch = step_chance(note, first + s)
                    if ch < 100:                        # CHANCE: only that share of the block is lit
                        xc = x + 1 + (w - 2) * ch / 100
                        d.rectangle([xc, y + 2, x + w - 1, y + ROW_H - 4], fill=shade(fill, off_fill, 0.3))
                else:
                    d.rectangle([x, y + 1, x + w, y + ROW_H - 3], fill=fill)
        if not wide:
            d.line([(x0, y + ROW_H / 2 - 1), (946, y + ROW_H / 2 - 1)], fill=OFF_BEAT if selected else OFF_STEP)
        for pos, n, vel, length in melody.get(note, []):   # keyboard notes: a bar per note
            start = pos - first
            if start + length < 0 or start >= cols:
                continue
            xa = x0 + max(0, start) * cw
            xb = x0 + min(cols, start + length) * cw
            d.rectangle([xa + 1, y + 2, max(xa + 3, xb - 2), y + ROW_H - 4],
                        fill=shade(on_colour, base, 0.4 + 0.6 * vel / 127))
    if shown > 1:                                       # bar lines over the lanes
        for b in range(1, shown):
            xx = x0 + b * STEPS * cw - 1
            d.line([(xx, row0 - 2), (xx, bottom)], fill=AXIS)
    play = current_step() - first
    if 0 <= play < cols and clock["playing"]:
        draw_playhead(d, x0 + play * cw, row0 - 1, bottom)

def chop_window():
    """The part of the sample on screen: centred on the current marker,
    or on the playhead while tapping."""
    L = len(chop_audio())
    span = max(1, L // chop["zoom"])
    head = chop_playhead() if chop["tap"] else None
    if head is not None:
        centre = head
    else:                                   # show the current slice: whole if it fits,
        a, b = slices()[min(chop["cur"], len(chop["markers"]) - 1)]
        centre = (a + b) // 2 if b - a <= span * 0.9 else a + span // 4   # else its start, 1/4 in
    if chop["view"] is not None:
        centre = chop["view"] + span // 2
    start = int(clamp(centre - span // 2, 0, max(0, L - span)))
    return start, start + span


def chop_zoom(direction):
    vs, ve = chop_window()
    i = CHOP_ZOOMS.index(chop["zoom"]) + direction
    chop["zoom"] = CHOP_ZOOMS[max(0, min(len(CHOP_ZOOMS) - 1, i))]
    if chop["view"] is not None:                        # scrolled: zoom around the middle of the screen
        chop["view"] = (vs + ve) // 2 - len(chop_audio()) // chop["zoom"] // 2


def chop_scroll(direction):
    """Left/Right arrows: move the view half a screen."""
    vs, ve = chop_window()
    L, span = len(chop_audio()), ve - vs
    chop["view"] = int(clamp(vs + direction * span // 2, 0, max(0, L - span)))


def chop_playhead():
    """Where the chop preview is right now (in the sample), or None."""
    v = chop["voice"]
    if v is None or v is not ui.get("preview") or v not in voices or v.pos < 0:
        return None
    since = max(0.0, time.monotonic() - audio_state["cb_time"] - audio_state["latency"])
    return chop["play_from"] + min(len(v.audio), v.pos + since * SR)


def chop_pad_playhead():
    """The pad itself playing (by hand or from the pattern): where it is in the sample."""
    v = last_voice.get(chop["pid"])
    if v is None or v not in voices or v.pos < 0 or len(v.audio) == 0:
        return None
    since = max(0.0, time.monotonic() - audio_state["cb_time"] - audio_state["latency"])
    return min(1.0, (v.pos + since * SR) / len(v.audio)) * len(chop_audio())   # tuned pads: same spot


def draw_chop(d):
    """SETTINGS · SAMPLE: slices. Same waveform look as the sample editor: the pad's
    colour, the selected slice lifted and full colour, its marker a filled flag."""
    top = strip_height() + 2
    card(d, [222, top, 954, 145])
    chop_state()
    if chop["tap"] and (chop["voice"] is None or chop["voice"] not in voices):
        chop["tap"] = False                                   # reached the end of the sample
    audio = chop_audio()
    L = len(audio)
    pc = pad_screen_colour(chop["pid"])
    if chop["tap"]:
        d.text((232, top + 8), "TAPPING: HIT ANY PAD TO START A SLICE  ·  RECORD TO STOP", font=F_NAV, fill=WARN,
               anchor="lm")
    else:
        d.text((232, top + 8), f"{len(chop['markers'])} SLICES  ·  SLICE {chop['cur'] + 1}"
               + (f"   {detect_summary(chop['pid'])}" if detect_summary(chop["pid"]) else ""), font=F_MONO,
               fill=GREY, anchor="lm")
    d.text((944, top + 8), (f"ZOOM ×{chop['zoom']}   " if chop["zoom"] > 1 else "") + f"{L / SR:.2f}s",
           font=F_MONO, fill=GREY, anchor="rm")
    x0, width = 232, 712
    y0, y1 = top + 18, 139
    mid, half = (y0 + y1) // 2, (y1 - y0) // 2 - 2
    vs, ve = chop_window()
    span = ve - vs
    col = lambda pos: (pos - vs) / span * width              # sample position -> screen column
    sl = slices()
    starts = [a for a, _ in sl]
    if 0 <= chop["cur"] < len(sl):                           # the selected slice, lifted
        a_, b_ = max(0, col(sl[chop["cur"]][0])), min(width, col(sl[chop["cur"]][1]))
        if b_ > a_:
            d.rectangle([x0 + a_, y0, x0 + b_, y1], fill=(32, 32, 32))
    d.line([(x0, mid), (x0 + width, mid)], fill=GRID_MAJOR)
    pk = peaks(audio[vs:ve], width)
    head, start = chop_playhead(), chop["play_from"]
    if head is None:
        head, start = chop_pad_playhead(), 0
    hx = None if head is None else int(col(head))
    fx = int(col(start))
    dim_a, dim_b = shade(pc, PANEL, 0.62), shade(pc, PANEL, 0.45)
    for i in range(width):
        pos = vs + i * span / width
        k = max([j for j, a in enumerate(starts) if a <= pos], default=0)
        colour = pc if k == chop["cur"] else (dim_a if k % 2 == 0 else dim_b)
        if hx is not None and fx <= i <= hx:
            colour = TXT                                          # heard = white
        h = max(1, int(pk[i] * half))
        d.line([(x0 + i, mid - h), (x0 + i, mid + h)], fill=colour)
    for k, a in enumerate(starts):
        if not vs <= a < ve:
            continue
        x = x0 + col(a)
        sel = k == chop["cur"]
        d.line([(x, y0), (x, y1)], fill=SEL_OUTLINE if sel else GREY, width=2 if sel else 1)
        label = str(k + 1)
        w = 8 + F_CHIP.getlength(label)
        d.rectangle([x, y0, x + w, y0 + 12], fill=pc if sel else RAISED)
        d.text((x + w / 2, y0 + 6), label, font=F_CHIP, fill=BG if sel else TXT, anchor="mm")
    if hx is not None and 0 <= hx < width:                  # playhead
        d.line([(x0 + hx, y0 + 13), (x0 + hx, y1)], fill=TXT, width=1)
    if chop["zoom"] > 1:                                    # where the zoomed view sits in the sample
        d.line([(x0, 142), (x0 + width, 142)], fill=DIM, width=2)
        d.line([(x0 + vs / L * width, 142), (x0 + max(ve / L * width, vs / L * width + 3), 142)], fill=TXT, width=2)


def draw_wave(d):
    top = strip_height() + 2
    card(d, [222, top, 954, 145])
    sel = ui["selected"]
    if sel is None:
        return
    caps(d, (232, top + 5), "SAMPLE", GREY)
    caps(d, (944, top + 5), f"{len(pitched[sel]) / SR:.2f} s", GREY, anchor="ra")
    x0, width = 232, 712
    mid, half = (top + 18 + 141) // 2, (141 - top - 18) // 2
    peaks_ = wave_peaks[sel]
    v = last_voice.get(sel)                               # live or sequenced: the latest hit
    playing = v is not None and v in voices and v.pos >= 0
    head = int(v.pos / len(v.audio) * width) if playing else -1
    for i in range(width):
        p = peaks_[int(i * len(peaks_) / width)]
        h = max(1, int(p * half))
        d.line([(x0 + i, mid - h), (x0 + i, mid + h)], fill=pad_screen_colour(sel) if i < head else (200, 200, 210))
    if playing:
        d.line([(x0 + head, top + 18), (x0 + head, 141)], fill=TXT)


FAV_C = (248, 200, 64)


def folder_icon(d, x, y, colour):
    d.rectangle([x, y + 2, x + 13, y + 10], outline=colour)
    d.rectangle([x, y, x + 5, y + 2], fill=colour)


def star_icon(d, cx, cy, r, colour):
    pts = []
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(a), cy - rr * math.sin(a)))
    d.polygon(pts, fill=colour)


def draw_browser(d):
    """Browse: the list on the left (selected row off-white), the preview on the right."""
    b = browser
    item_ = b.current()
    fav_on = bool(item_) and b.is_favorite(item_[1])
    d.rounded_rectangle([3, 1, SLOT_W - 3, 16], 2, fill=NAV)
    d.text((SLOT_W / 2, 9), "BROWSE", font=F_TAB, fill=BG, anchor="mm")
    d.text((SLOT_W + 8, 9), fit_text(b.where(), F_NAV, 4 * SLOT_W), font=F_NAV, fill=GREY, anchor="lm")
    if b.items:
        d.text((6 * SLOT_W - 10, 9), f"{b.index + 1} / {len(b.items)}", font=F_MONO, fill=GREY, anchor="rm")
    for slot, label in ((6, "UNFAV" if fav_on else "FAV"), (7, "SEARCH")):
        x0 = slot * SLOT_W + 3
        d.rounded_rectangle([x0, 1, x0 + SLOT_W - 6, 16], 2, fill=RAISED)
        d.text((x0 + (SLOT_W - 6) / 2, 9), label, font=F_TAB, fill=FAV_C if slot == 6 else TXT, anchor="mm")
    # list
    card(d, [3, 20, 556, 157])
    rows, rh = 6, 22
    first = max(0, min(b.index - rows // 2, len(b.items) - rows))
    for r in range(rows):
        i = first + r
        if i >= len(b.items):
            break
        name, path_, is_folder = b.items[i]
        y = 23 + r * rh
        sel = i == b.index
        if sel:
            d.rounded_rectangle([6, y, 553, y + rh - 3], 2, fill=NAV)
        ink = BG if sel else (GREY if is_folder else (205, 205, 200))
        if path_ == "*favorites":
            star_icon(d, 18, y + 9, 6, BG if sel else FAV_C)
        elif is_folder:
            folder_icon(d, 11, y + 4, BG if sel else BLUE)
        else:
            for k, hgt in enumerate((3, 7, 5, 8, 4)):            # a tiny waveform = audio file
                d.line([(12 + k * 2, y + 9 - hgt / 2), (12 + k * 2, y + 9 + hgt / 2)], fill=ink if sel else DIM)
        fav = b.is_favorite(path_)
        d.text((32, y + 9), fit_text(name, F_BODY, 495 if fav else 510), font=F_BODY, anchor="lm", fill=ink)
        if fav:
            star_icon(d, 543, y + 9, 5, BG if sel else FAV_C)
        if sel and is_folder:
            d.text((548, y + 9), "›", font=F_SMALL, fill=BG, anchor="rm")
    # preview pane
    card(d, [562, 20, 957, 157])
    if b.message:
        d.text((572, 31), fit_text(b.message, F_BODY, 375), font=F_BODY, fill=WARN, anchor="lm")
    if b.searching:
        d.text((572, 88), "SEARCHING…", font=F_SMALL, fill=FAV_C, anchor="lm")
    elif item_ and not item_[2] and b.preview and b.preview["path"] == item_[1]:
        pv = b.preview
        mid, half = 82, 32
        d.rectangle([572, mid - half - 2, 947, mid + half + 2], fill=(16, 16, 16))
        d.line([(572, mid), (947, mid)], fill=GRID_MAJOR)
        for i, pk in enumerate(pv["peaks"][:375]):
            h = max(1, int(pk * half))
            d.line([(572 + i, mid - h), (572 + i, mid + h)], fill=CYAN)
        d.text((572, 130), fit_text(os.path.splitext(os.path.basename(pv["path"]))[0], F_SMALL, 250),
               font=F_SMALL, fill=TXT, anchor="lm")
        d.text((572, 147), f"{pv['seconds']:.3f} s", font=F_MONO, fill=GREY, anchor="lm")
        d.rounded_rectangle([830, 138, 947, 154], 2, fill=RAISED, outline=BORDER)
        d.text((888, 146), "HIT A PAD TO LOAD", font=F_NAV, fill=TXT, anchor="mm")
    elif item_ and item_[2]:
        folder_icon(d, 572, 76, BLUE)
        d.text((592, 82), "Folder   ›  to open", font=F_BODY, fill=GREY, anchor="lm")
    if not b.message and not b.searching:
        d.text((572, 31), "KNOB 1 SCROLL · KNOB 2 FAST · ‹ BACK · › OPEN", font=F_MONO_S, fill=DIM, anchor="lm")


def draw_clear_confirm(d):
    i = ui["confirm_clear"]
    sl = capture() if i == pat["cur"] else slots[i]
    if sl is None:
        ui["confirm_clear"] = None
        return
    hits = sum(len(step_hits(e)) for p in sl["steps"].values() for e in p)
    d.rounded_rectangle([562, 22, 954, 145], 6, fill=CARD, outline=RED, width=2)
    caps(d, (576, 32), f"CLEAR PATTERN P{i + 1}?", RED)
    d.text((576, 52), f"{hits} hits on {len(sl['steps'])} pads, {sl['bars']} bar" + ("s" if sl["bars"] > 1 else ""),
           font=F_BODY, fill=GREY)
    d.text((576, 84), f"Press P{i + 1} again to clear", font=F_SMALL, fill=TXT)
    caps(d, (576, 118), "ANY OTHER BUTTON CANCELS  ·  UNDO BRINGS IT BACK", GREY)


def draw_confirm(d):
    note, path = ui["pending"]
    num = pad_name(note)
    d.rounded_rectangle([562, 22, 954, 156], 6, fill=CARD, outline=RED, width=2)
    caps(d, (576, 32), f"REPLACE PAD {num}?", RED)
    caps(d, (576, 56), "NOW", GREY)
    d.text((616, 54), os.path.splitext(pad_samples[note][0])[0][:30], font=F_BODY, fill=GREY)
    caps(d, (576, 78), "NEW", GREY)
    d.text((616, 76), os.path.splitext(os.path.basename(path))[0][:30], font=F_BODY, fill=TXT)
    d.text((576, 112), f"Hit pad {num} again to replace", font=F_SMALL, fill=TXT)
    caps(d, (576, 138), "< OR MOVE ON TO CANCEL", GREY)


def db(x):
    return 20 * math.log10(max(x, 1e-6))


def draw_sampling(d):
    caps(d, (10, 2), "SAMPLING", TXT)
    d.rectangle([10, 14, SLOT_W - 6, 15], fill=RED)
    caps(d, (SLOT_W + 10, 2), "SHIFT + RECORD: LEAVE  ·  HIT AN EMPTY PAD: SAMPLE THERE", GREY)
    for slot, param in enumerate(SMP_KNOBS):
        if param is None:
            continue
        x = slot * SLOT_W + 6
        card(d, [x, 19, x + SLOT_W - 8, 52], RED, side="top")
        caps(d, (x + 8, 24), param.label, GREY)
        v, u = split_value(param.text())
        value_unit(d, (x + 8, 33), v, u)
    # left: where it goes
    card(d, [6, 58, 214, 145], RED)
    t = smp["target"]
    caps(d, (18, 64), "SAMPLE TO", GREY)
    d.text((18, 76), f"PAD {pad_name(t)}" if t is not None else "NO EMPTY PAD", font=F_NAME, fill=TXT)
    caps(d, (18, 108), f"INPUT {setup['smp_input']}  ·  {clock['bpm']} BPM", GREY)
    st = smp["state"]
    status = {"idle": "RECORD TO START", "waiting": "WAITING FOR SIGNAL…",
              "recording": f"RECORDING  {smp['count'] / SR:.1f} s", "done": "SAVING…"}[st]
    if smp["error"]:
        status = smp["error"].upper()
    caps(d, (18, 124), status, RED if st in ("waiting", "recording") or smp["error"] else TXT)
    # right: meters + what's been recorded
    card(d, [222, 58, 954, 145])
    x0, x1 = 262, 944
    peak = smp["peak"].copy()
    smp["peak"] *= 0.8                                     # meters fall back between frames
    for ch in range(2):
        y = 66 + ch * 12
        caps(d, (232, y - 1), "LR"[ch], GREY)
        d.rectangle([x0, y, x1, y + 6], fill=CARD2)
        level = (max(db(peak[ch]), -60) + 60) / 60          # -60..0 dB
        fill = RED if peak[ch] >= 0.999 else (TXT if db(peak[ch]) > -6 else (150, 150, 165))
        d.rectangle([x0, y, x0 + level * (x1 - x0), y + 6], fill=fill)
    for mark in (-48, -36, -24, -12, -6, 0):                # dB scale
        xm = x0 + (mark + 60) / 60 * (x1 - x0)
        d.line([(xm, 64), (xm, 88)], fill=DIM)
    if setup["smp_thresh"] != "OFF":
        xt = x0 + (setup["smp_thresh"] + 60) / 60 * (x1 - x0)
        d.line([(xt, 62), (xt, 90)], fill=RED, width=2)
        caps(d, (xt + 4, 91), "THRESH", RED)
    if time.monotonic() - smp["clip"] < 1.0:
        caps(d, (944, 91), "CLIP", RED, anchor="ra")
    # the take so far, as a growing waveform
    if st == "recording" and smp["blocks"]:
        total = smp_length_samples() if setup["smp_length"] else max(smp["count"], 10 * SR)
        width = int(min(1, smp["count"] / total) * (x1 - x0))
        audio = np.concatenate(smp["blocks"][-int(total / BLOCK) - 2:]) if width > 0 else None
        if audio is not None and width > 1:
            pk = peaks(audio, width)
            mid, half = 122, 18
            for i in range(width):
                h = max(1, int(pk[i] * half))
                d.line([(x0 + i, mid - h), (x0 + i, mid + h)], fill=RED)
        if setup["smp_length"]:
            caps(d, (944, 104), f"{smp['count'] / smp_length_samples() * setup['smp_length']:.1f} / "
                                f"{setup['smp_length']} BARS", GREY, anchor="ra")


def draw_master(d):
    """Master page: gain reduction (like the SSL's needle) and the output level."""
    top = strip_height() + 2
    card(d, [222, top, 954, 145], MASTER_C)
    m = master_fx
    caps(d, (232, top + 6), "MASTER BUS", GREY)
    caps(d, (944, top + 6), ("COMP " + (f"{m['ratio']}:1" if m["ratio"] else "OFF")) +
         f"  ·  LIMIT {CEILING_DB} dB", GREY, anchor="ra")
    x0, x1 = 300, 868

    def meter(y, label, frac, colour, text):
        caps(d, (232, y + 2), label, GREY)
        d.rectangle([x0, y, x1, y + 10], fill=CARD2)
        w = int((x1 - x0) * max(0.0, min(1.0, frac)))
        if w > 0:
            d.rectangle([x0, y, x0 + w, y + 10], fill=colour)
        d.text((944, y + 5), text, font=F_SMALL, fill=TXT, anchor="rm")

    gr, lim = min(0.0, meters["gr"]), min(0.0, meters["lim"])
    meter(top + 22, "COMP GR", -gr / 20, (232, 160, 60), f"{gr:.1f} dB" if gr < -0.05 else "0 dB")
    meter(top + 36, "LIMITER", -lim / 6, RED, f"{lim:.1f} dB" if lim < -0.05 else "0 dB")
    for ch, name in enumerate(("OUT L", "OUT R")):
        pk = meters["out"][ch]
        db_ = 20 * math.log10(max(pk, 1e-6))
        meter(top + 54 + ch * 14, name, (db_ + 48) / 48, (90, 200, 120) if db_ < -6 else (230, 200, 70),
              "-inf" if pk < 1e-5 else f"{db_:.1f} dB")


# ---------------------------------------------------------------------------
# 4 FX buses (Device button). Each pad goes to MASTER (direct) or BUS 1-4 (PAD page
# knob BUS). A bus has 4 free slots: pick an effect per slot, its order, on/off.
# The settings live in `buses` (saved, undoable); BusEngine turns them into
# pedalboard chains in the main loop and the audio thread only runs the chains.
# ---------------------------------------------------------------------------
NUM_BUSES, NUM_SLOTS = 4, 4
BUS_COLOURS = [(60, 190, 200), (170, 110, 240), (240, 150, 60), (110, 200, 90)]
DELAY_NOTES = [("1/16", 0.25), ("1/8T", 1 / 3), ("1/8", 0.5), ("1/8D", 0.75), ("1/4", 1.0),
               ("1/4D", 1.5), ("1/2", 2.0)]


def _P(label, key, lo, hi, default, step=1, unit="", kind="lin", choices=None):
    return {"label": label, "key": key, "lo": lo, "hi": hi, "default": default, "step": step,
            "unit": unit, "kind": kind, "choices": choices}


EFFECTS = {
    "SAT": {"name": "SATURATION", "params": [                  # ours (native_fx.Saturation)
        _P("AMOUNT", "amount", 0, 100, 30, unit=""),
        _P("TYPE", "stype", 0, 0, "NEUTRAL", kind="choice", choices=["KEEP LOW", "NEUTRAL", "KEEP HIGH"]),
        _P("MIX", "mix", 0, 100, 100, unit="%"),
        _P("OUT", "out", -24, 6, 0, unit=" dB")]},
    "CRUSH": {"name": "BITCRUSHER", "params": [                # ours (native_fx.Crusher)
        _P("BITS", "bits", 1, 16, 8),
        _P("RATE", "rate", 500, 48000, 16000, kind="log", unit=" Hz"),
        _P("JITTER", "jitter", 0, 100, 0, unit="%"),
        _P("CRUNCH", "crunch", 0, 100, 0, unit="%"),
        _P("LOW CUT", "lowcut", 5, 1000, 20, kind="log", unit=" Hz"),
        _P("MIX", "mix", 0, 100, 100, unit="%")]},
    "FILTER": {"name": "FILTER", "params": [
        _P("MODE", "mode", 0, 0, "LPF", kind="choice", choices=["LPF", "HPF", "BPF"]),
        _P("CUTOFF", "cutoff", 20, 20000, 2000, kind="log", unit=" Hz"),
        _P("RESO", "reso", 0, 95, 20, unit="%"),
        _P("DRIVE", "drive", 1, 10, 1)]},
    "EQ": {"name": "EQ", "params": [                           # ours (native_fx.EQ), Nova-style
        _P("BAND", "band", 0, 0, 0, kind="choice", choices=[0, 1, 2, 3, 4]),
        _P("FREQ", "f", 20, 20000, 1000, kind="log", unit=" Hz"),
        _P("GAIN", "g", -15, 15, 0, unit=" dB"),
        _P("Q", "q", 3, 80, 10),
        _P("OUT", "out", -12, 12, 0, unit=" dB")]},
    "COMP": {"name": "COMPRESSOR", "params": [                 # ours (native_fx.Compressor), Pro-C style
        _P("THRESH", "thresh", -48, 0, -18, unit=" dB"),
        _P("RATIO", "ratio", 0, 0, 4, kind="choice", choices=[1.5, 2, 3, 4, 6, 10, 20], unit=":1"),
        _P("ATTACK", "attack", 1, 1000, 100, kind="log", unit=" ms"),       # tenths of a ms
        _P("RELEASE", "release", 10, 2600, 150, kind="log", unit=" ms"),    # past 2.5 s = AUTO
        _P("SC HPF", "schpf", 20, 500, 20, kind="log", unit=" Hz"),          # 20 = OFF
        _P("STYLE", "style", 0, 0, "CLEAN", kind="choice", choices=["CLEAN", "CHARACTER"]),
        _P("MIX", "mix", 0, 100, 100, unit="%")]},
    "TAPE": {"name": "TAPE / VINYL", "params": [               # ours (native_fx.TapeVinyl)
        _P("MODE", "mode", 0, 0, "TAPE", kind="choice", choices=["TAPE", "VINYL"]),
        _P("DRIVE", "drive", 0, 100, 30, unit="%"),
        _P("WOW", "wow", 0, 100, 20, unit="%"),
        _P("FLUTTER", "flutter", 0, 100, 15, unit="%"),
        _P("NOISE", "noise", 0, 100, 10, unit="%"),
        _P("WEAR", "wear", 0, 100, 10, unit="%"),
        _P("FILTER", "filter", -100, 100, 0, step=2, unit="")],
        "vinyl": [                                              # MODE = VINYL: these instead
        _P("MODE", "mode", 0, 0, "TAPE", kind="choice", choices=["TAPE", "VINYL"]),
        _P("ERA", "era", 0, 0, 1970, kind="choice", choices=[1930, 1950, 1960, 1970, 1980, 2000]),
        _P("DUST", "dust", 0, 100, 30, unit="%"),
        _P("SCRATCH", "scratch", 0, 100, 10, unit="%"),
        _P("WARP", "warp", 0, 100, 15, unit="%"),
        _P("WEAR", "vwear", 0, 100, 20, unit="%"),
        _P("MIX", "vmix", 0, 100, 100, unit="%")]},
    "REVERB": {"name": "REVERB", "params": [
        _P("SIZE", "size", 0, 100, 50, unit="%"),
        _P("DAMP", "damp", 0, 100, 50, unit="%"),
        _P("WET", "wet", 0, 100, 30, unit="%"),
        _P("DRY", "dry", 0, 100, 100, unit="%"),
        _P("WIDTH", "width", 0, 100, 100, unit="%")]},
    "DELAY": {"name": "BBD DELAY", "params": [                 # ours (native_fx.BBDDelay)
        _P("TIME", "time", 0, 0, "1/8D", kind="choice", choices=[n for n, _ in DELAY_NOTES]),
        _P("FEEDBACK", "fb", 0, 100, 35, unit="%"),
        _P("MIX", "mix", 0, 100, 30, unit="%"),
        _P("MODE", "mode", 0, 0, "L/R", kind="choice", choices=["L/R", "PING PONG"]),
        _P("WOBBLE", "wobble", 0, 100, 15, unit="%"),
        _P("QUALITY", "quality", 0, 100, 70, unit="%"),
        _P("FREEZE", "freeze", 0, 0, "OFF", kind="choice", choices=["OFF", "ON"])]},
}
EFFECT_TYPES = [None, "FILTER", "EQ", "COMP", "SAT", "CRUSH", "TAPE", "REVERB", "DELAY"]   # Luis's order
NATIVE = {"SAT": "Saturation", "DELAY": "BBDDelay", "CRUSH": "Crusher", "EQ": "EQ",
          "FILTER": "LadderFilter", "COMP": "Compressor", "TAPE": "TapeVinyl"}                                   # in native_fx   # in native_fx
SHORT_NAMES = {"SAT": "SATURATION", "CRUSH": "CRUSHER", "FILTER": "FILTER", "EQ": "EQ", "COMP": "COMP",
               "REVERB": "REVERB", "DELAY": "BBD DELAY", "TAPE": "TAPE/VINYL"}
buses = [{"slots": [None] * NUM_SLOTS, "mute": False, "vol": 0, "pan": 0} for _ in range(NUM_BUSES)]


def bus_muted(note):
    b = padp[note].get("bus")
    return b is not None and buses[b].get("mute", False)
dev = {"bus": 0, "slot": 0}                    # what the Device page shows


def load_buses(data):
    for b in range(NUM_BUSES):
        sl = (data[b]["slots"] if data and b < len(data) else [None] * NUM_SLOTS) + [None] * NUM_SLOTS
        out = []
        for s_ in sl[:NUM_SLOTS]:
            # effects that were dropped: carry on with ours
            swap = {"PLATE": "REVERB", "SOFTUBE": "SAT", "BRIGADE": "DELAY"}
            if s_ and s_.get("type") in swap:
                s_ = dict(s_, type=swap[s_["type"]])
            if s_ and s_.get("type") in EFFECTS:
                base = new_slot(s_["type"])
                base["on"] = bool(s_.get("on", True))
                base["p"].update({k: v for k, v in s_.get("p", {}).items() if k in base["p"]})
                out.append(base)
            else:
                out.append(None)
        buses[b]["slots"] = out
        old = data[b] if data and b < len(data) else {}
        buses[b]["mute"] = bool(old.get("mute", False))
        buses[b]["vol"] = int(old.get("vol", 0))
        buses[b]["pan"] = int(old.get("pan", 0))


def copy_buses():
    return [{"slots": [None if s_ is None else {"type": s_["type"], "on": s_["on"], "p": dict(s_["p"])}
                       for s_ in bus["slots"]], "mute": bus.get("mute", False),
             "vol": bus.get("vol", 0), "pan": bus.get("pan", 0)} for bus in buses]


EQ_DEFAULTS = {"f0": 100, "f1": 300, "f2": 1000, "f3": 3500, "f4": 8000,
               "g0": 0, "g1": 0, "g2": 0, "g3": 0, "g4": 0,
               "q0": 7, "q1": 10, "q2": 10, "q3": 10, "q4": 7}
PER_BAND = ("f", "g", "q")               # EQ knobs that act on the selected band


def new_slot(kind):
    specs = EFFECTS[kind]["params"] + EFFECTS[kind].get("vinyl", [])
    p_ = {p["key"]: p["default"] for p in specs if p["key"] not in PER_BAND or kind != "EQ"}
    if kind == "EQ":
        p_.update(EQ_DEFAULTS)
    return {"type": kind, "on": True, "p": p_}


def build_effect(kind):
    import pedalboard as pb
    if kind in NATIVE and native_fx is None:
        toast("needs scipy: .venv/bin/python -m pip install scipy")
        return pb.Gain()
    if kind in NATIVE:
        fx = getattr(native_fx, NATIVE[kind])()
        fx.process(np.zeros((2, 512), np.float32))        # warm up here, not in the audio thread
        return fx
    if kind == "REVERB":
        return pb.Reverb()

def apply_effect(kind, fx, p, bpm):
    import pedalboard as pb
    if kind in NATIVE:                                    # our own effects
        if hasattr(fx, "set"):
            fx.set(p, bpm)
    elif kind == "REVERB":
        fx.room_size, fx.damping = p["size"] / 100, p["damp"] / 100
        fx.wet_level, fx.dry_level, fx.width = p["wet"] / 100 * 0.6, p["dry"] / 100, p["width"] / 100


def fit_block(y, n):
    """Some effects (resampling) hand back a few samples less on their first block."""
    if y.shape[1] < n:
        return np.pad(y, ((0, 0), (n - y.shape[1], 0)))
    return y[:, :n] if y.shape[1] > n else y


class BusEngine:
    def __init__(self):
        import pedalboard as pb
        from collections import deque
        self.pb = pb
        self.chains = [None] * NUM_BUSES            # what the audio thread runs
        self.pending = deque()                      # changes the audio thread applies itself
        self.fx = [[None] * NUM_SLOTS for _ in range(NUM_BUSES)]     # (type, plugin) per slot
        self.keys = [None] * NUM_BUSES
        self.pkeys = {}
        self.loading = {}                           # (bus, slot) -> {"type", "fx"} while a plugin loads

    def sync(self):
        """Main loop: rebuild a bus chain when its slots change; push knob changes."""
        bpm = clock["bpm"]
        for b, bus in enumerate(buses):
            for i, s in enumerate(bus["slots"]):
                if not s:
                    self.fx[b][i] = None
                    continue
                if self.fx[b][i] is None or self.fx[b][i][0] != s["type"]:
                    fx = self.get_effect(b, i, s["type"])
                    if fx is None:                          # a plugin still loading
                        self.fx[b][i] = None
                        continue
                    self.fx[b][i] = (s["type"], fx)
                    self.pkeys.pop((b, i), None)
                pk = (s["type"], tuple(sorted(s["p"].items())), bpm)
                if self.pkeys.get((b, i)) != pk:
                    self.pending.append(("set", s["type"], self.fx[b][i][1], dict(s["p"]), bpm))
                    self.pkeys[(b, i)] = pk
            key = tuple((s["type"], s["on"], self.fx[b][i] is not None) if s else None
                        for i, s in enumerate(bus["slots"]))
            if key != self.keys[b]:
                active = [self.fx[b][i][1] for i, s in enumerate(bus["slots"])
                          if s and s["on"] and self.fx[b][i] is not None]
                self.pending.append(("chain", b, make_stages(active) if active else None))
                self.keys[b] = key

    def get_effect(self, b, i, kind):
        return build_effect(kind)

    def is_loading(self, b, i):
        return (b, i) in self.loading

    def audio_apply(self):
        """Audio thread, before processing: knob changes and new chains land here, so an
        effect is never changed while it is running (that can crash the program)."""
        while self.pending:
            item = self.pending.popleft()
            if item[0] == "set":
                _, kind, fx, p, bpm = item
                apply_effect(kind, fx, p, bpm)
            else:
                self.chains[item[1]] = item[2]


bus_engine = None


def pad_bus(note):
    """The bus a pad plays through, or None (straight to master)."""
    return padp[note].get("bus")


# ---------------------------------------------------------------------------
# Pad FX (the FILTER / EQ / COMP tabs): every pad has a fixed channel strip,
# FILTER > EQ > COMP, each on/off, before the pad's volume / pan and its BUS.
# Same engines as the bus effects, fewer knobs: a 3-band EQ (bands 1, 3, 5 of
# ours) and the compressor without SC HPF / STYLE. Off = costs nothing.
# padp[pad]["fx"] = {"FILTER": {"on", "p"}, "EQ": {...}, "COMP": {...}} or None.
# ---------------------------------------------------------------------------
PAD_FX = ("FILTER", "EQ", "COMP")
PAD_FX_COLOURS = {"FILTER": (255, 150, 60), "EQ": (90, 200, 255), "COMP": (240, 90, 110)}
PAD_EQ_BANDS = {0: "LOW", 2: "MID", 4: "HIGH"}


def new_pad_fx():
    fx = {k: {"on": False, "p": dict(new_slot(k)["p"])} for k in PAD_FX}
    fx["EQ"]["p"].update(f0=100, f2=1000, f4=8000, q0=7, q2=10, q4=7, band=2)
    return fx


def copy_pad_fx(fx):
    return None if not fx else {k: {"on": m["on"], "p": dict(m["p"])} for k, m in fx.items()}


def load_pad_fx(data):
    """From a saved project: missing settings fall back to the defaults."""
    if not isinstance(data, dict):
        return None
    fx = new_pad_fx()
    for k in PAD_FX:
        m = data.get(k) or {}
        fx[k]["on"] = bool(m.get("on", False))
        fx[k]["p"].update({kk: v for kk, v in (m.get("p") or {}).items() if kk in fx[k]["p"]})
    return fx


def pad_fx_active(note):
    fx = padp[note].get("fx")
    return [k for k in PAD_FX if fx and fx[k]["on"]]


class PadEngine:
    """Like BusEngine, for the pad strips: the main loop builds the effects and queues
    settings; the audio thread applies them between blocks and runs the chains."""

    def __init__(self):
        from collections import deque
        self.fx = {}                     # (pad, kind) -> effect
        self.chains = {}                 # pad -> stages (audio thread reads)
        self.pending = deque()
        self.pkeys, self.keys = {}, {}

    def sync(self):
        bpm = clock["bpm"]
        for note, pp in padp.items():
            fx = pp.get("fx")
            on = tuple(k for k in PAD_FX if fx and fx[k]["on"]) if note in pad_samples else ()
            for k in on:
                if (note, k) not in self.fx:
                    self.fx[(note, k)] = build_effect(k)
                pk = tuple(sorted(fx[k]["p"].items())) + (bpm,)
                if self.pkeys.get((note, k)) != pk:
                    self.pending.append(("set", k, self.fx[(note, k)], dict(fx[k]["p"]), bpm))
                    self.pkeys[(note, k)] = pk
            if on != self.keys.get(note, ()):
                self.keys[note] = on
                self.pending.append(("chain", note, make_stages([self.fx[(note, k)] for k in on]) if on else None))

    def audio_apply(self):
        while self.pending:
            item = self.pending.popleft()
            if item[0] == "set":
                apply_effect(item[1], item[2], item[3], item[4])
            elif item[2] is None:
                self.chains.pop(item[1], None)
            else:
                self.chains[item[1]] = item[2]


pad_engine = None


def render_pad_strip(note, audio, wrap=True):
    """Offline (resample / export): a pad's audio (2-D, pre-volume) through fresh copies
    of its strip. wrap = a loop: run twice so the state at the end carries into the start."""
    on = pad_fx_active(note)
    if not on:
        return audio
    chain = []
    for k in on:
        fx = build_effect(k)
        apply_effect(k, fx, padp[note]["fx"][k]["p"], clock["bpm"])
        chain.append(fx)
    stages = make_stages(chain)
    x = np.ascontiguousarray(audio.T, dtype=np.float32)
    if wrap:
        run_stages(stages, x)
    return run_stages(stages, x).T


def pad_fx_knobs(kind):
    """The knobs of the open FILTER / EQ / COMP tab, for the selected pad."""
    key = ("padfx", kind)
    if _pad_fx_knobs.get("key") == key:
        return _pad_fx_knobs["knobs"]

    def mod():
        sp = ui["selected"]
        if sp is None or sp not in pad_samples:
            return None
        if padp[sp].get("fx") is None:
            padp[sp]["fx"] = new_pad_fx()
        return padp[sp]["fx"][kind]

    def on_param():
        def put(d):
            m = mod()
            if m:
                m["on"] = d > 0
        return Param("ON", lambda: (mod() or {}).get("on") and "ON" or "OFF", put, per=4, needs_pad=True)

    def param(spec, key=None, label=None):
        base = fx_param(spec)
        k = key or spec["key"]

        def get():
            m = mod()
            return m["p"].get(k) if m else None

        def put(d):
            m = mod()
            if not m:
                return
            v = m["p"][k]
            if spec["kind"] == "choice":
                v = step_choice(spec["choices"], v, d)
            elif spec["kind"] == "log":
                nv = int(round(v * (1.04 ** d)))
                v = int(clamp(nv if nv != v else v + d, spec["lo"], spec["hi"]))
            else:
                v = clamp(v + d * spec["step"], spec["lo"], spec["hi"])
            m["p"][k] = v
            m["on"] = True                       # touching a setting switches the module on
        return Param(label or spec["label"], get, put, per=base.per, show=base.show,
                     needs_pad=True).auto_as(lambda: f"padfx:{ui['selected']}:{kind}:{k}")

    specs = {sp["key"]: sp for sp in EFFECTS[kind]["params"]}
    if kind == "FILTER":
        knobs = [on_param()] + [param(specs[k]) for k in ("mode", "cutoff", "reso", "drive")]
    elif kind == "EQ":
        knobs = [on_param()]
        for i, name in PAD_EQ_BANDS.items():
            knobs.append(param(specs["g"], f"g{i}", name))
            knobs.append(param(specs["f"], f"f{i}", f"{name} FREQ"))
        knobs.insert(5, param(specs["q"], "q2", "MID Q"))
    else:
        knobs = [on_param()] + [param(specs[k]) for k in ("thresh", "ratio", "attack", "release", "mix")]
    knobs = (knobs + [None] * 8)[:8]
    _pad_fx_knobs.update(key=key, knobs=knobs)
    return knobs


_pad_fx_knobs = {}


def pad_fx_open():
    return ui["page"] in PAD_FX and ui["open"]


def draw_pad_fx(d):
    """FILTER / EQ / COMP tab: the selected pad's module, its graph, and the strip."""
    top = strip_height() + 2
    kind = ui["page"]
    col = PAD_FX_COLOURS[kind]
    card(d, [222, top, 954, 145], col)
    sp = ui["selected"]
    if sp is None or sp not in pad_samples:
        caps(d, (232, top + 6), "PICK A PAD: ITS FILTER > EQ > COMP", GREY)
        return
    fx = padp[sp].get("fx") or new_pad_fx()
    m = fx[kind]
    caps(d, (232, top + 6), f"PAD {pad_name(sp)}", col)
    # the strip, top right: FILTER > EQ > COMP, lit = on, framed = this tab
    x = 944
    for k in reversed(PAD_FX):
        w = 58
        on, here = fx[k]["on"], k == kind
        d.rounded_rectangle([x - w, top + 3, x, top + 17], 3, fill=PAD_FX_COLOURS[k] if on else None,
                            outline=PAD_FX_COLOURS[k] if (here or on) else (70, 70, 78), width=1)
        caps(d, (x - w / 2, top + 6), k, (15, 15, 18) if on else (TXT if here else GREY), anchor="ma")
        x -= w + 6
    if not m["on"]:
        caps(d, (300, top + 6), "OFF  ·  TURN ANY KNOB TO SWITCH IT ON", DIM)
    if kind == "COMP":
        draw_comp_graph(d, m["p"], top, col if m["on"] else GREY,
                        comp=pad_engine.fx.get((sp, "COMP")) if pad_engine and m["on"] else None, header=False)
    else:
        draw_eq_graph(d, m["p"], top, col if m["on"] else GREY, kind=kind,
                      fxobj=pad_engine.fx.get((sp, kind)) if pad_engine and m["on"] else None,
                      bands=PAD_EQ_BANDS, header=False)


bus_meters = [0.0] * NUM_BUSES


def bus_lr(b):
    """Left/right gain of a bus fader: VOLUME in dB (-60 = off) and PAN."""
    bus = buses[b]
    vol = bus.get("vol", 0)
    g = 0.0 if vol <= -60 else 10 ** (vol / 20)
    pn = bus.get("pan", 0) / 50
    return np.array([min(1.0, 1 - pn), min(1.0, 1 + pn)], dtype=np.float32) * g


def make_stages(fxs):
    """A bus chain: runs of plugins become one pedalboard call, our own effects run as-is."""
    import pedalboard as pb
    stages, run = [], []
    for fx in fxs:
        if hasattr(fx, "process") and not isinstance(fx, pb.Plugin):
            if run:
                stages.append(pb.Pedalboard(run))
                run = []
            stages.append(fx)
        else:
            run.append(fx)
    if run:
        stages.append(pb.Pedalboard(run))
    return stages


def run_stages(stages, x):
    """x: (2, n). Plugins may hand back a few samples less at first: fit_block evens it."""
    n = x.shape[1]
    for st in stages:
        if hasattr(st, "process") and not hasattr(st, "reset"):
            x = st.process(x)
        else:
            x = np.ascontiguousarray(fit_block(st(x, SR, reset=False), n), dtype=np.float32)
    return x


def render_buses(per_bus, length, wrap=True):
    """Offline (resample / export): each bus's audio through fresh copies of its effects.
    Runs the loop twice so reverb and delay tails wrap around like when it plays."""
    import pedalboard as pb
    out = np.zeros((length, 2), np.float32)
    for b, audio in per_bus.items():
        slots = [s for s in buses[b]["slots"] if s and s["on"]]
        if not slots:
            out += audio * bus_lr(b)
            continue
        chain = []
        for s in slots:
            fx = build_effect(s["type"])
            apply_effect(s["type"], fx, s["p"], clock["bpm"])
            chain.append(fx)
        stages = make_stages(chain)
        x = np.ascontiguousarray(audio.T, dtype=np.float32)
        if wrap:                                    # a loop: first pass fills the tails
            run_stages(stages, x)
        y = run_stages(stages, x)
        out += y.T * bus_lr(b)
    return out


# --- Device page -------------------------------------------------------------
def dev_slot():
    return buses[dev["bus"]]["slots"][dev["slot"]]


def slot_type_param():
    def get():
        s = dev_slot()
        # short name on one line (a space would split it into value + unit)
        return SHORT_NAMES.get(s["type"], s["type"]).replace(" ", "\u00a0") if s else "—"

    def put(d):
        s = dev_slot()
        # A set-up effect only changes while you hold its slot button (or keep turning
        # right after picking it), so a stray turn can't throw its settings away.
        picking = time.monotonic() - dev.get("pick_t", 0) < 1.5
        if s and not picking and LOWER_BUTTONS[dev["slot"]] not in held_buttons:
            toast("hold the slot button to change the effect")
            return
        cur = s["type"] if s else None
        kind = step_choice(EFFECT_TYPES, cur, d)
        dev["pick_t"] = time.monotonic()
        if kind != cur:
            buses[dev["bus"]]["slots"][dev["slot"]] = new_slot(kind) if kind else None
    return Param("EFFECT", get, put, per=4)


def fx_param(spec):
    # The effect can change under a knob mid-draw (buttons run on another thread),
    # so a missing setting just shows "-" instead of stopping the program.
    def key(s):
        if s and s["type"] == "EQ" and spec["key"] in PER_BAND:
            return f"{spec['key']}{s['p'].get('band', 0)}"           # the selected band's setting
        return spec["key"]

    def get():
        s = dev_slot()
        return s["p"].get(key(s)) if s else None

    def put(d):
        s = dev_slot()
        if not s or key(s) not in s["p"]:
            return
        k = key(s)
        v = s["p"][k]
        if spec["kind"] == "choice":
            v = step_choice(spec["choices"], v, d)
        elif spec["kind"] == "log":
            nv = int(round(v * (1.04 ** d)))
            v = int(clamp(nv if nv != v else v + d, spec["lo"], spec["hi"]))   # always moves
        else:
            v = clamp(v + d * spec["step"], spec["lo"], spec["hi"])
        s["p"][k] = v

    def show(v):
        if v is None:
            return "-"
        if spec["key"] == "amount":
            return f"{v / 10:.1f}"
        if spec["key"] == "band":
            return (native_fx.EQ_BANDS[v] if native_fx else str(v + 1)).replace(" ", "\u00a0")
        if spec["key"] == "q":
            return f"{v / 10:.1f}"
        if spec["key"] == "release" and v > 2500:
            return "AUTO"
        if spec["key"] == "release" and v >= 1000:
            return f"{v / 1000:.1f} s"
        if spec["key"] == "filter":
            return "OFF" if v == 0 else (f"LP {-v}" if v < 0 else f"HP {v}")
        if spec["key"] == "schpf":
            return "OFF" if v <= 20 else f"{v} Hz"
        if spec["key"] == "attack":
            return f"{v / 10:g} ms"
        if spec["key"] == "rate" and v >= 1000:
            return f"{v / 1000:g} kHz"
        if spec["kind"] == "log" and v >= 1000:
            return f"{v / 1000:.1f} kHz"
        return f"{v:+d}{spec['unit']}" if spec["unit"] == " dB" and spec["lo"] < 0 else f"{v}{spec['unit']}"
    per = 4 if spec["kind"] == "choice" else 1

    def target():
        s = dev_slot()
        return f"bus:{dev['bus']}:{dev['slot']}:{s['type']}:{key(s)}" if s and spec["key"] != "band" else None
    return Param(spec["label"], get, put, per=per, show=show).auto_as(target)


_dev_knobs = {}
_dev_type_knob = None


def device_knobs():
    """Knob 1 = the slot's effect, knobs 2-6 = its settings."""
    global _dev_type_knob
    if _dev_type_knob is None:
        _dev_type_knob = slot_type_param()
    s = dev_slot()
    vinyl = bool(s and s["type"] == "TAPE" and s["p"].get("mode") == "VINYL")
    key = (dev["bus"], dev["slot"], s["type"] if s else None, vinyl)
    if key not in _dev_knobs:
        specs = (EFFECTS[s["type"]]["vinyl"] if vinyl else EFFECTS[s["type"]]["params"]) if s else []
        knobs = [_dev_type_knob] + [fx_param(p) for p in specs]
        knobs += [None] * (8 - len(knobs))
        _dev_knobs.clear()
        _dev_knobs[key] = knobs
    return _dev_knobs[key]


def device_lower_button(k):
    """Lower buttons on the Device page: 1-4 = slots, 5-8 = buses."""
    if k >= NUM_SLOTS:
        b = k - NUM_SLOTS
        if MUTE_BUTTON in held_buttons:                     # hold Mute + bus button
            checkpoint("bus mute")
            buses[b]["mute"] = not buses[b].get("mute", False)
            toast(f"BUS {b + 1} {'muted' if buses[b]['mute'] else 'on'}")
            mark_dirty()
            return
        dev["bus"], dev["slot"] = b, 0
        return
    if DELETE_BUTTON in held_buttons:
        if buses[dev["bus"]]["slots"][k]:
            checkpoint("clear slot")
            buses[dev["bus"]]["slots"][k] = None
            toast(f"slot {k + 1} cleared")
            mark_dirty()
        dev["slot"] = k
        return
    if SHIFT_BUTTON in held_buttons:
        s = buses[dev["bus"]]["slots"][k]
        if s:
            checkpoint("effect on/off")
            s["on"] = not s["on"]
            toast(f"{EFFECTS[s['type']]['name']} {'on' if s['on'] else 'off'}")
            mark_dirty()
    dev["slot"] = k


def device_move_slot(d):
    """Hold a slot button + Left/Right: move that effect earlier / later in the chain."""
    sl = buses[dev["bus"]]["slots"]
    j = dev["slot"] + d
    if 0 <= j < NUM_SLOTS:
        checkpoint("move effect")
        sl[dev["slot"]], sl[j] = sl[j], sl[dev["slot"]]
        held_buttons.discard(LOWER_BUTTONS[dev["slot"]])
        held_buttons.add(LOWER_BUTTONS[j])
        dev["slot"] = j
        mark_dirty()


def draw_device(d):
    """Device page: the bus's 4 slots as a chain, and which pads feed it."""
    top = strip_height() + 2
    b = dev["bus"]
    col = BUS_COLOURS[b]
    card(d, [222, top, 954, 145], col)
    pads_in = [pad_name(n) for n in kit_notes if padp[n].get("bus") == b]
    d.text((232, top + 8), f"BUS {b + 1}", font=F_NAV, fill=col, anchor="lm")
    if buses[b].get("mute"):
        d.rounded_rectangle([GX0 + 6, top + 2, GX0 + 46, top + 14], 2, fill=WARN)
        d.text((GX0 + 26, top + 8), "MUTED", font=F_MONO_S, fill=BG, anchor="mm")
    feed = ("PADS " + " ".join(pads_in[:14]) + (" …" if len(pads_in) > 14 else "")) if pads_in \
        else "NO PADS YET: PAD PAGE > BUS"
    d.text((944, top + 8), feed, font=F_MONO_S, fill=GREY, anchor="rm")
    sel_slot = buses[b]["slots"][dev["slot"]]
    if sel_slot and sel_slot["type"] == "COMP" and native_fx is not None:
        draw_comp_graph(d, sel_slot["p"], top, col)
        return
    if sel_slot and sel_slot["type"] in ("EQ", "FILTER") and native_fx is not None:
        draw_eq_graph(d, sel_slot["p"], top, col)
        return
    if sel_slot and sel_slot["type"] in FX_SCREENS and native_fx is not None:
        FX_SCREENS[sel_slot["type"]](d, sel_slot["p"], top, col)
        return
    w = (944 - 232 - 3 * 14) / NUM_SLOTS
    for i, s in enumerate(buses[b]["slots"]):
        x0 = 232 + i * (w + 14)
        box = [x0, top + 18, x0 + w, 138]
        selected = i == dev["slot"]
        on = bool(s and s["on"])
        d.rounded_rectangle(box, 2, fill=RAISED if on else GRAPH_BG,
                            outline=SEL_OUTLINE if selected else BORDER, width=2 if selected else 1)
        d.text((x0 + 8, top + 28), f"SLOT {i + 1}", font=F_NAV, fill=GREY, anchor="lm")
        if s:
            d.text((x0 + w - 8, top + 28), "ON" if s["on"] else "OFF", font=F_NAV, fill=col if s["on"] else DIM,
                   anchor="rm")
            d.text((x0 + 8, top + 50), EFFECTS[s["type"]]["name"], font=F_SMALL, fill=TXT if on else DIM, anchor="lm")
        else:
            d.text((x0 + 8, top + 50), "empty", font=F_BODY, fill=DIM, anchor="lm")
        if i < NUM_SLOTS - 1:                                   # signal flow arrow
            ax = x0 + w + 3
            d.polygon([(ax, top + 74), (ax + 8, top + 80), (ax, top + 86)], fill=AXIS)


_eq_cache = {}
EQ_FREQS = np.geomspace(20, 20000, 240)


def draw_eq_graph(d, p, top, col, kind=None, fxobj=None, bands=None, header=True):
    """The EQ's (or filter's) response, 20 Hz - 20 kHz, ±15 dB (filter ±24), on a quiet
    grid with the live analyzer behind. EQ: numbered band nodes, the selected one big
    and ringed. kind / fxobj / bands: a pad's strip (LOW, MID, HIGH only)."""
    x0, x1, y0, y1 = 236, 940, top + 17, 140
    ym = (y0 + y1) / 2
    pad_mode = kind is not None
    kind = kind or buses[dev["bus"]]["slots"][dev["slot"]]["type"]
    rng = 15 if kind == "EQ" else 24                          # filter: steeper, wider scale
    fx_ = lambda f: x0 + (x1 - x0) * math.log(f / 20) / math.log(1000)
    gy = lambda g: ym - (y1 - y0) / 2 * max(-1.0, min(1.0, g / rng))
    graph_box(d, x0, y0, x1, y1)
    freq_grid(d, x0, x1, y0, y1, fx_)
    for g in ((-12, -6, 6, 12) if rng == 15 else (-18, -12, -6, 6, 12, 18)):
        d.line([(x0, gy(g)), (x1, gy(g))], fill=GRID)
    for g in ((12, -12) if rng == 15 else (12, -12)):
        d.text((x0 + 3, gy(g)), f"{g:+d}", font=F_MONO_S, fill=DIM, anchor="lm")
    d.line([(x0, ym), (x1, ym)], fill=AXIS)
    key = (kind,) + tuple(sorted(p.items()))
    if _eq_cache.get("key") != key:
        fn = native_fx.eq_curve if kind == "EQ" else native_fx.filter_curve
        _eq_cache["key"], _eq_cache["curve"] = key, fn(p, EQ_FREQS)
    curve = _eq_cache["curve"]
    # live analyzer behind the curve (what comes out of the effect), -72..0 dBFS over the height
    eq = fxobj
    if not pad_mode and bus_engine is not None:
        item = bus_engine.fx[dev["bus"]][dev["slot"]]
        eq = item[1] if item and item[0] == kind else None
    if eq is not None and hasattr(eq, "tap"):
        spec = native_fx.spectrum(eq, EQ_FREQS, _eq_cache.get("spec"))
        _eq_cache["spec"] = spec
        sy = lambda v: y1 - (y1 - y0) * max(0.0, min(1.0, (v + 72) / 72))
        poly = [(x0, y1)] + [(fx_(f), sy(v)) for f, v in zip(EQ_FREQS, spec)] + [(x1, y1)]
        d.polygon(poly, fill=(40, 40, 40))
        d.line(poly[1:-1], fill=(72, 72, 72), width=1)
    pts = [(fx_(f), gy(g)) for f, g in zip(EQ_FREQS, curve)]
    fill_c = shade(col, GRAPH_BG, 0.22)
    if kind == "FILTER":
        d.polygon([(x0, y1)] + pts + [(x1, y1)], fill=fill_c)           # what passes
    else:
        d.polygon([(x0, ym)] + pts + [(x1, ym)], fill=fill_c)           # boost / cut against 0 dB
    d.line(pts, fill=col, width=2)
    if kind == "FILTER":                                  # one handle at the cutoff
        cx = fx_(max(20, min(20000, p["cutoff"])))
        cy = gy(float(np.interp(p["cutoff"], EQ_FREQS, curve)))
        d.line([(cx, y0), (cx, y1)], fill=shade(col, GRAPH_BG, 0.45))
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=col, outline=SEL_OUTLINE, width=2)
        if header:
            fx_header(d, top, col, "FILTER", f"{p['mode']} · {p['cutoff'] / 1000:.2f} kHz · RESO {p['reso']}%"
                      if p["cutoff"] >= 1000 else f"{p['mode']} · {p['cutoff']} Hz · RESO {p['reso']}%")
        else:
            d.text((x0 + 4, y1 - 7), f"FILTER · {p['mode']}", font=F_NAV, fill=col, anchor="lm")
        return
    if bands:                                             # a pad's 3-band EQ: L, M, H nodes
        for i, name in bands.items():
            cx, cy = fx_(p[f"f{i}"]), gy(p[f"g{i}"])
            d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=col, outline=SEL_OUTLINE)
            d.text((cx, cy), name[0], font=F_CHIP, fill=BG, anchor="mm")
        d.text((x0 + 4, y1 - 7), "EQ · LOW SHELF · MID BELL · HIGH SHELF", font=F_NAV, fill=col, anchor="lm")
        return
    sel_band = p.get("band", 0)
    for i in range(5):
        if i == sel_band:
            continue
        cx, cy = fx_(p[f"f{i}"]), gy(p[f"g{i}"])
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=GRAPH_BG, outline=GREY)
        d.text((cx, cy), str(i + 1), font=F_CHIP, fill=GREY, anchor="mm")
    f, g = p[f"f{sel_band}"], p[f"g{sel_band}"]
    cx, cy = fx_(f), gy(g)
    d.line([(cx, y0), (cx, y1)], fill=shade(col, GRAPH_BG, 0.45))
    d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=col, outline=SEL_OUTLINE, width=2)
    d.text((cx, cy), str(sel_band + 1), font=F_CHIP, fill=BG, anchor="mm")
    fs = f"{f / 1000:.2f} kHz" if f >= 1000 else f"{f} Hz"
    fx_header(d, top, col, f"EQ · {native_fx.EQ_BANDS[sel_band]}", f"{fs} · {g:+d} dB · Q {p[f'q{sel_band}'] / 10:.1f}"
              + (f" · OUT {p['out']:+d} dB" if p.get("out") else ""))


def draw_master_fx(d):
    """Master page: the last few seconds scrolling by (input grey, output white, the
    bus compressor's gain reduction as a red curtain, the limiter in amber) with the
    threshold; big GR / LIMIT readouts and the output meters with the ceiling."""
    top = strip_height() + 2
    col = MASTER_C
    card(d, [222, top, 954, 145], col)
    m = master_fx
    d.text((232, top + 8), "MASTER", font=F_NAV, fill=col, anchor="lm")
    d.text((232 + F_NAV.getlength("MASTER") + 10, top + 8),
           "BUS COMP " + (f"{m['ratio']}:1 · THR {m['thresh']} dB" if m["ratio"] else "OFF") +
           f" · LIMIT {CEILING_DB} dB" + (f" · DRIVE +{m['drive']} dB" if m["drive"] else ""), font=F_MONO_S,
           fill=GREY, anchor="lm")
    x0, x1, y0, y1 = 236, 790, top + 17, 140
    lo = -48.0
    ly = lambda db: y1 - (y1 - y0) * max(0.0, min(1.0, (db - lo) / -lo))
    gy = lambda g: y0 + (y1 - y0) * max(0.0, min(1.0, -g / 24))
    graph_box(d, x0, y0, x1, y1)
    for g in (-36, -24, -12):
        d.line([(x0, ly(g)), (x1, ly(g))], fill=GRID)
        d.text((x0 + 3, ly(g)), str(g), font=F_MONO_S, fill=DIM, anchor="lm")
    h = np.roll(master_hist, -master_hist_w[0], axis=1)
    n = h.shape[1]
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    d.polygon([(x0, y1)] + [(x, ly(v)) for x, v in zip(xs, h[0])] + [(x1, y1)], fill=(40, 40, 40))   # input
    if m["ratio"]:
        grl = [(x, gy(v)) for x, v in zip(xs, h[2])]
        d.polygon([(x0, y0)] + grl + [(x1, y0)], fill=(72, 16, 28))
        d.line(grl, fill=WARN, width=2)
    d.line([(x, gy(v)) for x, v in zip(xs, h[3])], fill=AMBER, width=1)          # limiter
    d.line([(x, ly(v)) for x, v in zip(xs, h[1])], fill=TXT, width=1)            # output
    if m["ratio"]:
        ty = ly(m["thresh"])
        for x in range(x0, x1, 10):
            d.line([(x, ty), (x + 5, ty)], fill=col)
    # readouts
    for i, (lab, val, c_) in enumerate((("GR", f"{meters['gr']:.1f}" if m["ratio"] else "—", WARN),
                                        ("LIMIT", f"{meters['lim']:.1f}", AMBER))):
        bx0, by0 = 798, y0 + i * 33
        d.rounded_rectangle([bx0, by0, 890, by0 + 30], 2, fill=GRAPH_BG, outline=BORDER)
        d.text((bx0 + 6, by0 + 9), lab, font=F_NAV, fill=GREY, anchor="lm")
        d.text((866, by0 + 24), val, font=F_VALUE_S, fill=c_ if val not in ("0.0", "—") else GREY, anchor="rs")
        d.text((884, by0 + 24), "dB", font=F_UNIT, fill=DIM, anchor="rs")
    # output meters with the ceiling
    my = level_meter(d, 902, y0, y1 - 10, meters["out"], lo=-48.0, hi=0.0, w=8)
    d.line([(899, my(CEILING_DB)), (922, my(CEILING_DB))], fill=TXT)
    d.text((906, y1 - 3), "L", font=F_MONO_S, fill=DIM, anchor="mm")
    d.text((916, y1 - 3), "R", font=F_MONO_S, fill=DIM, anchor="mm")
    d.text((944, y0 + 6), "OUT", font=F_MONO_S, fill=GREY, anchor="rm")


def synth_panel(d, x0, x1, top, title, col):
    d.rounded_rectangle([x0, top + 20, x1, 141], 2, fill=(16, 16, 16))
    d.text((x0 + 7, top + 28), title, font=F_NAV, fill=col, anchor="lm")


def draw_inst(d):
    """Synth page: two coordinated sections - oscillators (left) and filter (right) - in
    the synthesis amber, drawn from the real settings."""
    top = strip_height() + 2
    col = AMBER
    card(d, [222, top, 954, 145], col)
    sp = ui["selected"]
    if sp is None or not is_inst(sp):
        d.text((232, top + 10), "ADD DEVICE: PICK A SYNTH PAD, OR PRESS IT TO ADD ONE", font=F_NAV, fill=GREY,
               anchor="lm")
        return
    ip = padp[sp]["ip"]
    kind = padp[sp]["inst"]
    d.text((232, top + 9), f"{pad_samples[sp][0]}", font=F_NAV, fill=col, anchor="lm")
    d.text((232 + F_NAV.getlength(pad_samples[sp][0]) + 10, top + 9), f"{INST_LABELS[kind]}  ·  PAD {pad_name(sp)}",
           font=F_MONO_S, fill=GREY, anchor="lm")
    if kind == "KEYS":
        return draw_keys(d, sp, ip, top, col)
    # oscillators: two cycles of the mix
    synth_panel(d, 230, 530, top, "OSCILLATORS", col)
    x0, x1, y0, y1 = 240, 520, top + 36, 128
    t = np.linspace(0, 2, 300, endpoint=False)
    ph = t % 1.0
    shape = {"SAW": 2 * ph - 1, "TRI": 4 * np.abs(ph - 0.5) - 1,
             "SQUARE": np.where(ph < 0.5, 1.0, -1.0), "PULSE": np.where(ph < 0.25, 1.0, -1.0)}[ip["wave"]]
    sub = np.where((t / 2) % 1.0 < 0.5, 1.0, -1.0) * ip["sub"] / 100 * 0.6
    w = shape * (0.5 + (0.5 if ip["detune"] else 0)) + sub
    w = w / max(1e-6, float(np.abs(w).max()))
    mid, amp = (y0 + y1) / 2, (y1 - y0) / 2 - 3
    d.line([(x0, mid), (x1, mid)], fill=GRID_MAJOR)
    d.line([(x0 + (x1 - x0) * i / 299, mid - w[i] * amp) for i in range(300)], fill=col, width=2)
    d.text((x1, top + 28), f"{ip['wave']} · DETUNE {ip['detune']}ct · SUB {ip['sub']}%", font=F_MONO_S, fill=GREY,
           anchor="rm")
    # filter: closed (cutoff) and open (cutoff + contour)
    synth_panel(d, 538, 946, top, "FILTER · 24 dB", col)
    fx0, fx1 = 548, 936
    freqs = EQ_FREQS
    xf = lambda f: fx0 + (fx1 - fx0) * math.log(f / 20) / math.log(1000)
    gy = lambda g: y0 + (y1 - y0) * (1 - max(0.0, min(1.0, (g + 36) / 48)))
    for f in (100, 1000, 10000):
        d.line([(xf(f), y0), (xf(f), y1)], fill=GRID)
        d.text((xf(f) + 3, y1 - 5), {100: "100", 1000: "1K", 10000: "10K"}[f], font=F_MONO_S, fill=DIM, anchor="lm")
    d.line([(fx0, gy(0)), (fx1, gy(0))], fill=GRID_MAJOR)
    opened = min(18000, ip["cutoff"] * 2 ** (ip["contour"] / 100 * 5))
    for c_, (cut, colr) in enumerate(((opened, (112, 112, 112)), (ip["cutoff"], col))):
        curve = native_fx.filter_curve({"cutoff": cut, "reso": ip["emphasis"], "mode": "LPF"}, freqs)
        d.line([(xf(f), gy(g)) for f, g in zip(freqs, curve)], fill=colr, width=2 if c_ else 1)
    cx = xf(max(20, min(20000, ip["cutoff"])))
    d.ellipse([cx - 4, gy(0) - 4, cx + 4, gy(0) + 4], fill=col)
    d.text((fx1, top + 28), "GREY = OPEN (CONTOUR)", font=F_MONO_S, fill=GREY, anchor="rm")


def draw_keys(d, sp, ip, top, col):
    """KEYS page: the six programs (the current one lit), the volume shape of a note
    (ATTACK up, held, RELEASE down) and TONE as a low-pass curve."""
    synth_panel(d, 230, 524, top, "PROGRAM", col)
    x0, y0 = 238, top + 35
    for i, prog in enumerate(native_fx.KEYS_PROGRAMS):
        bx, by = x0 + (i % 3) * 94, y0 + (i // 3) * 19
        on = prog == ip.get("program")
        d.rounded_rectangle([bx, by, bx + 88, by + 16], 2, fill=col if on else RAISED)
        d.text((bx + 7, by + 8), prog, font=F_NAV, fill=BG if on else GREY, anchor="lm")
        if prog in native_fx.SAMPLED:
            d.text((bx + 82, by + 8), "S", font=F_MONO_S, fill=BG if on else DIM, anchor="rm")
    playing = sum(1 for v in voices if isinstance(v, SynthVoice) and v.note == sp and not v.synth.done)
    chord = ip.get("chord", "OFF")
    if chord != "OFF":
        chord = f"{chord} IN {NOTE_NAMES_12[setup['key_root']]} {setup['key_scale']}" if chord != "5TH" else "5TH"
    d.text((x0, 135), f"OCT {ip.get('octave', 0):+d} · CHORD {chord} · CHORUS {ip.get('chorus', 'OFF')} · "
           f"{playing}/{KEYS_POLY}", font=F_MONO_S, fill=GREY, anchor="lm")
    # envelope
    synth_panel(d, 532, 758, top, "ENVELOPE", col)
    ex0, ex1, ey0, ey1 = 542, 748, top + 38, 128
    att = 0.002 + (ip.get("attack", 0) / 100) ** 2 * 2.0
    rel = 0.03 + (ip.get("release", 30) / 100) ** 2 * 4.0
    total = att + 1.0 + rel * 3
    sx = lambda t: ex0 + (ex1 - ex0) * t / total
    d.line([(ex0, ey1), (ex1, ey1)], fill=GRID_MAJOR)
    pts = [(sx(0), ey1), (sx(att), ey0), (sx(att + 1.0), ey0)]
    pts += [(sx(att + 1.0 + rel * 3 * k / 20), ey1 - (ey1 - ey0) * math.exp(-3 * k / 20)) for k in range(1, 21)]
    d.polygon(pts + [(pts[-1][0], ey1)], fill=shade(col, (16, 16, 16), 0.22))
    d.line(pts, fill=col, width=2)
    d.text((ex1, top + 28), f"A {att * 1000:.0f}ms · R {rel:.2f}s", font=F_MONO_S, fill=GREY, anchor="rm")
    # tone
    synth_panel(d, 766, 946, top, "TONE", col)
    tx0, tx1 = 776, 936
    cut = 250 * (19000 / 250) ** min(1.0, ip.get("tone", 85) / 100)
    fr = np.geomspace(20, 20000, 60)
    g = -10 * np.log10(1 + (fr / cut) ** 4)
    gy = lambda v: ey0 + (ey1 - ey0) * min(1.0, -v / 30)
    d.line([(tx0, ey0), (tx1, ey0)], fill=GRID_MAJOR)
    tp = [(tx0 + (tx1 - tx0) * i / 59, gy(v)) for i, v in enumerate(g)]
    d.polygon(tp + [(tx1, ey1), (tx0, ey1)], fill=shade(col, (16, 16, 16), 0.22))
    d.line(tp, fill=col, width=2)
    d.text((tx1, top + 28), f"{ip.get('tone', 85)}%", font=F_MONO_S, fill=GREY, anchor="rm")


GX0, GX1, GY1 = 236, 940, 140                      # graph area (top comes from the page)
_scr = {}
SONG_C = (120, 220, 140)


def draw_comp_graph(d, p, top, col, comp=None, header=True):
    """Pro-C style: the last ~3 s scrolling by - input (grey), output (white), gain
    reduction hanging from the top (red curtain), the threshold dashed; GR readout."""
    x0, x1, y0, y1 = 236, 884, top + 17, 140
    lo = -48.0
    ly = lambda db: y1 - (y1 - y0) * max(0.0, min(1.0, (db - lo) / -lo))
    graph_box(d, x0, y0, x1, y1)
    for g in (-36, -24, -12):
        d.line([(x0, ly(g)), (x1, ly(g))], fill=GRID)
        d.text((x1 + 4, ly(g)), str(g), font=F_MONO_S, fill=DIM, anchor="lm")
    if header and bus_engine is not None:
        item = bus_engine.fx[dev["bus"]][dev["slot"]]
        comp = item[1] if item and item[0] == "COMP" else None
    gr_now = 0.0
    if comp is not None and hasattr(comp, "history"):
        h = comp.history()
        n = h.shape[1]
        xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
        gy = lambda g: y0 + (y1 - y0) * max(0.0, min(1.0, -g / 24))
        inp = [(x, ly(v)) for x, v in zip(xs, h[0])]
        d.polygon([(x0, y1)] + inp + [(x1, y1)], fill=(40, 40, 40))                  # input, filled
        grl = [(x, gy(v)) for x, v in zip(xs, h[2])]
        d.polygon([(x0, y0)] + grl + [(x1, y0)], fill=(72, 16, 28))                  # a curtain from the top
        d.line(grl, fill=WARN, width=2)
        d.line([(x, ly(v)) for x, v in zip(xs, h[1])], fill=TXT, width=1)           # output
        gr_now = float(h[2, -6:].min())
    ty = ly(p["thresh"])
    for x in range(int(x0), int(x1), 10):                     # dashed threshold
        d.line([(x, ty), (x + 5, ty)], fill=col)
    d.text((x0 + 4, ty - 6), f"THR {p['thresh']} dB", font=F_MONO_S, fill=col, anchor="lm")
    d.text((952, y0 + 10), "GR", font=F_NAV, fill=GREY, anchor="rm")
    d.text((952, y0 + 30), f"{gr_now:.1f}", font=F_SMALL, fill=WARN if gr_now < -0.05 else GREY, anchor="rm")
    d.text((952, y0 + 46), "dB", font=F_MONO_S, fill=DIM, anchor="rm")
    rel = "AUTO" if p["release"] > 2500 else (f"{p['release'] / 1000:.1f} s" if p["release"] >= 1000 else f"{p['release']} ms")
    sub = f"{p['ratio']}:1 · ATT {p['attack'] / 10:g} ms · REL {rel}" + \
        (f" · SC HPF {p['schpf']} Hz" if p["schpf"] > 20 else "") + (" · CHARACTER" if p["style"] == "CHARACTER" else "")
    if header:
        fx_header(d, top, col, "COMPRESSOR", sub)
    else:
        d.text((x0 + 4, y1 - 7), f"COMPRESSOR · {p['ratio']}:1", font=F_NAV, fill=col, anchor="lm")


def fx_header(d, top, col, text, sub=""):
    """An effect screen's title (after BUS n): name in the bus colour, details in mono."""
    d.text((GX0 + 52, top + 8), text, font=F_NAV, fill=col, anchor="lm")
    if sub:
        d.text((GX0 + 60 + F_NAV.getlength(text), top + 8), sub, font=F_MONO_S, fill=GREY, anchor="lm")


GRAPH_BG = (16, 16, 16)


def graph_box(d, x0, y0, x1, y1):
    d.rectangle([x0, y0, x1, y1], fill=GRAPH_BG)


def freq_grid(d, x0, x1, y0, y1, fx_):
    for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
        d.line([(fx_(f), y0), (fx_(f), y1)], fill=GRID_MAJOR if f in (100, 1000, 10000) else GRID)
    for f, lab in ((100, "100"), (1000, "1K"), (10000, "10K")):
        d.text((fx_(f) + 3, y1 - 6), lab, font=F_MONO_S, fill=DIM, anchor="lm")


def hbar(d, x, y, w, label, value, vmax, col, text=None):
    """A compact horizontal indicator: label, track + fill, value."""
    d.text((x, y), label, font=F_NAV, fill=GREY, anchor="lm")
    bx = x + 62
    d.rectangle([bx, y - 3, bx + w, y + 3], fill=(32, 32, 32))
    d.rectangle([bx, y - 3, bx + w * max(0.0, min(1.0, value / vmax)), y + 3], fill=col)
    d.text((bx + w + 6, y), text if text is not None else f"{value}%", font=F_MONO_S, fill=TXT, anchor="lm")


def draw_sat_screen(d, p, top, col):
    """Left: the transfer curve (output vs input) at this AMOUNT against the clean
    diagonal; lopsided = 2nd harmonic. Right: which frequencies get saturated + mix."""
    y0 = top + 17
    fx_header(d, top, col, "SATURATION", f"AMOUNT {p['amount'] / 10:.1f} · {p['stype']}")
    bx0, bx1 = GX0, GX0 + (GY1 - y0) * 2
    by0, by1 = y0, GY1
    graph_box(d, bx0, by0, bx1, by1)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    d.line([(cx, by0), (cx, by1)], fill=GRID)
    d.line([(bx0, cy), (bx1, cy)], fill=GRID)
    d.line([(bx0, by1), (bx1, by0)], fill=AXIS)                           # clean = straight
    key = ("sat", p["amount"])
    if _scr.get("satkey") != key:
        sat = native_fx.Saturation()
        sat.set(dict(p, mix=100, out=0))
        xs = np.linspace(-1, 1, 81)
        _scr["satkey"], _scr["satcurve"] = key, (xs, sat.shape(xs * 1.0))
    xs, ys = _scr["satcurve"]
    ys = ys / max(1e-6, float(np.abs(ys).max()))                    # show the shape full-size
    pts = [(cx + x * (bx1 - bx0) / 2, cy - max(-1.2, min(1.2, y)) * (by1 - by0) / 2) for x, y in zip(xs, ys)]
    d.line(pts, fill=col, width=2)
    d.text((bx0 + 4, by0 + 6), "OUT", font=F_MONO_S, fill=DIM, anchor="lm")
    d.text((bx1 - 4, by1 - 6), "IN", font=F_MONO_S, fill=DIM, anchor="rm")
    # frequency strip: the saturated region lit
    fx0, fx1 = bx1 + 24, GX1
    fy0, fy1 = y0 + 16, y0 + 34
    xf = lambda f: fx0 + (fx1 - fx0) * math.log(f / 20) / math.log(1000)
    lo, hi = {"KEEP LOW": (200, 20000), "KEEP HIGH": (20, 3000)}.get(p["stype"], (20, 20000))
    d.text((fx0, y0 + 10), "SATURATED RANGE", font=F_NAV, fill=GREY, anchor="lm")
    graph_box(d, fx0, fy0, fx1, fy1)
    d.rectangle([xf(lo), fy0, xf(hi), fy1], fill=shade(col, GRAPH_BG, 0.45))
    for f, lab in ((100, "100"), (1000, "1K"), (10000, "10K")):
        d.line([(xf(f), fy1), (xf(f), fy1 + 3)], fill=GREY)
        d.text((xf(f), fy1 + 8), lab, font=F_MONO_S, fill=DIM, anchor="mm")
    hw = (fx1 - fx0) // 2 - 62 - 56
    hbar(d, fx0, GY1 - 8, hw, "MIX", p["mix"], 100, col)
    hbar(d, fx0 + (fx1 - fx0) // 2 + 8, GY1 - 8, hw, "OUT", p["out"] + 24, 30, col, f"{p['out']:+d} dB")


def draw_crush_screen(d, p, top, col):
    """A clean sine (grey) and what the crusher makes of it (steps): RATE = how wide
    the steps are, BITS = how tall, JITTER = uneven widths, CRUNCH = clipping."""
    y0 = top + 17
    fx_header(d, top, col, "BITCRUSHER", f"{p['bits']} BITS · {p['rate'] / 1000:g} kHz · JITTER {p['jitter']}% · "
              f"CRUNCH {p['crunch']}% · MIX {p['mix']}%")
    graph_box(d, GX0, y0, GX1, GY1)
    n = GX1 - GX0
    xs = np.arange(n)
    wave = np.sin(2 * np.pi * xs / n * 2.2) * 0.85                         # 2.2 cycles across
    step_px = max(1.0, 48000 / p["rate"] * 3 / 3)
    rng = np.random.default_rng(7)                                         # steady jitter pattern
    edges, x = [], 0.0
    while x < n:
        edges.append(int(x))
        x += step_px * (1 + p["jitter"] / 100 * 0.9 * (rng.random() - 0.5) * 2)
    held = np.empty(n)
    for i, e in enumerate(edges):
        held[e:edges[i + 1] if i + 1 < len(edges) else n] = wave[e]
    q = 2 ** (p["bits"] - 1)
    held = np.round(held * q) / q
    if p["crunch"]:
        g = 1 + 9 * p["crunch"] / 100
        held = np.clip(held * g, -1, 1) / max(1.0, g * 0.6)
    mid, amp = (y0 + GY1) / 2, (GY1 - y0) / 2 - 4
    for k in (-0.5, 0.5):
        d.line([(GX0, mid - k * amp * 2), (GX1, mid - k * amp * 2)], fill=GRID)
    for k in range(1, 8):
        d.line([(GX0 + k * n / 8, y0), (GX0 + k * n / 8, GY1)], fill=GRID)
    d.line([(GX0, mid), (GX1, mid)], fill=GRID_MAJOR)
    d.line([(GX0 + i, mid - wave[i] * amp) for i in range(0, n, 2)], fill=(96, 96, 96), width=1)
    d.line([(GX0 + i, mid - held[i] * amp) for i in range(n)], fill=col, width=2)
    d.text((GX1 - 4, y0 + 7), "GREY = CLEAN", font=F_MONO_S, fill=DIM, anchor="rm")


def draw_reel(d, cx, cy, r, ang, col):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(144, 144, 144), width=2)
    d.ellipse([cx - r * 0.78, cy - r * 0.78, cx + r * 0.78, cy + r * 0.78], fill=shade(col, (24, 24, 24), 0.28))
    for k in range(3):                                   # spokes show the turning
        a = ang + k * 2 * math.pi / 3
        d.line([(cx + 7 * math.cos(a), cy + 7 * math.sin(a)),
                (cx + r * 0.72 * math.cos(a), cy + r * 0.72 * math.sin(a))], fill=(176, 176, 176), width=3)
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(200, 200, 200))
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=GRAPH_BG)


def draw_tape_screen(d, p, top, col):
    """TAPE: two reels (wobbling with WOW / FLUTTER) and the tape path, plus compact
    bars for the settings. VINYL: the record under the arm (dust specks, warp)."""
    y0 = top + 17
    t = time.monotonic()
    graph_box(d, GX0, y0, GX0 + 380, GY1)
    cy = (y0 + GY1) / 2
    r = (GY1 - y0) / 2 - 6
    bx, bw = GX0 + 400, 220
    if p["mode"] == "TAPE":
        f = p["filter"]
        fx_header(d, top, col, "TAPE", "FILTER " + ("OFF" if f == 0 else (f"LP {-f}" if f < 0 else f"HP {f}")))
        wob = p["wow"] / 100 * 0.25 * math.sin(t * 2 * math.pi * 0.45) + \
            p["flutter"] / 100 * 0.08 * math.sin(t * 2 * math.pi * 7)
        ang = t * 2.4 * (1 + wob)
        l, rr_ = GX0 + 80, GX0 + 300
        d.line([(l, cy + r), (rr_, cy + r)], fill=shade(col, GRAPH_BG, 0.6), width=2)     # the tape
        d.rectangle([(l + rr_) / 2 - 16, cy + r - 6, (l + rr_) / 2 + 16, cy + r + 2], fill=(64, 64, 64))  # head
        draw_reel(d, l, cy, r, ang, col)
        draw_reel(d, rr_, cy, r, ang * 1.1, col)
        rows = (("DRIVE", p["drive"]), ("WOW", p["wow"]), ("FLUTTER", p["flutter"]), ("NOISE", p["noise"]),
                ("WEAR", p["wear"]))
    else:
        fx_header(d, top, col, "VINYL", f"ERA {p['era']}")
        cx = GX0 + 150
        warp = p["warp"] / 100 * 3 * math.sin(t * 2 * math.pi * 0.555)
        d.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], fill=(32, 32, 32))   # platter
        d.ellipse([cx - r, cy - r + warp, cx + r, cy + r + warp], fill=(12, 12, 12), outline=(72, 72, 72))
        for k in range(3, 8):                                    # grooves
            rr = r * k / 8
            d.ellipse([cx - rr, cy - rr + warp, cx + rr, cy + rr + warp], outline=(32, 32, 32))
        d.ellipse([cx - r * 0.3, cy - r * 0.3 + warp, cx + r * 0.3, cy + r * 0.3 + warp], fill=col)
        a = t * 2 * math.pi * 0.555                              # 33 1/3 rpm
        d.line([(cx, cy + warp), (cx + r * 0.3 * math.cos(a), cy + warp + r * 0.3 * math.sin(a))],
               fill=GRAPH_BG, width=2)
        rnd = np.random.default_rng(int(t * 8))                  # dust specks flicker
        for _ in range(int(p["dust"] / 6)):
            ra, rr = rnd.random() * 2 * math.pi, r * (0.35 + 0.6 * rnd.random())
            px, py = cx + rr * math.cos(ra), cy + warp + rr * math.sin(ra)
            d.point([(px, py), (px + 1, py)], fill=(200, 200, 200))
        d.ellipse([cx + r + 26, cy - r - 2, cx + r + 40, cy - r + 12], fill=(160, 160, 160))     # arm pivot
        d.line([(cx + r + 33, cy - r + 5), (cx + r * 0.55, cy + r * 0.25)], fill=(184, 184, 184), width=3)
        rows = (("DUST", p["dust"]), ("SCRATCH", p["scratch"]), ("WARP", p["warp"]), ("WEAR", p["vwear"]),
                ("MIX", p["vmix"]))
    for i, (lab, v) in enumerate(rows):
        hbar(d, bx, y0 + 7 + i * 12, bw, lab, v, 100, col)


def draw_reverb_screen(d, p, top, col):
    """The tail: how long (SIZE), how dark it gets (DAMP: the fill fades darker), the
    dry hit and wet level as set (independent), and the stereo WIDTH."""
    y0 = top + 17
    fx_header(d, top, col, "REVERB", f"SIZE {p['size']}% · DAMP {p['damp']}%")
    x0, x1 = GX0, GX0 + 420
    graph_box(d, x0, y0, x1, GY1)
    for k in range(1, 4):
        d.line([(x0, y0 + k * (GY1 - y0) / 4), (x1, y0 + k * (GY1 - y0) / 4)], fill=GRID)
    base = GY1 - 1
    dry_h = (GY1 - y0 - 4) * p["dry"] / 100
    d.rectangle([x0 + 8, base - dry_h, x0 + 13, base], fill=TXT)          # the dry hit
    length = 0.15 + 0.85 * p["size"] / 100                                 # share of the width
    wet_h = (GY1 - y0 - 4) * 0.9 * p["wet"] / 100
    n = 70
    span = x1 - x0 - 24
    for i in range(n):
        f = i / n
        if f > length:
            break
        lvl = math.exp(-4.0 * f / length)
        dark = 1 - p["damp"] / 100 * f / length * 0.8                     # damping darkens the tail
        c = shade(col, GRAPH_BG, 0.3 + 0.7 * dark)
        xa = x0 + 18 + span * f
        d.rectangle([xa, base - wet_h * lvl, xa + span / n, base], fill=c)
    # stereo width: two sources spread from the centre
    wx0, wx1 = x1 + 14, x1 + 120
    graph_box(d, wx0, y0, wx1, y0 + 52)
    wc = (wx0 + wx1) / 2
    d.line([(wc, y0 + 6), (wc, y0 + 46)], fill=GRID_MAJOR)
    sp = 44 * p["width"] / 100
    for sx in (wc - sp, wc + sp):
        d.ellipse([sx - 5, y0 + 21, sx + 5, y0 + 31], fill=col)
    d.text((wc, y0 + 44), "WIDTH", font=F_MONO_S, fill=DIM, anchor="mm")
    bx = x1 + 136
    for i, (lab, v) in enumerate((("WET", p["wet"]), ("DRY", p["dry"]), ("WIDTH", p["width"]))):
        hbar(d, bx, y0 + 10 + i * 16, 70, lab, v, 100, col)


def draw_delay_screen(d, p, top, col):
    """The echoes in time over one bar: each tap at the synced TIME, shorter as FEEDBACK
    lets them fade; PING PONG alternates left (up) / right (down); lo-fi QUALITY makes
    later taps darker; WOBBLE bends the line; FREEZE keeps them."""
    y0 = top + 17
    beats = dict(DELAY_NOTES)[p["time"]]
    fx_header(d, top, col, "BBD DELAY", f"{p['time']} · {p['mode']} · FEEDBACK {p['fb']}% · MIX {p['mix']}% · "
              f"QUALITY {p['quality']}%" + (" · FREEZE" if p["freeze"] == "ON" else ""))
    graph_box(d, GX0, y0, GX1, GY1)
    mid = (y0 + GY1) / 2
    span = 4.0                                                    # one bar
    xb = lambda b: GX0 + 24 + (GX1 - GX0 - 48) * b / span
    for k in range(17):                                           # 16ths, beats brighter
        d.line([(xb(k / 4), y0 + 2), (xb(k / 4), GY1 - 12)], fill=GRID_MAJOR if k % 4 == 0 else GRID)
    for b in range(5):
        d.text((xb(b), GY1 - 6), str(b + 1) if b < 4 else "1", font=F_MONO_S, fill=DIM, anchor="mm")
    t = time.monotonic()
    wob = p["wobble"] / 100
    d.line([(x, mid + wob * 3 * math.sin(x / 25 + t * 3)) for x in range(GX0 + 24, GX1 - 24, 6)], fill=AXIS)
    h = (GY1 - 12 - y0) / 2 - 4
    d.rectangle([xb(0) - 3, mid - h, xb(0) + 3, mid + h], fill=TXT)       # the dry hit
    fb = 1.0 if p["freeze"] == "ON" else p["fb"] / 100
    lvl, k = 0.35 + 0.65 * p["mix"] / 100, 1
    while beats * k <= span and lvl > 0.02:
        x = xb(beats * k)
        q = p["quality"] / 100
        dark = q + (1 - q) * max(0.3, 1 - 0.18 * k)
        c = shade(col, GRAPH_BG, dark)
        hh = h * min(1.0, lvl)
        if p["mode"] == "PING PONG":
            d.rectangle([x - 3, mid - hh, x + 3, mid] if k % 2 else [x - 3, mid, x + 3, mid + hh], fill=c)
        else:
            d.rectangle([x - 3, mid - hh, x + 3, mid + hh], fill=c)
        lvl *= fb
        k += 1
    if p["mode"] == "PING PONG":
        d.text((GX0 + 6, y0 + 8), "L", font=F_MONO_S, fill=DIM, anchor="lm")
        d.text((GX0 + 6, GY1 - 18), "R", font=F_MONO_S, fill=DIM, anchor="lm")


FX_SCREENS = {"SAT": draw_sat_screen, "CRUSH": draw_crush_screen, "TAPE": draw_tape_screen,
              "REVERB": draw_reverb_screen, "DELAY": draw_delay_screen}


def draw_device_row(d):
    """Bottom edge on the Device page: the slots and the buses, above their buttons."""
    for k in range(8):
        if k < NUM_SLOTS:
            s = buses[dev["bus"]]["slots"][k]
            cur = k == dev["slot"]
            name = SHORT_NAMES.get(s["type"], s["type"]) if s else "—"
            off = bool(s and not s["on"])
            nav_cell(d, k, f"{k + 1} {name}", BUS_COLOURS[dev["bus"]] if cur else (GREY if s else DIM), active=cur,
                     sub="OFF" if off else None)
        else:
            b = k - NUM_SLOTS
            muted = buses[b].get("mute", False)
            used = any(s for s in buses[b]["slots"])
            nav_cell(d, k, f"BUS {b + 1}", WARN if muted else (BUS_COLOURS[b] if b == dev["bus"] else
                                                               (GREY if used else DIM)),
                     active=b == dev["bus"], sub="M" if muted else None, sub_colour=WARN)


def device_open():
    return ui["page"] == "DEVICE" and ui["open"]


# ---------------------------------------------------------------------------
# Master page drawn as hardware: a bus-compressor module (knobs 1-6, needle meter)
# and a limiter module (knobs 7-8, meters). Knobs sit under their encoders.
# The panels are drawn once at 2x and scaled down (smooth edges); only pointers,
# values, the needle and the bars are drawn each frame.
# ---------------------------------------------------------------------------
COMP_PANEL, LIM_PANEL = (54, 57, 63), (30, 31, 35)
KNOB_BLUE, KNOB_GREY = (58, 104, 176), (150, 152, 158)
VU = {"box": (14, 102, 254, 154), "pivot": (134, 313), "r": 195, "a0": -27, "a1": 27}
KNOB_Y, KNOB_R = 59, 18


def knob_ticks(i):
    """How many dots around knob i (stepped knobs: one per position)."""
    return [7, len(RATIO_CHOICES), len(ATTACK_CHOICES), len(RELEASE_CHOICES), 6, 11, 5, 11][i]


def knob_fraction(i):
    m = master_fx
    return [(m["thresh"] + 30) / 30, RATIO_CHOICES.index(m["ratio"]) / (len(RATIO_CHOICES) - 1),
            ATTACK_CHOICES.index(m["attack"]) / (len(ATTACK_CHOICES) - 1),
            RELEASE_CHOICES.index(m["release"]) / (len(RELEASE_CHOICES) - 1),
            m["makeup"] / 15, m["mix"] / 100, m["drive"] / 12, master["volume"]][i]


def knob_angle(frac):                                   # -135..+135 degrees, 0 = up
    return math.radians(-135 + 270 * max(0.0, min(1.0, frac)))


def vu_angle(db):                                       # 0 dB left .. 20 dB right
    a0, a1 = VU["a0"], VU["a1"]
    return math.radians(a0 + (a1 - a0) * max(0.0, min(1.0, db / 20)))


_master_bg = {}


def master_background():
    if "img" in _master_bg:
        return _master_bg["img"]
    k = 2
    img = Image.new("RGB", (W * k, H * k), BLACK)
    g = ImageDraw.Draw(img)
    S = lambda *v: [int(a * k) for a in v]
    g.rounded_rectangle(S(4, 18, 716, 157), 8 * k, fill=COMP_PANEL, outline=(80, 84, 92), width=k)
    g.rounded_rectangle(S(724, 18, 956, 157), 8 * k, fill=LIM_PANEL, outline=(70, 70, 78), width=k)
    for x, y in ((12, 26), (708, 26), (12, 149), (708, 149), (732, 26), (948, 26), (732, 149), (948, 149)):
        g.ellipse(S(x - 3, y - 3, x + 3, y + 3), fill=(28, 28, 32))                 # screws
        g.line(S(x - 2, y, x + 2, y), fill=(90, 90, 98), width=k)
    for i in range(8):
        cx, cy, r = i * SLOT_W + SLOT_W / 2, KNOB_Y, KNOB_R
        n = knob_ticks(i)
        for t in range(n):
            a = knob_angle(t / (n - 1))
            px, py = cx + (r + 6) * math.sin(a), cy - (r + 6) * math.cos(a)
            g.ellipse(S(px - 1.3, py - 1.3, px + 1.3, py + 1.3), fill=(215, 215, 220))
        body = KNOB_BLUE if i < 6 else KNOB_GREY
        g.ellipse(S(cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2), fill=(20, 20, 24))
        g.ellipse(S(cx - r, cy - r, cx + r, cy + r), fill=body)
        hi = tuple(min(255, c + 35) for c in body)
        g.ellipse(S(cx - r + 4, cy - r + 3, cx + r - 6, cy + r - 8), fill=hi)
        g.ellipse(S(cx - r + 5, cy - r + 5, cx + r - 5, cy + r - 5), fill=body)
    # needle meter face
    x0, y0, x1, y1 = VU["box"]
    g.rounded_rectangle(S(x0 - 3, y0 - 3, x1 + 3, y1 + 2), 4 * k, fill=(16, 16, 18))
    g.rectangle(S(x0, y0, x1, y1), fill=(232, 222, 196))
    px, py = VU["pivot"]
    rr = VU["r"]
    for db in range(0, 21, 2):
        a = vu_angle(db)
        long = db % 4 == 0
        r0 = rr - (9 if long else 5)
        g.line(S(px + r0 * math.sin(a), py - r0 * math.cos(a), px + rr * math.sin(a), py - rr * math.cos(a)),
               fill=(20, 20, 20), width=k * (2 if long else 1))
    g.arc(S(px - rr, py - rr, px + rr, py + rr), 270 + VU["a0"], 270 + VU["a1"], fill=(20, 20, 20), width=k)
    img = img.resize((W, H), Image.LANCZOS)
    g = ImageDraw.Draw(img)
    for db in range(0, 21, 4):                              # scale numbers at 1x (crisp text)
        a = vu_angle(db)
        tx, ty = px + (rr + 8) * math.sin(a), py - (rr + 8) * math.cos(a)
        g.text((tx, ty), str(db), font=F_LABEL, fill=(20, 20, 20), anchor="mm")
    g.text(((x0 + x1) / 2, y1 - 8), "dB  COMPRESSION", font=F_LABEL, fill=(60, 50, 40), anchor="mm")
    _master_bg["img"] = img
    return img


def draw_master_device(d):
    img = d._image
    img.paste(master_background())
    draw_tabs(d)
    m = master_fx
    params = PAGES["MASTER"]["knobs"]
    for i, prm in enumerate(params):
        cx = i * SLOT_W + SLOT_W / 2
        caps(d, (cx, 21), prm.label, (205, 205, 212), anchor="ma")
        a = knob_angle(knob_fraction(i))
        x1, y1 = cx + (KNOB_R - 3) * math.sin(a), KNOB_Y - (KNOB_R - 3) * math.cos(a)
        x0, y0 = cx + 5 * math.sin(a), KNOB_Y - 5 * math.cos(a)
        d.line([(x0, y0), (x1, y1)], fill=(250, 250, 250), width=3)
        d.text((cx, 91), prm.text(), font=F_ROW, fill=TXT, anchor="mm")
    # compressor: IN lamp, title, input level against the threshold
    on = bool(m["ratio"])
    d.rounded_rectangle([660, 104, 704, 122], 3, fill=(235, 235, 235) if on else (70, 72, 78),
                        outline=(20, 20, 24))
    d.text((682, 113), "IN", font=F_LABEL, fill=(20, 20, 20) if on else (140, 140, 148), anchor="mm")
    caps(d, (272, 106), "STEREO BUS COMPRESSOR", (200, 200, 208))
    lo, hi_db = -36, 0
    bx0, bx1, by = 272, 640, 128
    ind = 20 * math.log10(max(meters["in"], 1e-6))
    d.rectangle([bx0, by, bx1, by + 8], fill=(24, 25, 28))
    w = int((bx1 - bx0) * max(0, min(1, (ind - lo) / (hi_db - lo))))
    if w:
        d.rectangle([bx0, by, bx0 + w, by + 8], fill=(90, 200, 120) if ind < m["thresh"] else (232, 170, 60))
    tx = bx0 + (bx1 - bx0) * (m["thresh"] - lo) / (hi_db - lo)
    d.line([(tx, by - 3), (tx, by + 11)], fill=(250, 250, 250), width=2)
    caps(d, (bx0, by + 13), "IN", GREY)
    caps(d, (tx, by + 13), "THRESH", GREY, anchor="ma")
    # needle (a little inertia, like the real thing)
    target = -meters["gr"] if on else 0.0
    nd = _master_bg.get("needle", 0.0)
    nd += (target - nd) * 0.35
    _master_bg["needle"] = nd
    px, py = VU["pivot"]
    a = vu_angle(nd)
    tip = (px + (VU["r"] - 2) * math.sin(a), py - (VU["r"] - 2) * math.cos(a))
    base_r = (py - VU["box"][3]) / max(0.2, math.cos(a))          # start where the needle enters the window
    base = (px + base_r * math.sin(a), py - base_r * math.cos(a))
    d.line([base, tip], fill=(20, 20, 20), width=2)
    # limiter: reduction + output meters with the ceiling
    caps(d, (736, 104), f"LIMITER  {CEILING_DB} dB", (200, 200, 208))
    rows = [("GR", -meters["lim"] / 6, RED, meters["lim"])]
    for ch, nm in enumerate(("L", "R")):
        db_ = 20 * math.log10(max(meters["out"][ch], 1e-6))
        rows.append((nm, (db_ + 36) / 36, (90, 200, 120) if db_ < -6 else (230, 200, 70), db_))
    for r_, (nm, frac, col, val) in enumerate(rows):
        y = 118 + r_ * 12
        caps(d, (736, y), nm, GREY)
        d.rectangle([756, y, 944, y + 7], fill=(14, 14, 16))
        w = int(188 * max(0.0, min(1.0, frac)))
        if w:
            d.rectangle([756, y, 756 + w, y + 7], fill=col)
    cx_ = 756 + 188 * (CEILING_DB + 36) / 36
    d.line([(cx_, 128), (cx_, 145)], fill=(250, 250, 250), width=1)


def draw_toast(d):
    t = ui.get("toast")
    if t and time.monotonic() - t[1] < 2.0:
        w = int(F_BODY.getlength(t[0])) + 24
        d.rounded_rectangle([948 - w, 117, 948, 141], 3, fill=RAISED, outline=BORDER)
        d.text((948 - w + 12, 129), t[0], font=F_BODY, fill=TXT, anchor="lm")


def frame():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    draw_main(d)
    if ui.get("help"):
        draw_help(d)
    return np.asarray(img, dtype=np.uint8)


def draw_main(d):
    if cal_ui["on"]:
        draw_calibration(d)
        draw_toast(d)
        return
    if browser.active:
        item = browser.current()
        if ui.get("pending") and (item is None or item[1] != ui["pending"][1]):
            ui["pending"] = None                       # you moved on: forget the question
        draw_browser(d)
        if ui.get("pending"):
            draw_confirm(d)
        return
    if proj["open"]:
        draw_projects(d)
        draw_toast(d)
        return
    if smp["open"]:
        draw_sampling(d)
        draw_toast(d)
        return
    if song_open() and not ui.get("settings") and not held_steps and not held_note_steps \
            and not ui.get("pad_move"):
        draw_strip(d)
        draw_song(d)
        draw_toast(d)
        return
    if mix_open() and not ui.get("settings") and not held_steps and not held_note_steps \
            and not ui.get("pad_move"):
        draw_strip(d)
        draw_mixer(d)
        draw_toast(d)
        return
    draw_strip(d)
    draw_left(d)
    if ui.get("pad_move"):
        draw_pad_move(d)
        draw_toast(d)
        return
    if chop_view():
        draw_chop(d)
    elif sample_page():
        draw_trim(d)
    elif ui.get("seq_view") and ui["open"] and ui["page"] in ("MASTER", "DEVICE", "INST") + PAD_FX:
        draw_roll(d) if (kb["mode"] and ui["selected"] in pad_samples) else \
            (draw_grid(d) if ui["view"] == "seq" else draw_wave(d))     # Shift + Layout: the steps
    elif ui["page"] == "MASTER" and ui["open"] and not ui.get("settings"):
        draw_master_fx(d)
    elif device_open() and not ui.get("settings"):
        draw_device(d)
    elif pad_fx_open() and not ui.get("settings"):
        draw_pad_fx(d)
    elif ui["page"] == "INST" and ui["open"] and not ui.get("settings"):
        draw_inst(d)
    else:
        if kb["mode"] and ui["selected"] in pad_samples:
            draw_roll(d)
        else:
            draw_grid(d) if ui["view"] == "seq" else draw_wave(d)
    draw_device_row(d) if device_open() else draw_pattern_row(d)
    if ui.get("confirm_clear") is not None:
        draw_clear_confirm(d)
    else:
        draw_toast(d)

# ---------------------------------------------------------------------------


def dump_diag():
    """Slow audio blocks go to dropouts.log in the app folder (for Claude)."""
    if not diag["slow"]:
        return
    rows, diag["slow"] = diag["slow"], []
    with open(os.path.join(APP_DIR, "dropouts.log"), "a") as f:
        for t, took, gap, mms, nv, ph0, ph1, under, playing in rows:
            f.write(f"{t:10.3f} took {took:5.2f} ms  gap {gap:6.2f} ms  master {mms:4.2f} ms  voices {nv:2d}  "
                    f"main {ph0}->{ph1}  {'UNDERFLOW' if under else ''}  {'playing' if playing else 'stopped'}\n")


CRASH_LOG = os.path.join(APP_DIR, "crash.log")
_crash = {"last": 0.0}


def log_error(where):
    """Write the error to crash.log (for Claude) instead of quitting."""
    import traceback
    with open(CRASH_LOG, "a") as f:
        f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} in {where} ---\n")
        traceback.print_exc(file=f)
    traceback.print_exc()
    if time.monotonic() - _crash["last"] > 3:
        _crash["last"] = time.monotonic()
        toast("something went wrong: saved in crash.log")


def safe_on_message(msg):
    if audio_state.get("bounce_tid"):
        return                                      # rendering an export: hands off for a moment
    try:
        on_message(msg)
    except Exception:
        log_error("button/pad handling")


splash = {"msg": "", "stop": None, "thread": None, "t0": 0.0}


def splash_start(display):
    """The load screen (load_screen.png) while the project and the audio start up;
    sent from a thread because the Push blanks its screen when frames stop."""
    try:
        base = Image.open(os.path.join(APP_DIR, "load_screen.png")).convert("RGB")
    except OSError:
        return
    if base.size != (W, H):
        base = base.resize((W, H), Image.LANCZOS)
    stop = threading.Event()

    def run():
        shown, arr, frame = None, None, None
        try:
            while not stop.is_set():
                if splash["msg"] != shown:
                    shown = splash["msg"]
                    img = base.copy()
                    if shown:
                        d = ImageDraw.Draw(img)
                        tw = F_MONO_S.getlength(shown.upper())
                        d.rectangle([948 - tw - 10, 143, 952, 157], fill=BG)
                        d.text((948, 150), shown.upper(), font=F_MONO_S, fill=GREY, anchor="rm")
                    arr = display_correct(np.asarray(img, dtype=np.uint8))
                    frame = push2.encode_frame(arr)
                display.send(frame)
                time.sleep(1 / 30)
            for k in range(5, -1, -1):                          # a short fade out
                display.send(push2.encode_frame(display_correct((np.asarray(base, dtype=np.float32) * k / 6)
                                                                .astype(np.uint8))))
                time.sleep(1 / 30)
        except Exception:
            pass

    splash.update(stop=stop, t0=time.monotonic(), thread=threading.Thread(target=run, daemon=True))
    splash["thread"].start()


def splash_end(min_time=1.5):
    if splash["thread"] is None:
        return
    time.sleep(max(0.0, min_time - (time.monotonic() - splash["t0"])))
    splash["stop"].set()
    splash["thread"].join(timeout=2)
    splash["thread"] = None


def main():
    global midi
    import faulthandler                               # a hard crash still leaves a trace
    faulthandler.enable(open(CRASH_LOG, "a"), all_threads=True)
    display = push2.PushDisplay()
    ui["display"] = display
    load_calibration()
    splash["msg"] = "loading project"
    splash_start(display)
    open_initial_project()
    splash["msg"] = f"project {proj['cur']:02d}" + (f" · {proj['name']}" if proj["name"] else "") + " · starting audio"
    midi = push2.PushMidi(safe_on_message)
    midi.clear_pads()
    setup_button_colours()
    # ask the Push for pressure per pad (polyphonic aftertouch), used by note repeat
    midi.send(midi.mido.Message("sysex", data=[0x00, 0x21, 0x1D, 0x01, 0x01, 0x1E, 0x01]))
    global master_bus
    # The effects let go of Python's lock while they run; by default the audio thread
    # then waits up to 5 ms to get it back from the screen code (= dropouts at 2.7 ms
    # per block). 0.5 ms keeps the audio thread on time.
    no_lock = not getattr(sys, "_is_gil_enabled", lambda: True)()
    print("Python", sys.version.split()[0], "- lock (GIL)",
          "OFF: audio and screen run in parallel" if no_lock else "ON (start with ./run.sh for no dropouts)")
    if not no_lock:
        sys.setswitchinterval(0.00002)
    ui["fps"] = 30 if no_lock else 20
    global bus_engine, pad_engine
    try:
        master_bus = MasterBus()
        bus_engine = BusEngine()
        bus_engine.sync()
        pad_engine = PadEngine()
        pad_engine.sync()
    except ImportError:
        print("Master bus off: python -m pip install pedalboard")
    # Python's automatic garbage collection sometimes scans every object at once,
    # holding the lock for 10-20 ms while the audio thread waits = dropouts. So it's
    # off: we clear young objects every frame instead, and freeze what exists now.
    gc.disable()
    gc.collect()
    gc.freeze()
    def open_stream(block):
        """The audio output. latency="low" asks the AudioFuse for small chunks too, so
        the block size really sets the delay between hitting a pad and hearing it."""
        st = sd.OutputStream(samplerate=SR, channels=2, device="AudioFuse", blocksize=block,
                             dtype="float32", callback=audio_callback, latency="low")
        st.start()
        audio_state["latency"] = st.latency
        audio_state["block"] = block
        print(f"Audio: {block}-sample blocks, output latency {st.latency * 1000:.1f} ms")
        return st

    stream = open_stream(setup.get("buffer", 128))
    splash_end()
    print("\nMPushC sequencer. Hit a kit pad, then program steps on the top two rows.")
    print("Buttons under the knobs = pages: 1 PAD, 7 PATTERN, 8 SETUP. Labels are on screen.")
    print("Ctrl+C to quit.\n")
    try:
        while True:
            try:
                diag["phase"] = "ticks"
                apply_follow()
                browser.tick()
                sampling_tick()
                detect_tick()
                stems_tick()
                follow_tempo()
                if bus_engine is not None:
                    bus_engine.sync()
                if pad_engine is not None:
                    pad_engine.sync()
                follow_selection()
                vary_tick()
                auto_tick()
                run_main_jobs()
                if setup.get("buffer", 128) != audio_state["block"]:     # BUFFER changed (Settings)
                    stream.stop()
                    stream.close()
                    stream = open_stream(setup.get("buffer", 128))
                    toast(f"buffer {audio_state['block']}: output latency {stream.latency * 1000:.1f} ms")
                audio_state["cpu"] = 0.8 * audio_state["cpu"] + 0.2 * stream.cpu_load * 100   # smoothed
                diag["phase"] = "lights"
                update_lights()
                diag["phase"] = "autosave"
                autosave()
                diag["phase"] = "draw"
                img = frame()
                if ui.pop("shot", False):
                    save_screenshot(img)
                diag["phase"] = "display"
                display.show(display_correct(img))
                diag["phase"] = "sleep"
                dump_diag()
                if not no_lock:
                    gc.collect(1)                               # young objects only: well under 1 ms
                # (free-threaded Python has no "young objects only": every collection
                # pauses ALL threads, audio included, so there we don't collect at all;
                # plain reference counting still frees almost everything)
                time.sleep(1 / ui.get("fps", 20))               # with the lock on: 20 fps, less fighting
            except Exception:
                log_error("main loop")
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if save_state["dirty_since"] is not None:
                save_project()
                print("Saved.")
        except Exception as e:
            print("Could not save:", e)
        try:
            clock["playing"] = False
            midi.clear_pads()
            for c in (PLAY_BUTTON, SESSION_BUTTON, NOTE_BUTTON, PAGE_LEFT, PAGE_RIGHT,
                      DOUBLE_BUTTON, FOLLOW_BUTTON, ZOOM_IN, ZOOM_OUT, MUTE_BUTTON, SOLO_BUTTON,
                      BROWSE_BUTTON, ARROW_LEFT, ARROW_RIGHT, UNDO_BUTTON, DELETE_BUTTON, SHIFT_BUTTON,
                      DUPLICATE_BUTTON, *LOWER_BUTTONS, METRONOME_BUTTON, TAP_TEMPO_BUTTON,
                      RECORD_BUTTON, REPEAT_BUTTON, *RATE_BUTTONS, QUANTIZE_BUTTON, SELECT_BUTTON, NEW_BUTTON, CONVERT_BUTTON, SETUP_BUTTON, HELP_BUTTON,
                      ADD_TRACK_BUTTON, STOP_CLIP_BUTTON, FIXED_LENGTH_BUTTON,
                      ACCENT_BUTTON, SCALE_BUTTON,
                      OCTAVE_UP, OCTAVE_DOWN,
                      *UPPER_BUTTONS):
                midi.send(midi.mido.Message("control_change", control=c, value=0))
            stream.abort()
        except Exception:
            pass
        print("\nBye.")
        os._exit(0)


if __name__ == "__main__":
    main()
