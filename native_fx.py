"""MPushC's own effects, in numpy (+ scipy for filters). No plugins, exact tempo sync,
and they never hand Python's lock to the screen code mid-block like plugin calls do.

Every effect works on stereo blocks shaped (2, n), float32, and keeps its own state
between blocks. set(params, bpm) is called from the audio thread before process().
"""
import math

import numpy as np
from scipy.signal import lfilter

SR = 48000


# ---------------------------------------------------------------------------
# Small filter helpers (RBJ biquads), with per-channel state
# ---------------------------------------------------------------------------
def biquad(kind, f, q=0.707, sr=SR):
    w0 = 2 * math.pi * min(max(f, 5.0), sr * 0.45) / sr
    c, s_ = math.cos(w0), math.sin(w0)
    alpha = s_ / (2 * q)
    if kind == "lp":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
    else:                                                   # "hp"
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
    a = [1 + alpha, -2 * c, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


class Filter:
    """A biquad that remembers where it was, so it can run block after block."""

    def __init__(self, kind, f, q=0.707):
        self.kind, self.q = kind, q
        self.zi = np.zeros((2, 2))
        self.f = None
        self.set(f)

    def set(self, f):
        if f != self.f:
            self.f = f
            self.b, self.a = biquad(self.kind, f, self.q)

    def __call__(self, x):
        y, self.zi = lfilter(self.b, self.a, x, axis=-1, zi=self.zi)
        return y


# ---------------------------------------------------------------------------
# Saturation, in the spirit of Softube's Saturation Knob
# ---------------------------------------------------------------------------
class Oversampler:
    """x4 oversampling for a nonlinearity (keeps the harmonics it makes from folding
    back down as harsh, out-of-tune aliasing). Stateful, so it runs block by block."""
    F = 4

    def __init__(self, ch=2):
        from scipy.signal import ellip
        self.sos = ellip(10, 0.05, 100, 20000 / (SR * self.F / 2), output="sos")
        self.zu = np.zeros((self.sos.shape[0], ch, 2))
        self.zd = np.zeros((self.sos.shape[0], ch, 2))

    def up(self, x):
        from scipy.signal import sosfilt
        u = np.zeros((x.shape[0], x.shape[1] * self.F))
        u[:, ::self.F] = x * self.F
        y, self.zu = sosfilt(self.sos, u, axis=-1, zi=self.zu)
        return y

    def down(self, y):
        from scipy.signal import sosfilt
        z, self.zd = sosfilt(self.sos, y, axis=-1, zi=self.zd)
        return z[:, ::self.F]


class Saturation:
    """An analog output stage: a soft clipper with a slight bias.

    The bias makes the curve lopsided, which adds even (2nd) harmonics: warm, at low
    drive it acts like an exciter. Driving harder, the symmetric clipping takes over
    and odd (3rd, 5th...) harmonics grow: thicker, more compressed. Louder input pushes
    it harder too, like the real thing. Level is kept about constant.

    Creamy, not harsh: the curve runs 4x oversampled (no aliasing), the drive tops out
    at +18 dB, the highs going in are tilted down a little as the drive rises (so the
    top end saturates less: the fizz), and a gentle 6 dB/oct roll-off after the curve
    smooths the highest harmonics, like the transformer of a real output stage.

    TYPE (like the Knob's switch): NEUTRAL saturates everything; KEEP HIGH leaves the
    highs (above ~3 kHz) clean and saturates lows/mids; KEEP LOW leaves the lows (below
    ~200 Hz) clean and saturates the rest.
    """

    def __init__(self):
        self.split_hi = Filter("lp", 3000)
        self.split_lo = Filter("lp", 200)
        self.dc = Filter("hp", 12)
        self.os = Oversampler()
        self.tilt_in = Filter("lp", 20000)               # into the curve: highs a bit lower (a shelf, set below)
        self.tilt_out = Filter("lp", 20000)              # after it: back up (the tone stays)
        self.smooth = Filter("lp", 18000, 0.5)          # the transformer's gentle top roll-off
        self.amount, self.stype, self.mix, self.out, self.makeup = 3.0, "NEUTRAL", 1.0, 1.0, 1.0

    def set(self, p, bpm=None):
        amount = p["amount"] / 10                           # 0..10, like the Knob
        if amount != self.amount or not hasattr(self, "makeup"):
            self.amount = amount
            ref = 0.25 * np.sin(np.linspace(0, 2 * np.pi, 256, endpoint=False))   # a -12 dB tone
            got = float(np.sqrt(np.mean(self.shape(ref, 1.0) ** 2)))
            self.makeup = float(np.sqrt(np.mean(ref ** 2))) / max(got, 1e-6)       # keeps it level
            a = self.amount / 10
            tilt = -6.0 * a                                   # up to 6 dB less top end into the curve
            for f_, g_ in ((self.tilt_in, tilt), (self.tilt_out, -tilt)):
                sec = shelf("highshelf", 4000, g_)
                f_.b, f_.a = np.array(sec[:3]), np.array(sec[3:])
            self.smooth.set(20000 - 8000 * a)                 # 20 kHz .. 12 kHz
        self.stype, self.mix = p["stype"], p["mix"] / 100
        self.out = 10 ** (p["out"] / 20)

    def shape(self, x, makeup=None):
        a = self.amount / 10                                # 0..1
        if a <= 0:
            return x
        m = self.makeup if makeup is None else makeup
        g = 10 ** (a * 18 / 20)                             # up to +18 dB into the curve
        bias = 0.3 * (1 - 0.6 * a)                          # lopsided = 2nd harmonic; fades as drive rises
        tb = math.tanh(bias)
        y = np.tanh(g * x + bias) - tb
        return y / (g * (1 - tb * tb)) * m                  # level kept about the same

    def saturate(self, x):
        if self.amount <= 0:
            return x
        x = self.tilt_in(x)
        y = self.os.down(self.shape(self.os.up(x)))
        return self.smooth(self.tilt_out(y))

    def process(self, x):
        if self.stype == "KEEP HIGH":
            low = self.split_hi(x)                          # lows + mids get saturated
            wet = self.saturate(low) + (x - low)
        elif self.stype == "KEEP LOW":
            low = self.split_lo(x)                          # the sub stays clean
            wet = low + self.saturate(x - low)
        else:
            wet = self.saturate(x)
        wet = self.dc(wet)                                  # the bias leaves a little DC: remove it
        y = wet * self.mix + x * (1 - self.mix)
        return (y * self.out).astype(np.float32)


# ---------------------------------------------------------------------------
# BBD delay, in the spirit of analog bucket-brigade delays
# ---------------------------------------------------------------------------
class BBDDelay:
    """Echoes that get darker and dirtier on each repeat, with a slow warble.

    Each pass goes through the BBD's filter (darker), a soft clip (gentle grit) and,
    with QUALITY turned down, a lower 'clock' (sample-and-hold) and a little hiss -
    more so for long delays, as with real BBD chips. Changing TIME glides the pitch
    like the hardware. The time comes straight from the project BPM.
    """

    MAX = 4 * SR                                            # 4 s of memory

    def __init__(self):
        self.buf = np.zeros((2, self.MAX), np.float32)
        self.w = 0
        self.d = None                                       # current delay (samples), glides to target
        self.target = SR * 0.3
        self.fb, self.mix, self.mode, self.wobble, self.quality, self.freeze = 0.35, 0.3, "L/R", 0.15, 0.7, False
        self.phase, self.phase2 = 0.0, 0.0
        self.lp = Filter("lp", 6000)
        self.hp = Filter("hp", 60)

    def set(self, p, bpm):
        beats = dict(DELAY_NOTES)[p["time"]]
        self.target = min(self.MAX - 2048, beats * 60 / bpm * SR)
        self.fb, self.mix = p["fb"] / 100, p["mix"] / 100
        self.mode, self.wobble = p["mode"], p["wobble"] / 100
        self.quality, self.freeze = p["quality"] / 100, p["freeze"] == "ON"

    def process(self, x):
        n = x.shape[1]
        out = np.empty_like(x)
        i = 0
        while i < n:                                        # reads must stay behind writes
            step = min(n - i, 512, int((self.d or self.target) - 600))
            out[:, i:i + step] = self.block(x[:, i:i + step])
            i += step
        return out

    def block(self, x):
        n = x.shape[1]
        N = self.MAX
        d0 = self.target if self.d is None else self.d
        d1 = d0 + float(np.clip(self.target - d0, -0.3 * n, 0.3 * n))   # glide (pitch bend)
        self.d = d1
        t = np.arange(n)
        d = d0 + (d1 - d0) * t / n
        # warble: two slow LFOs, up to ~4 ms deep
        ph = self.phase + 2 * math.pi * 0.55 * t / SR
        ph2 = self.phase2 + 2 * math.pi * 0.13 * t / SR
        self.phase = (self.phase + 2 * math.pi * 0.55 * n / SR) % (2 * math.pi)
        self.phase2 = (self.phase2 + 2 * math.pi * 0.13 * n / SR) % (2 * math.pi)
        d = d + self.wobble * 0.004 * SR * (0.7 * np.sin(ph) + 0.3 * np.sin(ph2))
        d = np.maximum(d, n + 4)
        pos = (self.w + t - d) % N
        i0 = np.floor(pos).astype(np.int64)
        f = (pos - i0).astype(np.float32)
        i1 = (i0 + 1) % N
        wet = self.buf[:, i0] * (1 - f) + self.buf[:, i1] * f            # (2, n)
        # what goes back into the line
        ms = d1 / SR * 1000
        q = self.quality
        if self.freeze:
            feed = wet if self.mode != "PING PONG" else wet[::-1]
            write = feed
        else:
            fb = self.fb
            loop = np.tanh(1.3 * wet) / 1.3                                # gentle grit per repeat
            if self.mode == "PING PONG":
                mono = x.mean(axis=0)
                write = np.stack([mono + fb * loop[1], fb * loop[0]])
            else:
                write = x + fb * loop
            cutoff = 2500 + 9500 * q                                       # QUALITY: 2.5 to 12 kHz
            cutoff *= min(1.0, max(0.5, 400 / max(ms, 1))) ** (1 - q)      # longer = darker when lo-fi
            self.lp.set(cutoff)
            write = self.hp(self.lp(write))
            if q < 0.5:                                                    # a slower BBD clock
                k = 1 + int(round((0.5 - q) * 2 * 4 * min(2.0, ms / 400)))
                if k > 1:
                    write = np.repeat(write[:, ::k], k, axis=1)[:, :n]
                write = write + np.random.randn(2, n).astype(np.float32) * (0.5 - q) * 0.002
        idx = (self.w + t) % N
        self.buf[:, idx] = write
        self.w = (self.w + n) % N
        dry = min(1.0, 2 * (1 - self.mix))
        wv = min(1.0, 2 * self.mix)
        return (x * dry + wet * wv).astype(np.float32)


DELAY_NOTES = [("1/16", 0.25), ("1/8T", 1 / 3), ("1/8", 0.5), ("1/8D", 0.75), ("1/4", 1.0),
               ("1/4D", 1.5), ("1/2", 2.0)]


# ---------------------------------------------------------------------------
# Bitcrusher, in the spirit of "Bite"
# ---------------------------------------------------------------------------
class Crusher:
    """CRUNCH drives the signal into a hard-ish clipper first (grit that aliases nicely),
    RATE holds each sample for longer (any rate, not just fixed steps), JITTER makes that
    hold clock wobble randomly like cheap old converters, BITS drops resolution in
    steps, LOW CUT clears rumble and DC, MIX blends with the clean signal."""

    def __init__(self):
        self.ph = 0.0                                       # hold clock, carried across blocks
        self.held = np.zeros(2, np.float32)
        self.hp = Filter("hp", 20)
        self.bits, self.rate, self.jitter, self.crunch, self.mix = 8, 16000, 0.0, 0.0, 1.0

    def set(self, p, bpm=None):
        self.bits, self.rate = p["bits"], float(p["rate"])
        self.jitter, self.crunch = p["jitter"] / 100, p["crunch"] / 100
        self.hp.set(float(p["lowcut"]))
        self.mix = p["mix"] / 100

    def process(self, x):
        n = x.shape[1]
        y = x
        if self.crunch > 0:
            g = 1 + 9 * self.crunch
            y = np.clip(y * g, -1, 1) * 0.7 + np.tanh(y * g) * 0.3
            y = y / (1 + 0.6 * self.crunch * 3)             # keep it from jumping in level
        if self.rate < SR * 0.999:
            step = np.full(n, self.rate / SR)
            if self.jitter > 0:
                step = step * (1 + self.jitter * 0.9 * (np.random.rand(n) - 0.5) * 2)
            ph = self.ph + np.cumsum(step)
            k = np.floor(ph)
            prev = np.concatenate([[math.floor(self.ph)], k[:-1]])
            tick = k > prev                                 # the clock fires here: take a new sample
            idx = np.where(tick, np.arange(n), -1)
            idx = np.maximum.accumulate(idx)
            held = np.where(idx >= 0, y[:, np.maximum(idx, 0)], self.held[:, None])
            if tick.any():
                self.held = y[:, idx[-1]].copy()
            self.ph = float(ph[-1] % 1.0)
            y = held
        if self.bits < 16:
            q = 2 ** (self.bits - 1)
            y = np.round(y * q) / q
        y = self.hp(y)
        return (y * self.mix + x * (1 - self.mix)).astype(np.float32)


# ---------------------------------------------------------------------------
# 5-band EQ, in the spirit of Nova: low shelf, three bells, high shelf
# ---------------------------------------------------------------------------
EQ_BANDS = ["1 LO", "2", "3", "4", "5 HI"]          # same numbers as the dots on the graph
EQ_KINDS = ["lowshelf", "bell", "bell", "bell", "highshelf"]


def eq_section(kind, f, g, q, sr=SR):
    """One RBJ biquad as (b0, b1, b2, a0=1, a1, a2)."""
    A = 10 ** (g / 40)
    w0 = 2 * math.pi * min(max(f, 10.0), sr * 0.45) / sr
    c, s_ = math.cos(w0), math.sin(w0)
    alpha = s_ / (2 * max(q, 0.1))
    if kind == "bell":
        b = [1 + alpha * A, -2 * c, 1 - alpha * A]
        a = [1 + alpha / A, -2 * c, 1 - alpha / A]
    else:
        sq = 2 * math.sqrt(A) * alpha
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)]
            a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq]
        else:
            b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)]
            a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq]
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]


def eq_sos(p):
    return np.array([eq_section(k, p[f"f{i}"], p[f"g{i}"], p[f"q{i}"] / 10) for i, k in enumerate(EQ_KINDS)])


def eq_curve(p, freqs):
    """The EQ's response in dB at each frequency (for the graph)."""
    z = np.exp(-1j * 2 * np.pi * np.asarray(freqs) / SR)
    h = np.ones(len(freqs), complex)
    for b0, b1, b2, _, a1, a2 in eq_sos(p):
        h *= (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
    return 20 * np.log10(np.maximum(np.abs(h), 1e-9))


TAP = 4096                                           # analyzer window (85 ms)


class EQ:
    def __init__(self):
        self.sos = None
        self.zi = np.zeros((5, 2, 2))
        self.out = 1.0
        self.tap = np.zeros(TAP, np.float32)            # the last TAP samples of the output (mono)
        self.tap_w = 0

    def set(self, p, bpm=None):
        self.sos = eq_sos(p)
        self.out = 10 ** (p["out"] / 20)

    def process(self, x):
        if self.sos is None:
            return x
        from scipy.signal import sosfilt
        y, self.zi = sosfilt(self.sos, x, axis=-1, zi=self.zi)
        y = (y * self.out).astype(np.float32)
        tap_write(self, y)
        return y


def tap_write(obj, y):
    """Keep a mono copy of the last TAP output samples for the analyzer on screen."""
    m = y.mean(axis=0)[-TAP:]
    n, w = len(m), obj.tap_w
    first = min(n, TAP - w)
    obj.tap[w:w + first] = m[:first]
    obj.tap[:n - first] = m[first:]
    obj.tap_w = (w + n) % TAP


class LadderFilter:
    """pedalboard's ladder filter (the sound Luis approved), plus an analyzer tap."""

    MODES = {"LPF": "LPF24", "HPF": "HPF24", "BPF": "BPF24"}

    def __init__(self):
        import pedalboard as pb
        self.pb = pb
        self.f = pb.LadderFilter()
        self.tap = np.zeros(TAP, np.float32)
        self.tap_w = 0

    def set(self, p, bpm=None):
        self.f.mode = getattr(self.pb.LadderFilter.Mode, self.MODES[p["mode"]])
        self.f.cutoff_hz, self.f.resonance, self.f.drive = p["cutoff"], p["reso"] / 100, p["drive"]

    def process(self, x):
        n = x.shape[1]
        y = self.f(np.ascontiguousarray(x, dtype=np.float32), SR, reset=False)
        if y.shape[1] < n:
            y = np.pad(y, ((0, 0), (n - y.shape[1], 0)))
        y = np.ascontiguousarray(y[:, :n], dtype=np.float32)
        tap_write(self, y)
        return y


def filter_curve(p, freqs):
    """Approximate response of the 24 dB ladder for the graph (resonance = peak at cutoff)."""
    fc = p["cutoff"]
    q2 = 0.54 + (p["reso"] / 100) ** 1.5 * 9                # resonance shows as a peak
    z = np.exp(-1j * 2 * np.pi * np.asarray(freqs) / SR)
    h = np.ones(len(freqs), complex)
    kinds = {"LPF": ("lp", "lp"), "HPF": ("hp", "hp"), "BPF": ("hp", "lp")}[p["mode"]]
    for kind, q in zip(kinds, (0.54, q2) if p["mode"] != "BPF" else (q2 / 2 + 0.3, q2 / 2 + 0.3)):
        b, a = biquad(kind, fc, q)
        h *= (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    return 20 * np.log10(np.maximum(np.abs(h), 1e-9))


_win = np.hanning(TAP).astype(np.float32)


def spectrum(eq, freqs, prev=None):
    """Analyzer for the screen (main thread): level in dB at each graph frequency,
    smoothed so it falls back gently like a real analyzer."""
    x = np.roll(eq.tap, -eq.tap_w) * _win
    mag = np.abs(np.fft.rfft(x)) / (TAP / 4)
    bins = np.fft.rfftfreq(TAP, 1 / SR)
    db = 20 * np.log10(np.maximum(np.interp(freqs, bins, mag), 1e-7))
    db = np.convolve(db, np.ones(5) / 5, mode="same")   # a little smoothing across frequency
    db = db + 4.5 * np.log2(np.asarray(freqs) / 1000)   # +4.5 dB/octave tilt, like pro analyzers:
                                                        # music falls off up high, this levels it
    if prev is not None and len(prev) == len(db):
        db = np.maximum(db, prev - 2.5)                 # rises at once, falls ~2.5 dB per frame
    return db


# ---------------------------------------------------------------------------
# Compressor, in the spirit of Pro-C
# ---------------------------------------------------------------------------
class Compressor:
    """Feed-forward compressor with a soft knee.

    The level detector and gain run on 8-sample steps (0.17 ms), then the gain is
    smoothed out to every sample: fast enough for drum-bus attack, cheap in Python.
    ATTACK 0.1-100 ms and RELEASE 10 ms - 2.5 s cover drum bus to slow opto.
    AUTO (VCA-65 style): program-dependent - a fast and a slow release work together,
    so short peaks recover quickly while steady material is held smoothly.
    STYLE CHARACTER adds a little 2nd-harmonic colour that grows with gain reduction.
    Auto-gain makes up the level from THRESH and RATIO; a soft limiter at the end
    catches anything that would clip. DRY blends in the uncompressed signal.
    The last 3 seconds of input / output / gain reduction are kept for the screen.
    """

    STEP = 8
    HIST = 300                                          # blocks of history (~3 s at 512)

    def __init__(self):
        self.gr = 0.0                                   # current gain reduction, dB (<= 0)
        self.gr_slow = 0.0
        self.p = None
        self.hist = np.zeros((3, self.HIST), np.float32)   # in dB, out dB, GR dB
        self.hist_w = 0
        self.sc1, self.sc2 = Filter("hp", 20), Filter("hp", 20)   # side-chain HPF, 24 dB/oct

    def set(self, p, bpm=None):
        self.p = dict(p)

    def process(self, x):
        p = self.p
        if p is None:
            return x
        n, st = x.shape[1], self.STEP
        thr, ratio, knee = p["thresh"], p["ratio"], 6.0
        auto = p["release"] > 2500                      # RELEASE turned past 2.5 s = AUTO (SSL-style)
        att_ms = p["attack"] / 10
        rel_ms = 60.0 if auto else p["release"]
        dt = st / SR * 1000
        a_att = math.exp(-dt / max(att_ms, 0.05))
        a_rel = math.exp(-dt / max(rel_ms, 1.0))
        a_slow_att, a_slow_rel = math.exp(-dt / 200), math.exp(-dt / 1500)
        # level per 8-sample step (peak of both channels). SC HPF: the detector doesn't
        # hear the lows, so kicks and bass don't make the whole bus pump.
        det = x
        if p["schpf"] > 20:
            self.sc1.set(float(p["schpf"]))
            self.sc2.set(float(p["schpf"]))
            det = self.sc2(self.sc1(x))
        m = np.abs(det).max(axis=0)
        k = n // st
        lv = m[:k * st].reshape(k, st).max(axis=1)
        if n % st:
            lv = np.append(lv, m[k * st:].max())
        lv_db = 20 * np.log10(np.maximum(lv, 1e-6))
        over = lv_db - thr
        target = np.where(over <= -knee / 2, 0.0,
                          np.where(over >= knee / 2, -over * (1 - 1 / ratio),
                                   -(1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
        gr, gs = self.gr, self.gr_slow
        out_gr = np.empty(len(target))
        for i, t in enumerate(target.tolist()):         # the only per-step loop: ~64 steps
            a = a_att if t < gr else a_rel
            gr = t + (gr - t) * a
            if auto:                                    # slow companion holds steady material
                b = a_slow_att if gr < gs else a_slow_rel
                gs = gr + (gs - gr) * b
                out_gr[i] = min(gr, gs * 0.7)
            else:
                out_gr[i] = gr
        self.gr, self.gr_slow = gr, gs
        g_db = np.repeat(out_gr, st)[:n]
        g_db = np.convolve(np.concatenate([[g_db[0]] * (st - 1), g_db]), np.ones(st) / st, "valid")
        makeup = -thr * (1 - 1 / ratio) * 0.5          # auto-gain
        g = 10 ** ((g_db + makeup) / 20)
        y = x * g
        if p["style"] == "CHARACTER":                   # colour that grows with compression
            c = np.clip(-g_db / 12, 0, 1) * 0.25
            y = y + c * (y * np.abs(y))                 # 2nd harmonic
        dry = 1 - p["mix"] / 100                      # MIX: 100% = fully compressed
        y = y * (1 - dry) + x * dry
        a_ = np.abs(y)                                   # soft limiter above -1 dB
        lim = 0.89
        y = np.where(a_ > lim, np.sign(y) * (lim + 0.1 * np.tanh((a_ - lim) / 0.1)), y)
        y = y.astype(np.float32)
        h, w = self.hist, self.hist_w
        h[0, w] = 20 * math.log10(max(float(np.abs(x).max()), 1e-6))
        h[1, w] = 20 * math.log10(max(float(np.abs(y).max()), 1e-6))
        h[2, w] = float(out_gr.min())
        self.hist_w = (w + 1) % self.HIST
        return y

    def history(self):
        """(in dB, out dB, GR dB), oldest first."""
        return np.roll(self.hist, -self.hist_w, axis=1)


# ---------------------------------------------------------------------------
# Tape / Vinyl: one effect, MODE picks the machine
# ---------------------------------------------------------------------------
def shelf(kind, f, g, sr=SR):
    return eq_section(kind, f, g, 0.707, sr)


class ModLine:
    """A short delay line read at a moving position: pitch wobble (wow, flutter, warp)."""

    def __init__(self, base_ms=3.0):                    # 3 ms: just enough room for the wobble
        self.N = SR // 2
        self.buf = np.zeros((2, self.N), np.float32)
        self.w = 0
        self.base = base_ms / 1000 * SR

    def process(self, x, offs):
        """offs: extra delay per sample (samples), same length as the block."""
        n = x.shape[1]
        t = np.arange(n)
        idx = (self.w + t) % self.N
        self.buf[:, idx] = x
        pos = (self.w + t - self.base - offs) % self.N
        i0 = np.floor(pos).astype(np.int64)
        f = (pos - i0).astype(np.float32)
        y = self.buf[:, i0] * (1 - f) + self.buf[:, (i0 + 1) % self.N] * f
        self.w = (self.w + n) % self.N
        return y


class TapeVinyl:
    """TAPE (in the spirit of Tape MELLO-FI): DRIVE into a tape-style saturation
    (highs saturate first, through pre/de-emphasis), slow WOW and fast FLUTTER pitch
    wobble, hiss NOISE with a touch of motor hum, WEAR = duller highs plus the odd
    dropout, FILTER = one knob from low-pass (left) to high-pass (right).

    VINYL (in the spirit of RC-20 / iZotope Vinyl): ERA sets the bandwidth and mono
    of a record from 1930 to 2000, DUST = crackle, SCRATCH = pops and clicks (some at
    the record's turn), WARP = slow wobble at 33 1/3 rpm, WEAR = dull + rumble + grit,
    MIX blends the record with the clean sound."""

    ERAS = {1930: (250, 3500, 1.0), 1950: (90, 7000, 1.0), 1960: (60, 11000, 0.4),
            1970: (40, 15000, 0.15), 1980: (30, 17500, 0.05), 2000: (20, 19500, 0.0)}

    def __init__(self):
        self.p = None
        self.line = ModLine()
        self.t = 0                                      # samples since start (for LFOs)
        self.drift = 0.0
        self.pre = np.array([shelf("highshelf", 3000, 6)])
        self.post = np.array([shelf("highshelf", 3000, -6)])
        self.zi_pre, self.zi_post = np.zeros((1, 2, 2)), np.zeros((1, 2, 2))
        self.lp, self.hp = Filter("lp", 20000), Filter("hp", 20)
        self.wear_lp = Filter("lp", 20000)
        self.era_lp, self.era_hp = Filter("lp", 19500), Filter("hp", 20)
        self.noise_lp = Filter("lp", 6000)
        self.rumble_lp = Filter("lp", 60)
        self.click_bp = Filter("hp", 1500)
        self.dropout = 1.0
        self.last_click = 0

    def set(self, p, bpm=None):
        self.p = dict(p)

    def process(self, x):
        p = self.p
        if p is None:
            return x
        n = x.shape[1]
        tt = (self.t + np.arange(n)) / SR
        self.t += n
        return self.tape(x, p, tt) if p["mode"] == "TAPE" else self.vinyl(x, p, tt)

    def tape(self, x, p, tt):
        from scipy.signal import sosfilt
        n = x.shape[1]
        drive = p["drive"] / 100
        y = x
        if drive > 0:                                   # highs hit the tape first
            g = 1 + drive * 6
            y, self.zi_pre = sosfilt(self.pre, y, axis=-1, zi=self.zi_pre)
            y = np.tanh(y * g + 0.08 * drive) - math.tanh(0.08 * drive)
            y = y / g * (1 + drive * 0.5)                # roughly level as you drive it
            y, self.zi_post = sosfilt(self.post, y, axis=-1, zi=self.zi_post)
        # wow (slow, ~0.5 Hz, drifting) + flutter (fast, ~7 Hz, a little random)
        self.drift = 0.995 * self.drift + 0.005 * np.random.randn()
        wow = p["wow"] / 100 * 0.0018 * SR * (np.sin(2 * np.pi * 0.45 * tt) + 0.4 * self.drift)
        flut = p["flutter"] / 100 * 0.00012 * SR * (np.sin(2 * np.pi * 7.1 * tt)
                                                    + 0.5 * np.sin(2 * np.pi * 11.3 * tt))
        y = self.line.process(y.astype(np.float32), wow + flut)
        wear = p["wear"] / 100
        if wear > 0:
            self.wear_lp.set(20000 * (1 - wear) + 3500 * wear)
            y = self.wear_lp(y)
            if np.random.rand() < wear * 0.02:          # the odd dropout
                self.dropout = 0.35 + 0.4 * np.random.rand()
            self.dropout = min(1.0, self.dropout + 0.04)
            y = y * self.dropout
        noise = p["noise"] / 100
        if noise > 0:
            hiss = self.noise_lp(np.random.randn(2, n).astype(np.float32)) * 0.05 * noise
            hum = 0.004 * noise * np.sin(2 * np.pi * 50 * tt)
            y = y + hiss + hum
        f = p["filter"]                                 # -100 .. 0 .. +100
        self.lp.set(20000 if f >= 0 else 20000 * (500 / 20000) ** (-f / 100))
        self.hp.set(20 if f <= 0 else 20 * (2000 / 20) ** (f / 100))
        if f < 0:
            y = self.lp(y)
        elif f > 0:
            y = self.hp(y)
        return y.astype(np.float32)

    def vinyl(self, x, p, tt):
        n = x.shape[1]
        lo, hi, mono = self.ERAS[p["era"]]
        wear = p["vwear"] / 100
        self.era_hp.set(lo)
        self.era_lp.set(hi * (1 - 0.6 * wear))
        y = self.era_lp(self.era_hp(x))
        if mono:
            m = y.mean(axis=0)
            y = y * (1 - mono) + m * mono
        warp = p["warp"] / 100 * 0.0015 * SR * np.sin(2 * np.pi * 0.555 * tt)   # 33 1/3 rpm
        y = self.line.process(y.astype(np.float32), warp)
        if wear > 0:
            y = np.tanh(y * (1 + wear)) / (1 + wear * 0.5)
            y = y + self.rumble_lp(np.random.randn(2, n).astype(np.float32)) * 0.15 * wear
        clicks = np.zeros((2, n), np.float32)
        dust = p["dust"] / 100
        if dust > 0:                                    # crackle: lots of tiny ticks
            k = np.random.poisson(dust * 260 * n / SR)
            if k:
                at = np.random.randint(0, n, k)
                amp = np.random.exponential(0.02, k) * np.random.choice([-1.0, 1.0], k)
                clicks[0, at] += amp
                clicks[1, (at + np.random.randint(0, 3, k)) % n] += amp * 0.8
        scr = p["scratch"] / 100
        if scr > 0:                                     # pops: rarer, louder, some on the turn
            turn = int(SR / 0.555)
            k = np.random.poisson(scr * 1.5 * n / SR)
            for _ in range(k):
                at = np.random.randint(0, n)
                clicks[:, at:at + 3] += np.random.choice([-1, 1]) * (0.15 + 0.35 * np.random.rand())
            if scr > 0.3 and (self.t // turn) != ((self.t - n) // turn):
                at = (turn - (self.t - n) % turn) % n
                clicks[:, at:at + 4] += 0.25 * scr
        if dust > 0 or scr > 0:
            y = y + self.click_bp(clicks)
        mix = p["vmix"] / 100
        return (y * mix + x * (1 - mix)).astype(np.float32)


# ---------------------------------------------------------------------------
# Instruments. A synth voice renders its own audio block by block; the app mixes it
# like a sample voice (pad volume / pan / bus apply as usual).
# ---------------------------------------------------------------------------
def poly_blep(t, dt):
    """Smooths the jump of a saw/square so it doesn't alias (fold back as harsh tones)."""
    y = np.zeros_like(t)
    a = t < dt
    x = t[a] / dt[a] if np.ndim(dt) else t[a] / dt
    y[a] = x + x - x * x - 1
    b = t > 1 - dt
    x = (t[b] - 1) / (dt[b] if np.ndim(dt) else dt)
    y[b] = x * x + x + x + 1
    return y


def midi_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


class MonoBass:
    """In the spirit of a Minimoog Model D: two oscillators (the second detuned) plus a
    sub one octave down, into a 24 dB ladder filter with its own envelope (CONTOUR =
    how far it opens, DECAY = how fast it closes), EMPHASIS = resonance, GLIDE between
    notes, one note at a time. The amp holds while the key is held, then releases."""

    CHUNK = 64                                          # filter / envelopes step every 1.3 ms

    def __init__(self, p):
        import pedalboard as pb
        self.p = p
        self.lad = pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24)
        self.ph = np.zeros(3)
        self.f = None                                   # current frequency (glides)
        self.target = 110.0
        self.amp, self.fenv = 0.0, 0.0
        self.gate = False
        self.vel = 1.0
        self.done = False
        self.t_note = 0                                 # samples since the note started

    def note_on(self, n, vel):
        self.target = midi_hz(n)
        if self.f is None or self.p["glide"] == 0 or not self.gate:
            self.f = self.target if (self.f is None or self.p["glide"] == 0) else self.f
        if not self.gate or self.p.get("retrig", True):
            self.fenv = 1.0                             # filter contour restarts
        self.gate, self.vel, self.done, self.t_note = True, vel, False, 0

    def note_off(self):
        self.gate = False

    def osc(self, f, ph0, shape):
        dt = f / SR
        ph = (ph0 + np.cumsum(dt)) % 1.0
        if shape == "TRI":
            y = 4 * np.abs(ph - 0.5) - 1
        elif shape in ("SQUARE", "PULSE"):
            w = 0.5 if shape == "SQUARE" else 0.25
            y = np.where(ph < w, 1.0, -1.0)
            y = y + poly_blep(ph, dt) - poly_blep((ph + 1 - w) % 1.0, dt)
        else:                                            # SAW
            y = 2 * ph - 1 - poly_blep(ph, dt)
        return y, float(ph[-1])

    def render(self, n):
        p = self.p
        out = np.zeros(n, np.float32)
        glide_s = p["glide"] / 100 * 0.4                 # up to 0.4 s
        dec = 0.02 + (p["decay"] / 100) ** 2 * 2.5       # filter decay 20 ms .. 2.5 s
        rel = 0.01 + (p["release"] / 100) ** 2 * 1.5
        for i in range(0, n, self.CHUNK):
            m = min(self.CHUNK, n - i)
            k = m / SR
            # pitch (glide in log frequency)
            if glide_s > 0:
                a = math.exp(-k / glide_s)
                f0 = self.f
                self.f = self.target * (self.f / self.target) ** a
                f = np.geomspace(f0, self.f, m)
            else:
                self.f = self.target
                f = np.full(m, self.f)
            det = 2 ** (p["detune"] / 1200)
            o1, self.ph[0] = self.osc(f, self.ph[0], p["wave"])
            o2, self.ph[1] = self.osc(f * det, self.ph[1], p["wave"])
            o3, self.ph[2] = self.osc(f * 0.5, self.ph[2], "SQUARE")
            sig = o1 * 0.5 + o2 * (0.5 if p["detune"] else 0.0) + o3 * p["sub"] / 100 * 0.6
            # envelopes
            a_target = 1.0 if self.gate else 0.0
            tau = 0.003 if self.gate else rel
            a0 = self.amp
            self.amp = a_target + (self.amp - a_target) * math.exp(-k / tau)
            amp = np.linspace(a0, self.amp, m)
            fe_target = 0.15 if self.gate else 0.0       # contour falls to a low sustain
            self.fenv = fe_target + (self.fenv - fe_target) * math.exp(-k / dec)
            cut = p["cutoff"] * 2 ** (p["contour"] / 100 * 5 * self.fenv * (0.5 + 0.5 * self.vel))
            self.lad.cutoff_hz = float(min(18000.0, max(20.0, cut)))
            self.lad.resonance = p["emphasis"] / 100 * 0.95
            self.lad.drive = 1.0 + p["drive"] / 100 * 3
            y = self.lad(np.ascontiguousarray(np.stack([sig, sig]).astype(np.float32)), SR, reset=False)[0]
            if len(y) < m:
                y = np.pad(y, (m - len(y), 0))
            out[i:i + m] = y[:m] * amp * (0.4 + 0.6 * self.vel)
        self.t_note += n
        if not self.gate and self.amp < 1e-4:
            self.done = True
        return out


# ---------------------------------------------------------------------------
# KEYS: a ROMpler (recorded piano, strings, brass, organ: VSCO-2 Community Edition,
# public domain / CC0) plus a DX-style FM engine (electric piano, bells). One KeysNote
# per note played; the app keeps up to 4 per pad.
# ---------------------------------------------------------------------------
KEYS_PROGRAMS = ["PIANO", "STRINGS", "BRASS", "ORGAN", "E.PIANO", "BELL"]
SAMPLED = {"PIANO": False, "STRINGS": True, "BRASS": True, "ORGAN": True}   # True = loops while held
_banks = {}
KEYS_DIR = "keys"                                  # the app sets <app>/keys


def keys_bank(prog):
    """[(root note, stereo audio, loop or None)] for a sampled program, loaded once (call
    it from the main thread: reading files in the audio thread would drop out). Sustained
    programs get a crossfaded loop (1.5 s to the end) so a held note never runs out."""
    if prog not in SAMPLED:
        return []
    if prog in _banks:
        return _banks[prog]
    import os
    import soundfile as sf
    folder = os.path.join(KEYS_DIR, prog)
    bank = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if not name.endswith(".wav") or not name[:-4].isdigit():
            continue
        a, _ = sf.read(os.path.join(folder, name), dtype="float32", always_2d=True)
        if a.shape[1] == 1:
            a = np.repeat(a, 2, axis=1)
        a, loop = np.ascontiguousarray(a[:, :2]), None
        l0, xf = int(SR * 1.5), int(SR * 0.3)
        if SAMPLED[prog] and len(a) > l0 + xf + SR // 2:
            l1 = len(a) - int(SR * 0.1)                 # skip the fade-out at the very end
            fade = np.linspace(0, 1, xf, dtype=np.float32)[:, None]
            a = a[:l1].copy()
            a[l1 - xf:] = a[l1 - xf:] * (1 - fade) + a[l0 - xf:l0] * fade   # the end flows into l0
            loop = (l0, l1)
        bank.append((int(name[:-4]), a, loop))
    bank.sort(key=lambda b: b[0])
    _banks[prog] = bank
    return bank


class KeysNote:
    """One note (or chord) of the KEYS synth, stereo. Same interface as MonoBass:
    note_on / note_off / render / done; kill() = a fast release (voice stealing).
    chord = semitone offsets set by the app before note_on (CHORD knob); all the
    chord's notes share one envelope and tone filter, so a chord costs one voice."""
    stereo = True

    def __init__(self, p):
        self.p = p
        self.prog = p.get("program", "PIANO")
        self.done, self.gate, self.fast = False, False, False
        self.env, self.t, self.vel = 0.0, 0, 1.0
        self.tone = Filter("lp", 20000)
        self.chord = (0,)
        self.parts = []

    def note_on(self, n, vel):
        if self.fast:                                    # stolen before it even started
            self.done = True
            return
        sens = self.p.get("velo", 50) / 100
        self.vel = (1 - sens) + sens * vel
        self.gate, self.t, self.done = True, 0, False
        notes = [n + c for c in self.chord]
        self.level = 1 / math.sqrt(len(notes))
        if self.prog in SAMPLED:
            bank = _banks.get(self.prog)                 # preloaded by the app
            if not bank:
                self.done = True
                return
            self.parts = []
            for m in notes:
                root, audio, loop = min(bank, key=lambda b: abs(b[0] - m))
                self.parts.append([audio, loop, 2 ** ((m - root) / 12), 0.0])
        else:
            self.parts = [midi_hz(m) for m in notes]

    def note_off(self):
        self.gate = False

    def kill(self):
        self.gate, self.fast = False, True

    @staticmethod
    def sample_part(part, m):
        a, loop, rate, p0 = part
        N = len(a)
        pos = p0 + rate * np.arange(m)
        if loop:
            l0, l1 = loop
            pos = np.where(pos >= l1, l0 + (pos - l0) % (l1 - l0), pos)
            nxt = p0 + rate * m
            part[3] = nxt if nxt < l1 else l0 + (nxt - l0) % (l1 - l0)
        else:
            if p0 >= N - 3:
                return None
            part[3] = p0 + rate * m
        i = np.floor(pos).astype(np.int64)
        f = (pos - i)[:, None].astype(np.float32)
        g = lambda k: a[np.clip(i + k, 0, N - 1)]
        q0, q1, q2, q3 = g(-1), g(0), g(1), g(2)          # cubic (Hermite) interpolation
        c1 = 0.5 * (q2 - q0)
        c2 = q0 - 2.5 * q1 + 2 * q2 - 0.5 * q3
        c3 = 0.5 * (q3 - q0) + 1.5 * (q1 - q2)
        y = ((c3 * f + c2) * f + c1) * f + q1
        if not loop:
            y[i >= N - 2] = 0
        return y

    def sample_block(self, m):
        y = np.zeros((m, 2), np.float32)
        alive = False
        for part in self.parts:
            z = self.sample_part(part, m)
            if z is not None:
                y += z
                alive = True
        if not alive:
            self.done = True
        return y

    def fm_one(self, f, m):
        v = self.vel
        t = (self.t + np.arange(m)) / SR
        w = 2 * np.pi * f * t
        if self.prog == "E.PIANO":                        # DX7-style: body pair + tine pair
            idx = (0.3 + 1.4 * v) * np.exp(-t * 2.5)
            body = np.sin(w + idx * np.sin(w)) * np.exp(-t * (0.5 + f / 1500))
            tine = np.sin(w + 0.9 * v * np.exp(-t * 20) * np.sin(14 * w)) * 0.3 * np.exp(-t * 5)
            y = (body + tine) * 0.45
            spread = 0.15 * np.sin(2 * np.pi * 4.5 * t)    # a touch of tremolo pan, like a Rhodes
            return y * (1 + spread), y * (1 - spread)

        def bell(w):                                      # DX-style tubular bell, 3 layers
            idx = (1.5 + 2.5 * v) * np.exp(-t * 0.7)
            body = np.sin(w + idx * np.sin(3.5 * w)) * np.exp(-t * 0.35)          # main ring
            hum = np.sin(0.5 * w + 0.6 * np.sin(w)) * 0.45 * np.exp(-t * 0.25)   # low hum
            strike = np.sin(2.76 * w + (1.0 + 2 * v) * np.exp(-t * 8) * np.sin(5.4 * w)) \
                * (0.35 + 0.3 * v) * np.exp(-t * 3.0)                             # the hit
            return (body + hum + strike) * 0.5
        return bell(w), bell(w * 1.0025)                  # slightly detuned right side: wide

    def fm_block(self, m):
        l = np.zeros(m)
        r = np.zeros(m)
        for f in self.parts:
            a, b = self.fm_one(f, m)
            l += a
            r += b
        return np.stack([l, r], axis=1).astype(np.float32)

    def render(self, n):
        if self.done:
            return np.zeros((n, 2), np.float32)
        p = self.p
        y = self.sample_block(n) if self.prog in SAMPLED else self.fm_block(n)
        if self.done:
            return np.zeros((n, 2), np.float32)
        att = 0.002 + (p.get("attack", 0) / 100) ** 2 * 2.0
        rel = 0.008 if self.fast else 0.03 + (p.get("release", 30) / 100) ** 2 * 4.0
        k = n / SR
        e0 = self.env
        if self.gate:
            self.env = min(1.0, self.env + k / att)
        else:
            self.env *= math.exp(-k / rel)
        env = np.linspace(e0, self.env, n, dtype=np.float32)[:, None]
        # TONE: a gentle low-pass; harder playing opens it a little
        cut = 250 * (19000 / 250) ** min(1.0, p.get("tone", 100) / 100 + 0.1 * self.vel)
        self.tone.set(float(min(19000.0, cut)))
        y = self.tone(y.T).T * env * (self.vel * self.level)
        self.t += n
        if (not self.gate and self.env < 1e-3) or (self.prog not in SAMPLED and self.t > SR * 15):
            self.done = True
        return y.astype(np.float32)


class JunoChorus:
    """In the spirit of the Juno-60 chorus: one BBD-style delay line per side, swept by
    a triangle LFO in opposite directions left / right, mixed 50/50 with the dry sound.
    I = slow and gentle, II = deeper, I+II = fast and shallow (a vibrato-like shimmer)."""

    MODES = {"I": (0.513, 1.6), "II": (0.863, 2.4), "I+II": (9.75, 0.35)}   # Hz, depth ms

    def __init__(self):
        self.line = ModLine(base_ms=3.5)
        self.ph = 0.0
        self.tone = Filter("lp", 9000)                   # BBDs are a bit dark

    def process(self, x, mode):
        """x: (2, n)."""
        if mode not in self.MODES:
            return x
        rate, depth = self.MODES[mode]
        n = x.shape[1]
        ph = (self.ph + rate / SR * np.arange(n)) % 1.0
        self.ph = float((self.ph + rate / SR * n) % 1.0)
        tri = 4 * np.abs(ph - 0.5) - 1                   # -1 .. 1
        d = depth / 1000 * SR
        L = self.line
        t = np.arange(n)
        L.buf[:, (L.w + t) % L.N] = x
        out = []
        for c, offs in ((0, d * (1 + tri) / 2), (1, d * (1 - tri) / 2)):   # opposite sweeps
            pos = (L.w + t - L.base - offs) % L.N
            i0 = np.floor(pos).astype(np.int64)
            f = (pos - i0).astype(np.float32)
            out.append(L.buf[c, i0] * (1 - f) + L.buf[c, (i0 + 1) % L.N] * f)
        L.w = (L.w + n) % L.N
        wet = self.tone(np.stack(out))
        return ((x + wet) * 0.7).astype(np.float32)
