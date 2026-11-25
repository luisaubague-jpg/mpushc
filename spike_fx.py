"""MPushC v0.3 feasibility test: can the Python engine run effects (and a synth)
in real time on this Mac?

    python spike_fx.py            # offline numbers + live test through the AudioFuse
    python spike_fx.py --offline  # offline numbers only (no sound)

Needs: python -m pip install pedalboard
Plays a loop from projects/01/samples (or the WAV you pass) through each setup for
a few seconds. Turn the AudioFuse down a bit first. Nothing in your project changes.
"""
import glob
import os
import subprocess
import sys
import threading
import time

import numpy as np

SR, BLOCK = 48000, 128
BUDGET_MS = BLOCK / SR * 1000            # 2.67 ms: the callback must finish well inside this
SECONDS = 8                              # live test length per setup

try:
    import pedalboard as pb
except ImportError:
    sys.exit("pedalboard missing: python -m pip install pedalboard")


# ---------------------------------------------------------------------------
# Setups to test
# ---------------------------------------------------------------------------
def bus():
    """One full bus as planned: saturation, bitcrusher, filter, EQ, comp, reverb, delay."""
    return pb.Pedalboard([
        pb.Distortion(drive_db=6), pb.Bitcrush(bit_depth=12), pb.LadderFilter(cutoff_hz=4000),
        pb.PeakFilter(cutoff_frequency_hz=1000, gain_db=2), pb.Compressor(threshold_db=-18, ratio=3),
        pb.Reverb(room_size=0.4, wet_level=0.2), pb.Delay(delay_seconds=0.25, mix=0.2)])


def master():
    return pb.Pedalboard([pb.Compressor(threshold_db=-10, ratio=2), pb.Limiter(threshold_db=-1)])


class Setup:
    def __init__(self, name, buses=(), pads=0, master_fx=None, synth=None):
        self.name, self.buses, self.master, self.synth = name, list(buses), master_fx, synth
        self.pad_fx = [pb.Pedalboard([pb.LadderFilter(cutoff_hz=3000)]) for _ in range(pads)]

    def process(self, x, note_on=False):
        """x: (2, BLOCK). Pads -> 4 buses (split evenly) -> master, like the real engine will."""
        out = np.zeros_like(x)
        if self.pad_fx:
            parts = [fx(x * (1 / len(self.pad_fx)), SR, reset=False) for fx in self.pad_fx]
            x = np.sum(parts, axis=0)
        if self.buses:
            for b in self.buses:
                out += b(x * (1 / len(self.buses)), SR, reset=False)
        else:
            out = x
        if self.synth is not None:
            msgs = [([0x90, 48, 100], 0.0)] if note_on else []
            out = out + self.synth(msgs, duration=BLOCK / SR, sample_rate=SR,
                                   num_channels=2, buffer_size=BLOCK, reset=False)[:, :BLOCK] * 0.3
        if self.master is not None:
            out = self.master(out, SR, reset=False)
        return out


def find_plugins():
    """Third-party plugins installed on this Mac. VST3 preferred over AU when both exist."""
    roots = ["/Library/Audio/Plug-Ins", os.path.expanduser("~/Library/Audio/Plug-Ins")]
    vst3, au = {}, {}
    for r in roots:
        for f in glob.glob(os.path.join(r, "VST3", "*.vst3")):
            vst3[os.path.splitext(os.path.basename(f))[0].lower()] = f
        for f in glob.glob(os.path.join(r, "Components", "*.component")):
            au[os.path.splitext(os.path.basename(f))[0].lower()] = f
    return [vst3.get(k) or au[k] for k in sorted(set(vst3) | set(au))]


WANTED_FX = ("airwindows", "chow", "slplugins", "fig")
WANTED_SYNTH = ("surge xt", "ca-72", "ca72", "dexed", "vital")


def probe(path):
    """Load + run a plugin in a separate Python first: a plugin that crashes takes
    only that test process down. Returns 'fx', 'synth' or None."""
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--probe", path],
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"   {os.path.basename(path)}: hung while loading, skipped")
        return None
    kind = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
    if kind in ("fx", "synth"):
        return kind
    print(f"   {os.path.basename(path)}: crashed or failed to load headless, skipped")
    return None


def probe_main(path):
    p = pb.load_plugin(path)
    x = np.zeros((2, BLOCK), np.float32)
    if p.is_instrument:
        for _ in range(20):
            p([], duration=BLOCK / SR, sample_rate=SR, num_channels=2, buffer_size=BLOCK, reset=False)
        print("synth")
    else:
        for _ in range(20):
            p(x, SR, reset=False)
        print("fx")


# ---------------------------------------------------------------------------
def load_loop():
    import soundfile as sf
    path = sys.argv[-1] if sys.argv[-1].lower().endswith(".wav") else None
    if path is None:
        wavs = sorted(glob.glob("projects/*/samples/*.wav"))
        path = max(wavs, key=os.path.getsize) if wavs else None
    if path is None:
        print("no WAV found: using noise")
        return (np.random.randn(SR * 4, 2) * 0.1).astype(np.float32)
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    print("test loop:", os.path.basename(path))
    return a[:, :2]


def offline(setup, loop):
    """Time per block with nothing else running."""
    n = 2000
    times = []
    for i in range(n):
        x = np.ascontiguousarray(loop[(i * BLOCK) % (len(loop) - BLOCK):][:BLOCK].T)
        t = time.perf_counter()
        setup.process(x, note_on=(i % 400 == 0))
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times[50:])
    return times.mean(), np.percentile(times, 99.9)


def live(setup, loop):
    """Real stream through the AudioFuse, with a busy main thread (like the screen)."""
    import sounddevice as sd
    st = {"pos": 0, "blocks": 0, "under": 0, "times": [], "i": 0}

    def cb(out, frames, t_info, status):
        t = time.perf_counter()
        if status.output_underflow:
            st["under"] += 1
        p = st["pos"]
        if p + frames > len(loop):
            p = 0
        x = np.ascontiguousarray(loop[p:p + frames].T)
        st["pos"] = p + frames
        y = setup.process(x, note_on=(st["i"] % 400 == 0))
        st["i"] += 1
        out[:] = np.clip(y.T, -1, 1) * 0.5
        st["times"].append((time.perf_counter() - t) * 1000)

    try:
        stream = sd.OutputStream(samplerate=SR, channels=2, device="AudioFuse", blocksize=BLOCK,
                                 dtype="float32", callback=cb)
    except Exception:
        stream = sd.OutputStream(samplerate=SR, channels=2, blocksize=BLOCK, dtype="float32", callback=cb)
    from PIL import Image, ImageDraw                 # stand-in for drawing the Push screen
    stream.start()
    end = time.time() + SECONDS
    while time.time() < end:
        img = Image.new("RGB", (960, 160))
        d = ImageDraw.Draw(img)
        for k in range(200):
            d.line([(k * 4, 0), (k * 4, 160)], fill=(k, 80, 120))
        np.asarray(img).tobytes()
        time.sleep(1 / 30)
    stream.stop()
    stream.close()
    t = np.array(st["times"][100:])
    return t.mean(), np.percentile(t, 99.9), st["under"]


def main():
    if "--probe" in sys.argv:
        return probe_main(sys.argv[sys.argv.index("--probe") + 1])
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    loop = load_loop()
    plugins = find_plugins()
    extra = [p for p in plugins if any(w in os.path.basename(p).lower() for w in WANTED_FX)]
    synths = [p for p in plugins if any(w in os.path.basename(p).lower() for w in WANTED_SYNTH)
              and "effect" not in os.path.basename(p).lower()]
    print(f"\n{len(plugins)} plugins installed. Testing:", ", ".join(os.path.basename(p) for p in extra + synths)
          or "none of the listed ones")
    print("Checking each one loads safely (separate process)...", flush=True)
    extra = [p for p in extra if probe(p) == "fx"][:3]
    synths = [p for p in synths if probe(p) == "synth"][:2]

    setups = [
        Setup("dry (no effects)"),
        Setup("1 bus (7 effects)", buses=[bus()]),
        Setup("4 buses + master", buses=[bus() for _ in range(4)], master_fx=master()),
        Setup("4 buses + master + 16 pad filters", buses=[bus() for _ in range(4)], pads=16,
              master_fx=master()),
    ]
    for p in extra:
        setups.append(Setup(f"4 buses + master + {os.path.basename(p)[:28]}",
                            buses=[bus() for _ in range(3)] + [pb.Pedalboard([pb.load_plugin(p)])],
                            master_fx=master()))
    for p in synths:
        setups.append(Setup(f"full mix + synth {os.path.basename(p)[:24]}",
                            buses=[bus() for _ in range(4)], master_fx=master(), synth=pb.load_plugin(p)))

    print(f"\nBudget per block: {BUDGET_MS:.2f} ms  (128 samples @ 48 kHz). Safe = under ~50%.\n")
    print(f"{'setup':<48}{'avg':>8}{'worst':>9}{'avg %':>8}")
    rows = []
    for s in setups:
        avg, worst = offline(s, loop)
        rows.append((s, avg, worst))
        print(f"{s.name:<48}{avg:7.3f}ms{worst:7.2f}ms{avg / BUDGET_MS * 100:7.0f}%", flush=True)

    if "--offline" in sys.argv:
        return
    print(f"\nLive test: {SECONDS} s per setup through the AudioFuse (listen for clicks).\n")
    print(f"{'setup':<48}{'avg':>8}{'worst':>9}{'dropouts':>10}")
    for s, *_ in rows:
        print(f"{s.name:<48}", end="", flush=True)
        avg, worst, under = live(s, loop)
        print(f"{avg:7.3f}ms{worst:7.2f}ms{under:10d}")
    print("\nDone. Copy everything above and paste it to Claude.")


if __name__ == "__main__":
    main()
