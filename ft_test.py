"""Does free-threaded Python stop the audio from waiting on the screen?

Plays 10 s of noise through MPushC-like effects (our saturation + BBD delay, plus
reverb, compressor and limiter from pedalboard) on the AudioFuse, 512-sample blocks,
while the main thread draws a busy fake screen non-stop - like MPushC does.

    .venv-ft/bin/python ft_test.py                 # free-threaded (no lock)
    PYTHON_GIL=1 .venv-ft/bin/python ft_test.py    # same, with the old lock back on

Turn the AudioFuse down: it plays quiet noise.
"""
import os
import sys
import time

import numpy as np
import pedalboard as pb
import sounddevice as sd
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native_fx as nf

SR, BLOCK, SECONDS = 48000, 512, 10
gil = getattr(sys, "_is_gil_enabled", lambda: True)()
print(f"Python {sys.version.split()[0]}  -  lock (GIL) {'ON' if gil else 'OFF'}")

sat = nf.Saturation()
sat.set({"amount": 50, "stype": "KEEP HIGH", "mix": 100, "out": 0})
dly = nf.BBDDelay()
dly.set({"time": "1/8", "fb": 50, "mix": 30, "mode": "PING PONG", "wobble": 30, "quality": 30,
         "freeze": "OFF"}, 90)
plugs = [pb.Pedalboard([pb.Reverb()]), pb.Pedalboard([pb.Compressor(threshold_db=-20, ratio=3)]),
         pb.Pedalboard([pb.Limiter()])]
st = {"times": [], "under": 0}


def cb(out, frames, t_info, status):
    t = time.perf_counter()
    if status.output_underflow:
        st["under"] += 1
    x = (np.random.randn(2, frames) * 0.05).astype(np.float32)
    for _ in range(40):                                    # stand-in for mixing ~10 voices
        x = x * 0.9999
    y = dly.process(sat.process(x))
    for p in plugs:
        y = p(np.ascontiguousarray(y), SR, reset=False)
    out[:] = (y.T * 0.5)[:frames]
    st["times"].append((time.perf_counter() - t) * 1000)


try:
    stream = sd.OutputStream(samplerate=SR, channels=2, device="AudioFuse", blocksize=BLOCK,
                             dtype="float32", callback=cb)
except Exception:
    stream = sd.OutputStream(samplerate=SR, channels=2, blocksize=BLOCK, dtype="float32", callback=cb)
stream.start()
end, frames_drawn = time.time() + SECONDS, 0
while time.time() < end:                                   # a busy screen, never resting
    img = Image.new("RGB", (960, 160))
    d = ImageDraw.Draw(img)
    for k in range(400):
        d.line([(k * 2, 0), (k * 2, 160)], fill=(k % 255, 80, 120))
        d.text((k % 900, 20 + k % 120), str(k), fill=(255, 255, 255))
    np.asarray(img).tobytes()
    frames_drawn += 1
stream.stop()
stream.close()
t = np.array(st["times"][20:])
budget = BLOCK / SR * 1000
print(f"screen frames drawn: {frames_drawn}")
print(f"audio block: avg {t.mean():.2f} ms, worst {t.max():.2f} ms (budget {budget:.1f} ms), "
      f"over budget: {(t > budget).sum()} of {len(t)}, dropouts reported: {st['under']}")
