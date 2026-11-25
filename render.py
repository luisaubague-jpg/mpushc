"""
Draws the v0.0 test screen: title, a waveform, and the last pad hit.
Uses Pillow to draw, numpy to hand pixels to the display encoder.
"""

import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import push2

W, H = push2.WIDTH, push2.HEIGHT

BG = (12, 12, 16)
INK = (235, 235, 235)
QUIET = (120, 120, 130)
WAVE = (70, 160, 255)
PLAYED = (255, 120, 60)


def load_font(size):
    try:
        return ImageFont.load_default(size=size)          # Pillow 10.1+
    except TypeError:
        for path in ("/System/Library/Fonts/Helvetica.ttc",
                     "/System/Library/Fonts/SFNS.ttf"):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def test_waveform(n=640):
    """A fake drum loop: 8 hits, each a noise burst with a fast decay."""
    rng = np.random.default_rng(7)
    x = np.zeros(n)
    hit_len = n // 8
    for i in range(8):
        level = 1.0 if i % 2 == 0 else 0.55
        decay = np.exp(-np.linspace(0, 6, hit_len))
        x[i * hit_len:(i + 1) * hit_len] = level * decay * rng.uniform(-1, 1, hit_len)
    return x


class Renderer:
    def __init__(self):
        self.big = load_font(34)
        self.small = load_font(16)
        self.wave = test_waveform()
        self.last_pad = None            # (note, row, col, velocity)
        self.last_hit_time = 0.0

    def pad_hit(self, note, velocity):
        row, col = push2.pad_position(note)
        self.last_pad = (note, row, col, velocity)
        self.last_hit_time = time.monotonic()

    def frame(self):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)

        # Left: title
        d.text((20, 18), "MPushC", font=self.big, fill=INK)
        d.text((22, 64), "v0.0 hardware test", font=self.small, fill=QUIET)
        d.text((22, 128), time.strftime("%H:%M:%S"), font=self.small, fill=QUIET)

        # Middle: waveform with a playhead that sweeps after a pad hit
        x0, mid, half = 220, 80, 60
        n = len(self.wave)
        since = time.monotonic() - self.last_hit_time
        playhead = int(min(since / 1.0, 1.0) * n) if self.last_pad else -1
        for i in range(n):
            h = int(abs(self.wave[i]) * half)
            color = PLAYED if 0 <= i < playhead and since < 1.0 else WAVE
            d.line([(x0 + i, mid - h), (x0 + i, mid + h)], fill=color)

        # Right: last pad hit
        if self.last_pad:
            note, row, col, vel = self.last_pad
            d.text((880, 20), "PAD", font=self.small, fill=QUIET, anchor="ma")
            d.text((880, 44), f"{row + 1}·{col + 1}", font=self.big, fill=INK, anchor="ma")
            d.text((880, 92), f"note {note}", font=self.small, fill=QUIET, anchor="ma")
            bar = int(vel / 127 * 80)
            d.rectangle([840, 124, 840 + bar, 136], fill=PLAYED)
            d.rectangle([840, 124, 920, 136], outline=QUIET)
        else:
            d.text((880, 70), "hit a pad", font=self.small, fill=QUIET, anchor="ma")

        return np.asarray(img, dtype=np.uint8)
