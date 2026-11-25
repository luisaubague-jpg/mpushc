"""
Show the theme mockups on the Push 2 screen, so you can judge them on the
real display. A new one every 5 seconds. Ctrl+C to quit.
Run from ~/Projects/mpushc (it uses push2.py from there).
"""
import os, sys, time
import numpy as np
from PIL import Image
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
import push2

NAMES = ["A ARCADE", "B FIGHTER", "C EARLY DAW", "D HANDHELD", "E SYNTHWAVE",
         "F 90s RAP MAGAZINE", "G VINTAGE BUTCHER"]
frames = [np.asarray(Image.open(os.path.join(here, f"theme_{i}.png")).convert("RGB"), dtype=np.uint8)
          for i in range(len(NAMES))]
display = push2.PushDisplay()
try:
    i, shown = 0, 0
    while True:
        if time.monotonic() - shown > 5:
            i = (i + 1) % len(frames) if shown else 0
            shown = time.monotonic()
            print("Showing", NAMES[i])
        display.show(frames[i])        # the screen needs a fresh frame regularly
        time.sleep(1 / 15)
except KeyboardInterrupt:
    pass
finally:
    os._exit(0)
