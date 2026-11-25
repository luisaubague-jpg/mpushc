"""
Show load_screen.png on the Push 2 screen. Ctrl+C to quit.
(The Push blanks its screen if frames stop coming, so we keep sending it.)
"""
import os
import sys
import time
import numpy as np
from PIL import Image
import push2

path = sys.argv[1] if len(sys.argv) > 1 else "load_screen.png"
img = Image.open(path).convert("RGB")
if img.size != (push2.WIDTH, push2.HEIGHT):
    img = img.resize((push2.WIDTH, push2.HEIGHT), Image.LANCZOS)
frame = push2.encode_frame(np.asarray(img, dtype=np.uint8))   # encode once

display = push2.PushDisplay()
print(f"Showing {path} on the Push. Ctrl+C to quit.")
try:
    while True:
        display.send(frame)
        time.sleep(1 / 30)
except KeyboardInterrupt:
    pass
finally:
    print("\nBye.")
    os._exit(0)
