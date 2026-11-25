"""
LED colour test: the Push's LEDs don't show colours the way the screen does,
so we pick the button colours by eye.

The 8 buttons under the knobs light up with 4 candidate reds (buttons 1-4)
and 4 candidate blues (buttons 5-8). Pick the red and the blue that best
match the screen. Ctrl+C to quit.
"""
import os
import time
import mido

CANDIDATES = [
    ("R1", (255, 0, 0)), ("R2", (255, 14, 0)), ("R3", (255, 30, 12)), ("R4", (220, 0, 24)),
    ("B1", (0, 40, 255)), ("B2", (0, 85, 255)), ("B3", (20, 120, 255)), ("B4", (0, 0, 255)),
]
FIRST_SLOT = 20                        # palette entries 20-27, unused by MPushC

out_names = [n for n in mido.get_output_names() if "Push 2" in n]
outs = [mido.open_output(n) for n in out_names]


def send(msg):
    for o in outs:
        o.send(msg)


for i, (_, (r, g, b)) in enumerate(CANDIDATES):
    send(mido.Message("sysex", data=[0x00, 0x21, 0x1D, 0x01, 0x01, 0x03, FIRST_SLOT + i,
                                     r & 127, r >> 7, g & 127, g >> 7, b & 127, b >> 7, 0, 0]))
send(mido.Message("sysex", data=[0x00, 0x21, 0x1D, 0x01, 0x01, 0x05]))
for i in range(8):
    send(mido.Message("control_change", control=102 + i, value=FIRST_SLOT + i))

print("Buttons under the knobs, left to right:")
print("  " + "   ".join(name for name, _ in CANDIDATES))
print("Which red and which blue match the screen best? Ctrl+C to quit.")
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    for i in range(8):
        send(mido.Message("control_change", control=102 + i, value=0))
    os._exit(0)
