"""
Prints every MIDI message the Push sends (buttons, knobs, pads). Ctrl+C to quit.
Run it with MPushC closed, press the button you want to check, read the line.
"""
import os
import time
import mido

ports = [n for n in mido.get_input_names() if "Push 2" in n]
print("Listening on:", ports)


def show(name):
    def cb(msg):
        if msg.type != "clock":
            print(f"{name:32s} {msg}")
    return cb


inputs = [mido.open_input(n, callback=show(n)) for n in ports]
try:
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    os._exit(0)
