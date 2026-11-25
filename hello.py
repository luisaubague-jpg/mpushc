"""
Test 3 - everything together.

Hit a pad: it lights up (blue / green / red by velocity), and the Push
screen shows which pad and how hard, with a playhead sweeping the waveform.
Ctrl+C to quit.
"""

import time
import push2
from render import Renderer
from pads import color_for

midi = None
renderer = Renderer()


def on_message(msg):
    if msg.type == "note_on" and push2.is_pad(msg.note):
        if msg.velocity > 0:
            midi.pad_color(msg.note, color_for(msg.velocity))
            renderer.pad_hit(msg.note, msg.velocity)
        else:
            midi.pad_color(msg.note, push2.OFF)
    elif msg.type == "note_off" and push2.is_pad(msg.note):
        midi.pad_color(msg.note, push2.OFF)


def main():
    global midi
    display = push2.PushDisplay()
    midi = push2.PushMidi(on_message)
    midi.clear_pads()
    print("\nMPushC v0.0 running. Hit some pads. Ctrl+C to quit.\n")
    try:
        while True:
            display.show(renderer.frame())
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        midi.clear_pads()
        midi.close()
        display.close()
        print("\nBye.")


if __name__ == "__main__":
    main()
