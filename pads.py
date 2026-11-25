"""
Test 1 - pads only (MIDI).

Hit a pad: it lights up blue / green / red depending on how hard you hit,
and the terminal prints the note, position and velocity.
Turn a knob or press a button: the terminal prints it too.
Ctrl+C to quit (pads are cleared on exit).
"""

import time
import push2

midi = None


def color_for(velocity):
    if velocity < 50:
        return push2.BLUE
    if velocity < 100:
        return push2.GREEN
    return push2.RED


def on_message(msg):
    if msg.type == "note_on" and push2.is_pad(msg.note):
        if msg.velocity > 0:
            row, col = push2.pad_position(msg.note)
            print(f"PAD   note {msg.note:3d}  row {row} col {col}  velocity {msg.velocity:3d}")
            midi.pad_color(msg.note, color_for(msg.velocity))
        else:
            midi.pad_color(msg.note, push2.OFF)
    elif msg.type == "note_off" and push2.is_pad(msg.note):
        midi.pad_color(msg.note, push2.OFF)
    elif msg.type == "control_change":
        print(f"CC    {msg.control:3d}  value {msg.value:3d}")
    elif msg.type in ("polytouch", "aftertouch"):
        pass  # pressure messages: ignored for now, very chatty


def main():
    global midi
    midi = push2.PushMidi(on_message)
    midi.clear_pads()
    print("\nHit some pads. Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        midi.clear_pads()
        midi.close()
        print("\nBye.")


if __name__ == "__main__":
    main()
