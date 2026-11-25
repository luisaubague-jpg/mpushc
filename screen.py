"""
Test 2 - display only (USB).

Draws "MPushC" and a scrolling test waveform on the Push 2 screen.
Ctrl+C to quit.
"""

import time
import push2
from render import Renderer


def main():
    display = push2.PushDisplay()
    renderer = Renderer()
    print("Drawing on the Push screen. Ctrl+C to quit.")
    frames, t0 = 0, time.monotonic()
    try:
        while True:
            display.show(renderer.frame())
            frames += 1
            if frames % 60 == 0:
                fps = frames / (time.monotonic() - t0)
                print(f"{fps:5.1f} frames/s")
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        display.close()
        print("\nBye.")


if __name__ == "__main__":
    main()
