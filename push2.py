"""
MPushC v0.0 - Push 2 hardware layer.

Two independent parts, matching how the Push 2 works:
  * Pads, buttons, knobs and LEDs  -> plain MIDI (mido + python-rtmidi)
  * The 960x160 color display      -> raw USB bulk transfers (pyusb + libusb)

Protocol reference: Ableton's official Push 2 docs
https://github.com/Ableton/push-interface
"""

import os
import time

# --------------------------------------------------------------------------
# MIDI: pads, buttons, knobs, LEDs
# --------------------------------------------------------------------------

PAD_FIRST = 36           # bottom-left pad
PAD_LAST = 99            # top-right pad

# Colors from the Push 2 default LED palette (velocity = palette index)
OFF, WHITE, LIGHT_GRAY, DARK_GRAY, BLUE, GREEN, RED = 0, 122, 123, 124, 125, 126, 127


def pad_note(row, col):
    """row 0 = bottom, col 0 = left -> MIDI note number."""
    return PAD_FIRST + row * 8 + col


def pad_position(note):
    """MIDI note -> (row, col), row 0 = bottom."""
    i = note - PAD_FIRST
    return i // 8, i % 8


def is_pad(note):
    return PAD_FIRST <= note <= PAD_LAST


class PushMidi:
    """Opens every Push 2 MIDI port (Live Port and User Port).

    The Push sends on whichever port matches its current mode, so we listen
    to both and send LED messages to both.
    """

    def __init__(self, callback):
        import mido

        self.mido = mido
        in_names = [n for n in mido.get_input_names() if "Push 2" in n]
        out_names = [n for n in mido.get_output_names() if "Push 2" in n]
        if not in_names or not out_names:
            raise RuntimeError(
                "Push 2 MIDI ports not found. Is it plugged in and powered on?\n"
                f"Inputs seen: {mido.get_input_names()}"
            )
        self.inputs = [mido.open_input(n, callback=callback) for n in in_names]
        self.outputs = [mido.open_output(n) for n in out_names]
        print("MIDI in :", ", ".join(in_names))
        print("MIDI out:", ", ".join(out_names))

    def send(self, msg):
        for out in self.outputs:
            out.send(msg)

    def pad_color(self, note, color):
        self.send(self.mido.Message("note_on", note=note, velocity=color))

    def clear_pads(self):
        for note in range(PAD_FIRST, PAD_LAST + 1):
            self.pad_color(note, OFF)

    def close(self):
        for port in self.inputs + self.outputs:
            port.close()


# --------------------------------------------------------------------------
# USB: the display
# --------------------------------------------------------------------------

VENDOR_ID = 0x2982
PRODUCT_ID = 0x1967
ENDPOINT_OUT = 0x01

WIDTH, HEIGHT = 960, 160
LINE_BYTES = 2048                  # 960 px * 2 bytes = 1920, padded to 2048
FRAME_HEADER = bytes([0xFF, 0xCC, 0xAA, 0x88] + [0] * 12)
XOR_PATTERN = bytes([0xE7, 0xF3, 0xE7, 0xFF])
CHUNK = 16 * 1024

LIBUSB_PATHS = [
    "/opt/homebrew/lib/libusb-1.0.dylib",   # Homebrew on Apple Silicon
    "/usr/local/lib/libusb-1.0.dylib",      # Homebrew on Intel
]


def encode_frame(rgb):
    """rgb: numpy uint8 array (160, 960, 3) -> bytes ready for the Push.

    Pixel format: 16-bit little-endian, blue in the top 5 bits,
    green in the middle 6, red in the bottom 5. Each line is padded
    to 2048 bytes and the whole buffer is XOR-ed with a fixed pattern.
    """
    import numpy as np

    r = rgb[:, :, 0].astype(np.uint16) >> 3
    g = rgb[:, :, 1].astype(np.uint16) >> 2
    b = rgb[:, :, 2].astype(np.uint16) >> 3
    pixels = (b << 11) | (g << 5) | r                       # (160, 960) uint16

    buf = np.zeros((HEIGHT, LINE_BYTES), dtype=np.uint8)
    buf[:, : WIDTH * 2] = pixels.astype("<u2").view(np.uint8).reshape(HEIGHT, WIDTH * 2)

    mask = np.frombuffer(XOR_PATTERN * (LINE_BYTES // 4), dtype=np.uint8)
    buf ^= mask                                              # broadcast per line
    return buf.tobytes()


class PushDisplay:
    def __init__(self):
        import usb.core
        import usb.backend.libusb1

        lib = next((p for p in LIBUSB_PATHS if os.path.exists(p)), None)
        backend = usb.backend.libusb1.get_backend(
            find_library=(lambda _: lib) if lib else None
        )
        if backend is None:
            raise RuntimeError("libusb not found. Run: brew install libusb")

        self.dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID, backend=backend)
        if self.dev is None:
            raise RuntimeError("Push 2 display not found on USB. Is it plugged in?")
        try:
            self.dev.set_configuration()
        except usb.core.USBError:
            pass  # already configured, that's fine
        import usb.util
        usb.util.claim_interface(self.dev, 0)
        print("Display: connected")

    def send(self, frame_bytes):
        self.dev.write(ENDPOINT_OUT, FRAME_HEADER, timeout=1000)
        for i in range(0, len(frame_bytes), CHUNK):
            self.dev.write(ENDPOINT_OUT, frame_bytes[i : i + CHUNK], timeout=1000)

    def show(self, rgb):
        self.send(encode_frame(rgb))

    def close(self):
        import usb.util
        usb.util.release_interface(self.dev, 0)
        usb.util.dispose_resources(self.dev)


def now():
    return time.monotonic()
