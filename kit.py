"""
Step: a real kit. WAVs in the kit/ folder fill the bottom-left 4x4 pads,
in file-name order, starting at the bottom-left pad and going left to right,
then up. Loaded pads glow blue; a hit flashes white. Ctrl+C to quit.
"""

import glob
import os
import time
import numpy as np
import sounddevice as sd
import push2
from play import load_sample, velocity_to_gain, SR, BLOCK, MAX_VOICES

files = sorted(glob.glob(os.path.join("kit", "*.wav")))[:16]
if not files:
    raise SystemExit("No WAV files found in the kit folder.")

pad_samples = {}                     # MIDI note -> (name, audio)
for i, path in enumerate(files):
    note = push2.pad_note(i // 4, i % 4)
    pad_samples[note] = (os.path.basename(path), load_sample(path))

class Voice:
    """One playing sound. A plain object, so Python compares voices by
    identity (comparing lists that hold audio arrays caused a crash)."""

    __slots__ = ("audio", "pos", "gain")

    def __init__(self, audio, gain):
        self.audio, self.pos, self.gain = audio, 0, gain


voices = []
midi = None


def audio_callback(out, frames, time_info, status):
    out[:] = 0
    for voice in list(voices):
        chunk = voice.audio[voice.pos:voice.pos + frames]
        out[:len(chunk)] += chunk * voice.gain
        voice.pos += frames
        if voice.pos >= len(voice.audio):
            try:
                voices.remove(voice)
            except ValueError:
                pass              # already dropped by the voice limit
    np.clip(out, -1.0, 1.0, out=out)


def idle_color(note):
    return push2.BLUE if note in pad_samples else push2.OFF


def on_message(msg):
    if msg.type == "note_on" and push2.is_pad(msg.note):
        if msg.velocity > 0:
            if msg.note in pad_samples:
                name, audio = pad_samples[msg.note]
                if len(voices) >= MAX_VOICES:
                    voices.pop(0)
                voices.append(Voice(audio, velocity_to_gain(msg.velocity)))
                midi.pad_color(msg.note, push2.WHITE)
                print(f"{name:30s} velocity {msg.velocity:3d}")
        else:
            midi.pad_color(msg.note, idle_color(msg.note))
    elif msg.type == "note_off" and push2.is_pad(msg.note):
        midi.pad_color(msg.note, idle_color(msg.note))


def main():
    global midi
    midi = push2.PushMidi(on_message)
    midi.clear_pads()
    print()
    for note, (name, _) in pad_samples.items():
        row, col = push2.pad_position(note)
        print(f"  pad row {row + 1} col {col + 1}: {name}")
        midi.pad_color(note, push2.BLUE)
    stream = sd.OutputStream(samplerate=SR, channels=2, device="AudioFuse",
                             blocksize=BLOCK, dtype="float32", callback=audio_callback)
    stream.start()
    print("\nPlay! Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Closing audio + MIDI cleanly can hang on macOS, so: lights off,
        # stop audio immediately, then exit hard. Nothing is left running.
        try:
            midi.clear_pads()
            stream.abort()
        except Exception:
            pass
        print("\nBye.")
        os._exit(0)


if __name__ == "__main__":
    main()
