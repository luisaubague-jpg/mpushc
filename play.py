"""
Step: first real sound. Every pad plays kick.wav through the AudioFuse.
Harder hits = louder. Ctrl+C to quit.
"""

import time
import numpy as np
import soundfile as sf
import sounddevice as sd
import push2

SR = 48000            # AudioFuse sample rate
BLOCK = 128           # audio buffer: 128 samples = 2.7 ms
MAX_VOICES = 16


def load_sample(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:                       # mono -> stereo
        data = np.repeat(data, 2, axis=1)
    data = data[:, :2]
    if sr != SR:                                 # simple resample to 48 kHz
        n_out = int(len(data) * SR / sr)
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, n_out)
        data = np.column_stack([np.interp(x_new, x_old, data[:, c]) for c in range(2)]).astype(np.float32)
    return data


kick = load_sample("kick.wav")
voices = []          # each voice: [position, gain]
midi = None


def velocity_to_gain(v):
    # Your hardest hits land around 82, so treat 85+ as full volume for now.
    return min(1.0, v / 85.0) ** 1.5


def audio_callback(out, frames, time_info, status):
    out[:] = 0
    for voice in list(voices):
        pos, gain = voice
        chunk = kick[pos:pos + frames]
        out[:len(chunk)] += chunk * gain
        voice[0] = pos + frames
        if voice[0] >= len(kick):
            voices.remove(voice)
    np.clip(out, -1.0, 1.0, out=out)


def on_message(msg):
    if msg.type == "note_on" and push2.is_pad(msg.note):
        if msg.velocity > 0:
            if len(voices) >= MAX_VOICES:
                voices.pop(0)
            voices.append([0, velocity_to_gain(msg.velocity)])
            midi.pad_color(msg.note, push2.RED)
            print(f"hit  velocity {msg.velocity:3d}")
        else:
            midi.pad_color(msg.note, push2.OFF)
    elif msg.type == "note_off" and push2.is_pad(msg.note):
        midi.pad_color(msg.note, push2.OFF)


def main():
    global midi
    midi = push2.PushMidi(on_message)
    midi.clear_pads()
    with sd.OutputStream(samplerate=SR, channels=2, device="AudioFuse",
                         blocksize=BLOCK, dtype="float32", callback=audio_callback) as stream:
        print(f"\nAudio latency: {stream.latency * 1000:.1f} ms. Hit any pad. Ctrl+C to quit.\n")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    midi.clear_pads()
    midi.close()
    print("\nBye.")


if __name__ == "__main__":
    main()
