import time, numpy as np, sounddevice as sd
sr = 48000
t = np.arange(sr) / sr
tone = 0.2 * np.sin(2 * np.pi * 440 * t)
fade = np.linspace(0, 1, 480)
tone[:480] *= fade; tone[-480:] *= fade[::-1]
pos = [len(tone)]
def cb(out, frames, time_info, status):
    out[:] = 0
    chunk = tone[pos[0]:pos[0] + frames]
    out[:len(chunk), 0] = chunk; out[:len(chunk), 1] = chunk
    pos[0] += frames
with sd.OutputStream(samplerate=sr, channels=2, device='AudioFuse', blocksize=128, callback=cb):
    time.sleep(0.5)
    for i in range(3):
        pos[0] = 0; time.sleep(1.5)
