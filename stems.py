"""
Stem separation worker for MPushC (runs in its own process so the audio never glitches).
    python stems.py <input.wav> <output folder> [fast|best]
fast = htdemucs (seconds). best = htdemucs_ft + 2 shifts (~4-8x slower, cleaner).
The bass stem gets a gentle low-pass (1.5 kHz) to remove highs bleeding in.
Writes drums.wav, bass.wav, other.wav, vocals.wav into the output folder and prints
"DONE" (or "ERROR: ...") on the last line. Uses demucs-mlx (Apple Silicon, no PyTorch):
    pip install demucs-mlx
"""
import os
import sys

import numpy as np
import soundfile as sf


def gentle_lowpass(a, sr, fc=1500.0):
    """Zero-phase, 12 dB/octave roll-off above fc (Butterworth-shaped), done with one FFT."""
    n = len(a)
    spec = np.fft.rfft(a, axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    gain = 1 / np.sqrt(1 + (f / fc) ** 4)
    return np.fft.irfft(spec * gain[:, None], n=n, axis=0).astype(np.float32)


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "fast"
    try:
        from demucs_mlx import Separator
    except ImportError:
        print("ERROR: demucs-mlx is not installed (pip install demucs-mlx)", flush=True)
        return 1
    os.makedirs(out_dir, exist_ok=True)
    if mode == "best":
        separator = Separator(model="htdemucs_ft", shifts=2)
    else:
        separator = Separator()                   # htdemucs: drums, bass, other, vocals
    _, stems = separator.separate_audio_file(src)
    sr = (getattr(separator, "samplerate", None)
          or getattr(getattr(separator, "model", None), "samplerate", None) or 44100)
    for name, audio in stems.items():
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 3:                            # (batch, channels, time)
            a = a[0]
        if a.ndim == 2 and a.shape[0] <= 2 < a.shape[1]:
            a = a.T                                # -> (time, channels)
        if name == "bass":
            a = gentle_lowpass(a, int(sr))
        sf.write(os.path.join(out_dir, f"{name}.wav"), a, int(sr), subtype="PCM_24")
        print("STEM", name, flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:                       # report, never hang
        print(f"ERROR: {type(err).__name__}: {err}", flush=True)
        sys.exit(1)
