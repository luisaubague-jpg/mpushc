"""
Step: the kit + the screen. Hit a pad: the Push screen shows that sample's
name and its real waveform, with a playhead that follows the sound.
Ctrl+C to quit.
"""

import os
import time
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw

import push2
import kit
from kit import Voice, pad_samples, idle_color
from play import velocity_to_gain, SR, BLOCK, MAX_VOICES
from render import load_font, BG, INK, QUIET, WAVE, PLAYED

W, H = push2.WIDTH, push2.HEIGHT
WAVE_X, WAVE_W, WAVE_MID, WAVE_HALF = 220, 720, 84, 62


def peaks(audio, width=WAVE_W):
    """Loudest point of each screen column, 0..1, for drawing."""
    mono = np.abs(audio).max(axis=1)
    cols = np.array_split(mono, width)
    p = np.array([c.max() if len(c) else 0.0 for c in cols])
    top = p.max()
    return p / top if top > 0 else p


wave_peaks = {note: peaks(audio) for note, (_, audio) in pad_samples.items()}

# ---------------------------------------------------------------------------
# Tuning: every sample is assumed to be C3 until we add pitch detection.
# Knob 1 retunes the last pad you hit, in semitones (speed + pitch together).
# ---------------------------------------------------------------------------
ROOT = 60                     # C3 (Ableton naming)
TUNE_RANGE = 24               # +/- 2 octaves
TUNE_KNOB = 71                # knob 1, above the screen
TICKS_PER_SEMITONE = 4        # raise if the knob feels too fast
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

tune = {note: 0 for note in pad_samples}
pitched = {note: audio for note, (_, audio) in pad_samples.items()}
knob_ticks = [0]


def note_name(midi_note):
    return f"{NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 2}"


def repitch(audio, semitones):
    """Play faster/slower: +12 = one octave up and half as long."""
    if semitones == 0:
        return audio
    rate = 2 ** (semitones / 12)
    n_out = max(1, int(len(audio) / rate))
    x_new = np.arange(n_out) * rate
    x_old = np.arange(len(audio))
    return np.column_stack(
        [np.interp(x_new, x_old, audio[:, c]) for c in range(2)]
    ).astype(np.float32)


def set_tune(note, semitones):
    semitones = max(-TUNE_RANGE, min(TUNE_RANGE, semitones))
    if semitones == tune[note]:
        return
    tune[note] = semitones
    pitched[note] = repitch(pad_samples[note][1], semitones)
    wave_peaks[note] = peaks(pitched[note])
    print(f"{pad_samples[note][0]:30s} -> {note_name(ROOT + semitones)} ({semitones:+d})")

# ---------------------------------------------------------------------------
# Mute groups: pads in the same group cut each other off.
# Pads are numbered 1-16 from the bottom-left, left to right, then up.
# ---------------------------------------------------------------------------
MUTE_GROUPS = {
    1: [3, 4],          # closed hat (pad 3) and open hat (pad 4)
}
FADE = 96               # samples (2 ms) to fade a choked sound, avoids clicks


def pad_number_to_note(n):
    return push2.pad_note((n - 1) // 4, (n - 1) % 4)


group_of = {pad_number_to_note(n): g for g, pads in MUTE_GROUPS.items() for n in pads}


class PadVoice(Voice):
    __slots__ = ("note", "fade_left")

    def __init__(self, audio, gain, note):
        super().__init__(audio, gain)
        self.note, self.fade_left = note, None


def choke(group):
    for v in list(kit.voices):
        if group_of.get(v.note) == group and v.fade_left is None:
            v.fade_left = FADE


# ---------------------------------------------------------------------------
# Clock + metronome. Timing is counted in audio samples inside the audio
# callback, so every click lands on the exact sample, whatever Python does.
# ---------------------------------------------------------------------------
PLAY_BUTTON = 85              # bottom-left button on the Push
BPM_KNOB = 72                 # knob 2
TICKS_PER_BPM = 2
BPM_MIN, BPM_MAX = 40, 200


def make_click(freq, ms=35):
    n = int(SR * ms / 1000)
    t = np.arange(n) / SR
    c = 0.35 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 90)
    return np.column_stack([c, c]).astype(np.float32)


CLICK_ACCENT, CLICK = make_click(1600), make_click(1000)
clock = {"playing": False, "bpm": 90, "sample": 0, "next_beat": 0.0, "beat": -1}
click_tails = []              # clicks still sounding: [audio, position]
bpm_ticks = [0]


def start_clock():
    clock.update(sample=0, next_beat=0.0, beat=-1)
    click_tails.clear()
    clock["playing"] = True
    midi.send(midi.mido.Message("control_change", control=PLAY_BUTTON, value=push2.GREEN))
    print(f"Play  {clock['bpm']} BPM")


def stop_clock():
    clock["playing"] = False
    midi.send(midi.mido.Message("control_change", control=PLAY_BUTTON, value=push2.OFF))
    print("Stop")


def mix_clock(out, frames):
    # finish clicks that started in an earlier block
    for tail in list(click_tails):
        audio, pos = tail
        chunk = audio[pos:pos + frames]
        out[:len(chunk)] += chunk
        tail[1] = pos + frames
        if tail[1] >= len(audio):
            click_tails.remove(tail)
    if not clock["playing"]:
        return
    spb = SR * 60.0 / clock["bpm"]              # samples per beat
    start, end = clock["sample"], clock["sample"] + frames
    while clock["next_beat"] < end:
        offset = int(clock["next_beat"]) - start
        clock["beat"] += 1
        audio = CLICK_ACCENT if clock["beat"] % 4 == 0 else CLICK
        chunk = audio[:frames - offset]
        out[offset:offset + len(chunk)] += chunk
        if len(chunk) < len(audio):
            click_tails.append([audio, len(chunk)])
        clock["next_beat"] += spb
    clock["sample"] = end


def audio_callback(out, frames, time_info, status):
    out[:] = 0
    mix_clock(out, frames)
    for v in list(kit.voices):
        chunk = v.audio[v.pos:v.pos + frames] * v.gain
        if v.fade_left is not None:
            n = len(chunk)
            ramp = np.clip((v.fade_left - np.arange(n)) / FADE, 0, 1).astype(np.float32)
            chunk = chunk * ramp[:, None]
            v.fade_left -= n
        out[:len(chunk)] += chunk
        v.pos += frames
        if v.pos >= len(v.audio) or (v.fade_left is not None and v.fade_left <= 0):
            try:
                kit.voices.remove(v)
            except ValueError:
                pass
    np.clip(out, -1.0, 1.0, out=out)

state = {"note": None, "voice": None, "velocity": 0}
midi = None
big, small = load_font(30), load_font(16)


def on_message(msg):
    if msg.type == "note_on" and push2.is_pad(msg.note):
        if msg.velocity > 0:
            if msg.note in pad_samples:
                audio = pitched[msg.note]
                if msg.note in group_of:
                    choke(group_of[msg.note])
                voice = PadVoice(audio, velocity_to_gain(msg.velocity), msg.note)
                if len(kit.voices) >= MAX_VOICES:
                    kit.voices.pop(0)
                kit.voices.append(voice)
                state.update(note=msg.note, voice=voice, velocity=msg.velocity)
                midi.pad_color(msg.note, push2.WHITE)
        else:
            midi.pad_color(msg.note, idle_color(msg.note))
    elif msg.type == "note_off" and push2.is_pad(msg.note):
        midi.pad_color(msg.note, idle_color(msg.note))
    elif msg.type == "control_change" and msg.control == PLAY_BUTTON:
        if msg.value > 0:                                       # press, not release
            stop_clock() if clock["playing"] else start_clock()
    elif msg.type == "control_change" and msg.control == BPM_KNOB:
        step = msg.value if msg.value < 64 else msg.value - 128
        bpm_ticks[0] += step
        while abs(bpm_ticks[0]) >= TICKS_PER_BPM:
            direction = 1 if bpm_ticks[0] > 0 else -1
            bpm_ticks[0] -= direction * TICKS_PER_BPM
            clock["bpm"] = max(BPM_MIN, min(BPM_MAX, clock["bpm"] + direction))
    elif msg.type == "control_change" and msg.control == TUNE_KNOB:
        note = state["note"]
        if note is None:
            return
        step = msg.value if msg.value < 64 else msg.value - 128   # 1 = right, -1 = left
        knob_ticks[0] += step
        while abs(knob_ticks[0]) >= TICKS_PER_SEMITONE:
            direction = 1 if knob_ticks[0] > 0 else -1
            knob_ticks[0] -= direction * TICKS_PER_SEMITONE
            set_tune(note, tune[note] + direction)


def draw_clock(d):
    """Bottom-left: BPM and four beat squares (beat 1 = orange)."""
    d.text((22, 132), f"{clock['bpm']} BPM", font=small, fill=INK if clock["playing"] else QUIET)
    current = clock["beat"] % 4 if clock["playing"] and clock["beat"] >= 0 else -1
    for i in range(4):
        x = 110 + i * 22
        fill = (PLAYED if i == 0 else WAVE) if i == current else None
        d.rectangle([x, 134, x + 14, 148], fill=fill, outline=QUIET)


def frame():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    draw_clock(d)
    note = state["note"]
    if note is None:
        d.text((20, 18), "MPushC", font=big, fill=INK)
        d.text((22, 60), "kit loaded: hit a pad", font=small, fill=QUIET)
        return np.asarray(img, dtype=np.uint8)

    name = pad_samples[note][0]
    audio = pitched[note]
    row, col = push2.pad_position(note)
    st = tune[note]
    d.text((20, 16), os.path.splitext(name)[0][:12], font=big, fill=INK)
    d.text((22, 58), f"pad {row + 1}·{col + 1}   {len(audio) / SR:.2f} s", font=small, fill=QUIET)
    d.text((22, 84), f"{note_name(ROOT + st)}", font=big, fill=PLAYED if st else INK)
    d.text((110, 96), f"{st:+d} st", font=small, fill=QUIET)

    voice = state["voice"]
    playing = voice is not None and voice in kit.voices
    head = int(voice.pos / len(voice.audio) * WAVE_W) if playing else -1
    for i, p in enumerate(wave_peaks[note]):
        h = max(1, int(p * WAVE_HALF))
        color = PLAYED if i < head else WAVE
        d.line([(WAVE_X + i, WAVE_MID - h), (WAVE_X + i, WAVE_MID + h)], fill=color)
    if playing:
        d.line([(WAVE_X + head, 18), (WAVE_X + head, H - 18)], fill=INK)
    return np.asarray(img, dtype=np.uint8)


def main():
    global midi
    display = push2.PushDisplay()
    midi = push2.PushMidi(on_message)
    midi.clear_pads()
    for note in pad_samples:
        midi.pad_color(note, push2.BLUE)
    stream = sd.OutputStream(samplerate=SR, channels=2, device="AudioFuse",
                             blocksize=BLOCK, dtype="float32", callback=audio_callback)
    stream.start()
    print("\nPlay! Watch the Push screen. Ctrl+C to quit.\n")
    try:
        while True:
            display.show(frame())
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            midi.clear_pads()
            stop_clock()
            stream.abort()
        except Exception:
            pass
        print("\nBye.")
        os._exit(0)


if __name__ == "__main__":
    main()
