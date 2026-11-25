"""Downloads the KEYS synth's sounds (VSCO-2 Community Edition, CC0 / public domain)
and trims them to 48 kHz 16-bit WAVs named by MIDI note: keys/<PROGRAM>/<note>.wav"""
import json, os, re, subprocess, sys, urllib.request, urllib.parse

API = "https://api.github.com/repos/sgossner/VSCO-2-CE/contents/"
RAW = "https://raw.githubusercontent.com/sgossner/VSCO-2-CE/master/"
NAMES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
# program -> list of (folder, preferred dynamics in order, max seconds, MIDI range to use,
# octave fix: some folders name middle C "C3" - measured, they really sound an octave up)
PROGRAMS = {
    "PIANO": [("Keys/Upright Nr1", ["_mf_RR1", "_mf_RR2", "_f_RR1"], 7, (0, 127), 0)],
    "STRINGS": [("Strings/Cello Section/susvib", ["_v3_1", "_v1_1"], 6, (0, 54), 12),
                ("Strings/Violin Section/susVib", ["_v2", "_v1"], 6, (55, 127), 12)],
    "BRASS": [("Brass/Tenor Trombone/sus", ["_v3", "_v2", "_v1"], 5, (0, 57), 12),
              ("Brass/Trumpet/sus", ["_v3_rr1", "_v1_rr1"], 5, (58, 127), 12)],
    "ORGAN": [("Keys/Organ/Loud", ["Man3Open"], 5, (0, 127), 0)],
}


def listing(folder):
    with urllib.request.urlopen(API + urllib.parse.quote(folder)) as r:
        return [x["name"] for x in json.load(r) if x["name"].lower().endswith(".wav")]


def note_of(name):
    m = re.search(r"_([A-G]#?)(-?\d)", name)
    if m:
        return 12 * (int(m.group(2)) + 1) + NAMES[m.group(1)]       # C4 = 60
    m = re.search(r"Man3Open_(\d+)", name)                         # organ: key number 1 = C2
    if m:
        return 35 + int(m.group(1))
    return None


here = os.path.dirname(os.path.abspath(__file__))
for prog, parts in PROGRAMS.items():
    out_dir = os.path.join(here, prog)
    os.makedirs(out_dir, exist_ok=True)
    for folder, prefs, secs, (lo, hi), shift in parts:
        names = listing(folder)
        best = {}
        for nm in names:
            n = note_of(nm)
            n = None if n is None else n + shift
            if n is None or not lo <= n <= hi:
                continue
            rank = next((i for i, p in enumerate(prefs) if p in nm), None)
            if rank is None:
                continue
            if n not in best or rank < best[n][0]:
                best[n] = (rank, nm)
        for n, (_, nm) in sorted(best.items()):
            dst = os.path.join(out_dir, f"{n}.wav")
            if os.path.exists(dst):
                continue
            tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpushc_keys_tmp.wav")
            urllib.request.urlretrieve(RAW + urllib.parse.quote(f"{folder}/{nm}"), tmp)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-ar", "48000", "-ac", "2",
                            "-af", f"atrim=0:{secs},afade=t=out:st={secs - 0.4}:d=0.4", "-c:a", "pcm_s16le",
                            dst], check=True)
            os.remove(tmp)
            print(prog, n, nm, flush=True)
print("done")
