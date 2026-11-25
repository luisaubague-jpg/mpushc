"""Checks the plugins Luis picked as models for MPushC's effects.

    .venv/bin/python plugin_check.py

Each plugin is loaded in its own separate Python process, so one that crashes
can't take the check down. Writes plugin_report.txt (for Claude): does it load,
how much CPU it takes per audio block, and every knob it has.
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

SR, BLOCK = 48000, 512
WANTED = {                                   # what to look for in the plugin file names
    "Saturation Knob": ("saturation knob",),
    "FreeAmp": ("freeamp",),
    "RO-Gold": ("ro-gold", "rogold", "ro gold"),
    "Brigade Delay": ("brigade",),
    "Airwindows Consolidated": ("airwindows consolidated",),
    "CHOWTapeModel": ("chowtape",),
}


def find(keys):
    roots = ["/Library/Audio/Plug-Ins", os.path.expanduser("~/Library/Audio/Plug-Ins")]
    vst3, au = [], []
    for r in roots:
        vst3 += glob.glob(os.path.join(r, "VST3", "*.vst3"))
        au += glob.glob(os.path.join(r, "Components", "*.component"))
    for group in (vst3, au):                 # VST3 first: loads more reliably headless
        for p in sorted(group):
            name = os.path.basename(p).lower()
            if any(k in name for k in keys):
                return p
    return None


def one(path):
    """Runs in the child process: load, measure, list the parameters, print JSON."""
    import pedalboard as pb
    t = time.perf_counter()
    p = pb.load_plugin(path)
    load_ms = (time.perf_counter() - t) * 1000
    x = (np.random.randn(2, BLOCK) * 0.2).astype(np.float32)
    lengths = [p(x, SR, reset=False).shape[1] for _ in range(4)]
    times = []
    for _ in range(600):
        t = time.perf_counter()
        p(x, SR, reset=False)
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times[20:])
    params = {}
    for name, prm in p.parameters.items():
        info = {}
        for attr in ("range", "valid_values", "units", "label", "string_value", "raw_value"):
            try:
                v = getattr(prm, attr)
                if callable(v):
                    continue
                info[attr] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)[:300]
            except Exception:
                pass
        params[name] = info
    print(json.dumps({"ok": True, "load_ms": round(load_ms), "is_instrument": p.is_instrument,
                      "block_ms_avg": round(float(times.mean()), 4),
                      "block_ms_worst": round(float(times.max()), 3),
                      "budget_pct": round(float(times.mean()) / (BLOCK / SR * 1000) * 100, 2),
                      "first_lengths": lengths, "params": params}))


def main():
    if "--one" in sys.argv:
        return one(sys.argv[sys.argv.index("--one") + 1])
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    lines = [f"MPushC plugin check  {time.strftime('%Y-%m-%d %H:%M')}  ({BLOCK}-sample blocks, "
             f"{BLOCK / SR * 1000:.1f} ms budget)\n"]
    for label, keys in WANTED.items():
        path = find(keys)
        print(f"{label:<26}", end="", flush=True)
        if not path:
            print("not found")
            lines.append(f"## {label}: NOT FOUND\n")
            continue
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", path],
                               capture_output=True, text=True, timeout=120)
            out = r.stdout.strip().splitlines()
            data = json.loads(out[-1]) if r.returncode == 0 and out else None
        except subprocess.TimeoutExpired:
            data, r = None, None
        if not data:
            why = "hung" if r is None else f"crashed (exit {r.returncode}) {r.stderr.strip()[-300:]}"
            print(why[:60])
            lines.append(f"## {label}: FAILED — {why}\n   file: {path}\n")
            continue
        print(f"OK  {data['block_ms_avg']} ms per block ({data['budget_pct']}% of budget)")
        lines.append(f"## {label}: OK\n   file: {path}\n   load {data['load_ms']} ms, "
                     f"avg {data['block_ms_avg']} ms, worst {data['block_ms_worst']} ms, "
                     f"{data['budget_pct']}% of budget, first block lengths {data['first_lengths']}, "
                     f"instrument={data['is_instrument']}\n   {len(data['params'])} parameters:")
        for name, info in data["params"].items():
            lines.append(f"     {name}: {json.dumps(info)[:400]}")
        lines.append("")
    with open("plugin_report.txt", "w") as f:
        f.write("\n".join(lines))
    print("\nDone: plugin_report.txt written. Tell Claude.")


if __name__ == "__main__":
    main()
