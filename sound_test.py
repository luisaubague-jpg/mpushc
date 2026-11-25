"""Does a plugin really change the sound outside a DAW?
    .venv/bin/python sound_test.py "/Library/Audio/Plug-Ins/VST3/Delay BRIGADE.vst3"
Feeds it a click and checks whether anything other than the dry click comes out."""
import sys

import numpy as np
import pedalboard as pb

SR, N = 48000, 512
path = sys.argv[1] if len(sys.argv) > 1 else "/Library/Audio/Plug-Ins/VST3/Delay BRIGADE.vst3"
p = pb.load_plugin(path)
print("loaded:", p)
settings = {"blend": 1.0, "feedback": 0.5, "echo_level": 1.0, "left_delay_time": 0.3,
            "right_delay_time": 0.3, "on_off": "Active"} if "BRIGADE" in path.upper() else {}
for k, v in settings.items():
    try:
        setattr(p, k, v)
    except Exception as err:
        print("could not set", k, err)
x = np.zeros((2, SR * 3), np.float32)
x[:, 100] = 1.0
y = np.concatenate([p(x[:, i:i + N], SR, reset=False) for i in range(0, x.shape[1], N)], axis=1)
n = min(x.shape[1], y.shape[1])
print("difference from dry:", round(float(np.abs(y[:, :n] - x[:, :n]).sum()), 3))
print("sound after 0.1 s:  ", round(float(np.abs(y[:, SR // 10:]).sum()), 3))
peaks = np.where(np.abs(y[0]) > 0.05)[0]
print("echoes at (ms):     ", sorted({int(i / SR * 1000) for i in peaks})[:12])
print("\nAll zeros = the plugin is silent outside a DAW.")
