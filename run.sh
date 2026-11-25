#!/bin/sh
# Start MPushC on free-threaded Python (no lock: audio never waits for the screen).
# PYTHON_GIL=0 keeps the lock off even for the two add-ons not yet marked as safe for it
# (python-rtmidi, python-stretch).
cd "$(dirname "$0")"
PYTHON_GIL=0 exec .venv-ft/bin/python seq.py "$@"
