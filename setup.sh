#!/bin/bash
# MPushC v0.0 - one-time setup. Run from inside the mpushc folder:  bash setup.sh
set -e

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is not installed. Install it from https://brew.sh, then run this again."
  exit 1
fi

echo "==> Installing libusb (lets Python talk to the Push screen)"
brew list libusb >/dev/null 2>&1 || brew install libusb

echo "==> Creating a Python environment in .venv"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install mido python-rtmidi pyusb pillow numpy

echo
echo "Done. Each time you open a new Terminal window, run:"
echo "  cd $(pwd) && source .venv/bin/activate"
echo "Then:  python pads.py   |   python screen.py   |   python hello.py"
