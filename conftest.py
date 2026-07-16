"""Root conftest.py — ensure the repo root is on sys.path so tests can do
`from tools.gpurir_scenes.xxx import ...` without per-file path hacks."""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
