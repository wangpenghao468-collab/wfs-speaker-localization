"""User feedback from the web page (speaker_corrections.json).

Looked up in the finder folder first, then the newest one in ~/Downloads."""
import json, glob, os
from config import OUT_DIR


def load_corrections(verbose=True):
    paths = [f"{OUT_DIR}/speaker_corrections.json"]
    if not os.path.exists(paths[0]):
        paths = sorted(glob.glob(os.path.expanduser("~/Downloads/speaker_corrections*.json")), key=os.path.getmtime)[-1:]
    if not paths or not os.path.exists(paths[0]):
        return {}
    by = {}
    for c in json.load(open(paths[0], encoding="utf-8"))["corrections"]:
        by.setdefault(c["photo"], []).append(c)
    if verbose:
        print(f"corrections: {sum(map(len, by.values()))} from {paths[0]}")
    return by
