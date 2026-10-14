"""Debug: draw every identified speaker id on Sony previews, side by side."""
import json, sys, cv2, numpy as np
from config import WORK, OUT_DIR

F = json.load(open(f"{WORK}/speakers_final.json"))["speakers"]
per = {}
for s, v in F.items():
    for o in v["photos"]:
        per.setdefault(o["photo"], []).append((s, o))
tiles = []
for n in sys.argv[2:]:
    img = cv2.imread(f"{OUT_DIR}/previews/{n[:-4]}.jpg"); sc = img.shape[1] / 7008
    for s, o in per.get(n, []):
        c = (int(o["xy"][0] * sc), int(o["xy"][1] * sc)); r = max(5, int(o["r_px"] * sc))
        cv2.circle(img, c, r, (0, 255, 0), 2)
        cv2.putText(img, s, (c[0] - r, c[1] - r - 6), 0, 0.9, (0, 0, 0), 5)
        cv2.putText(img, s, (c[0] - r, c[1] - r - 6), 0, 0.9, (0, 255, 255), 2)
    cv2.putText(img, n, (10, 40), 0, 1.3, (0, 0, 255), 3)
    tiles.append(cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1]))))
w = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.zeros((w - t.shape[0], 800, 3), np.uint8)]) for t in tiles]
rows = [np.hstack(tiles[i:i + 2] + [np.zeros_like(tiles[0])] * (2 - len(tiles[i:i + 2]))) for i in range(0, len(tiles), 2)]
cv2.imwrite(sys.argv[1], np.vstack(rows))
