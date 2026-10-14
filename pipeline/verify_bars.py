"""Per-bar verification montage: several Sony photos spread over the sequence,
cropped around the bar, with the assigned ids drawn."""
import json, os, sys, cv2, numpy as np
from config import WORK, OUT_DIR

ONLY_UNREGISTERED = "--unregistered" in sys.argv     # check the photos COLMAP had not registered

N = 6


def main():
    F = json.load(open(f"{WORK}/speakers_final.json"))["speakers"]
    poses = json.load(open(f"{WORK}/sony_poses_final.json"))
    vdir = f"{OUT_DIR}/verify" + ("_unregistered" if ONLY_UNREGISTERED else "")
    os.makedirs(vdir, exist_ok=True)
    for b in range(14):
        ids = [b * 8 + i + 1 for i in range(8)]
        per = {}
        for s in ids:
            for o in F[str(s)]["photos"]:
                if ONLY_UNREGISTERED and poses[o["photo"]].get("registered_in_sony_model", True):
                    continue
                per.setdefault(o["photo"], []).append((s, o))
        names = sorted(n for n, v in per.items() if len(v) >= 4) or sorted(per, key=lambda n: -len(per[n]))[:N]
        pick = [names[int(i)] for i in np.linspace(0, len(names) - 1, min(N, len(names)))] if names else []
        tiles = []
        for n in pick:
            pv = f"{OUT_DIR}/previews/{n[:-4]}.jpg"
            if not os.path.exists(pv):
                from config import SONY_IMAGES
                full = cv2.imread(f"{SONY_IMAGES}/{n}"); s_ = 1600 / full.shape[1]
                cv2.imwrite(pv, cv2.resize(full, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
            img = cv2.imread(pv); sc = img.shape[1] / 7008 if img.shape[1] > img.shape[0] else img.shape[1] / 3944
            pts = []
            for s, o in per[n]:
                c = (int(o["xy"][0] * sc), int(o["xy"][1] * sc)); r = max(5, int(o["r_px"] * sc)); pts.append(c)
                cv2.circle(img, c, r, (0, 255, 0), 2)
                cv2.putText(img, str(s), (c[0] - r, c[1] - r - 6), 0, 0.8, (0, 0, 0), 5)
                cv2.putText(img, str(s), (c[0] - r, c[1] - r - 6), 0, 0.8, (0, 255, 255), 2)
            P = np.array(pts); x0, y0 = P.min(0); x1, y1 = P.max(0)
            w = max(x1 - x0, 300); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            half_w = int(w * 0.75) + 80; half_h = int(half_w * 0.6)
            crop = img[max(0, cy - half_h): cy + half_h, max(0, cx - half_w): cx + half_w]
            crop = cv2.resize(crop, (640, 384))
            cv2.putText(crop, n, (8, 26), 0, 0.8, (0, 0, 255), 2)
            tiles.append(crop)
        if not tiles:
            continue
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        m = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        cv2.putText(m, f"bar {b+1}: ids {ids[0]}-{ids[-1]}", (8, m.shape[0] - 12), 0, 1.0, (255, 255, 0), 2)
        cv2.imwrite(f"{vdir}/bar{b+1:02d}_{ids[0]}-{ids[-1]}.jpg", m, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(b + 1, len(per), "photos;", pick)


if __name__ == "__main__":
    main()
