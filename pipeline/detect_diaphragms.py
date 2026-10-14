"""Detect silver diaphragms in every selected Sony photo.

Detector: bright low-saturation compact blob surrounded by a dark speaker frame
(the look of the SAM2 diaphragm masks). Centres are also given undistorted.
Output: detections.json
"""
import cv2, numpy as np, json, os
from config import SONY_IMAGES, WORK

DET_W = 2400


def detect(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    bright = ((v > 140) & (s < 60)).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
    ii = cv2.integral((v < 70).astype(np.float32))
    H, W = v.shape
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 120 or a > 12000 or w > 2.5 * h or h > 2.5 * w or a / (w * h) < 0.55:
            continue
        cnt = cv2.findContours((lab[y:y + h, x:x + w] == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)[0][0]
        per = cv2.arcLength(cnt, True)
        if per == 0 or 4 * np.pi * a / per ** 2 < 0.6:
            continue
        cx, cy = x + w / 2, y + h / 2
        R = max(w, h) * 0.9
        x0, y0, x1, y1 = int(max(0, cx - R)), int(max(0, cy - R)), int(min(W, cx + R)), int(min(H, cy + R))
        ring = (x1 - x0) * (y1 - y0) - a
        dark = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / max(ring, 1)
        if dark < 0.35:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        out.append(dict(xy=(float(xs.mean() + x), float(ys.mean() + y)), r=float(np.sqrt(a / np.pi)),
                        bbox=(int(x), int(y), int(w), int(h)), dark=float(dark)))
    return out


def main():
    intr = json.load(open(f"{WORK}/sony_intrinsics.json"))
    path = f"{WORK}/detections.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for k, n in enumerate(sorted(intr)):
        if n in res and res[n] and "xy_und" in res[n][0] and len(res[n][0]) == 4:
            continue
        K, dist = np.array(intr[n]["K"]), np.array(intr[n]["dist"])
        img = cv2.imread(f"{SONY_IMAGES}/{n}")
        sc = DET_W / img.shape[1]
        dets = detect(cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))
        out = []
        for d in dets:
            c = np.array(d["xy"]) / sc
            und = cv2.undistortPoints(c[None, None], K, dist, P=K)[0, 0]
            out.append(dict(xy=c.tolist(), xy_und=und.tolist(), r=d["r"] / sc, dark=d["dark"]))
        res[n] = out
        if k % 50 == 0:
            print(f"[{k+1}/{len(intr)}] {n}: {len(out)} dets", flush=True)
            json.dump(res, open(path, "w"))
    json.dump(res, open(path, "w"))
    print("photos with detections:", sum(1 for v in res.values() if v))


if __name__ == "__main__":
    main()
