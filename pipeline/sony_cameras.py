"""Intrinsics for every selected Sony photo (registered or not).

Registered photos use their COLMAP camera. Unregistered photos borrow the
camera of registered photos with the same EXIF focal length (the lens zoom
setting), which is what COLMAP calibrated per focal group.
Output: sony_intrinsics.json
"""
import json, collections, csv
import numpy as np
from PIL import Image
from colmap_io import read_model
from config import SONY_MODEL, SONY_IMAGES, WORK, SONY_SELECTED, SONY_QUALITY

SELECTED = SONY_SELECTED
QUALITY = SONY_QUALITY


def exif_focal(name):
    e = Image.open(f"{SONY_IMAGES}/{name}").getexif().get_ifd(0x8769)
    return float(e.get(0x920A, 0)), Image.open(f"{SONY_IMAGES}/{name}").size


def main():
    cams, imgs = read_model(SONY_MODEL, with_points=False)
    reg = {im["name"]: im["camera_id"] for im in imgs.values()}
    names = [l.strip() for l in open(SELECTED) if l.strip()]
    sharp = {r["file"]: float(r["sharpness_lap_var_1280"]) for r in csv.DictReader(open(QUALITY, encoding="utf-8-sig"))}
    focal = {n: exif_focal(n) for n in names}
    by_focal = collections.defaultdict(collections.Counter)
    for n, cid in reg.items():
        by_focal[(focal[n][0], focal[n][1])][cid] += 1
    out, unmatched = {}, []
    for n in names:
        f_mm, size = focal[n]
        if n in reg:
            cid = reg[n]
        elif by_focal.get((f_mm, size)):
            cid = by_focal[(f_mm, size)].most_common(1)[0][0]
        else:                                               # nearest calibrated focal length
            cands = [(abs(k[0] - f_mm), k) for k in by_focal if k[1] == size]
            cid = by_focal[min(cands)[1]].most_common(1)[0][0] if cands else None
            unmatched.append(n)
        if cid is None:
            continue
        c = cams[cid]; fpx, cx, cy, k1 = c["params"]
        if (c["width"], c["height"]) != size:
            continue
        out[n] = dict(K=[[fpx, 0, cx], [0, fpx, cy], [0, 0, 1]], dist=[k1, 0, 0, 0, 0], width=c["width"],
                      height=c["height"], camera_id=cid, focal_mm=f_mm, registered=n in reg, sharpness=sharp.get(n))
    json.dump(out, open(f"{WORK}/sony_intrinsics.json", "w"), indent=1)
    print(f"{len(out)} photos ({sum(v['registered'] for v in out.values())} registered); "
          f"nearest-focal fallback for {len(unmatched)}; skipped {len(names) - len(out)}")


if __name__ == "__main__":
    main()
