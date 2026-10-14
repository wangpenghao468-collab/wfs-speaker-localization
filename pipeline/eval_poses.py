"""Check each Sony pose (in DJI frame) against diaphragm detections.
A correct pose puts the projected bar-model speakers onto detected diaphragms."""
import json, numpy as np
from config import WORK


def load():
    B = json.load(open(f"{WORK}/dji_bar_model.json"))
    S = {int(k): np.array(v["xyz"]) for k, v in B["speakers"].items()}
    ids = sorted(S); X = np.array([S[i] for i in ids])
    # outward normal: speakers face the array centre (horizontal component)
    c = X.mean(0)
    nrm = {}
    for b in range(14):
        bid = [i for i in ids if (i - 1) // 8 == b]
        P = np.array([S[i] for i in bid]); d = np.linalg.svd(P - P.mean(0))[2][0]
        v = c - P.mean(0); v -= (v @ d) * d
        for i in bid:
            nrm[i] = v / np.linalg.norm(v)
    return ids, S, nrm, B["pitch"]


def project(pose, X):
    R, t, K = np.array(pose["R"]), np.array(pose["t"]), np.array(pose["K"])
    Xc = X @ R.T + t
    uv = Xc[:, :2] / Xc[:, 2:3] * [K[0, 0], K[1, 1]] + K[:2, 2]
    return uv, Xc[:, 2]


def evaluate(pose, ids, S, nrm, pitch, dets):
    R, t = np.array(pose["R"]), np.array(pose["t"]); C = -R.T @ t
    X = np.array([S[i] for i in ids]); uv, z = project(pose, X)
    W, H, f = pose["width"], pose["height"], pose["K"][0][0]
    dxy = np.array([d["xy_und"] for d in dets]) if dets else np.zeros((0, 2))
    res = []
    for k, i in enumerate(ids):
        if z[k] <= 0 or (C - S[i]) @ nrm[i] <= 0:
            continue
        ppx = pitch * f / z[k]
        if not (0 < uv[k, 0] < W and 0 < uv[k, 1] < H) or ppx < 25:
            continue
        e = np.linalg.norm(dxy - uv[k], axis=1).min() / ppx if len(dxy) else 9
        res.append((i, e, ppx))
    return res


def main():
    ids, S, nrm, pitch = load()
    poses = json.load(open(f"{WORK}/sony_in_dji.json")); det = json.load(open(f"{WORK}/detections.json"))
    rows = []
    for n, p in sorted(poses.items()):
        if "R" not in p:
            rows.append((n, 0, 0, 0, None)); continue
        r = evaluate(p, ids, S, nrm, pitch, det.get(n, []))
        hit = sum(e < 0.3 for _, e, _ in r)
        med = float(np.median([e for _, e, _ in r if e < 0.5])) if hit else None
        rows.append((n, p["inliers"], len(r), hit, med))
    good = [r for r in rows if r[2] >= 3 and r[3] >= 0.6 * r[2]]
    print("photos", len(rows), "pose ok (>=60% visible speakers hit, >=3 visible):", len(good))
    for lo, hi in [(0, 30), (30, 60), (60, 120), (120, 10000)]:
        sel = [r for r in rows if lo <= r[1] < hi]
        print(f"inliers {lo}-{hi}: {len(sel)} photos, ok {sum(1 for r in sel if r in good)}")
    json.dump({r[0]: dict(inliers=int(r[1]), visible=int(r[2]), hits=int(r[3]), med_err_pitch=r[4]) for r in rows},
              open(f"{WORK}/pose_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
