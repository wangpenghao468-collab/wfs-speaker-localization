"""Speaker heights with an independent check of the floor plane.

The DJI sparse cloud has floor points mostly in the room corners, so a floor
plane fitted to it may be tilted. Here the floor is densified from Sony photo
pairs (known poses): SIFT matches gated by the epipolar constraint are
triangulated and points near the floor are kept. Then:
  * one global floor plane (RANSAC + least squares);
  * one plane per room region (left / right / front / back) to test flatness
    and tilt stability;
  * bootstrap of the floor points for the plane's uncertainty;
  * every speaker height relative to the global floor and to the local floor
    (floor points within 1 m horizontally of the speaker).
Output: outputs/.../speaker_heights.csv, work/.../floor.json
"""
import json, csv, sqlite3, numpy as np, torch
from common import mutual_ratio
from config import WORK, OUT_DIR, SONY_DB

DB = SONY_DB
MAX_PAIRS = 3000
EPI_PX = 3.0
BAND_MM = 40             # keep candidate floor points within this of the current floor estimate (frame floor is z = 0)


def proj_mat(p):
    return np.array(p["K"]) @ np.hstack([np.array(p["R"]), np.array(p["t"])[:, None]])


def undist(xy, p):
    import cv2
    return cv2.undistortPoints(xy[:, None].astype(np.float64), np.array(p["K"]), np.array(p["dist"]),
                               P=np.array(p["K"]))[:, 0]


def tri(P1, P2, a, b):
    A = np.stack([a[:, 0:1] * P1[2] - P1[0], a[:, 1:2] * P1[2] - P1[1],
                  b[:, 0:1] * P2[2] - P2[0], b[:, 1:2] * P2[2] - P2[1]], 1)
    X = np.linalg.svd(A)[2][:, -1]
    return X[:, :3] / X[:, 3:]


def plane_fit(P, rng, iters=800, thr=8.0):
    best = None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - s[0]) @ n) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]; c = Q.mean(0); n = np.linalg.svd(Q - c)[2][2]
    for _ in range(20):                       # deterministic finish: Huber-weighted LS on all points
        r = (P - c) @ n
        w = np.where(np.abs(r) < thr, 1.0, thr / np.maximum(np.abs(r), 1e-9))
        w[np.abs(r) > 4 * thr] = 0
        c = (P * w[:, None]).sum(0) / w.sum()
        n_new = np.linalg.svd((P - c) * np.sqrt(w)[:, None])[2][2]
        if abs(n_new @ n) > 1 - 1e-12:
            n = n_new; break
        n = n_new
    if n[2] < 0:
        n = -n
    best = np.abs((P - c) @ n) < thr
    return c, n, best


def main():
    C = json.load(open(f"{WORK}/centers.json")); mm = C["mm_per_unit"]
    o = np.array(C["origin"]); Bx = np.array([C["x"], C["y"], C["z"]])
    to_mm = lambda X: (X - o) @ Bx.T * mm                 # DJI units -> current floor frame (mm)
    poses = json.load(open(f"{WORK}/sony_poses_final.json"))
    db = sqlite3.connect(DB)
    idn = {i: n for i, n in db.execute("select image_id, name from images")}
    pairs = [(idn[p // 2147483647], idn[p % 2147483647], r) for p, r in
             db.execute("select pair_id, rows from two_view_geometries where rows >= 60")]
    pairs = [(a, b) for a, b, r in sorted(pairs, key=lambda x: -x[2]) if a in poses and b in poses][:MAX_PAIRS]
    print("photo pairs:", len(pairs))

    feats = {}
    def F(n):
        if n not in feats:
            z = np.load(f"{WORK}/feats2/sony_{n}.npz")
            feats[n] = (z["xy"], torch.tensor(z["desc"]).cuda(), undist(z["xy"], poses[n]))
        return feats[n]

    pts = []
    for k, (a, b) in enumerate(pairs):
        xa, da, ua = F(a); xb, dbb, ub = F(b)
        q, t = mutual_ratio(da, dbb, ratio=0.9)
        if len(q) < 10:
            continue
        Pa, Pb = proj_mat(poses[a]), proj_mat(poses[b])
        X = tri(Pa, Pb, ua[q], ub[t])
        ok = np.ones(len(X), bool)
        for P, u in ((Pa, ua[q]), (Pb, ub[t])):
            h = np.hstack([X, np.ones((len(X), 1))]) @ P.T
            ok &= (h[:, 2] > 0) & (np.linalg.norm(h[:, :2] / h[:, 2:] - u, axis=1) < EPI_PX)
        Ca = -np.array(poses[a]["R"]).T @ np.array(poses[a]["t"]); Cb = -np.array(poses[b]["R"]).T @ np.array(poses[b]["t"])
        ra, rb = X - Ca, X - Cb
        ang = np.degrees(np.arccos(np.clip((ra * rb).sum(1) / np.linalg.norm(ra, axis=1) / np.linalg.norm(rb, axis=1), -1, 1)))
        ok &= ang > 3
        Y = to_mm(X[ok])
        pts.append(Y[np.abs(Y[:, 2]) < BAND_MM])
        if k % 300 == 0:
            print(f"[{k}/{len(pairs)}] floor candidates so far {sum(map(len, pts))}", flush=True)
    P = np.concatenate(pts)
    rng = np.random.default_rng(0)
    c, n, inl = plane_fit(P, rng)
    Fp = P[inl]
    res = (Fp - c) @ n
    print(f"floor points {len(Fp)} (of {len(P)} candidates), plane rms {np.sqrt((res ** 2).mean()):.1f} mm")

    # region planes: tilt and offset relative to the global plane
    S = {r["id"]: np.array(r["xyz"]) for r in C["rows"]}
    xs = np.array([S[i][0] for i in S]); ys = np.array([S[i][1] for i in S])
    regions = {"左側 (x<-1000)": Fp[:, 0] < -1000, "右側 (x>1000)": Fp[:, 0] > 1000,
               "前方 (y>中位)": Fp[:, 1] > np.median(Fp[:, 1]), "後方 (y<中位)": Fp[:, 1] <= np.median(Fp[:, 1])}
    reg = {}
    for name, m in regions.items():
        if m.sum() < 50:
            reg[name] = None; continue
        cr, nr, _ = plane_fit(Fp[m], rng, thr=6)
        tilt = float(np.degrees(np.arccos(abs(nr @ n))))
        reg[name] = dict(n=int(m.sum()), tilt_deg=tilt, offset_mm=float((cr - c) @ n))
        print(f"{name}: {m.sum()} pts, tilt vs global {tilt:.2f} deg, offset {(cr - c) @ n:+.1f} mm")

    # bootstrap the plane -> height uncertainty from the floor alone
    boots = []
    for _ in range(200):
        idx = rng.integers(0, len(Fp), len(Fp)); Q = Fp[idx]; cq = Q.mean(0); nq = np.linalg.svd(Q - cq)[2][2]
        nq = nq if nq[2] > 0 else -nq
        boots.append((cq, nq))

    rows = []
    for i in sorted(S):
        X = S[i]
        h = float((X - c) @ n)
        hb = np.array([(X - cq) @ nq for cq, nq in boots])
        near = Fp[np.linalg.norm(Fp[:, :2] - X[:2], axis=1) < 1000]
        if len(near) >= 30:
            cl, nl, li = plane_fit(near, rng, iters=300, thr=6)
            hl = float((X - cl) @ nl); nl_n = int(li.sum())
            # bootstrap of the local plane -> uncertainty of the local height
            L = near[li]; hs = []
            for _ in range(100):
                q = L[rng.integers(0, len(L), len(L))]; cq = q.mean(0); nq = np.linalg.svd(q - cq)[2][2]
                hs.append(abs((X - cq) @ nq))
            sdl = float(np.std(hs))
        else:
            hl, nl_n, sdl = None, len(near), None
        rows.append(dict(id=i, height_global=h, sd_floor=float(hb.std()), height_local=hl, n_local=nl_n, sd_local=sdl))
    tilt_global = float(np.degrees(np.arccos(abs(n @ np.array([0, 0, 1.0])))))
    c_dji = o + Bx.T @ (c / mm); n_dji = Bx.T @ n           # same plane in DJI model units
    json.dump(dict(plane_c=c.tolist(), plane_n=n.tolist(), plane_c_dji=c_dji.tolist(), plane_n_dji=n_dji.tolist(),
                   n_points=len(Fp), regions=reg,
                   tilt_vs_previous_floor_deg=tilt_global, rows=rows), open(f"{WORK}/floor.json", "w"), indent=1)
    sd_spk = {r["id"]: r.get("sd_boot") for r in C["rows"]}
    with open(f"{OUT_DIR}/speaker_heights.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "bar", "height_mm(global floor)", "floor_plane_sd_mm", "speaker_sd_mm(3D)",
                    "height_mm(local floor within 1 m)", "local_floor_sd_mm", "local_floor_points"])
        for r in rows:
            w.writerow([r["id"], (r["id"] - 1) // 8 + 1, round(r["height_global"], 1), round(r["sd_floor"], 2),
                        "" if sd_spk[r["id"]] is None else round(sd_spk[r["id"]], 2),
                        "" if r["height_local"] is None else round(r["height_local"], 1),
                        "" if r["sd_local"] is None else round(r["sd_local"], 2), r["n_local"]])
    hl_ok = [r for r in rows if r["height_local"] is not None]
    print(f"local-floor heights available for {len(hl_ok)}/112; local minus global: median "
          f"{np.median([r['height_local'] - r['height_global'] for r in hl_ok]):+.1f} mm, "
          f"spread (p10..p90) {np.percentile([r['height_local'] - r['height_global'] for r in hl_ok], 10):+.1f}.."
          f"{np.percentile([r['height_local'] - r['height_global'] for r in hl_ok], 90):+.1f} mm")
    hg = np.array([r["height_global"] for r in rows])
    print(f"new floor vs previous floor: tilt {tilt_global:.2f} deg")
    print(f"heights (global floor): min {hg.min():.0f} median {np.median(hg):.0f} max {hg.max():.0f}")
    for b in range(14):
        v = hg[b * 8:(b + 1) * 8]
        print(f"bar {b+1:2d} ids {b*8+1}-{b*8+8}: mean height {v.mean():.1f} mm")


if __name__ == "__main__":
    main()
