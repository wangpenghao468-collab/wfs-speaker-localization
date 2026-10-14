"""Diaphragm centres in a physical frame, with uncertainty.

Per speaker: DLT initialisation, then robust (Huber) Gauss-Newton on the
reprojection error over all confirmed Sony photos. Uncertainty from the
covariance (pose errors excluded) and from a bootstrap over photos.
Weak speakers are also predicted from their bar (8 collinear speakers),
validated by leave-one-out on well-measured speakers.

Frame: origin on the floor under the array centre, z up (height above floor),
x along the front row (41 -> 72), y = z x x (towards the back row). Units mm.
Output: outputs/.../speaker_centers_mm.csv, work/.../centers.json
"""
import json, csv, numpy as np
from colmap_io import read_points3D
from config import WORK, OUT_DIR, DJI_MODEL

HUBER_PX = 3.0
N_BOOT = 200
MIN_VIEWS, MIN_SPREAD_DEG = 5, 25
MIN_SHARPNESS = 50      # leave the blurriest photos out of the centre estimate
MAX_VIEW_DEG = 75       # nearly edge-on views of a diaphragm are not used for its centre
BAD_PHOTO_FACTOR = 3.0  # a photo whose speakers all reproject > 3x the typical error has a bad pose
BAD_PHOTO_PX = 25.0


def proj(P, X):
    h = P @ np.append(X, 1)
    return h[:2] / h[2]


def huber_threshold(n):
    """Huber threshold from this speaker's own noise level, floored at HUBER_PX.
    2D residual lengths are Rayleigh(sigma): median = sigma*sqrt(2 ln 2), so sigma is
    estimated robustly from the median; the threshold 2.45 sigma is the Rayleigh 95 %
    point, i.e. with Gaussian noise only ~5 % of observations get down-weighted."""
    sigma = np.median(n) / np.sqrt(2 * np.log(2))
    return max(HUBER_PX, 2.45 * sigma)


def solve(obs, Ps):
    """DLT init, then IRLS Gauss-Newton with Huber weights (analytic Jacobian, vectorised)."""
    O = np.asarray(obs, float); P = np.asarray(Ps, float)            # (N,2), (N,3,4)
    A = np.concatenate([O[:, 0:1] * P[:, 2] - P[:, 0], O[:, 1:2] * P[:, 2] - P[:, 1]])
    X = np.linalg.svd(A)[2][-1]
    X = X[:3] / X[3]
    for _ in range(15):
        h = P[:, :, :3] @ X + P[:, :, 3]                               # (N,3)
        e = h[:, :2] / h[:, 2:] - O
        J = (P[:, :2, :3] * h[:, 2, None, None] - h[:, :2, None] * P[:, 2:3, :3]) / h[:, 2, None, None] ** 2
        n = np.linalg.norm(e, axis=1)
        thr = huber_threshold(n)
        wt = np.where(n < thr, 1.0, thr / np.maximum(n, 1e-12))
        Jw = (J * np.sqrt(wt)[:, None, None]).reshape(-1, 3); rw = (e * np.sqrt(wt)[:, None]).ravel()
        dx = np.linalg.lstsq(Jw, -rw, rcond=None)[0]
        X = X + dx
        if np.linalg.norm(dx) < 1e-9:
            break
    h = P[:, :, :3] @ X + P[:, :, 3]
    res = np.linalg.norm(h[:, :2] / h[:, 2:] - O, axis=1)
    sigma = np.sqrt((rw ** 2).sum() / max(len(rw) - 3, 1))
    cov = sigma ** 2 * np.linalg.inv(Jw.T @ Jw)
    return X, res, cov


def floor_frame(est, rng):
    """Floor plane: the Sony-densified floor (floor.json) when available, otherwise
    the DJI sparse points. z up, x along the front row."""
    allX = np.array([e["X"] for e in est.values() if "X" in e])
    ctr = allX.mean(0)
    try:
        fj = json.load(open(f"{WORK}/floor.json"))
        fc, fn = np.array(fj["plane_c_dji"]), np.array(fj["plane_n_dji"])
        fn /= np.linalg.norm(fn)
        if (ctr - fc) @ fn < 0:
            fn = -fn
        xd = est[72]["X"] - est[41]["X"]
        xd -= (xd @ fn) * fn; xd /= np.linalg.norm(xd)
        print("floor: Sony-densified plane from floor.json")
        return ctr - ((ctr - fc) @ fn) * fn, xd, np.cross(fn, xd), fn, 0.0, fj["n_points"]
    except (FileNotFoundError, KeyError):
        print("floor: DJI sparse points (run floor_height.py, then this again, for the densified floor)")
    nrm = np.linalg.svd(allX - ctr)[2][2]
    _, pts = read_points3D(f"{DJI_MODEL}/points3D.bin")
    h = (pts - ctr) @ nrm
    hist, edges = np.histogram(h, bins=400, range=(-3, 3))
    hist[np.abs(edges[:-1]) < 0.3] = 0                    # ignore the speaker level itself
    peak = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
    sel = pts[np.abs(h - peak) < 0.05]
    best = None
    for _ in range(500):
        a, b, c = sel[rng.choice(len(sel), 3, replace=False)]
        nn = np.cross(b - a, c - a)
        if np.linalg.norm(nn) < 1e-9:
            continue
        nn /= np.linalg.norm(nn)
        inl = np.abs((sel - a) @ nn) < 0.01
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), nn, a)
    fl = sel[np.abs((sel - best[2]) @ best[1]) < 0.01]
    fc = fl.mean(0)
    fn = np.linalg.svd(fl - fc)[2][2]
    if (ctr - fc) @ fn < 0:
        fn = -fn
    xd = est[72]["X"] - est[41]["X"]
    xd -= (xd @ fn) * fn; xd /= np.linalg.norm(xd)
    yd = np.cross(fn, xd)
    origin = ctr - ((ctr - fc) @ fn) * fn
    rms = float(np.sqrt((((fl - fc) @ fn) ** 2).mean()))
    return origin, xd, yd, fn, rms, len(fl)


def photo_check(F, poses, Pm):
    """Photos whose pose is off: after a first solve, every speaker seen in such a photo
    reprojects badly in the same photo. Judge per photo (not per observation), by the
    median reprojection error of all its speakers relative to the overall level."""
    per = {}
    for s, v in F["speakers"].items():
        use = [o for o in v["photos"] if (poses[o["photo"]].get("sharpness") or 999) >= MIN_SHARPNESS]
        if len(use) < 3:
            continue
        X, res, _ = solve([np.array(o["xy_und"]) for o in use], [Pm[o["photo"]] for o in use])
        for o, r in zip(use, res):
            per.setdefault(o["photo"], []).append(r)
    med = {n: float(np.median(r)) for n, r in per.items() if len(r) >= 2}
    overall = float(np.median(list(med.values())))
    bad = {n for n, m in med.items() if m > max(BAD_PHOTO_PX, BAD_PHOTO_FACTOR * overall)}
    print(f"photo check: typical photo median reprojection {overall:.1f}px; "
          f"{len(bad)} photos excluded from centre estimation (whole-photo offset)")
    return bad


def main():
    F = json.load(open(f"{WORK}/speakers_final.json"))
    mm = F["cm_per_unit"] * 10
    poses = json.load(open(f"{WORK}/sony_poses_final.json"))
    Pm = {n: np.array(p["K"]) @ np.hstack([np.array(p["R"]), np.array(p["t"])[:, None]]) for n, p in poses.items()}
    C = {n: -np.array(p["R"]).T @ np.array(p["t"]) for n, p in poses.items()}
    rng = np.random.default_rng(0)

    bad_photos = photo_check(F, poses, Pm)
    json.dump(sorted(bad_photos), open(f"{WORK}/photos_excluded_from_centres.json", "w"), indent=1)
    est = {}
    for s, v in F["speakers"].items():
        use = [o for o in v["photos"] if (poses[o["photo"]].get("sharpness") or 999) >= MIN_SHARPNESS
               and o["photo"] not in bad_photos and o["view_angle_deg"] <= MAX_VIEW_DEG]
        v = dict(v, photos=use)
        obs = [np.array(o["xy_und"]) for o in v["photos"]]
        Ps = [Pm[o["photo"]] for o in v["photos"]]
        if len(obs) < 2:
            est[int(s)] = dict(n=len(obs))
            continue
        X, res, cov = solve(obs, Ps)
        rays = np.array([(C[o["photo"]] - X) / np.linalg.norm(C[o["photo"]] - X) for o in v["photos"]])
        spread = float(np.degrees(np.arccos(np.clip(rays @ rays.T, -1, 1))).max())
        boot = []
        for _ in range(N_BOOT if len(obs) >= 4 else 0):
            idx = rng.integers(0, len(obs), len(obs))
            if len(set(idx)) >= 3:
                boot.append(solve([obs[i] for i in idx], [Ps[i] for i in idx])[0])
        b = np.array(boot) * mm if len(boot) > 5 else None
        est[int(s)] = dict(X=X, n=len(obs), spread=spread, reproj=float(np.median(res)),
                           sd_cov=float(np.sqrt(np.trace(cov)) * mm),
                           sd_boot=float(np.sqrt(np.trace(np.cov(b.T)))) if b is not None else None)
    good = {s for s, e in est.items() if "X" in e and e["n"] >= MIN_VIEWS and e["spread"] >= MIN_SPREAD_DEG}

    def bar_fit(members):
        k = np.array([(s - 1) % 8 for s in members], float)
        A = np.stack([np.ones_like(k), k], 1)
        return np.linalg.lstsq(A, np.array([est[s]["X"] for s in members]), rcond=None)[0]

    loo = []                                              # leave-one-out on the bar model
    for b in range(14):
        mem = [s for s in range(b * 8 + 1, b * 8 + 9) if s in good]
        for s in mem:
            rest = [m for m in mem if m != s]
            if len(rest) >= 3:
                c = bar_fit(rest)
                pred = c[0] + ((s - 1) % 8) * c[1]
                inside = min(m % 8 for m in rest) <= s % 8 <= max(m % 8 for m in rest) if s % 8 else False
                gap = min(abs((s - 1) % 8 - (m - 1) % 8) for m in rest)
                loo.append((gap, np.linalg.norm(pred - est[s]["X"]) * mm))
    loo = np.array(loo)
    pitches = []
    for b in range(14):
        mem = [s for s in range(b * 8 + 1, b * 8 + 9) if s in good]
        if len(mem) >= 3:
            c = bar_fit(mem)
            pitches.append(round(float(np.linalg.norm(c[1]) * mm), 1))
            for s in range(b * 8 + 1, b * 8 + 9):
                est[s]["bar_X"] = c[0] + ((s - 1) % 8) * c[1]
                est[s]["bar_from"] = mem

    origin, xd, yd, zd, floor_rms, n_floor = floor_frame(est, rng)
    T = lambda X: np.array([(X - origin) @ xd, (X - origin) @ yd, (X - origin) @ zd]) * mm

    rows = []
    for s in range(1, 113):
        e = est[s]
        tri = T(e["X"]) if "X" in e else None
        bar = T(e["bar_X"]) if "bar_X" in e else None
        use = "triangulated" if s in good else ("bar_model" if bar is not None else "triangulated_weak")
        rows.append(dict(id=s, use=use, xyz=tri if use != "bar_model" else bar, tri=tri, bar=bar, n=e["n"],
                         spread=e.get("spread"), reproj=e.get("reproj"), sd_cov=e.get("sd_cov"), sd_boot=e.get("sd_boot"),
                         bar_from=e.get("bar_from"),
                         diff_bar=float(np.linalg.norm(tri - bar)) if tri is not None and bar is not None else None))

    r1 = lambda a: [round(float(v), 1) for v in a] if a is not None else ["", "", ""]
    with open(f"{OUT_DIR}/speaker_centers_mm.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "bar", "x_mm", "y_mm", "height_z_mm", "source", "n_photos", "view_spread_deg",
                    "median_reproj_px", "sd_bootstrap_mm", "triangulated_x", "triangulated_y", "triangulated_z",
                    "bar_model_x", "bar_model_y", "bar_model_z", "triangulated_vs_bar_mm"])
        for r in rows:
            w.writerow([r["id"], (r["id"] - 1) // 8 + 1, *r1(r["xyz"]), r["use"], r["n"],
                        "" if r["spread"] is None else round(r["spread"]),
                        "" if r["reproj"] is None else round(r["reproj"], 1),
                        "" if r["sd_boot"] is None else round(r["sd_boot"], 2),
                        *r1(r["tri"]), *r1(r["bar"]), "" if r["diff_bar"] is None else round(r["diff_bar"], 1)])
    json.dump(dict(origin=origin.tolist(), x=xd.tolist(), y=yd.tolist(), z=zd.tolist(), mm_per_unit=mm,
                   rows=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in rows]),
              open(f"{WORK}/centers.json", "w"), indent=1)

    G = [r for r in rows if r["use"] == "triangulated"]
    print(f"good speakers {len(G)}/112; floor plane rms {floor_rms * mm:.1f} mm from {n_floor} points")
    print("bootstrap sd (mm): median %.2f, 90%% %.2f" % (np.median([r["sd_boot"] for r in G]), np.percentile([r["sd_boot"] for r in G], 90)))
    print("covariance sd (mm, poses fixed): median %.2f" % np.median([r["sd_cov"] for r in G]))
    print("reprojection median px: %.1f" % np.median([r["reproj"] for r in G]))
    d = [r["diff_bar"] for r in G if r["diff_bar"] is not None]
    print("triangulated vs bar line (good speakers): median %.1f mm, 90%% %.1f mm" % (np.median(d), np.percentile(d, 90)))
    for g in sorted(set(loo[:, 0])):
        e = loo[loo[:, 0] == g, 1]
        print(f"bar leave-one-out, {int(g)} slot(s) from nearest used member: median {np.median(e):.1f} mm, 90% {np.percentile(e, 90):.1f} mm (n={len(e)})")
    print("fitted in-bar pitch per bar (mm):", pitches)
    hz = np.array([r["xyz"][2] for r in rows])
    print("height z (mm): min %.0f median %.0f max %.0f" % (hz.min(), np.median(hz), hz.max()))
    for r in rows:
        if r["use"] != "triangulated":
            print(r["id"], r["use"], "photos", r["n"], "xyz", [round(float(v)) for v in r["xyz"]],
                  "| bar from", r["bar_from"], "| tri vs bar", r["diff_bar"] and round(r["diff_bar"], 1), "mm")


if __name__ == "__main__":
    main()
