"""Sony-only bundle adjustment, initialised from the verified poses.

The Sony COLMAP model folded the mirrored left/right sides because of wrong
cross-side matches. With poses that are now known to be on the right side,
every database match is checked against the epipolar geometry implied by
those poses; only consistent matches are kept. COLMAP then triangulates Sony
3D points with the poses fixed (point_triangulator) and refines everything
together (bundle_adjuster). The result is mapped back into the DJI frame by a
robust Sim3 on camera centres, so the rest of the pipeline is unchanged.

Writes work/ba/ (clean database, COLMAP models) and sony_poses_ba.json.
usage: python ba_sony.py            (all steps)
"""
import json, os, shutil, sqlite3, subprocess, sys, cv2, numpy as np
from colmap_io import read_model, qvec2rotmat
from config import WORK, SONY_IMAGES, COLMAP_EXE, SONY_DB

COLMAP = COLMAP_EXE
DB_SRC = SONY_DB
BA = f"{WORK}/ba"
REFINE_INTRINSICS = "--refine-intrinsics" in sys.argv   # default: lens parameters fixed, poses + points refined
MASK_BARS = "--mask-bars" in sys.argv                   # drop keypoints on the speaker bars (repeated structure)
BAR_MARGIN = 0.7                                        # pitches around each bar's centre line
SAMPSON_PX = 4.0          # a match must agree with the known poses within this (pixels)
MIN_PAIR_INLIERS = 15
PAIR_ID = 2147483647


def rotmat2qvec(R):
    q = np.empty(4); tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1); q[0] = 0.25 / s
        q[1:] = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * s
    else:
        i = np.argmax(np.diag(R)); j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    return q if q[0] >= 0 else -q


def run(args, log):
    env = dict(os.environ, PATH=os.path.dirname(COLMAP) + os.pathsep + os.environ["PATH"],
               QT_PLUGIN_PATH=os.path.join(os.path.dirname(os.path.dirname(COLMAP)), "plugins"))
    with open(log, "w") as fh:
        r = subprocess.run([COLMAP] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        raise SystemExit(f"COLMAP failed: {' '.join(args[:1])}, see {log}")


def umeyama(A, B):
    ma, mb = A.mean(0), B.mean(0); a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a / len(A)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (a ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


def main():
    os.makedirs(BA, exist_ok=True)
    poses = json.load(open(f"{WORK}/sony_poses_final.json"))
    intr = json.load(open(f"{WORK}/sony_intrinsics.json"))
    db_path = f"{BA}/database_clean.db"
    shutil.copy(DB_SRC, db_path)
    db = sqlite3.connect(db_path)
    name2id = {n: i for i, n in db.execute("select image_id, name from images")}
    posed = [n for n in poses if n in name2id]
    print(f"posed photos in database: {len(posed)}")

    # ---- camera groups: one SIMPLE_RADIAL camera per (EXIF focal, image size)
    groups, cam_of = {}, {}
    for n in posed:
        key = (n,) if not REFINE_INTRINSICS else (intr[n]["focal_mm"], intr[n]["width"], intr[n]["height"])
        if key not in groups:
            groups[key] = dict(id=len(groups) + 1, f=[], k1=[], cx=poses[n]["K"][0][2], cy=poses[n]["K"][1][2],
                               w=intr[n]["width"], h=intr[n]["height"])
        groups[key]["f"].append(poses[n]["K"][0][0]); groups[key]["k1"].append(poses[n]["dist"][0])
        cam_of[n] = groups[key]["id"]
    db.execute("delete from cameras")
    for g in groups.values():
        params = np.array([np.median(g["f"]), g["cx"], g["cy"], np.median(g["k1"])], np.float64)
        db.execute("insert into cameras values (?, ?, ?, ?, ?, ?)", (g["id"], 2, g["w"], g["h"], params.tobytes(), 1))
    for n, iid in name2id.items():
        db.execute("update images set camera_id = ? where image_id = ?", (cam_of.get(n, 1), iid))
    print(f"camera groups: {len(groups)}")

    # ---- clean matches with the epipolar geometry of the verified poses
    bar_lines = None
    if MASK_BARS:
        cmu = json.load(open(f"{WORK}/speakers_final.json"))["cm_per_unit"]
        S = {int(k): np.array(v["xyz"]) for k, v in json.load(open(f"{WORK}/speakers_final.json"))["speakers"].items()}
        pitch = 8.0 / cmu
        bar_lines = [(S[b * 8 + 1] - 0.6 * (S[b * 8 + 2] - S[b * 8 + 1]), S[b * 8 + 8] + 0.6 * (S[b * 8 + 8] - S[b * 8 + 7])) for b in range(14)]
    def on_bars(n, xy):
        p = poses[n]; K = np.array(p["K"]); R = np.array(p["R"]); t = np.array(p["t"])
        bad = np.zeros(len(xy), bool)
        for A, B in bar_lines:
            pa, pb = R @ A + t, R @ B + t
            if pa[2] <= 0 or pb[2] <= 0:
                continue
            ua = (K @ pa)[:2] / pa[2]; ub = (K @ pb)[:2] / pb[2]
            wpx = BAR_MARGIN * pitch * K[0, 0] / min(pa[2], pb[2])
            d = ub - ua; L = np.dot(d, d)
            tt = np.clip(((xy - ua) @ d) / max(L, 1e-9), 0, 1)
            bad |= np.linalg.norm(xy - (ua + tt[:, None] * d), axis=1) < wpx
        return bad
    kps = {}
    def keypoints(n):
        if n not in kps:
            iid = name2id[n]
            r, c, data = db.execute("select rows, cols, data from keypoints where image_id = ?", (iid,)).fetchone()
            xy = np.frombuffer(data, np.float32).reshape(r, c)[:, :2].astype(np.float64)
            p = poses[n]; K = np.array(p["K"]); und = cv2.undistortPoints(xy[:, None], K, np.array(p["dist"]))[:, 0]
            kps[n] = (xy, und, K, on_bars(n, xy) if MASK_BARS else np.zeros(len(xy), bool))
        return kps[n]
    rows = db.execute("select pair_id, rows, data from two_view_geometries where rows > 0").fetchall()
    db.execute("delete from two_view_geometries")
    id2name = {i: n for n, i in name2id.items()}
    kept_pairs = kept_m = total_m = 0
    for pid, nr, data in rows:
        a, b = id2name[pid // PAIR_ID], id2name[pid % PAIR_ID]
        total_m += nr
        if a not in poses or b not in poses:
            continue
        m = np.frombuffer(data, np.uint32).reshape(nr, 2)
        _, ua, Ka, ba_ = keypoints(a); _, ub, Kb, bb_ = keypoints(b)
        Ra, ta = np.array(poses[a]["R"]), np.array(poses[a]["t"]); Rb, tb = np.array(poses[b]["R"]), np.array(poses[b]["t"])
        R = Rb @ Ra.T; t = tb - R @ ta
        E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
        x1 = np.hstack([ua[m[:, 0]], np.ones((nr, 1))]); x2 = np.hstack([ub[m[:, 1]], np.ones((nr, 1))])
        Ex1 = x1 @ E.T; Etx2 = x2 @ E
        num = (x2 * Ex1).sum(1) ** 2
        den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
        samp_px = np.sqrt(num / np.maximum(den, 1e-18)) * np.sqrt(Ka[0, 0] * Kb[0, 0])
        ok = (samp_px < SAMPSON_PX) & ~ba_[m[:, 0]] & ~bb_[m[:, 1]]
        if ok.sum() < MIN_PAIR_INLIERS:
            continue
        mm = np.ascontiguousarray(m[ok])
        db.execute("insert into two_view_geometries (pair_id, rows, cols, data, config) values (?, ?, 2, ?, 2)",
                   (pid, int(ok.sum()), mm.tobytes()))
        kept_pairs += 1; kept_m += int(ok.sum())
    db.commit(); db.close()
    print(f"matches: kept {kept_m} of {total_m} in {kept_pairs} of {len(rows)} pairs (rest contradict the verified poses)")

    # ---- input model: verified poses, no points
    inp = f"{BA}/input"; os.makedirs(inp, exist_ok=True)
    with open(f"{inp}/cameras.txt", "w") as fh:
        for g in groups.values():
            fh.write(f"{g['id']} SIMPLE_RADIAL {g['w']} {g['h']} {np.median(g['f'])} {g['cx']} {g['cy']} {np.median(g['k1'])}\n")
    with open(f"{inp}/images.txt", "w") as fh:
        for n in posed:
            q = rotmat2qvec(np.array(poses[n]["R"])); t = poses[n]["t"]
            fh.write(f"{name2id[n]} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cam_of[n]} {n}\n\n")
    open(f"{inp}/points3D.txt", "w").close()

    tri, ba = f"{BA}/triangulated", f"{BA}/adjusted"
    os.makedirs(tri, exist_ok=True); os.makedirs(ba, exist_ok=True)
    print("COLMAP point_triangulator (poses fixed) ...", flush=True)
    run(["point_triangulator", "--database_path", db_path, "--image_path", SONY_IMAGES, "--input_path", inp,
         "--output_path", tri, "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
         "--Mapper.ba_refine_principal_point", "0"], f"{BA}/point_triangulator.log")
    print("COLMAP bundle_adjuster (everything refined) ...", flush=True)
    run(["bundle_adjuster", "--input_path", tri, "--output_path", ba,
         "--BundleAdjustment.refine_focal_length", "1" if REFINE_INTRINSICS else "0",
         "--BundleAdjustment.refine_principal_point", "0",
         "--BundleAdjustment.refine_extra_params", "1" if REFINE_INTRINSICS else "0",
         "--BundleAdjustment.max_num_iterations", "500"],
        f"{BA}/bundle_adjuster.log")

    # ---- back to the DJI frame (robust Sim3 on camera centres), same format as sony_poses_final.json
    cams, imgs = read_model(ba, with_points=False)
    by = {im["name"]: im for im in imgs.values()}
    common = [n for n in posed if n in by]
    A = np.array([by[n]["C"] for n in common])
    Bc = np.array([-np.array(poses[n]["R"]).T @ np.array(poses[n]["t"]) for n in common])
    rng = np.random.default_rng(0); best = None
    for _ in range(2000):
        idx = rng.choice(len(common), 3, replace=False); s, Rs, ts = umeyama(A[idx], Bc[idx])
        inl = np.linalg.norm(s * A @ Rs.T + ts - Bc, axis=1) < 0.05
        if best is None or inl.sum() > best.sum():
            best = inl
    s, Rs, ts = umeyama(A[best], Bc[best])
    move = np.linalg.norm(s * A @ Rs.T + ts - Bc, axis=1)
    out = {}
    for n in common:
        im = by[n]; cam = cams[im["camera_id"]]; f_, cx, cy, k1 = cam["params"]
        R = im["R"] @ Rs.T; t = s * im["t"] - R @ ts
        out[n] = dict(poses[n], K=[[f_, 0, cx], [0, f_, cy], [0, 0, 1]], dist=[k1, 0, 0, 0, 0],
                      R=R.tolist(), t=t.tolist(), source=poses[n]["source"] + "+ba",
                      ba_moved_dji_units=float(move[common.index(n)]))
    json.dump(out, open(f"{WORK}/sony_poses_ba.json", "w"), indent=1)
    rep = [l for l in open(f"{BA}/bundle_adjuster.log", errors="ignore") if "Final cost" in l or "Initial cost" in l]
    print("BA cost:", " | ".join(r.strip()[-60:] for r in rep[-2:]))
    print(f"registered after BA: {len(common)} of {len(posed)}; camera centres moved by BA: median "
          f"{np.median(move):.4f}, 90% {np.percentile(move, 90):.4f} DJI units; outside 0.05: {int((~best).sum())}")


if __name__ == "__main__":
    main()
