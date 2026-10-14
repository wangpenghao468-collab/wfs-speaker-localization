"""Localise every selected Sony photo directly in the DJI COLMAP frame.

The Sony model folds the mirrored left/right sides, so its poses are not used.
Each Sony photo is matched against DJI frames; DJI keypoints are lifted to DJI
3D points via the DJI COLMAP tracks; PnP-RANSAC with the Sony intrinsics gives
the pose in the DJI frame.
Output: sony_in_dji.json
"""
import cv2, numpy as np, torch, json, os
from colmap_io import read_model, read_points3D
from common import mutual_ratio
from config import DJI_MODEL, DJI_IMAGES, SONY_IMAGES, WORK

sift = cv2.SIFT_create(nfeatures=15000, contrastThreshold=0.02)
SONY_W = 3000
TOP_DJI = 8


def feats(gray):
    kp, d = sift.detectAndCompute(gray, None)
    d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7))
    return np.array([k.pt for k in kp], np.float32), d.astype(np.float16)


def cached(path, fn):
    if os.path.exists(path):
        z = np.load(path); return z["xy"], z["desc"]
    xy, d = fn(); np.savez(path, xy=xy, desc=d); return xy, d


def main():
    _, dimgs = read_model(DJI_MODEL)
    dcam = read_model(DJI_MODEL, with_points=False)[0][1]
    pid, pxyz = read_points3D(f"{DJI_MODEL}/points3D.bin"); P = dict(zip(pid, range(len(pid))))
    os.makedirs(f"{WORK}/feats2", exist_ok=True)
    os.makedirs(f"{WORK}/corr", exist_ok=True)

    dj = []
    for im in sorted(dimgs.values(), key=lambda x: x["name"]):
        n = im["name"]
        xy, d = cached(f"{WORK}/feats2/dji_{n}.npz", lambda: feats(cv2.imread(f"{DJI_IMAGES}/{n}", 0)))
        m = im["pids"] >= 0
        cxy, cp = torch.tensor(im["xy"][m], dtype=torch.float32).cuda(), im["pids"][m]
        dist, j = torch.cdist(torch.tensor(xy).cuda(), cxy).min(1)
        ok = (dist < 2.0).cpu().numpy()
        X = np.full((len(xy), 3), np.nan); X[ok] = pxyz[[P[p] for p in cp[j.cpu().numpy()[ok]]]]
        dj.append((n, torch.tensor(d).cuda(), X, torch.tensor(d[ok]).cuda(), np.where(ok)[0]))
    print("DJI features ready", flush=True)

    out = {}
    intr = json.load(open(f"{WORK}/sony_intrinsics.json"))
    for k, n in enumerate(sorted(intr)):
        K, dist = np.array(intr[n]["K"]), np.array(intr[n]["dist"])

        def f():
            g = cv2.imread(f"{SONY_IMAGES}/{n}", 0); sc = SONY_W / g.shape[1]
            xy, d = feats(cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))
            return xy / sc, d
        sxy, sd = cached(f"{WORK}/feats2/sony_{n}.npz", f)
        D = torch.tensor(sd).cuda()
        cand = []
        for dn, dd, X, dd3, idx3 in dj:
            q, t = mutual_ratio(D, dd3)          # match only DJI keypoints that have 3D
            cand.append((len(q), dn, q, X[idx3[t]]))
        cand.sort(key=lambda c: -c[0])
        p2 = np.ascontiguousarray(np.concatenate([sxy[c[2]] for c in cand[:TOP_DJI]]), dtype=np.float64)
        p3 = np.ascontiguousarray(np.concatenate([c[3] for c in cand[:TOP_DJI]]), dtype=np.float64)
        res = dict(top_dji=[(c[1], int(c[0])) for c in cand[:TOP_DJI]], n_corr=int(len(p2)))
        p3 = p3.reshape(-1, 3); p2 = p2.reshape(-1, 2)
        np.savez(f"{WORK}/corr/{n}.npz", p2=p2, p3=p3)
        if len(p2) >= 12:
            ok, inl = False, None
            for flag in (cv2.SOLVEPNP_AP3P, cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP):
                try:   # OpenCV 5 occasionally throws inside a minimal solver
                    ok, rv, tv, inl = cv2.solvePnPRansac(p3, p2, K, dist, reprojectionError=20.0,
                                                         iterationsCount=20000, confidence=0.9999, flags=flag)
                    break
                except cv2.error:
                    continue
            if ok and inl is not None and len(inl) >= 12:
                inl = inl.ravel()
                rv, tv = cv2.solvePnPRefineLM(p3[inl], p2[inl], K, dist, rv, tv)
                R = cv2.Rodrigues(rv)[0]
                pr = cv2.projectPoints(p3[inl], rv, tv, K, dist)[0][:, 0]
                res.update(R=R.tolist(), t=tv.ravel().tolist(), K=K.tolist(), dist=dist.tolist(),
                           width=intr[n]["width"], height=intr[n]["height"],
                           inliers=int(len(inl)), inlier_ratio=round(len(inl) / len(p2), 3),
                           reproj_px=float(np.median(np.linalg.norm(pr - p2[inl], axis=1))))
        out[n] = res
        print(f"[{k+1}/{len(intr)}] {n} corr {len(p2)} inliers {res.get('inliers', 0)} top {cand[0][1]}", flush=True)
    json.dump(out, open(f"{WORK}/sony_in_dji.json", "w"), indent=1)


if __name__ == "__main__":
    main()
