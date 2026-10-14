"""Triangulate annotated speakers in the DJI model (identity source of truth)."""
import json, numpy as np, collections
from colmap_io import read_model, read_points3D
from config import DJI_MODEL, WORK
from annotations import load_annotations


def main():
    cams, imgs = read_model(DJI_MODEL)
    cam = cams[1]; f, _, cx, cy = cam["params"]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    by = {im["name"]: im for im in imgs.values()}
    pid, pxyz = read_points3D(f"{DJI_MODEL}/points3D.bin"); P = dict(zip(pid, range(len(pid))))
    ann = load_annotations()
    obs = collections.defaultdict(list)
    for (img, sid), a in ann.items():
        if img in by:
            obs[sid].append((img, np.array(a["xy"]), a["r"]))
    out = {}
    for sid, ob in sorted(obs.items()):
        if len(ob) >= 2:
            A = []
            for n, (x, y), _ in ob:
                im = by[n]; M = K @ np.hstack([im["R"], im["t"][:, None]])
                A += [x * M[2] - M[0], y * M[2] - M[1]]
            X = np.linalg.svd(np.array(A))[2][-1]; X = X[:3] / X[3]
            errs = []
            for n, xy, _ in ob:
                im = by[n]; h = K @ (im["R"] @ X + im["t"]); errs.append(np.linalg.norm(h[:2] / h[2] - xy))
            out[sid] = dict(xyz=X.tolist(), method="multi_view", n=len(ob), err_px=float(np.max(errs)))
        else:
            # single view: depth from sparse points observed near the annotation
            n, xy, r = ob[0]; im = by[n]
            m = im["pids"] >= 0
            near = m & (np.linalg.norm(im["xy"] - xy, axis=1) < 2.5 * max(r, 8))
            if near.sum() < 2:
                continue
            Xs = pxyz[[P[p] for p in im["pids"][near]]]
            z = np.median((Xs @ im["R"].T + im["t"])[:, 2])
            ray = np.linalg.inv(K) @ np.array([xy[0], xy[1], 1.0])
            X = im["R"].T @ (ray * z - im["t"])
            out[sid] = dict(xyz=X.tolist(), method="single_view_depth", n=1, n_depth=int(near.sum()))
    json.dump(out, open(f"{WORK}/dji_speakers.json", "w"), indent=1)
    ids = sorted(out)
    X = {i: np.array(out[i]["xyz"]) for i in ids}
    for i in ids:
        if i + 1 in X:
            print(f"{i:3d}->{i+1:3d} {np.linalg.norm(X[i+1]-X[i]):.3f} {out[i]['method'][:5]} n={out[i]['n']}"
                  + (f" err={out[i]['err_px']:.1f}" if 'err_px' in out[i] else ''))


if __name__ == "__main__":
    main()
