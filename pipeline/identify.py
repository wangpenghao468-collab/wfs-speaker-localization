"""Speaker -> Sony photos.

A photo lists a speaker when, with the photo's DJI-frame pose:
  * the speaker faces the camera and the whole speaker (±0.45 pitch) is in frame,
  * a detected diaphragm lies within 0.25 pitch of the projected centre.
Speaker centres are then re-triangulated from those Sony detections and the
assignment repeated once with the refined centres.
Output: speakers_final.json
"""
import json, numpy as np
from eval_poses import load, project
from config import WORK
from corrections import load_corrections

GATE = (0.30, 0.25)       # match radius in pitches, per pass
MARGIN = 0.45             # half speaker size in pitches (complete speaker in frame)
R_RANGE = (0.11, 0.30)    # diaphragm radius / pitch (median 0.176) — rejects screws and glints


def triangulate(obs, poses):
    A = []
    for n, (x, y) in obs:
        p = poses[n]; K = np.array(p["K"]); M = K @ np.hstack([np.array(p["R"]), np.array(p["t"])[:, None]])
        A += [x * M[2] - M[0], y * M[2] - M[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def assign(X, sid, poses, det, nrm, pitch, gate, not_speaker=None):
    out = []
    for n, p in poses.items():
        R, t = np.array(p["R"]), np.array(p["t"]); C = -R.T @ t
        if (C - X) @ nrm[sid] <= 0:
            continue
        uv, z = project(p, X[None]); uv, z = uv[0], z[0]
        if z <= 0:
            continue
        ppx = pitch * p["K"][0][0] / z; m = MARGIN * ppx
        if not (m < uv[0] < p["width"] - m and m < uv[1] < p["height"] - m):
            continue
        ds = det.get(n, [])
        if not ds:
            continue
        xy = np.array([d["xy_und"] for d in ds]); e = np.linalg.norm(xy - uv, axis=1)
        size = np.array([d["r"] for d in ds]) / ppx
        for x, y in (not_speaker or {}).get(n, []):          # user said "this is not a speaker"
            e[np.hypot(np.array([d["xy"][0] for d in ds]) - x, np.array([d["xy"][1] for d in ds]) - y) < 0.3 * ppx] = np.inf
        e[(size < R_RANGE[0]) | (size > R_RANGE[1])] = np.inf
        j = int(e.argmin())
        if e[j] < gate * ppx:
            view = (C - X) / np.linalg.norm(C - X)
            out.append(dict(photo=n, j=j, xy_und=xy[j].tolist(), xy=ds[j]["xy"], r_px=ds[j]["r"], pitch_px=ppx,
                            err_pitch=float(e[j] / ppx), view_angle_deg=float(np.degrees(np.arccos(np.clip(view @ nrm[sid], -1, 1))))))
    return out


def main():
    ids, S, nrm, pitch = load()
    poses = json.load(open(f"{WORK}/sony_poses_final.json")); det = json.load(open(f"{WORK}/detections.json"))
    cm_per_unit = json.load(open(f"{WORK}/dji_bar_model.json"))["cm_per_unit"]
    not_speaker = {n: [(c["x_px"], c["y_px"]) for c in cs if c.get("correct_id") == 0]
                   for n, cs in load_corrections(verbose=False).items()}
    result = {}
    for sid in ids:
        X = S[sid]; seed = X.copy(); obs = []
        for gate in GATE:
            obs = assign(X, sid, poses, det, nrm, pitch, gate, not_speaker)
            if len(obs) >= 2:
                X = triangulate([(o["photo"], o["xy_und"]) for o in obs], poses)
        errs = []
        for o in obs:
            uv, _ = project(poses[o["photo"]], X[None]); errs.append(float(np.linalg.norm(uv[0] - o["xy_und"])))
            o["reproj_px"] = errs[-1]
        result[sid] = dict(xyz=X.tolist(), xyz_mm=(X * cm_per_unit * 10).tolist(), seed_xyz=seed.tolist(),
                           moved_mm=float(np.linalg.norm(X - seed) * cm_per_unit * 10), n_photos=len(obs),
                           reproj_px_median=float(np.median(errs)) if errs else None, photos=obs)
    json.dump(dict(cm_per_unit=cm_per_unit, pitch=pitch, speakers={str(k): v for k, v in result.items()}),
              open(f"{WORK}/speakers_final.json", "w"), indent=1)
    n = [result[s]["n_photos"] for s in ids]
    print("photos per speaker: min %d median %d max %d ; speakers with 0 photos: %s"
          % (min(n), np.median(n), max(n), [s for s in ids if result[s]["n_photos"] == 0]))
    print("speakers with <3 photos:", [s for s in ids if result[s]["n_photos"] < 3])
    mv = [result[s]["moved_mm"] for s in ids if result[s]["n_photos"] >= 2]
    print("refinement moved centres by median %.1f mm (90%% %.1f mm)" % (np.median(mv), np.percentile(mv, 90)))
    # in-bar spacing check after refinement
    d = [np.linalg.norm(np.array(result[s + 1]["xyz_mm"]) - np.array(result[s]["xyz_mm"]))
         for s in ids if s % 8 and s + 1 in result and result[s]["n_photos"] >= 3 and result[s + 1]["n_photos"] >= 3]
    print("in-bar neighbour spacing: median %.1f mm, std %.1f mm, n=%d (expected 80)" % (np.median(d), np.std(d), len(d)))
    db = [np.linalg.norm(np.array(result[s + 1]["xyz_mm"]) - np.array(result[s]["xyz_mm"]))
          for s in ids if s % 8 == 0 and s + 1 in result and (s // 8) % 4 != 0 and result[s]["n_photos"] >= 3 and result[s + 1]["n_photos"] >= 3]
    print("between-bar spacing:", [round(x, 1) for x in db])


if __name__ == "__main__":
    main()
