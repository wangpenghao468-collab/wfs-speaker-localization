"""Final Sony poses in the DJI frame, for every selected photo.

Each photo has a direct PnP pose in the DJI frame, but close-ups cannot tell the
mirrored left/right sides apart, so some direct poses land on the wrong side.
Every photo therefore collects pose hypotheses and votes:
  * its own direct pose;
  * one per neighbour with a decided pose: rotation = relative rotation x the
    neighbour's rotation, position solved from this photo's own 2D-3D
    correspondences. Relative rotations come from the Sony model (registered
    pairs sharing >= 40 points) or from the COLMAP database essential matrix
    (any pair with >= 60 verified matches).
Agreeing hypotheses are grouped; the largest group wins if it clearly leads.
User corrections (speaker_corrections.json) are hard constraints.
Output: sony_poses_final.json
"""
import json, sqlite3, cv2, numpy as np
from colmap_io import read_model
from config import SONY_MODEL, WORK, SONY_DB
from eval_poses import load, evaluate, project
from corrections import load_corrections

DB = SONY_DB
MIN_COVIS = 40          # Sony-model shared 3D points
MIN_MATCHES = 60        # database verified matches
C_TOL = 0.15            # DJI units (~5 cm): hypotheses agreeing in camera centre
R_TOL = 4.0             # degrees
PASSES = 4


def rot_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))


def centre(R, t):
    return -R.T @ t


class Photo:
    def __init__(self, n, intr):
        self.n = n
        self.K, self.dist = np.array(intr["K"]), np.array(intr["dist"])
        z = np.load(f"{WORK}/corr/{n}.npz")
        self.p2, self.p3 = z["p2"], z["p3"]
        und = cv2.undistortPoints(self.p2[:, None].astype(np.float64), self.K, self.dist)[:, 0] if len(self.p2) else np.zeros((0, 2))
        self.rays = np.hstack([und, np.ones((len(und), 1))])

    def solve_t(self, R, rng, thr=20.0):
        """Position for a fixed rotation: RANSAC over 2-point linear solves, then LM refine."""
        if len(self.p3) < 8:
            return None
        RX = self.p3 @ R.T

        def lin(idx):
            A, b = [], []
            for i in idx:
                r = self.rays[i]
                rx = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
                A.append(rx); b.append(-rx @ RX[i])
            return np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]

        rv = cv2.Rodrigues(R)[0]; best = None
        for _ in range(300):
            t = lin(rng.choice(len(RX), 2, replace=False))
            pr = cv2.projectPoints(self.p3, rv, t, self.K, self.dist)[0][:, 0]
            inl = np.linalg.norm(pr - self.p2, axis=1) < thr
            if best is None or inl.sum() > best.sum():
                best = inl
        if best.sum() < 12:
            return None
        return self.refine(R, lin(np.where(best)[0]))

    def refine(self, R, t, thr=25.0):
        rv = cv2.Rodrigues(R)[0]; tv = np.asarray(t, float).reshape(3, 1).copy(); n_in = 0
        for th in (thr * 2, thr, thr / 2):
            pr = cv2.projectPoints(self.p3, rv, tv, self.K, self.dist)[0][:, 0]
            inl = np.linalg.norm(pr - self.p2, axis=1) < th
            if inl.sum() < 12:
                break
            rv, tv = cv2.solvePnPRefineLM(self.p3[inl], self.p2[inl], self.K, self.dist, rv, tv)
            n_in = int(inl.sum())
        if n_in < 12:
            return None
        return cv2.Rodrigues(rv)[0], tv.ravel(), n_in


def main():
    ids, S, nrm, pitch = load()
    intr = json.load(open(f"{WORK}/sony_intrinsics.json"))
    names = sorted(intr)
    photos = {n: Photo(n, intr[n]) for n in names}
    rng = np.random.default_rng(0)

    # ---- relative rotations between neighbours: R_n = rel[(m, n)] @ R_m
    rel = {}
    _, simgs = read_model(SONY_MODEL)
    sony = {im["name"]: im for im in simgs.values()}
    pts = {n: set(im["pids"][im["pids"] >= 0].tolist()) for n, im in sony.items()}
    sn = sorted(sony)
    for i, a in enumerate(sn):
        for b in sn[i + 1:]:
            if len(pts[a] & pts[b]) >= MIN_COVIS:
                rel[(a, b)] = [sony[b]["R"] @ sony[a]["R"].T]
                rel[(b, a)] = [rel[(a, b)][0].T]
    db = sqlite3.connect(DB)
    idn = {i: n for i, n in db.execute("select image_id, name from images")}
    n_db = 0
    for pid, rows, E in db.execute("select pair_id, rows, E from two_view_geometries where rows >= ?", (MIN_MATCHES,)):
        a, b = idn[pid // 2147483647], idn[pid % 2147483647]
        if (a, b) in rel or a not in intr or b not in intr or E is None:
            continue
        R1, R2, _ = cv2.decomposeEssentialMat(np.frombuffer(E, np.float64).reshape(3, 3))
        rel[(a, b)] = [R1, R2]; rel[(b, a)] = [R1.T, R2.T]; n_db += 1
    nbrs = {n: [] for n in names}
    for a, b in rel:
        nbrs[b].append(a)
    print(f"neighbour pairs: {len(rel) // 2} ({n_db} from database only)")

    fixes = load_corrections()
    direct = json.load(open(f"{WORK}/sony_in_dji.json"))
    det = json.load(open(f"{WORK}/detections.json"))

    def passes_check(R, t, n):
        p = dict(R=R, t=t, K=photos[n].K, width=intr[n]["width"], height=intr[n]["height"])
        r = evaluate(p, ids, S, nrm, pitch, det.get(n, []))
        e = [x for _, x, _ in r if x < 0.3]
        return len(e) >= 3 and np.median(e) < 0.15

    def satisfies(R, t, n, labels):
        K, dist = photos[n].K, photos[n].dist
        for c in labels:
            xy = cv2.undistortPoints(np.array([[[c["x_px"], c["y_px"]]]], float), K, dist, P=K)[0, 0]
            uv, z = project(dict(R=R, t=t, K=K), S[c["correct_id"]][None])
            if z[0] <= 0 or np.linalg.norm(uv[0] - xy) > 0.35 * pitch * K[0, 0] / z[0]:
                return False
        return True

    ok = {}
    for n in names:
        p = direct.get(n, {})
        if "R" in p and passes_check(np.array(p["R"]), np.array(p["t"]), n):
            ok[n] = (np.array(p["R"]), np.array(p["t"]))
    print("direct poses passing detection check:", len(ok), "of", len(names))

    decided, cache = {}, {}
    for it in range(PASSES):
        sources = {**ok, **{n: (d["R"], d["t"]) for n, d in decided.items()}}
        new = {}
        for n in names:
            if any(c.get("exclude") for c in fixes.get(n, [])):
                continue
            hyps = [("self", ok[n])] if n in ok else []
            for m in nbrs[n]:
                if m not in sources:
                    continue
                key = (m, n, np.round(sources[m][0], 6).tobytes() + np.round(sources[m][1], 6).tobytes())
                if key not in cache:
                    cache[key] = None
                    best = None
                    for Rr in rel[(m, n)]:
                        sol = photos[n].solve_t(Rr @ sources[m][0], rng)
                        if sol and (best is None or sol[2] > best[2]):
                            best = sol
                    if best and passes_check(best[0], best[1], n):
                        cache[key] = (best[0], best[1])
                if cache[key] is not None:
                    hyps.append((m, cache[key]))
            labels = [c for c in fixes.get(n, []) if c.get("correct_id")]
            if labels:
                hyps = [h for h in hyps if satisfies(h[1][0], h[1][1], n, labels)]
            if not hyps:
                continue
            groups = []
            for h in hyps:
                for g in groups:
                    if np.linalg.norm(centre(*h[1]) - centre(*g[0][1])) < C_TOL and rot_deg(h[1][0], g[0][1][0]) < R_TOL:
                        g.append(h); break
                else:
                    groups.append([h])
            groups.sort(key=len, reverse=True)
            top, second = len(groups[0]), len(groups[1]) if len(groups) > 1 else 0
            if labels:
                src = "user_corrected"
            elif len(hyps) == 1:
                if hyps[0][0] != "self":
                    continue              # a single neighbour is not enough evidence
                src = "direct_only"
            elif top >= 2 and top >= 2 * second:
                src = "direct_confirmed" if any(h[0] == "self" for h in groups[0]) else "neighbour_votes"
            else:
                continue                  # ambiguous: leave the photo out
            best = next((h for h in groups[0] if h[0] == "self"), groups[0][0])
            new[n] = dict(R=best[1][0], t=best[1][1], source=src, votes=f"{top}/{len(hyps)}",
                          rejected_self=n in ok and not any(h[0] == "self" for h in groups[0]))
        decided = new
        print(f"pass {it + 1}: decided {len(decided)}", flush=True)

    out, stats = {}, {}
    for n, d in decided.items():
        ph = photos[n]
        r = ph.refine(np.array(d["R"]), np.array(d["t"]))
        if r is None:
            continue
        out[n] = dict(K=ph.K.tolist(), dist=ph.dist.tolist(), width=intr[n]["width"], height=intr[n]["height"],
                      R=r[0].tolist(), t=r[1].tolist(), source=d["source"], votes=d["votes"], refine_inliers=r[2],
                      registered_in_sony_model=intr[n]["registered"], sharpness=intr[n]["sharpness"])
        stats[d["source"]] = stats.get(d["source"], 0) + 1
        stats["self_rejected"] = stats.get("self_rejected", 0) + int(d["rejected_self"])
    print(stats, "final poses:", len(out), "of", len(names),
          f"({sum(v['registered_in_sony_model'] for v in out.values())} registered, "
          f"{sum(not v['registered_in_sony_model'] for v in out.values())} unregistered)")
    json.dump(out, open(f"{WORK}/sony_poses_final.json", "w"), indent=1)


if __name__ == "__main__":
    main()
