"""Robust DJI-frame position for all 112 speakers using the bar model:
each bar holds 8 collinear speakers at a fixed in-bar pitch (8 cm).
Output: dji_bar_model.json
"""
import json, numpy as np
from config import WORK

BAR = 8
IN_BAR_CM = 8.0


def main():
    S = {int(k): v for k, v in json.load(open(f"{WORK}/dji_speakers.json")).items()}
    X = {k: np.array(v["xyz"]) for k, v in S.items()}
    good = {k for k, v in S.items() if v["method"] == "single_view_depth" or v["err_px"] < 10}
    # global in-bar pitch from adjacent reliable multi-view pairs inside a bar
    d = [np.linalg.norm(X[k + 1] - X[k]) for k in good if k + 1 in good and k % BAR != 0
         and S[k]["method"] == "multi_view" and S[k + 1]["method"] == "multi_view"]
    pitch = float(np.median(d))
    out, report = {}, []
    for b in range(112 // BAR):
        ids = [b * BAR + i + 1 for i in range(BAR)]
        pts = [(i, X[s]) for i, s in enumerate(ids) if s in good]
        if len(pts) < 2:
            report.append((b, "too few"))
            continue
        best = None
        # RANSAC over pairs: line through two members with known pitch
        for a in range(len(pts)):
            for c in range(a + 1, len(pts)):
                (ia, xa), (ic, xc) = pts[a], pts[c]
                u = (xc - xa) / (ic - ia)
                if abs(np.linalg.norm(u) - pitch) > 0.15 * pitch:
                    continue
                p0 = xa - ia * u
                inl = [(i, x) for i, x in pts if np.linalg.norm(p0 + i * u - x) < 0.3 * pitch]
                if best is None or len(inl) > len(best):
                    best = inl
        if best is None or len(best) < 2:
            report.append((b, "no consistent pair"))
            continue
        I = np.array([i for i, _ in best], float); P = np.array([x for _, x in best])
        A = np.stack([np.ones_like(I), I], 1)
        coef = np.linalg.lstsq(A, P, rcond=None)[0]          # p0, u
        res = np.linalg.norm(A @ coef - P, axis=1)
        outliers = sorted(set(s for s in ids if s in good) - {ids[int(i)] for i in I})
        for i, s in enumerate(ids):
            out[s] = dict(xyz=(coef[0] + i * coef[1]).tolist(), bar=b + 1, annotated=s in X,
                          used=s in {ids[int(j)] for j in I}, outlier=s in outliers)
        report.append((b, f"{len(best)}/8 used, pitch {np.linalg.norm(coef[1]):.3f}, rms {np.sqrt((res**2).mean()):.4f}, outliers {outliers}"))
    json.dump(dict(pitch=pitch, cm_per_unit=IN_BAR_CM / pitch, speakers={str(k): v for k, v in out.items()}),
              open(f"{WORK}/dji_bar_model.json", "w"), indent=1)
    print(f"in-bar pitch {pitch:.4f} DJI units -> {IN_BAR_CM / pitch:.2f} cm/unit")
    for b, r in report:
        print(f"bar {b+1:2d} (ids {b*BAR+1}-{b*BAR+BAR}): {r}")
    print("speakers with position:", len(out))


if __name__ == "__main__":
    main()
