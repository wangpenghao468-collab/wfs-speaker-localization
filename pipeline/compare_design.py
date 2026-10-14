"""Compare computed diaphragm centres with the designed layout (Panoramix export).

The export lists each speaker as azimuth / elevation / distance (aed) around the
designer's origin (the listening centre). Our photogrammetry frame has its own
origin, so the two are related by an unknown rigid motion; it is fitted on all
speakers (rotation about the vertical + translation, no scale) and the residual
per speaker is the deviation between design/manual layout and measurement.
Output: design_vs_measured.csv + .json (work dir)
"""
import json, csv, sys, numpy as np
from config import WORK, OUT_DIR, DESIGN_TXT

AED = DESIGN_TXT


def load_design(path=AED):
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("/bus/10/speakers/aed "):
            v = np.array(line.split()[1:], float).reshape(-1, 3)
            return v                                    # az deg, el deg, dist m
    raise SystemExit("no /bus/10/speakers/aed line")


def aed_to_xyz(aed, az_sign):
    az = np.radians(aed[:, 0]) * az_sign; el = np.radians(aed[:, 1]); d = aed[:, 2] * 1000
    # Spat convention: 0 deg = front (+y), x to the side, z up
    return np.stack([d * np.cos(el) * np.sin(az), d * np.cos(el) * np.cos(az), d * np.sin(el)], 1)


def fit_rigid_2d(P, Q, scale=False):
    """Q ~ s R P + t in the horizontal plane (z handled separately)."""
    pm, qm = P.mean(0), Q.mean(0); a, b = P - pm, Q - qm
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum() if scale else 1.0
    return s, R, qm - s * R @ pm


def main():
    design = load_design()
    meas = {r["id"]: r for r in json.load(open(f"{WORK}/centers.json"))["rows"]}
    ids = list(range(1, len(design) + 1))
    M = np.array([meas[i]["xyz"] for i in ids])         # measured, our floor frame, mm
    best = None
    for sgn in (1, -1):
        Dz = aed_to_xyz(design, sgn)
        s, R, t = fit_rigid_2d(M[:, :2], Dz[:, :2])
        res = np.linalg.norm((M[:, :2] @ R.T + t) - Dz[:, :2], axis=1)
        if best is None or np.median(res) < best[0]:
            best = (np.median(res), sgn, R, t, Dz)
    _, sgn, R, t, Dz = best
    Mh = M[:, :2] @ R.T + t                             # measured, in the design frame
    s_fit = fit_rigid_2d(M[:, :2], Dz[:, :2], scale=True)[0]
    dxy = Mh - Dz[:, :2]
    hz = M[:, 2]; dz = hz - np.median(hz)              # design puts every speaker at one height
    # radial / tangential split w.r.t. the design origin (listener)
    rad = Dz[:, :2] / np.linalg.norm(Dz[:, :2], axis=1, keepdims=True)
    d_rad = (dxy * rad).sum(1); d_tan = dxy[:, 0] * rad[:, 1] - dxy[:, 1] * rad[:, 0]
    d_az = np.degrees(np.arctan2(Mh[:, 0], Mh[:, 1]) - np.arctan2(Dz[:, 0], Dz[:, 1])) * sgn
    d_az = (d_az + 180) % 360 - 180
    rows = []
    for k, i in enumerate(ids):
        rows.append(dict(id=i, design_xy=Dz[k, :2].tolist(), measured_xy=Mh[k].tolist(), dx=float(dxy[k, 0]),
                         dy=float(dxy[k, 1]), dxy=float(np.linalg.norm(dxy[k])), d_radial=float(d_rad[k]),
                         d_tangential=float(d_tan[k]), d_az_deg=float(d_az[k]), height=float(hz[k]), dz=float(dz[k]),
                         design_az=float(design[k, 0]), design_dist_mm=float(design[k, 2] * 1000),
                         measured_dist_mm=float(np.linalg.norm(Mh[k])), source=meas[i]["use"]))
    rot_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    summary = dict(az_sign=sgn, rotation_deg=rot_deg, translation_mm=t.tolist(), scale_if_free=float(s_fit),
                   median_dxy=float(np.median(np.linalg.norm(dxy, axis=1))),
                   p90_dxy=float(np.percentile(np.linalg.norm(dxy, axis=1), 90)),
                   max_dxy=float(np.linalg.norm(dxy, axis=1).max()), height_median=float(np.median(hz)),
                   height_range=[float(hz.min()), float(hz.max())])
    json.dump(dict(summary=summary, rows=rows), open(f"{WORK}/design_vs_measured.json", "w"), indent=1)
    with open(f"{OUT_DIR}/design_vs_measured.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "design_x_mm", "design_y_mm", "measured_x_mm", "measured_y_mm", "dx_mm", "dy_mm",
                    "horizontal_deviation_mm", "radial_mm(+ = farther)", "tangential_mm", "azimuth_dev_deg",
                    "design_distance_mm", "measured_distance_mm", "height_mm", "height_dev_from_median_mm", "measured_source"])
        for r in rows:
            w.writerow([r["id"], *[round(v, 1) for v in r["design_xy"]], *[round(v, 1) for v in r["measured_xy"]],
                        round(r["dx"], 1), round(r["dy"], 1), round(r["dxy"], 1), round(r["d_radial"], 1),
                        round(r["d_tangential"], 1), round(r["d_az_deg"], 2), round(r["design_dist_mm"]),
                        round(r["measured_dist_mm"]), round(r["height"], 1), round(r["dz"], 1), r["source"]])
    print(json.dumps(summary, indent=1))
    worst = sorted(rows, key=lambda r: -r["dxy"])[:10]
    print("largest deviations:", [(r["id"], round(r["dxy"]), round(r["d_radial"]), round(r["d_tangential"])) for r in worst])


if __name__ == "__main__":
    main()
