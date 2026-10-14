"""Export measured speaker positions and orientations for Panoramix (WFS bus 10).

Positions: measured centres, mapped into the designer's frame (Panoramix origin)
by the rigid horizontal fit from compare_design.py. Elevation stays 0 because the
bus has /bus/10/ignorez 1 (2.5D WFS); heights are reported separately.
Orientations: each 8-speaker bar is a rigid straight bar, so its facing direction
is the horizontal normal of the line through its 8 measured centres; the sign
follows the designed direction vector of that speaker.
Writes:
  outputs/.../panoramix_bus10_measured.txt  (the two OSC lines to paste)
  outputs/.../1012_measured.txt             (copy of 1012.txt with those lines replaced)
  outputs/.../panoramix_bus10_measured.csv  (per-speaker table)
"""
import json, csv, re, numpy as np
from config import WORK, OUT_DIR
from compare_design import AED, load_design

DIR_KEY = "/bus/10/speakers/directions/xyz"
AED_KEY = "/bus/10/speakers/aed"


def load_dirs(path=AED):
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith(DIR_KEY + " "):
            return np.array(line.split()[1:], float).reshape(-1, 3)
    raise SystemExit("no directions line")


def main():
    cmp_ = json.load(open(f"{WORK}/design_vs_measured.json"))
    rows = {r["id"]: r for r in cmp_["rows"]}
    ids = sorted(rows)
    M = np.array([rows[i]["measured_xy"] for i in ids]) / 1000.0        # metres, designer frame
    design_aed = load_design(); design_dir = load_dirs()

    normals = np.zeros((len(ids), 2)); line_rms = {}
    for b in range(len(ids) // 8):
        P = M[b * 8:(b + 1) * 8]; c = P.mean(0)
        u = np.linalg.svd(P - c)[2][0]                                  # bar direction
        nrm = np.array([-u[1], u[0]])
        for k in range(8):
            d = design_dir[b * 8 + k, :2]
            normals[b * 8 + k] = nrm if nrm @ d >= 0 else -nrm
        line_rms[b + 1] = float(np.sqrt((((P - c) @ nrm) ** 2).mean()) * 1000)

    az = np.degrees(np.arctan2(M[:, 0], M[:, 1])) * cmp_["summary"]["az_sign"]
    dist = np.linalg.norm(M, axis=1)
    aed_vals = " ".join(f"{a:.4f} 0.0 {d:.5f}" for a, d in zip(az, dist))
    dir_vals = " ".join(f"{n[0]:.6f} {n[1]:.6f} 0.0" for n in normals)
    lines = {AED_KEY: f"{AED_KEY} {aed_vals}", DIR_KEY: f"{DIR_KEY} {dir_vals}"}

    raw = open(AED, "rb").read()
    eol = b"\r\n" if b"\r\n" in raw else b"\n"                     # keep the original line endings
    with open(f"{OUT_DIR}/panoramix_bus10_measured.txt", "wb") as f:
        f.write(eol.join([b"# Measured WFS layout for Panoramix bus 10, from Sony photogrammetry.",
                          b"# Replace the same two lines in your session file. Elevation 0 (bus uses ignorez 1).",
                          lines[AED_KEY].encode(), lines[DIR_KEY].encode(), b""]))
    key = lambda l: l.split(b" ", 1)[0].decode("utf-8", "ignore")
    out = [lines[key(l)].encode() if key(l) in lines else l for l in raw.split(eol)]
    open(f"{OUT_DIR}/1012_measured.txt", "wb").write(eol.join(out))

    with open(f"{OUT_DIR}/panoramix_bus10_measured.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "bar", "x_m", "y_m", "azimuth_deg", "distance_m", "design_azimuth_deg",
                    "design_distance_m", "direction_x", "direction_y", "direction_angle_deg",
                    "design_direction_angle_deg", "direction_diff_deg", "bar_straightness_rms_mm"])
        for k, i in enumerate(ids):
            ang = np.degrees(np.arctan2(normals[k, 1], normals[k, 0]))
            dang = np.degrees(np.arctan2(design_dir[k, 1], design_dir[k, 0]))
            diff = (ang - dang + 180) % 360 - 180
            w.writerow([i, (i - 1) // 8 + 1, round(M[k, 0], 5), round(M[k, 1], 5), round(az[k], 4), round(dist[k], 5),
                        design_aed[k, 0], design_aed[k, 2], round(normals[k, 0], 6), round(normals[k, 1], 6),
                        round(ang, 3), round(dang, 3), round(diff, 3), round(line_rms[(i - 1) // 8 + 1], 2)])
    diffs = [(np.degrees(np.arctan2(normals[b * 8, 1], normals[b * 8, 0])) -
              np.degrees(np.arctan2(design_dir[b * 8, 1], design_dir[b * 8, 0])) + 180) % 360 - 180
             for b in range(len(ids) // 8)]
    print("bar orientation vs design (deg):", [round(float(d), 2) for d in diffs])
    print("bar straightness rms (mm):", {k: round(v, 2) for k, v in line_rms.items()})
    print("wrote panoramix_bus10_measured.txt / 1012_measured.txt / panoramix_bus10_measured.csv")


if __name__ == "__main__":
    main()
