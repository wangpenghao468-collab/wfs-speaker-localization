"""Express the measured speakers in a coordinate frame you choose.

    python pipeline/custom_frame.py --out DIR [--origin-speakers 41,72] [--x-from 41 --x-to 72]
                                    [--offset X Y] [--design 1012.txt]

Frame: z up from the floor (mm). The origin is the floor point below the mean of the
--origin-speakers (default: all 112), moved by --offset (mm, in the new x/y). The x axis
points horizontally from speaker --x-from to --x-to (default: the design-aligned x axis);
y = z cross x (Panoramix convention: x right, y front, z up, azimuth 0 = +y, positive to the right).
Without --origin-speakers/--x-from the Panoramix design alignment is reused.

Writes to --out:
  speakers.csv                 id, bar, x/y/z mm, azimuth, distance, facing direction
  panoramix_bus10.txt          the two OSC lines (aed + directions) for this frame
  <design name>_custom.txt     copy of the design file with only those two lines replaced (if --design)
  speakers_points.ply / speakers_discs.ply"""
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK, DESIGN_TXT

AED_KEY, DIR_KEY = "/bus/10/speakers/aed", "/bus/10/speakers/directions/xyz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--origin-speakers", default="")
    ap.add_argument("--x-from", type=int); ap.add_argument("--x-to", type=int)
    ap.add_argument("--offset", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--design", default=DESIGN_TXT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    C = json.load(open(f"{WORK}/centers.json"))
    X = np.array([r["xyz"] for r in sorted(C["rows"], key=lambda r: r["id"])])   # floor frame, mm, z = height
    ids = np.arange(1, len(X) + 1)
    D = json.load(open(f"{WORK}/design_vs_measured.json"))["summary"]
    th = np.radians(D["rotation_deg"]); R0 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t0 = np.array(D["translation_mm"])
    P0 = X[:, :2] @ R0.T + t0                                          # design-aligned (Panoramix) frame

    # new frame, defined in the design-aligned frame so "default" = the current Panoramix result
    if a.x_from and a.x_to:
        d = P0[a.x_to - 1] - P0[a.x_from - 1]; xa = d / np.linalg.norm(d)
    else:
        xa = np.array([1.0, 0.0])
    ya = np.array([-xa[1], xa[0]])                                     # z cross x
    if a.origin_speakers.strip():
        sel = [int(s) for s in a.origin_speakers.replace(" ", "").split(",") if s]
        org = P0[np.array(sel) - 1].mean(0)
    else:
        org = np.zeros(2)
    org = org + a.offset[0] * xa + a.offset[1] * ya
    B = np.stack([xa, ya])
    P = (P0 - org) @ B.T                                               # mm in the new frame

    # facing: normal of each 8-speaker bar line, pointing into the array
    cen = P.mean(0); N = np.zeros_like(P)
    for b in range(len(P) // 8):
        Q = P[b * 8:(b + 1) * 8]; c = Q.mean(0); u = np.linalg.svd(Q - c)[2][0]; n = np.array([-u[1], u[0]])
        N[b * 8:(b + 1) * 8] = n if n @ (cen - c) >= 0 else -n
    az = np.degrees(np.arctan2(P[:, 0], P[:, 1])); dist = np.linalg.norm(P, axis=1) / 1000

    with open(f"{a.out}/speakers.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "bar", "x_mm", "y_mm", "height_mm", "azimuth_deg", "distance_m", "direction_x", "direction_y", "direction_deg"])
        for k in range(len(P)):
            w.writerow([k + 1, k // 8 + 1, round(P[k, 0], 1), round(P[k, 1], 1), round(X[k, 2], 1), round(az[k], 4), round(dist[k], 5),
                        round(N[k, 0], 6), round(N[k, 1], 6), round(np.degrees(np.arctan2(N[k, 1], N[k, 0])), 3)])
    lines = {AED_KEY: AED_KEY + " " + " ".join(f"{x:.4f} 0.0 {d:.5f}" for x, d in zip(az, dist)),
             DIR_KEY: DIR_KEY + " " + " ".join(f"{n[0]:.6f} {n[1]:.6f} 0.0" for n in N)}
    open(f"{a.out}/panoramix_bus10.txt", "w", encoding="utf-8").write(
        "# Speaker layout in the custom frame. Elevation 0 (ignorez 1); heights are in speakers.csv.\n" + lines[AED_KEY] + "\n" + lines[DIR_KEY] + "\n")
    if a.design and os.path.exists(a.design):
        raw = open(a.design, "rb").read(); eol = b"\r\n" if b"\r\n" in raw else b"\n"
        key = lambda l: l.split(b" ", 1)[0].decode("utf-8", "ignore")
        out = [lines[key(l)].encode() if key(l) in lines else l for l in raw.split(eol)]
        name = os.path.splitext(os.path.basename(a.design))[0]
        open(f"{a.out}/{name}_custom.txt", "wb").write(eol.join(out))

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    from export_ply import write_ply, bar_colours, DISC_R, DISC_N
    XYZ = np.c_[P, X[:, 2]]; N3 = np.c_[N, np.zeros(len(N))]; col = bar_colours()[(ids - 1) // 8]
    write_ply(f"{a.out}/speakers_points.ply", XYZ, normals=N3, colours=col, ids=ids, comment="measured centres, custom frame")
    ang = np.linspace(0, 2 * np.pi, DISC_N, endpoint=False); V, F, Cc = [], [], []
    for k in range(len(XYZ)):
        n = N3[k]; u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(n, u); b0 = k * (DISC_N + 1)
        V += [XYZ[k], *(XYZ[k] + DISC_R * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v))]; Cc += [col[k]] * (DISC_N + 1)
        F += [[b0, b0 + 1 + j, b0 + 1 + (j + 1) % DISC_N] for j in range(DISC_N)]
    write_ply(f"{a.out}/speakers_discs.ply", np.array(V), faces=np.array(F), colours=np.array(Cc), comment="28 mm discs, custom frame")
    json.dump(dict(origin_speakers=a.origin_speakers, x_from=a.x_from, x_to=a.x_to, offset_mm=a.offset,
                   origin_in_design_frame_mm=org.round(2).tolist(), x_axis_in_design_frame=xa.round(6).tolist()),
              open(f"{a.out}/frame.json", "w"), indent=1)
    print(f"origin (design frame) {org.round(1)} mm, x axis {xa.round(4)}; speaker 1 at {P[0].round(1)} mm")


if __name__ == "__main__":
    main()
