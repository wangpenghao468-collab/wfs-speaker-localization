"""Export the speaker array as PLY files in the Panoramix frame (millimetres; x right, y front, z up from the floor).

    python tools/export_ply.py              # speakers + design + PGSR mesh crop
    python tools/export_ply.py --no-mesh    # skip the (slow, ~1 GB read) PGSR mesh crop

Writes to measurement_dir:
  speakers_measured_points.ply   112 diaphragm centres, normal = facing direction, colour per bar, speaker id
  speakers_measured_discs.ply    a 28 mm disc at each centre, facing its direction (easy to see in MeshLab/Blender)
  speakers_design_points.ply     your Panoramix design positions (at the median measured height)
  array_pgsr_mesh.ply            the PGSR scene mesh cropped to the speaker bars, in the same frame"""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from config import WORK, OUT_DIR, MEASURE_DIR, REPO

PGSR_MESH = os.path.join(os.path.dirname(OUT_DIR), "PGSR_DJI_WFS", "mesh", "tsdf_fusion_post.ply")
DISC_R, DISC_N = 14.0, 24              # diaphragm radius (mm, ~0.176 pitch) and polygon sides
CROP_XY, CROP_Z = 150.0, (850.0, 1400.0)  # keep mesh within this distance of a bar line / height band


def table(name):
    with open(f"{OUT_DIR}/{name}", encoding="utf-8-sig") as f:
        return {int(r["speaker_id"]): r for r in csv.DictReader(f)}


def bar_colours():
    rng = np.random.default_rng(3)
    base = np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189], [140, 86, 75], [227, 119, 194]])
    return np.vstack([base, base * 0.6 + 100]).astype(np.uint8)[:14]


def write_ply(path, verts, faces=None, normals=None, colours=None, ids=None, comment=""):
    n = len(verts)
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        dt += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if colours is not None:
        dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if ids is not None:
        dt += [("speaker_id", "<i4")]
    v = np.empty(n, dt)
    v["x"], v["y"], v["z"] = verts.T
    if normals is not None:
        v["nx"], v["ny"], v["nz"] = normals.T
    if colours is not None:
        v["red"], v["green"], v["blue"] = colours.T
    if ids is not None:
        v["speaker_id"] = ids
    head = ["ply", "format binary_little_endian 1.0", f"comment {comment}", "comment units: millimetres, Panoramix frame (x right, y front, z up from floor)",
            f"element vertex {n}"] + [f"property {'float' if t == '<f4' else 'uchar' if t == 'u1' else 'int'} {k}" for k, t in dt]
    if faces is not None:
        head += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    with open(path, "wb") as f:
        f.write(("\n".join(head + ["end_header"]) + "\n").encode())
        f.write(v.tobytes())
        if faces is not None:
            fa = np.empty(len(faces), [("n", "u1"), ("i", "<i4", (3,))]); fa["n"] = 3; fa["i"] = faces
            f.write(fa.tobytes())
    print(f"wrote {path}  ({n} vertices{'' if faces is None else f', {len(faces)} faces'})")


def read_open3d_ply(path):
    """Binary PLY from Open3D: double xyz + uchar rgb, faces as uchar count + uint indices."""
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        h = head.decode()
        nv = int(h.split("element vertex ")[1].split()[0]); nf = int(h.split("element face ")[1].split()[0])
        assert "property double x" in h and "property uchar red" in h and "uchar uint vertex_indices" in h, "unexpected PLY layout"
        v = np.fromfile(f, [("p", "<f8", (3,)), ("c", "u1", (3,))], nv)
        fc = np.fromfile(f, [("n", "u1"), ("i", "<u4", (3,))], nf)
    return v["p"], v["c"], fc["i"]


def main():
    P, H = table("panoramix_bus10_measured.csv"), table("speaker_heights.csv")
    D = json.load(open(f"{WORK}/design_vs_measured.json"))
    ids = np.arange(1, 113)
    X = np.array([[float(P[i]["x_m"]) * 1000, float(P[i]["y_m"]) * 1000, float(H[i]["height_mm(global floor)"])] for i in ids])
    N = np.array([[float(P[i]["direction_x"]), float(P[i]["direction_y"]), 0.0] for i in ids])
    col = bar_colours()[(ids - 1) // 8]
    os.makedirs(MEASURE_DIR, exist_ok=True)
    write_ply(f"{MEASURE_DIR}/speakers_measured_points.ply", X, normals=N, colours=col, ids=ids, comment="measured diaphragm centres 2026-10-13")

    # discs facing each speaker's direction
    ang = np.linspace(0, 2 * np.pi, DISC_N, endpoint=False)
    V, F, C = [], [], []
    for k in range(112):
        n = N[k] / np.linalg.norm(N[k]); u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u); w = np.cross(n, u)
        ring = X[k] + DISC_R * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w)
        b = len(V) * 0 + k * (DISC_N + 1)
        V += [X[k], *ring]; C += [col[k]] * (DISC_N + 1)
        F += [[b, b + 1 + j, b + 1 + (j + 1) % DISC_N] for j in range(DISC_N)]
    write_ply(f"{MEASURE_DIR}/speakers_measured_discs.ply", np.array(V), faces=np.array(F), colours=np.array(C), comment="28 mm discs at measured centres")

    Xd = np.array([[r["design_xy"][0], r["design_xy"][1], D["summary"]["height_median"]] for r in sorted(D["rows"], key=lambda r: r["id"])])
    write_ply(f"{MEASURE_DIR}/speakers_design_points.ply", Xd, colours=np.full((112, 3), 160, np.uint8), ids=ids, comment="Panoramix design (1012.txt) at median height")

    if "--no-mesh" in sys.argv or not os.path.exists(PGSR_MESH):
        return
    # PGSR mesh: DJI units -> floor frame (centers.json) -> Panoramix frame (design alignment)
    c = json.load(open(f"{WORK}/centers.json"))
    A = np.stack([c["x"], c["y"], c["z"]]); o = np.array(c["origin"]); s = c["mm_per_unit"]
    th = np.radians(D["summary"]["rotation_deg"]); R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t2 = np.array(D["summary"]["translation_mm"])
    check = (np.array(json.load(open(f"{WORK}/speakers_final.json"))["speakers"]["20"]["xyz"]) - o) @ A.T * s
    check[:2] = R2 @ check[:2] + t2
    print(f"frame check, speaker 20: {np.linalg.norm(check - X[19]):.1f} mm between DJI seed and measured centre")
    print(f"reading {PGSR_MESH} ...")
    pv, pc, pf = read_open3d_ply(PGSR_MESH)
    Q = (pv - o) @ A.T * s
    Q[:, :2] = Q[:, :2] @ R2.T + t2
    keep = (Q[:, 2] > CROP_Z[0]) & (Q[:, 2] < CROP_Z[1])
    near = np.zeros(len(Q), bool)
    for b in range(14):                                  # distance to each bar line, extended 60 mm past the ends
        a, e = X[b * 8, :2], X[b * 8 + 7, :2]; d = (e - a) / np.linalg.norm(e - a)
        a, e = a - 60 * d, e + 60 * d; L = np.linalg.norm(e - a)
        rel = Q[keep, :2] - a; tt = np.clip(rel @ d, 0, L)
        near[np.flatnonzero(keep)[np.linalg.norm(rel - tt[:, None] * d, axis=1) < CROP_XY]] = True
    idx = np.full(len(Q), -1, np.int64); idx[near] = np.arange(near.sum())
    ff = pf[near[pf].all(1)]
    write_ply(f"{MEASURE_DIR}/array_pgsr_mesh.ply", Q[near], faces=idx[ff], colours=pc[near], comment="PGSR tsdf_fusion_post cropped to the speaker bars")


if __name__ == "__main__":
    main()
