"""Minimal COLMAP binary model reader (cameras/images/points3D)."""
import struct
import numpy as np

CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
                 3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8)}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cid, mid, w, h = _read(f, "iiQQ")
            name, npar = CAMERA_MODELS[mid]
            params = np.array(_read(f, "d" * npar))
            cams[cid] = dict(model=name, width=w, height=h, params=params)
    return cams


def read_images(path, with_points=True):
    imgs = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "idddddddi")
            name = b""
            while True:
                c = f.read(1)
                if c == b"\0":
                    break
                name += c
            (np2d,) = _read(f, "Q")
            raw = f.read(24 * np2d)
            if with_points:
                arr = np.frombuffer(raw, dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("pid", "<i8")]))
                xy = np.stack([arr["x"], arr["y"]], 1)
                pids = arr["pid"].copy()
            else:
                xy, pids = None, None
            imgs[iid] = dict(name=name.decode(), q=np.array([qw, qx, qy, qz]), t=np.array([tx, ty, tz]),
                             camera_id=cid, xy=xy, pids=pids)
    return imgs


def read_points3D(path, with_rgb=False):
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        ids = np.empty(n, np.int64)
        xyz = np.empty((n, 3))
        rgb = np.empty((n, 3), np.uint8)
        for i in range(n):
            pid, x, y, z, r, g, b, err = _read(f, "QdddBBBd")
            (tl,) = _read(f, "Q")
            f.read(8 * tl)
            ids[i] = pid
            xyz[i] = (x, y, z)
            rgb[i] = (r, g, b)
    return (ids, xyz, rgb) if with_rgb else (ids, xyz)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_model(d, with_points=True):
    cams = read_cameras(f"{d}/cameras.bin")
    imgs = read_images(f"{d}/images.bin", with_points)
    for im in imgs.values():
        im["R"] = qvec2rotmat(im["q"])
        im["C"] = -im["R"].T @ im["t"]
    return cams, imgs
