"""Write the offline speaker photo finder (outputs/speaker_photo_finder):
previews/, data.js, speaker_photos.csv. index.html is static and reads data.js."""
import json, os, csv, collections, cv2, numpy as np
from eval_poses import load, project
from annotations import load_annotations
from config import WORK, OUT_DIR, SONY_IMAGES, DJI_IMAGES, COARSE_DIR

PREVIEW_W = 1600
BLURRY = 60             # global Laplacian sharpness below this is flagged as blurry
MARGIN = 0.45
MAX_VIEW_DEG = 75


def distort(uv, K, k1):
    x = (uv[0] - K[0][2]) / K[0][0]; y = (uv[1] - K[1][2]) / K[1][1]; s = 1 + k1 * (x * x + y * y)
    return x * s * K[0][0] + K[0][2], y * s * K[1][1] + K[1][2]


def main():
    ids, S, nrm, pitch = load()
    F = json.load(open(f"{WORK}/speakers_final.json")); cmu = F["cm_per_unit"]
    poses = json.load(open(f"{WORK}/sony_poses_final.json"))
    os.makedirs(f"{OUT_DIR}/previews", exist_ok=True)
    for n in sorted(poses):
        p = f"{OUT_DIR}/previews/{n[:-4]}.jpg"
        if not os.path.exists(p):
            img = cv2.imread(f"{SONY_IMAGES}/{n}"); sc = PREVIEW_W / img.shape[1]
            cv2.imwrite(p, cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])

    # floor-plane layout for the mini map (PCA of all centres)
    X = np.array([F["speakers"][str(i)]["xyz"] for i in ids]); mu = X.mean(0); V = np.linalg.svd(X - mu)[2]
    xy2 = (X - mu) @ V[:2].T * cmu * 10

    C3 = {r["id"]: r for r in json.load(open(f"{WORK}/centers.json"))["rows"]}
    speakers = {}
    for k, sid in enumerate(ids):
        s = F["speakers"][str(sid)]; Xs = np.array(s["xyz"])
        conf = {o["photo"] for o in s["photos"]}
        photos = []
        for o in s["photos"]:
            p = poses[o["photo"]]; W, H = p["width"], p["height"]
            photos.append(dict(f=o["photo"], x=round(o["xy"][0] / W, 5), y=round(o["xy"][1] / H, 5),
                               r=round(o["r_px"] / W, 5), s=round(o["pitch_px"] / W, 5),
                               a=round(o["view_angle_deg"], 1), e=round(o["reproj_px"], 1), c=1,
                               b=int((p.get("sharpness") or 999) < BLURRY), u=int(not p.get("registered_in_sony_model", True))))
        for n, p in poses.items():             # in frame & facing, but no diaphragm detected
            if n in conf:
                continue
            R, t = np.array(p["R"]), np.array(p["t"]); C = -R.T @ t; v = C - Xs
            ang = np.degrees(np.arccos(np.clip(v @ nrm[sid] / np.linalg.norm(v), -1, 1)))
            if ang > MAX_VIEW_DEG:
                continue
            uv, z = project(p, Xs[None]); uv, z = uv[0], z[0]
            if z <= 0:
                continue
            ppx = pitch * p["K"][0][0] / z; m = MARGIN * ppx
            if not (m < uv[0] < p["width"] - m and m < uv[1] < p["height"] - m) or ppx < 20:
                continue
            x, y = distort(uv, p["K"], p["dist"][0])
            photos.append(dict(f=n, x=round(x / p["width"], 5), y=round(y / p["height"], 5),
                               r=round(0.176 * ppx / p["width"], 5), s=round(ppx / p["width"], 5), a=round(ang, 1), c=0,
                               b=int((p.get("sharpness") or 999) < BLURRY), u=int(not p.get("registered_in_sony_model", True))))
        speakers[sid] = dict(id=sid, bar=(sid - 1) // 8 + 1, xyz_mm=[round(v, 1) for v in C3[sid]["xyz"]], xyz_src=C3[sid]["use"], sd_mm=C3[sid]["sd_boot"] and round(C3[sid]["sd_boot"], 1),
                             map=[round(float(v), 1) for v in xy2[k]], reproj=s["reproj_px_median"] and round(s["reproj_px_median"], 1),
                             n=len(s["photos"]), photos=photos)

    refs = collections.defaultdict(list)
    for (img, sid), a in load_annotations().items():
        p = f"{DJI_IMAGES}/{img}"
        if not os.path.exists(p):
            p = f"{COARSE_DIR}/original_images/{img}"
        refs[sid].append(dict(img=os.path.relpath(p, OUT_DIR).replace("\\", "/"), name=img, src=a["src"],
                              x=a["xy"][0] / 2560, y=a["xy"][1] / 1919, r=a["r"] / 2560))

    data = dict(generated=__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
                sony_dir=SONY_IMAGES.replace("\\", "/"), n_photos=len(poses),
                sizes={n: [p["width"], p["height"]] for n, p in poses.items()},
                speakers=speakers, refs={str(k): v for k, v in refs.items()})
    with open(f"{OUT_DIR}/data.js", "w", encoding="utf-8") as f:
        f.write("window.FINDER_DATA = "); json.dump(data, f, ensure_ascii=False, separators=(",", ":")); f.write(";\n")

    with open(f"{OUT_DIR}/speaker_photos.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "bar", "photo", "status", "center_x_px", "center_y_px", "diaphragm_radius_px",
                    "view_angle_deg", "reproj_err_px"])
        for sid in ids:
            for p in sorted(speakers[sid]["photos"], key=lambda q: (-q["c"], q["f"])):
                Wd, Hd = data["sizes"][p["f"]]
                w.writerow([sid, speakers[sid]["bar"], p["f"], "confirmed" if p["c"] else "possible",
                            round(p["x"] * Wd, 1), round(p["y"] * Hd, 1), round(p["r"] * Wd, 1), p["a"], p.get("e", "")])
    n_c = [speakers[i]["n"] for i in ids]; n_p = [sum(1 for p in speakers[i]["photos"] if not p["c"]) for i in ids]
    print(f"wrote {OUT_DIR}: confirmed photos/speaker median {np.median(n_c):.0f}, possible median {np.median(n_p):.0f}")


if __name__ == "__main__":
    main()
