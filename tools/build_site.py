"""Build the public GitHub Pages site in docs/ from the local results.

    python tools/build_site.py

Reads the local finder data (output_dir/data.js and previews/) and writes
  docs/sheets/NNN.jpg     one low-resolution contact sheet per speaker (no original photos are published)
  docs/data.js            per-speaker results and the tile layout of each sheet
  docs/figures/, docs/verify/, docs/data/   figures, per-bar check montages, result files
and copies the result files to measurement_dir. The html pages in docs/ are kept in the repository."""
import json, os, shutil, sys, datetime
import cv2, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from config import OUT_DIR, SITE_DIR, MEASURE_DIR, DESIGN_TXT

TILE_W, TILE_H, COLS, ZOOM = 192, 144, 10, 3.2
FIGURES = ["pipeline_overview.png", "detection_demo_20.png", "rays_speaker_20.png", "weights_speaker_20.png",
           "speaker_heights.png", "topview_manual_vs_photo.png", "design_vs_measured.png"]
RESULTS = ["1012_measured.txt", "panoramix_bus10_measured.txt", "panoramix_bus10_measured.csv",
           "speaker_centers_mm.csv", "speaker_heights.csv", "design_vs_measured.csv", "speaker_photos.csv"]


def crop(img, x, y, span, r, color, label):
    """Tile centred on (x, y) px, `span` px wide, circle of radius r px, caption at the bottom."""
    w, h = span, span * TILE_H / TILE_W
    M = np.array([[TILE_W / w, 0, TILE_W / 2 - x * TILE_W / w], [0, TILE_H / h, TILE_H / 2 - y * TILE_H / h]])
    t = cv2.warpAffine(img, M, (TILE_W, TILE_H), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
    cv2.circle(t, (TILE_W // 2, TILE_H // 2), max(int(r * TILE_W / w), 6), color, 2, cv2.LINE_AA)
    cv2.rectangle(t, (0, TILE_H - 16), (TILE_W, TILE_H), (0, 0, 0), -1)
    cv2.putText(t, label, (4, TILE_H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    return t


def main():
    D = json.loads(open(f"{OUT_DIR}/data.js", encoding="utf-8").read().split("=", 1)[1].rstrip().rstrip(";"))
    for d in ("sheets", "figures", "verify", "data"):
        os.makedirs(f"{SITE_DIR}/{d}", exist_ok=True)
    cache = {}

    def load(p):
        if p not in cache:
            if len(cache) > 300:
                cache.clear()
            cache[p] = cv2.imread(p)
        return cache[p]

    def table(name):
        import csv
        with open(f"{OUT_DIR}/{name}", encoding="utf-8-sig") as f:
            return {r["speaker_id"]: r for r in csv.DictReader(f)}
    pano, dev = table("panoramix_bus10_measured.csv"), table("design_vs_measured.csv")

    speakers, total = {}, 0
    for sid, s in sorted(D["speakers"].items(), key=lambda kv: int(kv[0])):
        tiles, items = [], []
        for r in D["refs"].get(sid, []):                       # your DJI annotations first (orange)
            im = load(os.path.normpath(os.path.join(OUT_DIR, r["img"])))
            if im is None:
                continue
            W = im.shape[1]; span = max(r["r"] * 5.5, 0.03) * ZOOM * W
            tiles.append(crop(im, r["x"] * W, r["y"] * im.shape[0], span, r["r"] * W, (0, 150, 255), "DJI " + r["name"][:-4]))
            items.append(dict(k="dji", f=r["name"], src=r["src"]))
        photos = sorted((p for p in s["photos"] if p["c"]), key=lambda p: -p["s"])
        for p in photos:                                          # confirmed Sony photos, largest first (green)
            im = load(f"{OUT_DIR}/previews/{p['f'][:-4]}.jpg")
            if im is None:
                continue
            W = im.shape[1]
            tiles.append(crop(im, p["x"] * W, p["y"] * im.shape[0], max(p["s"] * W * ZOOM, 60), p["r"] * W,
                              (90, 214, 43), f"{p['f'][:-4]}  {round(p['a'])}deg"))
            Wo, Ho = D["sizes"][p["f"]]
            items.append(dict(k="sony", f=p["f"], a=p["a"], e=p.get("e"), b=p["b"],
                              px=[round(p["x"] * Wo), round(p["y"] * Ho)]))
        rows = (len(tiles) + COLS - 1) // COLS
        sheet = np.full((max(rows, 1) * TILE_H, COLS * TILE_W, 3), 24, np.uint8)
        for i, t in enumerate(tiles):
            sheet[(i // COLS) * TILE_H:(i // COLS + 1) * TILE_H, (i % COLS) * TILE_W:(i % COLS + 1) * TILE_W] = t
        out = f"{SITE_DIR}/sheets/{int(sid):03d}.jpg"
        cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 72, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
        total += os.path.getsize(out)
        speakers[sid] = {k: s[k] for k in ("id", "bar", "xyz_mm", "xyz_src", "sd_mm", "map", "reproj", "n")}
        P, V = pano[sid], dev[sid]
        speakers[sid].update(
            maybe=sum(1 for p in s["photos"] if not p["c"]), items=items,
            pano=dict(x=float(P["x_m"]), y=float(P["y_m"]), az=float(P["azimuth_deg"]), d=float(P["distance_m"]),
                      dir=float(P["direction_angle_deg"]), dir_design=float(P["design_direction_angle_deg"])),
            design=dict(x=float(V["design_x_mm"]) / 1000, y=float(V["design_y_mm"]) / 1000,
                        dev=float(V["horizontal_deviation_mm"]), radial=float(V["radial_mm(+ = farther)"]),
                        tang=float(V["tangential_mm"])))
        if int(sid) % 16 == 0:
            print(f"sheets {sid}/112", flush=True)
    print(f"sheets: {total / 1e6:.1f} MB")

    site = dict(generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), n_photos=D["n_photos"],
                tile=[TILE_W, TILE_H], cols=COLS, speakers=speakers)
    with open(f"{SITE_DIR}/data.js", "w", encoding="utf-8") as f:
        f.write("window.SITE_DATA = "); json.dump(site, f, ensure_ascii=False, separators=(",", ":")); f.write(";\n")

    for n in FIGURES:
        shutil.copy2(f"{OUT_DIR}/{n}", f"{SITE_DIR}/figures/{n}")
    for d in ("verify", "verify_unregistered"):
        os.makedirs(f"{SITE_DIR}/verify/{d}", exist_ok=True)
        for n in sorted(os.listdir(f"{OUT_DIR}/{d}")):
            shutil.copy2(f"{OUT_DIR}/{d}/{n}", f"{SITE_DIR}/verify/{d}/{n}")
    os.makedirs(MEASURE_DIR, exist_ok=True)
    for n in RESULTS:
        shutil.copy2(f"{OUT_DIR}/{n}", f"{SITE_DIR}/data/{n}")
        shutil.copy2(f"{OUT_DIR}/{n}", f"{MEASURE_DIR}/{n}")
    shutil.copy2(DESIGN_TXT, f"{SITE_DIR}/data/1012_design.txt")
    shutil.copy2(DESIGN_TXT, f"{MEASURE_DIR}/1012_design.txt")
    open(f"{SITE_DIR}/.nojekyll", "w").close()
    print(f"site written to {SITE_DIR}; results copied to {MEASURE_DIR}")


if __name__ == "__main__":
    main()
