"""Top view, true scale: manual (Panoramix design) vs photo-measured positions in
two colours, with magnified insets where the differences are largest."""
import json, numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import WORK, OUT_DIR

from config import FONT
SURF, INK, INK2, GRID, FRAME = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
MANUAL, PHOTO = "#eb6a1e", "#2a78d6"                    # orange = manual design, blue = photo measured
DIAPHRAGM_R_MM = 14


def f(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=1 if bold else 0)


class View:
    def __init__(self, box, lo, hi):
        self.box = box
        x0, y0, x1, y1 = box
        self.s = min((x1 - x0) / (hi[0] - lo[0]), (y1 - y0) / (hi[1] - lo[1]))
        self.ox = x0 + ((x1 - x0) - (hi[0] - lo[0]) * self.s) / 2
        self.oy = y1 - ((y1 - y0) - (hi[1] - lo[1]) * self.s) / 2
        self.lo = lo

    def __call__(self, p):
        return (self.ox + (p[0] - self.lo[0]) * self.s, self.oy - (p[1] - self.lo[1]) * self.s)


def draw_points(d, V, Dm, Mm, ids, r_px, label_ids=(), lw=2, font=None):
    for k in ids:
        a, b = V(Dm[k]), V(Mm[k])
        d.line([a, b], fill=INK2, width=1)
    for k in ids:
        a = V(Dm[k]); d.ellipse([a[0] - r_px, a[1] - r_px, a[0] + r_px, a[1] + r_px], outline=MANUAL, width=lw)
    for k in ids:
        b = V(Mm[k]); d.ellipse([b[0] - r_px, b[1] - r_px, b[0] + r_px, b[1] + r_px], fill=PHOTO, outline=SURF, width=1)
    for k in label_ids:
        if k in ids:
            b = V(Mm[k]); d.text((b[0] + r_px + 3, b[1] - r_px - 18), str(k + 1), fill=INK, font=font)


def main():
    R = json.load(open(f"{WORK}/design_vs_measured.json"))["rows"]
    Dm = np.array([r["design_xy"] for r in R]); Mm = np.array([r["measured_xy"] for r in R])
    dev = np.array([r["dxy"] for r in R])
    W, H = 2400, 1600
    im = Image.new("RGB", (W, H), SURF); d = ImageDraw.Draw(im)
    d.text((40, 26), "俯視（真實比例）：手動設計位置 vs 照片實測位置", fill=INK, font=f(38, True))
    d.text((40, 78), f"橙色空心圈 = 你原本的手動量測 / Panoramix 設計；藍色實心點 = 照片三角化的實測位置；灰線連接同一顆喇叭。"
                     f"偏差中位數 {np.median(dev):.0f} mm，最大 {dev.max():.0f} mm。", fill=INK2, font=f(22))
    # legend
    lx, ly = 40, 120
    d.ellipse([lx, ly + 4, lx + 18, ly + 22], outline=MANUAL, width=3); d.text((lx + 28, ly), "手動設計", fill=INK, font=f(22))
    d.ellipse([lx + 160, ly + 4, lx + 178, ly + 22], fill=PHOTO); d.text((lx + 188, ly), "照片實測", fill=INK, font=f(22))

    # ---- main map
    lo = np.minimum(Dm.min(0), Mm.min(0)) - 180; hi = np.maximum(Dm.max(0), Mm.max(0)) + 180
    V = View((40, 170, 1420, 1560), lo, hi)
    for g in range(-3000, 3001, 500):
        if lo[0] <= g <= hi[0]:
            d.line([V((g, lo[1])), V((g, hi[1]))], fill=GRID, width=1)
        if lo[1] <= g <= hi[1]:
            d.line([V((lo[0], g)), V((hi[0], g))], fill=GRID, width=1)
    o = V((0, 0)); d.line([(o[0] - 14, o[1]), (o[0] + 14, o[1])], fill=INK, width=2); d.line([(o[0], o[1] - 14), (o[0], o[1] + 14)], fill=INK, width=2)
    d.text((o[0] + 10, o[1] + 6), "原點（聆聽位置）", fill=INK, font=f(22))
    r_main = max(3, DIAPHRAGM_R_MM * V.s)
    draw_points(d, V, Dm, Mm, range(112), r_main, label_ids=[0, 7, 8, 39, 40, 71, 72, 103, 104, 111], font=f(20))
    # scale bar 0.5 m
    sb0 = V((lo[0] + 60, lo[1] + 60)); d.line([sb0, (sb0[0] + 500 * V.s, sb0[1])], fill=INK, width=4)
    d.text((sb0[0], sb0[1] + 8), "0.5 m（格線 0.5 m）", fill=INK2, font=f(20))

    # ---- insets
    def centre(ids):
        return np.vstack([Dm[ids], Mm[ids]]).mean(0)
    insets = [("A  後排左段 1–4", list(range(0, 4))),
              ("B  後排右段 109–112", list(range(108, 112))),
              ("C  左前轉角 40–42", list(range(39, 42))),
              ("D  右前轉角 71–73", list(range(70, 73))),
              ("E  前排架間接縫 47–50", list(range(46, 50))),
              ("F  左側架間接縫 23–26", list(range(22, 26)))]
    col_w, row_h = 470, 450
    for n, (title, ids) in enumerate(insets):
        cx0 = 1450 + (n % 2) * (col_w + 10); cy0 = 170 + (n // 2) * row_h
        bw, bh = col_w, row_h - 90
        P = np.vstack([Dm[ids], Mm[ids]])
        ilo, ihi = P.min(0) - 35, P.max(0) + 35          # fit the points, keep the box aspect
        span = ihi - ilo; need = max(span[0] / bw, span[1] / bh)
        c = (ilo + ihi) / 2; ilo, ihi = c - np.array([bw, bh]) * need / 2, c + np.array([bw, bh]) * need / 2
        sub = Image.new("RGB", (bw, bh), SURF); sd = ImageDraw.Draw(sub)
        IV = View((0, 0, bw, bh), ilo, ihi)
        draw_points(sd, IV, Dm, Mm, ids, DIAPHRAGM_R_MM * IV.s, label_ids=ids, lw=3, font=f(18))
        sb = (10, bh - 16); sd.line([sb, (sb[0] + 50 * IV.s, sb[1])], fill=INK, width=3)
        sd.text((sb[0], sb[1] - 26), "5 cm", fill=INK2, font=f(16))
        sd.rectangle([0, 0, bw - 1, bh - 1], outline=GRID, width=1)
        im.paste(sub, (cx0, cy0 + 34))
        dv = dev[ids]
        d.text((cx0, cy0), title, fill=INK, font=f(22, True))
        d.text((cx0, cy0 + 34 + bh + 6), f"放大 {IV.s / V.s:.0f} 倍；偏差 {dv.min():.0f}–{dv.max():.0f} mm", fill=INK2, font=f(19))
        a, b = V(ilo), V(ihi)                             # mark the region on the main map
        d.rectangle([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])], outline=FRAME, width=2)
        d.text((min(a[0], b[0]) + 4, min(a[1], b[1]) + 2), title[0], fill=FRAME, font=f(22, True))
    out = f"{OUT_DIR}/topview_manual_vs_photo.png"
    im.save(out); print(out)


if __name__ == "__main__":
    main()
