"""One image: designed vs measured speaker positions (top view) + per-speaker deviation."""
import json, numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import WORK, OUT_DIR

from config import FONT
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DESIGN, MEAS = "#8a8984", "#2a78d6"                      # neutral = design, series-1 = measured
SEQ = ["#cfe0f6", "#86b6ef", "#4a90e2", "#2a78d6", "#1d5fb0", "#184f95", "#0f3566"]   # blue ramp light->dark
EXAG = 5


def f(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=1 if bold else 0)


def seq_color(v, vmax):
    return SEQ[min(len(SEQ) - 1, int(v / vmax * (len(SEQ) - 1) + 0.5))]


def main():
    D = json.load(open(f"{WORK}/design_vs_measured.json")); S, R = D["summary"], D["rows"]
    W, H = 2400, 1500
    im = Image.new("RGB", (W, H), SURF); d = ImageDraw.Draw(im)
    d.text((40, 28), "喇叭位置：設計（Panoramix）vs 照片實測", fill=INK, font=f(40, True))
    d.text((40, 84), f"以你的原點為準；實測經剛體對齊（旋轉 {S['rotation_deg']:.2f}°，不縮放）。"
                     f"水平偏差中位數 {S['median_dxy']:.0f} mm，90% 在 {S['p90_dxy']:.0f} mm 以內，最大 {S['max_dxy']:.0f} mm",
           fill=INK2, font=f(24))

    # ---------------- left: top view
    L = (40, 140, 1180, 1460)
    dx = np.array([r["design_xy"] for r in R]); mx = np.array([r["measured_xy"] for r in R])
    lo = np.minimum(dx.min(0), mx.min(0)) - 150; hi = np.maximum(dx.max(0), mx.max(0)) + 150
    sc = min((L[2] - L[0]) / (hi[0] - lo[0]), (L[3] - L[1] - 60) / (hi[1] - lo[1]))
    ox = L[0] + ((L[2] - L[0]) - (hi[0] - lo[0]) * sc) / 2
    P = lambda p: (ox + (p[0] - lo[0]) * sc, L[3] - 60 - (p[1] - lo[1]) * sc)
    for g in range(-3000, 3001, 500):                     # 0.5 m grid
        if lo[0] <= g <= hi[0]:
            d.line([P((g, lo[1])), P((g, hi[1]))], fill=GRID, width=1)
        if lo[1] <= g <= hi[1]:
            d.line([P((lo[0], g)), P((hi[0], g))], fill=GRID, width=1)
    o = P((0, 0)); d.line([(o[0] - 14, o[1]), (o[0] + 14, o[1])], fill=INK, width=2); d.line([(o[0], o[1] - 14), (o[0], o[1] + 14)], fill=INK, width=2)
    d.text((o[0] + 10, o[1] + 8), "原點", fill=INK, font=f(22))
    vmax = max(60.0, max(r["dxy"] for r in R))
    for r in R:
        a, b = P(r["design_xy"]), np.array(r["design_xy"]) + (np.array(r["measured_xy"]) - np.array(r["design_xy"])) * EXAG
        b = P(b)
        d.line([a, b], fill=seq_color(r["dxy"], vmax), width=3)
        d.ellipse([a[0] - 6, a[1] - 6, a[0] + 6, a[1] + 6], outline=DESIGN, width=2, fill=SURF)
        d.ellipse([b[0] - 5, b[1] - 5, b[0] + 5, b[1] + 5], fill=seq_color(r["dxy"], vmax), outline=SURF, width=2)
    for r in R:                                           # selective labels: first id of each bar
        if r["id"] % 8 == 1 or r["id"] in (8, 40, 72, 104, 112):
            a = P(r["design_xy"]); c = np.array(a) - np.array(o); c = c / (np.linalg.norm(c) + 1e-9)
            d.text((a[0] + c[0] * 26 - 14, a[1] + c[1] * 26 - 12), str(r["id"]), fill=INK2, font=f(20))
    d.text((L[0], L[3] - 40), f"俯視圖，格線 0.5 m。偏差箭頭放大 {EXAG} 倍。", fill=INK2, font=f(22))
    # legend
    lx, ly = L[0] + 10, L[1] + 4
    d.ellipse([lx, ly + 6, lx + 14, ly + 20], outline=DESIGN, width=2); d.text((lx + 22, ly), "設計位置", fill=INK, font=f(22))
    d.ellipse([lx + 150, ly + 6, lx + 164, ly + 20], fill=MEAS); d.text((lx + 172, ly), "實測（偏差×5）", fill=INK, font=f(22))
    for k, c in enumerate(SEQ):
        d.rectangle([lx + 380 + k * 34, ly + 6, lx + 410 + k * 34, ly + 20], fill=c)
    d.text((lx + 380, ly + 24), "0", fill=INK2, font=f(18)); d.text((lx + 380 + 6 * 34, ly + 24), f"{vmax:.0f} mm", fill=INK2, font=f(18))

    # ---------------- right top: per-speaker horizontal deviation (radial component signed)
    def panel(box, vals, title, unit, sym=True):
        x0, y0, x1, y1 = box
        d.text((x0, y0 - 44), title, fill=INK, font=f(26, True))
        vm = max(10, np.ceil(max(abs(v) for v in vals) / 10) * 10)
        zero = (y0 + y1) / 2 if sym else y1
        span = (y1 - y0) / 2 if sym else (y1 - y0)
        for g in np.arange(-vm if sym else 0, vm + 1, 20):
            yy = zero - g / vm * span
            d.line([(x0, yy), (x1, yy)], fill=GRID if g else INK2, width=1)
            d.text((x0 - 56, yy - 11), f"{g:.0f}", fill=INK2, font=f(18))
        bw = (x1 - x0) / len(vals)
        for k, v in enumerate(vals):
            xa = x0 + k * bw + 1; xb = xa + bw - 2; yv = zero - v / vm * span
            d.rectangle([xa, min(yv, zero), xb, max(yv, zero)], fill=MEAS if v >= 0 else "#eb6a1e")
        for b in range(15):
            xx = x0 + b * 8 * bw
            d.line([(xx, y0), (xx, y1)], fill=GRID, width=1)
            if b < 14:
                d.text((xx + 2, y1 + 6), str(b * 8 + 1), fill=INK2, font=f(18))
        d.text((x1 - 140, y1 + 30), "喇叭編號", fill=INK2, font=f(18))
        d.text((x1 + 8, zero - 11), unit, fill=INK2, font=f(18))

    rad = [r["d_radial"] for r in R]; tan_ = [r["d_tangential"] for r in R]; hz = [r["dz"] for r in R]
    panel((1300, 220, 2340, 520), rad, "徑向偏差（藍＝比設計更遠離原點，橙＝更靠近）", "mm")
    panel((1300, 640, 2340, 900), tan_, "切向偏差（沿著架子方向）", "mm")
    panel((1300, 1020, 2340, 1280), hz, f"高度偏差（相對中位數 {S['height_median']:.0f} mm；設計仰角全為 0）", "mm")
    d.text((1300, 1360), f"若允許縮放，最佳比例 {S['scale_if_free']:.4f}（實測陣列比設計小 {(S['scale_if_free'] - 1) * 100:.1f}%）。"
                         f"原點位置由所有喇叭擬合推得。", fill=INK2, font=f(22))
    n_tri = sum(r["source"] == "triangulated" for r in R)
    def gaps(key):
        a = [np.linalg.norm(np.subtract(R[s][key], R[s - 1][key])) for s in range(41, 72)]
        return np.mean([g for k, g in zip(range(41, 72), a) if k % 8]), np.mean([g for k, g in zip(range(41, 72), a) if not k % 8])
    di, db = gaps("design_xy"); mi, mb = gaps("measured_xy")
    d.text((1300, 1396), f"主因：設計的架間距 {db:.0f} mm，實測 {mb:.0f} mm（架內 {di:.0f} / {mi:.0f} mm）。"
                         f"{n_tri}/112 顆為多張照片三角化。", fill=INK2, font=f(22))
    out = f"{OUT_DIR}/design_vs_measured.png"
    im.save(out); print(out)


if __name__ == "__main__":
    main()
