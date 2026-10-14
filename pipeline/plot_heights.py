"""One image: speaker diaphragm-centre height above the floor, with uncertainty,
plus a top view coloured by height and the floor points used."""
import json, numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import WORK, OUT_DIR

from config import FONT
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIV = ["#2a78d6", "#6fa3e3", "#b3cdee", "#e6e5e1", "#f3b98f", "#ef8a4f", "#eb6a1e"]   # blue - gray - orange
SIDE = {1: "後左", 2: "左", 3: "左", 4: "左", 5: "左", 6: "前", 7: "前", 8: "前", 9: "前",
        10: "右", 11: "右", 12: "右", 13: "右", 14: "後右"}


def f(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=1 if bold else 0)


def main():
    Fl = json.load(open(f"{WORK}/floor.json"))
    C = {r["id"]: r for r in json.load(open(f"{WORK}/centers.json"))["rows"]}
    rows = Fl["rows"]
    h = np.array([r["height_global"] for r in rows])
    sd = np.array([np.hypot(C[r["id"]]["sd_boot"] or 0, r["sd_floor"]) for r in rows])
    hl = [r["height_local"] for r in rows]
    med = float(np.median(h))
    W, H = 2400, 1400
    im = Image.new("RGB", (W, H), SURF); d = ImageDraw.Draw(im)
    d.text((40, 28), "喇叭振膜中心離地高度（多張照片三角化，地板由照片加密擬合）", fill=INK, font=f(38, True))
    d.text((40, 82), f"中位數 {med:.0f} mm，範圍 {h.min():.0f}–{h.max():.0f} mm。誤差棒 = 喇叭三角化誤差與地板誤差合成（1σ）。"
                     f"空心點 = 以喇叭周圍 1 m 內局部地板量的高度（與全域地板差異中位數 "
                     f"{np.median([a - b for a, b in zip([x for x in hl if x is not None], [r['height_global'] for r in rows if r['height_local'] is not None])]):+.1f} mm）。",
           fill=INK2, font=f(22))

    # ---- per-speaker heights
    x0, y0, x1, y1 = 120, 190, 2340, 760
    lo, hi = np.floor(h.min() / 5) * 5 - 5, np.ceil(h.max() / 5) * 5 + 5
    Y = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
    for g in np.arange(lo, hi + 1, 5):
        d.line([(x0, Y(g)), (x1, Y(g))], fill=GRID, width=1); d.text((x0 - 70, Y(g) - 12), f"{g:.0f}", fill=INK2, font=f(18))
    d.text((x0 - 70, y0 - 40), "mm", fill=INK2, font=f(18))
    bw = (x1 - x0) / 112
    for b in range(14):
        xa = x0 + b * 8 * bw
        d.line([(xa, y0), (xa, y1)], fill=GRID, width=1)
        mb = h[b * 8:(b + 1) * 8].mean()
        d.line([(xa + 4, Y(mb)), (xa + 8 * bw - 4, Y(mb))], fill=INK2, width=2)
        d.text((xa + 6, y1 + 8), f"{b * 8 + 1}–{b * 8 + 8}", fill=INK2, font=f(17))
        d.text((xa + 6, y1 + 30), f"{SIDE[b + 1]} {mb:.0f}", fill=INK, font=f(18, True))
    for k, r in enumerate(rows):
        cx = x0 + (k + 0.5) * bw
        d.line([(cx, Y(h[k] - sd[k])), (cx, Y(h[k] + sd[k]))], fill="#2a78d6", width=2)
        d.ellipse([cx - 4, Y(h[k]) - 4, cx + 4, Y(h[k]) + 4], fill="#2a78d6")
        if hl[k] is not None:
            d.ellipse([cx - 5, Y(hl[k]) - 5, cx + 5, Y(hl[k]) + 5], outline="#eb6a1e", width=2)
    d.text((x0, y1 + 60), "每一架的灰色橫線與粗體數字 = 該架平均高度（mm）。", fill=INK2, font=f(20))

    # ---- top view coloured by height deviation from the median
    D = json.load(open(f"{WORK}/design_vs_measured.json"))["rows"]
    XY = np.array([r["measured_xy"] for r in D])
    bx0, by0, bx1, by1 = 120, 930, 900, 1380
    lo2, hi2 = XY.min(0) - 150, XY.max(0) + 150
    s = min((bx1 - bx0) / (hi2[0] - lo2[0]), (by1 - by0) / (hi2[1] - lo2[1]))
    P = lambda p: (bx0 + (p[0] - lo2[0]) * s, by1 - (p[1] - lo2[1]) * s)
    vm = max(8.0, np.abs(h - med).max())
    for k, p in enumerate(XY):
        c = DIV[int(np.clip(np.round((h[k] - med) / vm * 3 + 3), 0, 6))]
        q = P(p); d.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], fill=c, outline=SURF)
    o = P((0, 0)); d.line([(o[0] - 10, o[1]), (o[0] + 10, o[1])], fill=INK, width=2); d.line([(o[0], o[1] - 10), (o[0], o[1] + 10)], fill=INK, width=2)
    d.text((o[0] + 8, o[1] + 4), "原點", fill=INK, font=f(18))
    d.text((bx0, by0 - 50), "俯視：顏色 = 相對中位高度", fill=INK, font=f(24, True))
    for k, c in enumerate(DIV):
        d.rectangle([bx0 + 330 + k * 34, by0 - 42, bx0 + 360 + k * 34, by0 - 26], fill=c)
    d.text((bx0 + 330, by0 - 22), f"-{vm:.0f}", fill=INK2, font=f(16)); d.text((bx0 + 330 + 6 * 34, by0 - 22), f"+{vm:.0f} mm", fill=INK2, font=f(16))

    # ---- notes
    reg = Fl["regions"]
    tx, ty = 1000, 900
    d.text((tx, ty), "地板與誤差", fill=INK, font=f(26, True))
    notes = [f"地板：由 Sony 照片配對三角化 {Fl['n_points']} 個地磚點，平面殘差 RMS 約 4.6 mm。",
             "分區擬合（左 / 右 / 前 / 後）相對全域地板：",
             *[f"   {k}：傾斜 {v['tilt_deg']:.2f}°，偏移 {v['offset_mm']:+.1f} mm" for k, v in reg.items() if v],
             f"喇叭三角化誤差（bootstrap）中位數 {np.median([C[i]['sd_boot'] for i in C]):.2f} mm；地板平面誤差中位數 {np.median([r['sd_floor'] for r in rows]):.2f} mm。",
             "注意：DJI 稀疏點雲在角落的平面比真實地面高約 8 cm（貨架底層），已改用照片加密的地板。",
             "建議用捲尺量 1–2 顆喇叭中心離地高度做最終驗證。"]
    for k, t in enumerate(notes):
        d.text((tx, ty + 46 + k * 34), t, fill=INK2, font=f(21))
    out = f"{OUT_DIR}/speaker_heights.png"; im.save(out); print(out)


if __name__ == "__main__":
    main()
