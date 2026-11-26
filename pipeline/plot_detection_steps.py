"""The diaphragm detector step by step, with the black-and-white masks it actually works on.
usage: python plot_detection_steps.py [speaker id] [photo]     -> OUT_DIR/detection_steps_<id>.png

1 crop of the photo  2 brightness (V)  3 black/white mask: V > 140 and S < 60 (bright, colourless = silver)
4 after removing specks (3x3 opening), each white blob tested: size, roundness >= 0.6
5 dark-surround mask (V < 70): at least 35 % of the box around the blob must be dark (the black speaker frame)
6 accepted blob: outline + centroid = this photo's observation of the diaphragm centre"""
import json, sys, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import WORK, OUT_DIR, SONY_IMAGES, FONT
from detect_diaphragms import DET_W

T = 360          # tile size


def f(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=1 if bold else 0)


def tile(arr, gray=False):
    a = cv2.resize(arr, (T, T), interpolation=cv2.INTER_NEAREST if gray else cv2.INTER_CUBIC)
    return Image.fromarray(a if gray else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).convert("RGB")


def main(sid, photo=None):
    F = json.load(open(f"{WORK}/speakers_final.json"))["speakers"][str(sid)]
    obs = sorted(F["photos"], key=lambda o: o["view_angle_deg"])
    o = next((x for x in obs if x["photo"] == photo), obs[0])
    img = cv2.imread(f"{SONY_IMAGES}/{o['photo']}"); sc = DET_W / img.shape[1]
    small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    c = np.array(o["xy"]) * sc; half = int(o["pitch_px"] * sc * 1.1)          # a bit more than one speaker
    x0, y0 = int(c[0] - half), int(c[1] - half)
    crop = small[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV); v, s = hsv[..., 2], hsv[..., 1]
    raw = ((v > 140) & (s < 60)).astype(np.uint8)
    mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = (v < 70).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    ii = cv2.integral(dark.astype(np.float32)); H, W = v.shape
    verdict = {}
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 120 or a > 12000 or w > 2.5 * h or h > 2.5 * w or a / (w * h) < 0.55:
            verdict[i] = ("size", None); continue
        cnt = cv2.findContours((lab[y:y + h, x:x + w] == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
        per = cv2.arcLength(cnt, True); circ = 4 * np.pi * a / per ** 2 if per else 0
        if circ < 0.6:
            verdict[i] = ("round", circ); continue
        cx, cy = x + w / 2, y + h / 2; R = max(w, h) * 0.9
        bx0, by0, bx1, by1 = int(max(0, cx - R)), int(max(0, cy - R)), int(min(W, cx + R)), int(min(H, cy + R))
        dk = (ii[by1, bx1] - ii[by0, bx1] - ii[by1, bx0] + ii[by0, bx0]) / max((bx1 - bx0) * (by1 - by0) - a, 1)
        verdict[i] = ("ok" if dk >= 0.35 else "dark", dk, (bx0, by0, bx1, by1))
    k = S = T / crop.shape[1]

    # tiles
    t1 = tile(crop)
    t2 = tile(cv2.cvtColor(v, cv2.COLOR_GRAY2BGR))
    t3 = tile(raw * 255, gray=True)
    col = np.zeros((*mask.shape, 3), np.uint8)
    for i, vd in verdict.items():
        col[lab == i] = (60, 200, 90) if vd[0] == "ok" else (230, 70, 60)
    t4 = tile(col, gray=True)
    t5 = tile(255 - dark * 255, gray=True)            # dark = black, rest white
    t6 = tile(crop)
    d5, d6 = ImageDraw.Draw(t5), ImageDraw.Draw(t6)
    for i, vd in verdict.items():
        if vd[0] == "ok":
            bx0, by0, bx1, by1 = vd[2]
            d5.rectangle([bx0 * S, by0 * S, bx1 * S, by1 * S], outline="#e5484d", width=3)
            d5.text((bx0 * S + 4, by0 * S + 2), f"暗 {vd[1]:.0%}", fill="#e5484d", font=f(20, True))
            m = (lab == i).astype(np.uint8)
            for cn in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
                d6.line([tuple(p) for p in (cn[:, 0] * S).tolist()] + [tuple((cn[0, 0] * S).tolist())], fill="#30d158", width=3)
            ys, xs = np.nonzero(m); q = (xs.mean() * S, ys.mean() * S)
            d6.line([q[0] - 12, q[1], q[0] + 12, q[1]], fill="#30d158", width=3); d6.line([q[0], q[1] - 12, q[0], q[1] + 12], fill="#30d158", width=3)
    ok = sum(1 for vd in verdict.values() if vd[0] == "ok")
    caps = [("1 原圖（裁切）", f"{o['photo']}，縮到寬 {DET_W} px 偵測"),
            ("2 亮度 V", "HSV 的 V：越白越亮"),
            ("3 黑白遮罩", "白 = V > 140 且 S < 60（又亮又沒顏色＝銀色）"),
            ("4 每塊白色逐一檢查", f"去雜點後：綠 = 大小、形狀、圓度 ≧ 0.6 通過；紅 = 不通過（{len(verdict) - ok} 塊）"),
            ("5 周圍要夠暗", "黑 = V < 70。紅框內除了亮區，至少 35% 是黑的（黑色喇叭框）"),
            ("6 結果", "綠線 = 亮區邊界，十字 = 重心 = 這張照片的振膜中心")]
    tiles = [t1, t2, t3, t4, t5, t6]
    cols, pad, cap_h = 3, 20, 92
    out = Image.new("RGB", (cols * (T + pad) + pad, 90 + 2 * (T + cap_h + pad)), "#fcfcfb"); od = ImageDraw.Draw(out)
    od.text((pad, 14), f"#{sid} 振膜偵測：「高亮法」每一步的黑白圖", fill="#0b0b0b", font=f(30, True))
    od.text((pad, 54), "沒有訓練模型，只用亮度、顏色、形狀三個規則；最後的中心再由幾十張照片三角化。", fill="#555", font=f(19))
    for j, (t, (h1, h2)) in enumerate(zip(tiles, caps)):
        x = pad + (j % cols) * (T + pad); y = 90 + (j // cols) * (T + cap_h + pad)
        out.paste(t, (x, y)); od.rectangle([x - 1, y - 1, x + T, y + T], outline="#ccc")
        od.text((x, y + T + 8), h1, fill="#0b0b0b", font=f(22, True))
        words, line, ly = h2, "", y + T + 40
        for ch in words:                                 # simple wrap
            if od.textlength(line + ch, font=f(17)) > T:
                od.text((x, ly), line, fill="#555", font=f(17)); line, ly = "", ly + 22
            line += ch
        od.text((x, ly), line, fill="#555", font=f(17))
    path = f"{OUT_DIR}/detection_steps_{sid}.png"; out.save(path); print(path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, sys.argv[2] if len(sys.argv) > 2 else None)
