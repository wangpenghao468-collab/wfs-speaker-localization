"""Show what the diaphragm detector actually segments: for one speaker, a frontal,
a medium and an oblique photo, with the bright-region outline, its centroid, and the
final multi-view centre projected back.  usage: python plot_detection_demo.py <id>"""
import json, sys, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import WORK, OUT_DIR, SONY_IMAGES
from detect_diaphragms import DET_W

from config import FONT


def f(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=1 if bold else 0)


def main(sid):
    F = json.load(open(f"{WORK}/speakers_final.json"))["speakers"][str(sid)]; P = json.load(open(f"{WORK}/sony_poses_final.json"))
    X = np.array(json.load(open(f"{WORK}/speakers_final.json"))["speakers"][str(sid)]["xyz"])
    obs = sorted(F["photos"], key=lambda o: o["view_angle_deg"])
    pick = [obs[0], min(obs, key=lambda o: abs(o["view_angle_deg"] - 35)), min(obs, key=lambda o: abs(o["view_angle_deg"] - 60))]
    tiles = []
    for o in pick:
        img = cv2.imread(f"{SONY_IMAGES}/{o['photo']}"); sc = DET_W / img.shape[1]
        small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        bright = ((hsv[..., 2] > 140) & (hsv[..., 1] < 60)).astype(np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        c = np.array(o["xy"]) * sc; half = int(o["pitch_px"] * sc * 0.75)
        x0, y0 = int(c[0] - half), int(c[1] - half)
        crop = small[max(0, y0):y0 + 2 * half, max(0, x0):x0 + 2 * half].copy()
        n, lab = cv2.connectedComponents(bright[max(0, y0):y0 + 2 * half, max(0, x0):x0 + 2 * half])
        cl = c - [max(0, x0), max(0, y0)]
        k = lab[int(cl[1]), int(cl[0])] if 0 <= int(cl[1]) < lab.shape[0] and 0 <= int(cl[0]) < lab.shape[1] else 0
        S = 480 / crop.shape[1]
        big = cv2.resize(crop, (480, int(crop.shape[0] * S)), interpolation=cv2.INTER_CUBIC)
        pil = Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
        if k:
            m = (lab == k).astype(np.uint8)
            cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            for cn in cnts:
                pts = [(p[0][0] * S, p[0][1] * S) for p in cn]
                d.line(pts + [pts[0]], fill="#2bd65a", width=3)
            ys, xs = np.nonzero(m); cen = (xs.mean() * S, ys.mean() * S)
            d.line([(cen[0] - 12, cen[1]), (cen[0] + 12, cen[1])], fill="#2bd65a", width=3)
            d.line([(cen[0], cen[1] - 12), (cen[0], cen[1] + 12)], fill="#2bd65a", width=3)
        p = P[o["photo"]]; K = np.array(p["K"]); h = K @ (np.array(p["R"]) @ X + np.array(p["t"]))
        uv = h[:2] / h[2]; x_, y_ = (uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1]; s_ = 1 + p["dist"][0] * (x_ * x_ + y_ * y_)
        uv = np.array([x_ * s_ * K[0, 0] + K[0, 2], y_ * s_ * K[1, 1] + K[1, 2]]) * sc - [max(0, x0), max(0, y0)]
        q = uv * S; d.ellipse([q[0] - 7, q[1] - 7, q[0] + 7, q[1] + 7], outline="#ff3b30", width=3)
        canvas = Image.new("RGB", (480, pil.height + 70), "#fcfcfb"); canvas.paste(pil, (0, 0))
        cd = ImageDraw.Draw(canvas)
        cd.text((6, pil.height + 6), f"{o['photo']}  視角 {o['view_angle_deg']:.0f}°", fill="#0b0b0b", font=f(20, True))
        cd.text((6, pil.height + 36), f"振膜在照片中直徑約 {2 * o['r_px']:.0f} px（原圖）", fill="#52514e", font=f(18))
        tiles.append(canvas)
    Hm = max(t.height for t in tiles)
    out = Image.new("RGB", (len(tiles) * 500 + 20, Hm + 130), "#fcfcfb"); od = ImageDraw.Draw(out)
    od.text((20, 14), f"#{sid}：「高亮法」實際抓到的區域", fill="#0b0b0b", font=f(30, True))
    od.text((20, 58), "綠線 = 偵測到的亮區邊界（又亮又淡的銀色區域），綠十字 = 亮區重心（這張照片的觀測）；"
                      "紅圈 = 多張照片算出的最終中心投影回這張照片。", fill="#52514e", font=f(19))
    for i, t in enumerate(tiles):
        out.paste(t, (20 + i * 500, 110))
    path = f"{OUT_DIR}/detection_demo_{sid}.png"; out.save(path); print(path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
