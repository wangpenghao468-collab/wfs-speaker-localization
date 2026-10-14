"""How much each photo counts for one speaker's centre, and how many photos the
result needs: share of every photo (sorted), and the distance to the final centre
when only the k most influential photos (or k random photos) are used.
usage: python plot_weights.py <speaker id>"""
import json, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import WORK, OUT_DIR
from triangulate_centers import solve, huber_threshold, MIN_SHARPNESS, MAX_VIEW_DEG

from config import FONT
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#0f3566", "#1d5fb0", "#2a78d6", "#4a90e2", "#86b6ef"]
TOPK, RAND = "#0b0b0b", "#8a8984"


def f(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=1 if bold else 0)


def main(sid):
    F = json.load(open(f"{WORK}/speakers_final.json"))["speakers"]; P = json.load(open(f"{WORK}/sony_poses_final.json"))
    mm = json.load(open(f"{WORK}/centers.json"))["mm_per_unit"]
    bad = set(json.load(open(f"{WORK}/photos_excluded_from_centres.json")))
    Pm = lambda n: np.array(P[n]["K"]) @ np.hstack([np.array(P[n]["R"]), np.array(P[n]["t"])[:, None]])
    obs = [o for o in F[str(sid)]["photos"] if (P[o["photo"]].get("sharpness") or 999) >= MIN_SHARPNESS
           and o["photo"] not in bad and o["view_angle_deg"] <= MAX_VIEW_DEG]
    excl = [o for o in F[str(sid)]["photos"] if o not in obs]
    xy = [np.array(o["xy_und"]) for o in obs]; Ps = [Pm(o["photo"]) for o in obs]
    X, res, _ = solve(xy, Ps)
    thr = huber_threshold(res); hub = np.where(res < thr, 1.0, thr / res)
    depth = np.array([(Pm(o["photo"]) @ np.append(X, 1))[2] for o in obs])
    foc = np.array([P[o["photo"]]["K"][0][0] for o in obs])
    share = hub * (foc / depth) ** 2; share /= share.sum()
    dist_m = depth * mm / 1000
    order = np.argsort(-share)
    ks = list(range(3, len(obs) + 1))
    top = [np.linalg.norm(solve([xy[i] for i in order[:k]], [Ps[i] for i in order[:k]])[0] - X) * mm for k in ks]
    rng = np.random.default_rng(0)
    rnd = [np.median([np.linalg.norm(solve([xy[i] for i in idx], [Ps[i] for i in idx])[0] - X) * mm
                      for idx in (rng.choice(len(obs), k, replace=False) for _ in range(30))]) for k in ks]
    cum = np.cumsum(share[order])

    W, H = 2200, 1250
    im = Image.new("RGB", (W, H), SURF); d = ImageDraw.Draw(im)
    d.text((40, 24), f"#{sid}：每張照片的份量，以及需要多少張照片", fill=INK, font=f(40, True))
    d.text((40, 82), f"中心是用全部 {len(obs)} 張照片一起算的（不是只挑某個範圍），每張按份量計入。"
                     f"份量 = 看得多清楚（相機越近，振膜在照片裡越大）× 是否偏太多（偏多的降權）。", fill=INK2, font=f(23))

    # ---- panel 1: share of every photo, sorted
    x0, y0, x1, y1 = 120, 190, 2140, 600
    d.text((40, 140), f"① 每張照片的份量（由大到小排列；顏色 = 相機距離，近深遠淺）", fill=INK, font=f(26, True))
    vmax = share.max() * 1.1
    for g in np.arange(0, vmax * 100, 1.0):
        yy = y1 - g / 100 / vmax * (y1 - y0); d.line([(x0, yy), (x1, yy)], fill=GRID); d.text((x0 - 60, yy - 12), f"{g:.0f}%", fill=INK2, font=f(18))
    n = len(obs) + len(excl); bw = (x1 - x0) / n
    lo, hi = dist_m.min(), dist_m.max()
    for j, k in enumerate(order):
        c = RAMP[min(4, int((dist_m[k] - lo) / max(hi - lo, 1e-6) * 4.999))]
        xa = x0 + j * bw; yb = y1 - share[k] / vmax * (y1 - y0)
        d.rectangle([xa + 1, yb, xa + bw - 2, y1], fill=c)
    for j in range(len(excl)):
        xa = x0 + (len(obs) + j) * bw; d.rectangle([xa + 1, y1 - 4, xa + bw - 2, y1], fill="#eb6a1e")
    d.text((x0 + len(obs) * bw + 4, y1 - 40), f"← 排除 {len(excl)} 張", fill="#eb6a1e", font=f(18))
    for kk in (10, 30):
        if kk < len(obs):
            xx = x0 + kk * bw; d.line([(xx, y0), (xx, y1)], fill=INK2, width=2)
            d.text((xx + 6, y0 + 4), f"前 {kk} 張合計 {cum[kk - 1] * 100:.0f}%", fill=INK, font=f(20, True))
    d.text((x0, y1 + 8), "每一根柱子 = 一張照片", fill=INK2, font=f(19))

    # ---- panel 2: how far from the final centre with only k photos
    x0, y0, x1, y1 = 120, 720, 2140, 1100
    d.text((40, 670), "② 只用 k 張照片時，結果離最終中心多遠（mm）", fill=INK, font=f(26, True))
    ymax = max(3.0, np.ceil(max(max(top[2:]), max(rnd[2:])) * 1.1))
    for g in np.arange(0, ymax + 0.01, 0.5):
        yy = y1 - g / ymax * (y1 - y0); d.line([(x0, yy), (x1, yy)], fill=GRID); d.text((x0 - 60, yy - 12), f"{g:.1f}", fill=INK2, font=f(18))
    X_ = lambda k: x0 + (k - ks[0]) / (ks[-1] - ks[0]) * (x1 - x0)
    Y_ = lambda v: y1 - min(v, ymax) / ymax * (y1 - y0)
    d.line([(X_(k), Y_(v)) for k, v in zip(ks, rnd)], fill=RAND, width=3)
    d.line([(X_(k), Y_(v)) for k, v in zip(ks, top)], fill=TOPK, width=3)
    for k in (5, 10, 20, 40):
        if k <= ks[-1]:
            d.line([(X_(k), y1), (X_(k), y1 + 8)], fill=INK2); d.text((X_(k) - 10, y1 + 10), str(k), fill=INK2, font=f(18))
    d.text((X_(ks[-1]) - 40, y1 + 10), str(ks[-1]), fill=INK2, font=f(18))
    d.text((x1 - 300, y1 + 34), "使用的照片張數 k", fill=INK2, font=f(19))
    kx = ks[int(len(ks) * 0.45)]
    d.text((X_(kx), Y_(rnd[ks.index(kx)]) - 34), "隨機挑 k 張（30 次的中位數）", fill=RAND, font=f(21, True))
    k10 = ks.index(min(10, ks[-1]))
    d.text((X_(ks[k10]) + 10, Y_(top[k10]) - 34), "挑份量最大的前 k 張", fill=TOPK, font=f(21, True))
    stable = lambda curve: next((k for k in ks if all(v < 0.5 for v in curve[ks.index(k):])), ks[-1])
    k_top, k_rnd = stable(top), stable(rnd)
    # angular spread of the 10 most influential photos vs all photos (why "top-k" alone is not enough)
    C = np.array([-np.array(P[o["photo"]]["R"]).T @ np.array(P[o["photo"]]["t"]) for o in obs])
    rays = (C - X) / np.linalg.norm(C - X, axis=1)[:, None]
    spread = lambda idx: np.degrees(np.arccos(np.clip(rays[idx] @ rays[idx].T, -1, 1))).max()
    s10, sall = spread(order[:10]), spread(np.arange(len(obs)))
    vw = np.array([o["view_angle_deg"] for o in obs])
    d.text((40, 1150), f"結論：中心由全部 {len(obs)} 張照片按份量決定，約 {max(k_top, k_rnd)} 張以上就穩定在最終結果 0.5 mm 以內。"
                       f"只挑份量最大的照片並沒有比較好：", fill=INK, font=f(23))
    d.text((40, 1188), f"它們都是近照、也比較斜（視角中位數 {np.median(vw[order[:10]]):.0f}°，全部 {np.median(vw):.0f}°），"
                       f"斜拍會讓振膜中心系統性地往前偏約 1 mm；混合各種角度的照片可以互相抵消，所以要用全部照片。", fill=INK, font=f(23))
    out = f"{OUT_DIR}/weights_speaker_{sid}.png"; im.save(out)
    print(out, "stable (<0.5 mm) from k: top", k_top, "random", k_rnd, "| spread top10 %.0f deg, all %.0f deg" % (s10, sall))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
