"""Readable picture of how one speaker's centre is found from many photos:
left = where the cameras were and their rays to the speaker (top view),
right = zoom on the speaker with the diaphragm drawn to scale, showing how
closely the rays pass the solved centre; bottom = how far each ray misses it.
usage: python plot_rays.py <speaker id>"""
import json, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import WORK, OUT_DIR
from triangulate_centers import solve, huber_threshold, MIN_SHARPNESS, MAX_VIEW_DEG

from config import FONT
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAY, DOWN, SPK = "#2a78d6", "#eb6a1e", "#0b0b0b"
DIAPHRAGM_R = 14.0                                       # mm


def f(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=1 if bold else 0)


def canvas(w, h):
    im = Image.new("RGB", (w, h), SURF)
    return im, ImageDraw.Draw(im)


def main(sid):
    F = json.load(open(f"{WORK}/speakers_final.json")); poses = json.load(open(f"{WORK}/sony_poses_final.json"))
    Cj = json.load(open(f"{WORK}/centers.json")); mm = Cj["mm_per_unit"]
    o = np.array(Cj["origin"]); B = np.array([Cj["x"], Cj["y"], Cj["z"]])
    T = lambda X: (np.asarray(X) - o) @ B.T * mm
    bad = set(json.load(open(f"{WORK}/photos_excluded_from_centres.json")))       # same selection as the centres
    obs = [p for p in F["speakers"][str(sid)]["photos"] if (poses[p["photo"]].get("sharpness") or 999) >= MIN_SHARPNESS
           and p["photo"] not in bad and p["view_angle_deg"] <= MAX_VIEW_DEG]
    Ps = [np.array(poses[p["photo"]]["K"]) @ np.hstack([np.array(poses[p["photo"]]["R"]), np.array(poses[p["photo"]]["t"])[:, None]]) for p in obs]
    xy = [np.array(p["xy_und"]) for p in obs]
    X, res, _ = solve(xy, Ps)
    thr = huber_threshold(res); down = res > thr
    Xm = T(X)
    cams, dirs, depth, foc = [], [], [], []
    for p, u in zip(obs, xy):
        P = poses[p["photo"]]; K = np.array(P["K"]); R = np.array(P["R"]); t = np.array(P["t"])
        cams.append(T(-R.T @ t)); dv = B @ (R.T @ np.linalg.inv(K) @ np.array([u[0], u[1], 1.0])); dirs.append(dv / np.linalg.norm(dv))
        foc.append(K[0, 0])
    cams, dirs, foc = np.array(cams), np.array(dirs), np.array(foc)
    v = Xm - cams; along = (v * dirs).sum(1)
    offs = cams + dirs * along[:, None] - Xm                  # where each sightline passes the centre (3D, mm)
    miss = np.linalg.norm(offs, axis=1)
    # influence of each photo on the solution: Huber weight x (pixels per mm at that distance)^2
    hub = np.where(res < thr, 1.0, thr / res)
    infl = hub * (foc / along) ** 2; share = infl / infl.sum()
    dist_m = along / 1000
    boot = np.array([T(solve([xy[i] for i in idx], [Ps[i] for i in idx])[0])
                     for idx in np.random.default_rng(0).integers(0, len(xy), (200, len(xy)))])
    sd = np.sqrt((boot.std(0) ** 2).sum())
    others = {int(k): T(v["xyz"]) for k, v in F["speakers"].items() if int(k) != sid}
    # face plane of the speaker: horizontal axis along its bar, vertical axis = height
    rows = {r["id"]: np.array(r["xyz"]) for r in Cj["rows"]}
    bar = np.array([rows[i] for i in range((sid - 1) // 8 * 8 + 1, (sid - 1) // 8 * 8 + 9)])
    eh = np.linalg.svd(bar - bar.mean(0))[2][0]; eh[2] = 0; eh /= np.linalg.norm(eh)
    ez = np.array([0, 0, 1.0])
    RAMP = ["#0f3566", "#1d5fb0", "#2a78d6", "#4a90e2", "#86b6ef"]           # near -> far
    dlo, dhi = dist_m.min(), dist_m.max()
    col = lambda k: RAMP[min(4, int((dist_m[k] - dlo) / max(dhi - dlo, 1e-6) * 4.999))]

    W, H = 2200, 1360
    im, d = canvas(W, H)
    d.text((40, 24), f"#{sid} 的中心是怎麼算出來的：{len(obs)} 張照片，{len(obs)} 條視線", fill=INK, font=f(40, True))
    d.text((40, 82), "每張照片都告訴我們「振膜中心在這台相機的哪個方向上」，也就是一條從相機出發的直線。"
                     "如果完全沒有誤差，所有直線會交在同一點。", fill=INK2, font=f(23))

    # ---- left: room top view (camera colour = distance, same as the right panel)
    LW, LH = 1000, 900
    left, ld = canvas(LW, LH)
    pts = np.vstack([cams[:, :2], Xm[None, :2]]); lo, hi = pts.min(0) - 250, pts.max(0) + 250
    s = min(LW / (hi[0] - lo[0]), LH / (hi[1] - lo[1]))
    P = lambda p: ((p[0] - lo[0]) * s + (LW - (hi[0] - lo[0]) * s) / 2, LH - (p[1] - lo[1]) * s - (LH - (hi[1] - lo[1]) * s) / 2)
    for k2, q in others.items():
        a = P(q[:2]); ld.ellipse([a[0] - 3, a[1] - 3, a[0] + 3, a[1] + 3], fill="#c9c8c3")
    for k in range(len(cams)):
        ld.line([P(cams[k, :2]), P(Xm[:2])], fill="#c8dbf3", width=1)
    for k in np.argsort(-dist_m):
        a = P(cams[k, :2]); r = 4 + 10 * np.sqrt(share[k] / share.max())
        ld.ellipse([a[0] - r, a[1] - r, a[0] + r, a[1] + r], fill=col(k), outline=DOWN if down[k] else SURF, width=3 if down[k] else 1)
    b = P(Xm[:2]); ld.ellipse([b[0] - 9, b[1] - 9, b[0] + 9, b[1] + 9], fill=SPK)
    ld.text((b[0] + 12, b[1] - 30), f"#{sid}", fill=INK, font=f(26, True))
    ld.rectangle([0, 0, LW - 1, LH - 1], outline=GRID)
    im.paste(left, (40, 170))
    d.text((40, 136), "① 俯視整個房間：每個圓點 = 拍照時相機的位置", fill=INK, font=f(24, True))
    d.text((40, 1076), f"相機離喇叭 {dlo:.1f}–{dhi:.1f} m，從很多不同方向拍，所以位置能定得準。", fill=INK2, font=f(22))

    # ---- right: "dartboard" seen from in front of the speaker
    RW, RH = 1080, 900; HALF = 8.0
    right, rd = canvas(RW, RH)
    zs = (RH - 60) / (2 * HALF); cx, cy = RW / 2 - 60, RH / 2
    Z = lambda h, v_: (cx + h * zs, cy - v_ * zs)
    for rr in (1, 2, 3, 5):
        rd.ellipse([cx - rr * zs, cy - rr * zs, cx + rr * zs, cy + rr * zs], outline="#cfcec9", width=2 if rr == 5 else 1)
        rd.text((cx + rr * zs * 0.71 + 4, cy - rr * zs * 0.71 - 22), f"{rr} mm", fill="#8a8984", font=f(18))
    for k in np.argsort(share):                                       # big dots last (on top)
        h, v_ = offs[k] @ eh, offs[k] @ ez
        h, v_ = np.clip(h, -HALF + 0.3, HALF - 0.3), np.clip(v_, -HALF + 0.3, HALF - 0.3)   # outliers pinned to the edge
        q = Z(h, v_); r = 3 + 13 * np.sqrt(share[k] / share.max())
        rd.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=col(k), outline=DOWN if down[k] else SURF, width=3 if down[k] else 1)
    rd.line([(cx - 16, cy), (cx + 16, cy)], fill=SPK, width=3); rd.line([(cx, cy - 16), (cx, cy + 16)], fill=SPK, width=3)
    rd.text(Z(-HALF + 0.3, -HALF + 1.4), "← 沿著架子方向 →", fill=INK2, font=f(19))
    rd.text(Z(-HALF + 0.3, HALF - 0.4), "↑ 高度", fill=INK2, font=f(19))
    # inset: the whole diaphragm with the same dots, for size
    ix, iy, ir = RW - 135, RH - 140, 110
    rd.rectangle([ix - ir - 10, iy - ir - 36, ix + ir + 10, iy + ir + 10], fill=SURF, outline=GRID)
    rd.text((ix - ir, iy - ir - 32), "整個振膜（直徑 28 mm）", fill=INK2, font=f(17))
    rd.ellipse([ix - ir, iy - ir, ix + ir, iy + ir], outline="#b8b7b2", width=2)
    isc = ir / DIAPHRAGM_R
    for k in range(len(offs)):
        q = (ix + (offs[k] @ eh) * isc, iy - (offs[k] @ ez) * isc); rd.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=col(k))
    zz = 8 * isc; rd.rectangle([ix - zz, iy - zz, ix + zz, iy + zz], outline=INK2, width=1)
    rd.text((ix - zz, iy + zz + 2), "放大區", fill=INK2, font=f(15))
    # legend (right side of the panel)
    lx, ly = RW - 230, 30
    rd.text((lx, ly), "點的大小 = 份量", fill=INK, font=f(21, True))
    for n, frac in enumerate((1.0, 0.4, 0.1)):
        r = 3 + 13 * np.sqrt(frac); yy = ly + 50 + n * 40
        rd.ellipse([lx + 20 - r, yy - r, lx + 20 + r, yy + r], fill="#2a78d6")
        rd.text((lx + 48, yy - 12), ("最大", "中", "小")[n], fill=INK2, font=f(19))
    rd.text((lx, ly + 190), "顏色 = 相機距離", fill=INK, font=f(21, True))
    for n, c in enumerate(RAMP):
        rd.rectangle([lx + n * 36, ly + 228, lx + n * 36 + 30, ly + 248], fill=c)
    rd.text((lx, ly + 252), f"近 {dlo:.1f} m", fill=INK2, font=f(17)); rd.text((lx + 120, ly + 252), f"遠 {dhi:.1f} m", fill=INK2, font=f(17))
    rd.ellipse([lx + 6, ly + 300, lx + 30, ly + 324], fill="#2a78d6", outline=DOWN, width=3)
    rd.text((lx + 40, ly + 298), "橙框 = 被降權", fill=INK2, font=f(19))
    rd.line([(lx + 6, ly + 360), (lx + 30, ly + 360)], fill=SPK, width=3); rd.line([(lx + 18, ly + 348), (lx + 18, ly + 372)], fill=SPK, width=3)
    rd.text((lx + 40, ly + 348), "算出的中心", fill=INK2, font=f(19))
    rd.rectangle([0, 0, RW - 1, RH - 1], outline=GRID)
    im.paste(right, (1080, 170))
    d.text((1080, 136), "② 站在喇叭正前方看振膜（真實比例）：每個圓點 = 一張照片", fill=INK, font=f(24, True))
    top_share = np.sort(share)[::-1]
    d.text((1080, 1076), "點的位置 = 這張照片的視線從中心旁邊經過的地方；正中十字 = 所有點按份量平衡後的中心。", fill=INK2, font=f(22))
    d.text((1080, 1108), f"近的照片看得清楚，份量大：最近的 10 張佔 {top_share[:10].sum() * 100:.0f}%，最遠的一半只佔 "
                         f"{np.sort(share)[:len(share) // 2].sum() * 100:.0f}%。", fill=INK2, font=f(22))

    # ---- bottom: plain-language result + miss histogram
    d.text((40, 1170), "③ 結果", fill=INK, font=f(26, True))
    d.text((40, 1212), f"每條視線離中心：一半在 {np.median(miss):.1f} mm 以內，90% 在 {np.percentile(miss, 90):.1f} mm 以內。"
                       f"用 {len(obs)} 條一起求「最吻合所有視線的點」，", fill=INK2, font=f(23))
    d.text((40, 1250), f"中心的不確定度約 ±{sd:.1f} mm。離得特別遠的少數幾張（橙框，{int(down.sum())} 張）自動降低份量，不會把結果拉偏。"
                       f"中心：x {Xm[0]:.1f}，y {Xm[1]:.1f}，離地 {Xm[2]:.1f} mm。", fill=INK2, font=f(23))
    out = f"{OUT_DIR}/rays_speaker_{sid}.png"; im.save(out)
    print(out, len(obs), "rays; miss median", round(float(np.median(miss)), 2), "down", int(down.sum()), "thr", round(thr, 1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
