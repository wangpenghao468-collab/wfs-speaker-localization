"""Figures for the thesis (JPEG, 300 dpi, white background).

    python tools/paper_figures.py

Writes to <repo>/paper_figures/:
  fig1_pipeline.jpg            the complete localisation workflow, methods and key numbers per stage
  fig2_diaphragm_detection.jpg how a diaphragm is found and identified in one Sony photo (real data)"""
import json, os, sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "pipeline"))
from config import WORK, OUT_DIR, SONY_IMAGES, FONT  # noqa: E402

OUT = os.path.join(REPO, "paper_figures"); os.makedirs(OUT, exist_ok=True)
INK, MUTED, LINE = (29, 29, 27), (95, 94, 90), (190, 188, 182)
C_IN, C_GEO, C_ID, C_OUT, C_VAL = (234, 240, 250), (232, 244, 236), (253, 241, 229), (243, 236, 250), (248, 248, 246)
E_IN, E_GEO, E_ID, E_OUT = (42, 120, 214), (26, 158, 92), (235, 106, 30), (130, 80, 200)


def font(sz, bold=False):
    try:
        return ImageFont.truetype(FONT, sz, index=1 if bold else 0)
    except Exception:
        return ImageFont.truetype(FONT, sz)


def text_w(d, t, f):
    return d.textlength(t, font=f)


def wrap(d, t, f, w):
    lines, cur = [], ""
    for ch in t:
        if ch == "\n":
            lines.append(cur); cur = ""; continue
        if text_w(d, cur + ch, f) > w and cur:
            lines.append(cur); cur = ch
        else:
            cur += ch
    return lines + ([cur] if cur else [])


def box(d, xy, fill, edge, r=18, w=3):
    d.rounded_rectangle(xy, r, fill=fill, outline=edge, width=w)


def arrow(d, x, y0, y1, col=MUTED, w=5):
    d.line([(x, y0), (x, y1 - 16)], fill=col, width=w)
    d.polygon([(x - 13, y1 - 20), (x + 13, y1 - 20), (x, y1)], fill=col)


# ---------------------------------------------------------------- figure 1
STAGES = [
    ("輸入資料", C_IN, E_IN, [
        ("DJI 影片幀", "286 幀，2560×1919；連續拍攝，提供正確的整體形狀"),
        ("Sony 照片", "ILCE-7M4，944 張，7008×3944；高解析度，用於量測"),
        ("人工標註（DJI 幀）", "SAM2 振膜遮罩 144 筆、粗框 113 個；提供喇叭編號"),
        ("Panoramix 設計檔", "1012.txt：設計原點、方位角、距離、朝向"),
    ]),
    ("① 參考座標系（DJI 運動恢復結構）", C_GEO, E_GEO, [
        ("方法", "COLMAP 3.11：SIFT 特徵、匹配、增量式 SfM、光束法平差；PINHOLE 共用相機"),
        ("結果", "286 個相機位姿、42,129 個 3D 點，平均重投影誤差 0.81 px"),
    ]),
    ("② 喇叭身份與尺度", C_ID, E_ID, [
        ("標註三角化", "多幀 DLT 三角化人工標註的振膜，得到已標註喇叭的 3D 位置"),
        ("陣列模型", "每架 8 顆共線等距，RANSAC 擬合 14 架 → 112 顆初始位置（公分級）"),
        ("尺度", "架內間距 80 mm → 1 模型單位 = 345.9 mm"),
    ]),
    ("③ Sony 相機定位（對齊 DJI 座標）", C_GEO, E_GEO, [
        ("2D–3D 對應", "RootSIFT（3000 px）與 8 張最相似 DJI 幀做 GPU 互為最近鄰 + 比值測試"),
        ("PnP", "AP3P / SQPnP RANSAC + Levenberg–Marquardt 精修"),
        ("鄰居投票", "共視 ≧ 40 點或驗證匹配 ≧ 60 的照片互相檢驗相對旋轉與位置，排除左右錯置"),
        ("結果", "938 張中 663 張取得可靠位姿"),
    ]),
    ("④ 振膜偵測與身份辨識（每張 Sony 照片）", C_ID, E_ID, [
        ("偵測", "HSV 門檻（V > 140、S < 60）→ 連通區域 → 圓度 ≧ 0.6、暗色外圈 ≧ 35% → 質心 → 去畸變"),
        ("辨識", "將 112 顆初始位置投影到照片，在 0.30 → 0.25 間距內配對最近偵測；半徑 0.11–0.30 間距、喇叭完整在畫面內"),
    ]),
    ("⑤ 多視角三角化振膜中心", C_GEO, E_GEO, [
        ("估計", "DLT 初值 → Gauss–Newton 最小化重投影誤差；自適應 Huber 權重（2.45σ）"),
        ("篩選", "視角 ≦ 75°、清晰度 ≧ 50；整張位姿偏移的照片（> 25 px 且 > 3 倍典型值）排除"),
        ("結果", "112 顆，每顆約 25–110 張照片（中位數約 80）"),
    ]),
    ("⑥ 地板平面與高度", C_GEO, E_GEO, [
        ("方法", "Sony 照片兩兩三角化地磚點（極線門檻 3 px）→ 穩健平面擬合（Huber 加權）"),
        ("結果", "振膜中心離地 1115–1130 mm（中位數 1120 mm）"),
    ]),
    ("⑦ 對齊設計原點並輸出", C_OUT, E_OUT, [
        ("對齊", "水平剛體轉換（旋轉 + 平移，不縮放）最小化與設計位置的差；旋轉 0.23°"),
        ("朝向", "每架 8 顆中心擬合直線，法線方向即喇叭朝向"),
        ("輸出", "Panoramix /bus/10/speakers/aed 與 directions/xyz（仰角 0、ignorez 1）"),
    ]),
]
VALIDATION = [
    ("Bootstrap（200 次）", "中位數 0.74 mm\n90% ≦ 1.17 mm"),
    ("架內相鄰間距", "80.10 ± 0.74 mm\n（實際 80 mm）"),
    ("直線度", "離擬合直線\n中位數 0.6 mm"),
    ("留一驗證", "中位數 0.8 mm"),
    ("架間距離", "88–93 mm\n（人工量測約 90 mm）"),
    ("與設計差異", "中位數 15 mm\n最大 64 mm"),
    ("WFS 時間誤差", "RMS 31 µs → 2.1 µs\n（設計值 → 實測值）"),
]


def fig1():
    W, M = 3000, 70
    main_w, side_x = 2080, 2230
    img = Image.new("RGB", (W, 6000), "white"); d = ImageDraw.Draw(img)
    fT, fH, fK, fV, fS = font(62, True), font(40, True), font(31, True), font(31), font(27)
    d.text((M, 50), "WFS 喇叭陣列光學定位流程", font=fT, fill=INK)
    d.text((M, 132), "從 DJI 影片、Sony 照片與人工標註，得到 112 顆振膜中心的 3D 位置與朝向", font=fS, fill=MUTED)
    y = 220; key_w = 300; pad = 26; centres = []
    for title, fill, edge, rows in STAGES:
        lines = [(k, wrap(d, v, fV, main_w - key_w - 2 * pad - 20)) for k, v in rows]
        h = 30 + 56 + sum(max(1, len(v)) * 44 + 10 for _, v in lines) + 18
        box(d, (M, y, M + main_w, y + h), fill, edge)
        d.rectangle((M + 3, y + 3, M + 16, y + h - 3), fill=edge)
        d.text((M + pad + 6, y + 24), title, font=fH, fill=INK)
        yy = y + 90
        for k, v in lines:
            d.text((M + pad + 6, yy), k, font=fK, fill=edge)
            for j, l in enumerate(v):
                d.text((M + pad + key_w, yy + j * 44), l, font=fV, fill=INK)
            yy += max(1, len(v)) * 44 + 10
        centres.append((y, y + h))
        y += h + 56
    for (a0, a1), (b0, _) in zip(centres, centres[1:]):
        arrow(d, M + main_w // 2, a1 + 4, b0 - 4)
    # side: what each source contributes + validation
    top = centres[1][0]
    step = lambda v: 42 + 38 * len(v.split("\n")) + 34
    vb_h = 92 + sum(step(v) for _, v in VALIDATION)
    box(d, (side_x, top, W - M, top + vb_h), C_VAL, LINE, w=3)
    d.text((side_x + 26, top + 22), "精度驗證", font=fH, fill=INK)
    yy = top + 92
    for k, v in VALIDATION:
        d.text((side_x + 26, yy), k, font=fK, fill=E_GEO)
        for j, l in enumerate(v.split("\n")):
            d.text((side_x + 26, yy + 42 + j * 38), l, font=fV, fill=INK)
        yy += step(v)
    ly = top + vb_h + 50
    d.text((side_x, ly), "圖例", font=fK, fill=INK); ly += 50
    for col, edge, lab in [(C_IN, E_IN, "輸入資料"), (C_GEO, E_GEO, "幾何計算"), (C_ID, E_ID, "喇叭身份"), (C_OUT, E_OUT, "輸出")]:
        box(d, (side_x, ly, side_x + 46, ly + 34), col, edge, r=6, w=3); d.text((side_x + 62, ly - 2), lab, font=fV, fill=INK); ly += 52
    ly += 20
    note = "DJI：提供整體座標與喇叭編號（影像不參與量測）\nSony：提供量測用的振膜像素位置\n尺度：只依賴架內 80 mm"
    for j, l in enumerate(note.split("\n")):
        for k, ll in enumerate(wrap(d, l, fS, W - M - side_x)):
            d.text((side_x, ly), ll, font=fS, fill=MUTED); ly += 38
        ly += 8
    H = max(y, ly) + 20
    img = img.crop((0, 0, W, H))
    img.save(os.path.join(OUT, "fig1_pipeline.jpg"), quality=95, dpi=(300, 300))
    print("wrote fig1_pipeline.jpg", img.size)


# ---------------------------------------------------------------- figure 2
def fig2(photo="DSC03610.JPG", target=20):
    sys.path.insert(0, os.path.join(REPO, "pipeline"))
    from detect_diaphragms import detect, DET_W
    from eval_poses import load, project
    img = cv2.imread(os.path.join(SONY_IMAGES, photo))
    sc = DET_W / img.shape[1]; small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV); v, s = hsv[..., 2], hsv[..., 1]
    bright = ((v > 140) & (s < 60)).astype(np.uint8) * 255
    dets = detect(small)
    ids, S, nrm, pitch = load()
    poses = json.load(open(f"{WORK}/sony_poses_final.json")); p = poses[photo]
    F = json.load(open(f"{WORK}/speakers_final.json"))
    uv, z = project(p, np.array([S[i] for i in ids]))
    fx = p["K"][0][0]
    # crop around the target speaker, ~4 pitches wide
    k = ids.index(target); cx, cy = uv[k] * sc; span = 4.2 * pitch * fx / z[k] * sc
    x0, y0 = int(cx - span / 2), int(cy - span * 0.32); x1, y1 = int(cx + span / 2), int(cy + span * 0.32)
    x0, y0 = max(0, x0), max(0, y0)
    crop = lambda a: a[y0:y1, x0:x1]
    T = 900; zoom = T / (x1 - x0)
    def up(a):
        return cv2.resize(crop(a), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST if a.ndim == 2 else cv2.INTER_CUBIC)
    P = lambda x, y: (int((x - x0) * zoom), int((y - y0) * zoom))
    a = up(small)
    b = cv2.cvtColor(up(bright), cv2.COLOR_GRAY2BGR)
    c = a.copy()
    for d_ in dets:
        x, y = d_["xy"]; r = d_["r"]
        cv2.circle(c, P(x, y), int(r * zoom * 1.0), (60, 200, 60), 3)
        R = int(r * 1.8 * zoom); q = P(x, y); cv2.rectangle(c, (q[0] - R, q[1] - R), (q[0] + R, q[1] + R), (40, 140, 255), 2)
    dd = a.copy()
    for i, sid in enumerate(ids):
        if z[i] <= 0:
            continue
        q = P(*(uv[i] * sc))
        if 0 <= q[0] < dd.shape[1] and 0 <= q[1] < dd.shape[0]:
            g = int(0.25 * pitch * fx / z[i] * sc * zoom)
            cv2.circle(dd, q, g, (30, 120, 245), 3)
            cv2.putText(dd, str(sid), (q[0] - 18, q[1] - g - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 120, 245), 2, cv2.LINE_AA)
    for d_ in dets:
        cv2.drawMarker(dd, P(*d_["xy"]), (60, 200, 60), cv2.MARKER_CROSS, 22, 3)
    panels = [(a, "(a) Sony 照片（縮至 2400 px 偵測）"), (b, "(b) 亮且不飽和的像素：V > 140、S < 60"),
              (c, "(c) 形狀篩選：面積、長寬比、圓度 ≧ 0.6、周圍暗色 ≧ 35%（綠圈；橙框 = 檢查暗色的範圍）。螺絲等小亮點也會通過，留到 (d) 以大小排除"),
              (dd, "(d) 身份：112 顆初始 3D 位置投影到照片（橙圈 = 配對範圍 0.25 間距，數字 = 編號）；綠十字 = 偵測到的質心。只接受半徑 0.11–0.30 間距的偵測，所以螺絲不會被配對")]
    ph = max(pp.shape[0] for pp, _ in panels); pw = max(pp.shape[1] for pp, _ in panels)
    W = pw * 2 + 3 * 40; Hh = 200 + 2 * (ph + 160) + 330
    canvas = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(canvas)
    dr.text((40, 40), "振膜偵測與身份辨識（不需訓練模型）", font=font(54, True), fill=INK)
    dr.text((40, 112), f"例：照片 {photo[:-4]}，第 {target} 號喇叭附近", font=font(30), fill=MUTED)
    for n, (pp, lab) in enumerate(panels):
        gx, gy = 40 + (n % 2) * (pw + 40), 190 + (n // 2) * (ph + 160)
        canvas.paste(Image.fromarray(cv2.cvtColor(pp, cv2.COLOR_BGR2RGB)), (gx, gy))
        for j, l in enumerate(wrap(dr, lab, font(28), pw)):
            dr.text((gx, gy + pp.shape[0] + 12 + j * 36), l, font=font(28), fill=INK)
    ty = 190 + 2 * (ph + 160) + 10
    notes = ["偵測只用亮度、飽和度、形狀與周圍暗色外框等固定規則（OpenCV），沒有訓練任何神經網路。",
             "編號不是從影像外觀判斷（112 顆外觀相同），而是由人工在 DJI 幀上的標註、經 3D 投影傳到每張 Sony 照片。",
             "SAM2 只用於人工標註工具（預訓練模型，未再訓練）；振膜中心的量測全部來自 Sony 照片的質心。"]
    for j, l in enumerate(notes):
        for k2, ll in enumerate(wrap(dr, "• " + l, font(29), W - 80)):
            dr.text((40, ty), ll, font=font(29), fill=INK); ty += 40
        ty += 8
    canvas = canvas.crop((0, 0, W, ty + 30))
    canvas.save(os.path.join(OUT, "fig2_diaphragm_detection.jpg"), quality=95, dpi=(300, 300))
    print("wrote fig2_diaphragm_detection.jpg", canvas.size)


if __name__ == "__main__":
    fig1(); fig2()
