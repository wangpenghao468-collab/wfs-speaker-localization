"""Overview flowchart: from the photos to the Panoramix txt (what each step does and
which problem it solves)."""
from PIL import Image, ImageDraw, ImageFont
from config import OUT_DIR

from config import FONT
SURF, INK, INK2, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
FILL = {"you": "#fdeee4", "geo": "#e8f0fb", "out": "#e6f4ea", "check": "#f1f0ec"}
EDGE = {"you": "#eb6a1e", "geo": "#2a78d6", "out": "#1a9e5c", "check": "#8a8984"}


def f(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=1 if bold else 0)


STEPS = [
    ("you", "① 拍照", "DJI 環繞影片（286 幀）＋ Sony 相機 938 張照片",
     "提供兩組素材：DJI 連續、方便重建；Sony 解析度高、適合量測"),
    ("you", "② 你標註喇叭編號", "在 DJI 幀上用 SAM2 遮罩和粗框標出部分喇叭的號碼",
     "112 顆喇叭長得一模一樣，必須有人給出「誰是幾號」；frame_00059 標錯側已更正"),
    ("geo", "③ 建立 DJI 三維模型（COLMAP）", "用連續影片算出每一幀的相機位置",
     "Sony 自己的模型把左右兩側「摺疊」在一起（兩側太像），DJI 連續影片沒有這個問題，所以以它為座標基準"),
    ("geo", "④ 112 顆喇叭的初始位置", "把你的標註在 DJI 中三角化，再用「8 顆一架、等距共線」補齊",
     "得到每一顆喇叭的編號和大概位置，作為後面配對照片的依據"),
    ("geo", "⑤ 每張 Sony 照片定位到同一座標", "特徵匹配到 DJI 幀＋PnP，求出每張照片的相機位置",
     "繞過 Sony 的摺疊模型；COLMAP 沒接上的 542 張也能用（1–8 號的照片幾乎都在其中）"),
    ("geo", "⑥ 鄰居投票防止左右錯置", "每張照片和它的匹配鄰居互相確認相機位置",
     "近景特寫分不出左右兩側（例如 DSC03829 被定到對面），投票把錯的改回來；663 張取得可靠位置"),
    ("geo", "⑦ 偵測振膜、對上編號", "在每張照片找出銀色振膜，和投影過去的喇叭配對",
     "每顆喇叭得到 25–118 張確認照片 → 照片查找網頁；大小檢查排除螺絲反光"),
    ("geo", "⑧ 多張照片求振膜中心", "求最吻合所有視線的點：近的照片份量大，偏的降權，相機位置算錯的整張排除",
     "單張照片只能給方向；幾十條視線交會才得到 mm 級的位置（±0.7 mm）"),
    ("geo", "⑨ 公釐比例與地板", "用架內 8 cm 校正比例；用照片配對三角化地磚點擬合地板",
     "得到真實尺寸和離地高度；修正 DJI 點雲裡比地面高 8 cm 的假地板"),
    ("out", "⑩ 對齊你的原點，輸出 txt", "實測位置對齊到你的 Panoramix 設計；每根架子的直線給出朝向",
     "1012_measured.txt：只改 aed（位置）與 directions（朝向）兩行，可直接載入"),
]
CHECKS = [("架內間距", "79.9 ± 0.8 mm（實際 80）"), ("架間距離", "約 91 mm（你量約 9.0 cm）"),
          ("架子直線度", "8 顆偏離直線 0.6 mm"), ("編號抽查", "每架 6 張照片拼圖，背景一致"),
          ("你的回報", "DSC04160、frame_00059 已修正")]


def main():
    W, BH, GAP, X0, BW = 2000, 150, 46, 60, 1440
    H = 170 + len(STEPS) * (BH + GAP) + 40
    im = Image.new("RGB", (W, H), SURF); d = ImageDraw.Draw(im)
    d.text((X0, 30), "從拍照到 Panoramix txt：做了什麼、解決了什麼", fill=INK, font=f(42, True))
    lx = X0
    for k, (lab, txt) in enumerate((("you", "你提供"), ("geo", "幾何計算"), ("out", "輸出"), ("check", "驗證"))):
        d.rectangle([lx, 98, lx + 30, 122], fill=FILL[lab], outline=EDGE[lab], width=3)
        d.text((lx + 40, 94), txt, fill=INK2, font=f(22)); lx += 180
    y = 160
    centres = []
    for kind, title, what, why in STEPS:
        d.rounded_rectangle([X0, y, X0 + BW, y + BH], radius=14, fill=FILL[kind], outline=EDGE[kind], width=3)
        d.text((X0 + 24, y + 14), title, fill=INK, font=f(28, True))
        d.text((X0 + 24, y + 62), "做了：" + what, fill=INK, font=f(22))
        d.text((X0 + 24, y + 102), "解決：" + why, fill=INK2, font=f(21))
        centres.append(y + BH / 2)
        if y > 160:
            ax = X0 + 60
            d.line([(ax, y - GAP + 4), (ax, y - 6)], fill=LINE, width=4)
            d.polygon([(ax - 10, y - 18), (ax + 10, y - 18), (ax, y - 4)], fill=LINE)
        y += BH + GAP
    # verification column, linked to the measurement steps
    cx0 = X0 + BW + 60; cy0 = 160 + 6 * (BH + GAP)
    d.rounded_rectangle([cx0, cy0, W - 40, cy0 + 3 * BH + 2 * GAP], radius=14, fill=FILL["check"], outline=EDGE["check"], width=3)
    d.text((cx0 + 20, cy0 + 16), "獨立驗證", fill=INK, font=f(28, True))
    for k, (head, body) in enumerate(CHECKS):
        yy = cy0 + 70 + k * 84
        cx, cyy = cx0 + 30, yy + 14                       # drawn check mark (no glyph needed)
        d.line([(cx - 9, cyy), (cx - 2, cyy + 8), (cx + 11, cyy - 9)], fill=EDGE["out"], width=4)
        d.text((cx0 + 54, yy), head, fill=INK, font=f(22, True))
        d.text((cx0 + 54, yy + 32), body, fill=INK2, font=f(20))
    for c in centres[6:9]:
        d.line([(X0 + BW, c), (cx0, c)], fill=LINE, width=2)
    out = f"{OUT_DIR}/pipeline_overview.png"; im.save(out); print(out)


if __name__ == "__main__":
    main()
