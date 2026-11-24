# WFS 喇叭陣列定位

用照片量測 112 顆波場合成（WFS）喇叭的振膜中心位置與朝向，輸出可直接載入 Panoramix（Max/MSP）的設定檔。

**網站：<https://wangpenghao468-collab.github.io/wfs-speaker-localization/>**
（照片查找、原理教學、結果與驗證、操作手冊）

## 結果（2026-10-13 量測）

| 項目 | 結果 |
|---|---|
| 每顆中心誤差 | 中位數 0.74 mm，90% ≤ 1.17 mm（bootstrap） |
| 架內相鄰間距 | 80.10 ± 0.74 mm（實際 80 mm） |
| 離地高度 | 1115–1130 mm，中位數 1120 mm |
| 與手動設計的差 | 中位數 15 mm，最大 64 mm |
| 使用照片 | 663 張 Sony 照片，每顆中位數約 80 張 |

給 Panoramix 的檔案：[`measurements/2026-10-13/1012_measured.txt`](measurements/2026-10-13/1012_measured.txt)
（設計檔的副本，只改了 `/bus/10/speakers/aed` 與 `/bus/10/speakers/directions/xyz` 兩行）

## 方法

DJI 連續影片建立正確的整體座標並提供人工編號；每張高解析 Sony 照片以 PnP 對齊到 DJI 座標，並以鄰居投票排除左右認錯；喇叭投影到照片中配對振膜偵測，再以幾十張照片做穩健的多視角三角化；尺度由架內 80 mm 決定，地板由 Sony 照片重新擬合，最後以旋轉＋平移對齊設計原點。詳見網站的[原理教學](https://wangpenghao468-collab.github.io/wfs-speaker-localization/learn.html)。

## 結構

```
pipeline/            主流程（run_all.py 依序執行 14 步；config.py 讀 settings.yaml）
tools/               check_env.py（環境檢查）、build_site.py（產生網站）、coarse_annotation_ui.py（標註）
measurements/DATE/   每次量測的結果
docs/                GitHub Pages 網站
settings.example.yaml  路徑設定範本
上傳更新.bat          重新產生網站並推送
```

原始照片、COLMAP 資料庫與快取不在倉庫中，位置由 `settings.yaml` 指定。

## 網頁處理工具（本機）

雙擊 `處理工具.bat`（或 `python tools/webtool/server.py`），瀏覽器會打開本機頁面：

- **3DGS 場景重建**：選照片資料夾 → 選相機位姿來源（COLMAP 自動／已定位的 Sony 位姿／既有模型）、方法（PGSR／原版 3DGS）、步數、解析度、GPU、省記憶體、曝光補償 → 開始。結果是高斯 `point_cloud.ply`，PGSR 另有網格 `tsdf_fusion_post.ply`。
- **新空間定位**：新的房間或聽音區，不需要影片標註。選照片資料夾 → COLMAP → 自動找出所有振膜並分架 → 在俯視圖上設定編號與原點 → 輸出 CSV、Panoramix 兩行、PLY 與畫上編號的檢查照片（`pipeline/auto_locate.py`）。
- **喇叭空間位置**：用目前結果重設原點與 x 軸（幾秒），或加入新照片後重算；輸出 CSV、PLY、Panoramix 兩行及設定檔副本。

只監聽 127.0.0.1，網頁和每個訓練都自動挑沒人用的連接埠，不會和其他本機服務衝突。每個工作有自己的資料夾（`settings.yaml` 的 `jobs_dir`），關掉工具視窗工作仍會繼續；失敗可「重試」，已完成的步驟會跳過。

## 執行

```bash
pip install numpy opencv-python pillow pyyaml torch
copy settings.example.yaml settings.yaml     # 修改成你的路徑
python tools/check_env.py                    # 檢查環境與資料
python pipeline/run_all.py                   # 全部重算
python tools/build_site.py                   # 更新網站
```

詳細說明見網站的[操作手冊](https://wangpenghao468-collab.github.io/wfs-speaker-localization/manual.html)。
