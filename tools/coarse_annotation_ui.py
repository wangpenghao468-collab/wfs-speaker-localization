"""Local, separate UI for coarse speaker identity boxes (no SAM masks)."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import gradio as gr
from PIL import Image


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from config import DJI_IMAGES, COARSE_DIR  # paths from settings.yaml

IMAGE_DIR = Path(DJI_IMAGES)
OUTPUT_DIR = Path(COARSE_DIR)
ORIGINALS_DIR = OUTPUT_DIR / "original_images"
OVERLAYS_DIR = OUTPUT_DIR / "original_size_overlays"
ANNOTATIONS_JSON = OUTPUT_DIR / "speaker_boxes.json"
ANNOTATIONS_CSV = OUTPUT_DIR / "speaker_boxes.csv"


def load_records() -> dict[str, list[dict]]:
    if not ANNOTATIONS_JSON.exists():
        return {}
    try:
        data = json.loads(ANNOTATIONS_JSON.read_text(encoding="utf-8"))
        return data.get("images", {}) if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


RECORDS = load_records()
IMAGE_NAMES = sorted(p.name for p in IMAGE_DIR.glob("*.jpg"))


def html_view(image_name: str, boxes: list[dict]) -> str:
    path = IMAGE_DIR / image_name
    with Image.open(path) as im:
        width, height = im.size
        import base64
        import io

        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=94)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    safe_boxes = json.dumps(boxes, ensure_ascii=False)
    return f"""<div class="box-annotator" data-src="data:image/jpeg;base64,{encoded}"
      data-width="{width}" data-height="{height}" data-boxes='{safe_boxes}'>
      <div class="box-toolbar">拖曳框住一顆喇叭；滾輪以游標位置縮放；按住空白鍵拖曳可平移。框選後選序號並按「加入框」，可在同一張照片繼續標多顆。</div>
      <div class="box-viewport"><canvas class="box-canvas"></canvas></div>
      <div class="box-hint">座標以原圖像素儲存。這是粗框標註，不會生成或修改 SAM 遮罩。</div>
    </div>"""


HEAD = r"""
<style>
.box-annotator{font-family:system-ui,sans-serif;color:#243047}.box-toolbar{padding:8px 10px;background:#f3f6fb;border-radius:8px;margin-bottom:8px;font-size:14px}
.box-viewport{height:min(68vh,760px);min-height:420px;overflow:auto;background:#20242b;border-radius:8px;position:relative}
.box-canvas{display:block;max-width:none;cursor:crosshair;touch-action:none}.box-hint{font-size:12px;color:#64748b;padding:6px 2px}
.box-json-transport{display:none!important}
</style>
<script>
(() => {
  let active = null;
  const init = (root) => {
    const canvas = root.querySelector('.box-canvas');
    const viewport = root.querySelector('.box-viewport');
    if (!canvas || canvas.dataset.ready) return;
    canvas.dataset.ready = '1'; active = canvas;
    const ctx = canvas.getContext('2d');
    const img = new Image(); img.src = root.dataset.src;
    const W = Number(root.dataset.width), H = Number(root.dataset.height);
    const boxes = JSON.parse(root.dataset.boxes || '[]');
    const state = {boxes, draft:null, pending:null, scale:1, pan:false, sx:0, sy:0, scrollL:0, scrollT:0};
    window.__speakerBoxDraft = null;
    const publishDraft = value => {window.__speakerBoxDraft=value;const wrap=document.querySelector('#speaker-box-draft');const input=wrap&&wrap.querySelector('textarea,input');if(input){input.value=JSON.stringify(value);input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new Event('change',{bubbles:true}));}};
    publishDraft(null);
    window.__speakerBoxFrame = root.dataset.frame || '';
    canvas.width=W; canvas.height=H;
    const resize = () => {
      const fit = Math.min(1, (viewport.clientWidth-4)/W, (viewport.clientHeight-4)/H);
      state.scale = fit; canvas.style.width=(W*fit)+'px'; canvas.style.height=(H*fit)+'px'; paint();
    };
    const paint = () => {
      if (!img.complete) return;
      ctx.clearRect(0,0,W,H); ctx.drawImage(img,0,0,W,H);
      for (const b of state.boxes) {
        ctx.lineWidth=Math.max(2,W/900);ctx.strokeStyle='#00e676';ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);
        const fs=22/state.scale,pad=5/state.scale,label=`#${b.speaker_id}`;ctx.font=`bold ${fs}px sans-serif`;
        const tw=ctx.measureText(label).width,bh=fs+2*pad,bw=tw+2*pad,ly=b.y1>=bh?b.y1-bh:b.y1;
        ctx.fillStyle='rgba(12,18,28,.92)';ctx.fillRect(b.x1,ly,bw,bh);ctx.fillStyle='#ffeb3b';ctx.textBaseline='middle';ctx.fillText(label,b.x1+pad,ly+bh/2);ctx.textBaseline='alphabetic';
      }
      if (state.pending) {const b=state.pending;ctx.lineWidth=Math.max(3,W/700);ctx.strokeStyle='#ffeb3b';ctx.fillStyle='rgba(255,235,59,.12)';ctx.fillRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);ctx.setLineDash([12,8]);ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);ctx.setLineDash([]);ctx.font=`bold ${Math.max(18,W/55)}px sans-serif`;ctx.fillStyle='#ffeb3b';ctx.fillText('待加入',b.x1+4,b.y1+Math.max(21,W/52));}
      if (state.draft) {ctx.lineWidth=Math.max(2,W/900);ctx.setLineDash([8,6]);ctx.strokeStyle='#ffeb3b';ctx.strokeRect(state.draft.x1,state.draft.y1,state.draft.x2-state.draft.x1,state.draft.y2-state.draft.y1);ctx.setLineDash([]);}
    };
    img.onload=()=>{resize();};
    window.addEventListener('resize',resize);
    const coords=e=>{const r=canvas.getBoundingClientRect();return {x:Math.max(0,Math.min(W,(e.clientX-r.left)*W/r.width)),y:Math.max(0,Math.min(H,(e.clientY-r.top)*H/r.height))};};
    canvas.addEventListener('pointerdown',e=>{if(e.button===1||e.buttons===4||e.getModifierState(' ')){state.pan=true;state.sx=e.clientX;state.sy=e.clientY;state.scrollL=viewport.scrollLeft;state.scrollT=viewport.scrollTop;canvas.setPointerCapture(e.pointerId);return;}state.pending=null;publishDraft(null);const p=coords(e);state.draft={x1:p.x,y1:p.y,x2:p.x,y2:p.y};canvas.setPointerCapture(e.pointerId);paint();});
    canvas.addEventListener('pointermove',e=>{if(state.pan){viewport.scrollLeft=state.scrollL-(e.clientX-state.sx);viewport.scrollTop=state.scrollT-(e.clientY-state.sy);return;}if(!state.draft)return;const p=coords(e);state.draft.x2=p.x;state.draft.y2=p.y;paint();});
    canvas.addEventListener('pointerup',()=>{if(state.pan){state.pan=false;return;}if(!state.draft)return;let b=state.draft;state.draft=null;let x1=Math.min(b.x1,b.x2),y1=Math.min(b.y1,b.y2),x2=Math.max(b.x1,b.x2),y2=Math.max(b.y1,b.y2);if(x2-x1>=5&&y2-y1>=5){state.pending={x1:Math.round(x1),y1:Math.round(y1),x2:Math.round(x2),y2:Math.round(y2)};publishDraft(state.pending);}paint();});
    canvas.addEventListener('wheel',e=>{e.preventDefault();const old=state.scale;const rect=canvas.getBoundingClientRect();const px=e.clientX-rect.left,py=e.clientY-rect.top;const next=Math.max(Math.min(1,(viewport.clientWidth-4)/W,(viewport.clientHeight-4)/H),Math.min(5,old*(e.deltaY<0?1.2:1/1.2)));if(next===old)return;const imageX=px/old,imageY=py/old;state.scale=next;canvas.style.width=(W*next)+'px';canvas.style.height=(H*next)+'px';viewport.scrollLeft=(rect.left-viewport.getBoundingClientRect().left+viewport.scrollLeft)+imageX*next-px;viewport.scrollTop=(rect.top-viewport.getBoundingClientRect().top+viewport.scrollTop)+imageY*next-py;paint();},{passive:false});
    canvas.addEventListener('dblclick',()=>{window.__speakerBoxDraft=null;});
    window.__speakerBoxes = boxes;
  };
  const scan=()=>document.querySelectorAll('.box-annotator').forEach(init);
  new MutationObserver(scan).observe(document.documentElement,{childList:true,subtree:true});
  document.addEventListener('DOMContentLoaded',scan);setTimeout(scan,100);setTimeout(scan,500);
})();
</script>
"""


def current_markup(image_name: str, records: dict[str, list[dict]]) -> str:
    return html_view(image_name, records.get(image_name, []))


def add_box(image_name: str, speaker_id: int, draft_json: str, records: dict[str, list[dict]]):
    records = {k: list(v) for k, v in (records or {}).items()}
    try:
        box = json.loads(draft_json or "null")
    except json.JSONDecodeError:
        box = None
    if not box:
        return records, current_markup(image_name, records), "請先在照片上拖曳畫出喇叭框。", gr.update(value="")
    if not 1 <= int(speaker_id) <= 112:
        return records, current_markup(image_name, records), "序號需介於 1–112。", gr.update(value="")
    entries = records.setdefault(image_name, [])
    if any(int(x["speaker_id"]) == int(speaker_id) for x in entries):
        return records, current_markup(image_name, records), f"這張照片已經有 {speaker_id} 號；請刪除舊框或改用其他序號。", gr.update(value="")
    with Image.open(IMAGE_DIR / image_name) as im:
        w, h = im.size
    x1, y1 = max(0, int(box["x1"])), max(0, int(box["y1"]))
    x2, y2 = min(w, int(box["x2"])), min(h, int(box["y2"]))
    if x2 <= x1 or y2 <= y1:
        return records, current_markup(image_name, records), "框選範圍無效，請重新框選。", gr.update(value="")
    entries.append({"speaker_id": int(speaker_id), "bbox_xyxy": [x1, y1, x2, y2], "visibility": "visible"})
    entries.sort(key=lambda x: int(x["speaker_id"]))
    msg = f"已加入喇叭 {speaker_id}；本張目前 {len(entries)} 個框。可繼續框下一顆，完成後按儲存。"
    return records, current_markup(image_name, records), msg, gr.update(value="")


def remove_last(image_name: str, records: dict[str, list[dict]]):
    records = {k: list(v) for k, v in (records or {}).items()}
    entries = records.get(image_name, [])
    if entries:
        entries.pop()
    return records, current_markup(image_name, records), f"已撤回最後一個框；本張目前 {len(entries)} 個框。"


def save_boxes(image_name: str, records: dict[str, list[dict]]):
    records = {k: list(v) for k, v in (records or {}).items()}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    source = IMAGE_DIR / image_name
    target = ORIGINALS_DIR / image_name
    if not target.exists():
        shutil.copy2(source, target)
    # Store one full-resolution review image per annotated source photo.
    from PIL import ImageDraw
    with Image.open(source) as im:
        canvas = im.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        line_w = max(3, round(canvas.width / 700))
        for item in records.get(image_name, []):
            x1, y1, x2, y2 = item["bbox_xyxy"]
            sid = int(item["speaker_id"])
            draw.rectangle((x1, y1, x2, y2), outline=(0, 230, 118), width=line_w)
            font_size = max(36, canvas.width // 55)
            try:
                from PIL import ImageFont
                font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", font_size)
            except OSError:
                from PIL import ImageFont
                font = ImageFont.load_default()
            label = f"#{sid}"
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font, stroke_width=max(1, line_w // 2))
            label_w, label_h = right - left + 20, bottom - top + 14
            label_y = max(0, y1 - label_h) if y1 >= label_h else y1
            draw.rounded_rectangle((x1, label_y, x1 + label_w, label_y + label_h), radius=max(5, line_w * 2), fill=(12, 18, 28), outline=(255, 235, 59), width=max(2, line_w // 2))
            draw.text((x1 + 10, label_y + 4), label, font=font, fill=(255, 235, 59), stroke_width=max(1, line_w // 3), stroke_fill=(0, 0, 0))
        canvas.save(OVERLAYS_DIR / f"{Path(image_name).stem}_boxes.jpg", quality=95)
        width, height = canvas.size
    payload = {
        "schema": "speaker_coarse_boxes_v1",
        "coordinate_system": "original_image_pixels_xyxy",
        "image_root": str(IMAGE_DIR),
        "images": records,
    }
    ANNOTATIONS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with ANNOTATIONS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_file", "speaker_id", "x1", "y1", "x2", "y2", "image_width", "image_height", "label_type"])
        writer.writeheader()
        for fname, items in sorted(records.items()):
            with Image.open(IMAGE_DIR / fname) as im:
                iw, ih = im.size
            for item in items:
                x1, y1, x2, y2 = item["bbox_xyxy"]
                writer.writerow({"image_file": fname, "speaker_id": item["speaker_id"], "x1": x1, "y1": y1, "x2": x2, "y2": y2, "image_width": iw, "image_height": ih, "label_type": "coarse_box_no_mask"})
    total = sum(len(v) for v in records.values())
    return records, f"已儲存 {image_name}：{len(records.get(image_name, []))} 個框；全部 {total} 個框。原圖副本與原尺寸框選檢視圖已分開存放。"


def select_image(image_name: str, records: dict[str, list[dict]]):
    boxes = (records or {}).get(image_name, [])
    return current_markup(image_name, records or {}), f"{image_name}：已有 {len(boxes)} 個框。"


with gr.Blocks(title="喇叭序號粗標註", head=HEAD) as demo:
    gr.Markdown("# 喇叭序號粗標註（非 Mask）\n框選只記錄喇叭身分與位置，不呼叫 SAM 2、不產生遮罩。每張照片可標多顆；所有座標都換算成原圖像素。")
    records_state = gr.State(RECORDS)
    with gr.Row():
        image_select = gr.Dropdown(choices=IMAGE_NAMES, value=IMAGE_NAMES[0] if IMAGE_NAMES else None, label="原始訓練照片", scale=4)
        speaker_select = gr.Number(value=1, minimum=1, maximum=112, precision=0, label="喇叭序號（1–112）", scale=1)
    box_canvas = gr.HTML(value=current_markup(IMAGE_NAMES[0], RECORDS) if IMAGE_NAMES else "找不到訓練照片。", elem_id="speaker-box-view")
    with gr.Row():
        draft_json = gr.Textbox(value="", label="框選座標", elem_id="speaker-box-draft", elem_classes=["box-json-transport"])
        add_button = gr.Button("加入目前框", variant="primary")
        undo_button = gr.Button("撤回本張最後一框")
        save_button = gr.Button("儲存本張標註")
    status = gr.Markdown("選照片後拖曳框選；滾輪可縮放。")
    gr.Markdown(f"**粗標註專用路徑**\n\n- 框座標與清單：`{ANNOTATIONS_JSON}`、`{ANNOTATIONS_CSV}`\n- 原始尺寸原圖副本：`{ORIGINALS_DIR}`\n- 疊框檢視圖（仍為原始尺寸）：`{OVERLAYS_DIR}`\n\n不會寫入 `masks` 資料夾，也不會裁切或覆寫原圖。")

    image_select.change(select_image, inputs=[image_select, records_state], outputs=[box_canvas, status])
    add_button.click(add_box, inputs=[image_select, speaker_select, draft_json, records_state], outputs=[records_state, box_canvas, status, draft_json])
    undo_button.click(remove_last, inputs=[image_select, records_state], outputs=[records_state, box_canvas, status])
    save_button.click(save_boxes, inputs=[image_select, records_state], outputs=[records_state, status])


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7861, inbrowser=True, share=False)
