"""User speaker-ID annotations on DJI frames (SAM2 diaphragm masks + coarse boxes)."""
import csv, json, cv2, numpy as np
from config import MASK_DIR, COARSE_DIR


def load_annotations():
    """{(dji_image, speaker_id): dict(xy, r, src)} ; masks override coarse boxes."""
    ann = {}
    js = json.load(open(f"{COARSE_DIR}/speaker_boxes.json", encoding="utf-8"))
    for img, boxes in js["images"].items():
        for b in boxes:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            ann[(img, int(b["speaker_id"]))] = dict(xy=((x1 + x2) / 2, (y1 + y2) / 2),
                                                     r=min(x2 - x1, y2 - y1) / 2 * 0.6, src="coarse_box")
    with open(f"{MASK_DIR}/manual_mask_annotations.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            m = cv2.imread(f"{MASK_DIR}/{row['mask_file']}", cv2.IMREAD_GRAYSCALE)
            if m is None or (m > 127).sum() < 10:
                continue
            ys, xs = np.nonzero(m > 127)
            ann[(row["image_file"], int(row["speaker_id"]))] = dict(
                xy=(xs.mean(), ys.mean()), r=float(np.sqrt(len(xs) / np.pi)), src="sam2_mask")
    return ann
