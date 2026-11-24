"""New-room speaker localisation job: photos -> (COLMAP) -> diaphragms in 3D -> bars, then waits for the
speaker numbers (given in the web tool, which runs `auto_locate.py assign`).

    python tools/webtool/locate_job.py JOB_DIR"""
import json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
import gs_job  # noqa: E402  (reads JOB_DIR/job.json; gives status(), run(), colmap_reconstruct())
from gs_job import O, JOB, status, STEPS, n_images

AUTO = os.path.join(REPO, "pipeline", "auto_locate.py")


def run_progress(cmd):
    """Run auto_locate and turn its 'PROGRESS nn text' lines into status updates."""
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                         env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
    for line in p.stdout:
        line = line.rstrip()
        m = re.match(r"PROGRESS (\d+) (.*)", line)
        if m:
            status(gs_job.STATUS_STEP[0], progress=int(m.group(1)), detail=m.group(2))
        print(line, flush=True)
    if p.wait() != 0:
        raise RuntimeError("auto_locate.py 失敗，見記錄")


def main():
    images = O["images"]
    STEPS[:] = (["COLMAP 特徵", "COLMAP 匹配", "COLMAP 重建"] if O.get("poses") == "colmap" else []) + ["找振膜與分架", "等待編號", "完成"]
    status(STEPS[0], state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
    n_in = sum(1 for f in os.listdir(images) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")))
    if O.get("poses") == "colmap":
        work = os.path.join(JOB, "colmap")
        done = [d for d in (os.path.join(work, "sparse", s) for s in os.listdir(os.path.join(work, "sparse"))) if os.path.exists(os.path.join(d, "images.bin"))] \
            if os.path.isdir(os.path.join(work, "sparse")) else []
        model = max(done, key=n_images) if done else gs_job.colmap_reconstruct(images, work)
    else:
        model = O["sparse"]
    status("找振膜與分架", n_images=n_images(model), n_input=n_in, model=model)
    out = os.path.join(JOB, "out")
    if not os.path.exists(os.path.join(out, "analysis.json")):
        run_progress([sys.executable, "-u", AUTO, "analyse", "--model", model, "--images", images, "--out", out, "--per-bar", str(O.get("per_bar", 8))])
    A = json.load(open(os.path.join(out, "analysis.json")))
    status("等待編號", state="needs_ids", progress=None, n_found=len(A["points"]), n_bars=len(A["bars"]),
           result={"outputs": out})
    print(f"找到 {len(A['points'])} 顆、{len(A['bars'])} 根架子。請在左邊「新空間定位」設定編號。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"錯誤：{e}", flush=True)
        status("錯誤", state="error", error=str(e))
        sys.exit(1)
