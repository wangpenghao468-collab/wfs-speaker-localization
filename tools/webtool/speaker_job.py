"""One speaker-localisation job (started by the web tool).

    python tools/webtool/speaker_job.py JOB_DIR      # JOB_DIR/job.json holds the options

mode "frame":     re-express the current results in a chosen origin / axis (seconds).
mode "recompute": run the whole pipeline on a photo folder (the original Sony photos plus any new ones),
                  in a separate work folder so the adopted results are never overwritten; then the frame step."""
import glob, json, os, shutil, subprocess, sys, time
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "pipeline"))
import config  # noqa: E402

JOB = os.path.abspath(sys.argv[1])
O = json.load(open(os.path.join(JOB, "job.json"), encoding="utf-8"))
STEPS = []
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def status(step, **kw):
    p = os.path.join(JOB, "status.json")
    st = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    st.update(step=step, steps=STEPS, time=time.strftime("%Y-%m-%d %H:%M:%S"), **kw)
    json.dump(st, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def run(cmd, env=None):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen([str(c) for c in cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=os.path.join(REPO, "pipeline"))
    for line in p.stdout:
        print(line.rstrip(), flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"{os.path.basename(str(cmd[1]))} failed (exit {p.returncode})")


def prepare_recompute():
    """Job settings: own work/output folders, caches shared, photo list extended with new photos."""
    S = dict(config._S)
    work = os.path.join(JOB, "work"); out = os.path.join(JOB, "outputs"); os.makedirs(work, exist_ok=True)
    src = config.WORK
    for f in glob.glob(os.path.join(src, "*.json")):                     # small intermediate files
        shutil.copy2(f, work)
    if os.path.isdir(os.path.join(src, "corr")):
        shutil.copytree(os.path.join(src, "corr"), os.path.join(work, "corr"), dirs_exist_ok=True)
    feats = os.path.join(src, "feats2")                                  # ~1 GB feature cache: shared, not copied
    if os.path.isdir(feats) and not os.path.exists(os.path.join(work, "feats2")):
        subprocess.run(["cmd", "/c", "mklink", "/J", os.path.join(work, "feats2"), feats], check=True, capture_output=True)
    photos = O.get("images") or config.SONY_IMAGES
    have = sorted(f for f in os.listdir(photos) if f.lower().endswith(IMG_EXT))
    old = [l.strip() for l in open(config.SONY_SELECTED) if l.strip()]
    missing = [n for n in old if n not in set(have)]
    if missing:
        raise RuntimeError(f"這個資料夾少了 {len(missing)} 張原本的 Sony 照片（例如 {missing[0]}）。"
                           "請把新照片放進含有原始照片的資料夾，因為鏡頭參數和喇叭編號都依賴原本的照片。")
    excluded = set()
    ex = os.path.join(os.path.dirname(config.SONY_SELECTED), "excluded_images.csv")
    if os.path.exists(ex):
        excluded = {l.split(",")[0].strip() for l in open(ex, encoding="utf-8-sig")}
    new = [n for n in have if n not in set(old) and n not in excluded]
    sel = os.path.join(JOB, "selected_image_list.txt")
    open(sel, "w", encoding="utf-8").write("\n".join(old + new) + "\n")
    print(f"photos: {len(old)} original + {len(new)} new", flush=True)
    S.update(sony_images=photos, sony_selected_list=sel, work_dir=work, output_dir=out,
             measurement_dir=os.path.join(JOB, "measurement"))
    if O.get("design"):
        S["design_txt"] = O["design"]
    path = os.path.join(JOB, "settings.yaml")
    yaml.safe_dump(S, open(path, "w", encoding="utf-8"), allow_unicode=True)
    return path, len(new)


def main():
    mode = O.get("mode", "frame")
    STEPS[:] = (["準備", "主流程（約 30–90 分鐘）", "匯出 PLY"] if mode == "recompute" else []) + ["設定原點", "完成"]
    status(STEPS[0], state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    py = sys.executable
    if mode == "recompute":
        settings, n_new = prepare_recompute()
        env["WFS_SETTINGS"] = settings
        status(STEPS[1], new_photos=n_new)
        run([py, os.path.join(REPO, "pipeline", "run_all.py")], env)
        status(STEPS[2])
        run([py, os.path.join(REPO, "tools", "export_ply.py"), "--no-mesh"], env)
    status("設定原點")
    out = os.path.join(JOB, "frame")
    cmd = [py, os.path.join(REPO, "pipeline", "custom_frame.py"), "--out", out]
    if O.get("origin_speakers"):
        cmd += ["--origin-speakers", O["origin_speakers"]]
    if O.get("x_from") and O.get("x_to"):
        cmd += ["--x-from", O["x_from"], "--x-to", O["x_to"]]
    if O.get("offset"):
        cmd += ["--offset", *O["offset"]]
    if O.get("design"):
        cmd += ["--design", O["design"]]
    run(cmd, env)
    result = {"folder": out, "files": sorted(os.path.join(out, f) for f in os.listdir(out))}
    if mode == "recompute":
        result["outputs"] = os.path.join(JOB, "outputs")
    status("完成", state="done", result=result, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("完成", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"錯誤：{e}", flush=True)
        status("錯誤", state="error", error=str(e))
        sys.exit(1)
