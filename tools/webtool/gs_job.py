"""One 3D Gaussian Splatting job: (COLMAP) -> undistort + resize -> train (3DGS or PGSR) -> render / mesh.

    python tools/webtool/gs_job.py JOB_DIR        # JOB_DIR/job.json holds the options (written by the web tool)

Progress goes to JOB_DIR/status.json, the console output to JOB_DIR/log.txt (the web tool shows both).
Ports: the trainers open a viewer socket; each job gets its own free port so parallel jobs never clash."""
import glob, json, os, re, shutil, socket, struct, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline"))
import config  # noqa: E402

JOB = os.path.abspath(sys.argv[1])
O = json.load(open(os.path.join(JOB, "job.json"), encoding="utf-8"))
S = config._S
COLMAP = config.COLMAP_EXE
STEPS = []
STATUS_STEP = [""]                      # current step name (for progress updates)


def setting(key):
    v = S.get(key)
    return None if v is None else (v if os.path.isabs(v) else os.path.join(config.REPO, v))


def status(step, **kw):
    STATUS_STEP[0] = step
    st = {"step": step, "steps": STEPS, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    old = {}
    if os.path.exists(os.path.join(JOB, "status.json")):
        old = json.load(open(os.path.join(JOB, "status.json"), encoding="utf-8"))
    old.update(st)
    json.dump(old, open(os.path.join(JOB, "status.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def run(cmd, cwd=None, env=None):
    print("$ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), flush=True)
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    shown, traceback, error = -1, False, None
    for line in p.stdout:
        line = line.rstrip()
        m = re.search(r"(\d+)%\|", line)
        if m:                                          # progress bars: report the percentage, log only the final line
            pct = int(m.group(1))
            if "Training progress" in line and pct != shown:
                shown = pct; status(STATUS_STEP[0], progress=pct)
            if pct < 100:
                continue
        if line:
            print(line, flush=True)
        traceback = traceback or line.startswith("Traceback")
        if traceback and re.match(r"^\w+(Error|Exception)\b", line):
            error = line                               # a crashed Python step can hang in its native threads:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)   # stop it now
            break
    code = p.wait()
    if error or code != 0:
        raise RuntimeError(f"{os.path.basename(str(cmd[1] if len(cmd) > 1 else cmd[0]))} failed: {error or f'exit {code}'}")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close(); return port


def n_images(model):
    return struct.unpack("<Q", open(os.path.join(model, "images.bin"), "rb").read(8))[0]


def colmap_reconstruct(images, work):
    db = os.path.join(work, "database.db"); sparse = os.path.join(work, "sparse"); os.makedirs(sparse, exist_ok=True)
    gpu = ["--SiftExtraction.use_gpu", "1"]
    status("COLMAP 特徵")
    run([COLMAP, "feature_extractor", "--database_path", db, "--image_path", images,
         "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1" if O.get("single_camera") else "0",
         "--SiftExtraction.max_image_size", "3200", "--SiftExtraction.max_num_features", "8192"] + gpu)
    status("COLMAP 匹配")
    m = O.get("matcher", "exhaustive")
    if m == "sequential":
        run([COLMAP, "sequential_matcher", "--database_path", db, "--SequentialMatching.overlap", "20", "--SiftMatching.use_gpu", "1"])
    elif m == "vocab":
        run([COLMAP, "vocab_tree_matcher", "--database_path", db, "--VocabTreeMatching.vocab_tree_path", setting("vocab_tree"), "--SiftMatching.use_gpu", "1"])
    else:
        run([COLMAP, "exhaustive_matcher", "--database_path", db, "--SiftMatching.use_gpu", "1"])
    status("COLMAP 重建")
    run([COLMAP, "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse])
    models = [d for d in glob.glob(os.path.join(sparse, "*")) if os.path.exists(os.path.join(d, "images.bin"))]
    if not models:
        raise RuntimeError("COLMAP 沒有重建出任何模型（照片之間重疊不夠？）")
    best = max(models, key=n_images)
    print(f"COLMAP: {len(models)} model(s); using {best} with {n_images(best)} images", flush=True)
    if len(models) > 1:
        print("注意：照片分成多個互不相連的模型，只用最大的一個。", flush=True)
    return best


def undistort(images, model, data):
    run([COLMAP, "image_undistorter", "--image_path", images, "--input_path", model, "--output_path", data,
         "--output_type", "COLMAP", "--max_image_size", str(O.get("max_size", 1600))])
    sp = os.path.join(data, "sparse"); os.makedirs(os.path.join(sp, "0"), exist_ok=True)
    unify_sizes(data)
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(os.path.join(sp, f), os.path.join(sp, "0", f))      # PGSR reads sparse/, 3DGS reads sparse/0/


def unify_sizes(data):
    """Centre-crop every undistorted image to the smallest common size and shift its principal point.
    Photos with different aspect ratios (16:9 and 3:2) otherwise break PGSR's mesh fusion."""
    import cv2
    cam_file = os.path.join(data, "sparse", "cameras.bin")
    cams = []
    with open(cam_file, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            if mid != 1:
                raise RuntimeError(f"unexpected camera model {mid} after undistortion")
            cams.append([cid, mid, w, h, list(struct.unpack("<4d", f.read(32)))])
    W, H = min(c[2] for c in cams), min(c[3] for c in cams)
    if all((c[2], c[3]) == (W, H) for c in cams):
        return
    sys.path.insert(0, os.path.join(config.REPO, "pipeline"))
    from colmap_io import read_images
    by_cam = {}
    for im in read_images(os.path.join(data, "sparse", "images.bin"), with_points=False).values():
        by_cam.setdefault(im["camera_id"], []).append(im["name"])
    for c in cams:
        dx, dy = (c[2] - W) // 2, (c[3] - H) // 2
        if dx or dy:
            for name in by_cam.get(c[0], []):
                p = os.path.join(data, "images", name)
                img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                cv2.imwrite(p, img[dy:dy + H, dx:dx + W], [cv2.IMWRITE_JPEG_QUALITY, 95])
            c[4][2] -= dx; c[4][3] -= dy
        c[2], c[3] = W, H
    with open(cam_file, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, mid, w, h, prm in cams:
            f.write(struct.pack("<iiQQ4d", cid, mid, w, h, *prm))
    print(f"images centre-cropped to a common {W}x{H} (mixed aspect ratios)", flush=True)


def scene_depth(model):
    """Robust scene depth in model units (for the mesh depth cut-off and voxel size)."""
    sys.path.insert(0, os.path.join(config.REPO, "pipeline"))
    from colmap_io import read_model
    _, imgs = read_model(model, with_points=False)
    ids, xyz = __import__("colmap_io").read_points3D(os.path.join(model, "points3D.bin"))
    C = np.array([im["C"] for im in imgs.values()])
    sub = xyz[:: max(1, len(xyz) // 5000)]
    d = np.linalg.norm(sub[None] - C[:: max(1, len(C) // 200), None], axis=2)
    return float(np.percentile(d, 90))


def main():
    os.makedirs(JOB, exist_ok=True)
    images = O["images"]; method = O.get("method", "pgsr")
    STEPS[:] = (["COLMAP 特徵", "COLMAP 匹配", "COLMAP 重建"] if O["poses"] == "colmap" else []) + \
               ["去畸變與縮圖", "訓練", "渲染與網格" if method == "pgsr" else "渲染預覽", "完成"]
    status(STEPS[0], state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))

    if O["poses"] == "colmap":
        model = colmap_reconstruct(images, os.path.join(JOB, "colmap"))
    elif O["poses"] == "sony_posed":
        model = setting("sony_posed_model")
    else:
        model = O["sparse"]
    if not os.path.exists(os.path.join(model, "images.bin")):
        raise RuntimeError(f"找不到 COLMAP 模型：{model}")

    status("去畸變與縮圖")
    data = os.path.join(JOB, "dataset")
    done_marker = os.path.join(data, "sparse", "0", "points3D.bin")
    if os.path.exists(done_marker):
        print("dataset already prepared (retry): skipping undistortion", flush=True)
    else:
        if os.path.exists(data):
            shutil.rmtree(data)
        undistort(images, model, data)
    sp = os.path.join(data, "sparse")
    n_img = n_images(sp)
    n_in =sum(1 for f in os.listdir(images) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")))
    print(f"dataset: {n_img} of {n_in} photos have camera poses; longest side {O.get('max_size', 1600)} px", flush=True)
    if n_img < 0.7 * n_in:
        print(f"注意：只有 {n_img}/{n_in} 張照片定位成功，其餘沒有參與重建（照片之間重疊不夠，或改用其他匹配方式）。", flush=True)

    status("訓練", n_images=n_img, n_input=n_in)
    it = int(O.get("iterations", 30000)); out = os.path.join(JOB, "model")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(O.get("gpu", 0)), PYTHONIOENCODING="utf-8")
    repo, py = (setting("gs_pgsr_repo"), setting("gs_pgsr_python")) if method == "pgsr" else (setting("gs_3dgs_repo"), setting("gs_3dgs_python"))
    cmd = [py, "train.py", "-s", data, "-m", out, "-r", "1", "--iterations", it, "--port", free_port(),
           "--test_iterations", it, "--save_iterations", *sorted({min(7000, it), it})]
    if O.get("low_mem"):
        cmd += ["--data_device", "cpu"]
    if method == "3dgs":
        cmd += ["--disable_viewer"]
        if O.get("exposure"):
            cmd += ["--train_test_exp"]              # per-image exposure (3DGS >= 2024)
    else:
        if O.get("low_mem"):
            cmd += ["--max_all_points", "2000000"]
        if O.get("exposure"):
            cmd += ["--exposure_compensation"]       # per-image exposure (photos taken with auto exposure)
    ply = os.path.join(out, "point_cloud", f"iteration_{it}", "point_cloud.ply")
    if os.path.exists(ply):
        print("model already trained (retry): skipping training", flush=True)
    else:
        run(cmd, cwd=repo, env=env)

    status(STEPS[-2])
    result = {"gaussians_ply": ply}
    if method == "pgsr" and O.get("mesh", True):
        depth = scene_depth(model)
        div = {"coarse": 300, "medium": 600, "fine": 1000}.get(O.get("mesh_detail", "medium"), 600)
        print(f"mesh: depth cut-off {depth:.2f}, voxel {depth / div:.4f} (scene units; detail '{O.get('mesh_detail', 'medium')}')", flush=True)
        run([py, "render.py", "-m", out, "--max_depth", f"{depth:.3f}", "--voxel_size", f"{depth / div:.5f}"], cwd=repo, env=env)
        result["mesh"] = os.path.join(out, "mesh", "tsdf_fusion_post.ply")
    elif O.get("render", True):
        run([py, "render.py", "-m", out, "--skip_test"], cwd=repo, env=env)
    rd = sorted(glob.glob(os.path.join(out, "train", f"ours_{it}", "renders", "*")))
    gt = sorted(glob.glob(os.path.join(out, "train", f"ours_{it}", "gt", "*")))
    result["previews"] = [r for r in rd[:: max(1, len(rd) // 6)][:6]]
    result["gt"] = [g for g in gt[:: max(1, len(gt) // 6)][:6]] if gt else []
    log = open(os.path.join(JOB, "log.txt"), encoding="utf-8", errors="replace").read()
    ps = re.findall(r"Evaluating train: L1 [\d.e-]+ PSNR ([\d.]+)", log)
    if ps:
        result["psnr_train"] = round(float(ps[-1]), 2)
    if os.path.exists(ply):
        with open(ply, "rb") as f:
            head = f.read(2000).decode("latin-1")
        m = re.search(r"element vertex (\d+)", head)
        result["gaussians"] = int(m.group(1)) if m else None
    status("完成", state="done", result=result, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("完成：" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"錯誤：{e}", flush=True)
        status("錯誤", state="error", error=str(e))
        sys.exit(1)
