"""Check that this computer can run the pipeline.

    python tools/check_env.py                    # check packages, GPU, settings.yaml paths, COLMAP
    python tools/check_env.py --download-colmap  # also download COLMAP 3.11.1 (Windows, CUDA) into tools/colmap

Nothing is changed unless --download-colmap is given."""
import importlib, os, shutil, subprocess, sys, urllib.request, zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLMAP_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/colmap-x64-windows-cuda.zip"
OK, BAD, OPT = "  ok ", "  !! ", "  -- "
problems = 0


def say(tag, msg):
    global problems
    problems += tag == BAD
    print(tag + msg)


print("Python", sys.version.split()[0], sys.executable)
for mod, pip, why in [("numpy", "numpy", "all steps"), ("cv2", "opencv-python>=4.4", "SIFT, PnP, undistort"),
                      ("PIL", "pillow", "figures"), ("yaml", "pyyaml", "settings.yaml"),
                      ("torch", "torch (CUDA build)", "GPU feature matching in localize_sony.py")]:
    try:
        m = importlib.import_module(mod)
        say(OK, f"{mod} {getattr(m, '__version__', '')}")
    except ImportError:
        say(BAD, f"{mod} missing ({why}): pip install {pip}")
try:
    import cv2
    cv2.SIFT_create()
except Exception as e:
    say(BAD, f"OpenCV has no SIFT ({e}); install opencv-python 4.4 or newer")
try:
    import torch
    say(OK if torch.cuda.is_available() else OPT,
        f"CUDA GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "no CUDA GPU: matching runs on CPU (much slower)")
except Exception:
    pass

print("\nsettings.yaml")
path = os.path.join(REPO, "settings.yaml")
if not os.path.exists(path):
    say(BAD, "settings.yaml not found: copy settings.example.yaml to settings.yaml and edit the paths")
    S = {}
else:
    import yaml
    S = yaml.safe_load(open(path, encoding="utf-8")) or {}
need = {"sony_images": "Sony photos", "dji_model": "DJI COLMAP model", "dji_images": "DJI frames",
        "sony_model": "Sony COLMAP model", "sony_database": "Sony COLMAP database", "mask_dir": "SAM2 masks",
        "coarse_dir": "coarse boxes", "design_txt": "Panoramix design file"}
optional = {"sony_selected_list": "selected photo list", "sony_quality_csv": "photo quality csv",
            "colmap_exe": "COLMAP (only for ba_sony.py and new reconstructions)", "font": "font for figures"}
for group, tag in ((need, BAD), (optional, OPT)):
    for k, what in group.items():
        v = S.get(k)
        if v is None:
            say(tag, f"{k}: not set ({what})"); continue
        p = v if os.path.isabs(v) else os.path.join(REPO, v)
        if os.path.exists(p):
            n = len(os.listdir(p)) if os.path.isdir(p) else None
            say(OK, f"{k}: {p}" + (f"  ({n} items)" if n is not None else ""))
        else:
            say(tag, f"{k}: missing {p} ({what})")
for k in ("work_dir", "output_dir"):
    v = S.get(k)
    if v:
        p = v if os.path.isabs(v) else os.path.join(REPO, v)
        say(OK, f"{k}: {p}" + ("" if os.path.exists(p) else "  (will be created)"))

print("\nCOLMAP")
exe = S.get("colmap_exe")
exe = exe and (exe if os.path.isabs(exe) else os.path.join(REPO, exe))
if not (exe and os.path.exists(exe)):
    exe = shutil.which("colmap")
if exe:
    out = subprocess.run([exe, "-h"], capture_output=True, text=True).stdout.splitlines()
    say(OK, f"{exe}: {out[0] if out else 'found'}")
elif "--download-colmap" in sys.argv:
    dst = os.path.join(REPO, "tools", "colmap"); os.makedirs(dst, exist_ok=True)
    z = os.path.join(dst, "colmap.zip")
    print(f"  downloading {COLMAP_URL}")
    urllib.request.urlretrieve(COLMAP_URL, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(dst)
    os.remove(z)
    found = [os.path.join(r, "colmap.exe") for r, _, fs in os.walk(dst) if "colmap.exe" in fs]
    say(OK if found else BAD, f"COLMAP extracted: {found[0]}  -> set colmap_exe to this path" if found else "colmap.exe not found in the download")
else:
    say(OPT, "COLMAP not found. Only needed for ba_sony.py or a new reconstruction. "
             "Run with --download-colmap, or get it from https://github.com/colmap/colmap/releases")

print("\nOK: ready to run pipeline/run_all.py" if not problems else f"\n{problems} problem(s) above need fixing first")
sys.exit(1 if problems else 0)
