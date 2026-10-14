"""All paths come from settings.yaml in the repository root (copy settings.example.yaml
and edit it). Relative paths in that file are relative to the repository root."""
import os, sys
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
_path = f"{REPO}/settings.yaml"
if not os.path.exists(_path):
    sys.exit(f"settings.yaml not found in {REPO}. Copy settings.example.yaml to settings.yaml and set your paths "
             f"(python tools/check_env.py checks them).")
_S = yaml.safe_load(open(_path, encoding="utf-8")) or {}


def _p(key, default=None):
    v = _S.get(key, default)
    if v is None:
        return None
    v = os.path.expanduser(str(v)).replace("\\", "/")
    return v if os.path.isabs(v) else f"{REPO}/{v}"


WORK = _p("work_dir")                  # caches and intermediate json (about 1.5 GB)
OUT_DIR = _p("output_dir")             # full local outputs: finder with previews, csv, figures
DJI_MODEL = _p("dji_model")            # COLMAP sparse model of the DJI frames (coordinate frame)
DJI_IMAGES = _p("dji_images")
SONY_MODEL = _p("sony_model")          # Sony COLMAP model (local relations only)
SONY_DB = _p("sony_database")          # Sony COLMAP database.db (verified matches)
SONY_SELECTED = _p("sony_selected_list")
SONY_QUALITY = _p("sony_quality_csv")
SONY_IMAGES = _p("sony_images")
MASK_DIR = _p("mask_dir")              # SAM2 diaphragm masks drawn on DJI frames
COARSE_DIR = _p("coarse_dir")          # coarse boxes drawn on DJI frames
DESIGN_TXT = _p("design_txt")          # Panoramix design file (never modified)
COLMAP_EXE = _p("colmap_exe")
FONT = _p("font", "C:/Windows/Fonts/msjh.ttc")
SITE_DIR = f"{REPO}/docs"              # GitHub Pages site (built by tools/build_site.py)
MEASURE_DIR = _p("measurement_dir", "measurements/latest")
