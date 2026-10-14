"""Rebuild everything end to end. Re-run after adding or fixing DJI annotations
(masks/ or coarse_annotations/) or after downloading speaker_corrections.json from
the web page. Cached features are reused, so only new work is recomputed."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    (["sony_cameras.py"], "intrinsics for all selected Sony photos"),
    (["detect_diaphragms.py"], "detect diaphragms in every Sony photo"),
    (["dji_speakers.py"], "triangulate the annotated speakers in the DJI model (identity source)"),
    (["dji_bars.py"], "8-per-bar model -> seeds for all 112 speakers"),
    (["localize_sony.py"], "localise each Sony photo in the DJI frame (PnP)"),
    (["consensus_poses.py"], "neighbour voting + user corrections: reject mirrored-side poses"),
    (["identify.py"], "speaker -> photos"),
    (["triangulate_centers.py"], "centres from the Sony photos (Huber LS + bootstrap)"),
    (["floor_height.py"], "floor densified from photo pairs, heights"),
    (["triangulate_centers.py"], "re-express centres on the densified floor"),
    (["floor_height.py"], "heights on the final frame"),
    (["compare_design.py"], "deviation from the Panoramix design"),
    (["export_panoramix.py"], "Panoramix aed + directions lines, 1012_measured.txt"),
    (["plot_design_vs_measured.py"], "figure: design vs measured"),
    (["plot_heights.py"], "figure: heights"),
    (["build_finder.py"], "web page data, previews, CSVs"),
    (["verify_bars.py"], "per-bar verification montages"),
    (["verify_bars.py", "--unregistered"], "montages from photos COLMAP had not registered"),
]

for args, what in STEPS:
    t = time.time(); print(f"== {' '.join(args)}: {what}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, args[0])] + args[1:], check=True, cwd=HERE)
    print(f"   done in {time.time() - t:.0f}s", flush=True)
