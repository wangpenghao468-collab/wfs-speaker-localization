"""How much do speaker position errors matter for WFS, and what does switching to
the measured layout gain?

The renderer computes each speaker's delay from the position it is TOLD; the sound
leaves from where the speaker REALLY is. For a virtual source the per-speaker
timing error is the change in source-to-speaker distance caused by the position
error. A common delay shared by all active speakers is harmless, so it is removed.
Coherence at frequency f: |mean exp(j 2 pi f dt)| over active speakers (1 = all
contributions in phase; the loss in dB shows how much the synthesized wavefront
is degraded).

Cases
  manual   : renderer uses the Panoramix design, speakers are at the measured spots
  measured : renderer uses the photo-measured layout; remaining error = photo
             measurement error (bootstrap sd per speaker, simulated as Gaussian)
"""
import json, numpy as np
from config import WORK

C = 343.0
FREQS = (500, 1000, 2000)


def active(P, nrm, src=None, k=None):
    """WFS secondary-source selection: speakers facing the propagation direction."""
    v = k[None, :] if k is not None else (P - src)
    return (v * nrm).sum(1) > 0


def timing(P_told, P_true, src=None, k=None):
    if k is not None:
        return ((P_told - P_true) @ k) / C                # plane wave: projection on the direction
    return (np.linalg.norm(P_told - src, axis=1) - np.linalg.norm(P_true - src, axis=1)) / C


def coherence_db(dt, f):
    return 20 * np.log10(np.abs(np.exp(2j * np.pi * f * dt).mean()))


def main():
    R = json.load(open(f"{WORK}/design_vs_measured.json"))["rows"]
    Cen = {r["id"]: r for r in json.load(open(f"{WORK}/centers.json"))["rows"]}
    D = np.array([r["design_xy"] for r in R]) / 1000; M = np.array([r["measured_xy"] for r in R]) / 1000
    sd = np.array([Cen[r["id"]]["sd_boot"] for r in R]) / 1000
    # inward normals per bar from the measured layout
    nrm = np.zeros_like(M)
    for b in range(14):
        P = M[b * 8:(b + 1) * 8]; u = np.linalg.svd(P - P.mean(0))[2][0]; n = np.array([-u[1], u[0]])
        if n @ (-P.mean(0)) < 0:
            n = -n
        nrm[b * 8:(b + 1) * 8] = n
    rng = np.random.default_rng(0)
    scenes = [("plane", np.array([np.sin(a), np.cos(a)])) for a in np.radians(np.arange(0, 360, 5))]
    for ang in range(0, 360, 30):                         # point sources 1 m outside the array
        a = np.radians(ang); scenes.append(("point", np.array([np.sin(a), np.cos(a)]) * 3.2))
    res = {"manual": [], "measured": []}
    for kind, v in scenes:
        k, src = (v, None) if kind == "plane" else (None, v)
        act = active(M, nrm, src=src, k=-v if kind == "plane" else None) if kind == "plane" else active(M, nrm, src=src)
        if act.sum() < 8:
            continue
        dt = timing(D, M, src, k)[act]; dt -= dt.mean()
        res["manual"].append(dt)
        for _ in range(20):                               # measured layout: only the photo error is left
            Pm = M + rng.normal(size=M.shape) * sd[:, None] / np.sqrt(3) * np.sqrt(2)   # 3D sd -> per horizontal axis
            dt = timing(Pm, M, src, k)[act]; dt -= dt.mean()
            res["measured"].append(dt)
    out = {}
    for case, L in res.items():
        rms = np.array([np.sqrt((x ** 2).mean()) for x in L]) * 1e6
        mx = np.array([np.abs(x).max() for x in L]) * 1e6
        coh = {f: np.array([coherence_db(x, f) for x in L]) for f in FREQS}
        out[case] = dict(rms_us_median=float(np.median(rms)), rms_us_worst=float(rms.max()),
                         max_us_worst=float(mx.max()),
                         coherence_db={f: dict(median=float(np.median(c)), worst=float(c.min())) for f, c in coh.items()})
    print(json.dumps(out, indent=1))
    # equivalent path error and phase at 1 kHz / 2 kHz for the worst speaker in the manual layout
    dev = np.linalg.norm(D - M, axis=1)
    for f in FREQS:
        print(f"{f} Hz: wavelength {C / f * 100:.1f} cm; worst manual error {dev.max() * 1000:.0f} mm = "
              f"{dev.max() / (C / f) * 360:.0f} deg phase; median {np.median(dev) * 1000:.0f} mm = {np.median(dev) / (C / f) * 360:.0f} deg; "
              f"photo error ~{np.median(sd) * 1000:.1f} mm = {np.median(sd) / (C / f) * 360:.1f} deg")
    json.dump(out, open(f"{WORK}/wfs_error_sim.json", "w"), indent=1)


if __name__ == "__main__":
    main()
