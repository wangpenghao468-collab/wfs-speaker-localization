"""Locate a speaker array in a NEW room from a COLMAP model of ordinary photos (phone or camera),
without DJI annotations. Speaker numbers are given afterwards on a top view (web tool).

    python pipeline/auto_locate.py analyse --model SPARSE_DIR --images IMG_DIR --out DIR [--per-bar 8]
    python pipeline/auto_locate.py assign  --out DIR --spec spec.json

analyse
  1. detect diaphragms in every registered photo (same classical detector as the main pipeline)
  2. pair candidates: detections on the epipolar line in covisible photos, same physical size
  3. 3D clusters of candidates -> verified by reprojection into ALL photos -> robust Huber triangulation
  4. scale from the in-bar pitch (median nearest-neighbour distance = pitch_mm), bars = collinear chains
     with ~1 pitch steps (the gap between bars is longer), vertical = normal of all bar directions,
     floor = densest horizontal level of the COLMAP points below the speakers
  writes analysis.json (+ detections.json cache)
assign
  spec.json: {"start_bar": 0, "first_id": 1, "clockwise": true, "overrides": {"3": {"first": 41, "reverse": false}},
              "origin_ids": [..], "x_from": a, "x_to": b, "offset": [x, y], "pitch_mm": 80,
              "tape": {"a": 41, "b": 48, "mm": 557.5} | null, "known_height": {"id": 41, "mm": 1120} | null,
              "design": "path.txt" | "", "bus": 10}
  writes speakers.csv, panoramix_bus10.txt, <design>_custom.txt, speakers_points.ply, report.json, check_*.jpg"""
import argparse, csv, json, os, sys, time
from collections import defaultdict
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colmap_io import read_model, read_points3D          # noqa: E402
from detect_diaphragms import detect, DET_W              # noqa: E402
from triangulate_centers import solve                    # noqa: E402

TOL = 0.0012          # match tolerance, fraction of image width (~5 px on a 4000 px photo)
NEIGH = 12            # covisible partner photos per photo
MIN_VIEWS = 5
MIN_SPREAD_DEG = 15
IN_BAR = 1.06         # in-bar step < 1.06 pitch; bar-to-bar gap is longer (~1.12 pitch here)


def log(*a):
    print(*a, flush=True)


def intrinsics(cam):
    m, p = cam["model"], cam["params"]
    d = np.zeros(4)
    if m == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1:3]
    elif m == "PINHOLE":
        fx, fy, cx, cy = p
    elif m == "SIMPLE_RADIAL":
        fx = fy = p[0]; cx, cy = p[1:3]; d[0] = p[3]
    elif m == "RADIAL":
        fx = fy = p[0]; cx, cy = p[1:3]; d[:2] = p[3:5]
    elif m == "OPENCV":
        fx, fy, cx, cy = p[:4]; d[:] = p[4:8]
    else:
        raise RuntimeError(f"camera model {m} not supported")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), d


# ---------------------------------------------------------------- 1 detection
def detect_all(model, images, out):
    cams, imgs = read_model(model, with_points=False)
    path = os.path.join(out, "detections.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    todo = [im for im in imgs.values() if im["name"] not in res]
    t0 = time.time()
    for k, im in enumerate(todo):
        n = im["name"]
        img = cv2.imread(os.path.join(images, n), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)   # COLMAP ignores EXIF rotation too
        if img is None:
            res[n] = []; continue
        cam = cams[im["camera_id"]]; K, dist = intrinsics(cam)
        to_cam = cam["width"] / img.shape[1]
        sc = min(1.0, DET_W / img.shape[1])
        small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else img
        D = detect(small)
        if D:
            xy = np.array([d["xy"] for d in D]) / sc * to_cam
            und = cv2.undistortPoints(xy.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
            res[n] = [dict(xy=[round(v, 2) for v in xy[i]], u=[round(v, 2) for v in und[i]], r=round(D[i]["r"] / sc * to_cam, 2))
                      for i in range(len(D))]
        else:
            res[n] = []
        if k % 25 == 0 or k == len(todo) - 1:
            log(f"PROGRESS {int(100 * (k + 1) / len(todo))} 偵測振膜 {k + 1}/{len(todo)}  {n}: {len(res[n])}  ({time.time() - t0:.0f}s)")
            json.dump(res, open(path, "w"))
    json.dump(res, open(path, "w"))
    return res


# ---------------------------------------------------------------- 2-3 triangulation
def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def load_views(model, det):
    cams, imgs = read_model(model)
    V = []
    for iid, im in sorted(imgs.items()):
        D = det.get(im["name"]) or []
        cam = cams[im["camera_id"]]; K, _ = intrinsics(cam)
        V.append(dict(name=im["name"], K=K, Ki=np.linalg.inv(K), R=im["R"], t=im["t"], C=im["C"], P=K @ np.c_[im["R"], im["t"]],
                      w=cam["width"], h=cam["height"], f=K[0, 0],
                      u=np.array([d["u"] for d in D]).reshape(-1, 2), xy=np.array([d["xy"] for d in D]).reshape(-1, 2),
                      r=np.array([d["r"] for d in D]), pids=np.unique(im["pids"][im["pids"] >= 0])))
    return V


def covisible_pairs(V):
    owners = defaultdict(list)
    for k, v in enumerate(V):
        if len(v["u"]):
            for p in v["pids"]:
                owners[int(p)].append(k)
    cnt = defaultdict(int)
    for lst in owners.values():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                cnt[(lst[a], lst[b])] += 1
    best = defaultdict(list)
    for (a, b), c in cnt.items():
        if c >= 20:
            best[a].append((c, b)); best[b].append((c, a))
    pairs = set()
    for a, l in best.items():
        for c, b in sorted(l, reverse=True)[:NEIGH]:
            pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def pair_candidates(A, B):
    if not len(A["u"]) or not len(B["u"]):
        return None
    Rr = B["R"] @ A["R"].T; tr = B["t"] - Rr @ A["t"]
    F = B["Ki"].T @ skew(tr) @ Rr @ A["Ki"]
    xa = np.c_[A["u"], np.ones(len(A["u"]))]; xb = np.c_[B["u"], np.ones(len(B["u"]))]
    L = xa @ F.T
    dist = np.abs(L @ xb.T) / np.linalg.norm(L[:, :2], axis=1)[:, None]
    ia, ib = np.nonzero(dist < TOL * max(A["w"], B["w"]))
    if not len(ia):
        return None
    da = (A["R"].T @ (A["Ki"] @ xa[ia].T)).T; da /= np.linalg.norm(da, axis=1)[:, None]
    db = (B["R"].T @ (B["Ki"] @ xb[ib].T)).T; db /= np.linalg.norm(db, axis=1)[:, None]
    w0 = A["C"] - B["C"]; bb = (da * db).sum(1); dd = da @ w0; ee = db @ w0
    den = np.maximum(1 - bb ** 2, 1e-12)
    s = (bb * ee - dd) / den; t = (ee - bb * dd) / den
    X = (A["C"] + s[:, None] * da + B["C"] + t[:, None] * db) / 2
    rho_a = A["r"][ia] * s / A["f"]; rho_b = B["r"][ib] * t / B["f"]
    ok = (s > 0) & (t > 0) & (bb < np.cos(np.radians(3))) & (rho_a / np.maximum(rho_b, 1e-12) > 0.75) & (rho_a / np.maximum(rho_b, 1e-12) < 1.33)
    return X[ok], ((rho_a + rho_b) / 2)[ok]


def support(V, X, rho):
    """All photos in which X reprojects onto a detection of the same physical size."""
    obs, Ps, refs = [], [], []
    for k, v in enumerate(V):
        if not len(v["u"]):
            continue
        Xc = v["R"] @ X + v["t"]
        if Xc[2] <= 0:
            continue
        p = v["K"] @ Xc; p = p[:2] / p[2]
        if not (0 <= p[0] < v["w"] and 0 <= p[1] < v["h"]):
            continue
        d = np.linalg.norm(v["u"] - p, axis=1); j = int(np.argmin(d))
        if d[j] < TOL * v["w"] and 0.7 < v["r"][j] * Xc[2] / v["f"] / rho < 1.4:
            obs.append(v["u"][j]); Ps.append(v["P"]); refs.append((k, j))
    return obs, Ps, refs


def spread_deg(V, X, refs):
    d = np.array([V[k]["C"] - X for k, _ in refs]); d /= np.linalg.norm(d, axis=1)[:, None]
    m = d.mean(0); m /= np.linalg.norm(m)
    return float(np.degrees(np.arccos(np.clip(d @ m, -1, 1))).max() * 2)


def triangulate(V):
    pairs = covisible_pairs(V)
    log(f"PROGRESS 0 三角化：{len(pairs)} 組共視照片")
    Xs, rs = [], []
    for k, (a, b) in enumerate(pairs):
        c = pair_candidates(V[a], V[b])
        if c is not None and len(c[0]):
            Xs.append(c[0]); rs.append(c[1])
        if k % 500 == 0:
            log(f"PROGRESS {int(40 * k / max(len(pairs), 1))} 配對候選 {k}/{len(pairs)}")
    if not Xs:
        raise RuntimeError("找不到任何跨照片一致的振膜（照片是否對得上？振膜偵測是否成功？）")
    X = np.concatenate(Xs); rho = np.concatenate(rs); rho0 = float(np.median(rho))
    log(f"{len(X)} 個兩兩候選，振膜半徑中位數 {rho0:.5f}（模型單位）")
    keys, inv, cnt = np.unique(np.floor(X / rho0).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    alive = np.ones(len(X), bool); found = []
    order = np.argsort(-cnt); t0 = time.time()
    for n_done, vi in enumerate(order):
        if cnt[vi] < 3:
            break
        m = alive & (inv == vi)
        if m.sum() < 3:
            continue
        seed = np.median(X[m], 0)
        near = alive & (np.linalg.norm(X - seed, axis=1) < rho0)
        if near.sum() < 3:
            alive[m] = False; continue
        seed = np.median(X[near], 0)
        res = None
        obs, Ps, refs = support(V, seed, rho0)
        if len(obs) >= MIN_VIEWS:
            Xr = solve(obs, Ps)[0]
            obs, Ps, refs = support(V, Xr, rho0)
            if len(obs) >= MIN_VIEWS and spread_deg(V, Xr, refs) >= MIN_SPREAD_DEG:
                Xr, e, cov = solve(obs, Ps)
                if np.median(e) < 0.5 * TOL * np.median([V[k]["w"] for k, _ in refs]):
                    res = (Xr, refs, e, obs, Ps)
        if res:
            found.append(res); alive &= np.linalg.norm(X - res[0], axis=1) > 2.5 * rho0
        else:
            alive &= np.linalg.norm(X - seed, axis=1) > rho0
        if n_done % 200 == 0:
            log(f"PROGRESS {40 + min(50, int(50 * (1 - alive.mean())))} 驗證候選：已找到 {len(found)} 顆（{time.time() - t0:.0f}s）")
    # non-maximum suppression: two speakers are never closer than ~5 diaphragm radii (80 mm vs 15 mm)
    found.sort(key=lambda r: -len(r[1])); keep = []
    for r in found:
        if all(np.linalg.norm(r[0] - k[0]) > 3 * rho0 for k in keep):
            keep.append(r)
    log(f"找到 {len(keep)} 顆喇叭（候選 {len(found)}）")
    return keep, rho0


# ---------------------------------------------------------------- 4 bars, vertical, floor
def chains(P, pitch):
    n = len(P); D = np.linalg.norm(P[:, None] - P[None], axis=2)
    adj = defaultdict(list)
    for i in range(n):
        nb = [j for j in np.argsort(D[i])[1:4] if 0.9 * pitch < D[i, j] < IN_BAR * pitch]
        adj[i] = nb[:2]
    ed = {(min(i, j), max(i, j)) for i in adj for j in adj[i] if i in adj[j]}     # mutual neighbours only
    g = defaultdict(list)
    for i, j in ed:
        g[i].append(j); g[j].append(i)
    seen, out = set(), []
    for s in range(n):
        if s in seen:
            continue
        comp, st = [], [s]
        while st:
            a = st.pop()
            if a in seen:
                continue
            seen.add(a); comp.append(a); st += g[a]
        c = P[comp].mean(0); u = np.linalg.svd(P[comp] - c)[2][0] if len(comp) > 1 else np.array([1.0, 0, 0])
        comp.sort(key=lambda i: (P[i] - c) @ u)
        out.append(comp)
    # merge collinear pieces separated by missing speakers (gap 2-3 pitches)
    merged = True
    while merged:
        merged = False
        for a in range(len(out)):
            for b in range(len(out)):
                if a == b or len(out[a]) < 2:
                    continue
                A, B = out[a], out[b]
                u = P[A[-1]] - P[A[0]]; u /= np.linalg.norm(u)
                for bb in (B, B[::-1]):
                    g_ = P[bb[0]] - P[A[-1]]; L = np.linalg.norm(g_); m = round(L / pitch)
                    if m in (2, 3) and abs(L - m * pitch) < 0.08 * pitch and g_ @ u / L > 0.995:
                        out[a] = A + [None] * (m - 1) + list(bb); out.pop(b); merged = True; break
                if merged:
                    break
            if merged:
                break
    return out


# ---------------------------------------------------------------- 3b per-bar refinement and growth
def support_f(V, X, rho, n, pitch, gate=None, max_deg=70):
    """Like support(), but only frontal views (diaphragm normal n) with the whole speaker in the frame.
    gate: match radius in pitches (for predicted positions); default the strict pixel tolerance."""
    obs, Ps, refs = [], [], []
    for k, v in enumerate(V):
        if not len(v["u"]):
            continue
        d = v["C"] - X
        if n is not None and d @ n < np.cos(np.radians(max_deg)) * np.linalg.norm(d):
            continue
        Xc = v["R"] @ X + v["t"]
        if Xc[2] <= 0:
            continue
        p = v["K"] @ Xc; p = p[:2] / p[2]
        ppx = pitch * v["f"] / Xc[2]; m = 0.45 * ppx
        if not (m <= p[0] < v["w"] - m and m <= p[1] < v["h"] - m):
            continue
        dd = np.linalg.norm(v["u"] - p, axis=1); j = int(np.argmin(dd))
        lim = gate * ppx if gate else TOL * v["w"]
        if dd[j] < lim and 0.7 < v["r"][j] * Xc[2] / v["f"] / rho < 1.4 and 0.11 < v["r"][j] / ppx < 0.30:
            obs.append(v["u"][j]); Ps.append(v["P"]); refs.append((k, j))
    return obs, Ps, refs


def fit_point(V, X0, rho, n, pitch, gate=None):
    obs, Ps, refs = support_f(V, X0, rho, n, pitch, gate)
    if len(obs) < MIN_VIEWS:
        return None
    X = solve(obs, Ps)[0]
    for _ in range(2):
        obs, Ps, refs = support_f(V, X, rho, n, pitch)
        if len(obs) < MIN_VIEWS or spread_deg(V, X, refs) < MIN_SPREAD_DEG:
            return None
        X, e, _ = solve(obs, Ps)
    return (X, refs, e, obs, Ps)


def bar_normal(V, Q, refs, up):
    c = Q.mean(0); u = np.linalg.svd(Q - c)[2][0]
    n = np.cross(up, u); n /= np.linalg.norm(n)
    cam = np.mean([V[k]["C"] for k, _ in refs], 0)
    return n if n @ (cam - c) >= 0 else -n


def refine_grow(V, keep, bars, pitch, up, rho, per_bar):
    """Re-solve every bar member from frontal, complete views only, then look for the missing members
    at the positions the bar line predicts (the main pipeline's bar model, without annotations)."""
    allX = [k[0] for k in keep]
    result = []
    for bi, b in enumerate(bars):
        mem = [keep[i] for i in b if i is not None]
        if len(mem) < 2:
            continue
        n = bar_normal(V, np.array([m[0] for m in mem]), [r for m in mem for r in m[1]], up)
        new = {}
        for s_, i in enumerate(b):
            if i is not None:
                r = fit_point(V, keep[i][0], rho, n, pitch) or fit_point(V, keep[i][0], rho, None, pitch)
                if r:
                    new[s_] = r
        if len(new) < 2:
            continue
        for _ in range(per_bar):
            if len(new) >= per_bar:
                break
            lo, hi = min(new), max(new)
            cand = [s_ for s_ in range(hi - per_bar + 1, lo + per_bar) if s_ not in new and max(hi, s_) - min(lo, s_) < per_bar]
            ks = sorted(new); coef = np.polyfit(np.array(ks, float), np.array([new[k][0] for k in ks]), 1)
            added = False
            for s_ in sorted(cand, key=lambda t: min(abs(t - k) for k in ks)):
                Xp = coef[0] * s_ + coef[1]
                r = fit_point(V, Xp, rho, n, pitch, gate=0.25)
                if r and np.linalg.norm(r[0] - Xp) < 0.1 * pitch and all(np.linalg.norm(r[0] - new[k][0]) > 0.5 * pitch for k in new):
                    new[s_] = r; added = True; break
            if not added:
                break
        result.append(new)
    # drop duplicates claimed by two bars (keep the bar where it has more views)
    flat = sorted(((len(r[1]), bi, s_) for bi, nw in enumerate(result) for s_, r in nw.items()), reverse=True)
    taken = []
    for _, bi, s_ in flat:
        X = result[bi][s_][0]
        if any(np.linalg.norm(X - t) < 0.5 * pitch for t in taken):
            del result[bi][s_]
        else:
            taken.append(X)
    keep2, bars2 = [], []
    for nw in result:
        if not nw:
            continue
        lo, hi = min(nw), max(nw); row = []
        for s_ in range(lo, hi + 1):
            if s_ in nw:
                row.append(len(keep2)); keep2.append(nw[s_])
            else:
                row.append(None)
        bars2.append(row)
    return keep2, bars2


def split_long(bars, P, per_bar):
    """A chain longer than one bar (two collinear bars) is cut at its widest step (bar gap > pitch)."""
    out, todo = [], list(bars)
    while todo:
        b = todo.pop()
        if len(b) <= per_bar:
            out.append(b); continue
        steps = [np.linalg.norm(P[b[k + 1]] - P[b[k]]) if b[k] is not None and b[k + 1] is not None else 0 for k in range(len(b) - 1)]
        k = int(np.argmax(steps))
        todo += [b[:k + 1], b[k + 1:]]
    return out


def analyse(a):
    os.makedirs(a.out, exist_ok=True)
    det = detect_all(a.model, a.images, a.out)
    log(f"偵測完成：{sum(1 for v in det.values() if v)} 張照片有振膜")
    if a.only:                                   # restrict to a list of photo names (testing)
        ok = set(json.load(open(a.only)))
        det = {k: (v if k in ok else []) for k, v in det.items()}
    V = load_views(a.model, det)
    keep, rho0 = triangulate(V)
    if len(keep) < 4:
        raise RuntimeError(f"只找到 {len(keep)} 顆喇叭，無法繼續")
    P = np.array([k[0] for k in keep])
    D = np.linalg.norm(P[:, None] - P[None], axis=2); np.fill_diagonal(D, np.inf)
    pitch = float(np.median(D.min(1)))
    log(f"架內間距（模型單位）{pitch:.5f}")
    log("PROGRESS 92 分架、地板")
    min_len = max(3, a.per_bar // 2)
    def grouped(P):
        bs = split_long([[None if i is None else int(i) for i in b] for b in chains(P, pitch)], P, a.per_bar)
        return [b for b in bs if sum(m is not None for m in b) >= min_len]
    bars = grouped(P)
    # vertical: all bars are horizontal -> smallest principal direction of the bar directions
    dirs = []
    for b in bars:
        m = [i for i in b if i is not None]
        if len(m) >= 3:
            dirs.append(P[m[-1]] - P[m[0]])
    dirs = np.array([d / np.linalg.norm(d) for d in dirs]) if dirs else np.zeros((0, 3))
    down_cam = np.mean([v["R"].T @ np.array([0, 1.0, 0]) for v in V], 0)
    ev = np.linalg.eigh(dirs.T @ dirs)[0] if len(dirs) else None
    if len(dirs) >= 2 and ev[1] > 0.05 * ev[2]:
        up = np.linalg.eigh(dirs.T @ dirs)[1][:, 0]
        up_src = "bars"
    else:                                       # one straight row only: camera "down" orthogonalised to the row
        up = -down_cam
        if len(dirs):
            up -= (up @ dirs[0]) * dirs[0]
        up_src = "camera"
    up /= np.linalg.norm(up)
    if up @ down_cam > 0:
        up = -up
    n0 = len(keep)
    for it in range(2):
        keep, bars = refine_grow(V, keep, bars, pitch, up, rho0, a.per_bar)
        P = np.array([k[0] for k in keep])
        bars = grouped(P)
        log(f"PROGRESS {94 + 2 * it} 逐架精修與補找（第 {it + 1} 輪）：{len(keep)} 顆、{len(bars)} 根架")
    used = {i for b in bars for i in b if i is not None}
    stray = [i for i in range(len(keep)) if i not in used]
    log(f"精修後 {len(used)} 顆在架上（第一輪 {n0} 個點）；{len(stray)} 個不在架上的點已排除")
    e1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    # floor: densest level of the sparse points at least ~0.3 m (3.75 pitches) below the speakers
    _, pts = read_points3D(os.path.join(a.model, "points3D.bin"))
    h = pts @ up; hs = P @ up
    low = h[h < hs.min() - 3.75 * pitch]
    floor_h, floor_n = None, int(len(low))
    if len(low) > 50:
        bw = 0.125 * pitch                     # ~10 mm bins
        hist, edges = np.histogram(low, bins=max(10, int((low.max() - low.min()) / bw)))
        k = int(np.argmax(hist)); c = (edges[k] + edges[k + 1]) / 2
        sel = low[np.abs(low - c) < 0.4 * pitch]
        floor_h, floor_n = float(np.median(sel)), int(len(sel))
    # which side the photos were taken from (for the facing direction)
    items = []
    for i, (X, refs, e, obs, Ps) in enumerate(keep):
        cams_seen = np.array([V[k]["C"] for k, _ in refs])
        boot = []
        rng = np.random.default_rng(i)
        for _ in range(60):
            s = rng.integers(0, len(obs), len(obs))
            if len({refs[j][0] for j in s}) >= 3:
                try:
                    boot.append(solve([obs[j] for j in s], [Ps[j] for j in s])[0])
                except np.linalg.LinAlgError:
                    pass
        sd = float(np.sqrt(np.trace(np.cov(np.array(boot).T)))) if len(boot) > 5 else None
        items.append(dict(X=X.tolist(), n_views=len(refs), res_px=float(np.median(e)), sd_units=sd,
                          cam_mean=cams_seen.mean(0).tolist(), refs=[[int(k), int(j)] for k, j in refs]))
    json.dump(dict(model=os.path.abspath(a.model), images=os.path.abspath(a.images), per_bar=a.per_bar,
                   views=[v["name"] for v in V], view_w=[v["w"] for v in V], pitch_units=pitch, rho_units=rho0, up=up.tolist(), up_source=up_src,
                   e1=e1.tolist(), e2=e2.tolist(), floor_h=floor_h, floor_points=floor_n,
                   points=items, bars=bars, stray=stray), open(os.path.join(a.out, "analysis.json"), "w"), indent=0)
    lens = [len(b) for b in bars]
    log(f"分出 {len(bars)} 根架子，長度 {sorted(lens, reverse=True)}；地板點 {floor_n}；垂直方向來源 {up_src}")
    log("PROGRESS 100 完成，等待設定編號")


# ---------------------------------------------------------------- assign numbers, frame, outputs
def bar_slots(A, b):
    """Positions (model units) of every slot of a bar, including missing speakers (predicted on the line)."""
    P = np.array([p["X"] for p in A["points"]])
    idx = [i for i, m in enumerate(b) if m is not None]
    t = np.array(idx, float); Q = P[[b[i] for i in idx]]
    if len(idx) >= 2:
        coef = np.polyfit(t, Q, 1)
        pred = lambda s: coef[0] * s + coef[1]
    else:
        pred = lambda s: Q[0]
    return [(P[m] if m is not None else pred(s), m) for s, m in enumerate(b)]


def numbering(A, spec):
    """Ring order of the bars around the array centre, then IDs along that direction."""
    up, e1, e2 = (np.array(A[k]) for k in ("up", "e1", "e2"))
    bars = A["bars"]
    slots = [bar_slots(A, b) for b in bars]
    allp = np.array([p for s in slots for p, _ in s]); c = allp.mean(0)
    ang = lambda X: np.arctan2((X - c) @ e2, (X - c) @ e1)
    sign = -1 if spec.get("clockwise", True) else 1                         # clockwise seen from above
    bang = np.array([ang(np.mean([p for p, _ in s], 0)) for s in slots])
    start = int(spec.get("start_bar", 0)) % len(bars)
    rel = (sign * (bang - bang[start])) % (2 * np.pi)
    order = [int(b) for b in np.argsort(rel)]
    ov = {int(k): v for k, v in (spec.get("overrides") or {}).items()}
    nid = int(spec.get("first_id", 1)); out = {}
    order = [bi for bi in order if not (spec.get("overrides") or {}).get(str(bi), {}).get("exclude")]
    for bi in order:
        s = slots[bi]
        a0 = ang(s[0][0]); a1 = ang(s[-1][0])
        fwd = (sign * ((a1 - a0 + np.pi) % (2 * np.pi) - np.pi)) > 0
        seq = list(range(len(s))) if fwd else list(range(len(s)))[::-1]
        if ov.get(bi, {}).get("reverse"):
            seq = seq[::-1]
        if ov.get(bi, {}).get("first"):
            nid = int(ov[bi]["first"])
        ids = []
        for k in seq:
            out[nid] = dict(X=s[k][0], point=s[k][1], bar=bi, slot=k); ids.append(nid); nid += 1
    return out, order


def assign(a):
    A = json.load(open(os.path.join(a.out, "analysis.json")))
    spec = json.load(open(a.spec, encoding="utf-8"))
    up, e1, e2 = (np.array(A[k]) for k in ("up", "e1", "e2"))
    S, order = numbering(A, spec)
    ids = sorted(S)
    X = np.array([S[i]["X"] for i in ids])
    pitch_mm = float(spec.get("pitch_mm") or 80.0)
    mmu, scale_src = pitch_mm / A["pitch_units"], f"架內間距 {pitch_mm} mm"
    tape = spec.get("tape")
    if tape and tape.get("mm") and tape.get("a") in S and tape.get("b") in S:
        mmu = float(tape["mm"]) / np.linalg.norm(S[tape["a"]]["X"] - S[tape["b"]]["X"])
        scale_src = f"捲尺 {tape['a']}→{tape['b']} = {tape['mm']} mm"
    T = np.c_[X @ e1, X @ e2] * mmu                                        # top view, mm
    H = X @ up
    kh = spec.get("known_height")
    if kh and kh.get("id") in S and kh.get("mm") is not None:
        h0, h_src = S[kh["id"]]["X"] @ up - float(kh["mm"]) / mmu, f"已知 {kh['id']} 號高 {kh['mm']} mm"
    elif A["floor_h"] is not None:
        h0, h_src = A["floor_h"], f"地板（{A['floor_points']} 個點）"
    else:
        h0, h_src = H.min() - 1000 / mmu, "找不到地板：高度只是相對值（最低一顆 = 1000 mm）"
    Z = (H - h0) * mmu
    pos = {i: k for k, i in enumerate(ids)}
    # frame: origin = floor point under the mean of origin_ids (+ offset), x from x_from to x_to
    oid = [i for i in (spec.get("origin_ids") or []) if i in pos] or ids
    xf, xt = spec.get("x_from"), spec.get("x_to")
    if xf in pos and xt in pos:
        xa = T[pos[xt]] - T[pos[xf]]
    else:
        b0 = [i for i in ids if S[i]["bar"] == order[0]]; xa = T[pos[b0[-1]]] - T[pos[b0[0]]]
    xa /= np.linalg.norm(xa); ya = np.array([-xa[1], xa[0]])
    org = T[[pos[i] for i in oid]].mean(0) + float(spec.get("offset", [0, 0])[0]) * xa + float(spec.get("offset", [0, 0])[1]) * ya
    Pm = (T - org) @ np.stack([xa, ya]).T
    # facing: normal of the bar line, towards the side the photos were taken from
    N = np.zeros_like(Pm)
    for bi in set(S[i]["bar"] for i in ids):
        m = [pos[i] for i in ids if S[i]["bar"] == bi]
        Q = Pm[m]; c = Q.mean(0)
        u = np.linalg.svd(Q - c)[2][0] if len(m) > 1 else xa
        n = np.array([-u[1], u[0]])
        cams = [A["points"][S[i]["point"]]["cam_mean"] for i in ids if S[i]["bar"] == bi and S[i]["point"] is not None]
        cm = ((np.c_[np.array(cams) @ e1, np.array(cams) @ e2] * mmu - org) @ np.stack([xa, ya]).T).mean(0)
        N[m] = n if n @ (cm - c) >= 0 else -n
    az = np.degrees(np.arctan2(Pm[:, 0], Pm[:, 1])); dist = np.linalg.norm(Pm, axis=1) / 1000
    if a.cmd == "preview":
        json.dump(dict(speakers=[dict(id=i, x=round(Pm[k, 0], 1), y=round(Pm[k, 1], 1), z=round(Z[k], 1), bar=S[i]["bar"],
                                      predicted=S[i]["point"] is None, nx=round(N[k, 0], 4), ny=round(N[k, 1], 4)) for k, i in enumerate(ids)],
                       order=[int(b) for b in order], n_bars=len(order), per_bar=A["per_bar"], scale=scale_src, height=h_src),
                  open(os.path.join(a.out, "preview.json"), "w", encoding="utf-8"), ensure_ascii=False)
        return
    rows = []
    for k, i in enumerate(ids):
        pi = S[i]["point"]; p = A["points"][pi] if pi is not None else None
        rows.append(dict(id=i, bar=order.index(S[i]["bar"]) + 1, x_mm=round(Pm[k, 0], 1), y_mm=round(Pm[k, 1], 1), height_mm=round(Z[k], 1),
                         azimuth_deg=round(az[k], 4), distance_m=round(dist[k], 5), dir_x=round(N[k, 0], 6), dir_y=round(N[k, 1], 6),
                         n_views=p["n_views"] if p else 0, sd_mm=round(p["sd_units"] * mmu, 2) if p and p["sd_units"] else None,
                         predicted=p is None))
    with open(os.path.join(a.out, "speakers.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bus = int(spec.get("bus", 10)); AK, DK = f"/bus/{bus}/speakers/aed", f"/bus/{bus}/speakers/directions/xyz"
    lines = {AK: AK + " " + " ".join(f"{r['azimuth_deg']:.4f} 0.0 {r['distance_m']:.5f}" for r in rows),
             DK: DK + " " + " ".join(f"{r['dir_x']:.6f} {r['dir_y']:.6f} 0.0" for r in rows)}
    open(os.path.join(a.out, f"panoramix_bus{bus}.txt"), "w", encoding="utf-8").write(
        f"# {len(rows)} speakers, ids {ids[0]}..{ids[-1]} in this order. Elevation 0; heights in speakers.csv.\n" + lines[AK] + "\n" + lines[DK] + "\n")
    design = spec.get("design") or ""
    if design and os.path.exists(design):
        raw = open(design, "rb").read(); eol = b"\r\n" if b"\r\n" in raw else b"\n"
        key = lambda l: l.split(b" ", 1)[0].decode("utf-8", "ignore")
        o = [lines[key(l)].encode() if key(l) in lines else l for l in raw.split(eol)]
        open(os.path.join(a.out, os.path.splitext(os.path.basename(design))[0] + "_custom.txt"), "wb").write(eol.join(o))
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
    from export_ply import write_ply
    write_ply(os.path.join(a.out, "speakers_points.ply"), np.c_[Pm, Z], normals=np.c_[N, np.zeros(len(N))], ids=np.array(ids),
              comment="auto_locate, mm")
    # report
    gaps, straight = [], []
    for bi in set(S[i]["bar"] for i in ids):
        m = [k for k, i in enumerate(ids) if S[i]["bar"] == bi and not rows[k]["predicted"]]
        Q = np.c_[Pm[m], Z[m]]
        if len(m) >= 2:
            sl = sorted(m, key=lambda k: S[ids[k]]["slot"])
            for p_, q_ in zip(sl, sl[1:]):
                if S[ids[q_]]["slot"] - S[ids[p_]]["slot"] == 1:
                    gaps.append(float(np.linalg.norm(np.r_[Pm[q_], Z[q_]] - np.r_[Pm[p_], Z[p_]])))
        if len(m) >= 3:
            c = Q.mean(0); u = np.linalg.svd(Q - c)[2][0]; r = (Q - c) - np.outer((Q - c) @ u, u)
            straight += list(np.linalg.norm(r, axis=1))
    sds = [r["sd_mm"] for r in rows if r["sd_mm"] is not None]
    rep = dict(n_speakers=len(rows), ids=[ids[0], ids[-1]], n_predicted=sum(r["predicted"] for r in rows),
               bars=len(order), bar_lengths=[sum(1 for i in ids if S[i]["bar"] == b) for b in order],
               scale=scale_src, mm_per_unit=mmu, height=h_src, up_source=A["up_source"],
               in_bar_gap_mm=dict(mean=round(float(np.mean(gaps)), 2), sd=round(float(np.std(gaps)), 2)) if gaps else None,
               straightness_mm_median=round(float(np.median(straight)), 2) if straight else None,
               sd_mm_median=round(float(np.median(sds)), 2) if sds else None, sd_mm_p90=round(float(np.percentile(sds, 90)), 2) if sds else None,
               views_median=int(np.median([r["n_views"] for r in rows if r["n_views"]])),
               height_mm_range=[round(float(Z.min()), 1), round(float(Z.max()), 1)])
    rep["warnings"] = [w for w in [
        f"{rep['n_predicted']} 顆照片不夠，由同架其他顆推算" if rep["n_predicted"] else "",
        f"有架子不是 {A['per_bar']} 顆：{rep['bar_lengths']}" if any(l != A["per_bar"] for l in rep["bar_lengths"]) else "",
        "高度沒有地板可參考" if "找不到" in h_src else ""] if w]
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    check_images(a.out, A, S, ids)
    log(json.dumps(rep, ensure_ascii=False))


def check_images(out, A, S, ids, n=4):
    """Photos with the most speakers, numbers drawn on them, to compare with the labels on the bars."""
    det = json.load(open(os.path.join(out, "detections.json")))
    per_view = defaultdict(list)
    for i in ids:
        p = S[i]["point"]
        if p is not None:
            for k, j in A["points"][p]["refs"]:
                per_view[k].append((i, j))
    best = sorted(per_view, key=lambda k: -len(per_view[k]))
    chosen = []
    for k in best:                              # spread over different parts of the array
        if all(len(set(i for i, _ in per_view[k]) & set(i for i, _ in per_view[c])) < 0.5 * len(per_view[k]) for c in chosen):
            chosen.append(k)
        if len(chosen) == n:
            break
    for f in os.listdir(out):
        if f.startswith("check_") and f.endswith(".jpg"):
            os.remove(os.path.join(out, f))
    for c, k in enumerate(chosen):
        name = A["views"][k]
        img = cv2.imread(os.path.join(A["images"], name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            continue
        D = det[name]
        sc = 1600 / A["view_w"][k]; img = cv2.resize(img, (1600, int(round(img.shape[0] * 1600 / img.shape[1]))), interpolation=cv2.INTER_AREA)
        for i, j in per_view[k]:
            x, y = D[j]["xy"]
            x, y = x * sc, y * sc
            r = max(6, D[j]["r"] * sc)
            cv2.circle(img, (int(x), int(y)), int(r * 1.6), (0, 140, 255), 2)
            cv2.putText(img, str(i), (int(x - r), int(y - 2.2 * r)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, str(i), (int(x - r), int(y - 2.2 * r)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out, f"check_{c + 1}_{os.path.splitext(os.path.basename(name))[0]}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyse", "assign", "preview"])
    ap.add_argument("--model"); ap.add_argument("--images"); ap.add_argument("--out", required=True)
    ap.add_argument("--per-bar", type=int, default=8); ap.add_argument("--spec"); ap.add_argument("--only")
    a = ap.parse_args()
    analyse(a) if a.cmd == "analyse" else assign(a)  # assign also does "preview" (numbers + frame only)


if __name__ == "__main__":
    main()
