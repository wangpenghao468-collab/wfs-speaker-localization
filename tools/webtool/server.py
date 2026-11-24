"""Local web tool: pick a photo folder and options, then build a 3D Gaussian Splatting scene or
locate the speakers in a coordinate frame you choose.

    python tools/webtool/server.py            # opens http://127.0.0.1:<free port>/ in the browser

Only listens on 127.0.0.1 (this computer). It takes the first free port from 8790 upwards, so it never
collides with other local servers (SAM2 7860/7861, file preview 8765, 3DGS viewers 6007/6009 ...);
the 3DGS trainers it starts also get their own free ports. Python standard library only."""
import json, os, string, subprocess, sys, threading, time, urllib.parse, webbrowser, socket
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "pipeline"))
import config  # noqa: E402

S = config._S
JOBS_DIR = config._p("jobs_dir", "work/webtool_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
LOCK = threading.Lock()


def pid_alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0); return True
        except OSError:
            return False
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong(); ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code)); ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259                                                  # STILL_ACTIVE


def job_pid(jid):
    p = os.path.join(JOBS_DIR, jid, "pid")
    return int(open(p).read()) if os.path.exists(p) else None


def free_port(start=8790):
    for p in range(start, start + 200):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p)); s.close(); return p
        except OSError:
            s.close()
    raise SystemExit("no free port")


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return [dict(zip(["index", "name", "mem_used", "mem_total", "util", "temp"], [v.strip() for v in l.split(",")]))
                for l in out.strip().splitlines()]
    except Exception:
        return []


def list_dir(path):
    if not path:
        return {"path": "", "parent": None, "dirs": [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")],
                "images": 0, "home": [os.path.expanduser("~\\Desktop"), os.path.expanduser("~\\Documents"), config.SONY_IMAGES.replace("/", "\\")]}
    path = os.path.abspath(path)
    dirs, n = [], 0
    try:
        for e in os.scandir(path):
            if e.is_dir() and not e.name.startswith(("$", ".")):
                dirs.append(e.name)
            elif e.is_file() and e.name.lower().endswith(IMG_EXT):
                n += 1
    except OSError as ex:
        return {"path": path, "error": str(ex), "dirs": [], "images": 0, "parent": os.path.dirname(path)}
    parent = os.path.dirname(path) if os.path.dirname(path) != path else ""
    is_model = os.path.exists(os.path.join(path, "images.bin"))
    return {"path": path, "parent": parent, "dirs": sorted(dirs, key=str.lower), "images": n, "colmap_model": is_model}


def job_info(jid):
    d = os.path.join(JOBS_DIR, jid)
    try:
        job = json.load(open(os.path.join(d, "job.json"), encoding="utf-8"))
    except Exception:
        return None
    st = {}
    if os.path.exists(os.path.join(d, "status.json")):
        try:
            st = json.load(open(os.path.join(d, "status.json"), encoding="utf-8"))
        except Exception:
            pass
    pid = job_pid(jid)
    running = pid is not None and pid_alive(pid) and st.get("state") not in ("done", "error", "needs_ids")
    if st.get("state") == "running" and not running:
        st["state"] = "stopped"
    return {"id": jid, "dir": d, "job": job, "status": st, "running": running}


def start_job(opts):
    kind = opts.get("type")
    name = "".join(c for c in (opts.get("name") or kind) if c.isalnum() or c in "-_")[:40] or kind
    jid = time.strftime("%Y%m%d-%H%M%S") + "_" + name
    d = os.path.join(JOBS_DIR, jid); os.makedirs(d)
    json.dump(opts, open(os.path.join(d, "job.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    launch(d, kind)
    return jid


def launch(d, kind, append=False):
    """Start the job in folder d, or continue it after a failure (the job script skips finished steps)."""
    script = {"gs": "gs_job.py", "speakers": "speaker_job.py", "locate": "locate_job.py"}[kind]
    log = open(os.path.join(d, "log.txt"), "a" if append else "w", encoding="utf-8")
    if append:
        log.write(f"\n===== 重試 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"); log.flush()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    # own hidden console: the job keeps running if the tool window is closed, and COLMAP opens no windows
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    p = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, script), d], stdout=log, stderr=subprocess.STDOUT, env=env,
                         creationflags=flags, cwd=REPO)
    open(os.path.join(d, "pid"), "w").write(str(p.pid))


def speakers_preview():
    import csv
    rows = list(csv.DictReader(open(os.path.join(config.OUT_DIR, "panoramix_bus10_measured.csv"), encoding="utf-8-sig")))
    return [{"id": int(r["speaker_id"]), "x": float(r["x_m"]) * 1000, "y": float(r["y_m"]) * 1000} for r in rows]


ALLOWED = [JOBS_DIR, config.OUT_DIR, os.path.join(REPO, "measurements")]


def allowed(path):
    a = os.path.normcase(os.path.abspath(path))
    return any(a.startswith(os.path.normcase(os.path.abspath(r))) for r in ALLOWED if r)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = dict(urllib.parse.parse_qsl(u.query))
        if u.path in ("/", "/index.html"):
            return self.send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            jobs = [job_info(j) for j in sorted(os.listdir(JOBS_DIR), reverse=True)[:40]]
            return self.send(200, {"gpus": gpus(), "jobs": [j for j in jobs if j], "jobs_dir": JOBS_DIR.replace("/", "\\"),
                                   "defaults": {"sony_images": config.SONY_IMAGES.replace("/", "\\"),
                                                "design": (config.DESIGN_TXT or "").replace("/", "\\"),
                                                "sony_posed": bool(S.get("sony_posed_model"))}})
        if u.path == "/api/ls":
            return self.send(200, list_dir(q.get("path", "")))
        if u.path == "/api/speakers":
            return self.send(200, speakers_preview())
        if u.path == "/api/log":
            p = os.path.join(JOBS_DIR, q["id"], "log.txt")
            txt = open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
            lines = [l.split("\r")[-1] for l in txt.splitlines()]
            return self.send(200, {"lines": lines[-int(q.get("n", 200)):]})
        if u.path == "/api/locate":
            out = os.path.join(JOBS_DIR, q["id"], "out")
            r = {"ready": os.path.exists(os.path.join(out, "analysis.json"))}
            for k in ("preview", "report", "spec"):
                f = os.path.join(out, k + ".json")
                if os.path.exists(f):
                    r[k] = json.load(open(f, encoding="utf-8"))
            r["checks"] = sorted(os.path.join(out, f) for f in os.listdir(out) if f.startswith("check_")) if os.path.isdir(out) else []
            return self.send(200, r)
        if u.path == "/api/file":
            p = q.get("path", "")
            if not (allowed(p) and os.path.isfile(p)):
                return self.send(403, {"error": "not allowed"})
            ext = os.path.splitext(p)[1].lower()
            ctype = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".txt": "text/plain; charset=utf-8",
                     ".csv": "text/csv; charset=utf-8", ".json": "application/json"}.get(ext, "application/octet-stream")
            return self.send(200, open(p, "rb").read(), ctype)
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if u.path == "/api/jobs":
            if body.get("type") in ("gs", "locate") and not os.path.isdir(body.get("images", "")):
                return self.send(400, {"error": "照片資料夾不存在"})
            with LOCK:
                return self.send(200, {"id": start_job(body)})
        if u.path == "/api/assign":                       # numbers + frame for a new-room job (seconds)
            jd = os.path.join(JOBS_DIR, body["id"]); out = os.path.join(jd, "out")
            if not os.path.exists(os.path.join(out, "analysis.json")):
                return self.send(400, {"error": "這個工作還沒找完喇叭"})
            spec = os.path.join(out, "spec.json")
            json.dump(body["spec"], open(spec, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            cmd = "preview" if body.get("preview") else "assign"
            p = subprocess.run([sys.executable, "-u", os.path.join(REPO, "pipeline", "auto_locate.py"), cmd, "--out", out, "--spec", spec],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
                               env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if p.returncode != 0:
                return self.send(500, {"error": (p.stdout + p.stderr)[-1500:]})
            if cmd == "assign":
                files = [os.path.join(out, f) for f in sorted(os.listdir(out)) if f.endswith((".csv", ".txt", ".ply")) or f.startswith("check_")]
                sp = os.path.join(jd, "status.json"); st = json.load(open(sp, encoding="utf-8"))
                st.update(state="done", step="完成", result={"outputs": out, "files": files}, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
                json.dump(st, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                with open(os.path.join(jd, "log.txt"), "a", encoding="utf-8") as lg:
                    lg.write(f"\n===== 輸出 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n" + p.stdout[-3000:] + "\n")
            return self.send(200, {"ok": True})
        if u.path == "/api/retry":
            info = job_info(body["id"])
            if not info or info["running"]:
                return self.send(400, {"error": "工作不存在或仍在執行"})
            with LOCK:
                launch(info["dir"], info["job"]["type"], append=True)
            return self.send(200, {"ok": True})
        if u.path == "/api/cancel":
            pid = job_pid(body["id"])
            if pid and pid_alive(pid):
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
                sp = os.path.join(JOBS_DIR, body["id"], "status.json")
                st = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
                st.update(state="cancelled", step="已取消"); json.dump(st, open(sp, "w", encoding="utf-8"), ensure_ascii=False)
            return self.send(200, {"ok": True})
        if u.path == "/api/open":
            p = body.get("path", "")
            if os.path.exists(p):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(p)] if os.path.isfile(p) else ["explorer", os.path.normpath(p)])
                return self.send(200, {"ok": True})
            return self.send(404, {"error": "不存在"})
        return self.send(404, {"error": "not found"})


def main():
    port = free_port(int(os.environ.get("WEBTOOL_PORT", 8790)))
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    open(os.path.join(HERE, ".port"), "w").write(str(port))
    print(f"處理工具：{url}  （只在這台電腦上可用；關閉這個視窗即停止。執行中的工作會繼續跑完）", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
