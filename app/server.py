"""
Local web app for the neuromarketing pipeline (one server, several pages).

  python app/server.py            -> http://127.0.0.1:8765/

Pages:
  /setup    experiment setup: participant, which ads, questions per ad, baseline;
            checks the EEG stream and launches the recording task
  /review   scene editor: check and fix the labeller's scenes and brand events,
            then approve (freeze) them

Listens on 127.0.0.1 only (this laptop). Standard library (+ pylsl for the
EEG-stream check).
"""
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "pipeline"))
sys.path.insert(0, os.path.join(PROJ, "lab_task_psychopy"))
sys.path.insert(0, HERE)
import label_scenes as LS  # noqa: E402
import session_config as SC  # noqa: E402

TASK = os.path.join(PROJ, "lab_task_psychopy", "ad_viewing_task.py")
RUN = {"proc": None, "pid": None, "log": None}       # the one recording that may be running
RUN_LOCK = threading.Lock()
NOISE = ("MovieStim.__del__", "_freePlayer", "self.unload()", "^^^^", "Exception ignored", "Traceback (most recent call last)",
         "movies\\__init__.py", "INFO|", "[h264 @", "ffmpeg-devel", "Late SEI")

ADS_DIR = os.path.join(PROJ, "data", "lab_session", "ads")
SCENES_DIR = os.path.join(PROJ, "data", "lab_session", "scenes")
STATIC = os.path.join(HERE, "static")
PORT = int(os.environ.get("NM_PORT", "8765"))
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".webm")


def _ads():
    out = []
    for f in sorted(os.listdir(ADS_DIR)):
        base, ext = os.path.splitext(f)
        if ext.lower() not in VIDEO_EXTS:
            continue
        sp = os.path.join(SCENES_DIR, base + ".scenes.json")
        status = None
        if os.path.exists(sp):
            try:
                status = json.load(open(sp)).get("status")
            except Exception:
                status = "unreadable"
        out.append({"ad": base, "file": f, "scenes_status": status})
    return out


def _ad_info():
    out = []
    for f in SC.list_ads():
        base = os.path.splitext(f)[0]
        info = {"file": f, "ad": base, "duration_s": None, "brand": "", "product": "", "scenes_status": None}
        sp = os.path.join(SCENES_DIR, base + ".scenes.json")
        if os.path.exists(sp):
            try:
                d = json.load(open(sp, encoding="utf-8"))
                info.update(duration_s=d.get("duration_s"), brand=d.get("story", {}).get("brand", ""),
                            product=d.get("story", {}).get("product", ""), scenes_status=d.get("status"))
            except Exception:
                pass
        prof = SC.load_profile(base) or {}
        info["recall_ready"] = bool(prof.get("brand") and prof.get("brand") in (prof.get("recall_options") or []))
        info["has_objectives"] = bool(prof.get("profile"))
        if not info["brand"] and prof.get("brand"):
            info["brand"] = prof["brand"]
        if info["duration_s"] is None:
            try:
                import cv2
                cap = cv2.VideoCapture(os.path.join(ADS_DIR, f))
                n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                info["duration_s"] = round(n / fps, 2) if fps else None
            except Exception:
                pass
        out.append(info)
    return out


def _existing_sessions():
    lab = SC.LAB
    ids = set()
    for f in os.listdir(lab):
        m = re.match(r"^([A-Za-z0-9_\-]+)_(raw\.fif|ratings\.csv)$", f)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def _run_status():
    with RUN_LOCK:
        p = RUN["proc"]
        if p is None:
            return {"state": "idle"}
        code = p.poll()
        tail = []
        try:
            with open(RUN["log"], encoding="utf-8", errors="replace") as fh:
                tail = [l.rstrip() for l in fh if l.strip() and not any(n in l for n in NOISE)][-40:]
        except OSError:
            pass
        out_base = os.path.join(SC.LAB, RUN["pid"])
        files = [os.path.basename(out_base + s) for s in
                 ("_raw.fif", "_ratings.csv", "_framelog.csv", "_session.json") if os.path.exists(out_base + s)]
        # progress from the whole log: which phase the task is in
        prog = {"baseline_eo": False, "baseline_ec": False, "ads_done": 0, "current": None, "planned": None}
        try:
            with open(RUN["log"], encoding="utf-8", errors="replace") as fh:
                for l in fh:
                    if "BASELINE_EO_END" in l:
                        prog["baseline_eo"] = True
                    elif "BASELINE_EC_END" in l:
                        prog["baseline_ec"] = True
                    elif "[marker] AD_START:" in l:
                        prog["current"] = l.split("AD_START:", 1)[1].strip()
                    elif "[marker] AD_END:" in l:
                        prog["ads_done"] += 1
                        prog["current"] = None
                    elif l.startswith("[session]") and "ad(s)" in l:
                        m = re.search(r"(\d+) ad\(s\)", l)
                        prog["planned"] = int(m.group(1)) if m else None
        except OSError:
            pass
        return {"state": "running" if code is None else ("finished" if code == 0 else "failed"),
                "returncode": code, "participant_id": RUN["pid"], "log": tail, "files": files, "progress": prog}


def _analysis(ad, session=None):
    import sqlite3
    db_path = os.path.join(SC.LAB, "neuro.db")
    if not os.path.exists(db_path):
        return {"error": "no database yet -- run pipeline/scene_values.py"}
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    a = db.execute("SELECT * FROM ads WHERE ad=?", (ad,)).fetchone()
    if not a:
        return {"error": f"unknown ad {ad}"}
    sessions = [r[0] for r in db.execute("SELECT DISTINCT session FROM scene_values WHERE ad=? ORDER BY session", (ad,))
                if r[0] not in _excluded()]
    session = session if session in sessions else (sessions[0] if sessions else None)
    scenes = [dict(r) for r in db.execute("SELECT * FROM scenes WHERE ad=? ORDER BY idx", (ad,))]
    vals = {}
    for r in db.execute("SELECT * FROM scene_values WHERE ad=? AND session=?", (ad, session)):
        vals.setdefault(r["scene_idx"], {})[r["measure"]] = dict(r)
    for s in scenes:
        s["values"] = vals.get(s["idx"], {})
    tl = {}
    for r in db.execute("SELECT t_s, measure, value FROM timeline WHERE ad=? AND session=? ORDER BY t_s", (ad, session)):
        tl.setdefault(r["measure"], []).append([r["t_s"], r["value"]])
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    rel = {"timeline_v2": {}, "reliability": {}, "brand_test": {}, "scene_replication": {}, "recall_text": {}}
    if "brand_test" in tables and session:
        for r in db.execute("SELECT t_s, measure, value, noisy_frac FROM timeline_repeat WHERE ad=? AND session=? "
                            "AND viewing=2 ORDER BY t_s", (ad, session)):
            noisy = r["noisy_frac"] is not None and r["noisy_frac"] > 0.3          # same gate as scenes
            rel["timeline_v2"].setdefault(r["measure"], []).append([r["t_s"], None if noisy else r["value"]])
        rel["reliability"] = {r["measure"]: dict(r) for r in db.execute(
            "SELECT * FROM reliability WHERE ad=? AND session=?", (ad, session))}
        for r in db.execute("SELECT * FROM brand_test WHERE ad=? AND session=? ORDER BY viewing", (ad, session)):
            rel["brand_test"].setdefault(r["measure"], []).append(dict(r))
        for r in db.execute("SELECT * FROM scene_replication WHERE ad=? AND session=?", (ad, session)):
            rel["scene_replication"].setdefault(r["measure"], {})[r["scene_idx"]] = r["status"]
        rel["recall_text"] = {r["question"]: r["text"] for r in db.execute(
            "SELECT question, text FROM answers_text WHERE ad=? AND session=?", (ad, session))}
    rel["viewings"] = 2 if rel["timeline_v2"] else 1
    return {
        "profile": SC.load_profile(ad), **rel,
        "ad": dict(a), "sessions": sessions, "session": session, "scenes": scenes, "timeline": tl,
        "ad_values": {r["measure"]: r["value"] for r in db.execute(
            "SELECT measure, value FROM ad_values WHERE ad=? AND session=?", (ad, session))},
        "ratings": {r["question"]: r["answer"] for r in db.execute(
            "SELECT question, answer FROM ratings WHERE ad=? AND session=?", (ad, session))},
        "transcript": [dict(r) for r in db.execute("SELECT * FROM transcript WHERE ad=? ORDER BY start_s", (ad,))],
        "brand_events": [dict(r) for r in db.execute("SELECT * FROM brand_events WHERE ad=? ORDER BY start_s", (ad,))],
        "measures": [dict(r) for r in db.execute("SELECT * FROM measures")],
        "all_ads": [dict(r) for r in db.execute(
            "SELECT a.ad, a.brand, a.product, a.duration_s, a.scenes_status, "
            "(SELECT COUNT(DISTINCT session) FROM scene_values v WHERE v.ad=a.ad) AS n, "
            "(SELECT GROUP_CONCAT(DISTINCT session) FROM scene_values v WHERE v.ad=a.ad) AS viewers FROM ads a ORDER BY a.ad")],
        "ad_detail": {r["measure"]: {"value": r["value"], "verdict": r["verdict"], "noisy_frac": r["noisy_frac"]}
                      for r in db.execute("SELECT measure, value, verdict, noisy_frac FROM ad_values WHERE ad=? AND session=?",
                                          (ad, session))},
        "session_info": dict(db.execute("SELECT * FROM sessions WHERE session=?", (session,)).fetchone() or {}) if session else {},
    }


def _timelines(ad):
    """Every clean viewing of an ad: {viewings: [{session, viewing, measures: {m: [[t, v|null]]}}]}.
    Values are relative to that viewing's own median (0 = this ad's typical level for that viewer),
    so viewers with different overall levels can be drawn together. Noisy windows are null."""
    import sqlite3
    import statistics
    db_path = os.path.join(SC.LAB, "neuro.db")
    if not os.path.exists(db_path):
        return {"viewings": []}
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    raw = {}
    for r in db.execute("SELECT session, t_s, measure, value, noisy_frac FROM timeline WHERE ad=? ORDER BY t_s", (ad,)):
        raw.setdefault((r["session"], 1), {}).setdefault(r["measure"], []).append((r["t_s"], r["value"], r["noisy_frac"]))
    try:
        for r in db.execute("SELECT session, viewing, t_s, measure, value, noisy_frac FROM timeline_repeat WHERE ad=? ORDER BY t_s", (ad,)):
            raw.setdefault((r["session"], r["viewing"]), {}).setdefault(r["measure"], []).append((r["t_s"], r["value"], r["noisy_frac"]))
    except sqlite3.OperationalError:
        pass
    out = []
    for (ses, v), meas in sorted(raw.items()):
        if ses in _excluded():
            continue
        mm = {}
        for m, pts in meas.items():
            clean = [x[1] for x in pts if x[1] is not None and (x[2] is None or x[2] <= 0.3)]
            if len(clean) < 4:
                continue
            med = statistics.median(clean)
            mm[m] = [[round(t, 2), None if (val is None or (nf is not None and nf > 0.3)) else round(val - med, 2)]
                     for t, val, nf in pts]
        if mm:
            out.append({"session": ses, "viewing": v, "measures": mm})
    return {"viewings": out}


def _excluded():
    """Viewer ids excluded by the team (data/lab_session/viewers.json)."""
    p = os.path.join(SC.LAB, "viewers.json")
    try:
        with open(p, encoding="utf-8") as fh:
            return {k for k, v in json.load(fh).items() if v.get("excluded")}
    except (OSError, ValueError):
        return set()


def _export_csv(ad):
    import csv
    import io
    import sqlite3
    db = sqlite3.connect(os.path.join(SC.LAB, "neuro.db"))
    db.row_factory = sqlite3.Row
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ad", "scene", "start_s", "end_s", "label", "story_role", "viewer", "viewing", "measure",
                "value_vs_rest", "vs_ad_typical", "verdict", "noisy_fraction"])
    sc = {r["idx"]: r for r in db.execute("SELECT * FROM scenes WHERE ad=?", (ad,))}
    rows = [(r, 1) for r in db.execute("SELECT * FROM scene_values WHERE ad=? ORDER BY session, scene_idx", (ad,))]
    try:
        rows += [(r, r["viewing"]) for r in db.execute("SELECT * FROM scene_values_repeat WHERE ad=? ORDER BY session, scene_idx", (ad,))]
    except sqlite3.OperationalError:
        pass
    for r, v in rows:
        s = sc.get(r["scene_idx"])
        if s is None:
            continue
        w.writerow([ad, r["scene_idx"], round(s["start_s"], 2), round(s["end_s"], 2), s["label"], s["story_role"],
                    r["session"], v, r["measure"], r["value"], r["vs_ad_median"], r["verdict"], r["noisy_frac"]])
    return buf.getvalue()


THUMBS = {}


def _thumb(ad, t, w):
    """JPEG frame of an ad at time t (s), width w -- cached in memory."""
    key = (ad, round(t, 1), w)
    if key in THUMBS:
        return THUMBS[key]
    import cv2
    files = [f for f in os.listdir(ADS_DIR) if os.path.splitext(f)[0] == ad and f.lower().endswith(VIDEO_EXTS)]
    if not files:
        return None
    cap = cv2.VideoCapture(os.path.join(ADS_DIR, files[0]))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 25, cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(t * fps))))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    h0, w0 = fr.shape[:2]
    fr = cv2.resize(fr, (w, int(round(h0 * w / w0))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    data = buf.tobytes() if ok else None
    if len(THUMBS) > 400:
        THUMBS.clear()
    THUMBS[key] = data
    return data


PROC = {"proc": None, "pid": None, "log": None}      # the one analysis job that may be running
LABEL = {"proc": None, "ads": [], "log": None}       # the one labelling job that may be running
LABELLER = os.path.join(PROJ, "pipeline", "label_scenes.py")


def _label_status():
    p = LABEL["proc"]
    unl = [a["ad"] for a in _ads() if a["scenes_status"] is None]
    if p is None:
        return {"state": "idle", "unlabelled": unl}
    code = p.poll()
    tail = []
    try:
        with open(LABEL["log"], encoding="utf-8", errors="replace") as fh:
            tail = [l.rstrip() for l in fh if l.strip() and "warn" not in l.lower() and "return _methods" not in l
                    and "ret = ret" not in l][-8:]
    except OSError:
        pass
    return {"state": "running" if code is None else ("finished" if code == 0 else "failed"),
            "ads": LABEL["ads"], "log": tail, "unlabelled": unl}
SCENE_VALUES = os.path.join(PROJ, "pipeline", "scene_values.py")


def _db_sessions():
    """Recorded viewers: processed ones (in the database, with the ads they watched)
    plus recordings on disk that have not been analysed yet."""
    import sqlite3
    out, seen = [], set()
    db_path = os.path.join(SC.LAB, "neuro.db")
    if os.path.exists(db_path):
        db = sqlite3.connect(db_path)
        try:
            for s, hb in db.execute("SELECT session, has_baseline FROM sessions ORDER BY session"):
                ads = [r[0] for r in db.execute("SELECT DISTINCT ad FROM scene_values WHERE session=? ORDER BY ad", (s,))]
                out.append({"session": s, "processed": True, "has_baseline": bool(hb), "ads": ads})
                seen.add(s)
        except sqlite3.Error:
            pass
        db.close()
    for s in _existing_sessions():
        if s not in seen and os.path.exists(os.path.join(SC.LAB, f"{s}_raw.fif")) and s != "channel_calibration":
            out.append({"session": s, "processed": False, "has_baseline": None, "ads": []})
    vw = _viewer_meta()
    for r in out:
        r["excluded"] = bool(vw.get(r["session"], {}).get("excluded"))
        r["note"] = vw.get(r["session"], {}).get("note", "")
    return sorted(out, key=lambda r: r["session"])


def _viewer_meta():
    try:
        with open(os.path.join(SC.LAB, "viewers.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _stop(job):
    p = job["proc"]
    if p is None or p.poll() is not None:
        return False
    p.terminate()
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
    return True


def _proc_status():
    p = PROC["proc"]
    if p is None:
        return {"state": "idle"}
    code = p.poll()
    tail = []
    try:
        with open(PROC["log"], encoding="utf-8", errors="replace") as fh:
            tail = [l.rstrip() for l in fh if l.strip() and "warn" not in l.lower()][-15:]
    except OSError:
        pass
    sess = next((r for r in _db_sessions() if r["session"] == PROC["pid"]), None)
    return {"state": "running" if code is None else ("finished" if code == 0 else "failed"),
            "participant_id": PROC["pid"], "log": tail, "ads": (sess or {}).get("ads", [])}


def _eeg_check():
    try:
        from pylsl import resolve_byprop
    except ImportError:
        return {"ok": False, "error": "pylsl is not installed in this environment"}
    streams = resolve_byprop("type", "EEG", timeout=2.5)
    rows = [{"name": s.name(), "channels": s.channel_count(), "rate": s.nominal_srate(),
             "host": s.hostname()} for s in streams]
    return {"ok": bool(rows), "streams": rows}


class Handler(BaseHTTPRequestHandler):
    server_version = "nm-app/1"

    def log_message(self, fmt, *args):  # quieter console
        if "/media/" not in self.path:
            sys.stderr.write("  " + (fmt % args) + "\n")

    # ------------------------------------------------------------------ helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype=None):
        if not os.path.isfile(path):
            return self._json({"error": "not found"}, 404)
        size = os.path.getsize(path)
        ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        if ctype.startswith("text/html"):
            self.send_header("Cache-Control", "no-store")     # always serve the current page
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass  # the browser cancels range requests while seeking; harmless

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ------------------------------------------------------------------ routes
    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/home"):
            f = "home.html" if os.path.exists(os.path.join(STATIC, "home.html")) else "results.html"
            return self._file(os.path.join(STATIC, f), "text/html; charset=utf-8")
        if p == "/ads":
            f = "ads.html" if os.path.exists(os.path.join(STATIC, "ads.html")) else "review.html"
            return self._file(os.path.join(STATIC, f), "text/html; charset=utf-8")
        if p == "/setup":
            return self._file(os.path.join(STATIC, "setup.html"), "text/html; charset=utf-8")
        if p == "/review":
            return self._file(os.path.join(STATIC, "review.html"), "text/html; charset=utf-8")
        if p in ("/analysis", "/results"):
            return self._file(os.path.join(STATIC, "results.html"), "text/html; charset=utf-8")
        if p == "/record":
            return self._file(os.path.join(STATIC, "setup.html"), "text/html; charset=utf-8")
        m = re.match(r"^/static/fonts/([A-Za-z0-9_\-]+\.woff2)$", p)
        if m:
            return self._file(os.path.join(STATIC, "fonts", m.group(1)), "font/woff2")
        m = re.match(r"^/api/timelines/([^/]+)$", p)
        if m and NAME_RE.match(m.group(1)):
            return self._json(_timelines(m.group(1)))
        m = re.match(r"^/api/export/([^/]+)\.csv$", p)
        if m and NAME_RE.match(m.group(1)):
            body = _export_csv(m.group(1)).encode("utf-8-sig")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{m.group(1)}_scenes.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        m = re.match(r"^/static/([A-Za-z0-9_\-]+\.(css|js|svg))$", p)
        if m:
            return self._file(os.path.join(STATIC, m.group(1)))
        m = re.match(r"^/api/thumb/([^/]+)$", p)
        if m and NAME_RE.match(m.group(1)):
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            try:
                t = float((qs.get("t") or ["0"])[0])
                w = max(64, min(960, int((qs.get("w") or ["320"])[0])))
            except ValueError:
                return self._json({"error": "bad t or w"}, 400)
            data = _thumb(m.group(1), t, w)
            if data is None:
                return self._json({"error": "no frame"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)
            return
        m = re.match(r"^/api/analysis/([^/]+)$", p)
        if m and NAME_RE.match(m.group(1)):
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            return self._json(_analysis(m.group(1), (qs.get("session") or [None])[0]))
        m = re.match(r"^/api/copilot/suggest/([^/]+)$", p)
        if m and NAME_RE.match(m.group(1)):
            try:
                import copilot
                return self._json({"questions": copilot.suggest(m.group(1))})
            except Exception as e:
                return self._json({"questions": [], "error": f"{type(e).__name__}: {e}"})
        m = re.match(r"^/api/copilot/brief/([^/]+)$", p)
        if m and NAME_RE.match(m.group(1)):
            from urllib.parse import parse_qs
            regen = (parse_qs(urlparse(self.path).query).get("regenerate") or ["0"])[0] == "1"
            try:
                import copilot
                return self._json(copilot.brief(m.group(1), regenerate=regen))
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        if p == "/api/ads":
            return self._json(_ads())
        if p == "/api/setup":
            return self._json({"ads": _ad_info(), "bank": SC.load_bank(), "default_questions": SC.DEFAULT_QUESTIONS,
                               "sessions": _existing_sessions(), "run": _run_status()})
        if p == "/api/session/status":
            return self._json(_run_status())
        if p == "/api/sessions":
            return self._json(_db_sessions())
        m = re.match(r"^/api/insights/([^/]+)$", p)
        if m and NAME_RE.match(m.group(1)):
            try:
                import insights
                return self._json(insights.build(m.group(1)))
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        if p == "/api/label/status":
            return self._json(_label_status())
        if p == "/api/process/status":
            return self._json(_proc_status())
        if p == "/api/contact_check":
            if RUN["proc"] is not None and RUN["proc"].poll() is None:
                return self._json({"ok": False, "error": "a recording is running"})
            try:
                import contact_check
                return self._json(contact_check.check(8.0))
            except Exception as e:
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"})
        m = re.match(r"^/api/profile/([^/]+)$", p)
        if m and NAME_RE.match(m.group(1)):
            return self._json(SC.load_profile(m.group(1)) or {"ad": m.group(1), "brand": "", "recall_options": []})
        if p == "/api/eeg_check":
            return self._json(_eeg_check())
        m = re.match(r"^/api/scenes/([^/]+)$", p)
        if m and NAME_RE.match(m.group(1)):
            ad = m.group(1)
            sp = os.path.join(SCENES_DIR, ad + ".scenes.json")
            cp = os.path.join(SCENES_DIR, ad + ".cuts.json")
            if not os.path.exists(sp):
                return self._json({"error": f"{ad} has no scenes yet -- run pipeline/label_scenes.py {ad}"}, 404)
            doc = json.load(open(sp))
            doc.pop("model_log", None)  # big and not needed by the editor; kept on disk
            cuts = json.load(open(cp))["candidates"] if os.path.exists(cp) else []
            ap = os.path.join(SCENES_DIR, ad + ".audio.json")
            audio = json.load(open(ap, encoding="utf-8")) if os.path.exists(ap) else None
            return self._json({"doc": doc, "cuts": cuts, "audio": audio,
                               "roles": LS.STORY_ROLES, "brand_kinds": LS.BRAND_KINDS})
        m = re.match(r"^/media/([^/]+)$", p)
        if m:
            f = os.path.basename(m.group(1))
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                return self._file(os.path.join(ADS_DIR, f))
        return self._json({"error": "not found"}, 404)

    def _post_question(self):
        b = self._body()
        try:
            q = SC.add_question(b.get("text", ""), b.get("type", "scale9"), b.get("anchors", ""))
        except ValueError as e:
            return self._json({"ok": False, "error": str(e)}, 400)
        return self._json({"ok": True, "question": q, "bank": SC.load_bank()})

    def _post_session(self):
        b = self._body()
        cfg, start, overwrite = b.get("config", {}), bool(b.get("start")), bool(b.get("overwrite"))
        probs = SC.validate(cfg)
        pid = cfg.get("participant_id", "")
        if not probs and start and SC.session_exists(pid) and not overwrite:
            probs.append(f"session '{pid}' already has a recording -- choose a new id or tick 'replace'")
        if probs:
            return self._json({"ok": False, "problems": probs}, 400)
        with RUN_LOCK:
            if start and RUN["proc"] is not None and RUN["proc"].poll() is None:
                return self._json({"ok": False, "problems": [f"a recording ({RUN['pid']}) is still running"]}, 409)
        path = SC.save_config(cfg)
        if not start:
            return self._json({"ok": True, "config": os.path.relpath(path, PROJ)})
        cmd = [sys.executable, TASK, "--id", pid, "--config", path] + (["--overwrite"] if overwrite else []) \
            + (["--auto"] if b.get("test_auto") else [])          # test hook, not exposed in the UI
        log = os.path.join(SC.SESS_DIR, f"{pid}.log")
        with RUN_LOCK:
            fh = open(log, "w", encoding="utf-8")
            RUN.update(proc=subprocess.Popen(cmd, cwd=PROJ, stdout=fh, stderr=subprocess.STDOUT,
                                             env={**os.environ, "PYTHONUNBUFFERED": "1"}),
                       pid=pid, log=log)
        return self._json({"ok": True, "config": os.path.relpath(path, PROJ), "started": True,
                           "command": " ".join(f'"{c}"' if " " in c else c for c in cmd)})

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/api/questions":
            return self._post_question()
        if p == "/api/session":
            return self._post_session()
        if p == "/api/process/stop":
            return self._json({"ok": True, "stopped": _stop(PROC)})
        if p == "/api/label/stop":
            return self._json({"ok": True, "stopped": _stop(LABEL)})
        m = re.match(r"^/api/viewer/([^/]+)$", p)
        if m and NAME_RE.match(m.group(1)):
            b = self._body()
            vw = _viewer_meta()
            cur = vw.get(m.group(1), {})
            if "excluded" in b:
                cur["excluded"] = bool(b["excluded"])
            if "note" in b:
                cur["note"] = str(b["note"])[:300]
            vw[m.group(1)] = cur
            with open(os.path.join(SC.LAB, "viewers.json"), "w", encoding="utf-8") as fh:
                json.dump(vw, fh, indent=1, ensure_ascii=False)
            return self._json({"ok": True, "viewer": cur})
        m = re.match(r"^/api/profile/([^/]+)$", p)
        if m and NAME_RE.match(m.group(1)):
            b = self._body()
            prof = SC.load_profile(m.group(1)) or {"ad": m.group(1)}
            brand = str(b.get("brand", "")).strip()[:60]
            opts = [str(o).strip()[:40] for o in (b.get("recall_options") or []) if str(o).strip()][:6]
            if brand and brand not in opts:
                opts.insert(0, brand)
            prof.update(brand=brand, recall_options=opts)
            for k in ("title", "company", "goal"):
                if k in b:
                    prof[k] = str(b[k])[:300]
            os.makedirs(SC.PROFILES_DIR, exist_ok=True)
            with open(os.path.join(SC.PROFILES_DIR, m.group(1) + ".json"), "w", encoding="utf-8") as fh:
                json.dump(prof, fh, indent=1, ensure_ascii=False)
            return self._json({"ok": True, "profile": prof})
        if p == "/api/ads/upload":
            from urllib.parse import parse_qs
            orig = os.path.basename((parse_qs(urlparse(self.path).query).get("name") or ["video.mp4"])[0])
            ext = os.path.splitext(orig)[1].lower()
            n = int(self.headers.get("Content-Length") or 0)
            if ext not in (".mp4", ".mov", ".m4v", ".webm") or not 0 < n <= 300 * 1024 * 1024:
                return self._json({"ok": False, "error": "need an .mp4 / .mov / .webm video under 300 MB"}, 400)
            nums = [int(x.group(1)) for x in (re.match(r"^ad(\d+)\.", f) for f in os.listdir(ADS_DIR)) if x]
            name = f"ad{(max(nums) + 1 if nums else 1):02d}{ext}"
            left, dst = n, os.path.join(ADS_DIR, name)
            with open(dst, "wb") as fh:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    fh.write(chunk)
                    left -= len(chunk)
            with open(os.path.join(ADS_DIR, "ad_name_map.csv"), "a", encoding="utf-8") as fh:
                fh.write(f'"{name}","{orig.replace(chr(34), "")}"\n')
            return self._json({"ok": True, "ad": os.path.splitext(name)[0], "file": name})
        m = re.match(r"^/api/process/([^/]+)$", p)
        if m:
            pid = m.group(1)
            if not NAME_RE.match(pid) or not os.path.exists(os.path.join(SC.LAB, f"{pid}_raw.fif")):
                return self._json({"ok": False, "error": f"no recording for '{pid}'"}, 404)
            if RUN["proc"] is not None and RUN["proc"].poll() is None:
                return self._json({"ok": False, "error": "a recording is still running -- wait until it finishes"}, 409)
            if PROC["proc"] is not None and PROC["proc"].poll() is None:
                return self._json({"ok": False, "error": f"already analysing {PROC['pid']}"}, 409)
            os.makedirs(SC.SESS_DIR, exist_ok=True)
            log = os.path.join(SC.SESS_DIR, f"{pid}.analysis.log")
            fh = open(log, "w", encoding="utf-8")
            PROC.update(proc=subprocess.Popen([sys.executable, SCENE_VALUES, "--update", pid], cwd=PROJ, stdout=fh,
                                              stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"}),
                        pid=pid, log=log)
            return self._json({"ok": True, "started": pid})
        if p == "/api/label":
            b = self._body()
            allads = {a["ad"]: a["scenes_status"] for a in _ads()}
            todo = [a for a, st in allads.items() if st is None]
            over = bool(b.get("overwrite"))
            ads = [a for a in (b.get("ads") or todo) if a in allads and (allads[a] is None or over)]
            if over and any(allads[a] == "reviewed" for a in ads) and not b.get("confirm_reviewed"):
                return self._json({"ok": False, "error": "this ad was reviewed and approved -- re-labelling would discard that"}, 409)
            lang = str(b.get("language", "en"))
            if not ads:
                return self._json({"ok": False, "error": "nothing to label"}, 400)
            if not re.fullmatch(r"[a-z]{0,3}", lang):
                return self._json({"ok": False, "error": "bad language code"}, 400)
            if LABEL["proc"] is not None and LABEL["proc"].poll() is None:
                return self._json({"ok": False, "error": "labelling is already running"}, 409)
            if RUN["proc"] is not None and RUN["proc"].poll() is None:
                return self._json({"ok": False, "error": "a recording is running -- label after it finishes"}, 409)
            log = os.path.join(SCENES_DIR, "label.log")
            fh = open(log, "w", encoding="utf-8")
            LABEL.update(proc=subprocess.Popen([sys.executable, LABELLER, *ads, "--model", os.environ.get("OPENAI_MODEL") or "gpt-5.5",
                                                "--language", lang] + (["--overwrite"] if over else []), cwd=PROJ, stdout=fh, stderr=subprocess.STDOUT,
                                               env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}),
                         ads=ads, log=log)
            return self._json({"ok": True, "started": ads})
        if p == "/api/copilot":
            b = self._body()
            q = (b.get("question") or "").strip()
            if not q:
                return self._json({"error": "empty question"}, 400)
            try:
                import copilot
                return self._json(copilot.answer(q, history=b.get("history"), context=b.get("context")))
            except Exception as e:
                msg = str(e)
                if "insufficient_quota" in msg or "429" in msg:
                    msg = "OpenAI says the API key has no credit left (insufficient quota)."
                elif "Connection" in type(e).__name__ or "connect" in msg.lower():
                    msg = "Cannot reach OpenAI -- check the internet connection (phone hotspot?)."
                return self._json({"error": msg}, 502)
        m = re.match(r"^/api/scenes/([^/]+)(/approve)?$", p)
        if not (m and NAME_RE.match(m.group(1))):
            return self._json({"error": "not found"}, 404)
        ad, approving = m.group(1), bool(m.group(2))
        sp = os.path.join(SCENES_DIR, ad + ".scenes.json")
        if not os.path.exists(sp):
            return self._json({"error": "no scenes file"}, 404)
        disk = json.load(open(sp))
        edit = self._body()
        # Only the editable parts come from the browser; everything else (model
        # log, fps, frame count, provenance) is kept from disk.
        disk["scenes"] = edit["scenes"]
        disk["brand_events"] = edit["brand_events"]
        disk.setdefault("edit_history", []).append({"saved": LS.dt.datetime.now().isoformat(timespec="seconds"),
                                                   "approve": approving})
        LS.normalise(disk)
        if approving:
            probs = LS.approve_doc(disk)
            if probs:
                return self._json({"ok": False, "problems": probs}, 400)
        else:
            if disk.get("status") == "reviewed":
                disk["status"] = "draft"   # an edit after approval needs a new approval
            disk.pop("reviewed", None) if disk["status"] == "draft" else None
        problems = LS.check(disk)
        tmp = sp + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(disk, fh, indent=1)
        os.replace(tmp, sp)
        return self._json({"ok": True, "status": disk["status"], "problems": problems})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/setup"
    print(f"Neuromarketing app running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
