"""
Per-scene brain values + the project database (data/lab_session/neuro.db).

For every recorded session and every ad it watched, each labelled scene gets
the four measures computed from EXACTLY that scene's EEG samples:

  attention   posterior alpha suppression, % vs rest   (+ = alpha dropped = more visual attention)
  alertness   beta/alpha engagement index, % vs rest   (+ = more alert/engaged)
  effort      frontal-midline theta, dB vs rest        (+ = more mental processing)
  wanting     frontal alpha asymmetry, change vs rest  (+ = more approach; noisy for one person)

How a scene finds its EEG
  scene = video frames [start_frame, end_frame)  (from <ad>.scenes.json, labeller)
  frame -> EEG time:
    "frame_log"           <session>_framelog.csv: when each video frame was actually
                          on screen, on the EEG time axis (recordings from 2026-11-13 on)
    "two_marker_estimate" older recordings: frames spread evenly between the
                          AD_START and AD_END markers (can be off by ~0.1-0.3 s)
  + LAG_S (default 0.3 s) brain delay: the EEG window is shifted later, no samples dropped.

The same recipe for every scene:
  Welch spectrum, 1 s pieces, 50 % overlap, MEDIAN-averaged (one movement blip
  can't dominate a scene), same channels as extract_neuromarketing_features.py.

Normal range ("is this more than ordinary wobble?"):
  for a scene of length D, the same measure is computed on every D-second chunk
  of the person's eyes-open rest (0.5 s steps); the central 90 % of those chunk
  values is the normal range. A scene outside it is "above rest" / "below rest".
  Caveat (stored with the results): rest = fixation cross, ads = moving video, so
  attention is expected to sit above rest for most scenes; compare scenes WITHIN
  an ad with `vs_ad_median`.

Controls stored per scene: fraction of noisy 0.5 s blocks (>150 uV peak-to-peak
on the channels used), screen brightness, loudness, sudden loud sounds, number
of camera cuts, words spoken -- things that move EEG for reasons other than
the story.

Single-viewer reliability (attention, alertness, effort; asymmetry is experimental
and never tested). All within ONE person -- no group needed:
  brand-moment test   timeline windows with the brand on screen (logo / product /
                      brand text / tagline from the labeller) vs the rest of the ad.
                      Timeline linearly detrended first (an end-card logo must not be
                      confused with attention fading towards the end). Statistic =
                      median(brand) - median(other); p = circular shifts of the brand
                      on/off pattern against the EEG (shifts >= 4 s, two-sided).
                      Attention is the pre-specified primary test; alertness/effort are
                      secondary and uncorrected -> reported as such.
  repeat viewings     sessions recorded with viewings = 2 show each ad twice
                      (markers 'AD_START:ad06.mp4#2'). Viewing 1 fills the normal
                      tables; viewing 2 goes to *_repeat tables. reliability = the
                      viewer's own timeline correlated across viewings (circular-shift
                      p, one-sided). A scene is 'confirmed' when the measure passed the
                      viewer's quality check, the scene is clean in both viewings, is
                      outside the rest range in viewing 1 and sits on the same side of
                      the ad median both times. Brand test 'confirmed for this viewer' =
                      p < 0.05 in viewing 1 and same direction in viewing 2.
  quality gate        attention counts only if the posterior electrodes used show the
                      eyes-closed alpha jump (median ratio >= 1.3).

Usage:
  python pipeline/scene_values.py                 # every session with a recording
  python pipeline/scene_values.py --session test1 [--lag 0.3]
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import sqlite3
import sys

import cv2
import mne
import numpy as np
from mne.time_frequency import psd_array_welch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "lab_task_psychopy"))
import extract_neuromarketing_features as F  # noqa: E402
from preprocess_mne import load_recording, quality_check, clean_raw, fix_reference  # noqa: E402

LAB = os.path.join(PROJ, "data", "lab_session")
ADS_DIR = os.path.join(LAB, "ads")
SCENES_DIR = os.path.join(LAB, "scenes")
DB_PATH = os.path.join(LAB, "neuro.db")
LAG_S = 0.3
RANGE_PCT = (5, 95)
CHUNK_STEP_S = 0.5
NOISE_UV = 150.0
NOISY_MAX = 0.3            # scenes with more noisy blocks than this get no verdict
BERGER_MIN = 1.3           # attention electrodes must show at least this eyes-closed alpha jump
TIMELINE_WIN_S, TIMELINE_STEP_S = 2.0, 0.5
SKIP_SESSIONS = {"channel_calibration"}

MEASURES = {
    #  key          label                         unit   higher means
    "attention": ("Visual attention (posterior alpha suppression)", "%", "more visual attention"),
    "alertness": ("Alertness (beta/alpha engagement index)", "%", "more alert / engaged"),
    "effort":    ("Mental effort (frontal-midline theta)", "dB", "more mental processing"),
    # internal key stays "wanting" (older tables/tools use it); shown as Approach, EXPERIMENTAL:
    # least reliable EEG ad metric (van Diepen 2025), null in Moya 2020 -> never used for conclusions.
    "wanting":   ("Approach (frontal alpha asymmetry) -- experimental", "", "more approach (experimental; not used for conclusions)"),
}
# Measures that the reliability tests run on (asymmetry is experimental -> not tested).
TEST_MEASURES = ("attention", "alertness", "effort")
VISUAL_BRAND = ("logo", "product", "brand_text", "tagline")   # "brand on screen" for the brand-moment test
MIN_SHIFT_S = 4.0          # circular-shift tests never use shifts shorter than this (neighbouring windows overlap)
P_LIMIT = 0.05


# ----------------------------------------------------------------------------- spectra
def _bp(seg, sf, band):
    n = min(seg.shape[-1], int(sf))
    if n < int(0.5 * sf):
        return np.full(seg.shape[0], np.nan)
    psd, _ = psd_array_welch(seg, sfreq=sf, fmin=band[0], fmax=band[1], n_per_seg=n,
                             n_overlap=n // 2, average="median", verbose=False)
    return psd.mean(axis=-1)


class Measurer:
    """Raw log-domain values of the four measures for any span of the cleaned recording."""

    def __init__(self, raw):
        self.sf = raw.info["sfreq"]
        self.data = raw.get_data()
        names, bad = raw.ch_names, list(raw.info["bads"])
        try:
            l, r = F.choose_faa_pair(names, bad)
            self.faa = ([names.index(l)], [names.index(r)])
        except RuntimeError:
            self.faa = None
        self.mid = [names.index(c) for c in F.choose_theta_channels(names, bad)]
        self.post = [names.index(c) for c in F.choose_posterior_channels(names, bad)]
        self.eng = [names.index(c) for c in (F._find(names, n, bad) for n in F.ENGAGEMENT_CHANNELS) if c]
        self.used = sorted(set(self.mid + self.post + self.eng + (self.faa[0] + self.faa[1] if self.faa else [])))
        self.channels = {
            "attention": [names[i] for i in self.post], "alertness": [names[i] for i in self.eng],
            "effort": [names[i] for i in self.mid],
            "wanting": [names[self.faa[0][0]], names[self.faa[1][0]]] if self.faa else []}

    def span(self, t0, t1):
        a, b = max(0, int(round(t0 * self.sf))), min(self.data.shape[1], int(round(t1 * self.sf)))
        return self.data[:, a:b]

    def logs(self, t0, t1):
        seg = self.span(t0, t1)
        out = {}
        out["attention"] = float(np.log10(_bp(seg[self.post], self.sf, F.ALPHA_BAND).mean())) if self.post else np.nan
        if self.eng:
            out["alertness"] = float(np.log10(_bp(seg[self.eng], self.sf, F.BETA_BAND).mean()
                                              / _bp(seg[self.eng], self.sf, F.ALPHA_BAND).mean()))
        else:
            out["alertness"] = np.nan
        out["effort"] = float(np.log10(_bp(seg[self.mid], self.sf, F.THETA_BAND).mean()))
        if self.faa:
            out["wanting"] = float(np.log(_bp(seg[self.faa[1]], self.sf, F.ALPHA_BAND).mean())
                                   - np.log(_bp(seg[self.faa[0]], self.sf, F.ALPHA_BAND).mean()))
        else:
            out["wanting"] = np.nan
        return out

    def noisy_fraction(self, t0, t1):
        seg = self.span(t0, t1)[self.used] * 1e6
        blk = int(0.5 * self.sf)
        n = seg.shape[1] // blk
        if n == 0:
            return None
        p2p = np.array([np.ptp(seg[:, k * blk:(k + 1) * blk], axis=1).max() for k in range(n)])
        return float(np.mean(p2p > NOISE_UV))


def change(measure, v, ref):
    """log-domain value -> the reported change vs rest (same formulas as the existing pipeline)."""
    if v is None or ref is None or np.isnan(v) or np.isnan(ref):
        return None
    d = v - ref
    if measure == "attention":
        return 100 * (1 - 10 ** d)
    if measure == "alertness":
        return 100 * (10 ** d - 1)
    if measure == "effort":
        return 10 * d
    return d                                   # wanting: FAA is already a log ratio


class RestRange:
    """Normal range of each measure for chunks of a given length, from eyes-open rest."""

    def __init__(self, meas, rest):
        self.m, self.rest = meas, rest
        self.ref = meas.logs(*rest)
        self.cache = {}

    def get(self, dur):
        key = round(dur * 4) / 4                # 0.25 s resolution
        if key in self.cache:
            return self.cache[key]
        r0, r1 = self.rest
        starts = np.arange(r0, r1 - key + 1e-9, CHUNK_STEP_S)
        vals = {m: [] for m in MEASURES}
        for s in starts:
            lg = self.m.logs(s, s + key)
            for m in MEASURES:
                c = change(m, lg[m], self.ref[m])
                if c is not None:
                    vals[m].append(c)
        out = {}
        for m, v in vals.items():
            if len(v) >= 5:
                lo, hi = np.percentile(v, RANGE_PCT)
                out[m] = (float(lo), float(hi), len(v))
            else:
                out[m] = (None, None, len(v))
        self.cache[key] = out
        return out


# ----------------------------------------------------------------------------- timing
def frame_mapper(session, ad_file, ad_start, ad_end, n_frames):
    """Return (f -> recording seconds, method). f may be n_frames (= end of the last frame)."""
    fl = os.path.join(LAB, f"{session}_framelog.csv")
    if os.path.exists(fl):
        vf, rt = [], []
        with open(fl) as fh:
            for row in csv.DictReader(fh):
                if row["ad"] == ad_file:
                    vf.append(int(row["video_frame"]))
                    rt.append(float(row["rec_time_s"]))
        if vf:
            vf, rt = np.array(vf), np.array(rt)
            first_idx = {}
            for i, f in enumerate(vf):
                first_idx.setdefault(f, i)
            shown = np.array(sorted(first_idx))
            shown_t = np.array([rt[first_idx[f]] for f in shown])

            def fmap(f):
                if f >= n_frames:
                    return float(ad_end)
                k = np.searchsorted(shown, f)  # first frame shown at or after f (skipped frames -> next shown)
                return float(shown_t[min(k, len(shown) - 1)])
            return fmap, "frame_log"
    span = ad_end - ad_start
    return (lambda f: float(ad_start + span * min(f, n_frames) / n_frames)), "two_marker_estimate"


def inverse_mapper(fmap, n_frames, fps):
    """recording seconds -> video seconds (for the timeline)."""
    fr = np.arange(0, n_frames + 1)
    rec = np.array([fmap(f) for f in fr])
    return lambda t: float(np.interp(t, rec, fr / fps))


def scene_brightness(ad_file, scenes, fps):
    cap = cv2.VideoCapture(os.path.join(ADS_DIR, ad_file))
    step = max(1, int(round(fps / 4)))
    out = []
    for s in scenes:
        vals = []
        for f in range(s["start_frame"], s["end_frame"], step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if ok:
                vals.append(float(cv2.cvtColor(cv2.resize(fr, (64, 36)), cv2.COLOR_BGR2HSV)[..., 2].mean()))
        out.append(round(float(np.mean(vals)), 1) if vals else None)
    cap.release()
    return out


# ----------------------------------------------------------------------------- database
SCHEMA = """
CREATE TABLE ads(ad TEXT PRIMARY KEY, file TEXT, brand TEXT, product TEXT, duration_s REAL, fps REAL,
  n_frames INTEGER, summary TEXT, audio_summary TEXT, scenes_status TEXT);
CREATE TABLE scenes(ad TEXT, idx INTEGER, start_frame INTEGER, end_frame INTEGER, start_s REAL, end_s REAL,
  dur_s REAL, label TEXT, description TEXT, heard TEXT, story_role TEXT, boundary_method TEXT,
  human_checked INTEGER, flags TEXT, loudness_db_mean REAL, loud_jumps TEXT, n_words INTEGER,
  n_cuts INTEGER, brightness REAL, PRIMARY KEY(ad, idx));
CREATE TABLE brand_events(ad TEXT, kind TEXT, start_s REAL, end_s REAL, what TEXT, human_checked INTEGER);
CREATE TABLE transcript(ad TEXT, start_s REAL, end_s REAL, text TEXT, kind TEXT, reliable INTEGER);
CREATE TABLE questions(id TEXT PRIMARY KEY, text TEXT, type TEXT, anchors TEXT);
CREATE TABLE measures(measure TEXT PRIMARY KEY, label TEXT, unit TEXT, higher_means TEXT);
CREATE TABLE sessions(session TEXT PRIMARY KEY, recorded TEXT, has_baseline INTEGER, n_ads INTEGER,
  bad_channels TEXT, channels_used TEXT, cap_check_ratio REAL, lag_s REAL, range_pct TEXT, note TEXT,
  reference_fix TEXT, posterior_check TEXT);
CREATE TABLE ratings(session TEXT, ad TEXT, order_shown INTEGER, question TEXT, answer REAL, rt_s REAL);
CREATE TABLE scene_values(session TEXT, ad TEXT, scene_idx INTEGER, measure TEXT, value REAL,
  rest_lo REAL, rest_hi REAL, n_rest_chunks INTEGER, verdict TEXT, vs_ad_median REAL,
  eeg_start_s REAL, eeg_end_s REAL, noisy_frac REAL, timing TEXT);
CREATE TABLE ad_values(session TEXT, ad TEXT, measure TEXT, value REAL, rest_lo REAL, rest_hi REAL,
  verdict TEXT, timing TEXT, on_screen_s REAL, noisy_frac REAL);
CREATE TABLE timeline(session TEXT, ad TEXT, t_s REAL, measure TEXT, value REAL, noisy_frac REAL);
CREATE TABLE scene_values_repeat(session TEXT, ad TEXT, viewing INTEGER, scene_idx INTEGER, measure TEXT, value REAL,
  rest_lo REAL, rest_hi REAL, n_rest_chunks INTEGER, verdict TEXT, vs_ad_median REAL,
  eeg_start_s REAL, eeg_end_s REAL, noisy_frac REAL, timing TEXT);
CREATE TABLE timeline_repeat(session TEXT, ad TEXT, viewing INTEGER, t_s REAL, measure TEXT, value REAL, noisy_frac REAL);
CREATE TABLE reliability(session TEXT, ad TEXT, measure TEXT, n_windows INTEGER, r REAL, p REAL,
  scenes_same_side INTEGER, scenes_both_clean INTEGER, measure_ok INTEGER, verdict TEXT, note TEXT);
CREATE TABLE brand_test(session TEXT, ad TEXT, viewing INTEGER, measure TEXT, n_brand INTEGER, n_other INTEGER,
  brand_median REAL, other_median REAL, diff REAL, p REAL, n_shifts INTEGER, brand_late INTEGER, status TEXT, note TEXT);
CREATE TABLE scene_replication(session TEXT, ad TEXT, scene_idx INTEGER, measure TEXT, v1_vs_median REAL,
  v2_vs_median REAL, v1_verdict TEXT, v2_verdict TEXT, status TEXT);
CREATE TABLE answers_text(session TEXT, ad TEXT, question TEXT, text TEXT);
CREATE TABLE copilot_log(ts TEXT, question TEXT, tool_calls TEXT, answer TEXT, flags TEXT);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
"""


def verdict(v, lo, hi, noisy=None):
    if noisy is not None and noisy > NOISY_MAX:
        return "too noisy to judge"
    if v is None or lo is None:
        return "no range"
    return "above rest" if v > hi else ("below rest" if v < lo else "within rest range")


def load_static(db):
    sys.path.insert(0, os.path.join(PROJ, "lab_task_psychopy"))
    import session_config as SC
    for q in SC.load_bank():
        db.execute("INSERT OR REPLACE INTO questions VALUES(?,?,?,?)", (q["id"], q["text"], q["type"], q["anchors"]))
    for m, (label, unit, hm) in MEASURES.items():
        db.execute("INSERT INTO measures VALUES(?,?,?,?)", (m, label, unit, hm))
    scenes_by_ad = {}
    for p in sorted(glob.glob(os.path.join(SCENES_DIR, "*.scenes.json"))):
        d = json.load(open(p, encoding="utf-8"))
        ad = os.path.splitext(d["video"])[0]
        cuts_p = os.path.join(SCENES_DIR, ad + ".cuts.json")
        cuts = [c["frame"] for c in json.load(open(cuts_p))["candidates"] if c["kind"] == "cut"] if os.path.exists(cuts_p) else []
        st = d.get("story", {})
        db.execute("INSERT INTO ads VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (ad, d["video"], st.get("brand"), st.get("product"), d["duration_s"], d["fps"], d["n_frames"],
                    st.get("summary"), st.get("audio_summary"), d.get("status")))
        bright = scene_brightness(d["video"], d["scenes"], d["fps"])
        for s, br in zip(d["scenes"], bright):
            am = s.get("audio_measures") or {}
            db.execute("INSERT INTO scenes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (ad, s["idx"], s["start_frame"], s["end_frame"], s["start_s"], s["end_s"], s["dur_s"],
                        s["label"], s["description"], s.get("audio", ""), s["story_role"], s["boundary_method"],
                        int(bool(s.get("human_checked"))), json.dumps(s.get("flags", [])),
                        am.get("loudness_db_mean"), json.dumps(am.get("loud_jumps", [])), am.get("n_words"),
                        sum(1 for c in cuts if s["start_frame"] < c < s["end_frame"]), br))
        for b in d.get("brand_events", []):
            db.execute("INSERT INTO brand_events VALUES(?,?,?,?,?,?)",
                       (ad, b["kind"], b["start_s"], b["end_s"], b["what"], int(bool(b.get("human_checked")))))
        ap = os.path.join(SCENES_DIR, ad + ".audio.json")
        if os.path.exists(ap):
            a = json.load(open(ap, encoding="utf-8"))
            for sg in ((a.get("transcript") or {}).get("segments") or []):
                db.execute("INSERT INTO transcript VALUES(?,?,?,?,?,?)",
                           (ad, sg["start"], sg["end"], sg["text"], sg.get("kind", "speech"), int(sg["reliable"])))
        scenes_by_ad[ad] = d
    return scenes_by_ad


def load_ratings(db, session):
    p = os.path.join(LAB, f"{session}_ratings.csv")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            ad = os.path.splitext(row["ad"])[0]
            if str(row.get("viewing") or "1") != "1":
                continue                                  # repeat viewings have no answers
            for k, v in row.items():
                if k in ("order", "ad", "duration_s", "viewing") or k.endswith("_rt_s") or v in ("", None):
                    continue
                if k.endswith("_choice"):                 # recall answer text (the 1/0 score is its own column)
                    db.execute("INSERT INTO answers_text VALUES(?,?,?,?)", (session, ad, k[:-len("_choice")], v))
                    continue
                rt = row.get(f"{k}_rt_s")
                db.execute("INSERT INTO ratings VALUES(?,?,?,?,?,?)",
                           (session, ad, int(row["order"]), k, float(v), float(rt) if rt else None))


# ----------------------------------------------------------------------------- one session
def process_session(db, session, scenes_by_ad, lag):
    path = os.path.join(LAB, f"{session}_raw.fif")
    mne.set_log_level("ERROR")
    raw = load_recording(path)
    raw, ref_info = fix_reference(raw, F.KNOWN_BAD)       # only acts if REF noise is shared by all channels
    raw.info["bads"] = sorted(set(quality_check(raw)) | set(F.KNOWN_BAD)
                              | set(ref_info.get("floating_before_fix", [])))
    ann = raw.annotations
    segs = F.baseline_segments(ann)
    ads_seen = F.ad_segments(ann)
    load_ratings(db, session)
    if "EO" not in segs:
        db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (session, None, 0, len(ads_seen), json.dumps(raw.info["bads"]), None, None, lag,
                    json.dumps(RANGE_PCT), "no eyes-open baseline -- values relative to rest not computable",
                    json.dumps(ref_info), None))
        print(f"{session}: no eyes-open baseline -> skipped (ratings stored)")
        return
    # Attention electrodes must show real brain alpha: the eyes-closed/eyes-open alpha
    # jump (Berger effect, >= 1.3x). On p01 Pz passed the amplitude check (146 uV) but
    # showed 0.21x -- no brain alpha -- while P4 showed 3.0x.
    post_check = {}
    if "EC" in segs:
        sfq = raw.info["sfreq"]
        fr = raw.copy().filter(1, 40, verbose=False)
        cand = [c for c in F.POSTERIOR_PRIMARY + F.POSTERIOR_FALLBACK if c in fr.ch_names and c not in raw.info["bads"]]
        if cand:
            def _alpha(seg):
                a, b = int(seg[0] * sfq), int(seg[1] * sfq)
                return _bp(fr.get_data(picks=cand)[:, a:b], sfq, F.ALPHA_BAND)
            ratios = _alpha(segs["EC"]) / _alpha(segs["EO"])
            post_check = {c: round(float(r), 2) for c, r in zip(cand, ratios)}
            no_alpha = [c for c, r in post_check.items() if r < BERGER_MIN]
            if len(no_alpha) < len(cand):             # keep at least one; else leave as is (noted in DB)
                raw.info["bads"] = sorted(set(raw.info["bads"]) | set(no_alpha))
            print(f"{session}: posterior eyes-closed alpha ratios {post_check} -> attention uses "
                  f"{[c for c in cand if c not in raw.info['bads']] or cand}")
    cleaned = clean_raw(raw)
    tl_off = F.sliding_window_features(cleaned)
    _, ref_off = F.baseline_reference(tl_off, segs)
    meas = Measurer(cleaned)
    rest = RestRange(meas, segs["EO"])
    # Cap check = eyes-closed/eyes-open alpha on the attention electrodes actually used,
    # median-averaged spectrum (the older mean-based check read 0.17 on p01 because a few
    # movement bursts in the eyes-open block swamped it, while the robust ratio was 3-5x).
    used_post = [c for c in post_check if c not in raw.info["bads"]]
    cap = (used_post, float(np.median([post_check[c] for c in used_post]))) if used_post else None
    db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
               (session, raw.info.get("meas_date").isoformat() if raw.info.get("meas_date") else None, 1,
                len(ads_seen), json.dumps(cleaned.info["bads"]), json.dumps(meas.channels),
                round(cap[1], 3) if cap else None, lag, json.dumps(RANGE_PCT),
                "rest = eyes-open fixation; ads are moving video, so attention tends to sit above rest -- "
                "compare scenes within an ad with vs_ad_median", json.dumps(ref_info), json.dumps(post_check)))
    per_ad = {}      # ad -> {viewing: {"tl": [...], "scenes": rows}} for the reliability tests
    for ad_file, a_start, a_end in ads_seen:
        base_file, _, vtag = ad_file.partition("#")          # repeat viewings are 'ad06.mp4#2'
        viewing = int(vtag) if vtag.isdigit() else 1
        ad = os.path.splitext(base_file)[0]
        d = scenes_by_ad.get(ad)
        if d is None:
            print(f"{session}/{ad}: no scenes file -- run label_scenes.py {ad}")
            continue
        n, fps = d["n_frames"], d["fps"]
        fmap, timing = frame_mapper(session, ad_file, a_start, a_end, n)
        if viewing > 1:
            rows = scene_rows(session, ad, d, fmap, meas, rest, lag, timing)
            db.executemany("INSERT INTO scene_values_repeat VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           [r[:2] + [viewing] + r[2:] for r in rows])
            tl = ad_timeline(meas, rest.ref, fmap, n, fps, lag)
            db.executemany("INSERT INTO timeline_repeat VALUES(?,?,?,?,?,?,?)",
                           [(session, ad, viewing, t, m, v[m], nf) for t, v, nf in tl for m in MEASURES])
            per_ad.setdefault(ad, {})[viewing] = {"tl": tl, "scenes": rows}
            print(f"{session}/{ad} (viewing {viewing}): {len(d['scenes'])} scenes, timing = {timing}")
            continue
        # whole ad: the OFFICIAL numbers -- exactly the report / replay method
        # (3 s windows, median over windows inside AD_START..AD_END vs the
        # eyes-open baseline), so copilot, report and slides always agree.
        # A 60 s ad has no rest range: the baseline itself is only ~60 s long.
        off = F.relative_to_baseline(F.summary_features(tl_off, a_start, a_end), ref_off)
        key = {"attention": "alpha_suppression_pct", "alertness": "engagement_change_pct",
               "effort": "theta_change_db", "wanting": "faa_change"}
        ad_noisy = meas.noisy_fraction(fmap(0) + lag, fmap(n) + lag)
        for m in MEASURES:
            v = off.get(key[m])
            vd = ("too noisy to judge" if ad_noisy is not None and ad_noisy > NOISY_MAX
                  else "no range (ad longer than half the rest block)")
            db.execute("INSERT INTO ad_values VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (session, ad, m, v, None, None, vd,
                        "report method: 3 s windows, median", round(a_end - a_start, 3), ad_noisy))
        # scenes
        rows = scene_rows(session, ad, d, fmap, meas, rest, lag, timing)
        db.executemany("INSERT INTO scene_values VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        # timeline (video time), 2 s windows every 0.5 s, centred
        tl = ad_timeline(meas, rest.ref, fmap, n, fps, lag)
        db.executemany("INSERT INTO timeline VALUES(?,?,?,?,?,?)",
                       [(session, ad, t, m, v[m], nf) for t, v, nf in tl for m in MEASURES])
        per_ad.setdefault(ad, {})[1] = {"tl": tl, "scenes": rows}
        print(f"{session}/{ad}: {len(d['scenes'])} scenes, timing = {timing}, on screen {a_end - a_start:.2f} s "
              f"(file {n / fps:.2f} s)")

    # ---- single-viewer reliability: brand-moment test, repeat reliability, scene confirmation
    ok = measure_ok(meas, cap)
    for ad, views in per_ad.items():
        d = scenes_by_ad[ad]
        brand_iv = [(b["start_s"], b["end_s"]) for b in d.get("brand_events", []) if b["kind"] in VISUAL_BRAND]
        for m in TEST_MEASURES:
            res = {v: brand_test(views[v]["tl"], m, brand_iv, d["duration_s"]) for v in sorted(views)}
            st = brand_status(res, ok[m])
            for v, r in res.items():
                db.execute("INSERT INTO brand_test VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (session, ad, v, m, r["n_brand"], r["n_other"], r["brand_median"], r["other_median"],
                            r["diff"], r["p"], r["n_shifts"], r["brand_late"], st if v == 1 else None, r["note"]))
            if 1 in views and 2 in views:
                rel = repeat_reliability(views[1], views[2], m)
                verdict_r = ("measure failed this viewer's quality check" if not ok[m] else
                             "repeats" if rel["p"] is not None and rel["p"] < P_LIMIT and rel["r"] > 0 else
                             "does not repeat" if rel["r"] is not None else "not testable")
                db.execute("INSERT INTO reliability VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (session, ad, m, rel["n"], rel["r"], rel["p"], rel["same"], rel["both"], int(ok[m]),
                            verdict_r, "timeline correlation between viewing 1 and 2, circular-shift p (one-sided)"))
                db.executemany("INSERT INTO scene_replication VALUES(?,?,?,?,?,?,?,?,?)",
                               scene_replication(session, ad, views[1]["scenes"], views[2]["scenes"], m, ok[m]))
        print(f"{session}/{ad}: reliability -> " + ", ".join(
            f"{m}: brand {db.execute('SELECT status FROM brand_test WHERE session=? AND ad=? AND measure=? AND viewing=1', (session, ad, m)).fetchone()[0]}"
            for m in TEST_MEASURES if 1 in views))


# ----------------------------------------------------------------------------- shared pieces
def scene_rows(session, ad, d, fmap, meas, rest, lag, timing):
    rows = []
    for s in d["scenes"]:
        t0, t1 = fmap(s["start_frame"]) + lag, fmap(s["end_frame"]) + lag
        lg = meas.logs(t0, t1)
        rng = rest.get(t1 - t0)
        nf = meas.noisy_fraction(t0, t1)
        for m in MEASURES:
            v = change(m, lg[m], rest.ref[m])
            lo, hi, nch = rng[m]
            rows.append([session, ad, s["idx"], m, v, lo, hi, nch, verdict(v, lo, hi, nf), None,
                         round(t0, 3), round(t1, 3), nf, timing])
    for m in MEASURES:   # within-ad comparison: vs the median of this ad's CLEAN scenes
        vals = [r[4] for r in rows if r[3] == m and r[4] is not None and (r[12] is None or r[12] <= NOISY_MAX)]
        med = float(np.median(vals)) if vals else None
        for r in rows:
            if r[3] == m and r[4] is not None and med is not None and (r[12] is None or r[12] <= NOISY_MAX):
                r[9] = r[4] - med
    return rows


def ad_timeline(meas, rest_ref, fmap, n, fps, lag):
    """[(video_time_s, {measure: change vs rest}, noisy_frac)] -- 2 s windows every 0.5 s."""
    inv = inverse_mapper(fmap, n, fps)
    out = []
    t = fmap(0) + lag + TIMELINE_WIN_S / 2
    while t + TIMELINE_WIN_S / 2 <= fmap(n) + lag:
        a, b = t - TIMELINE_WIN_S / 2, t + TIMELINE_WIN_S / 2
        lg = meas.logs(a, b)
        out.append((round(inv(t - lag), 3), {m: change(m, lg[m], rest_ref[m]) for m in MEASURES},
                    meas.noisy_fraction(a, b)))
        t += TIMELINE_STEP_S
    return out


def measure_ok(meas, cap):
    """Per-viewer quality gate for each tested measure. Attention needs posterior
    electrodes that show real alpha (eyes-closed check >= BERGER_MIN)."""
    return {"attention": bool(meas.post) and cap is not None and cap[1] >= BERGER_MIN,
            "alertness": bool(meas.eng), "effort": bool(meas.mid)}


def _series(tl, m):
    t = np.array([x[0] for x in tl])
    v = np.array([np.nan if x[1][m] is None else x[1][m] for x in tl], dtype=float)
    good = np.array([(x[2] is None or x[2] <= NOISY_MAX) for x in tl]) & np.isfinite(v)
    return t, v, good


def _detrend(t, v, good):
    """Remove the straight-line trend over the clean windows, so a slow drift
    through the ad (e.g. attention fading towards the end, where logos usually are)
    can't masquerade as a brand effect."""
    if good.sum() < 4:
        return v
    k, c = np.polyfit(t[good], v[good], 1)
    return v - (k * t + c)


def brand_test(tl, m, brand_iv, duration):
    """Brand on screen vs the rest of the ad, within ONE viewer and ONE viewing.
    Statistic: median(brand windows) - median(other windows) of the detrended
    timeline. p-value: circular shifts of the brand on/off pattern against the
    EEG (keeps the EEG's own slow wobble and the brand's block structure), all
    shifts >= MIN_SHIFT_S, two-sided."""
    out = {"n_brand": 0, "n_other": 0, "brand_median": None, "other_median": None, "diff": None,
           "p": None, "n_shifts": 0, "brand_late": None, "note": ""}
    if not tl:
        out["note"] = "no timeline"
        return out
    t, v, good = _series(tl, m)
    lab = np.array([any(a <= x < b for a, b in brand_iv) for x in t])
    out["n_brand"], out["n_other"] = int((lab & good).sum()), int((~lab & good).sum())
    if not brand_iv:
        out["note"] = "no brand on screen in the labels"
        return out
    late = t[lab].min() >= 0.8 * duration if lab.any() else None
    out["brand_late"] = None if late is None else int(late)
    if out["n_brand"] < 3 or out["n_other"] < 3:
        out["note"] = "too few clean windows with / without the brand"
        return out
    vd = _detrend(t, v, good)
    out["brand_median"] = round(float(np.median(v[lab & good])), 2)
    out["other_median"] = round(float(np.median(v[~lab & good])), 2)

    def stat(lb):
        a, b = vd[lb & good], vd[~lb & good]
        return np.median(a) - np.median(b) if len(a) >= 3 and len(b) >= 3 else np.nan
    obs = stat(lab)
    k0 = int(round(MIN_SHIFT_S / TIMELINE_STEP_S))
    null = np.array([stat(np.roll(lab, k)) for k in range(k0, len(lab) - k0 + 1)])
    null = null[np.isfinite(null)]
    out["diff"] = round(float(obs), 2)
    out["n_shifts"] = int(len(null))
    if len(null) >= 10:
        out["p"] = round(float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null))), 4)
    else:
        out["note"] = "ad too short for the shift test"
    if late:
        out["note"] = (out["note"] + "; " if out["note"] else "") + \
            "brand appears only in the last 20 % of the ad -- detrended, but still partly confounded with the ending"
    return out


def brand_status(res, ok):
    """Combine viewings into one label for the brand-moment test."""
    if not ok:
        return "measure failed this viewer's quality check"
    r1 = res.get(1)
    if not r1 or r1["p"] is None:
        return "not testable"
    sig1 = r1["p"] < P_LIMIT
    r2 = res.get(2)
    if r2 is None or r2["diff"] is None:
        return "seen once (one viewing)" if sig1 else "no clear difference"
    same = np.sign(r1["diff"]) == np.sign(r2["diff"])
    if sig1 and same:
        return "confirmed for this viewer"
    if sig1 or (r2["p"] is not None and r2["p"] < P_LIMIT):
        return "seen once (did not repeat)"
    return "no clear difference"


def repeat_reliability(v1, v2, m):
    """Correlation of the viewer's own timeline across two viewings (aligned on video
    time); p from circular shifts of viewing 2 (one-sided: reliability means r > 0)."""
    t1, a, g1 = _series(v1["tl"], m)
    t2, b, g2 = _series(v2["tl"], m)
    out = {"n": 0, "r": None, "p": None, "same": 0, "both": 0}
    if g1.sum() < 10 or g2.sum() < 10:
        return out | _scene_side(v1, v2, m)
    bi = np.interp(t1, t2[g2], b[g2])
    gi = np.interp(t1, t2, g2.astype(float)) > 0.5
    ok = g1 & gi

    def r_of(x, mask):
        if mask.sum() < 10:
            return np.nan
        return float(np.corrcoef(a[mask], x[mask])[0, 1])
    obs = r_of(bi, ok)
    k0 = int(round(MIN_SHIFT_S / TIMELINE_STEP_S))
    null = np.array([r_of(np.roll(bi, k), g1 & np.roll(gi, k)) for k in range(k0, len(bi) - k0 + 1)])
    null = null[np.isfinite(null)]
    out.update(n=int(ok.sum()), r=round(obs, 3))
    if len(null) >= 10:
        out["p"] = round(float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null))), 4)
    return out | _scene_side(v1, v2, m)


def _scene_side(v1, v2, m):
    s1 = {r[2]: r for r in v1["scenes"] if r[3] == m}
    s2 = {r[2]: r for r in v2["scenes"] if r[3] == m}
    both = [i for i in s1 if i in s2 and s1[i][9] is not None and s2[i][9] is not None]
    same = sum(1 for i in both if np.sign(s1[i][9]) == np.sign(s2[i][9]))
    return {"same": same, "both": len(both)}


def scene_replication(session, ad, rows1, rows2, m, ok):
    """Per scene: is its position relative to the ad's own median the same in both
    viewings? 'confirmed' also needs: measure passed this viewer's quality check,
    clean in both viewings, outside the rest range in viewing 1."""
    s2 = {r[2]: r for r in rows2 if r[3] == m}
    out = []
    for r in rows1:
        if r[3] != m:
            continue
        q = s2.get(r[2])
        a, b = r[9], (q[9] if q else None)
        if a is None or b is None or not ok:
            st = "not judgeable"
        elif abs(a) < 1e-9 or abs(b) < 1e-9:
            st = "typical scene (no difference)"   # it IS the ad's median scene: no direction to repeat
        elif np.sign(a) != np.sign(b):
            st = "did not repeat"
        elif r[8] in ("above rest", "below rest"):
            st = "confirmed"
        else:
            st = "repeated, within normal variation"
        out.append((session, ad, r[2], m, a, b, r[8], q[8] if q else None, st))
    return out


def build(sessions=None, lag=LAG_S):
    # Rebuild in place (drop + recreate tables) rather than deleting the file: on
    # Windows the app server or OneDrive may hold the file open, and deleting fails.
    db = sqlite3.connect(DB_PATH, timeout=30)
    for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        db.execute(f'DROP TABLE IF EXISTS "{t}"')
    db.commit()
    db.executescript(SCHEMA)
    db.execute("INSERT INTO meta VALUES('built', ?)", (dt.datetime.now().isoformat(timespec="seconds"),))
    db.execute("INSERT INTO meta VALUES('method', ?)", (__doc__,))
    scenes_by_ad = load_static(db)
    if sessions is None:
        sessions = sorted(os.path.basename(p)[:-len("_raw.fif")] for p in glob.glob(os.path.join(LAB, "*_raw.fif")))
        sessions = [s for s in sessions if s not in SKIP_SESSIONS and not s.startswith("zz_test")]
    for s in sessions:
        process_session(db, s, scenes_by_ad, lag)
    db.commit()
    db.close()
    print(f"database -> {DB_PATH}")


def update_session(session, lag=LAG_S):
    """Add or refresh ONE viewer without rebuilding everyone (used by the app's
    'Analyse & view results' button). Labels/ads tables are refreshed too, since
    scenes may have been edited. Falls back to a full build if the database is
    missing or has an older layout."""
    if not os.path.exists(os.path.join(LAB, f"{session}_raw.fif")):
        sys.exit(f"no recording {session}_raw.fif")
    db = sqlite3.connect(DB_PATH, timeout=30)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    try:
        if "sessions" not in tables:
            raise sqlite3.OperationalError("no database yet")
        for t in ("ads", "scenes", "brand_events", "transcript", "questions", "measures"):
            db.execute(f"DELETE FROM {t}")
        scenes_by_ad = load_static(db)
        for t in ("sessions", "ratings", "scene_values", "ad_values", "timeline", "scene_values_repeat",
                  "timeline_repeat", "reliability", "brand_test", "scene_replication", "answers_text"):
            db.execute(f"DELETE FROM {t} WHERE session=?", (session,))
        process_session(db, session, scenes_by_ad, lag)
        db.execute("INSERT OR REPLACE INTO meta VALUES('built', ?)", (dt.datetime.now().isoformat(timespec="seconds"),))
        db.commit()
        db.close()
        print(f"database updated for {session} -> {DB_PATH}")
    except sqlite3.OperationalError as e:
        db.close()
        print(f"incremental update not possible ({e}) -- full rebuild")
        build(None, lag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", action="append", help="only these sessions (repeatable)")
    ap.add_argument("--update", help="add/refresh one viewer without rebuilding the others")
    ap.add_argument("--lag", type=float, default=LAG_S)
    a = ap.parse_args()
    if a.update:
        update_session(a.update, a.lag)
    else:
        build(a.session, a.lag)


if __name__ == "__main__":
    main()
