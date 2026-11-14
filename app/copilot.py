"""
Neuromarketing copilot: answers questions about the ads and the brain data in
plain English -- but every number comes from the project database
(data/lab_session/neuro.db) through fixed, tested tools. The model never sees
raw EEG and never writes its own queries.

Safeguards
  * ground rules in the instructions (numbers only from tools, cite seconds,
    no verdict inside the normal rest range, say how many viewers, label any
    "why" as a hypothesis, propose the study that would settle it)
  * number checker: every number in the answer must appear in this turn's tool
    results (or the question); otherwise the answer is flagged in the UI
  * everything is logged to copilot_log (question, tool calls, answer, flags)

What is sent to OpenAI: the question, computed numbers, scene labels and
transcripts of public ads, ad frames. No raw EEG, no names. store=False.

CLI test:  python app/copilot.py "Did people notice the Budweiser logo?"
"""
import base64
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
LAB = os.path.join(PROJ, "data", "lab_session")
DB_PATH = os.path.join(LAB, "neuro.db")
ADS_DIR = os.path.join(LAB, "ads")
BRIEF_DIR = os.path.join(LAB, "briefs")
sys.path.insert(0, os.path.join(PROJ, "lab_task_psychopy"))
from session_config import load_profile as SC_load_profile  # noqa: E402
MODEL = os.environ.get("COPILOT_MODEL", "gpt-5.4-mini")   # cheaper; gpt-5.5 used up credits fast
MAX_ROUNDS = 8

# Marketing question -> brain measure -> how much to trust it. Grounded in the
# team's research guide (van Diepen et al. 2025 J. Advertising reliability study,
# Silberstein 2020, Barnett & Cerf 2017, Kislov 2023, Varan 2015).
TRANSLATOR = [
    {"asks": "attention, noticing, did they watch, was it boring, did the logo land (attention part)",
     "measure": "attention", "evidence": "good",
     "note": "Posterior alpha suppression is one of the more reliable EEG ad measures (van Diepen et al. 2025). "
             "Brightness changes on screen also move it -- check the scene's brightness."},
    {"asks": "alertness, energy, engagement, excitement, arousal, emotional intensity, how strongly they felt",
     "measure": "alertness", "evidence": "good",
     "note": "Beta/alpha engagement index; tracked large-scale viewing behaviour in Kislov 2023. It is the INTENSITY "
             "(arousal) side of emotion: it says how strongly, not whether pleasant. "
             "Muscle tension (jaw, frowning) can inflate beta on a dry cap."},
    {"asks": "effort, thinking, confusion, processing, was it hard to follow, memory encoding",
     "measure": "effort", "evidence": "moderate",
     "note": "Frontal-midline theta rises with working-memory load and encoding; it does not say if the effort was pleasant."},
    {"asks": "emotion, feeling, liking, desire, approach, pleasant or unpleasant, emotional direction, valence",
     "measure": "wanting (shown in the app as 'Approach')", "evidence": "moderate -- use with care",
     "note": "Frontal alpha asymmetry is the DIRECTION (valence) side of emotion: + = leaning toward the content "
             "(approach), - = pulling away (avoidance) (Davidson; Allen). It was the least reliable EEG ad metric in "
             "van Diepen et al. 2025, and on 30 s of one person's data pure noise moves it by about 0.18 (SD), so "
             "only state it as a finding when it repeats (second viewing or several viewers), and set it beside the "
             "viewer's own liking / feeling ratings."},
    {"asks": "will they remember the brand, brand recall, branding moment",
     "measure": "attention + effort at the brand moment", "evidence": "moderate",
     "note": "Brand memory links to engagement at the moment the brand appears, especially at story closure "
             "(Silberstein 2020); the first ~20 s of an ad matter most for attention (Barnett & Cerf 2017)."},
    {"asks": "is this better than another ad / which ad wins",
     "measure": "any, compared across ads", "evidence": "needs many viewers",
     "note": "Vendors disagree on ad rankings (Varan 2015); reliable rankings need ~30-40 viewers (van Diepen et al. 2025)."},
]

INSTRUCTIONS = f"""You are the analysis copilot of a student neuromarketing study (BMS College of Engineering,
industry collaboration with ITIE Knowledge Solutions). People watched TV ads while wearing a 19-channel
dry-electrode EEG cap; each ad was split into story scenes by an AI labeller with frame-exact boundaries
(scenes_status "reviewed" = a human checked them; "draft" = not yet checked -- say so if relevant).
You answer questions from marketers and engineers about what the brain data says about the ads.

The four brain measures (always relative to the same person's eyes-open rest), in order of trust:
Use these display names in answers: attention = "Visual attention", alertness = "Engagement",
effort = "Mental effort", wanting = "Approach".
  attention  (posterior alpha suppression, %): PRIMARY.     + = more visual attention
  alertness  (beta/alpha engagement index, %): SUPPORTING.  + = more alert / engaged
  effort     (frontal theta, dB; app name "Processing"): EXPLORATORY. + = more mental processing
  wanting    (frontal alpha asymmetry; app name "Approach"): SUPPORTING, use with care. + = leaning toward the
             content (pleasant / approach), - = pulling away. Noisier than the others: a finding only when it repeats.
EMOTION: there is no single "emotion" signal. Describe emotion as two parts: Approach = emotional DIRECTION
(valence: drawn in vs pulling away) and Engagement = emotional INTENSITY (arousal: how strongly). Never name a
specific emotion (joy, sadness, fear) from EEG -- the cap cannot tell those apart.

GROUND RULES -- follow every time:
0. Before answering ANY question about how viewers responded to an ad, call get_scenes or find_moments
   (and get_ad_overview for ratings / whole-ad values). Never answer from brand moments or memory alone.
1. Every number you state must come from a tool result in this conversation, copied as given (same
   rounding). Never compute new numbers, never estimate, never invent. If you need a number, call a tool.
2. Cite moments as [t=SECONDS] in video seconds (e.g. [t=57.6]) for the ad the user is looking at, and as
   [adNN t=SECONDS] (e.g. [ad09 t=61.0]) for ANY OTHER ad -- the app turns both into buttons that open that ad
   and jump the video. Cite a scene as its label plus its time. Never cite another ad's moment as plain [t=..].
3. Each scene value has a normal range from rest. If the verdict is "within rest range", say the data
   cannot tell it apart from ordinary resting wobble -- do not call it high or low. "too noisy to judge"
   means no conclusion at all.
4. Rest is a fixation cross, the ads are moving video, so attention is usually above rest everywhere.
   To say which part of an ad stood out, use vs_ad_median (difference from the ad's typical clean scene,
   in POINTS of the same unit -- write e.g. "18.9 points above the ad's typical scene", not "18.9%").
5. Always say how many viewers the answer rests on. With 1-2 viewers everything is a hypothesis.
   With several viewers, tools give per-viewer values, the median, and agreement counts -- report the
   agreement ("2 of 2 viewers..."); never present one viewer's value as everyone's.
6. Any "why" (e.g. "because the music swells") is a hypothesis -- say so. You may use scene labels,
   what is heard, brightness, loudness and cuts to suggest one; mention confounds (brightness moves
   attention, loud sounds and fast cuts move everything).
7. When the data cannot answer, say so plainly and end with one short "To settle this:" line --
   the study that would (what comparison, which measure, and viewers: about 30-40 for a reliable
   group result, per van Diepen et al. 2025 -- use exactly that number, no other).
8. Map marketing words to measures with explain_measure when unsure. Approach ("wanting") is noisier than the
   others: treat a single viewing as a hint, and state it as a finding only when it repeats; for emotion questions
   use Approach (direction) + Engagement (intensity) and the viewer's own feeling ratings if present.
9. Use look_at_frames when what was on screen matters. Use seek_video to show the key moment.
10. Always write units with brain values: attention and alertness in %, effort in dB (e.g. "attention +35.0%",
   "effort -3.7 dB"); approach has no unit.
11. When your answer points at one key moment, call seek_video for it (once) so the video jumps there.
12. Be brief: 3-6 sentences or a short list. Plain English, no jargon without a gloss. No markdown headings.
13. Before calling any pattern real for a viewer, or answering about the brand moment, brand recall or
   "how sure", call get_reliability. Use its status words exactly: "confirmed for this viewer",
   "seen once", "no clear difference", "repeats" / "does not repeat"; per scene "confirmed",
   "repeated, within normal variation", "did not repeat", "typical scene (no difference)" (= it is the ad's
   typical scene, never a finding). Only "confirmed" findings may be stated as
   findings for that viewer; everything else is a hypothesis. A viewer watched an ad once unless
   get_reliability says viewings = 2.
15. For "what worked", "what should we change", a summary, or brand timing, call get_insights FIRST and keep your
   answer consistent with it: only its "what_held_viewers" and "ideas_to_test" items are data-backed findings.
   Say "idea to test", never "you should". If it lists no ideas, say there is no data-backed suggestion yet.
16. Cross-ad questions ("what worked for this viewer in ad A compared with ad B", viewer preferences): call
   compare_viewer_ads for that viewer (if no viewer is named, use list_ads and pick the viewer who watched the ad in
   context and the most other ads, and say which viewer you used). Describe how THIS viewer's responses differed
   between the ads and what the stand-out scenes have in common (story role, brand on screen, what was said), and
   set it beside their own ratings. Say clearly it is one viewer. Do not declare any company's ad better or worse
   than another's -- describe the viewer's response, not the ads' quality.
   Wording: "this viewer responded more strongly to ...", never "ad X worked better", "outlier", "weakest" or a ranking.
   Ads sell different things to different audiences: a low purchase-intent rating for a business product from a
   student viewer says nothing about the ad -- say so when ratings differ. If an ad's recording was too noisy,
   say only that it could not be judged.
14. For "did the ad achieve its objective" or business questions, call get_objectives: separate what the
   company documented (with source) from the team's assumptions, and never attribute public campaign
   results (other people) to our viewers' EEG. Purchase intent is a rating, not a purchase.
"""


# ----------------------------------------------------------------------------- data access
def _db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def _r(x, nd=1):
    return None if x is None else round(float(x), nd)


def _excluded():
    try:
        with open(os.path.join(LAB, "viewers.json"), encoding="utf-8") as fh:
            return {k for k, v in json.load(fh).items() if v.get("excluded")}
    except (OSError, ValueError):
        return set()


def _sessions_for(db, ad):
    ex = _excluded()
    return [r["session"] for r in db.execute(
        "SELECT DISTINCT session FROM scene_values WHERE ad=? ORDER BY session", (ad,)) if r["session"] not in ex]


def _check_ad(db, ad):
    if not db.execute("SELECT 1 FROM ads WHERE ad=?", (ad,)).fetchone():
        ads = [r["ad"] for r in db.execute("SELECT ad FROM ads")]
        raise ValueError(f"unknown ad '{ad}'. Known: {ads}")


def tool_list_ads():
    db = _db()
    out = []
    for a in db.execute("SELECT * FROM ads ORDER BY ad"):
        sess = _sessions_for(db, a["ad"])
        out.append({"ad": a["ad"], "brand": a["brand"], "product": a["product"],
                    "duration_s": _r(a["duration_s"], 2), "n_scenes": db.execute(
                        "SELECT COUNT(*) FROM scenes WHERE ad=?", (a["ad"],)).fetchone()[0],
                    "scenes_status": a["scenes_status"], "viewers_with_eeg": len(sess), "sessions": sess})
    return {"ads": out}


def tool_get_ad_overview(ad):
    db = _db()
    _check_ad(db, ad)
    a = dict(db.execute("SELECT * FROM ads WHERE ad=?", (ad,)).fetchone())
    sess = _sessions_for(db, ad)
    per = []
    for s in sess:
        se = dict(db.execute("SELECT * FROM sessions WHERE session=?", (s,)).fetchone())
        vals, ad_verdict, ad_noisy = {}, None, None
        for r in db.execute("SELECT measure, value, verdict, noisy_frac FROM ad_values WHERE session=? AND ad=?", (s, ad)):
            vals[r["measure"]] = _r(r["value"], 2)
            ad_verdict, ad_noisy = r["verdict"], r["noisy_frac"]
        if ad_verdict == "too noisy to judge":
            vals = {"verdict": "too noisy to judge -- do not quote these whole-ad numbers",
                    "noisy_fraction": _r(ad_noisy, 2)}
        ratings = {r["question"]: r["answer"] for r in db.execute(
            "SELECT question, answer FROM ratings WHERE session=? AND ad=?", (s, ad))}
        timing = db.execute("SELECT timing FROM scene_values WHERE session=? AND ad=? LIMIT 1", (s, ad)).fetchone()
        per.append({"session": s, "whole_ad_vs_rest": vals, "self_report_ratings_1to9": ratings,
                    "scene_timing": timing["timing"] if timing else None,
                    "cap_check_eyes_closed_ratio": se["cap_check_ratio"],
                    "reference_fix": json.loads(se["reference_fix"] or "{}") if "reference_fix" in se.keys() else None,
                    "bad_electrodes": json.loads(se["bad_channels"] or "[]")})
    return {"ad": ad, "brand": a["brand"], "product": a["product"], "duration_s": _r(a["duration_s"], 2),
            "scenes_status": a["scenes_status"],
            "story": a["summary"], "sound": a["audio_summary"], "viewers_with_eeg": len(sess),
            "per_viewer": per,
            "notes": "whole-ad values: median of 3 s windows vs eyes-open rest (no rest range for a whole ad). "
                     "two_marker_estimate timing can be off by ~0.1-0.3 s."}


def _scene_rows(db, ad, session):
    rows = []
    for s in db.execute("SELECT * FROM scenes WHERE ad=? ORDER BY idx", (ad,)):
        row = {"scene": s["idx"], "label": s["label"], "story_role": s["story_role"],
               "start_s": _r(s["start_s"], 2), "end_s": _r(s["end_s"], 2), "dur_s": _r(s["dur_s"], 2),
               "heard": s["heard"], "brightness_0to255": _r(s["brightness"], 0),
               "loudness_db": s["loudness_db_mean"], "camera_cuts": s["n_cuts"]}
        if session:
            v = {}
            for r in db.execute("SELECT * FROM scene_values WHERE session=? AND ad=? AND scene_idx=?",
                                (session, ad, s["idx"])):
                v[r["measure"]] = {"value": _r(r["value"], 2 if r["measure"] == "wanting" else 1),
                                   "rest_range": [_r(r["rest_lo"]), _r(r["rest_hi"])],
                                   "verdict": r["verdict"], "vs_ad_median": _r(r["vs_ad_median"])}
                row["noisy_fraction"] = _r(r["noisy_frac"], 2)
            row["brain"] = v
        rows.append(row)
    return rows


def _combined(db, ad, sess):
    """All viewers together: per scene and measure, the median across viewers plus how many agree."""
    rows = _scene_rows(db, ad, None)
    for row in rows:
        brain = {}
        for m in ("attention", "alertness", "effort", "wanting"):
            per = {}
            for s in sess:
                r = db.execute("SELECT * FROM scene_values WHERE session=? AND ad=? AND scene_idx=? AND measure=?",
                               (s, ad, row["scene"], m)).fetchone()
                if r:
                    per[s] = {"value": _r(r["value"], 2 if m == "wanting" else 1), "verdict": r["verdict"],
                              "vs_ad_median": _r(r["vs_ad_median"], 2 if m == "wanting" else 1)}
            judged = [p for p in per.values() if p["vs_ad_median"] is not None]
            brain[m] = {
                "per_viewer": per,
                "median_value": _r(np.median([p["value"] for p in judged]), 2 if m == "wanting" else 1) if judged else None,
                "median_vs_ad_median": _r(np.median([p["vs_ad_median"] for p in judged]), 2 if m == "wanting" else 1) if judged else None,
                "viewers_judged": len(judged),
                "viewers_above_typical": sum(1 for p in judged if p["vs_ad_median"] > 0),
                "viewers_below_typical": sum(1 for p in judged if p["vs_ad_median"] < 0),
                "viewers_above_rest": sum(1 for p in per.values() if p["verdict"] == "above rest"),
                "viewers_below_rest": sum(1 for p in per.values() if p["verdict"] == "below rest")}
        row["brain"] = brain
    return rows


def tool_get_scenes(ad, session=None):
    db = _db()
    _check_ad(db, ad)
    sess = _sessions_for(db, ad)
    units = {"attention": "%", "alertness": "%", "effort": "dB", "wanting": "log ratio"}
    notes = ("rest_range = central 90% of same-length rest chunks; vs_ad_median = scene minus the ad's typical "
             "clean scene (use it to say what stood out within the ad).")
    if session or len(sess) <= 1:
        session = session or (sess[0] if sess else None)
        return {"ad": ad, "session": session, "viewers_with_eeg": len(sess), "units": units,
                "scenes": _scene_rows(db, ad, session), "notes": notes}
    return {"ad": ad, "viewers_with_eeg": len(sess), "sessions": sess, "units": units,
            "scenes": _combined(db, ad, sess),
            "notes": notes + " Several viewers: per_viewer values plus the median across viewers and how many "
                     "viewers agree. Report agreement (e.g. '2 of 2 viewers above the ad's typical scene')."}


def tool_get_timeline(ad, measure, start_s, end_s, session=None):
    db = _db()
    _check_ad(db, ad)
    sess = _sessions_for(db, ad)
    session = session or (sess[0] if sess else None)
    pts = [(r["t_s"], r["value"]) for r in db.execute(
        "SELECT t_s, value FROM timeline WHERE session=? AND ad=? AND measure=? AND t_s BETWEEN ? AND ? ORDER BY t_s",
        (session, ad, measure, start_s, end_s))]
    step = max(1, len(pts) // 30)
    return {"ad": ad, "session": session, "measure": measure,
            "points": [{"t": _r(t, 1), "value": _r(v)} for t, v in pts[::step]],
            "note": "2 s windows every 0.5 s (subsampled), vs rest; single windows are noisy -- prefer scene values."}


def tool_find_moments(ad, measure, session=None):
    db = _db()
    _check_ad(db, ad)
    sess = _sessions_for(db, ad)
    if not session and len(sess) > 1:
        rows = []
        for row in _combined(db, ad, sess):
            b = row["brain"][measure]
            if b["median_vs_ad_median"] is not None:
                rows.append({"scene": row["scene"], "label": row["label"], "story_role": row["story_role"],
                             "start_s": row["start_s"], "median_value": b["median_value"],
                             "median_vs_ad_median": b["median_vs_ad_median"], "viewers_judged": b["viewers_judged"],
                             "viewers_above_typical": b["viewers_above_typical"],
                             "viewers_below_typical": b["viewers_below_typical"]})
        rows.sort(key=lambda r: r["median_vs_ad_median"], reverse=True)
        return {"ad": ad, "measure": measure, "viewers_with_eeg": len(sess), "sessions": sess,
                "highest": rows[:3], "lowest": rows[::-1][:3],
                "note": "median across viewers; trust a scene more when all viewers agree on the direction"}
    session = session or (sess[0] if sess else None)
    rows = [dict(r) for r in db.execute(
        """SELECT s.idx, s.label, s.story_role, s.start_s, s.end_s, v.value, v.vs_ad_median, v.verdict, v.noisy_frac
           FROM scenes s JOIN scene_values v ON v.ad=s.ad AND v.scene_idx=s.idx
           WHERE s.ad=? AND v.session=? AND v.measure=? AND v.vs_ad_median IS NOT NULL""", (ad, session, measure))]
    rows.sort(key=lambda r: r["vs_ad_median"], reverse=True)
    fmt = lambda r: {"scene": r["idx"], "label": r["label"], "story_role": r["story_role"],
                     "start_s": _r(r["start_s"], 2), "value": _r(r["value"]), "vs_ad_median": _r(r["vs_ad_median"]),
                     "verdict": r["verdict"]}
    skipped = [r[0] for r in db.execute(
        "SELECT scene_idx FROM scene_values WHERE ad=? AND session=? AND measure=? AND vs_ad_median IS NULL",
        (ad, session, measure))]
    return {"ad": ad, "session": session, "measure": measure,
            "highest": [fmt(r) for r in rows[:3]], "lowest": [fmt(r) for r in rows[::-1][:3]],
            "excluded_noisy_scenes": skipped}


def tool_compare_ads(measure, story_role=None):
    db = _db()
    out = []
    for a in db.execute("SELECT ad, brand FROM ads ORDER BY ad"):
        for s in _sessions_for(db, a["ad"]):
            if story_role:
                v = [r["value"] for r in db.execute(
                    """SELECT v.value FROM scene_values v JOIN scenes s ON s.ad=v.ad AND s.idx=v.scene_idx
                       WHERE v.ad=? AND v.session=? AND v.measure=? AND s.story_role=? AND v.verdict!='too noisy to judge'""",
                    (a["ad"], s, measure, story_role)) if r["value"] is not None]
                val = _r(float(np.median(v))) if v else None
            else:
                r = db.execute("SELECT value, verdict FROM ad_values WHERE ad=? AND session=? AND measure=?",
                               (a["ad"], s, measure)).fetchone()
                val = (_r(r["value"]) if r["verdict"] != "too noisy to judge" else "too noisy to judge") if r else None
            out.append({"ad": a["ad"], "brand": a["brand"], "session": s, "value": val})
    return {"measure": measure, "story_role": story_role, "rows": out,
            "note": "ads without a recording are missing. Rankings need ~30-40 viewers to be trustworthy."}


def tool_get_brand_moments(ad, session=None):
    db = _db()
    _check_ad(db, ad)
    sess = _sessions_for(db, ad)
    session = session or (sess[0] if sess else None)
    ev = [dict(r) for r in db.execute("SELECT kind, start_s, end_s, what FROM brand_events WHERE ad=? ORDER BY start_s", (ad,))]
    out = []
    for e in ev:
        sc = db.execute("SELECT idx, label FROM scenes WHERE ad=? AND start_s<=? AND end_s>? ",
                        (ad, e["start_s"], e["start_s"])).fetchone()
        out.append({"kind": e["kind"], "start_s": _r(e["start_s"], 2), "end_s": _r(e["end_s"], 2), "what": e["what"],
                    "in_scene": {"scene": sc["idx"], "label": sc["label"]} if sc else None})
    return {"ad": ad, "session": session, "brand_moments": out,
            "tip": "get the brain values of those scenes with get_scenes"}


def tool_explain_measure(question_words):
    return {"translator": TRANSLATOR}


def tool_look_at_frames(ad, start_s, end_s, n=3):
    db = _db()
    _check_ad(db, ad)
    a = db.execute("SELECT file, fps, n_frames FROM ads WHERE ad=?", (ad,)).fetchone()
    n = int(max(1, min(4, n)))
    cap = cv2.VideoCapture(os.path.join(ADS_DIR, a["file"]))
    imgs, ts = [], np.linspace(start_s, max(start_s, end_s), n) if n > 1 else [start_s]
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(a["n_frames"] - 1, int(t * a["fps"])))
        ok, fr = cap.read()
        if ok:
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (384, int(384 * h / w)))
            ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            imgs.append((round(float(t), 1), base64.b64encode(buf.tobytes()).decode()))
    cap.release()
    return {"_images": imgs, "ad": ad, "times": [t for t, _ in imgs]}


def tool_seek_video(ad, t):
    return {"_action": {"seek": {"ad": ad, "t": float(t)}}, "ok": True}


def _has_table(db, name):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())


def tool_get_reliability(ad, session=None):
    """Single-viewer reliability: brand-moment test, repeat viewing, scene confirmation, brand recall."""
    db = _db()
    _check_ad(db, ad)
    sess = [session] if session else _sessions_for(db, ad)
    if not _has_table(db, "brand_test"):
        return {"ad": ad, "note": "database built before the reliability tests existed -- re-analyse the viewer"}
    out = []
    for s in sess:
        bt = [{"viewing": r["viewing"], "measure": r["measure"], "brand_minus_rest_of_ad": _r(r["diff"], 2),
               "p": _r(r["p"], 3), "brand_windows": r["n_brand"], "other_windows": r["n_other"],
               "status": r["status"], "note": r["note"] or None}
              for r in db.execute("SELECT * FROM brand_test WHERE session=? AND ad=? ORDER BY measure, viewing", (s, ad))]
        rel = [{"measure": r["measure"], "timeline_r_between_viewings": _r(r["r"], 2), "p": _r(r["p"], 3),
                "scenes_same_side_both_viewings": r["scenes_same_side"], "scenes_judged_both": r["scenes_both_clean"],
                "verdict": r["verdict"]}
               for r in db.execute("SELECT * FROM reliability WHERE session=? AND ad=?", (s, ad))]
        reps = {}
        for r in db.execute("""SELECT v.measure, v.status, v.scene_idx, s.label, s.start_s FROM scene_replication v
                               JOIN scenes s ON s.ad=v.ad AND s.idx=v.scene_idx WHERE v.session=? AND v.ad=?""", (s, ad)):
            reps.setdefault(r["measure"], []).append({"scene": r["scene_idx"], "label": r["label"],
                                                      "start_s": _r(r["start_s"], 2), "status": r["status"]})
        viewings = 2 if db.execute("SELECT 1 FROM timeline_repeat WHERE session=? AND ad=? LIMIT 1", (s, ad)).fetchone() else 1
        recall = db.execute("SELECT answer FROM ratings WHERE session=? AND ad=? AND question='brand_recall'", (s, ad)).fetchone()
        choice = db.execute("SELECT text FROM answers_text WHERE session=? AND ad=? AND question='brand_recall'", (s, ad)).fetchone()
        out.append({"session": s, "viewings": viewings, "brand_moment_test": bt, "repeat_reliability": rel,
                    "scene_replication": reps,
                    "brand_recall_end_of_session": None if recall is None else
                    {"correct": bool(recall["answer"]), "chose": choice["text"] if choice else None}})
    return {"ad": ad, "viewers": out,
            "method": "Within one viewer. brand_moment_test: brand-on-screen windows vs the rest of the ad after removing "
                      "the ad's straight-line trend; p from circular shifts (two-sided). Attention is the planned test; "
                      "alertness/effort are secondary, uncorrected. An ad of about 15 s cannot reach p < 0.05 with this "
                      "test. repeat_reliability: the viewer's timeline correlated between viewing 1 and 2."}


def tool_get_objectives(ad):
    prof = SC_load_profile(ad)
    if not prof or not prof.get("profile"):
        return {"ad": ad, "note": "no campaign-objective profile for this ad"}
    return {"ad": ad, "company": prof.get("company"), "title": prof.get("title"), "year": prof.get("year"),
            **prof["profile"],
            "rules": "documented = public source (give it); assumed = the team's reading; public_evidence = other "
                     "people, campaign level -- never caused by or measured in our viewers."}


def tool_compare_viewer_ads(session, ads=None):
    """One viewer across every ad they watched: what differs between the ads for THIS person."""
    db = _db()
    if session in _excluded():
        return {"error": f"viewer {session} is excluded"}
    watched = [r["ad"] for r in db.execute("SELECT DISTINCT ad FROM scene_values WHERE session=? ORDER BY ad", (session,))]
    if not watched:
        return {"error": f"no analysed recording for viewer '{session}'",
                "viewers": [r["session"] for r in db.execute("SELECT DISTINCT session FROM scene_values")]}
    pick = [a for a in (ads or watched) if a in watched] or watched
    names = {"attention": "Visual attention", "alertness": "Engagement", "effort": "Mental effort"}
    out, pool_role, pool_brand = [], {}, {}
    for ad in pick:
        a = db.execute("SELECT * FROM ads WHERE ad=?", (ad,)).fetchone()
        whole, noisy = {}, False
        for r in db.execute("SELECT measure, value, verdict FROM ad_values WHERE session=? AND ad=?", (session, ad)):
            if r["verdict"] == "too noisy to judge":
                noisy = True
            elif r["measure"] in names:
                whole[names[r["measure"]]] = _r(r["value"], 1)
        ratings = {r["question"]: r["answer"] for r in db.execute("SELECT question, answer FROM ratings WHERE session=? AND ad=?", (session, ad))}
        ev = [dict(r) for r in db.execute("SELECT kind, start_s, end_s FROM brand_events WHERE ad=?", (ad,))]
        named = sorted(e["start_s"] for e in ev if e["kind"] in ("logo", "brand_text", "tagline"))
        rel = {}
        if _has_table(db, "reliability"):
            rel = {names[r["measure"]]: r["verdict"] for r in db.execute(
                "SELECT measure, verdict FROM reliability WHERE session=? AND ad=?", (session, ad)) if r["measure"] in names}
        scenes = {r["idx"]: r for r in db.execute("SELECT * FROM scenes WHERE ad=?", (ad,))}
        best = {}
        for m, nm in names.items():
            rows = [r for r in db.execute("SELECT scene_idx, vs_ad_median, verdict FROM scene_values WHERE session=? AND ad=? AND measure=? "
                                          "AND vs_ad_median IS NOT NULL", (session, ad, m))
                    if r["scene_idx"] in scenes and (scenes[r["scene_idx"]]["start_s"] + scenes[r["scene_idx"]]["end_s"]) / 2 >= 2.0]
            for r in rows:
                sc = scenes[r["scene_idx"]]
                on = any(e["kind"] != "spoken_brand" and e["start_s"] < sc["end_s"] and e["end_s"] > sc["start_s"] for e in ev)
                pool_role.setdefault(nm, {}).setdefault(sc["story_role"], []).append(r["vs_ad_median"])
                pool_brand.setdefault(nm, {}).setdefault("brand_on_screen" if on else "no_brand", []).append(r["vs_ad_median"])
            if rows:
                hi, lo = max(rows, key=lambda r: r["vs_ad_median"]), min(rows, key=lambda r: r["vs_ad_median"])
                f = lambda r: {"scene": scenes[r["scene_idx"]]["label"], "story_role": scenes[r["scene_idx"]]["story_role"],
                               "start_s": _r(scenes[r["scene_idx"]]["start_s"], 2), "vs_ad_typical": _r(r["vs_ad_median"], 1),
                               "vs_rest": r["verdict"]}
                best[nm] = {"highest_scene": f(hi), "lowest_scene": f(lo)}
        out.append({"ad": ad, "brand": a["brand"], "product": a["product"], "duration_s": _r(a["duration_s"], 1),
                    "brand_first_shown_pct_into_ad": round(100 * named[0] / a["duration_s"]) if named else None,
                    "ratings_1to9": ratings,
                    "whole_ad_vs_rest": "first viewing too noisy to judge" if noisy else whole,
                    "repeats_on_second_viewing": rel or "watched once",
                    "scenes_that_stood_out": best})
    mean = lambda v: _r(sum(v) / len(v), 1)
    return {"viewer": session, "ads_compared": len(out), "ads": out,
            "patterns_across_these_ads": {
                "by_story_role": {nm: {role: {"mean_vs_ad_typical": mean(v), "scenes": len(v)} for role, v in d.items()}
                                  for nm, d in pool_role.items()},
                "brand_on_screen_vs_not": {nm: {k: {"mean_vs_ad_typical": mean(v), "scenes": len(v)} for k, v in d.items()}
                                           for nm, d in pool_brand.items()}},
            "how_to_read": "whole_ad_vs_rest uses the same resting baseline for every ad, so levels are comparable for this viewer "
                           "(Visual attention and Engagement in %, Mental effort in dB). vs_ad_typical compares a scene with that "
                           "ad's own typical scene, so it can be pooled across ads. One viewer and few ads: describe tendencies "
                           "of THIS viewer; a pattern resting on fewer than 4 scenes is too thin to mention as a pattern."}


def tool_get_insights(ad):
    """The same findings the Results page shows in its Takeaways card."""
    import insights
    d = insights.build(ad, polish=True)
    if d.get("error"):
        return d
    return {"ad": ad, "ad_facts": d["audit"], "viewers": d["viewers"], "clean_viewings": d["viewings"],
            "what_held_viewers": [{k: i.get(k) for k in ("text", "evidence", "level", "t", "experimental", "cautions")} for i in d["strengths"]],
            "ideas_to_test": [{k: i.get(k) for k in ("label", "evidence", "idea", "level", "t", "experimental", "cautions")} for i in d["ideas"]],
            "scenes": [{"scene": s["idx"], "label": s["label"], "start_s": s["start_s"], "brand_on_screen": s["brand_visible"],
                        "agreement": {insights.NAMES[m]: f"{c['direction'] or 'no direction'} ({c['level']}, {c['agree']} of {c['of']} clean viewings)"
                                      for m, c in s["measures"].items() if c["level"] != "none"}} for s in d["scenes"] if not s["skip"]],
            "rules": d["rules"]}


TOOLS = {
    "get_insights": (tool_get_insights, "START HERE for 'what worked', 'what to change', 'summary' or brand-timing questions: "
                     "the ad's facts (brand first shown/said), what held viewers, and the data-backed ideas to test -- "
                     "exactly what the Results page shows.", {"ad": {"type": "string"}}),
    "compare_viewer_ads": (tool_compare_viewer_ads, "ONE viewer across the ads they watched (cross-ad): ratings, whole-ad brain "
                           "values on the same rest baseline, repeat reliability, the scenes that stood out in each ad, and "
                           "patterns pooled across ads (by story role; brand on screen vs not). Use for 'what works for this "
                           "viewer', 'compare ad X with ad Y for p04', viewer preferences.",
                           {"session": {"type": "string"}, "ads": {"type": ["array", "null"], "items": {"type": "string"}}}),
    "list_ads": (tool_list_ads, "List the ads, their brands, and how many viewers have EEG for each.", {}),
    "get_ad_overview": (tool_get_ad_overview, "Story, sound, whole-ad brain values vs rest and self-report ratings for one ad.",
                        {"ad": {"type": "string"}}),
    "get_scenes": (tool_get_scenes, "Every scene of an ad with times, labels, what is heard, controls and the four brain "
                   "values (value, rest range, verdict, vs_ad_median). Main tool for 'which part' questions.",
                   {"ad": {"type": "string"}, "session": {"type": ["string", "null"]}}),
    "get_timeline": (tool_get_timeline, "Second-by-second values of one measure over a time range of an ad.",
                     {"ad": {"type": "string"}, "measure": {"type": "string", "enum": ["attention", "alertness", "effort", "wanting"]},
                      "start_s": {"type": "number"}, "end_s": {"type": "number"}, "session": {"type": ["string", "null"]}}),
    "find_moments": (tool_find_moments, "The scenes that stood out most (highest/lowest vs the ad's typical scene) for one measure.",
                     {"ad": {"type": "string"}, "measure": {"type": "string", "enum": ["attention", "alertness", "effort", "wanting"]},
                      "session": {"type": ["string", "null"]}}),
    "compare_ads": (tool_compare_ads, "One measure across all recorded ads, whole ad or only scenes with a given story role.",
                    {"measure": {"type": "string", "enum": ["attention", "alertness", "effort", "wanting"]},
                     "story_role": {"type": ["string", "null"], "enum": ["setup", "build_up", "climax", "resolution", "brand", "other", None]}}),
    "get_brand_moments": (tool_get_brand_moments, "When the brand is visible or said aloud in an ad, and which scene that falls in.",
                          {"ad": {"type": "string"}, "session": {"type": ["string", "null"]}}),
    "explain_measure": (tool_explain_measure, "Which brain measure answers a marketing question, and how much to trust it.",
                        {"question_words": {"type": "string"}}),
    "look_at_frames": (tool_look_at_frames, "See 1-4 frames of an ad between two times (seconds).",
                       {"ad": {"type": "string"}, "start_s": {"type": "number"}, "end_s": {"type": "number"},
                        "n": {"type": "integer"}}),
    "seek_video": (tool_seek_video, "Jump the video player to a moment so the user can see it.",
                   {"ad": {"type": "string"}, "t": {"type": "number"}}),
    "get_reliability": (tool_get_reliability, "Is a pattern real for this viewer? Brand-moment test (brand on screen vs "
                        "rest of ad), repeat-viewing reliability, per-scene confirmation, and end-of-session brand recall.",
                        {"ad": {"type": "string"}, "session": {"type": ["string", "null"]}}),
    "get_objectives": (tool_get_objectives, "The ad's campaign-objective profile: documented facts with sources, team "
                       "assumptions, public outcome evidence, and how each objective is tested.",
                       {"ad": {"type": "string"}}),
}


def tool_specs():
    specs = []
    for name, (_, desc, props) in TOOLS.items():
        specs.append({"type": "function", "name": name, "description": desc, "strict": True,
                      "parameters": {"type": "object", "properties": props, "required": list(props),
                                     "additionalProperties": False}})
    return specs


# ----------------------------------------------------------------------------- number checker
NUM_RE = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?")


def _numbers(text):
    out = []
    for m in NUM_RE.finditer(text.replace("−", "-")):
        s = m.group(0)
        dec = len(s.split(".")[1]) if "." in s else 0
        out.append((abs(float(s)), dec, s))
    return out


def check_numbers(answer, sources):
    """Numbers in the answer that appear nowhere in the tool results / question (rounding-tolerant)."""
    pool = set()
    for src in sources:
        for v, _, _ in _numbers(src):
            pool.add(v)
    unverified = []
    for v, dec, s in _numbers(answer):
        if v <= 10 and dec == 0:          # small integers: counts, scene numbers, ratings 1-9
            continue
        tol = 0.5 * 10 ** (-dec) + 1e-9
        if not any(abs(v - p) <= tol for p in pool):
            unverified.append(s)
    return sorted(set(unverified))


# ----------------------------------------------------------------------------- the loop
def _ensure_key():
    """setx only reaches NEW processes; fall back to the user's saved Windows variable."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                os.environ["OPENAI_API_KEY"] = winreg.QueryValueEx(k, "OPENAI_API_KEY")[0]
                return
        except OSError:
            pass
    raise RuntimeError("OPENAI_API_KEY is not set (setx OPENAI_API_KEY \"sk-...\").")


def ask(question, history=None, context=None, model=None, log=True):
    """history: [{"role": "user"|"assistant", "content": str}]; context: {"ad", "t", "session"}."""
    from openai import OpenAI
    _ensure_key()
    client = OpenAI()
    model = model or MODEL
    ctx = ""
    if context and context.get("ad"):
        ctx = f"\n\n(Context: the user is looking at {context['ad']}" + (
            f", video at {context['t']:.1f} s" if context.get("t") is not None else "") + (
            f", viewer session {context['session']}" if context.get("session") else "") + ".)"
    inp = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-8:]]
    inp.append({"role": "user", "content": question + ctx})
    calls, actions, sources = [], [], [question, INSTRUCTIONS]
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "requests": 0}
    t0 = time.time()
    answer = ""
    for _ in range(MAX_ROUNDS):
        r = client.responses.create(model=model, instructions=INSTRUCTIONS, input=inp, tools=tool_specs(),
                                    store=False, include=["reasoning.encrypted_content"],
                                    reasoning={"effort": "low"})
        u = getattr(r, "usage", None)
        if u is not None:
            usage["requests"] += 1
            usage["input_tokens"] += u.input_tokens or 0
            usage["output_tokens"] += u.output_tokens or 0
            det = getattr(u, "input_tokens_details", None)
            usage["cached_input_tokens"] += (getattr(det, "cached_tokens", 0) or 0) if det else 0
        fcs = [o for o in r.output if o.type == "function_call"]
        inp += [o.model_dump(exclude_none=True) for o in r.output]
        if not fcs:
            answer = r.output_text
            break
        for fc in fcs:
            fn = TOOLS.get(fc.name, (None,))[0]
            try:
                args = json.loads(fc.arguments or "{}")
                res = fn(**args) if fn else {"error": f"unknown tool {fc.name}"}
            except Exception as e:  # tool errors go back to the model, not to the user
                args, res = {}, {"error": str(e)}
            if "_action" in res:
                actions.append(res.pop("_action"))
            imgs = res.pop("_images", None)
            text = json.dumps(res, ensure_ascii=False, default=str)
            sources.append(text)
            calls.append({"tool": fc.name, "args": args, "result_chars": len(text)})
            out = text
            if imgs:
                out = [{"type": "input_text", "text": text}] + [
                    {"type": "input_image", "image_url": "data:image/jpeg;base64," + b64, "detail": "low"}
                    for _, b64 in imgs]
            inp.append({"type": "function_call_output", "call_id": fc.call_id, "output": out})
    else:
        answer = "I could not finish this one within the step limit -- try asking more specifically."
    unverified = check_numbers(answer, sources)
    res = {"answer": answer, "actions": actions, "tool_calls": calls, "unverified_numbers": unverified,
           "model": model, "seconds": round(time.time() - t0, 1), "usage": usage}
    if log:
        try:
            db = sqlite3.connect(DB_PATH)
            db.execute("INSERT INTO copilot_log VALUES(?,?,?,?,?)",
                       (dt.datetime.now().isoformat(timespec="seconds"), question, json.dumps(calls),
                        answer, json.dumps({"unverified_numbers": unverified, "model": model, "usage": usage})))
            db.commit()
        except sqlite3.Error:
            pass
    return res


BRIEF_PROMPT = """Write a one-page brief on ad {ad} for a marketing team, using the tools (get_ad_overview,
get_scenes, get_brand_moments, find_moments for attention and effort; look_at_frames for the key moments).
Plain text. Use exactly these section titles, each on its own line followed by a colon, then 1-3 short lines:
What the ad is:
What held attention:
Where attention dipped:
The brand moment:
Ratings vs brain:
What we would change (hypotheses):
How sure we are:
To settle this:
Cite moments with [t=..]. Mention whether the scenes are human-reviewed or still draft.
Follow the ground rules strictly."""


def brief(ad, regenerate=False, model=None):
    os.makedirs(BRIEF_DIR, exist_ok=True)
    p = os.path.join(BRIEF_DIR, f"{ad}.json")
    db = _db()
    stamp = {"sessions": _sessions_for(db, ad),
             "scenes_status": (db.execute("SELECT scenes_status FROM ads WHERE ad=?", (ad,)).fetchone() or [None])[0],
             "db_built": (db.execute("SELECT value FROM meta WHERE key='built'").fetchone() or [None])[0]}
    if os.path.exists(p) and not regenerate:
        d = json.load(open(p, encoding="utf-8"))
        if d.get("stamp") == stamp:          # same viewers, same scenes, same database build
            d["cached"] = True
            return d
    res = ask(BRIEF_PROMPT.format(ad=ad), context={"ad": ad}, model=model)
    res["ad"], res["created"], res["stamp"] = ad, dt.datetime.now().isoformat(timespec="seconds"), stamp
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    res["cached"] = False
    return res


# ----------------------------------------------------------------------------- demo reliability
def suggest(ad):
    """Questions offered in the Ask panel, built from this ad's actual findings."""
    import insights
    d = insights.build(ad, polish=False)
    brand = (d.get("audit") or {}).get("brand") or "the brand"
    qs = ["What held viewers in this ad?", "What should we test changing?", f"How did the {brand} brand moment land?"]
    ideas = [i for i in d.get("ideas", []) if i.get("scene") is not None]
    if ideas:
        qs.append(f"What happened in the scene \u201c{ideas[0]['label']}\u201d?")
    qs.append("How did this viewer respond to this ad compared with the other ads they watched?")
    qs.append("How sure can we be about these findings?")
    return qs


def _qa_stamp(db, ad):
    return {"sessions": _sessions_for(db, ad),
            "scenes_status": (db.execute("SELECT scenes_status FROM ads WHERE ad=?", (ad,)).fetchone() or [None])[0],
            "db_built": (db.execute("SELECT value FROM meta WHERE key='built'").fetchone() or [None])[0]}


def _offline_answer(ad, why):
    import insights
    d = insights.build(ad, polish=False)
    if d.get("error"):
        return None
    a, lines = d["audit"], []
    if a.get("first_brand_s") is not None:
        lines.append(f"The {a['brand']} brand first appears at [t={a['first_brand_s']}], {a['first_brand_pct']}% into the ad.")
    if d["strengths"]:
        lines.append("What held viewers: " + " ".join(f"{i['text']} {i['evidence']}" for i in d["strengths"][:2]))
    if d["ideas"]:
        lines.append("Ideas to test: " + " ".join(f"{i['label']} [t={i['t']}] - {i['evidence']} {i['idea']}" for i in d["ideas"][:2]))
    if not d["strengths"] and not d["ideas"]:
        lines.append("There is no data-backed finding for this ad yet.")
    lines.append(f"This rests on {len(d['viewers'])} viewer(s), so treat it as exploratory.")
    return {"answer": "\n".join(lines), "offline": True, "offline_reason": why, "tool_calls": [], "actions": [],
            "unverified_numbers": [], "seconds": 0}


def answer(question, history=None, context=None):
    """Ask with (1) saved answers for the suggested questions and (2) an offline fallback."""
    ad = (context or {}).get("ad")
    session = (context or {}).get("session")
    cacheable = bool(ad) and not history and not session and question in suggest(ad)
    p = os.path.join(BRIEF_DIR, f"{ad}.qa.json") if ad else None
    store = {}
    if cacheable:
        db = _db()
        stamp = _qa_stamp(db, ad)
        try:
            store = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            store = {}
        if store.get("stamp") != stamp:
            store = {"stamp": stamp, "qa": {}}
        hit = store["qa"].get(question)
        if hit:
            return dict(hit, cached=True)
    try:
        res = ask(question, history=history, context=context)
    except Exception as e:
        off = _offline_answer(ad, f"{type(e).__name__}") if ad else None
        if off:
            return off
        raise
    if cacheable and not res.get("unverified_numbers"):
        os.makedirs(BRIEF_DIR, exist_ok=True)
        store["qa"][question] = {k: res.get(k) for k in ("answer", "tool_calls", "actions", "unverified_numbers", "seconds", "model")}
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=1, ensure_ascii=False)
    return res


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Did people notice the Budweiser logo?"
    out = ask(q, context={"ad": "ad04"}, log=False)
    print(out["answer"])
    print("\n-- tools:", [c["tool"] for c in out["tool_calls"]], "| actions:", out["actions"],
          "| unverified numbers:", out["unverified_numbers"], "|", out["model"], out["seconds"], "s")
