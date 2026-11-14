"""
Insights for one ad = what the AI saw in the ad + what viewers' brains did, with the evidence.

  audit     facts about the ad itself, from the approved labels (no EEG): when the brand first
            appears / is first said, how much of the ad shows the brand, speech, scene count.
  scenes    per scene and measure: each clean viewing's value relative to that viewing's own
            typical scene (vs_ad_median), combined over viewers AND repeat viewings:
              direction  up / down / mixed
              agree      k of m clean viewings in that direction
              level      "consistent"  all m agree, m >= 3
                         "repeated"    all m agree, m == 2
                         "mixed"       viewings disagree
                         "single"      one clean viewing
                         "none"        no clean data
  findings  strengths and ideas-to-test. A finding exists ONLY when the data flags it:
              - a scene is consistent/repeated low or high on visual attention or engagement
                or approach (emotional direction); mental effort is reported but low effort
                 never raises an idea on its own
              - the brand-moment test is confirmed
              - the audit shows the brand arriving in the second half of the ad
            Guards: scenes whose midpoint is in the first 2 s are skipped (eyes settling
            after the fixation cross); noisy viewings never count; a sharp brightness or
            loudness change is attached as a caution.
  ideas     wording of "idea to test" comes from fixed templates; optionally re-phrased once
            per ad by the language model (cached). The model gets no numbers and its text is
            rejected if it contains any digit, so it cannot invent data.
"""
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
LAB = os.path.join(PROJ, "data", "lab_session")
DB_PATH = os.path.join(LAB, "neuro.db")
CACHE_DIR = os.path.join(LAB, "briefs")
SKIP_START_S = 2.0
NOISY_MAX = 0.3

NAMES = {"attention": "Visual attention", "alertness": "Engagement", "effort": "Mental effort",
         "wanting": "Approach"}
GUIDE = {
    "attention": {"high": "eyes and brain locked on the screen", "low": "looking without taking much in",
                  "note": "Brightness and fast cuts also move it -- compare scenes within the same ad."},
    "alertness": {"high": "alert and activated", "low": "relaxed or passive",
                  "note": "Jaw or forehead tension can inflate it on a dry cap."},
    "effort": {"high": "working to follow the message", "low": "easy to follow, or tuned out",
               "note": "Exploratory: needs many viewers for a firm reading."},
    "wanting": {"high": "leaning toward the content", "low": "pulling away",
                "note": "Emotional direction (valence). Noisier than the other measures, so it counts only "
                        "when viewings agree. Needs a working left/right frontal electrode pair."},
}
TRIGGER = ("attention", "alertness", "wanting")          # measures that can raise a finding on their own
VISUAL = ("logo", "product", "brand_text", "tagline")


def _db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def audit(db, ad):
    a = db.execute("SELECT * FROM ads WHERE ad=?", (ad,)).fetchone()
    dur = a["duration_s"]
    ev = [dict(r) for r in db.execute("SELECT * FROM brand_events WHERE ad=? ORDER BY start_s", (ad,))]
    vis = [e for e in ev if e["kind"] in VISUAL]
    named = [e for e in vis if e["kind"] in ("logo", "brand_text", "tagline")]     # brand identifiable
    spoken = [e for e in ev if e["kind"] == "spoken_brand"]
    cover, t = 0.0, 0.0
    for e in sorted(vis, key=lambda e: e["start_s"]):
        s, en = max(e["start_s"], t), e["end_s"]
        if en > s:
            cover += en - s
            t = en
    tr = [dict(r) for r in db.execute("SELECT * FROM transcript WHERE ad=? ORDER BY start_s", (ad,))]
    speech = [s for s in tr if s["kind"] == "speech"]
    n_sc = db.execute("SELECT COUNT(*) FROM scenes WHERE ad=?", (ad,)).fetchone()[0]
    first_named = named[0]["start_s"] if named else None
    return {
        "duration_s": round(dur, 1), "scenes": n_sc,
        "brand": a["brand"],
        "first_brand_s": None if first_named is None else round(first_named, 1),
        "first_brand_pct": None if first_named is None else round(100 * first_named / dur),
        "first_brand_what": named[0]["what"] if named else None,
        "first_product_s": round(min((e["start_s"] for e in vis if e["kind"] == "product"), default=-1), 1)
        if any(e["kind"] == "product" for e in vis) else None,
        "first_spoken_s": round(spoken[0]["start_s"], 1) if spoken else None,
        "times_spoken": len(spoken),
        "brand_on_screen_pct": round(100 * cover / dur),
        "first_speech_s": round(speech[0]["start_s"], 1) if speech else None,
        "scenes_status": a["scenes_status"],
    }


def _viewings(db, ad):
    """{(session, viewing): {scene_idx: {measure: (vs_ad_median or None, verdict)}}}"""
    out = {}
    try:
        with open(os.path.join(LAB, "viewers.json"), encoding="utf-8") as fh:
            ex = {k for k, v in json.load(fh).items() if v.get("excluded")}
    except (OSError, ValueError):
        ex = set()
    for r in db.execute("SELECT * FROM scene_values WHERE ad=?", (ad,)):
        if r["session"] in ex:
            continue
        out.setdefault((r["session"], 1), {}).setdefault(r["scene_idx"], {})[r["measure"]] = (r["vs_ad_median"], r["verdict"])
    try:
        for r in db.execute("SELECT * FROM scene_values_repeat WHERE ad=?", (ad,)):
            if r["session"] in ex:
                continue
            out.setdefault((r["session"], r["viewing"]), {}).setdefault(r["scene_idx"], {})[r["measure"]] = (r["vs_ad_median"], r["verdict"])
    except sqlite3.OperationalError:
        pass
    return out


def _combine(vals):
    """vals: vs_ad_median of each clean viewing -> direction / agreement / level / mean."""
    m = len(vals)
    if m == 0:
        return {"level": "none", "direction": None, "agree": 0, "of": 0, "mean": None}
    # a value of exactly 0 is the ad's typical (median) scene itself: it votes neither way,
    # so it can never make a scene "consistent" or "repeated"
    up = sum(1 for v in vals if v > 1e-9)
    down = sum(1 for v in vals if v < -1e-9)
    k, direction = (up, "up") if up >= down else (down, "down")
    if k == 0:
        direction = None
    if m == 1:
        level = "single"
    elif k == m:
        level = "consistent" if m >= 3 else "repeated"
    else:
        level, direction = "mixed", ("up" if up > down else "down" if down > up else None)
    return {"level": level, "direction": direction, "agree": k, "of": m,
            "mean": round(sum(vals) / m, 2)}


def scenes(db, ad):
    views = _viewings(db, ad)
    rows = [dict(r) for r in db.execute("SELECT * FROM scenes WHERE ad=? ORDER BY idx", (ad,))]
    ev = [dict(r) for r in db.execute("SELECT * FROM brand_events WHERE ad=?", (ad,))]
    out, prev_b = [], None
    for s in rows:
        brand = sorted({e["kind"] for e in ev if e["start_s"] < s["end_s"] and e["end_s"] > s["start_s"]})
        cautions = []
        try:
            if json.loads(s["loud_jumps"] or "[]"):
                cautions.append("sudden loud sound")
        except ValueError:
            pass
        if prev_b is not None and s["brightness"] is not None and abs(s["brightness"] - prev_b) > 40:
            cautions.append("sharp brightness change")
        prev_b = s["brightness"] if s["brightness"] is not None else prev_b
        meas = {}
        for mkey in NAMES:
            vals = [v[s["idx"]][mkey][0] for v in views.values()
                    if s["idx"] in v and mkey in v[s["idx"]] and v[s["idx"]][mkey][0] is not None]
            meas[mkey] = _combine(vals)
        out.append({"idx": s["idx"], "label": s["label"], "role": s["story_role"],
                    "start_s": round(s["start_s"], 2), "end_s": round(s["end_s"], 2),
                    "description": s["description"], "heard": s["heard"],
                    "brand": brand, "brand_visible": any(b in VISUAL for b in brand),
                    "skip": (s["start_s"] + s["end_s"]) / 2 < SKIP_START_S,
                    "cautions": cautions, "measures": meas})
    viewers = sorted({k[0] for k in views})
    return out, {"viewers": viewers, "viewings": len(views)}


def _evidence(c, mkey):
    unit = {"attention": "points", "alertness": "points", "effort": "dB", "wanting": ""}[mkey]
    word = "above" if c["direction"] == "up" else "below"
    who = (f"{c['agree']} of {c['of']} clean viewings" if c["level"] == "consistent"
           else "both clean viewings")
    mag = f"{abs(c['mean']):.1f} {unit}".strip()
    return f"{NAMES[mkey]} was {word} this ad's typical scene in {who} (average {mag})."


def findings(db, ad, au, sc):
    strengths, ideas = [], []
    for s in sc:
        if s["skip"]:
            continue
        for mkey, c in s["measures"].items():
            if c["level"] not in ("consistent", "repeated"):
                continue
            item = {"scene": s["idx"], "label": s["label"], "t": s["start_s"], "end": s["end_s"],
                    "measure": mkey, "measure_name": NAMES[mkey], "level": c["level"],
                    "evidence": _evidence(c, mkey), "cautions": s["cautions"],
                    "experimental": False, "brand_visible": s["brand_visible"]}
            if c["direction"] == "up":
                item["text"] = (f"“{s['label']}” drew viewers in (approach rose)." if mkey == "wanting"
                                else f"“{s['label']}” held {NAMES[mkey].lower()}.")
                if mkey in TRIGGER:
                    strengths.append(item)
            else:
                if mkey not in TRIGGER:
                    continue                         # low mental effort is not a problem by itself
                low = "viewers pulled away (approach fell)" if mkey == "wanting" else f"{NAMES[mkey].lower()} is low"
                if s["brand_visible"]:
                    item["kind"] = "brand_low"
                    item["idea"] = (f"The brand is on screen while {low}. "
                                    f"Idea to test: show the brand at a moment that holds viewers better, or make it more prominent here.")
                else:
                    item["kind"] = "scene_low"
                    item["idea"] = (f"Idea to test: change the tone of “{s['label']}” and check whether viewers still pull away."
                                    if mkey == "wanting" else
                                    f"Idea to test: shorten or re-pace “{s['label']}” and check whether "
                                    f"{NAMES[mkey].lower()} holds.")
                ideas.append(item)
    try:
        for r in db.execute("SELECT session, status, diff FROM brand_test WHERE ad=? AND measure='attention' AND viewing=1", (ad,)):
            if r["status"] == "confirmed for this viewer":
                up = r["diff"] > 0
                strengths.append({
                    "scene": None, "label": "Brand moment", "t": au["first_brand_s"], "measure": "attention",
                    "measure_name": NAMES["attention"], "level": "repeated", "experimental": False, "cautions": [],
                    "text": f"Visual attention {'rose' if up else 'fell'} while the brand was on screen.",
                    "evidence": f"Viewer {r['session']}: brand-on-screen vs rest of the ad, confirmed on the second viewing "
                                f"({abs(r['diff']):.1f} points {'higher' if up else 'lower'})."})
    except sqlite3.OperationalError:
        pass
    if au["first_brand_pct"] is not None and au["first_brand_pct"] >= 50:
        ideas.append({"scene": None, "label": "Brand timing", "t": au["first_brand_s"], "kind": "late_brand",
                      "measure": None, "measure_name": "Ad structure", "level": "fact", "experimental": False, "cautions": [],
                      "evidence": f"The {au['brand']} name or logo first appears at {au['first_brand_s']} s "
                                  f"({au['first_brand_pct']}% into the ad).",
                      "idea": "Idea to test: a cut with an earlier brand cue, compared with this one on brand recall."})
    return strengths, ideas


def _polish(ad, sc, ideas, stamp):
    """Optional: let the language model re-phrase each idea using the scene description. One call per ad,
    cached. Falls back to the template wording on any problem."""
    if not ideas:
        return ideas
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, f"{ad}.ideas.json")
    key = json.dumps([stamp, [(i["scene"], i.get("kind"), i["measure"]) for i in ideas]], sort_keys=True)
    try:
        if os.path.exists(p):
            c = json.load(open(p, encoding="utf-8"))
            if c.get("key") == key:
                for i, t in zip(ideas, c["texts"]):
                    i["idea"], i["ai_worded"] = t, True
                return ideas
        sys.path.insert(0, HERE)
        import copilot
        from openai import OpenAI
        copilot._ensure_key()
        by = {s["idx"]: s for s in sc}
        payload = [{"n": k, "problem": i.get("kind"), "measure": i["measure_name"],
                    "scene": by[i["scene"]]["label"] if i["scene"] is not None else None,
                    "on_screen": by[i["scene"]]["description"] if i["scene"] is not None else None,
                    "heard": by[i["scene"]]["heard"] if i["scene"] is not None else None,
                    "template": i["idea"]} for k, i in enumerate(ideas)]
        r = OpenAI().responses.create(
            model=copilot.MODEL, store=False, reasoning={"effort": "low"},
            instructions=("You help an ad-testing team. For each item, write ONE sentence starting with 'Idea to test:' "
                          "that proposes a concrete, testable change to that scene of the ad, using what is on screen "
                          "and heard. It is a hypothesis to test, never a verdict. Do NOT use any numbers or digits. "
                          "Do not claim the change will work. Max 30 words. Reply as JSON: {\"ideas\": [strings in the same order]}."),
            input=json.dumps(payload, ensure_ascii=False))
        txt = r.output_text.strip()
        txt = txt[txt.index("{"): txt.rindex("}") + 1]
        texts = json.loads(txt)["ideas"]
        ok = (len(texts) == len(ideas) and all(isinstance(t, str) and t.startswith("Idea to test:")
                                                and not any(ch.isdigit() for ch in t) and len(t) < 260 for t in texts))
        if ok:
            json.dump({"key": key, "texts": texts}, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            for i, t in zip(ideas, texts):
                i["idea"], i["ai_worded"] = t, True
    except Exception as e:                                    # no key, no credit, offline, bad JSON -> templates stay
        print(f"[insights] idea wording kept as template ({type(e).__name__}: {e})")
    return ideas


def build(ad, polish=True):
    if not os.path.exists(DB_PATH):
        return {"error": "no database yet"}
    db = _db()
    if not db.execute("SELECT 1 FROM ads WHERE ad=?", (ad,)).fetchone():
        return {"error": f"unknown ad {ad}"}
    au = audit(db, ad)
    sc, who = scenes(db, ad)
    strengths, ideas = findings(db, ad, au, sc)
    built = (db.execute("SELECT value FROM meta WHERE key='built'").fetchone() or [None])[0]
    if polish:
        ideas = _polish(ad, sc, ideas, [built, who["viewings"], au["scenes_status"]])
    order = {"consistent": 0, "repeated": 1, "fact": 2}
    strengths.sort(key=lambda i: (order.get(i["level"], 9), i["experimental"]))
    ideas.sort(key=lambda i: (order.get(i["level"], 9), i["experimental"]))
    return {"ad": ad, "audit": au, "scenes": sc, **who, "strengths": strengths[:4], "ideas": ideas[:4],
            "names": NAMES, "guide": GUIDE,
            "rules": "A finding appears only when every clean viewing of that scene agrees (at least two). "
                     "The first 2 s of an ad and noisy viewings are ignored. Ideas are hypotheses to test, not verdicts."}


if __name__ == "__main__":
    d = build(sys.argv[1] if len(sys.argv) > 1 else "ad06", polish="--polish" in sys.argv)
    print(json.dumps({k: d[k] for k in ("audit", "viewers", "viewings", "strengths", "ideas")}, indent=1, ensure_ascii=False))
    for s in d["scenes"]:
        print(s["idx"], s["label"][:30], {m: (c["level"], c["direction"], f"{c['agree']}/{c['of']}") for m, c in s["measures"].items()})
