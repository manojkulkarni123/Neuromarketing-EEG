"""
Session configuration shared by the setup page (app/server.py) and the
recording task (ad_viewing_task.py): the question bank and the per-session
config file.

Question bank: data/lab_session/question_bank.json (created with the defaults
below on first use; custom questions added from the setup page are appended).
  type "scale9" -> answered with number keys 1-9 (anchors explain the ends)
  type "yesno"  -> answered with Y / N, stored as 1 / 0
  type "brand_choice" -> "Which brand was this ad for?" -- asked at the END of the
                     session (after every viewing, so it can't point the viewer at
                     the brand before a repeat viewing), with a still frame from the
                     ad; options = the brand + 3 competitors from
                     data/lab_session/ad_profiles/<ad>.json, shuffled, plus "not sure".
                     Stored as 1 (correct) / 0 (wrong or not sure) + the choice made.

Session config: data/lab_session/sessions/<pid>.config.json
  {
    "participant_id": "p01",
    "order": "random" | "fixed",        # random = shuffled, seeded by the id (reproducible)
    "viewings": 1 | 2,                  # 2 = every ad is shown twice (second block reshuffled);
                                        # questions only after the first viewing
    "baseline": true,
    "ads": [{"file": "ad04.mp4", "questions": ["liking", "purchase_intent"]}, ...],
    "questions": {id: {...}},           # snapshot of every question used -> self-describing
    "notes": "", "created": "..."
  }
"""
import datetime as dt
import json
import os
import re

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(PROJ, "data", "lab_session")
ADS_DIR = os.path.join(LAB, "ads")
SESS_DIR = os.path.join(LAB, "sessions")
BANK_PATH = os.path.join(LAB, "question_bank.json")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

DEFAULT_BANK = [
    {"id": "liking", "text": "How much did you like this ad?", "type": "scale9",
     "anchors": "1 = not at all   ...   9 = very much"},
    {"id": "purchase_intent", "text": "How likely would you be to buy this product?", "type": "scale9",
     "anchors": "1 = very unlikely   ...   9 = very likely"},
    {"id": "brand_impression", "text": "What is your overall impression of this brand?", "type": "scale9",
     "anchors": "1 = very negative   ...   9 = very positive"},
    # Self-Assessment Manikin style (Bradley & Lang 1994): the viewer's own report of emotional
    # direction (valence) and intensity (arousal) -- set beside Approach and Engagement from the EEG.
    {"id": "feeling_valence", "text": "How did this ad make you feel?", "type": "scale9",
     "anchors": "1 = very unpleasant   ...   9 = very pleasant"},
    {"id": "feeling_arousal", "text": "How calm or excited did this ad make you feel?", "type": "scale9",
     "anchors": "1 = very calm   ...   9 = very excited"},
    {"id": "seen_before", "text": "Had you seen this ad before today?", "type": "yesno",
     "anchors": "Y = yes        N = no"},
    {"id": "brand_recall", "text": "Which brand was this ad for?", "type": "brand_choice",
     "anchors": "press the number of your answer"},
]
DEFAULT_QUESTIONS = ["liking", "purchase_intent", "feeling_valence", "feeling_arousal", "seen_before"]
QTYPES = ("scale9", "yesno", "brand_choice")
PROFILES_DIR = os.path.join(LAB, "ad_profiles")


def load_bank():
    if not os.path.exists(BANK_PATH):
        save_bank(DEFAULT_BANK)
    with open(BANK_PATH, encoding="utf-8") as fh:
        bank = json.load(fh)
    # banks created before a default question existed get it added (never removes or edits anything)
    have = {q["id"] for q in bank}
    missing = [q for q in DEFAULT_BANK if q["id"] not in have]
    if missing:
        bank += missing
        save_bank(bank)
    return bank


def load_profile(ad):
    """data/lab_session/ad_profiles/<ad>.json (ad = file name without extension), or None."""
    p = os.path.join(PROFILES_DIR, f"{os.path.splitext(ad)[0]}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def recall_frame_time(ad):
    """Still frame shown with the brand recall question: the profile's recall_frame_s,
    else the middle of the longest stretch with no brand event at all (from the
    labeller's scenes file), else 30 % into the ad. Returns (seconds, how)."""
    ad = os.path.splitext(ad)[0]
    prof = load_profile(ad) or {}
    if prof.get("recall_frame_s") is not None:
        return float(prof["recall_frame_s"]), "set in profile"
    sp = os.path.join(LAB, "scenes", f"{ad}.scenes.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            d = json.load(fh)
        dur = d["duration_s"]
        ev = sorted((b["start_s"], b["end_s"]) for b in d.get("brand_events", []))
        gaps, t = [], 0.0
        for a, b in ev:
            if a > t:
                gaps.append((t, a))
            t = max(t, b)
        if t < dur:
            gaps.append((t, dur))
        gaps = [g for g in gaps if g[1] - g[0] >= 1.0]
        if gaps:
            a, b = max(gaps, key=lambda g: g[1] - g[0])
            return round((a + b) / 2, 2), "longest stretch without the brand"
        return round(0.3 * dur, 2), "no brand-free stretch found -- brand may be visible"
    return None, "no scenes file"


def save_bank(bank):
    os.makedirs(LAB, exist_ok=True)
    with open(BANK_PATH, "w", encoding="utf-8") as fh:
        json.dump(bank, fh, indent=1, ensure_ascii=False)


def add_question(text, qtype="scale9", anchors=""):
    text = (text or "").strip()
    if not text:
        raise ValueError("question text is empty")
    if qtype not in ("scale9", "yesno"):
        raise ValueError("type must be scale9 or yesno (brand recall is built in)")
    bank = load_bank()
    base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:32] or "q"
    qid, k = base, 2
    while any(q["id"] == qid for q in bank):
        qid, k = f"{base}_{k}", k + 1
    if not anchors:
        anchors = "Y = yes        N = no" if qtype == "yesno" else "1 = not at all   ...   9 = very much"
    q = {"id": qid, "text": text, "type": qtype, "anchors": anchors, "custom": True}
    bank.append(q)
    save_bank(bank)
    return q


def list_ads():
    if not os.path.isdir(ADS_DIR):
        return []
    return sorted(f for f in os.listdir(ADS_DIR) if f.lower().endswith(VIDEO_EXTS))


def config_path(pid):
    return os.path.join(SESS_DIR, f"{pid}.config.json")


def session_exists(pid):
    base = os.path.join(LAB, pid)
    return any(os.path.exists(base + s) for s in ("_raw.fif", "_raw.npz", "_ratings.csv"))


def validate(cfg):
    """Return a list of problems (empty = OK)."""
    probs = []
    pid = str(cfg.get("participant_id", ""))
    if not ID_RE.match(pid):
        probs.append("participant id: use only letters, digits, _ or - (e.g. p01)")
    if cfg.get("order") not in ("random", "fixed"):
        probs.append("order must be random or fixed")
    if cfg.get("viewings", 1) not in (1, 2):
        probs.append("viewings must be 1 or 2")
    ads = cfg.get("ads") or []
    if not ads:
        probs.append("choose at least one ad")
    have = set(list_ads())
    bank = {q["id"]: q for q in load_bank()}
    for a in ads:
        f = a.get("file", "")
        if f not in have:
            probs.append(f"ad file not found: {f}")
        if any(c in f for c in " ()[]"):
            probs.append(f"rename {f}: spaces/brackets break PsychoPy playback")
        if not a.get("questions"):
            probs.append(f"{f}: choose at least one question")
        for q in a.get("questions", []):
            if q not in bank:
                probs.append(f"{f}: unknown question '{q}'")
            elif bank[q]["type"] == "brand_choice":
                prof = load_profile(f) or {}
                opts = prof.get("recall_options") or []
                if not prof.get("brand") or prof.get("brand") not in opts or len(opts) < 2:
                    probs.append(f"{f}: brand recall needs data/lab_session/ad_profiles/"
                                 f"{os.path.splitext(f)[0]}.json with 'brand' and 'recall_options'")
    return probs


def save_config(cfg):
    probs = validate(cfg)
    if probs:
        raise ValueError("; ".join(probs))
    bank = {q["id"]: q for q in load_bank()}
    used = sorted({q for a in cfg["ads"] for q in a["questions"]})
    cfg = dict(cfg)
    cfg["questions"] = {q: bank[q] for q in used}
    cfg["created"] = dt.datetime.now().isoformat(timespec="seconds")
    os.makedirs(SESS_DIR, exist_ok=True)
    with open(config_path(cfg["participant_id"]), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1, ensure_ascii=False)
    return config_path(cfg["participant_id"])


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    probs = validate(cfg)
    if probs:
        raise SystemExit("Session config problems:\n  " + "\n  ".join(probs))
    return cfg
