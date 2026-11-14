"""
PsychoPy ad-viewing task -- plays short video ads and asks two 1-9 questions
after each one (liking the ad, likelihood of buying the product), while
recording EEG itself: it pulls
EEGStudio's LSL *Outlet* stream (turn that on in EEGStudio's Add-on tab
first) via lsl_recorder.py, and stamps AD_START/AD_END markers on the same
LSL clock. Saves one <participant>_raw.fif with EEG + markers already aligned.
(EEGStudio's LSL *Inlet* direction never connected, so we don't use it.)

Drop real ad video files (.mp4/.avi/.mov) into data/lab_session/ads/ before
running. If that folder is empty, runs with placeholder "ads" (a plain
colored screen shown for a fixed duration) instead, so the whole
task/marker/rating/logging flow can be tested today without real ad files --
swap in real videos later and nothing else changes.

Usage:
    .venv\\Scripts\\python.exe lab_task_psychopy\\ad_viewing_task.py --id p01
      --config FILE  session config from the setup page (app/server.py -> /setup):
                     which ads, fixed/random order, which questions after each ad,
                     baseline on/off. Without it: every ad in the folder, random
                     order, liking + purchase intent.
      --one          quick test: one ad only
      --no-baseline  skip the baseline block
      --overwrite    allow replacing an existing session with the same id
      --auto         TEST ONLY: no key presses needed (random answers) -- for
                     checking the recording/frame-log chain without a participant

Outputs (data/lab_session/): <id>_raw.fif (+ .npz backup), <id>_ratings.csv,
<id>_framelog.csv (screen time of every video frame, on the EEG time axis),
<id>_session.json (config used + order shown + playback stats).

Known harmless noise: after each real video ad finishes, PsychoPy may print
an "Exception ignored in: MovieStim.__del__ ... AttributeError: no attribute
'_freePlayer'" during garbage collection. Verified this does not stop the
script, affect timing, or corrupt output (tested: a 3.00s clip measured as
3.02s end-to-end, markers fired correctly) -- it's a cosmetic bug in this
PsychoPy version's cleanup code, not a real error.
"""
import os
import sys
import csv
import json
import time
import random

from psychopy import visual, core, event

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lsl_recorder import LSLRecorder
import session_config as SC

PARTICIPANT_ID = None  # set from --id in main(); also seeds the per-participant ad order
AUTO = False           # --auto: test mode, no key presses
N_PLACEHOLDER_ADS = 4
PLACEHOLDER_DURATION_S = 10.0

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADS_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session", "ads")
BASELINE_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session", "baseline")  # one neutral video, e.g. nature footage
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session")

# Baseline block, run before the ads (skip with --no-baseline for quick tests).
# Every EEG measure is later expressed relative to the person's own baseline --
# raw dry-cap values aren't comparable between people. Marker pairs:
#   BASELINE_EO_*      eyes open, fixation cross  -> reference for posterior alpha
#   BASELINE_EC_*      eyes closed                -> cap check: alpha should jump vs EO
#   BASELINE_NEUTRAL_* neutral video (if present) -> reference for watching-video measures
BASELINE_EO_S = 60
BASELINE_EC_S = 60

# Padding around every ad (not used as a reference in the analysis -- just time):
#   PRE_AD_S   fixation cross before the ad, so movement from the previous
#              rating keypress settles before the ad starts (in test1 it
#              contaminated the first ~5 s of the ad: every band jumped at once).
#   POST_AD_S  gray screen after the ad, before the questions, so the brain's
#              (slightly delayed) reaction to the ending/brand logo is recorded
#              instead of landing on the rating screen.
# Markers per ad: PRE_AD_START:<name>, AD_START:<name>, AD_END:<name>, POST_AD_END:<name>
PRE_AD_S = 4
POST_AD_S = 3

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


DISPLAY_FRACTION = 0.9  # every ad is scaled to fill this much of the screen, whatever its native resolution


def find_ads():
    """Shuffled per participant (seeded by PARTICIPANT_ID, so a session is
    reproducible), so order/fatigue effects don't systematically hit the same
    ad for everyone. The order shown is logged in the ratings CSV."""
    if not os.path.isdir(ADS_DIR):
        return []
    ads = sorted(os.path.join(ADS_DIR, f) for f in os.listdir(ADS_DIR) if f.lower().endswith(VIDEO_EXTS))
    # PsychoPy's video player fails to decode files whose names contain spaces or
    # brackets (verified: a WhatsApp "... PM (1).mp4" failed, the same file renamed played fine).
    bad = [os.path.basename(a) for a in ads if any(c in os.path.basename(a) for c in " ()[]")]
    if bad:
        raise RuntimeError(f"Rename these ad files to simple names (e.g. ad06.mp4) -- spaces/brackets "
                           f"break video playback: {bad}")
    random.Random(PARTICIPANT_ID).shuffle(ads)
    if "--one" in sys.argv:  # quick test: just the first ad in this participant's order
        return ads[:1]
    return ads


def display_size(win, path):
    """Same on-screen size for every ad (native resolutions range 384x288 to
    1280x720 -- showing them at native size would make screen size a confound
    for engagement). Fits DISPLAY_FRACTION of the window, keeping aspect ratio.
    Also returns the file's frame rate (for the frame log)."""
    import cv2
    cap = cv2.VideoCapture(path)
    vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    win_w, win_h = win.size
    scale = DISPLAY_FRACTION * min(win_w / vw, win_h / vh)
    return (2 * vw * scale / win_w, 2 * vh * scale / win_h), vfps  # norm units, fps


def push(recorder, label):
    recorder.mark(label)
    print(f"[marker] {label}")


START_ATTEMPTS = 10


def viewing_name(path, viewing=1):
    """Name used in markers and the frame log. Viewing 1 keeps the plain file name
    (so every older recording and tool still works); a repeat viewing is
    'ad06.mp4#2'."""
    name = os.path.basename(path)
    return name if viewing == 1 else f"{name}#{viewing}"


def play_real_ad(win, rec, path, start_label=None, end_label=None, viewing=1):
    """PsychoPy's player intermittently fails to deliver the first frame of
    these H.264 files (~50% of attempts, no pattern by path/size/warm-up), so
    retry the start. The start marker is sent only once the first frame is
    actually on screen, so a retry never shifts it away from true onset."""
    name = viewing_name(path, viewing)
    start_label = start_label or f"AD_START:{name}"
    end_label = end_label or f"AD_END:{name}"
    size, vfps = display_size(win, path)
    for attempt in range(1, START_ATTEMPTS + 1):
        mov = visual.MovieStim(win, filename=path, noAudio=False, size=size, units="norm")
        # Root cause of the flaky starts (2026-11-13): PsychoPy's reader waits only
        # ~one frame interval (41 polls x 1 ms) for the FIRST decoded frame, then
        # raises -- and a fresh MovieStim faces the same limit, so under load every
        # retry failed. Allow up to ~3 s for the first frame only, then restore the
        # normal limit. Any start delay this causes is captured by the frame log.
        reader = getattr(mov, "_player", None)
        default_attempts = getattr(reader, "_maxGetFrameAttempts", None)
        if default_attempts is not None:
            reader._maxGetFrameAttempts = 3000
        try:
            mov.draw()
            win.flip()
            if default_attempts is not None:
                reader._maxGetFrameAttempts = default_attempts
            break
        except RuntimeError:
            mov.pause()
            core.wait(0.2)
    else:
        raise RuntimeError(f"Could not start {name} after {START_ATTEMPTS} attempts.")
    # Frame log: after every flip, which frame (by its presentation timestamp)
    # is now on screen and when (LSL clock, read right after the flip returns at
    # the screen refresh). MovieStim.frameIndex is a stub (always 0) in this
    # PsychoPy version, so the frame is identified by mov.pts instead.
    rec.log_frame(name, mov.pts, vfps)
    if attempt > 1:
        print(f"[video] {name} started on attempt {attempt}")
    push(rec, start_label)
    t0 = time.perf_counter()
    dropped = 0
    while not mov.isFinished:
        try:
            mov.draw()
        except RuntimeError:
            dropped += 1  # a missed frame mid-ad: keep showing the last one rather than crashing
        win.flip()
        rec.log_frame(name, mov.pts, vfps)
        if "escape" in event.getKeys():
            core.quit()
    if dropped:
        print(f"[video] {name}: {dropped} frame-fetch failures during playback")
    duration = time.perf_counter() - t0
    push(rec, end_label)
    return name, duration


def _show_until_space(win, text):
    visual.TextStim(win, text=text, color="white", height=0.06, wrapWidth=1.8).draw()
    win.flip()
    if AUTO:
        core.wait(0.5)
        return
    keys = event.waitKeys(keyList=["space", "escape"])
    if "escape" in keys:
        core.quit()


def _timed_block(win, rec, label, seconds, stim=None):
    push(rec, f"{label}_START")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        if stim is not None:
            stim.draw()
        win.flip()
        if "escape" in event.getKeys():
            core.quit()
    push(rec, f"{label}_END")


def _beep():
    try:
        import winsound
        winsound.Beep(880, 400)
    except Exception:
        pass


def run_baseline(win, rec):
    fixation = visual.TextStim(win, text="+", color="white", height=0.15)
    _show_until_space(win, "First, a short resting recording.\n\n"
                           f"Look at the cross in the middle of the screen for {BASELINE_EO_S} seconds.\n"
                           "Stay relaxed and still, and try not to blink too much.\n\nPress SPACE to start.")
    _timed_block(win, rec, "BASELINE_EO", BASELINE_EO_S, fixation)

    _show_until_space(win, f"Now close your eyes and relax for {BASELINE_EC_S} seconds.\n\n"
                           "Open them when you hear the beep.\n\nPress SPACE, then close your eyes.")
    _timed_block(win, rec, "BASELINE_EC", BASELINE_EC_S)
    _beep()

    neutral = sorted(f for f in os.listdir(BASELINE_DIR) if f.lower().endswith(VIDEO_EXTS)) \
        if os.path.isdir(BASELINE_DIR) else []
    if neutral:
        _show_until_space(win, "Open your eyes.\n\nNext, just watch a short calm video.\n\nPress SPACE to start.")
        play_real_ad(win, rec, os.path.join(BASELINE_DIR, neutral[0]),
                     start_label="BASELINE_NEUTRAL_START", end_label="BASELINE_NEUTRAL_END")
    else:
        print(f"[baseline] no neutral video in {BASELINE_DIR} -- skipping that part "
              f"(analysis will fall back to the eyes-open baseline)")


def pre_ad_fixation(win, rec, name):
    push(rec, f"PRE_AD_START:{name}")
    cross = visual.TextStim(win, text="+", color="white", height=0.15)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < PRE_AD_S:
        cross.draw()
        win.flip()
        if "escape" in event.getKeys():
            core.quit()


def post_ad_gray(win, rec, name):
    gray = visual.Rect(win, width=2, height=2, fillColor=(0.0, 0.0, 0.0), lineWidth=0)  # mid-gray in PsychoPy's -1..1 color space
    cross = visual.TextStim(win, text="+", color="white", height=0.15)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < POST_AD_S:
        gray.draw()
        cross.draw()
        win.flip()
        if "escape" in event.getKeys():
            core.quit()
    push(rec, f"POST_AD_END:{name}")


def show_ad(win, rec, play_fn, *args, **kw):
    """Fixation cross -> ad -> gray screen, around any ad player."""
    name_hint = (viewing_name(args[0], kw.get("viewing", 1)) if args and isinstance(args[0], str)
                 else f"placeholder_{args[0]}")
    pre_ad_fixation(win, rec, name_hint)
    name, duration = play_fn(win, rec, *args, **kw)
    post_ad_gray(win, rec, name)
    return name, duration


def play_placeholder_ad(win, rec, index):
    name = f"placeholder_{index}"
    rect = visual.Rect(win, width=2, height=2, fillColor=random.choice(
        ["red", "blue", "green", "orange", "purple"]))
    txt = visual.TextStim(win, text=f"Placeholder Ad {index}\n(real video would play here)",
                           color="white", height=0.08)
    push(rec, f"AD_START:{name}")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < PLACEHOLDER_DURATION_S:
        rect.draw()
        txt.draw()
        win.flip()
        if "escape" in event.getKeys():
            core.quit()
    push(rec, f"AD_END:{name}")
    return name, PLACEHOLDER_DURATION_S


# Questions come from the question bank (session_config.py); which ones follow
# which ad is set per session on the setup page. Same wording/order each time an
# ad uses them. Liking the ad and wanting the product are deliberately separate
# (often diverge). Not "how engaging was it" -- engagement is what the EEG
# measures, so rating it too would compare the brain against the participant's
# guess about their brain. Without a config: liking + purchase intent.
DEFAULT_RATING_IDS = ["liking", "purchase_intent"]
DIGITS = [str(d) for d in range(1, 10)]


def ask_rating(win, q):
    """q: question dict from the bank. scale9 -> 1..9, yesno -> 1 (Y) / 0 (N)."""
    keys = DIGITS if q["type"] == "scale9" else ["y", "n"]
    txt1 = visual.TextStim(win, text=q["text"], color="white", height=0.08, pos=(0, 0.15), wrapWidth=1.8)
    txt2 = visual.TextStim(win, text=q["anchors"], color="white", height=0.06, pos=(0, -0.05), wrapWidth=1.8)
    # Discard keys pressed earlier (during the ad, or a double-press on the
    # previous question) so they can't auto-answer this one.
    event.clearEvents()
    txt1.draw()
    txt2.draw()
    win.flip()
    t0 = time.perf_counter()
    if AUTO:
        core.wait(0.3)
        k = random.choice(keys)
        return (int(k) if q["type"] == "scale9" else int(k == "y")), time.perf_counter() - t0
    while True:
        for k in event.getKeys():
            if k == "escape":
                core.quit()
            if k in keys:
                return (int(k) if q["type"] == "scale9" else int(k == "y")), time.perf_counter() - t0
        txt1.draw()
        txt2.draw()
        win.flip()


def get_ratings(win, questions):
    """Questions asked right after an ad (brand recall is asked at the end instead)."""
    out = {}
    for q in questions:
        if q["type"] == "brand_choice":
            continue
        value, rt = ask_rating(win, q)
        out[q["id"]], out[f"{q['id']}_rt_s"] = value, round(rt, 3)
    return out


def _still_frame(path, t_s):
    """PNG of the ad at t_s (temp file) for the recall question, or None."""
    import cv2
    import tempfile
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_s * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None, None
    out = os.path.join(tempfile.gettempdir(), f"recall_{os.path.basename(path)}.png")
    cv2.imwrite(out, fr)
    return out, fr.shape[1] / fr.shape[0]


def ask_brand(win, path, q):
    """End-of-session brand recall: a still frame (chosen to avoid the brand when
    possible), then the brand + 3 competitors in a per-viewer shuffled order + 'not sure'."""
    ad_file = os.path.basename(path)
    prof = SC.load_profile(ad_file) or {}
    opts = list(prof.get("recall_options") or [])
    random.Random(f"{PARTICIPANT_ID}-{ad_file}").shuffle(opts)
    opts.append("Not sure")
    keys = [str(i + 1) for i in range(len(opts))]
    t_s, how = SC.recall_frame_time(ad_file)
    img_path, aspect = _still_frame(path, t_s) if t_s is not None else (None, None)
    stims = []
    if img_path:
        h = 0.75
        w = h * aspect * win.size[1] / win.size[0]
        stims.append(visual.ImageStim(win, image=img_path, size=(w, h), pos=(0, 0.5), units="norm"))
    stims.append(visual.TextStim(win, text=q["text"] + "  (the ad shown in this picture)", color="white",
                                 height=0.065, pos=(0, 0.02), wrapWidth=1.8))
    stims.append(visual.TextStim(win, text="\n".join(f"{k}    {o}" for k, o in zip(keys, opts)), color="white",
                                 height=0.065, pos=(0, -0.45), wrapWidth=1.2, alignText="left", anchorHoriz="center"))
    event.clearEvents()
    t0 = time.perf_counter()
    choice = None
    if AUTO:
        core.wait(0.3)
        choice = random.choice(opts)
    while choice is None:
        for s in stims:
            s.draw()
        win.flip()
        for k in event.getKeys():
            if k == "escape":
                core.quit()
            if k in keys:
                choice = opts[keys.index(k)]
    rt = time.perf_counter() - t0
    print(f"[recall] {ad_file}: chose {choice!r} (brand {prof.get('brand')!r}; still frame {t_s}s, {how})")
    return {q["id"]: int(choice == prof.get("brand")), f"{q['id']}_rt_s": round(rt, 3),
            f"{q['id']}_choice": choice}


def ads_from_config(cfg):
    """[(path, [question dicts], viewing)] in the order to show them. With
    viewings = 2 the second block is reshuffled (random order) and never starts
    with the ad that ended the first block."""
    items = [(os.path.join(ADS_DIR, a["file"]), [cfg["questions"][q] for q in a["questions"]]) for a in cfg["ads"]]
    if cfg["order"] == "random":
        random.Random(PARTICIPANT_ID).shuffle(items)
    plan = [(p, qs, 1) for p, qs in items]
    if cfg.get("viewings", 1) == 2:
        second = list(items)
        if cfg["order"] == "random":
            random.Random(f"{PARTICIPANT_ID}-v2").shuffle(second)
        if len(second) > 1 and second[0][0] == items[-1][0]:
            second[0], second[1] = second[1], second[0]
        plan += [(p, qs, 2) for p, qs in second]
    return plan


def main():
    global PARTICIPANT_ID, AUTO
    import argparse
    import re
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="participant id, e.g. p01 (letters, digits, _ or -)")
    ap.add_argument("--config", help="session config JSON from the setup page")
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--auto", action="store_true", help="TEST ONLY: no key presses, random answers")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.id):
        raise SystemExit(f"Invalid --id {args.id!r}: use only letters, digits, _ or -")
    PARTICIPANT_ID, AUTO = args.id, args.auto

    cfg = SC.load_config(args.config) if args.config else None
    if cfg and cfg["participant_id"] != args.id:
        raise SystemExit(f"--id {args.id} does not match the config's participant id {cfg['participant_id']}")

    os.makedirs(ADS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_base = os.path.join(OUT_DIR, PARTICIPANT_ID)
    existing = [out_base + s for s in ("_raw.fif", "_raw.npz", "_ratings.csv") if os.path.exists(out_base + s)]
    if existing and not args.overwrite:
        raise SystemExit(f"Session '{PARTICIPANT_ID}' already exists ({[os.path.basename(e) for e in existing]}). "
                         f"Use a new --id, or add --overwrite if you really mean to replace it.")

    if cfg:
        items = ads_from_config(cfg)
        do_baseline = cfg.get("baseline", True) and not args.no_baseline
    else:
        bank = {q["id"]: q for q in SC.load_bank()}
        items = [(p, [bank[q] for q in DEFAULT_RATING_IDS], 1) for p in find_ads()]
        do_baseline = not args.no_baseline
    if args.one:
        items = items[:1]
    all_qids = []
    for _, qs, _v in items:
        for q in qs:
            if q["id"] not in all_qids:
                all_qids.append(q["id"])
    repeats = any(v == 2 for _, _, v in items)
    print(f"[session] {PARTICIPANT_ID}: {len(items)} ad(s) in this order: "
          f"{[viewing_name(p, v) for p, _, v in items]}; baseline {'on' if do_baseline else 'off'}"
          + ("; AUTO TEST MODE" if AUTO else ""))

    rec = LSLRecorder()  # fails fast with a clear message if EEGStudio's LSL Outlet is off
    rec.start()
    win = visual.Window(size=(1024, 768), fullscr=False, units="norm", color="black")
    rows = []
    t_session = time.time()

    try:
        push(rec, "session_start")
        if do_baseline:
            run_baseline(win, rec)
        any_yesno = any(q["type"] == "yesno" for _, qs, _v in items for q in qs)
        n_unique = len({p for p, _, _v in items})
        _show_until_space(
            win, f"{'Ads to watch: ' + str(n_unique) if items else 'No ad videos found -- using placeholders'}"
                 + ("\n\nEach ad will be shown TWICE (in a mixed order).\nJust watch it again normally the second time."
                    if repeats else "") + "\n\n"
                 "After " + ("the first showing of " if repeats else "") + "each ad you'll answer a few quick questions,\n"
                 "using the number keys 1-9" + (" (or Y / N)" if any_yesno else "") + ".\n\nPress SPACE to begin.")
        if items:
            for order, (path, qs, viewing) in enumerate(items, 1):
                name, duration = show_ad(win, rec, play_real_ad, path, viewing=viewing)
                answers = get_ratings(win, qs) if viewing == 1 else {}
                rows.append({"order": order, "ad": os.path.basename(path), "viewing": viewing,
                             "duration_s": round(duration, 3), **answers})
            # Brand recall LAST -- after every viewing, so the question can't point the
            # viewer at the brand before a repeat. One per ad, in first-viewing order.
            recall = [(p, q) for p, qs, v in items if v == 1 for q in qs if q["type"] == "brand_choice"]
            if recall:
                _show_until_space(win, "Last part: a few quick memory questions about the ads.\n\n"
                                       "Answer with the number keys.\n\nPress SPACE to continue.")
                for path, q in recall:
                    res = ask_brand(win, path, q)
                    row = next(r for r in rows if r["ad"] == os.path.basename(path) and r["viewing"] == 1)
                    row.update(res)
                    if f"{q['id']}_choice" not in all_qids:
                        all_qids.append(f"{q['id']}_choice")
        else:
            bank = {q["id"]: q for q in SC.load_bank()}
            qs = [bank[q] for q in DEFAULT_RATING_IDS]
            all_qids = list(DEFAULT_RATING_IDS)
            for i in range(N_PLACEHOLDER_ADS):
                name, duration = show_ad(win, rec, play_placeholder_ad, i)
                rows.append({"order": i + 1, "ad": name, "viewing": 1, "duration_s": duration, **get_ratings(win, qs)})
        push(rec, "session_end")
    finally:
        # Runs even on ESC (core.quit raises SystemExit) -- never lose a recorded session.
        rec.stop()
        with open(out_base + "_ratings.csv", "w", newline="", encoding="utf-8") as f:
            # one column pair per question used anywhere; blank where an ad didn't ask it
            # (repeat viewings have no answers; *_choice columns hold the recall answer text)
            rating_cols = [c for qid in all_qids for c in ((qid,) if qid.endswith("_choice") else (qid, f"{qid}_rt_s"))]
            writer = csv.DictWriter(f, fieldnames=["order", "ad", "viewing", "duration_s"] + rating_cols, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved ratings: {out_base}_ratings.csv ({len(rows)} ads)")
        if rec.n_samples():
            rec.save(out_base)
        frames_per_ad = {}
        for ad, _, pts, _ in rec.frames:
            frames_per_ad.setdefault(ad, set()).add(round(pts, 4))
        with open(out_base + "_session.json", "w", encoding="utf-8") as fh:
            json.dump({"participant_id": PARTICIPANT_ID, "config_file": args.config, "config": cfg,
                       "auto_test_mode": AUTO, "baseline": do_baseline,
                       "viewings": cfg.get("viewings", 1) if cfg else 1,
                       "order_shown": [r["ad"] + (f"#{r['viewing']}" if r.get("viewing", 1) > 1 else "") for r in rows],
                       "completed_ads": len(rows),
                       "planned_ads": len(items), "started_unix": t_session,
                       "distinct_video_frames_shown": {a: len(s) for a, s in frames_per_ad.items()}},
                      fh, indent=1, ensure_ascii=False)
        win.close()


if __name__ == "__main__":
    main()
