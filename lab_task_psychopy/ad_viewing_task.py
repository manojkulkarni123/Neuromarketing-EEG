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
      --one          quick test: one ad only
      --no-baseline  skip the baseline block
      --overwrite    allow replacing an existing session with the same id

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
import time
import random

from psychopy import visual, core, event

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lsl_recorder import LSLRecorder

PARTICIPANT_ID = None  # set from --id in main(); also seeds the per-participant ad order
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
    for engagement). Fits DISPLAY_FRACTION of the window, keeping aspect ratio."""
    import cv2
    cap = cv2.VideoCapture(path)
    vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    win_w, win_h = win.size
    scale = DISPLAY_FRACTION * min(win_w / vw, win_h / vh)
    return (2 * vw * scale / win_w, 2 * vh * scale / win_h)  # norm units


def push(recorder, label):
    recorder.mark(label)
    print(f"[marker] {label}")


START_ATTEMPTS = 10


def play_real_ad(win, rec, path, start_label=None, end_label=None):
    """PsychoPy's player intermittently fails to deliver the first frame of
    these H.264 files (~50% of attempts, no pattern by path/size/warm-up), so
    retry the start. The start marker is sent only once the first frame is
    actually on screen, so a retry never shifts it away from true onset."""
    name = os.path.basename(path)
    start_label = start_label or f"AD_START:{name}"
    end_label = end_label or f"AD_END:{name}"
    size = display_size(win, path)
    for attempt in range(1, START_ATTEMPTS + 1):
        mov = visual.MovieStim(win, filename=path, noAudio=False, size=size, units="norm")
        try:
            mov.draw()
            win.flip()
            break
        except RuntimeError:
            mov.pause()
            core.wait(0.2)
    else:
        raise RuntimeError(f"Could not start {name} after {START_ATTEMPTS} attempts.")
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


def show_ad(win, rec, play_fn, *args):
    """Fixation cross -> ad -> gray screen, around any ad player."""
    name_hint = os.path.basename(args[0]) if args and isinstance(args[0], str) else f"placeholder_{args[0]}"
    pre_ad_fixation(win, rec, name_hint)
    name, duration = play_fn(win, rec, *args)
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


# Asked after every ad, same wording/order/scale each time. Liking the ad and
# wanting the product are deliberately separate (often diverge). Not "how
# engaging was it" -- engagement is what the EEG measures, so rating it too
# would compare the brain against the participant's guess about their brain.
RATING_QUESTIONS = [
    ("liking", "How much did you like this ad?", "1 = not at all   ...   9 = very much"),
    ("purchase_intent", "How likely would you be to buy this product?", "1 = very unlikely   ...   9 = very likely"),
]
DIGITS = [str(d) for d in range(1, 10)]


def ask_rating(win, question, anchors):
    txt1 = visual.TextStim(win, text=question, color="white", height=0.08, pos=(0, 0.15), wrapWidth=1.8)
    txt2 = visual.TextStim(win, text=anchors, color="white", height=0.06, pos=(0, -0.05), wrapWidth=1.8)
    # Discard keys pressed earlier (during the ad, or a double-press on the
    # previous question) so they can't auto-answer this one.
    event.clearEvents()
    txt1.draw()
    txt2.draw()
    win.flip()
    t0 = time.perf_counter()
    while True:
        for k in event.getKeys():
            if k == "escape":
                core.quit()
            if k in DIGITS:
                return int(k), time.perf_counter() - t0
        txt1.draw()
        txt2.draw()
        win.flip()


def get_ratings(win):
    out = {}
    for key, question, anchors in RATING_QUESTIONS:
        value, rt = ask_rating(win, question, anchors)
        out[key], out[f"{key}_rt_s"] = value, rt
    return out


def main():
    global PARTICIPANT_ID
    import argparse
    import re
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="participant id, e.g. p01 (letters, digits, _ or -)")
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.id):
        raise SystemExit(f"Invalid --id {args.id!r}: use only letters, digits, _ or -")
    PARTICIPANT_ID = args.id

    os.makedirs(ADS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_base = os.path.join(OUT_DIR, PARTICIPANT_ID)
    existing = [out_base + s for s in ("_raw.fif", "_raw.npz", "_ratings.csv") if os.path.exists(out_base + s)]
    if existing and not args.overwrite:
        raise SystemExit(f"Session '{PARTICIPANT_ID}' already exists ({[os.path.basename(e) for e in existing]}). "
                         f"Use a new --id, or add --overwrite if you really mean to replace it.")

    ads = find_ads()
    rec = LSLRecorder()  # fails fast with a clear message if EEGStudio's LSL Outlet is off
    rec.start()
    win = visual.Window(size=(1024, 768), fullscr=False, units="norm", color="black")
    rows = []

    try:
        push(rec, "session_start")
        if not args.no_baseline:
            run_baseline(win, rec)
        _show_until_space(
            win, f"{'Real ads found: ' + str(len(ads)) if ads else 'No ad videos found -- using placeholders'}\n\n"
                 "After each ad you'll answer two quick questions,\n"
                 "using the number keys 1-9.\n\nPress SPACE to begin.")
        if ads:
            for order, path in enumerate(ads, 1):
                name, duration = show_ad(win, rec, play_real_ad, path)
                rows.append({"order": order, "ad": name, "duration_s": duration, **get_ratings(win)})
        else:
            for i in range(N_PLACEHOLDER_ADS):
                name, duration = show_ad(win, rec, play_placeholder_ad, i)
                rows.append({"order": i + 1, "ad": name, "duration_s": duration, **get_ratings(win)})
        push(rec, "session_end")
    finally:
        # Runs even on ESC (core.quit raises SystemExit) -- never lose a recorded session.
        rec.stop()
        with open(out_base + "_ratings.csv", "w", newline="") as f:
            rating_cols = [c for key, _, _ in RATING_QUESTIONS for c in (key, f"{key}_rt_s")]
            writer = csv.DictWriter(f, fieldnames=["order", "ad", "duration_s"] + rating_cols)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved ratings: {out_base}_ratings.csv ({len(rows)} ads)")
        if rec.n_samples():
            rec.save(out_base)
        win.close()


if __name__ == "__main__":
    main()
