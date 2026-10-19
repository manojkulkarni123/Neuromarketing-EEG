"""
PsychoPy ad-viewing task -- plays short video ads and collects a 1-9
engagement rating after each one, while recording EEG itself: it pulls
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
    .venv\\Scripts\\python.exe lab_task_psychopy\\ad_viewing_task.py

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

PARTICIPANT_ID = "pilot01_ads"
N_PLACEHOLDER_ADS = 4
PLACEHOLDER_DURATION_S = 10.0

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADS_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session", "ads")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session")

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


def play_real_ad(win, rec, path):
    """PsychoPy's player intermittently fails to deliver the first frame of
    these H.264 files (~50% of attempts, no pattern by path/size/warm-up), so
    retry the start. AD_START is marked only once the first frame is actually
    on screen, so a retry never shifts the marker away from true ad onset."""
    name = os.path.basename(path)
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
    push(rec, f"AD_START:{name}")
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
    push(rec, f"AD_END:{name}")
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


def get_rating(win):
    txt1 = visual.TextStim(win, text="How engaging was that ad?", color="white",
                            height=0.08, pos=(0, 0.15))
    txt2 = visual.TextStim(win, text="1 = not at all   ...   9 = extremely",
                            color="white", height=0.06, pos=(0, -0.05))
    t0 = time.perf_counter()
    rating = None
    while rating is None:
        txt1.draw()
        txt2.draw()
        win.flip()
        keys = event.getKeys()
        for k in keys:
            if k == "escape":
                core.quit()
            if k in [str(d) for d in range(1, 10)]:
                rating = int(k)
    rt = time.perf_counter() - t0
    return rating, rt


def main():
    os.makedirs(ADS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    ads = find_ads()
    rec = LSLRecorder()  # fails fast with a clear message if EEGStudio's LSL Outlet is off
    rec.start()
    win = visual.Window(size=(1024, 768), fullscr=False, units="norm", color="black")
    out_base = os.path.join(OUT_DIR, PARTICIPANT_ID)
    rows = []

    try:
        intro = visual.TextStim(
            win, color="white", height=0.06,
            text=(f"{'Real ads found: ' + str(len(ads)) if ads else 'No ad videos found -- using placeholders'}\n\n"
                  "Watch each ad, then rate how engaging it was (1-9).\n\n"
                  "Press SPACE to begin."))
        intro.draw()
        win.flip()
        event.waitKeys(keyList=["space"])

        push(rec, "session_start")
        if ads:
            for order, path in enumerate(ads, 1):
                name, duration = play_real_ad(win, rec, path)
                rating, rt = get_rating(win)
                rows.append({"order": order, "ad": name, "duration_s": duration,
                             "rating": rating, "rating_rt_s": rt})
        else:
            for i in range(N_PLACEHOLDER_ADS):
                name, duration = play_placeholder_ad(win, rec, i)
                rating, rt = get_rating(win)
                rows.append({"order": i + 1, "ad": name, "duration_s": duration,
                             "rating": rating, "rating_rt_s": rt})
        push(rec, "session_end")
    finally:
        # Runs even on ESC (core.quit raises SystemExit) -- never lose a recorded session.
        rec.stop()
        with open(out_base + "_ratings.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["order", "ad", "duration_s", "rating", "rating_rt_s"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved ratings: {out_base}_ratings.csv ({len(rows)} ads)")
        if rec.n_samples():
            rec.save(out_base)
        win.close()


if __name__ == "__main__":
    main()
