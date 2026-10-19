"""
Random-dot-motion decision task with confidence rating -- replicates the
paradigm shown in your own PPT's "Overview of the Experiment" diagram:
fixation -> moving dots (left/right) -> choice (left/right key) -> confidence
report (1-9) -> repeat.

Built with pygame + numpy only (both already installed, no dependency risk).

TIMING / SYNC WITH THE EEG RECORDING -- read this before running tomorrow:
This script logs every event with a high-resolution timer
(time.perf_counter(), microsecond-ish precision) on ITS OWN clock. That is
NOT automatically the same clock as the Mitsar/EEGStudio recording. You have
three options, in order of how good the resulting data will be:

  1. BEST -- hardware/LSL trigger: if EEGStudio can receive external event
     markers (parallel port TTL, or LSL), wire that up and every event here
     lands in the EEG recording at the exact right sample. Ask the lab
     tomorrow morning whether this is possible; if yes, tell me and I'll
     wire in the actual trigger call (a few lines) around send_marker() below
     -- it's a no-op placeholder right now.
  2. OK -- manual marker key: if EEGStudio has an "insert marker" hotkey the
     operator can press live, press it at the exact moment you press SPACE
     on the "session start" screen below. That gives you ONE known reference
     point to align this script's clock to the EEG recording's clock
     afterward. Good to maybe +/-50-100ms -- fine for eyeballing, shaky for
     single-trial decoding.
  3. FALLBACK -- no sync at all: you still get clean behavioral data (choices,
     confidence, RTs) and a recording that happened during roughly the same
     wall-clock window, but you CANNOT align individual trials to individual
     EEG samples. Useful only for a rehearsal of the task itself, not for any
     EEG analysis.

Whichever applies, note it in SESSION_NOTES.txt (written into the output
folder) -- decode_lda.py-style analysis on this data downstream needs to know
which sync quality it's working with.

Output: CSV with one row per trial (tstim, tresp, tconf, RT, choice, accuracy,
confidence, dotdirection, plus perf_counter AND wall-clock timestamps) --
column names deliberately match the original dataset's events files so the
existing pipeline/decode_lda.py feature-extraction logic can be pointed at it
later with minimal changes.
"""
import os
import sys
import csv
import time
import random

import numpy as np
import pygame

# ---- SESSION CONFIG -- edit before running ----
PARTICIPANT_ID = "pilot01"
N_TRIALS = 60
COHERENCE = 0.2          # fraction of dots moving in the true direction; lower = harder
PRACTICE_TRIALS = 5      # shown first, not saved, so the participant can calibrate to the difficulty

# Manual-marker sync: since EEGStudio only has a manual "insert marker" hotkey
# (no automatic trigger), a single sync point at the start isn't enough --
# clock drift between this PC and the EEG system accumulates over the session.
# Instead we emit a beep+flash every SYNC_CHECKPOINT_INTERVAL trials; the
# operator presses the EEGStudio marker key each time. With several of these
# spread across the session, you can fit a line (EEG-time = a*script-time + b)
# through the checkpoint pairs afterward and correct for drift, instead of
# trusting one single reference point.
SYNC_CHECKPOINT_INTERVAL = 10
SYNC_BEEP_FREQ_HZ = 1000
SYNC_BEEP_DURATION_S = 0.15

FIXATION_DURATION_S = 0.5
MAX_STIM_DURATION_S = 5.0   # safety cap in case someone never responds
ITI_DURATION_S = 0.5

N_DOTS = 150
DOT_RADIUS_PX = 2
APERTURE_RADIUS_PX = 200
DOT_SPEED_PX_PER_FRAME = 4

SCREEN_SIZE = (800, 600)
BG_COLOR = (0, 0, 0)
DOT_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)
FIXATION_COLOR = (255, 255, 255)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "lab_session")
# ------------------------------------------------


def send_marker(label):
    """Placeholder for a real hardware/LSL trigger -- see module docstring
    option 1. Currently a no-op; the event is still logged in the CSV with a
    precise local timestamp regardless."""
    pass


def make_beep_sound():
    fs = 44100
    t = np.linspace(0, SYNC_BEEP_DURATION_S, int(fs * SYNC_BEEP_DURATION_S), False)
    tone = np.sin(2 * np.pi * SYNC_BEEP_FREQ_HZ * t)
    # short fade in/out to avoid a click
    fade = np.linspace(0, 1, int(fs * 0.01))
    tone[:len(fade)] *= fade
    tone[-len(fade):] *= fade[::-1]
    stereo = np.stack([tone, tone], axis=1)
    audio = (stereo * 32767 * 0.5).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(audio))


def sync_checkpoint(screen, font, center, beep_sound, label, log_rows):
    """Flash + beep the operator can react to by pressing the EEGStudio manual
    marker key. Logs OUR side's precise timestamp for later drift correction."""
    screen.fill((255, 255, 255))
    txt = font.render("SYNC MARK -- press EEG marker key NOW", True, (0, 0, 0))
    screen.blit(txt, (center[0] - txt.get_width() // 2, center[1]))
    pygame.display.flip()
    t_perf = time.perf_counter()
    t_wall = time.time()
    beep_sound.play()
    log_rows.append({"label": label, "perf_counter": t_perf, "wall_clock": t_wall})
    pygame.time.delay(int(SYNC_BEEP_DURATION_S * 1000) + 200)
    screen.fill(BG_COLOR)
    pygame.display.flip()


class DotField:
    def __init__(self, n_dots, aperture_radius, coherence, direction_deg):
        self.aperture_radius = aperture_radius
        self.coherence = coherence
        self.direction_rad = np.deg2rad(direction_deg)
        angles = np.random.uniform(0, 2 * np.pi, n_dots)
        radii = aperture_radius * np.sqrt(np.random.uniform(0, 1, n_dots))
        self.x = radii * np.cos(angles)
        self.y = radii * np.sin(angles)

    def step(self):
        n = len(self.x)
        coherent_mask = np.random.uniform(0, 1, n) < self.coherence
        angles = np.where(coherent_mask, self.direction_rad,
                           np.random.uniform(0, 2 * np.pi, n))
        self.x += DOT_SPEED_PX_PER_FRAME * np.cos(angles)
        self.y += DOT_SPEED_PX_PER_FRAME * np.sin(angles)

        dist = np.sqrt(self.x ** 2 + self.y ** 2)
        out_of_bounds = dist > self.aperture_radius
        n_out = int(out_of_bounds.sum())
        if n_out:
            new_angles = np.random.uniform(0, 2 * np.pi, n_out)
            new_radii = self.aperture_radius * np.sqrt(np.random.uniform(0, 1, n_out))
            self.x[out_of_bounds] = new_radii * np.cos(new_angles)
            self.y[out_of_bounds] = new_radii * np.sin(new_angles)

    def draw(self, surface, center):
        pts = np.stack([self.x + center[0], self.y + center[1]], axis=1)
        for px, py in pts:
            pygame.draw.circle(surface, DOT_COLOR, (int(px), int(py)), DOT_RADIUS_PX)


def run_trial(screen, clock, font, center, is_practice=False):
    t0 = time.perf_counter()

    # 1. fixation
    screen.fill(BG_COLOR)
    pygame.draw.line(screen, FIXATION_COLOR, (center[0] - 10, center[1]), (center[0] + 10, center[1]), 2)
    pygame.draw.line(screen, FIXATION_COLOR, (center[0], center[1] - 10), (center[0], center[1] + 10), 2)
    pygame.display.flip()
    _wait_or_quit(FIXATION_DURATION_S)

    # 2. moving dots -- wait for LEFT/RIGHT response
    true_direction = random.choice(["left", "right"])
    direction_deg = 180 if true_direction == "left" else 0
    dots = DotField(N_DOTS, APERTURE_RADIUS_PX, COHERENCE, direction_deg)

    abs_perf_stim_onset = time.perf_counter()
    t_stim = abs_perf_stim_onset - t0
    wall_stim = time.time()
    send_marker("STIM_ON")

    choice = None
    t_resp = None
    stim_start = time.perf_counter()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_LEFT:
                    choice = "left"
                elif event.key == pygame.K_RIGHT:
                    choice = "right"
                if choice is not None:
                    t_resp = time.perf_counter() - t0
                    send_marker("RESP")

        if choice is not None:
            break
        if time.perf_counter() - stim_start > MAX_STIM_DURATION_S:
            break

        dots.step()
        screen.fill(BG_COLOR)
        dots.draw(screen, center)
        pygame.display.flip()
        clock.tick(60)

    if choice is None:
        # no response within MAX_STIM_DURATION_S -- log as a miss
        return None

    # 3. confidence report
    t_conf_onset = time.perf_counter() - t0
    send_marker("CONF_ON")
    confidence = None
    screen.fill(BG_COLOR)
    prompt1 = font.render("How confident are you?", True, TEXT_COLOR)
    prompt2 = font.render("1 = not confident at all   ...   9 = very confident", True, TEXT_COLOR)
    screen.blit(prompt1, (center[0] - prompt1.get_width() // 2, center[1] - 40))
    screen.blit(prompt2, (center[0] - prompt2.get_width() // 2, center[1] + 10))
    pygame.display.flip()

    while confidence is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if pygame.K_1 <= event.key <= pygame.K_9:
                    confidence = event.key - pygame.K_0
        clock.tick(60)
    t_conf_resp = time.perf_counter() - t0
    send_marker("CONF_RESP")

    # 4. ITI
    screen.fill(BG_COLOR)
    pygame.display.flip()
    _wait_or_quit(ITI_DURATION_S)

    accuracy = int(choice == true_direction)
    return {
        "tstim": t_stim,
        "tresp": t_resp,
        "tconf_onset": t_conf_onset,
        "tconf_resp": t_conf_resp,
        "RT": t_resp - t_stim,
        "conf_RT": t_conf_resp - t_conf_onset,
        "choice": 1 if choice == "left" else 2,
        "dotdirection": 1 if true_direction == "left" else 2,
        "accuracy": accuracy,
        "confidence": confidence,
        "wall_clock_stim_onset": wall_stim,
        "abs_perf_stim_onset": abs_perf_stim_onset,
    }


def _wait_or_quit(duration_s):
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    pygame.init()
    beep_sound = make_beep_sound()
    sync_log = []
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Random Dot Motion Task")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    center = (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2)

    # instructions + sync point
    screen.fill(BG_COLOR)
    lines = [
        "Dots will move LEFT or RIGHT.",
        "Press LEFT or RIGHT arrow to say which way.",
        "Then rate your confidence 1-9.",
        "",
        f"{PRACTICE_TRIALS} practice trials first, then {N_TRIALS} real trials.",
        "",
        "Press SPACE to begin.",
        "You'll see a white SYNC MARK screen + hear a beep periodically --",
        "the operator presses the EEGStudio manual marker key each time this happens.",
    ]
    for i, line in enumerate(lines):
        txt = font.render(line, True, TEXT_COLOR)
        screen.blit(txt, (center[0] - txt.get_width() // 2, 150 + i * 32))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

    session_start_wall = time.time()
    session_start_perf = time.perf_counter()
    print(f"SESSION START -- wall clock: {time.ctime(session_start_wall)}, "
          f"perf_counter reference: {session_start_perf:.6f}")

    sync_checkpoint(screen, font, center, beep_sound, "session_start", sync_log)

    for i in range(PRACTICE_TRIALS):
        screen.fill(BG_COLOR)
        txt = font.render(f"Practice trial {i+1}/{PRACTICE_TRIALS}", True, TEXT_COLOR)
        screen.blit(txt, (center[0] - txt.get_width() // 2, center[1] - 60))
        pygame.display.flip()
        _wait_or_quit(1.0)
        run_trial(screen, clock, font, center, is_practice=True)

    rows = []
    for i in range(N_TRIALS):
        if i > 0 and i % SYNC_CHECKPOINT_INTERVAL == 0:
            sync_checkpoint(screen, font, center, beep_sound, f"checkpoint_trial_{i}", sync_log)

        result = run_trial(screen, clock, font, center)
        if result is not None:
            result["trial_index"] = i
            rows.append(result)
        else:
            print(f"Trial {i}: no response (miss)")

    sync_checkpoint(screen, font, center, beep_sound, "session_end", sync_log)

    out_path = os.path.join(OUT_DIR, f"{PARTICIPANT_ID}_behavioral.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trial_index", "tstim", "tresp", "tconf_onset", "tconf_resp", "RT", "conf_RT",
            "choice", "dotdirection", "accuracy", "confidence", "wall_clock_stim_onset",
            "abs_perf_stim_onset",
        ])
        writer.writeheader()
        writer.writerows(rows)

    sync_path = os.path.join(OUT_DIR, f"{PARTICIPANT_ID}_sync_checkpoints.csv")
    with open(sync_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "perf_counter", "wall_clock"])
        writer.writeheader()
        writer.writerows(sync_log)

    notes_path = os.path.join(OUT_DIR, f"{PARTICIPANT_ID}_SESSION_NOTES.txt")
    with open(notes_path, "w") as f:
        f.write(f"Participant: {PARTICIPANT_ID}\n")
        f.write(f"Session start wall clock: {time.ctime(session_start_wall)} ({session_start_wall})\n")
        f.write(f"Coherence: {COHERENCE}, N_TRIALS: {N_TRIALS}\n")
        f.write(f"Sync method used: manual EEGStudio marker key, {len(sync_log)} checkpoints "
                f"logged in {os.path.basename(sync_path)} (see column 'label').\n")
        f.write("IMPORTANT -- fill this in right after the session, from the EEGStudio operator:\n")
        f.write("  For each checkpoint label above, the corresponding marker timestamp EEGStudio recorded "
                "(or the EEG sample index it landed on). Without this, the sync_checkpoints.csv timestamps "
                "are only useful for measuring drift/consistency, not for aligning to the EEG file.\n")

    n_correct = sum(r["accuracy"] for r in rows)
    print(f"\nDone. {len(rows)}/{N_TRIALS} trials completed, accuracy = {n_correct}/{len(rows)} "
          f"({100*n_correct/len(rows):.1f}%)" if rows else "\nNo trials completed.")
    print(f"Saved: {out_path}")
    print(f"Fill in: {notes_path}")

    pygame.quit()


if __name__ == "__main__":
    main()
