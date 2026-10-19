"""
Preference/neuromarketing decision task -- same structure as random_dot_task.py
(fixation -> decision -> confidence -> repeat, with the same periodic
beep+flash sync-checkpoint system for manual EEG marker alignment), but the
decision is a real one: "which of these two products do you prefer?" instead
of "which way are the dots moving?"

This directly extends RQ1 from the original study (can pre-response EEG
predict an upcoming choice?) to a naturalistic preference decision -- the
neuromarketing angle -- while reusing the exact same sync/logging/analysis
infrastructure already built and tested for the dot-motion task.

Images: drop real product/brand photos (any .png/.jpg) into
data/lab_session/product_images/ before running -- pairs are drawn randomly
from whatever's there. If that folder is empty, the script generates simple
placeholder "products" (colored shape + random label) instead, so it still
runs today even without real images ready -- swap in real ones later and
nothing else about the pipeline needs to change.

No ground truth exists for a preference choice (unlike dot direction), so
there's no "accuracy" here -- only choice (which side) and confidence (how
strong the preference felt). Analysis-wise this means decode_lda.py's
"choice" and "control_stimulus" targets don't apply the same way; only
choice-decoding (RQ1-style) is meaningful for this task.
"""
import os
import sys
import csv
import time
import random

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from random_dot_task import make_beep_sound, sync_checkpoint  # reuse, don't duplicate

# ---- SESSION CONFIG -- edit before running ----
PARTICIPANT_ID = "pilot01_preference"
N_TRIALS = 60
PRACTICE_TRIALS = 3

FIXATION_DURATION_S = 0.5
MAX_DECISION_DURATION_S = 8.0   # preference decisions can take longer than perceptual ones
ITI_DURATION_S = 0.5

SYNC_CHECKPOINT_INTERVAL = 10

SCREEN_SIZE = (900, 600)
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)
FIXATION_COLOR = (255, 255, 255)
IMAGE_SIZE = (300, 300)
IMAGE_GAP_PX = 100

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session", "product_images")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session")
# ------------------------------------------------


def send_marker(label):
    """See random_dot_task.py's module docstring -- same placeholder, wire up
    a real hardware/LSL trigger here later if one becomes available."""
    pass


def load_real_images():
    if not os.path.isdir(IMAGES_DIR):
        return []
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    paths = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.lower().endswith(exts)]
    images = []
    for p in paths:
        try:
            img = pygame.image.load(p)
            img = pygame.transform.smoothscale(img, IMAGE_SIZE)
            images.append((os.path.basename(p), img))
        except Exception as e:
            print(f"Skipping {p}: {e}")
    return images


PLACEHOLDER_COLORS = [
    (220, 60, 60), (60, 140, 220), (60, 200, 100), (230, 190, 50),
    (180, 90, 220), (240, 130, 40), (60, 200, 200), (220, 80, 160),
]
PLACEHOLDER_SHAPES = ["circle", "square", "triangle"]


def make_placeholder_image(font, label):
    surf = pygame.Surface(IMAGE_SIZE)
    color = random.choice(PLACEHOLDER_COLORS)
    shape = random.choice(PLACEHOLDER_SHAPES)
    surf.fill((245, 245, 245))
    cx, cy = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
    if shape == "circle":
        pygame.draw.circle(surf, color, (cx, cy), 100)
    elif shape == "square":
        pygame.draw.rect(surf, color, (cx - 90, cy - 90, 180, 180))
    else:
        pygame.draw.polygon(surf, color, [(cx, cy - 110), (cx - 100, cy + 90), (cx + 100, cy + 90)])
    txt = font.render(label, True, (20, 20, 20))
    surf.blit(txt, (cx - txt.get_width() // 2, IMAGE_SIZE[1] - 40))
    return label, surf


def build_stimulus_pool(font, real_images):
    if real_images:
        print(f"Using {len(real_images)} real product images from {IMAGES_DIR}")
        return real_images
    print(f"No images found in {IMAGES_DIR} -- generating placeholder stimuli instead.")
    print("(Drop real product/brand photos in that folder and rerun for a real neuromarketing session.)")
    n_placeholders = 20
    labels = [f"Product {chr(65 + i)}" for i in range(n_placeholders)]
    return [make_placeholder_image(font, l) for l in labels]


def _wait_or_quit(duration_s):
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()


def run_trial(screen, clock, font, center, stimulus_pool):
    t0 = time.perf_counter()

    screen.fill(BG_COLOR)
    pygame.draw.line(screen, FIXATION_COLOR, (center[0] - 10, center[1]), (center[0] + 10, center[1]), 2)
    pygame.draw.line(screen, FIXATION_COLOR, (center[0], center[1] - 10), (center[0], center[1] + 10), 2)
    pygame.display.flip()
    _wait_or_quit(FIXATION_DURATION_S)

    left_label, left_img = random.choice(stimulus_pool)
    right_label, right_img = random.choice(stimulus_pool)
    while right_label == left_label and len(stimulus_pool) > 1:
        right_label, right_img = random.choice(stimulus_pool)

    left_x = center[0] - IMAGE_GAP_PX // 2 - IMAGE_SIZE[0]
    right_x = center[0] + IMAGE_GAP_PX // 2
    y = center[1] - IMAGE_SIZE[1] // 2

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
        if time.perf_counter() - stim_start > MAX_DECISION_DURATION_S:
            break

        screen.fill(BG_COLOR)
        screen.blit(left_img, (left_x, y))
        screen.blit(right_img, (right_x, y))
        prompt = font.render("Which do you prefer?  <- LEFT   RIGHT ->", True, TEXT_COLOR)
        screen.blit(prompt, (center[0] - prompt.get_width() // 2, y + IMAGE_SIZE[1] + 20))
        pygame.display.flip()
        clock.tick(60)

    if choice is None:
        return None

    t_conf_onset = time.perf_counter() - t0
    send_marker("CONF_ON")
    confidence = None
    screen.fill(BG_COLOR)
    prompt1 = font.render("How strong is that preference?", True, TEXT_COLOR)
    prompt2 = font.render("1 = barely a preference   ...   9 = very strong preference", True, TEXT_COLOR)
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

    screen.fill(BG_COLOR)
    pygame.display.flip()
    _wait_or_quit(ITI_DURATION_S)

    return {
        "tstim": t_stim,
        "tresp": t_resp,
        "tconf_onset": t_conf_onset,
        "tconf_resp": t_conf_resp,
        "RT": t_resp - t_stim,
        "conf_RT": t_conf_resp - t_conf_onset,
        "choice": 1 if choice == "left" else 2,
        "confidence": confidence,
        "left_item": left_label,
        "right_item": right_label,
        "chosen_item": left_label if choice == "left" else right_label,
        "wall_clock_stim_onset": wall_stim,
        "abs_perf_stim_onset": abs_perf_stim_onset,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    pygame.init()
    beep_sound = make_beep_sound()
    sync_log = []

    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Preference Task")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 26)
    center = (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2)

    real_images = load_real_images()
    stimulus_pool = build_stimulus_pool(font, real_images)

    screen.fill(BG_COLOR)
    lines = [
        "Two products will appear side by side.",
        "Press LEFT or RIGHT arrow for the one you prefer.",
        "Then rate how strong that preference is, 1-9.",
        "",
        f"{PRACTICE_TRIALS} practice trials first, then {N_TRIALS} real trials.",
        "",
        "Press SPACE to begin.",
        "You'll see a white SYNC MARK screen + hear a beep periodically --",
        "the operator presses the EEGStudio manual marker key each time this happens.",
    ]
    for i, line in enumerate(lines):
        txt = font.render(line, True, TEXT_COLOR)
        screen.blit(txt, (center[0] - txt.get_width() // 2, 130 + i * 32))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

    session_start_wall = time.time()
    print(f"SESSION START -- wall clock: {time.ctime(session_start_wall)}")
    sync_checkpoint(screen, font, center, beep_sound, "session_start", sync_log)

    for i in range(PRACTICE_TRIALS):
        screen.fill(BG_COLOR)
        txt = font.render(f"Practice trial {i+1}/{PRACTICE_TRIALS}", True, TEXT_COLOR)
        screen.blit(txt, (center[0] - txt.get_width() // 2, center[1] - 60))
        pygame.display.flip()
        _wait_or_quit(1.0)
        run_trial(screen, clock, font, center, stimulus_pool)

    rows = []
    for i in range(N_TRIALS):
        if i > 0 and i % SYNC_CHECKPOINT_INTERVAL == 0:
            sync_checkpoint(screen, font, center, beep_sound, f"checkpoint_trial_{i}", sync_log)

        result = run_trial(screen, clock, font, center, stimulus_pool)
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
            "choice", "confidence", "left_item", "right_item", "chosen_item",
            "wall_clock_stim_onset", "abs_perf_stim_onset",
        ])
        writer.writeheader()
        writer.writerows(rows)

    sync_path = os.path.join(OUT_DIR, f"{PARTICIPANT_ID}_sync_checkpoints.csv")
    with open(sync_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "perf_counter", "wall_clock"])
        writer.writeheader()
        writer.writerows(sync_log)

    print(f"\nDone. {len(rows)}/{N_TRIALS} trials completed.")
    print(f"Saved: {out_path}")
    print(f"Saved: {sync_path}")
    print("Use lab_task/align_sync.py the same way as for the dot-motion task to align these "
          "timestamps to the EEG recording.")

    pygame.quit()


if __name__ == "__main__":
    main()
