"""
Verifies lsl_recorder.py's CHANNEL_ORDER against real physiology, plus the
stream's sample rate and units. Two phases, ~35s total:

  1. BLINK (15s)       -> blink artifacts should peak on Fp1/Fp2.
  2. EYES CLOSED (20s) -> alpha (8-13Hz) should rise most at the back of the
                          head (O1/O2/P3/P4/Pz/T5/T6) vs. the blink phase.

Why both: Fp1/Fp2 are channels 1-2 under BOTH the amplifier's hardware order
and EEGStudio's on-screen order, so blinks alone can't tell the two orders
apart. Posterior channels sit at different indices in each, so the alpha
check is what actually confirms the mapping.

Usage (EEGStudio's LSL Outlet on, cap on the participant):
    .venv\\Scripts\\python.exe lab_task_psychopy\\blink_test.py
"""
import os
import sys
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lsl_recorder import LSLRecorder, SCALP_CHANNELS

BLINK_S = 15
CLOSED_S = 20
POSTERIOR = {"O1", "O2", "P3", "P4", "Pz", "T5", "T6"}


def countdown(seconds, msg):
    print(f"\n>>> {msg} ({seconds}s)", flush=True)
    for remaining in range(seconds, 0, -5):
        print(f"    {remaining}s left", flush=True)
        time.sleep(min(5, remaining))


def alpha_power(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(fs * 2), axis=-1)
    band = (f >= 8) & (f <= 13)
    return p[:, band].mean(axis=-1)


def main():
    rec = LSLRecorder()
    rec.start()
    time.sleep(1)
    countdown(BLINK_S, "Eyes OPEN. BLINK HARD every ~2 seconds")
    n_blink = len(rec._stamps)
    countdown(CLOSED_S, "Now CLOSE YOUR EYES and relax -- don't blink or move")
    rec.stop()

    if not rec._chunks:
        print("\nNo EEG samples arrived. EEGStudio's LSL Outlet is visible but not sending -- "
              "is acquisition running? Toggle LSL Outlet off/on, wait ~5s, then rerun.")
        return

    data = np.vstack(rec._chunks).T
    stamps = np.asarray(rec._stamps)
    fs = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    labels = rec.ch_labels
    scalp = [i for i, ch in enumerate(labels) if ch in SCALP_CHANNELS]
    if not scalp:
        print(f"\nChannel labels weren't mapped ({labels[:5]}...) -- can't run the checks.")
        return

    print("\n================ RESULTS ================")
    rate_ok = abs(fs - rec.sfreq) / rec.sfreq < 0.02
    print(f"1. SAMPLE RATE: nominal {rec.sfreq:.1f}Hz, measured {fs:.1f}Hz -> "
          f"{'OK' if rate_ok else 'UNSTABLE'}")

    sos = butter(4, [1, 40], btype="band", fs=fs, output="sos")
    filt = sosfiltfilt(sos, data[scalp], axis=1)
    edge = int(fs)
    blink_seg = filt[:, edge:n_blink]
    closed_seg = filt[:, n_blink + edge:-edge]

    typical = np.median(closed_seg.std(axis=1))
    print(f"2. AMPLITUDE: typical scalp signal {typical:.1f} (1-40Hz, eyes closed). "
          f"Real EEG in uV is roughly 5-50; ~10x that suggests 0.1uV units.")

    sos_b = butter(4, [1, 5], btype="band", fs=fs, output="sos")
    blink_pow = sosfiltfilt(sos_b, data[scalp][:, :n_blink], axis=1)[:, edge:].std(axis=1)
    b_order = np.argsort(blink_pow)[::-1]
    top_blink = [labels[scalp[i]] for i in b_order[:2]]
    blink_ok = set(top_blink) == {"Fp1", "Fp2"}
    print(f"3a. BLINKS peak on: {top_blink} -> {'PASS' if blink_ok else 'CHECK'} (expect Fp1, Fp2)")

    ratio = alpha_power(closed_seg, fs) / alpha_power(blink_seg, fs)
    a_order = np.argsort(ratio)[::-1]
    top_alpha = [labels[scalp[i]] for i in a_order[:3]]
    alpha_ok = set(top_alpha) <= POSTERIOR
    print(f"3b. EYES-CLOSED ALPHA rises most on: {top_alpha} -> {'PASS' if alpha_ok else 'CHECK'} "
          f"(expect back-of-head: O1/O2/P3/P4/Pz/T5/T6)")

    print("\n    Full alpha-increase ranking (closed/open):")
    for rank, i in enumerate(a_order, 1):
        print(f"      {rank:2d}. {labels[scalp[i]]:4s} x{ratio[i]:.2f}")

    print("\n=========================================")
    if rate_ok and blink_ok and alpha_ok:
        print("ALL CHECKS PASS -- channel order confirmed, ready for real sessions.")
    else:
        print("Some checks need attention -- send me this whole output.")


if __name__ == "__main__":
    main()
