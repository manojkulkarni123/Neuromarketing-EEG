"""
Analysis for the lab session's PRIMARY, foolproof deliverable: eyes-closed vs
eyes-open alpha blocking (the Berger effect). This needs no hardware trigger
sync at all -- just a rough note of when you said "close your eyes" / "open
your eyes" during the recording, because the effect is a sustained change over
seconds, not a fast single-trial event. It is the single most reliable, easy
to demonstrate phenomenon in EEG, which is exactly why it's the right target
for a first hands-on session: if this doesn't show up, something about the
setup (electrode contact, reference, grounding) needs fixing before trusting
anything more subtle from the same session.

Usage:
    1. Export the recording from EEGStudio as EDF.
    2. Edit EDF_PATH and the four *_START_S / *_END_S values below to match
       your actual recording (seconds from the start of the file).
    3. Run: python pipeline/analyze_alpha_blocking.py
"""
import os
import sys

import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edf_reader import read_edf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- EDIT THESE for your actual recording ----
EDF_PATH = os.path.join(PROJECT_ROOT, "data", "lab_session", "session1.edf")

EYES_CLOSED_START_S = 10
EYES_CLOSED_END_S = 40

EYES_OPEN_START_S = 50
EYES_OPEN_END_S = 80

# Posterior/occipital channels show alpha blocking most clearly. Edit if your
# cap's channel labels differ (script does a case-insensitive substring match).
POSTERIOR_CHANNEL_HINTS = ["O1", "O2", "OZ", "PZ", "P3", "P4"]

ALPHA_BAND = (8, 13)
# ------------------------------------------------


def find_posterior_channels(labels):
    matches = []
    for label in labels:
        clean = label.upper().replace("EEG", "").replace("-", "").replace(" ", "")
        for hint in POSTERIOR_CHANNEL_HINTS:
            if hint in clean:
                matches.append(label)
                break
    return matches


def band_power(signal, fs, band):
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 2)))
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapz(psd[band_mask], freqs[band_mask])), freqs, psd


def main():
    print(f"Loading {EDF_PATH} ...")
    d = read_edf(EDF_PATH)
    print(f"Duration: {d['duration_s']:.1f}s, channels: {d['labels']}")

    posterior = find_posterior_channels(d["labels"])
    if not posterior:
        print("\nWARNING: no posterior channel labels matched "
              f"{POSTERIOR_CHANNEL_HINTS}. Actual labels in this file:")
        print(d["labels"])
        print("Edit POSTERIOR_CHANNEL_HINTS above to match, then rerun.")
        sys.exit(1)

    print(f"\nUsing posterior channels: {posterior}\n")
    print(f"{'channel':>10} {'closed alpha power':>20} {'open alpha power':>18} {'ratio (closed/open)':>20}")

    results = []
    for ch in posterior:
        sig = d["signals"][ch]
        fs = d["fs"][ch]

        closed_seg = sig[int(EYES_CLOSED_START_S * fs):int(EYES_CLOSED_END_S * fs)]
        open_seg = sig[int(EYES_OPEN_START_S * fs):int(EYES_OPEN_END_S * fs)]

        closed_power, _, _ = band_power(closed_seg, fs, ALPHA_BAND)
        open_power, _, _ = band_power(open_seg, fs, ALPHA_BAND)
        ratio = closed_power / open_power if open_power > 0 else float("nan")

        results.append((ch, closed_power, open_power, ratio))
        print(f"{ch:>10} {closed_power:>20.2f} {open_power:>18.2f} {ratio:>20.2f}")

    mean_ratio = np.mean([r[3] for r in results])
    print(f"\nMean closed/open alpha power ratio across posterior channels: {mean_ratio:.2f}")
    if mean_ratio > 1.3:
        print(">>> Alpha blocking clearly present (closed > open, as expected). "
              "The cap + recording chain is working correctly.")
    elif mean_ratio > 1.0:
        print(">>> Alpha blocking present but weak -- check electrode impedance/contact, "
              "or whether the participant actually relaxed with eyes closed.")
    else:
        print(">>> No alpha blocking detected (closed <= open). This suggests a setup "
              "problem (bad reference/ground, high impedance, noisy environment) rather "
              "than a real physiological finding -- worth troubleshooting before trusting "
              "anything else from this session.")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(posterior), figsize=(5 * len(posterior), 4), squeeze=False)
        for i, ch in enumerate(posterior):
            sig = d["signals"][ch]
            fs = d["fs"][ch]
            closed_seg = sig[int(EYES_CLOSED_START_S * fs):int(EYES_CLOSED_END_S * fs)]
            open_seg = sig[int(EYES_OPEN_START_S * fs):int(EYES_OPEN_END_S * fs)]
            _, f_c, psd_c = band_power(closed_seg, fs, ALPHA_BAND)
            _, f_o, psd_o = band_power(open_seg, fs, ALPHA_BAND)
            ax = axes[0][i]
            ax.plot(f_c, psd_c, label="eyes closed")
            ax.plot(f_o, psd_o, label="eyes open")
            ax.axvspan(*ALPHA_BAND, color="gray", alpha=0.2, label="alpha band")
            ax.set_title(ch)
            ax.set_xlabel("Hz")
            ax.set_xlim(0, 30)
            ax.legend(fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(PROJECT_ROOT, "data", "lab_session", "alpha_blocking_result.png")
        fig.savefig(out_png, dpi=150)
        print(f"\nSaved plot: {out_png}")
    except Exception as e:
        print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
