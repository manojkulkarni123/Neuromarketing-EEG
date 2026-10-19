"""
Live channel-mapping check: tap one electrode at a time with a finger, and
this prints which channel spiked, using lsl_recorder.py's CHANNEL_ORDER
labels. If you tap F3 and it prints F3, that channel is mapped right.

More reliable than physiological checks (blink/alpha) on a dry-electrode
cap: a tap produces a huge, localized artifact on exactly one channel,
no interpretation needed. Ctrl+C to stop.

Usage (EEGStudio's LSL Outlet on):
    .venv\\Scripts\\python.exe lab_task_psychopy\\tap_test.py
"""
import os
import sys
import time

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lsl_recorder import LSLRecorder, SCALP_CHANNELS

WINDOW_S = 1.0
SPIKE_FACTOR = 4.0  # a channel counts as "tapped" if its activity is this many times the median channel's


def main():
    rec = LSLRecorder()
    labels = rec.ch_labels
    scalp = [i for i, ch in enumerate(labels) if ch in SCALP_CHANNELS]
    sos = butter(2, 1, btype="high", fs=rec.sfreq, output="sos")
    rec.start()
    print("\nTap ONE electrode at a time (e.g. F3, then F4, then Fz, then O1), ~2 seconds apart.")
    print("Each tap should print that electrode's name. Ctrl+C to stop.\n", flush=True)

    last_seen = 0
    try:
        while True:
            time.sleep(WINDOW_S)
            n = len(rec._stamps)
            if n - last_seen < rec.sfreq * 0.5:
                if n == last_seen:
                    print("  (no data arriving -- check EEGStudio's LSL Outlet)", flush=True)
                continue
            data = np.vstack(rec._chunks).T[scalp, last_seen:n]
            last_seen = n
            act = sosfilt(sos, data - data[:, :1], axis=1).std(axis=1)
            top = int(np.argmax(act))
            ratio = act[top] / max(np.median(act), 1e-9)
            if ratio >= SPIKE_FACTOR:
                print(f"  TAP DETECTED on '{labels[scalp[top]]}'  ({ratio:.0f}x the typical channel)", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        rec.stop()


if __name__ == "__main__":
    main()
