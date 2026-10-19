"""
Recovers the true LSL channel -> electrode mapping by comparing our LSL
recording against EEGStudio's own labeled recording of the same moment.

Why: EEGStudio's LSL Outlet sends no channel labels, and the amplifier's
Electrode Layout list turned out NOT to match the stream order (tapping O1
lit up the stream slot we'd labeled "Cz"). EEGStudio's native recordings do
carry correct labels, so we let the data tell us: the same electrode recorded
twice correlates near 1.0.

Step 1 -- record (~75s):
    Start recording in EEGStudio FIRST, then run:
        .venv\\Scripts\\python.exe lab_task_psychopy\\calibrate_channels.py record
    While it runs: tap several different electrodes, blink, clench your jaw --
    varied activity makes every channel distinctive. When it finishes, STOP the
    EEGStudio recording and export it as EDF.

Step 2 -- match:
        .venv\\Scripts\\python.exe lab_task_psychopy\\calibrate_channels.py match <path_to_exported.edf>
    Writes lab_task_psychopy/channel_map.json, which lsl_recorder.py then uses
    automatically instead of its hardcoded CHANNEL_ORDER.
"""
import os
import sys
import json
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample, correlate

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CALIB_BASE = os.path.join(PROJECT_ROOT, "data", "lab_session", "channel_calibration")
MAP_PATH = os.path.join(HERE, "channel_map.json")

RECORD_S = 75
FS_COMMON = 100.0      # both recordings resampled to this before comparing
WINDOW_S = 10.0        # per-window lag refinement absorbs small clock drift between the two recordings
MIN_CORR = 0.8         # a trustworthy match between identical signals should be well above this


def record():
    sys.path.insert(0, HERE)
    from lsl_recorder import LSLRecorder
    rec = LSLRecorder()
    rec.start()
    print(f"\nRecording {RECORD_S}s. Make sure EEGStudio is ALSO recording right now.")
    print("Tap different electrodes, blink, clench jaw -- varied activity helps.\n", flush=True)
    for remaining in range(RECORD_S, 0, -15):
        print(f"  {remaining}s left", flush=True)
        time.sleep(min(15, remaining))
    rec.stop()
    rec.save(CALIB_BASE)
    print("\nDone. Now STOP the EEGStudio recording, export it as EDF, then run:\n"
          f"  .venv\\Scripts\\python.exe lab_task_psychopy\\calibrate_channels.py match <exported.edf>")


def _clean_label(name):
    name = name.replace("EEG", "").strip()
    return name.split("-")[0].strip()


def _prep(data, fs):
    sos = butter(4, [1, 15], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, data, axis=1)
    n_out = int(round(x.shape[1] * FS_COMMON / fs))
    x = resample(x, n_out, axis=1)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)


def _gfp(x):
    return np.abs(x).mean(axis=0)


def _best_lag(a, b, max_lag):
    """Lag (in samples) to shift b so it lines up with a, searched within +-max_lag."""
    c = correlate(a - a.mean(), b - b.mean(), mode="full")
    lags = np.arange(-len(b) + 1, len(a))
    ok = np.abs(lags) <= max_lag
    return int(lags[ok][np.argmax(c[ok])])


def match(edf_path):
    import mne
    lsl = np.load(CALIB_BASE + "_raw.npz")
    stamps = lsl["timestamps"]
    fs_lsl = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    lsl_x = _prep(lsl["data"], fs_lsl)
    varying = lsl["data"].std(axis=1) > 0  # skip flat stream slots (e.g. REF)

    edf = mne.io.read_raw(edf_path, preload=True, verbose=False)
    edf_labels = [_clean_label(ch) for ch in edf.ch_names]
    edf_x = _prep(edf.get_data(), edf.info["sfreq"])
    edf_ok = edf.get_data().std(axis=1) > 0

    # Coarse alignment: the two recordings start at different moments.
    lag = _best_lag(_gfp(edf_x), _gfp(lsl_x[varying]), max_lag=len(lsl_x[0]))
    # lag>0: LSL content appears `lag` samples later in the EDF
    win = int(WINDOW_S * FS_COMMON)
    corr_sum = np.zeros((lsl_x.shape[0], edf_x.shape[0]))
    n_windows = 0
    for start in range(0, lsl_x.shape[1] - win, win):
        l_seg = lsl_x[:, start:start + win]
        e_start = start + lag
        if e_start - 50 < 0 or e_start + win + 50 > edf_x.shape[1]:
            continue
        # fine = offset of l_seg inside the padded EDF segment (50 = no drift since coarse alignment)
        fine = _best_lag(_gfp(edf_x[:, e_start - 50:e_start + win + 50]),
                         _gfp(l_seg[varying]), max_lag=100)
        e_seg = edf_x[:, e_start - 50 + fine: e_start - 50 + fine + win]
        if e_seg.shape[1] != win:
            continue
        l_n = (l_seg - l_seg.mean(1, keepdims=True)) / (l_seg.std(1, keepdims=True) + 1e-12)
        e_n = (e_seg - e_seg.mean(1, keepdims=True)) / (e_seg.std(1, keepdims=True) + 1e-12)
        corr_sum += l_n @ e_n.T / win
        n_windows += 1

    if n_windows == 0:
        raise RuntimeError("Couldn't find overlapping time between the two recordings -- was EEGStudio "
                           "recording during the whole LSL capture?")
    corr = corr_sum / n_windows
    corr[:, ~edf_ok] = 0

    print(f"Compared {n_windows} x {WINDOW_S:.0f}s windows (EDF offset {lag / FS_COMMON:+.1f}s).\n")
    print(f"{'LSL slot':>8}  {'-> electrode':<12} {'corr':>6}  {'2nd best':<16} note")
    mapping, weak = [], []
    for i in range(lsl_x.shape[0]):
        if not varying[i]:
            mapping.append(f"FLAT{i + 1}")
            print(f"{i + 1:>8}  {'(flat)':<12}")
            continue
        order = np.argsort(np.abs(corr[i]))[::-1]
        best, second = order[0], order[1]
        c1, c2 = corr[i, best], corr[i, second]
        note = "" if abs(c1) >= MIN_CORR else "WEAK -- check"
        if abs(c1) < MIN_CORR:
            weak.append(i + 1)
        mapping.append(edf_labels[best])
        print(f"{i + 1:>8}  {edf_labels[best]:<12} {c1:6.2f}  {edf_labels[second]:<8}({c2:.2f})  {note}")

    dupes = {m for m in mapping if mapping.count(m) > 1 and not m.startswith("FLAT")}
    if dupes:
        print(f"\nWARNING: electrode(s) {sorted(dupes)} matched to more than one LSL slot.")
    if weak or dupes:
        print("Not saving the map -- rerun the calibration with more varied activity (taps on "
              "different electrodes), or send me this output.")
        return

    with open(MAP_PATH, "w") as f:
        json.dump({"channel_order": mapping, "source_edf": os.path.basename(edf_path),
                   "min_corr": float(np.min([abs(corr[i]).max() for i in range(len(mapping)) if varying[i]]))},
                  f, indent=2)
    print(f"\nAll slots matched confidently. Saved -> {MAP_PATH}")
    print("lsl_recorder.py will use this mapping from now on.")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "record":
        record()
    elif len(sys.argv) >= 3 and sys.argv[1] == "match":
        match(sys.argv[2])
    else:
        print(__doc__)
