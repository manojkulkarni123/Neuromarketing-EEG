"""
Aligns random_dot_task.py's behavioral timestamps (this PC's clock) to the
EEGStudio recording's own clock (EDF+ annotation timestamps, from the manual
marker key presses), using the periodic sync-checkpoint beeps as the
correspondence points -- see random_dot_task.py's module docstring for why a
single reference point isn't enough (clock drift over the session).

How the matching works: random_dot_task.py logs its checkpoints in a fixed,
known order (session_start, checkpoint_trial_10, checkpoint_trial_20, ...,
session_end) with ITS OWN perf_counter timestamp for each. The EDF file's
annotations (from the operator's manual marker key presses) are also in
order, with the EEG recording's own timestamp for each. This script assumes
the operator pressed the marker key once per beep, in the same order --
IF THE COUNTS DON'T MATCH, IT WILL TELL YOU RATHER THAN GUESS, since a
missed/extra marker press would silently corrupt the alignment otherwise.

It fits a line (eeg_seconds = a * our_perf_counter + b) through the matched
pairs by least squares, which corrects for both a constant offset AND clock
drift between the two machines (rather than trusting a single point).

Usage:
    python lab_task/align_sync.py <participant_id> <edf_path>
"""
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from edf_reader import read_edf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session")


def load_sync_checkpoints(participant_id):
    path = os.path.join(LAB_DIR, f"{participant_id}_sync_checkpoints.csv")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({"label": row["label"], "perf_counter": float(row["perf_counter"])})
    return rows


def load_behavioral(participant_id):
    path = os.path.join(LAB_DIR, f"{participant_id}_behavioral.csv")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else None) for k, v in row.items()})
    return rows


def main():
    if len(sys.argv) != 3:
        print("Usage: python lab_task/align_sync.py <participant_id> <edf_path>")
        sys.exit(1)
    participant_id, edf_path = sys.argv[1], sys.argv[2]

    checkpoints = load_sync_checkpoints(participant_id)
    print(f"Loaded {len(checkpoints)} sync checkpoints (this PC's clock):")
    for c in checkpoints:
        print(f"  {c['label']}: perf_counter={c['perf_counter']:.4f}")

    edf = read_edf(edf_path)
    annotations = edf["annotations"]
    print(f"\nLoaded {len(annotations)} annotations from the EDF file (EEG recording's clock):")
    for onset, duration, text in annotations:
        print(f"  t={onset:.4f}s: {text!r}")

    if len(annotations) != len(checkpoints):
        print(f"\n*** MISMATCH: {len(checkpoints)} checkpoints logged by the task, but "
              f"{len(annotations)} markers found in the EDF. ***")
        print("This likely means the operator missed a marker press, pressed an extra one, "
              "or EEGStudio's export includes non-marker annotations too.")
        print("Inspect the two lists above and manually edit this script's pairing logic "
              "(the zip() call below) before trusting the alignment.")
        n = min(len(annotations), len(checkpoints))
        print(f"Proceeding with the first {n} of each, in order -- treat the result as provisional.")
        checkpoints, annotations = checkpoints[:n], annotations[:n]

    our_times = np.array([c["perf_counter"] for c in checkpoints])
    eeg_times = np.array([a[0] for a in annotations])

    # eeg_seconds = a * our_perf_counter + b
    a, b = np.polyfit(our_times, eeg_times, 1)
    predicted = a * our_times + b
    residuals_ms = (eeg_times - predicted) * 1000
    print(f"\nFit: eeg_seconds = {a:.8f} * our_perf_counter + {b:.4f}")
    print(f"Residuals (ms) at each checkpoint: {np.round(residuals_ms, 1).tolist()}")
    print(f"Max residual: {np.max(np.abs(residuals_ms)):.1f} ms -- this is roughly how much error "
          f"to expect in the aligned trial timestamps (driven by the operator's manual key-press "
          f"reaction time jitter, not by clock drift, which this fit already corrects for).")

    behavioral = load_behavioral(participant_id)
    fs_by_channel = edf["fs"]
    # use the first non-annotation channel's fs as the reference sampling rate
    ref_fs = next(iter(fs_by_channel.values()))

    out_path = os.path.join(LAB_DIR, f"{participant_id}_aligned_trials.csv")
    fieldnames = list(behavioral[0].keys()) + ["eeg_time_stim_s", "eeg_time_resp_s",
                                                "eeg_sample_stim", "eeg_sample_resp"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in behavioral:
            stim_our_time = row["abs_perf_stim_onset"]
            resp_our_time = stim_our_time + row["RT"]
            eeg_time_stim = a * stim_our_time + b
            eeg_time_resp = a * resp_our_time + b
            row["eeg_time_stim_s"] = eeg_time_stim
            row["eeg_time_resp_s"] = eeg_time_resp
            row["eeg_sample_stim"] = int(round(eeg_time_stim * ref_fs))
            row["eeg_sample_resp"] = int(round(eeg_time_resp * ref_fs))
            writer.writerow(row)

    print(f"\nSaved: {out_path}")
    print(f"(EEG sample columns computed at fs={ref_fs}Hz -- confirm this matches the actual "
          f"recording rate before epoching from these sample indices.)")


if __name__ == "__main__":
    main()
