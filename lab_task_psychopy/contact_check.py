"""
Live cap-contact check (read-only: pulls a few seconds from the EEG LSL stream, saves nothing).

Per electrode, after removing mains hum (50/100 Hz notch) and band-passing 1-40 Hz:
  good  < 100 uV and not a copy of an unconnected input
  weak  < 150 uV and not a copy of an unconnected input
  bad   >= 150 uV, or |r| >= 0.8 with a spare (AUX) input = no real contact
Same limits as the analysis (preprocess_mne.quality_check: 150 uV, r 0.8).
Ready to record = Fz and Cz not bad (engagement / effort) and at least one of Pz, P3, P4
not bad (visual attention).

Run directly for a console check:  .venv\\Scripts\\python.exe lab_task_psychopy\\contact_check.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOOD_UV, WEAK_UV, AUX_R = 100.0, 150.0, 0.8
NEED_ALL = ["Fz", "Cz"]
NEED_ONE = ["Pz", "P3", "P4"]


def check(seconds=8.0):
    try:
        from pylsl import resolve_byprop, StreamInlet
        from scipy.signal import iirnotch, butter, filtfilt
    except ImportError as e:
        return {"ok": False, "error": f"missing package: {e}"}
    streams = [s for s in resolve_byprop("type", "EEG", timeout=2.5)]
    if not streams:
        return {"ok": False, "error": "No EEG stream. In EEGStudio: start acquisition and turn on LSL Outlet."}
    info = sorted(streams, key=lambda s: -s.channel_count())[0]
    try:
        labels = json.load(open(os.path.join(HERE, "channel_map.json")))["channel_order"]
    except (OSError, KeyError, ValueError):
        return {"ok": False, "error": "channel_map.json is missing - run calibrate_channels.py"}
    if info.channel_count() != len(labels):
        return {"ok": False, "error": f"Stream has {info.channel_count()} channels but the channel map has {len(labels)}. "
                                      f"Set EEGStudio to 'Raw data (in uV)'."}
    inlet = StreamInlet(info)
    chunks, stamps, t0 = [], [], time.time()
    while time.time() - t0 < seconds:
        c, ts = inlet.pull_chunk(timeout=0.5)
        if ts:
            chunks.append(np.asarray(c))
            stamps += ts
    inlet.close_stream()
    if len(stamps) < 500:
        return {"ok": False, "error": "The stream is visible but sends no data. In EEGStudio, toggle LSL Outlet off and on."}
    x = np.vstack(chunks).T.astype(float)
    sf = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    for f0 in (50, 100):
        if f0 < sf / 2 - 1:
            b, a = iirnotch(f0, 30, sf)
            x = filtfilt(b, a, x, axis=1)
    b, a = butter(4, [1, 40], btype="band", fs=sf)
    y = filtfilt(b, a, x, axis=1)[:, int(sf):-int(sf)]
    std = y.std(axis=1)
    aux = [i for i, c in enumerate(labels) if c.upper().startswith("AUX") and 0 < std[i] < 1e5]
    scalp = [i for i, c in enumerate(labels) if not c.upper().startswith(("AUX", "FLAT"))]
    out = []
    for i in scalp:
        r = max((abs(float(np.corrcoef(y[i], y[j])[0, 1])) for j in aux), default=0.0)
        uv = float(std[i])
        state = "bad" if (uv >= WEAK_UV or r >= AUX_R) else ("weak" if uv >= GOOD_UV else "good")
        out.append({"name": labels[i], "uv": round(uv, 1), "r_aux": round(r, 2), "state": state,
                    "why": "no contact (copies an unconnected input)" if r >= AUX_R else
                           "too noisy" if uv >= WEAK_UV else ""})
    st = {c["name"]: c["state"] for c in out}
    cc = np.corrcoef(y[scalp])
    shared = float(np.median(cc[np.triu_indices(len(scalp), 1)]))
    ready = all(st.get(n) != "bad" for n in NEED_ALL) and any(st.get(n) != "bad" for n in NEED_ONE)
    n_ok = sum(1 for c in out if c["state"] != "bad")
    if n_ok == 0 or shared > 0.9:
        msg = "All electrodes carry the same large signal: check the REF and GND electrodes first."
        ready = False
    elif ready:
        msg = f"{n_ok} of {len(out)} electrodes usable. Ready to record."
    else:
        miss = [n for n in NEED_ALL if st.get(n) == "bad"]
        if not any(st.get(n) != "bad" for n in NEED_ONE):
            miss.append("one of Pz / P3 / P4")
        msg = "Not ready: re-seat " + ", ".join(miss) + "."
    return {"ok": True, "rate_hz": round(sf, 1), "seconds": round(len(stamps) / sf, 1), "channels": out,
            "usable": n_ok, "shared_noise": round(shared, 2), "ready": bool(ready), "message": msg}


if __name__ == "__main__":
    r = check(float(sys.argv[1]) if len(sys.argv) > 1 else 8.0)
    if not r.get("ok"):
        sys.exit(r.get("error"))
    for c in r["channels"]:
        print(f"  {c['name']:4s} {c['uv']:8.1f} uV  r_aux={c['r_aux']:.2f}  {c['state']} {c['why']}")
    print(r["message"])
