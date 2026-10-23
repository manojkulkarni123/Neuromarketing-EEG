"""
Loads a recorded neuromarketing session (.xdf from an LSL recording tool like
LabRecorder, or .edf as a fallback) into an MNE Raw object, then cleans it:
bandpass filter + ICA-based eye-blink removal.

XDF is preferred because it already has the EEG stream and our task's marker
stream on one common clock -- no post-hoc alignment regression needed (unlike
the manual-marker EDF path in lab_task/align_sync.py). If only an EDF export
is available, falls back to MNE's own EDF reader (more robust than this
project's hand-rolled edf_reader.py, which stays as-is for the legacy
pygame/manual-marker pipeline that doesn't have MNE installed).

Run standalone to sanity-check a file:
    python pipeline/preprocess_mne.py <path.xdf|path.edf>
"""
import os
import sys

import numpy as np
import mne

FRONTAL_CHANNEL_HINTS = ["FP1", "FP2", "FPZ"]  # used as an EOG proxy if no real EOG channel exists
CORR_THRESHOLD = 0.5  # |corr| (both <4Hz) between an IC and the blink proxy above which it's excluded.
# Was 0.3 on full-band signals; on live dry-cap data that flagged 3 of 15 components at r~0.55,
# which looked like shared noise rather than 3 distinct blink sources. A true blink IC
# correlates ~0.9+ with the proxy (verified on synthetic data).


def _find_eeg_stream(streams):
    """XDF files can contain multiple streams (EEG, markers, maybe others).
    Pick the one that looks like continuous multi-channel EEG: many channels,
    a real sampling rate, and far more samples than a sparse marker stream."""
    eeg_candidates = [s for s in streams if float(s["info"]["nominal_srate"][0]) > 0]
    if not eeg_candidates:
        raise RuntimeError("No stream with a nonzero sampling rate found -- can't identify the EEG stream.")
    return max(eeg_candidates, key=lambda s: int(s["info"]["channel_count"][0]))


def _find_marker_streams(streams, eeg_stream):
    return [s for s in streams if s is not eeg_stream and float(s["info"]["nominal_srate"][0]) == 0]


def load_xdf(path):
    import pyxdf
    streams, _header = pyxdf.load_xdf(path)
    eeg_stream = _find_eeg_stream(streams)

    fs = float(eeg_stream["info"]["nominal_srate"][0])
    try:
        ch_names = [c["label"][0] for c in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]]
    except (KeyError, TypeError, IndexError):
        n_ch = int(eeg_stream["info"]["channel_count"][0])
        ch_names = [f"ch{i}" for i in range(n_ch)]
        print(f"WARNING: no channel labels found in the XDF stream metadata; using generic names {ch_names}. "
              f"FAA/theta extraction needs real 10-20 labels (F3/F4/Fz/etc) -- fix this before trusting features.")

    data = np.asarray(eeg_stream["time_series"]).T  # MNE wants (n_channels, n_samples)
    data_uv_to_v = data * 1e-6  # MNE expects volts; EEG amplifiers typically report microvolts
    info = mne.create_info(ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data_uv_to_v, info)

    eeg_t0 = eeg_stream["time_stamps"][0]
    for marker_stream in _find_marker_streams(streams, eeg_stream):
        onsets = np.asarray(marker_stream["time_stamps"]) - eeg_t0
        descriptions = [str(v[0]) if isinstance(v, list) else str(v) for v in marker_stream["time_series"]]
        valid = onsets >= 0
        if valid.sum() < len(onsets):
            print(f"NOTE: {len(onsets) - valid.sum()} marker(s) had a timestamp before the EEG stream's "
                  f"start and were dropped (likely markers sent before recording began).")
        raw.set_annotations(mne.Annotations(onset=onsets[valid], duration=0, description=list(np.array(descriptions)[valid])))

    return raw


def load_edf(path):
    return mne.io.read_raw_edf(path, preload=True)


def _drop_flat_channels(raw):
    """EEGStudio's LSL stream has a 20th 'EXTRA' channel that's constant zero --
    a flat channel makes ICA rank-deficient, so drop any such channel."""
    data = raw.get_data()
    flat = [ch for ch, row in zip(raw.ch_names, data) if np.ptp(row) == 0]
    if flat:
        print(f"Dropping flat channel(s): {flat}")
        raw.drop_channels(flat)
    return raw


# Thresholds set from real sessions on this dry cap. In test1 (2026-10-19), electrodes
# with confirmed good contact read 26-52uV (Fz, Cz, Fp1, Fp2, C3, C4, F4); Pz (466uV,
# r_aux=0.86) and O2 (404uV) passed the old 500uV / 0.9 limits, yet the eyes-closed
# check showed no alpha increase at all on them -- i.e. no real brain signal.
DEAD_CORR = 0.8       # |r| with an unconnected AUX input above this = electrode picks up mostly ambient noise
MAX_STD_UV = 150.0    # good dry-cap channels sit ~25-100uV after 1-40Hz filtering


def quality_check(raw):
    """Flags scalp electrodes with no real contact. Two independent signs:
    (1) near-identical to an unconnected spare input (AUX slots) -- this is how
    dead F3/F8 showed up (r=1.00, 0.94) while looking a normal ~40uV;
    (2) amplitude far beyond physiological range (e.g. Fp1 at 1,640uV while
    Fp2 read 26uV). Returns the list of channels to treat as bad."""
    r = raw.copy().filter(1, 40, picks="all", verbose=False)
    trim = int(r.info["sfreq"])
    eeg = [ch for ch, t in zip(r.ch_names, r.get_channel_types()) if t == "eeg"]
    aux = [ch for ch in r.ch_names if ch.upper().startswith("AUX")]
    aux_d = [r.get_data(picks=[a])[0][trim:-trim] for a in aux if r.get_data(picks=[a]).std() > 0]
    bad, rows = [], []
    for ch in eeg:
        x = r.get_data(picks=[ch])[0][trim:-trim]
        std_uv = x.std() * 1e6
        aux_r = max((abs(np.corrcoef(x, a)[0, 1]) for a in aux_d), default=0.0)
        reason = ("matches unconnected input" if aux_r > DEAD_CORR else
                  "amplitude far too high" if std_uv > MAX_STD_UV else "")
        if reason:
            bad.append(ch)
        rows.append((ch, std_uv, aux_r, reason))
    print("Electrode quality check (1-40Hz):")
    for ch, s, a, reason in rows:
        print(f"  {ch:4s} {s:8.1f}uV  r_aux={a:.2f}  {'BAD: ' + reason if reason else 'ok'}")
    return bad


def load_recording(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fif":  # written by lab_task_psychopy/lsl_recorder.py -- the primary path now
        raw = mne.io.read_raw_fif(path, preload=True)
    elif ext == ".xdf":
        raw = load_xdf(path)
    elif ext == ".edf":
        raw = load_edf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext} (expected .fif, .xdf or .edf)")
    return _drop_flat_channels(raw)


def clean_raw(raw, n_ica_components=15, random_state=0):
    """Bandpass filter + ICA-based eye-blink removal. Returns a new cleaned Raw."""
    raw = raw.copy().load_data()
    raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)

    if raw.get_montage() is None:
        try:
            raw.set_montage("standard_1020", on_missing="warn")
        except Exception as e:
            print(f"NOTE: couldn't set a standard 10-20 montage automatically ({e}). "
                  f"ICA will still run, but scalp-position-based diagnostics won't be available.")

    n_good_eeg = sum(1 for ch, t in zip(raw.ch_names, raw.get_channel_types())
                     if t == "eeg" and ch not in raw.info["bads"])
    n_components = min(n_ica_components, n_good_eeg - 1)
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state,
                                 max_iter=1000, fit_params=dict(tol=1e-3))
    ica.fit(raw, verbose=False)

    # NOTE: deliberately not using ica.find_bads_eog() -- tested against synthetic
    # data with a known, obvious blink component (correlation -0.999 with the
    # frontal proxy channel) and it returned an empty list regardless, silently.
    # Computing the correlation directly and thresholding on it is more
    # transparent and was verified to actually work on that same test case.
    # Blink proxy = mean of the good frontal-pole channels, and the comparison is
    # done on the <4Hz band only. Blinks are slow; FAA alpha is 8-13Hz. With F3 and
    # F8 dead on this cap, FAA runs on Fp1/Fp2 -- the same channels used as the blink
    # proxy -- so correlating full-band signals would risk stripping real Fp1/Fp2
    # alpha along with the blinks.
    eog_chs = [ch for ch in raw.ch_names
               if ch.upper() in FRONTAL_CHANNEL_HINTS and ch not in raw.info["bads"]]
    exclude = []
    if eog_chs:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, 4.0, btype="low", fs=raw.info["sfreq"], output="sos")
        sources = sosfiltfilt(sos, ica.get_sources(raw).get_data(), axis=1)
        eog_signal = sosfiltfilt(sos, raw.get_data(picks=eog_chs).mean(axis=0))
        corrs = np.array([np.corrcoef(sources[i], eog_signal)[0, 1] for i in range(sources.shape[0])])
        exclude = list(np.where(np.abs(corrs) > CORR_THRESHOLD)[0])
        print(f"ICA: blink proxy = mean of {eog_chs} (<4Hz), component correlations "
              f"{np.round(corrs, 3).tolist()}, flagged {exclude} (|corr| > {CORR_THRESHOLD}) as blink-related.")
    else:
        print(f"WARNING: no frontal channel found among {FRONTAL_CHANNEL_HINTS} to use as an EOG proxy -- "
              f"no components auto-excluded. Inspect ica.plot_components() manually before trusting this data.")

    ica.exclude = exclude
    cleaned = raw.copy()
    ica.apply(cleaned, verbose=False)
    return cleaned


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline/preprocess_mne.py <path.xdf|path.edf>")
        sys.exit(1)
    raw = load_recording(sys.argv[1])
    print(f"Loaded: {len(raw.ch_names)} channels, {raw.n_times} samples @ {raw.info['sfreq']}Hz")
    print(f"Channels: {raw.ch_names}")
    print(f"Annotations: {len(raw.annotations)}")
    cleaned = clean_raw(raw)
    print("Cleaning complete.")
