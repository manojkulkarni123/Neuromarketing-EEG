"""
Records EEGStudio's LSL Outlet stream from inside our own task process, and
logs task markers on the same LSL clock -- so EEG and markers come out
already aligned, in one file.

Why this direction: EEGStudio's LSL *Inlet* (receiving our markers) never
connected, despite text markers, int32 markers, a continuous marker channel,
a KnownPeers config in its own folder, and firewall off. Its *Outlet*
(broadcasting its EEG) is the direction acquisition software is normally
built for, so we record it ourselves instead.

Output (per session):
  <name>_raw.fif  -- MNE Raw with EEG + marker annotations (what preprocess_mne.py loads)
  <name>_raw.npz  -- raw backup: data, LSL timestamps, channel labels, markers

Timing: EEG samples arrive with EEGStudio's LSL timestamps; the inlet applies
LSL clock-sync + dejitter, putting them on this machine's local_clock().
Markers are stamped with local_clock() directly, so both share one clock.
"""
import os
import json
import time
import threading

import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_streams, local_clock, proc_clocksync, proc_dejitter

OWN_STREAM_NAMES = {"PsychoPyMarkers"}

# EEGStudio's 'Mitsar' LSL stream carries no channel labels in its metadata.
# Order = EEGStudio's ON-SCREEN montage order -- confirmed empirically by
# calibrate_channels.py (correlating our LSL capture against EEGStudio's own
# labeled EDF export: slots 1-19 matched this order at r~0.9+ where contact
# was good) and by a tap test (tapping O1 lit slot 18). NOT the numbered list
# in Settings -> Electrode Layout -- that's the amplifier's physical input
# order, and using it briefly mislabeled channels (e.g. F8's data as "F3").
# Slots 20-26 are unconnected spare inputs (they matched nothing but shared
# noise); slot 27 is flat. "Processed data" mode sends the first 20 slots,
# "Raw data (in uV)" mode sends all 27. Only the 19 scalp channels are analyzed.
CHANNEL_ORDER = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
                 "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
                 "AUX20", "AUX21", "AUX22", "AUX23", "AUX24", "AUX25", "AUX26", "FLAT27"]
SCALP_CHANNELS = set(CHANNEL_ORDER[:19])
MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "channel_map.json")


def _find_eeg_stream(timeout):
    candidates = [s for s in resolve_byprop("type", "EEG", timeout=timeout)
                  if s.name() not in OWN_STREAM_NAMES]
    if not candidates:  # EEGStudio may not label its stream type as 'EEG'
        candidates = [s for s in resolve_streams(wait_time=timeout)
                      if s.name() not in OWN_STREAM_NAMES and s.nominal_srate() > 0]
    if not candidates:
        raise RuntimeError("No EEG stream found on LSL. In EEGStudio: Add-on tab -> click 'LSL Outlet', "
                           "make sure acquisition is running, then retry.")
    # The same 'Mitsar' stream often resolves twice (one entry per network interface,
    # or a stale copy after toggling LSL Outlet) -- and a stale entry delivers nothing.
    # Pick the first candidate that actually sends samples within a short probe.
    for s in sorted(candidates, key=lambda s: -s.channel_count()):
        probe = StreamInlet(s)
        try:
            _, stamps = probe.pull_chunk(timeout=2.0)
            if not stamps:
                time.sleep(0.5)
                _, stamps = probe.pull_chunk(timeout=1.0)
        finally:
            probe.close_stream()
        if stamps:
            return s
    raise RuntimeError(f"Found {len(candidates)} EEG stream(s) but none is sending data. In EEGStudio, "
                       "toggle LSL Outlet off/on, wait ~5s, and retry.")


class LSLRecorder:
    def __init__(self, resolve_timeout=10.0):
        info = _find_eeg_stream(resolve_timeout)
        self.inlet = StreamInlet(info, max_buflen=360, processing_flags=proc_clocksync | proc_dejitter)
        full = self.inlet.info(timeout=5)
        self.sfreq = full.nominal_srate()
        self.name = full.name()
        self.ch_labels = self._read_labels(full)
        self._chunks, self._stamps = [], []
        self.markers = []  # (local_clock time, label)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pull_loop, daemon=True)
        print(f"[recorder] found '{self.name}': {len(self.ch_labels)} ch @ {self.sfreq}Hz, "
              f"labels={self.ch_labels}")

    @staticmethod
    def _read_labels(info):
        labels, ch = [], info.desc().child("channels").child("channel")
        while not ch.empty():
            labels.append(ch.child_value("label"))
            ch = ch.next_sibling()
        if len(labels) == info.channel_count() and all(labels):
            return labels
        if os.path.exists(MAP_PATH):  # written by calibrate_channels.py -- authoritative when present
            with open(MAP_PATH) as f:
                calibrated = json.load(f)["channel_order"]
            if len(calibrated) == info.channel_count():
                print(f"[recorder] using calibrated channel map ({MAP_PATH})")
                return calibrated
            print(f"[recorder] WARNING: channel_map.json has {len(calibrated)} channels but stream has "
                  f"{info.channel_count()} (EEGStudio data mode changed?) -- rerun calibrate_channels.py.")
        if info.channel_count() <= len(CHANNEL_ORDER):
            print(f"[recorder] no channel labels in stream metadata -- applying the amplifier's hardware "
                  f"order (first {info.channel_count()} of CHANNEL_ORDER). Verify with blink_test.py.")
            return list(CHANNEL_ORDER[:info.channel_count()])
        print(f"[recorder] WARNING: no labels and {info.channel_count()} channels (expected "
              f"{len(CHANNEL_ORDER)}) -- using ch0..chN; FAA/theta won't find F3/F4 until mapped.")
        return [f"ch{i}" for i in range(info.channel_count())]

    def _pull_loop(self):
        while not self._stop.is_set():
            chunk, stamps = self.inlet.pull_chunk(timeout=0.2)
            if stamps:
                self._chunks.append(np.asarray(chunk, dtype=np.float64))
                self._stamps.extend(stamps)

    def start(self):
        self._thread.start()

    def mark(self, label):
        self.markers.append((local_clock(), label))

    def stop(self):
        t_stop = local_clock()
        self._stop.set()
        self._thread.join(timeout=5)
        # Drain samples up to the stop moment, so a marker sent right before stop()
        # isn't orphaned past the end. Bounded by t_stop (and a 3s deadline) --
        # a live stream never runs empty, so "drain until empty" would loop forever.
        deadline = local_clock() + 3.0
        while (not self._stamps or self._stamps[-1] < t_stop) and local_clock() < deadline:
            chunk, stamps = self.inlet.pull_chunk(timeout=0.5)
            if stamps:
                self._chunks.append(np.asarray(chunk, dtype=np.float64))
                self._stamps.extend(stamps)

    def n_samples(self):
        return len(self._stamps)

    def save(self, out_base, units_scale=1e-6):
        """units_scale converts EEGStudio's units to volts (MNE's unit). 1e-6 assumes
        the stream is in microvolts -- sanity-check typical amplitudes afterward."""
        import mne
        if not self._chunks:
            raise RuntimeError("No EEG samples were recorded -- was EEGStudio's LSL Outlet active?")
        data = np.vstack(self._chunks).T  # (n_channels, n_samples)
        stamps = np.asarray(self._stamps)

        np.savez_compressed(out_base + "_raw.npz", data=data, timestamps=stamps,
                            ch_labels=np.array(self.ch_labels), sfreq=self.sfreq,
                            marker_times=np.array([m[0] for m in self.markers]),
                            marker_labels=np.array([m[1] for m in self.markers]))

        # EEGStudio's stream claims 250Hz, but measured delivery varies run to run
        # (250.0Hz in one test, 235.8Hz in another, with no gaps). Assuming the
        # nominal rate made the saved timeline drift ~6% from real time -- markers
        # landed seconds off and 10Hz alpha would read as ~10.6Hz. So: use the
        # MEASURED rate for the Raw, and place markers by the real per-sample
        # LSL timestamps rather than by index/nominal-rate arithmetic.
        n = data.shape[1]
        span = stamps[-1] - stamps[0]
        sfreq_eff = (n - 1) / span if span > 0 else self.sfreq
        max_gap_ms = 1000 * np.diff(stamps).max() if n > 1 else 0.0

        # Only the 19 scalp electrodes are 'eeg'; REF/A1/A2/aux are 'misc' so MNE's
        # filtering/ICA/feature steps ignore them automatically.
        scalp_upper = {c.upper() for c in SCALP_CHANNELS}
        if scalp_upper & {c.upper() for c in self.ch_labels}:
            ch_types = ["eeg" if ch.upper() in scalp_upper else "misc" for ch in self.ch_labels]
        else:  # unmapped fallback labels (ch0..chN) -- keep everything as eeg
            ch_types = ["eeg"] * len(self.ch_labels)
        info = mne.create_info(self.ch_labels, sfreq=sfreq_eff, ch_types=ch_types)
        raw = mne.io.RawArray(data * units_scale, info, verbose=False)
        m_times = np.array([m[0] for m in self.markers])
        keep = (m_times >= stamps[0]) & (m_times <= stamps[-1])
        idx = np.searchsorted(stamps, m_times[keep])
        raw.set_annotations(mne.Annotations(onset=idx / sfreq_eff, duration=0,
                                            description=[m[1] for m, k in zip(self.markers, keep) if k]))
        raw.save(out_base + "_raw.fif", overwrite=True, verbose=False)

        dropped = [m[1] for m, k in zip(self.markers, keep) if not k]
        print(f"[recorder] saved {n} samples x {data.shape[0]} ch over {span:.1f}s -> {out_base}_raw.fif")
        print(f"[recorder] rate: nominal {self.sfreq:.1f}Hz, measured {sfreq_eff:.1f}Hz (used); "
              f"largest gap between samples {max_gap_ms:.1f}ms")
        print(f"[recorder] markers kept {int(keep.sum())}/{len(self.markers)}"
              + (f" -- outside recording: {dropped}" if dropped else ""))
        if abs(sfreq_eff - self.sfreq) / self.sfreq > 0.02:
            print("[recorder] NOTE: measured rate differs from nominal by >2% -- handled, but report "
                  "it in the Methods section.")
        if max_gap_ms > 50:
            print("[recorder] WARNING: a gap >50ms suggests dropped data -- check this session before using it.")
        return out_base + "_raw.fif"
