"""
Minimal, dependency-free EDF/EDF+ reader (no mne/pyedflib -- installing mne
here tried to downgrade the numpy this project's torch/tensorflow/sklearn
stack depends on, so this avoids that risk entirely; EDF is a simple enough
binary format to parse directly with the standard library + numpy).

EDF spec: https://www.edfplus.info/specs/edf.html
Mitsar's EEGStudio can export recordings as EDF -- if it only offers a
proprietary format instead, send a sample file and this will need adapting.
"""
import numpy as np


def _read_header_field(f, n):
    return f.read(n).decode("ascii", errors="replace").strip()


def parse_tal_bytes(raw_bytes):
    """Decode an EDF+ 'EDF Annotations' channel's raw bytes into a list of
    (onset_seconds, duration_seconds_or_None, text) tuples. TAL format:
    +onset[0x15 duration] 0x14 (text 0x14)* 0x00 -- see edfplus.info/specs.
    Mitsar/EEGStudio manual markers pressed during recording typically show up
    here with the operator's key-press label as the annotation text."""
    text = raw_bytes.decode("latin-1")
    annotations = []
    for tal in text.split("\x00"):
        if not tal:
            continue
        parts = tal.split("\x14")
        onset_duration = parts[0]
        if not onset_duration or onset_duration[0] not in "+-":
            continue
        if "\x15" in onset_duration:
            onset_str, duration_str = onset_duration.split("\x15", 1)
        else:
            onset_str, duration_str = onset_duration, None
        try:
            onset = float(onset_str)
        except ValueError:
            continue
        duration = None
        if duration_str:
            try:
                duration = float(duration_str)
            except ValueError:
                pass
        texts = [p for p in parts[1:] if p]
        for t in texts:
            annotations.append((onset, duration, t))
    return annotations


def read_edf(path):
    """Returns dict: {labels: [str], fs: {label: float}, signals: {label: np.ndarray}, duration_s: float}."""
    with open(path, "rb") as f:
        _version = _read_header_field(f, 8)
        _patient_id = _read_header_field(f, 80)
        _recording_id = _read_header_field(f, 80)
        _start_date = _read_header_field(f, 8)
        _start_time = _read_header_field(f, 8)
        n_header_bytes = int(_read_header_field(f, 8))
        _reserved = _read_header_field(f, 44)
        n_records = int(_read_header_field(f, 8))
        record_duration_s = float(_read_header_field(f, 8))
        n_signals = int(_read_header_field(f, 4))

        labels = [_read_header_field(f, 16) for _ in range(n_signals)]
        _transducer = [_read_header_field(f, 80) for _ in range(n_signals)]
        _phys_dim = [_read_header_field(f, 8) for _ in range(n_signals)]
        phys_min = [float(_read_header_field(f, 8)) for _ in range(n_signals)]
        phys_max = [float(_read_header_field(f, 8)) for _ in range(n_signals)]
        dig_min = [float(_read_header_field(f, 8)) for _ in range(n_signals)]
        dig_max = [float(_read_header_field(f, 8)) for _ in range(n_signals)]
        _prefilter = [_read_header_field(f, 80) for _ in range(n_signals)]
        n_samples_per_record = [int(_read_header_field(f, 8)) for _ in range(n_signals)]
        _reserved2 = [_read_header_field(f, 32) for _ in range(n_signals)]

        assert f.tell() == n_header_bytes, f"header parse mismatch: read {f.tell()}, expected {n_header_bytes}"

        raw = np.frombuffer(f.read(), dtype="<i2")  # little-endian int16

    samples_per_record_total = sum(n_samples_per_record)
    expected_len = samples_per_record_total * n_records
    if raw.size < expected_len:
        # some recordings are truncated / n_records == -1 (unknown); use what's available
        n_records = raw.size // samples_per_record_total
        raw = raw[: n_records * samples_per_record_total]

    raw = raw.reshape(n_records, samples_per_record_total)

    signals = {}
    fs = {}
    annotations = []
    offset = 0
    for i, label in enumerate(labels):
        nsr = n_samples_per_record[i]
        raw_int16 = raw[:, offset:offset + nsr]
        offset += nsr

        if label.strip().lower() == "edf annotations":
            # Not a real signal -- these int16 samples are actually raw bytes
            # of ASCII TAL-encoded annotations (manual EEGStudio markers etc).
            annotations.extend(parse_tal_bytes(raw_int16.tobytes()))
            continue

        digital = raw_int16.reshape(-1).astype(np.float64)
        d_range = dig_max[i] - dig_min[i]
        p_range = phys_max[i] - phys_min[i]
        scale = p_range / d_range if d_range != 0 else 1.0
        physical = (digital - dig_min[i]) * scale + phys_min[i]

        signals[label] = physical.astype(np.float32)
        fs[label] = nsr / record_duration_s

    annotations.sort(key=lambda a: a[0])
    return {
        "labels": [l for l in labels if l.strip().lower() != "edf annotations"],
        "fs": fs,
        "signals": signals,
        "duration_s": n_records * record_duration_s,
        "n_records": n_records,
        "annotations": annotations,  # list of (onset_seconds, duration_or_None, text)
    }


if __name__ == "__main__":
    import sys
    d = read_edf(sys.argv[1])
    print(f"Duration: {d['duration_s']:.1f}s, {len(d['labels'])} channels")
    for label in d["labels"]:
        print(f"  {label}: fs={d['fs'][label]:.1f}Hz, n_samples={len(d['signals'][label])}")
    if d["annotations"]:
        print(f"\n{len(d['annotations'])} annotations/markers found:")
        for onset, duration, text in d["annotations"]:
            print(f"  t={onset:.3f}s{f' (dur {duration:.3f}s)' if duration else ''}: {text}")
