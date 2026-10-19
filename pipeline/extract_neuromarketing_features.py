"""
Sliding-window Frontal Alpha Asymmetry (FAA) + frontal theta power timeline --
the system's job here is to produce the complete, accurate chart of brain
activity across an ad's duration. Matching specific chart moments against the
ad's content (logo appears, product shown, etc.) is a manual, after-the-fact
step you do by eyeballing the plot -- not something this script does.

FAA sign convention (Davidson-style): FAA = ln(alpha power at F4) - ln(alpha
power at F3). Alpha power is inversely related to cortical activation, so a
MORE NEGATIVE FAA (F4 alpha < F3 alpha, i.e. more right-side activation)
conventionally indicates greater LEFT-hemisphere activation = approach /
positive engagement. A MORE POSITIVE FAA indicates avoidance/negative
engagement. Double-check this convention against your source paper before
writing the Results section -- sign conventions vary across the FAA
literature and getting this backwards is an easy, paper-invalidating mistake.

Model training (classify_engagement.py) is intentionally deferred -- this
script's output (the timeline CSV + plot, and a summary feature per ad) is
useful on its own before any classifier exists.

Run standalone to sanity-check against a file:
    python pipeline/extract_neuromarketing_features.py <path.xdf|path.edf> [--start S] [--end S]
"""
import os
import sys

import numpy as np
from mne.time_frequency import psd_array_welch

from preprocess_mne import load_recording, clean_raw, quality_check

ALPHA_BAND = (8, 13)
THETA_BAND = (4, 8)
WINDOW_S = 3.0
STEP_S = 1.0

# FAA uses ONE homologous left/right pair: the first pair in this list where
# both electrodes are present and not marked bad. Never mixes pairs or averages
# a bad electrode in -- with this cap F3 is dead (and F8 was during
# calibration), and an earlier version averaged F3 into "left" regardless.
# F7/F8 is an established FAA pair in its own right; Fp1/Fp2 is a weaker last
# resort (more eye-movement contamination, leans on ICA) -- disclose whichever
# pair was used in the paper.
FAA_PAIRS = [("F3", "F4"), ("F7", "F8"), ("Fp1", "Fp2")]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session")


def _find(ch_names, name, bad):
    """Exact, case-insensitive channel lookup that skips bad channels."""
    bad_u = {b.upper() for b in bad}
    for ch in ch_names:
        if ch.upper() == name.upper() and ch.upper() not in bad_u:
            return ch
    return None


def choose_faa_pair(ch_names, bad=(), pairs=FAA_PAIRS):
    for left, right in pairs:
        l, r = _find(ch_names, left, bad), _find(ch_names, right, bad)
        if l and r:
            return l, r
    raise RuntimeError(f"No usable FAA pair among {pairs} (bad: {list(bad)}). "
                       f"Channels available: {ch_names}")


def choose_theta_channels(ch_names, bad=()):
    """Fz, plus F3 and F4 only if BOTH are good (keeps the ROI left/right symmetric)."""
    chans = [c for c in [_find(ch_names, "Fz", bad)] if c]
    f3, f4 = _find(ch_names, "F3", bad), _find(ch_names, "F4", bad)
    if f3 and f4:
        chans += [f3, f4]
    if not chans:
        raise RuntimeError(f"No usable frontal-midline channel for theta (bad: {list(bad)}).")
    return chans


def _band_power(data_segment, sfreq, band):
    """data_segment: (n_channels, n_samples) for one window. Returns mean power
    in `band`, averaged across the given channels."""
    psds, freqs = psd_array_welch(data_segment, sfreq=sfreq, fmin=band[0], fmax=band[1],
                                   n_per_seg=min(data_segment.shape[-1], int(sfreq * 2)), verbose=False)
    return psds.mean(axis=-1)  # (n_channels,) -- mean power across frequency bins in-band


def sliding_window_features(raw, window_s=WINDOW_S, step_s=STEP_S,
                             alpha_band=ALPHA_BAND, theta_band=THETA_BAND, bad=None):
    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # (n_channels, n_samples), volts
    ch_names = raw.ch_names
    n_samples = data.shape[1]
    bad = list(raw.info["bads"]) if bad is None else list(bad)

    try:
        left, right = choose_faa_pair(ch_names, bad)
        left_ch, right_ch = [left], [right]
    except RuntimeError as e:
        print(f"FAA SKIPPED: {e}")  # no valid pair -> report no FAA rather than a number from dead electrodes
        left_ch, right_ch = [], []
    mid_ch = choose_theta_channels(ch_names, bad)

    left_idx = [ch_names.index(c) for c in left_ch]
    right_idx = [ch_names.index(c) for c in right_ch]
    mid_idx = [ch_names.index(c) for c in mid_ch]

    window_samples = int(window_s * sfreq)
    step_samples = int(step_s * sfreq)

    times, faa_values, theta_values = [], [], []
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        segment = data[:, start:end]

        if left_idx:
            left_alpha = _band_power(segment[left_idx], sfreq, alpha_band).mean()
            right_alpha = _band_power(segment[right_idx], sfreq, alpha_band).mean()
            faa = np.log(right_alpha) - np.log(left_alpha)
        else:
            faa = np.nan

        theta = _band_power(segment[mid_idx], sfreq, theta_band).mean()

        times.append((start + window_samples / 2) / sfreq)
        faa_values.append(faa)
        theta_values.append(theta)

    return {
        "times_s": np.array(times),
        "faa": np.array(faa_values),
        "theta": np.array(theta_values),
        "left_channels_used": left_ch,
        "right_channels_used": right_ch,
        "midline_channels_used": mid_ch,
    }


def summary_features(timeline, tmin=None, tmax=None):
    """One number per ad for the (later) classifier -- computed from the exact
    same values as the plotted timeline, restricted to windows fully inside
    [tmin, tmax] (the ad's AD_START..AD_END) when given."""
    t = timeline["times_s"]
    half = WINDOW_S / 2
    m = np.ones_like(t, dtype=bool)
    if tmin is not None:
        m &= t - half >= tmin
    if tmax is not None:
        m &= t + half <= tmax
    faa, theta = timeline["faa"][m], timeline["theta"][m]
    has_faa = faa.size and not np.all(np.isnan(faa))
    return {
        "n_windows": int(m.sum()),
        "faa_mean": float(np.nanmean(faa)) if has_faa else None,
        "theta_mean": float(np.mean(theta)) if theta.size else None,
        "theta_peak": float(np.max(theta)) if theta.size else None,
    }


def ad_segments(annotations):
    """[(name, start, end)] from AD_START:<name> / AD_END:<name> markers."""
    starts = {a["description"].split(":", 1)[1]: a["onset"] for a in annotations
              if a["description"].startswith("AD_START:")}
    segs = []
    for a in annotations:
        if a["description"].startswith("AD_END:"):
            name = a["description"].split(":", 1)[1]
            if name in starts:
                segs.append((name, starts[name], a["onset"]))
    return segs


def plot_timeline(timeline, out_path, title="Engagement timeline", segments=()):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for ax in (ax1, ax2):
        for name, s, e in segments:
            ax.axvspan(s, e, color="tab:green", alpha=0.12)
    if segments:
        for name, s, e in segments:
            ax2.text((s + e) / 2, 0.97, name, transform=ax2.get_xaxis_transform(),
                     ha="center", va="top", fontsize=8, color="darkgreen")

    if np.all(np.isnan(timeline["faa"])):
        ax1.text(0.5, 0.5, "FAA unavailable -- no working left/right frontal electrode pair",
                 transform=ax1.transAxes, ha="center", va="center", color="gray")
        ax1.set_title(f"{title}  (green = ad playing)")
    else:
        ax1.plot(timeline["times_s"], timeline["faa"], color="tab:blue")
        ax1.axhline(0, color="gray", linewidth=0.5)
        ax1.set_title(f"{title}  (green = ad playing)\nleft={timeline['left_channels_used']} "
                      f"right={timeline['right_channels_used']} "
                      f"(negative = approach/positive engagement, positive = avoidance)")
    ax1.set_ylabel("FAA\n(ln(R alpha) - ln(L alpha))")

    ax2.plot(timeline["times_s"], timeline["theta"], color="tab:orange")
    ax2.set_ylabel(f"Frontal theta power\n({timeline['midline_channels_used']})")
    ax2.set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help=".fif (from ad_viewing_task.py), .xdf or .edf")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--bad", default="", help="comma-separated dead electrodes, e.g. F3,F8 -- "
                                               "excluded from ICA and all features")
    args = ap.parse_args()
    path = args.path

    raw = load_recording(path)
    bad = [b.strip() for b in args.bad.split(",") if b.strip()]
    missing = [b for b in bad if b not in raw.ch_names]
    if missing:
        raise SystemExit(f"--bad names not in recording: {missing}. Channels: {raw.ch_names}")
    auto_bad = quality_check(raw)
    bad = sorted(set(bad) | set(auto_bad))
    raw.info["bads"] = bad
    if bad:
        print(f"Marked bad (excluded from ICA and features): {bad}")
    if args.start is not None or args.end is not None:
        raw = raw.crop(tmin=args.start or 0, tmax=args.end)
    cleaned = clean_raw(raw)

    timeline = sliding_window_features(cleaned)
    print(f"FAA pair used: {timeline['left_channels_used'] or 'NONE'} vs "
          f"{timeline['right_channels_used'] or 'NONE'} | theta from: {timeline['midline_channels_used']}")

    segments = ad_segments(cleaned.annotations)
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_png = os.path.join(OUT_DIR, f"{base}_engagement_timeline.png")
    plot_timeline(timeline, out_png, title=base, segments=segments)

    print(f"Timeline: {len(timeline['times_s'])} windows over {timeline['times_s'][-1]:.1f}s")
    for name, s, e in segments:
        print(f"Ad '{name}' ({s:.1f}-{e:.1f}s): {summary_features(timeline, s, e)}")
    if not segments:
        print(f"Whole recording (no AD markers): {summary_features(timeline)}")
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
