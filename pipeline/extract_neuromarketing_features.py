"""
Sliding-window Frontal Alpha Asymmetry (FAA) + frontal theta power timeline --
the system's job here is to produce the complete, accurate chart of brain
activity across an ad's duration. Matching specific chart moments against the
ad's content (logo appears, product shown, etc.) is a manual, after-the-fact
step you do by eyeballing the plot -- not something this script does.

FAA sign convention (Davidson; Allen et al.): FAA = ln(right alpha) - ln(left
alpha), e.g. ln(F4) - ln(F3), or ln(Fp2) - ln(Fp1) on this cap. Alpha power is
inversely related to cortical activation, so a POSITIVE FAA (right alpha >
left alpha) means relatively MORE LEFT activation = approach / positive
engagement; NEGATIVE means relatively more right activation = avoidance /
withdrawal. (An earlier version of this docstring and the plot label had this
backwards.)

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
# Beta for the engagement index. Capped at 25Hz rather than 30: jaw/scalp muscle
# activity leaks into the upper beta range, and dry electrodes pick it up more.
BETA_BAND = (13, 25)
ENGAGEMENT_CHANNELS = ["Cz", "Fz"]  # top of head: away from eyes and jaw
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

# Permanently dead on our SmartDRY cap (no scalp contact, confirmed in calibration
# and the pilot). Always excluded, so an automatic-detection miss can't let them
# into ICA or the features. --bad adds to this list.
KNOWN_BAD = ["F3", "F8"]

# Posterior alpha suppression: alpha over visual cortex drops when people look
# closely at something. First choice O1/O2/Pz; P3/P4 used only if none of those work.
POSTERIOR_PRIMARY = ["O1", "O2", "Pz"]
POSTERIOR_FALLBACK = ["P3", "P4"]

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


def choose_posterior_channels(ch_names, bad=()):
    chans = [c for c in (_find(ch_names, n, bad) for n in POSTERIOR_PRIMARY) if c]
    if not chans:
        chans = [c for c in (_find(ch_names, n, bad) for n in POSTERIOR_FALLBACK) if c]
    return chans


def _band_power(data_segment, sfreq, band):
    """data_segment: (n_channels, n_samples) for one window. Returns mean power
    in `band` per channel. 1s segments with 50% overlap, so the whole 3s window
    is used (an earlier 2s/no-overlap setting silently ignored the last second
    of every window, shifting the timeline ~0.5s early)."""
    n_per_seg = min(data_segment.shape[-1], int(sfreq))
    psds, freqs = psd_array_welch(data_segment, sfreq=sfreq, fmin=band[0], fmax=band[1],
                                   n_per_seg=n_per_seg, n_overlap=n_per_seg // 2, verbose=False)
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
    post_ch = choose_posterior_channels(ch_names, bad)
    if not post_ch:
        print(f"POSTERIOR ALPHA SKIPPED: none of {POSTERIOR_PRIMARY + POSTERIOR_FALLBACK} usable.")
    eng_ch = [c for c in (_find(ch_names, n, bad) for n in ENGAGEMENT_CHANNELS) if c]
    if not eng_ch:
        print(f"ENGAGEMENT INDEX SKIPPED: none of {ENGAGEMENT_CHANNELS} usable.")

    left_idx = [ch_names.index(c) for c in left_ch]
    right_idx = [ch_names.index(c) for c in right_ch]
    mid_idx = [ch_names.index(c) for c in mid_ch]
    post_idx = [ch_names.index(c) for c in post_ch]
    eng_idx = [ch_names.index(c) for c in eng_ch]

    window_samples = int(window_s * sfreq)
    step_samples = int(step_s * sfreq)

    # Powers are stored as log10(power): band power is heavily right-skewed (one
    # movement artifact makes one window huge), and summaries take the median of
    # these log values, so a single bad window can't drag an ad's number.
    times, faa_values, theta_values, post_values, eng_values = [], [], [], [], []
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        segment = data[:, start:end]

        if left_idx:
            left_alpha = _band_power(segment[left_idx], sfreq, alpha_band).mean()
            right_alpha = _band_power(segment[right_idx], sfreq, alpha_band).mean()
            faa = np.log(right_alpha) - np.log(left_alpha)
        else:
            faa = np.nan

        theta = np.log10(_band_power(segment[mid_idx], sfreq, theta_band).mean())
        post = np.log10(_band_power(segment[post_idx], sfreq, alpha_band).mean()) if post_idx else np.nan
        if eng_idx:
            eng = np.log10(_band_power(segment[eng_idx], sfreq, BETA_BAND).mean()
                           / _band_power(segment[eng_idx], sfreq, alpha_band).mean())
        else:
            eng = np.nan

        times.append((start + window_samples / 2) / sfreq)
        faa_values.append(faa)
        theta_values.append(theta)
        post_values.append(post)
        eng_values.append(eng)

    return {
        "times_s": np.array(times),
        "faa": np.array(faa_values),
        "theta": np.array(theta_values),        # log10 power
        "post_alpha": np.array(post_values),    # log10 power
        "engagement": np.array(eng_values),     # log10(beta/alpha)
        "left_channels_used": left_ch,
        "right_channels_used": right_ch,
        "midline_channels_used": mid_ch,
        "posterior_channels_used": post_ch,
        "engagement_channels_used": eng_ch,
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
    def med(key):
        v = timeline[key][m]
        return float(np.nanmedian(v)) if v.size and not np.all(np.isnan(v)) else None
    return {
        "n_windows": int(m.sum()),
        "faa": med("faa"),
        "theta_log": med("theta"),
        "post_alpha_log": med("post_alpha"),
        "engagement_log": med("engagement"),
    }


def baseline_segments(annotations):
    """{kind: (start, end)} for BASELINE_<KIND>_START / _END markers (EO, EC, NEUTRAL)."""
    starts, segs = {}, {}
    for a in annotations:
        d = a["description"]
        if d.startswith("BASELINE_") and d.endswith("_START"):
            starts[d[len("BASELINE_"):-len("_START")]] = a["onset"]
        elif d.startswith("BASELINE_") and d.endswith("_END"):
            kind = d[len("BASELINE_"):-len("_END")]
            if kind in starts:
                segs[kind] = (starts[kind], a["onset"])
    return segs


def baseline_reference(timeline, segs):
    """Baseline FAA/theta from the neutral video if recorded (closest match to
    watching an ad), else from eyes-open fixation. None if neither exists."""
    for kind in ("NEUTRAL", "EO"):
        if kind in segs:
            ref = summary_features(timeline, *segs[kind])
            if ref["n_windows"]:
                return kind, ref
    return None, None


def relative_to_baseline(ad, ref):
    """Express an ad's summary relative to the person's own baseline -- the
    numbers to compare across people.
      faa_change            difference (FAA is already a log ratio); + = more approach
      theta_change_db       + = more processing than baseline (+3 dB = power doubled)
      alpha_suppression_pct + = posterior alpha DROPPED vs baseline = more visual attention
      engagement_change_pct + = beta/alpha ratio ROSE vs baseline = more alert/engaged"""
    out = {}
    if ad["faa"] is not None and ref["faa"] is not None:
        out["faa_change"] = ad["faa"] - ref["faa"]
    if ad["theta_log"] is not None and ref["theta_log"] is not None:
        out["theta_change_db"] = 10 * (ad["theta_log"] - ref["theta_log"])
    if ad["post_alpha_log"] is not None and ref["post_alpha_log"] is not None:
        out["alpha_suppression_pct"] = 100 * (1 - 10 ** (ad["post_alpha_log"] - ref["post_alpha_log"]))
    if ad["engagement_log"] is not None and ref["engagement_log"] is not None:
        out["engagement_change_pct"] = 100 * (10 ** (ad["engagement_log"] - ref["engagement_log"]) - 1)
    return out


def eyes_closed_check(raw, segs, bad=()):
    """Cap sanity check from the baseline block: posterior alpha should rise
    clearly with eyes closed vs open (Berger effect). A ratio near 1 suggests
    poor posterior contact."""
    if "EO" not in segs or "EC" not in segs:
        return None
    chans = [c for c in ("O1", "O2", "Pz", "P3", "P4") if c in raw.ch_names and c not in bad]
    if not chans:
        return None
    sf = raw.info["sfreq"]
    def alpha(seg):
        s, e = int(seg[0] * sf), int(seg[1] * sf)
        return _band_power(raw.get_data(picks=chans)[:, s:e], sf, ALPHA_BAND).mean()
    return chans, alpha(segs["EC"]) / alpha(segs["EO"])


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


def post_ad_ends(annotations):
    """{ad name: end of its post-ad gray screen}, from POST_AD_END:<name> markers.
    Older recordings (before the post-ad buffer existed) have none."""
    return {a["description"].split(":", 1)[1]: a["onset"] for a in annotations
            if a["description"].startswith("POST_AD_END:")}


def plot_timeline(timeline, out_path, title="Engagement timeline", segments=(), baseline_segs=None, ref=None,
                  post_ends=None):
    import matplotlib.pyplot as plt
    fig, (ax1, ax4, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    for ax in (ax1, ax4, ax2, ax3):
        for name, s, e in segments:
            ax.axvspan(s, e, color="tab:green", alpha=0.12)
            pe = (post_ends or {}).get(name)
            if pe:
                ax.axvspan(e, pe, color="tab:green", alpha=0.05, hatch="//", edgecolor="tab:green", linewidth=0)
        for kind, (s, e) in (baseline_segs or {}).items():
            ax.axvspan(s, e, color="gray", alpha=0.12)
            ax.text((s + e) / 2, 0.02, f"baseline {kind}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color="dimgray")
    if ref is not None:
        for ax, key in ((ax1, "faa"), (ax4, "engagement_log"), (ax2, "theta_log"), (ax3, "post_alpha_log")):
            if ref[key] is not None:
                ax.axhline(ref[key], color="gray", linestyle="--", linewidth=1)
    if segments:
        for name, s, e in segments:
            ax2.text((s + e) / 2, 0.97, name, transform=ax2.get_xaxis_transform(),
                     ha="center", va="top", fontsize=8, color="darkgreen")

    if np.all(np.isnan(timeline["faa"])):
        ax1.text(0.5, 0.5, "FAA unavailable -- no working left/right frontal electrode pair",
                 transform=ax1.transAxes, ha="center", va="center", color="gray")
        ax1.set_title(f"{title}  (green = ad playing, hatched = gray screen after the ad)")
    else:
        ax1.plot(timeline["times_s"], timeline["faa"], color="tab:blue")
        ax1.axhline(0, color="gray", linewidth=0.5)
        ax1.set_title(f"{title}  (green = ad playing, hatched = gray screen after the ad)\nleft={timeline['left_channels_used']} "
                      f"right={timeline['right_channels_used']} "
                      f"(positive = approach/positive engagement, negative = avoidance)")
    ax1.set_ylabel("FAA\n(ln(R alpha) - ln(L alpha))")

    if np.all(np.isnan(timeline["engagement"])):
        ax4.text(0.5, 0.5, "Engagement index unavailable -- Cz/Fz not usable",
                 transform=ax4.transAxes, ha="center", va="center", color="gray")
    else:
        ax4.plot(timeline["times_s"], timeline["engagement"], color="tab:green")
    ax4.set_ylabel(f"Engagement index\nlog10(beta/alpha) ({timeline['engagement_channels_used']})")
    ax4.set_title("up = more alert/engaged", fontsize=8, loc="right")

    ax2.plot(timeline["times_s"], timeline["theta"], color="tab:orange")
    ax2.set_ylabel(f"Frontal theta\nlog10 power ({timeline['midline_channels_used']})")
    ax2.set_title("up = more processing", fontsize=8, loc="right")

    if np.all(np.isnan(timeline["post_alpha"])):
        ax3.text(0.5, 0.5, "Posterior alpha unavailable -- no working O1/O2/Pz/P3/P4",
                 transform=ax3.transAxes, ha="center", va="center", color="gray")
    else:
        ax3.plot(timeline["times_s"], timeline["post_alpha"], color="tab:purple")
        ax3.invert_yaxis()  # so "up" = more suppression = more attention, like the other panels
    ax3.set_ylabel(f"Posterior alpha\nlog10 power ({timeline['posterior_channels_used']})")
    ax3.set_title("up = alpha suppressed = more visual attention (axis inverted)", fontsize=8, loc="right")
    ax3.set_xlabel("Time (s)")

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
    known = [b for b in KNOWN_BAD if b in raw.ch_names]
    bad = sorted(set(bad) | set(auto_bad) | set(known))
    raw.info["bads"] = bad
    if bad:
        print(f"Marked bad (excluded from ICA and features): {bad}")
    if args.start is not None or args.end is not None:
        raw = raw.crop(tmin=args.start or 0, tmax=args.end)
    cleaned = clean_raw(raw)

    timeline = sliding_window_features(cleaned)
    print(f"FAA pair used: {timeline['left_channels_used'] or 'NONE'} vs "
          f"{timeline['right_channels_used'] or 'NONE'} | theta from: {timeline['midline_channels_used']} | "
          f"posterior alpha from: {timeline['posterior_channels_used'] or 'NONE'} | "
          f"engagement from: {timeline['engagement_channels_used'] or 'NONE'}")

    segments = ad_segments(cleaned.annotations)
    bsegs = baseline_segments(cleaned.annotations)
    ref_kind, ref = baseline_reference(timeline, bsegs)

    ec = eyes_closed_check(cleaned, bsegs, bad)
    if ec:
        chans, ratio = ec
        print(f"Cap check: eyes-closed/eyes-open posterior alpha = {ratio:.2f}x on {chans} "
              f"({'OK' if ratio >= 1.3 else 'WEAK -- check posterior contact'})")
    if ref is None:
        print("No baseline markers in this recording -- reporting raw values only "
              "(not comparable across participants).")
    else:
        print(f"Baseline reference: {ref_kind} ({ref['n_windows']} windows)")

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_png = os.path.join(OUT_DIR, f"{base}_engagement_timeline.png")
    plot_timeline(timeline, out_png, title=base, segments=segments, baseline_segs=bsegs, ref=ref,
                  post_ends=post_ad_ends(cleaned.annotations))

    print(f"Timeline: {len(timeline['times_s'])} windows over {timeline['times_s'][-1]:.1f}s")
    for name, s, e in segments:
        ad = summary_features(timeline, s, e)
        rel = relative_to_baseline(ad, ref) if ref else {}
        print(f"Ad '{name}' ({s:.1f}-{e:.1f}s): {ad}" + (f"  | vs baseline: {rel}" if rel else ""))
    if not segments:
        print(f"Whole recording (no AD markers): {summary_features(timeline)}")
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
