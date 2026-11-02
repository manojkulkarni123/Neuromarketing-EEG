"""
Builds an offline replay page for one session: the ad video plays, with the
four brain-measure timelines underneath and a moving line synced to the video.
Click (or drag) on any graph to jump the video to that moment.

    .venv\\Scripts\\python.exe pipeline\\build_replay.py data\\lab_session\\test1_raw.fif

Writes data/lab_session/<recording>_replay.html next to the ads/ folder (the
page plays the videos straight from data/lab_session/ads/, so it must stay in
that folder). Open it in Chrome/Edge. Same analysis as
extract_neuromarketing_features.py (quality check, ICA, sliding windows),
shown relative to the participant's baseline where one was recorded.
"""
import os
import sys
import csv
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_mne import load_recording, clean_raw, quality_check
import extract_neuromarketing_features as ef

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session")


def _clean(arr):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), 4) for v in arr]


def _ratings(path):
    """{ad name: {question: score}} from the session's ratings CSV, if it exists.
    Older sessions have a single 'rating' column instead of liking/purchase_intent."""
    base = os.path.splitext(path)[0]
    if base.endswith("_raw"):
        base = base[:-4]
    csv_path = base + "_ratings.csv"
    if not os.path.exists(csv_path):
        return {}
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["ad"]] = {k: int(row[k]) for k in ("liking", "purchase_intent", "rating")
                              if row.get(k) not in (None, "")}
    return out


def _session_map(annotations, n_seconds, segs, post_ends):
    """Blocks for the session strip at the top of the page."""
    blocks = []
    for kind, (s, e) in ef.baseline_segments(annotations).items():
        label = {"EO": "Baseline: eyes open", "EC": "Eyes closed (cap check)",
                 "NEUTRAL": "Baseline: neutral video"}.get(kind, kind)
        blocks.append({"kind": "baseline_" + kind.lower(), "label": label, "start": round(s, 2), "end": round(e, 2)})
    for name, s, e in segs:
        blocks.append({"kind": "ad", "label": name, "start": round(s, 2), "end": round(e, 2)})
        if name in post_ends:
            blocks.append({"kind": "post", "label": "gray", "start": round(e, 2), "end": round(post_ends[name], 2)})
    return {"length": round(float(n_seconds), 2), "blocks": sorted(blocks, key=lambda b: b["start"])}


def build(path, extra_bad=()):
    raw = load_recording(path)
    raw.info["bads"] = sorted(set(quality_check(raw)) | set(ef.KNOWN_BAD) | set(extra_bad))
    cleaned = clean_raw(raw)
    tl = ef.sliding_window_features(cleaned)
    segs = ef.ad_segments(cleaned.annotations)
    if not segs:
        raise SystemExit("No AD_START/AD_END markers in this recording -- nothing to replay.")
    bsegs = ef.baseline_segments(cleaned.annotations)
    ref_kind, ref = ef.baseline_reference(tl, bsegs)
    ratings = _ratings(path)
    t = tl["times_s"]
    half = ef.WINDOW_S / 2

    def series(key, ref_key, transform, fallback_center):
        vals = tl[key]
        if ref is not None and ref[ref_key] is not None:
            base = ref[ref_key]
        else:
            base = fallback_center(vals)
        return transform(vals, base)

    post_ends = ef.post_ad_ends(cleaned.annotations)
    center = lambda v: np.nanmedian(v) if np.any(np.isfinite(v)) else 0.0

    def entry(name, label, kind, s, e, end, video=None):
        m = (t - half >= s) & (t + half <= end)
        return {
            "name": name,
            "label": label,
            "kind": kind,                                 # "ad" or "baseline_eo" / "baseline_ec" / ...
            "video": video,
            "duration": round(float(e - s), 2),          # video length on the recording clock
            "total": round(float(end - s), 2),           # video + post-ad gray screen
            "t": _clean(t[m] - s),
            "faa": _clean(series("faa", "faa", lambda v, b: v[m] - b, center)),
            "engagement": _clean(series("engagement", "engagement_log",
                                        lambda v, b: 100 * (10 ** (v[m] - b) - 1), center)),
            "theta": _clean(series("theta", "theta_log", lambda v, b: 10 * (v[m] - b), center)),
            # % drop in alpha vs baseline (same formula as alpha_suppression_pct): UP = more visual attention
            "alpha": _clean(series("post_alpha", "post_alpha_log", lambda v, b: 100 * (1 - 10 ** (v[m] - b)), center)),
            # whole-segment numbers: exactly what extract_neuromarketing_features.py reports for ads
            "summary": ({k: round(float(v), 3) for k, v in
                         ef.relative_to_baseline(ef.summary_features(tl, s, e), ref).items()} if ref else {}),
            "ratings": ratings.get(name, {}),
        }

    # Baseline blocks first, so the page can show what "rest" itself looks like.
    ads = []
    for kind, label in (("EO", "Baseline: eyes open"), ("EC", "Eyes closed (cap check)"), ("NEUTRAL", "Baseline: neutral video")):
        if kind in bsegs:
            s, e = bsegs[kind]
            ads.append(entry(label, label, "baseline_" + kind.lower(), s, e, e))
    for name, s, e in segs:
        # Include the post-ad gray screen (if recorded): the brain reacts to the
        # ending/logo a little late, so that reaction lands just after AD_END.
        ads.append(entry(name, name, "ad", s, e, post_ends.get(name, e), video=f"ads/{name}"))

    meta = {
        "recording": os.path.basename(path),
        "baseline": ref_kind or "none (values relative to the session median instead)",
        "baseline_span": [round(v, 1) for v in bsegs[ref_kind]] if ref_kind else None,
        "baseline_windows": ref["n_windows"] if ref else 0,
        # official cap check (same as extract_neuromarketing_features.py): whole-block alpha, eyes closed / open
        "cap_check": (lambda ec: {"ratio": round(float(ec[1]), 2), "channels": ec[0]} if ec else None)(
            ef.eyes_closed_check(cleaned, bsegs, raw.info["bads"])),
        "faa_noise": 0.18,   # SD of FAA change from noise alone on ~30 s (tested on synthetic data)
        "session": _session_map(cleaned.annotations, cleaned.n_times / cleaned.info["sfreq"], segs, post_ends),
        "window_s": ef.WINDOW_S,
        "bad": raw.info["bads"],
        "channels": {
            "faa": f"{tl['left_channels_used']} vs {tl['right_channels_used']}" if tl["left_channels_used"] else "unavailable",
            "engagement": str(tl["engagement_channels_used"] or "unavailable"),
            "theta": str(tl["midline_channels_used"]),
            "alpha": str(tl["posterior_channels_used"] or "unavailable"),
        },
    }
    base = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(OUT_DIR, f"{base}_replay.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATA__", json.dumps({"meta": meta, "ads": ads})))
    print(f"Saved replay page: {out}")
    print("Open it in Chrome or Edge (double-click). Keep it in data/lab_session/ so it can find the ads.")
    return out


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>EEG Ad Replay</title>
<style>
  :root { --ink:#1c1c1c; --muted:#5f6368; --line:#dcdcdc; --bg:#f5f5f4; --card:#fff;
          --attn:#7c3aed; --alert:#16a34a; --effort:#ea580c; --want:#2563eb; --base:#6b7280; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, "Segoe UI", sans-serif; margin: 0; background: var(--bg); color: var(--ink); }
  header { padding: 10px 18px; background: var(--card); border-bottom: 1px solid var(--line);
           display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 17px; margin: 0; }
  header .meta { font-size: 12px; color: var(--muted); }
  select, button { font-size: 13px; padding: 5px 10px; }
  .wrap { padding: 12px 18px 24px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
  h2 { font-size: 14px; margin: 0 0 8px; }
  .sub { font-size: 12px; color: var(--muted); line-height: 1.45; }
  /* session map */
  .map { position: relative; height: 34px; background: #eceff1; border-radius: 6px; margin: 8px 0 4px; }
  .blk { position: absolute; top: 0; bottom: 0; border-radius: 5px; font-size: 11px; color: #fff;
         display: flex; align-items: center; justify-content: center; overflow: hidden; white-space: nowrap; }
  .blk.baseline_eo { background: var(--base); box-shadow: 0 0 0 2px #111 inset; font-weight: 600; }
  .blk.baseline_ec { background: #9ca3af; }
  .blk.baseline_neutral { background: var(--base); }
  .blk.ad { background: #15803d; cursor: pointer; }
  .blk.sel { box-shadow: 0 0 0 3px #facc15 inset !important; }
  .blk.click { cursor: pointer; }
  .cross { display: none; aspect-ratio: 16 / 9; background: #111; border-radius: 8px; color: #fff; flex-direction: column;
           align-items: center; justify-content: center; text-align: center; padding: 20px; }
  .cross .plus { font-size: 72px; line-height: 1; font-weight: 300; }
  .cross p { font-size: 13px; color: #d4d4d4; max-width: 460px; line-height: 1.5; }
  .blk.post { background: #a3a3a3; }
  .ticks { position: relative; height: 14px; font-size: 10px; color: var(--muted); }
  .ticks span { position: absolute; transform: translateX(-50%); }
  main { display: grid; grid-template-columns: minmax(340px, 44%) 1fr; gap: 14px; margin-top: 12px; }
  video { width: 100%; background: #000; border-radius: 8px; display: block; }
  .now { margin-top: 10px; }
  .now .t { font-size: 18px; font-weight: 600; }
  .row { display: grid; grid-template-columns: 14px 1fr auto; gap: 8px; align-items: baseline;
         padding: 5px 0; border-top: 1px solid #f0f0f0; font-size: 13px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .row .v { font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }
  .row .w { grid-column: 2 / 4; font-size: 12px; color: var(--muted); margin-top: -2px; }
  .panel { background: var(--card); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 10px; padding: 8px 10px 4px; }
  .panel .title { font-size: 13px; font-weight: 600; display: flex; justify-content: space-between; gap: 8px; }
  .panel .title span { font-weight: 400; color: var(--muted); font-size: 11.5px; }
  canvas { width: 100%; height: 132px; display: block; cursor: crosshair; }
  /* conclusion */
  .concl { margin-top: 12px; }
  .meas { display: grid; grid-template-columns: 240px 1fr 80px; gap: 10px; align-items: center; padding: 7px 0; border-top: 1px solid #f0f0f0; font-size: 13px; }
  .meas .name b { display: block; }
  .meas .name small { color: var(--muted); }
  .gauge { position: relative; height: 12px; background: #f1f1f1; border-radius: 6px; }
  .gauge .mid { position: absolute; left: 50%; top: -3px; bottom: -3px; border-left: 2px dashed var(--base); }
  .gauge .bar { position: absolute; top: 0; bottom: 0; border-radius: 6px; }
  .gauge .noise { position: absolute; top: 0; bottom: 0; background: repeating-linear-gradient(45deg, #d4d4d4 0 3px, transparent 3px 6px); }
  .meas .num { font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }
  .verdict { grid-column: 1 / 4; font-size: 12px; color: var(--muted); margin-top: -4px; }
  .ratings { display: flex; gap: 10px; margin-top: 10px; }
  .rate { flex: 1; background: #f8fafc; border: 1px solid var(--line); border-radius: 6px; padding: 8px; font-size: 12px; }
  .rate b { font-size: 20px; display: block; }
  .story { margin-top: 10px; padding: 10px 12px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; font-size: 13px; line-height: 1.5; }
  .warn { margin-top: 8px; font-size: 12px; color: #92400e; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 6px 10px; }
  details { margin-top: 12px; }
  summary { cursor: pointer; font-weight: 600; font-size: 13px; }
  ol { font-size: 13px; line-height: 1.55; padding-left: 20px; }
  .legend { display: flex; gap: 14px; font-size: 11.5px; color: var(--muted); flex-wrap: wrap; margin-top: 2px; }
  .legend i { display: inline-block; width: 18px; height: 0; border-top: 2px dashed var(--base); vertical-align: middle; margin-right: 4px; }
  .legend u { display: inline-block; width: 14px; height: 10px; vertical-align: middle; margin-right: 4px; text-decoration: none; }
</style></head>
<body>
<header>
  <h1>EEG Ad Replay</h1>
  <label>Ad: <select id="adSel"></select></label>
  <button id="playBtn">Play / Pause</button>
  <span class="meta" id="metaLine"></span>
</header>
<div class="wrap">
  <div class="card">
    <h2>1 · Where the baseline comes from</h2>
    <div class="sub" id="mapText"></div>
    <div class="map" id="map"></div>
    <div class="ticks" id="ticks"></div>
  </div>
  <main>
    <div>
      <video id="vid" controls preload="auto"></video>
      <div class="cross" id="cross"><div class="plus" id="crossPlus">+</div><p id="crossText"></p></div>
      <div class="card now">
        <div class="t" id="tNow">0.0 s</div>
        <div class="sub" id="winLine"></div>
        <div id="nowRows"></div>
      </div>
    </div>
    <div>
      <div class="legend">
        <span><i></i>dashed line "rest" = this person's level during the baseline</span>
        <span><u style="background:rgba(22,163,74,.25)"></u>above rest</span>
        <span><u style="background:rgba(107,114,128,.22)"></u>below rest</span>
        <span>click or drag a graph to jump the video</span>
      </div>
      <div id="panels"></div>
    </div>
  </main>
  <div class="card concl">
    <h2 id="conclTitle">2 · Conclusion for this ad</h2>
    <div class="sub" id="conclSub"></div>
    <div id="meas"></div>
    <div class="ratings" id="ratings"></div>
    <div class="story" id="story"></div>
    <div class="warn">One person, one viewing: this shows how the system works, not how people in general react. Findings need the same pattern across many participants.</div>
    <details>
      <summary>3 · How these numbers are calculated</summary>
      <ol>
        <li><b>Record.</b> The cap measures tiny voltages on the scalp, about 250 times a second, while the participant rests and then watches the ad.</li>
        <li><b>Clean.</b> Electrodes without real contact are dropped, the signal is filtered to 1–40 Hz (removing electrical hum and slow drift) and eye blinks are removed with ICA.</li>
        <li><b>Measure rhythms.</b> Every second, the last 3 seconds are split into brain rhythms: theta (4–8 Hz), alpha (8–13 Hz) and beta (13–25 Hz). That gives one point on each graph.</li>
        <li><b>Compare with rest.</b> Each point is shown against this person's level during the 60-second eyes-open baseline, so 0 means "same as resting" for everyone, whatever their skull or electrode contact.</li>
        <li><b>Summarise.</b> For the conclusion, the middle value over the whole ad is compared with the middle value of the baseline.</li>
      </ol>
      <div class="sub">
        <b>Visual attention</b> = drop in back-of-head alpha (alpha is the brain's idling rhythm; it falls when the eyes take in something interesting).
        <b>Alertness</b> = beta ÷ alpha on top of the head.
        <b>Mental effort</b> = frontal theta, in decibels (+3 dB ≈ double, −3 dB ≈ half).
        <b>Wanting</b> = frontal alpha asymmetry, right minus left (positive = leaning toward "I want this").
      </div>
    </details>
  </div>
</div>
<script>
const D = __DATA__;
const NOISE = D.meta.faa_noise;
// Ordered from most to least reliable for a single person.
const PANELS = [
  {key: "alpha", sum: "alpha_suppression_pct", name: "Visual attention", tech: "Posterior alpha suppression (alpha ERD)", band: "alpha 8–13 Hz, parieto-occipital",
   color: "#7c3aed", fmt: v => (v > 0 ? "+" : "") + v.toFixed(0) + "%",
   words: v => v >= 20 ? `alpha ${v.toFixed(0)}% below rest: strong visual attention`
             : v >= 5 ? `alpha ${v.toFixed(0)}% below rest: some extra visual attention`
             : v > -5 ? "about the same as rest" : `alpha ${(-v).toFixed(0)}% above rest: less visual attention`,
   gaugeMax: 60},
  {key: "engagement", sum: "engagement_change_pct", name: "Alertness", tech: "Beta/alpha engagement index", band: "beta 13–25 Hz ÷ alpha 8–13 Hz, central",
   color: "#16a34a", fmt: v => (v > 0 ? "+" : "") + v.toFixed(0) + "%",
   words: v => v >= 10 ? `${v.toFixed(0)}% more alert than at rest` : v <= -10 ? `${(-v).toFixed(0)}% less alert than at rest` : "about as alert as at rest",
   gaugeMax: 60},
  {key: "theta", sum: "theta_change_db", name: "Mental effort", tech: "Frontal midline theta (FMθ)", band: "theta 4–8 Hz, frontal midline",
   color: "#ea580c", fmt: v => (v > 0 ? "+" : "") + v.toFixed(1) + " dB",
   words: v => v >= 3 ? "much more mental effort than at rest (about double or more)" : v >= 1 ? "a little more mental effort than at rest"
             : v <= -3 ? "much less mental effort than at rest" : v <= -1 ? "a little less mental effort than at rest" : "about the same as rest",
   gaugeMax: 6},
  {key: "faa", sum: "faa_change", name: "Wanting", tech: "Frontal alpha asymmetry (FAA)", band: "alpha 8–13 Hz, right − left frontal",
   color: "#2563eb", fmt: v => (v > 0 ? "+" : "") + v.toFixed(2),
   words: v => Math.abs(v) < NOISE ? `inside the noise range (±${NOISE}): can't be judged from one person`
             : v > 0 ? "leans toward wanting it (approach)" : "leans away from it (avoidance)",
   gaugeMax: 0.5, noise: true},
];
const vid = document.getElementById("vid"), sel = document.getElementById("adSel");
const half = D.meta.window_s / 2;
let ad = null, adIdx = 0, canvases = {};
let tNow = 0, postStart = null, vStart = null;   // vStart: clock for blocks without a video (baselines)
function curTime() {
  if (!ad.video) return vStart === null ? tNow : Math.min(ad.total, (performance.now() - vStart) / 1000);
  if (postStart !== null) return Math.min(ad.total, ad.duration + (performance.now() - postStart) / 1000);
  return vid.currentTime;
}
function setTime(t) {
  tNow = Math.max(0, Math.min(ad.total, t));
  if (!ad.video) { vStart = null; return; }
  postStart = null; vid.pause();
  vid.currentTime = Math.min(tNow, Math.max(0, vid.duration ? vid.duration - 0.05 : ad.duration));
}

document.getElementById("metaLine").textContent =
  `${D.meta.recording} · ${D.meta.window_s}s windows · electrodes excluded (no contact): ${D.meta.bad.join(", ") || "none"}`;

// ---- session map ----
const S = D.meta.session, mapEl = document.getElementById("map"), ticksEl = document.getElementById("ticks");
const pct = s => (100 * s / S.length) + "%";
S.blocks.forEach(b => {
  const d = document.createElement("div");
  d.className = "blk " + b.kind;
  d.style.left = pct(b.start); d.style.width = pct(b.end - b.start);
  d.textContent = b.kind === "post" ? "" : b.label;
  d.title = `${b.label}: ${b.start.toFixed(0)}–${b.end.toFixed(0)} s`;
  const i = D.ads.findIndex(a => a.name === b.label);
  if (i >= 0) { d.dataset.ad = b.label; d.classList.add("click"); d.addEventListener("click", () => { sel.value = i; loadAd(i); }); }
  mapEl.appendChild(d);
});
for (let s = 0; s <= S.length; s += 30) { const sp = document.createElement("span"); sp.style.left = pct(s); sp.textContent = s + " s"; ticksEl.appendChild(sp); }
const bs = D.meta.baseline_span;
document.getElementById("mapText").innerHTML = bs
  ? `The whole session. The <b>dark block (${bs[0].toFixed(0)}–${bs[1].toFixed(0)} s)</b> is the baseline: the participant sat still with eyes open, looking at a cross, for 60 s. ` +
    `That is their personal "resting level". Every graph and number below is measured <b>against it</b>: 0 = same as rest. ` +
    `Green blocks are ads. Click any block, including the baseline, to see its readings; the eyes-closed block only checks that the cap is touching the scalp.`
  : `No baseline was recorded in this session, so values are shown relative to the session's middle value instead.`;

D.ads.forEach((a, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `${a.label} (${a.duration.toFixed(0)} s)`; sel.appendChild(o); });

// ---- graphs ----
const panelsEl = document.getElementById("panels");
PANELS.forEach(p => {
  const div = document.createElement("div"); div.className = "panel";
  div.innerHTML = `<div class="title"><span><span style="color:${p.color};font-weight:600;font-size:13px">${p.name}</span> <span style="color:#1c1c1c;font-weight:600;font-size:12px">· ${p.tech}</span></span><span>${p.band} · ${D.meta.channels[p.key]}</span></div>`;
  const c = document.createElement("canvas"); div.appendChild(c); panelsEl.appendChild(div);
  canvases[p.key] = c;
  const seek = ev => { if (!ad) return; const r = c.getBoundingClientRect();
    setTime((ev.clientX - r.left - 54) / (r.width - 62) * ad.total); draw(); };
  let dragging = false;
  c.addEventListener("mousedown", ev => { dragging = true; vid.pause(); seek(ev); });
  window.addEventListener("mousemove", ev => { if (dragging) seek(ev); });
  window.addEventListener("mouseup", () => dragging = false);
});

const nowRows = document.getElementById("nowRows");
PANELS.forEach(p => {
  const r = document.createElement("div"); r.className = "row";
  r.innerHTML = `<span class="dot" style="background:${p.color}"></span><span>${p.name} <small style="color:#5f6368">· ${p.tech}</small></span><span class="v" id="v_${p.key}">–</span><span class="w" id="w_${p.key}"></span>`;
  nowRows.appendChild(r);
});

function valueAt(arr, t) {
  let best = -1, bd = 1e9;
  ad.t.forEach((tt, i) => { const d = Math.abs(tt - t); if (d < bd) { bd = d; best = i; } });
  return (best >= 0 && bd <= half) ? arr[best] : null;
}

function hexA(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }

function drawPanel(p) {
  const c = canvases[p.key], dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight; c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const L = 54, R = 8, T = 8, B = 18, pw = W - L - R, ph = H - T - B;
  const ys = ad[p.key].filter(v => v !== null);
  const tx = t => L + (t / ad.total) * pw;
  g.font = "11px system-ui";
  if (!ys.length) { g.fillStyle = "#666"; g.fillText("Unavailable for this session (no working electrodes)", L + 8, T + ph / 2); return; }
  // Scale so one extreme movement spike can't flatten the whole curve: the typical
  // range (3rd-97th percentile) widened by half its span on each side. Only points
  // beyond that are pinned to the edge and flagged; ordinary peaks stay on scale.
  const srt = ys.slice().sort((a, b) => a - b), q = f => srt[Math.round(f * (srt.length - 1))];
  const span = q(0.97) - q(0.03);
  const vLo = Math.max(srt[0], q(0.03) - 0.5 * span), vHi = Math.min(srt[srt.length - 1], q(0.97) + 0.5 * span);
  let lo = Math.min(0, vLo), hi = Math.max(0, vHi);
  if (p.noise) { lo = Math.min(lo, -NOISE); hi = Math.max(hi, NOISE); }
  const pad = (hi - lo) * 0.12 || 1; lo -= pad; hi += pad;
  const ty = v => T + (1 - (Math.max(vLo, Math.min(vHi, v)) - lo) / (hi - lo)) * ph;
  const y0 = ty(0);
  if (ad.total > ad.duration) {
    g.fillStyle = "rgba(120,120,120,0.10)"; g.fillRect(tx(ad.duration), T, tx(ad.total) - tx(ad.duration), ph);
    g.fillStyle = "#777"; g.textAlign = "center"; g.fillText("gray screen after ad", (tx(ad.duration) + tx(ad.total)) / 2, T + 10);
  }
  if (p.noise) {
    g.fillStyle = "rgba(0,0,0,0.06)"; g.fillRect(L, ty(NOISE), pw, ty(-NOISE) - ty(NOISE));
    g.fillStyle = "#888"; g.textAlign = "left"; g.fillText("noise range for one person", L + 4, ty(-NOISE) - 3);
  }
  // fill between the curve and the resting level
  const runs = []; let cur = [];
  ad.t.forEach((t, i) => { const v = ad[p.key][i]; if (v === null) { if (cur.length) runs.push(cur); cur = []; } else cur.push([tx(t), ty(v)]); });
  if (cur.length) runs.push(cur);
  [[T, y0, hexA(p.color, 0.22)], [y0, T + ph, "rgba(107,114,128,0.20)"]].forEach(([top, bot, col]) => {
    g.save(); g.beginPath(); g.rect(L, top, pw, bot - top); g.clip();
    runs.forEach(r => { g.beginPath(); g.moveTo(r[0][0], y0); r.forEach(([x, y]) => g.lineTo(x, y)); g.lineTo(r[r.length - 1][0], y0); g.closePath(); g.fillStyle = col; g.fill(); });
    g.restore();
  });
  // current 3 s window
  g.fillStyle = "rgba(0,0,0,0.06)"; g.fillRect(tx(Math.max(0, tNow - half)), T, tx(Math.min(ad.total, tNow + half)) - tx(Math.max(0, tNow - half)), ph);
  // baseline
  g.strokeStyle = "#4b5563"; g.lineWidth = 1.5; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(L, y0); g.lineTo(L + pw, y0); g.stroke(); g.setLineDash([]);
  g.font = "bold 11px system-ui"; g.textAlign = "right";
  g.fillStyle = "#374151"; g.fillText("rest", L - 6, y0 + 4);
  g.font = "11px system-ui"; g.fillStyle = "#666";
  if (ty(vHi) < y0 - 14) g.fillText(p.fmt(vHi), L - 6, ty(vHi) + 4);
  if (ty(vLo) > y0 + 14) g.fillText(p.fmt(vLo), L - 6, ty(vLo) + 4);
  // flag points beyond the typical range
  g.fillStyle = "#b91c1c"; g.textAlign = "left";
  let flagged = false;
  ad.t.forEach((t, i) => { const v = ad[p.key][i];
    if (v === null || (v >= vLo && v <= vHi)) return;
    const x = tx(t), y = ty(v), up = v > vHi;
    g.beginPath(); g.moveTo(x - 4, y + (up ? 6 : -6)); g.lineTo(x + 4, y + (up ? 6 : -6)); g.lineTo(x, y + (up ? -1 : 1)); g.fill();
    if (!flagged) { g.fillText(`off scale (${p.fmt(v)}, likely movement)`, x + 7, y + (up ? 10 : -4)); flagged = true; } });
  g.fillStyle = "#666"; g.textAlign = "center";
  for (let s = 0; s <= ad.total; s += (ad.total > 60 ? 10 : 5)) g.fillText(s + "s", tx(s), H - 4);
  g.strokeStyle = p.color; g.lineWidth = 2;
  runs.forEach(r => { g.beginPath(); r.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke(); });
  g.strokeStyle = "#dc2626"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(tx(tNow), T); g.lineTo(tx(tNow), T + ph); g.stroke();
}

function draw() {
  if (!ad) return;
  if (!vid.paused || postStart !== null || vStart !== null) tNow = curTime();
  if (vStart !== null && tNow >= ad.total) vStart = null;
  PANELS.forEach(drawPanel);
  const t = tNow;
  document.getElementById("tNow").textContent = "Right now: " + t.toFixed(1) + " s" + (t > ad.duration ? " (gray screen after the ad)" : "");
  document.getElementById("winLine").textContent = `Brain signal from ${Math.max(0, t - half).toFixed(1)}–${Math.min(ad.total, t + half).toFixed(1)} s, compared with rest:`;
  PANELS.forEach(p => {
    const v = valueAt(ad[p.key], t);
    document.getElementById("v_" + p.key).textContent = v === null ? "–" : p.fmt(v);
    document.getElementById("w_" + p.key).textContent = v === null ? "" : p.words(v);
  });
}

// ---- conclusion ----
function conclusion() {
  const isBase = ad.kind !== "ad";
  document.getElementById("conclTitle").textContent = isBase ? `2 · Summary for ${ad.label}` : `2 · Conclusion for ${ad.name}`;
  document.getElementById("conclSub").textContent = `Whole ${isBase ? "block" : "ad"} compared with the baseline: the middle value (median) of all 3-second windows inside the ${isBase ? "block" : "ad"}, against the middle value during the eyes-open baseline.`;
  const box = document.getElementById("meas"); box.innerHTML = "";
  const S = ad.summary || {};
  PANELS.forEach(p => {
    const v = S[p.sum];
    const row = document.createElement("div"); row.className = "meas";
    if (v === undefined) { row.innerHTML = `<div class="name"><b>${p.name}</b><small>${p.tech}<br>${p.band}</small></div><div class="sub">not available</div><div></div>`; box.appendChild(row); return; }
    const frac = Math.max(-1, Math.min(1, v / p.gaugeMax));
    const left = frac >= 0 ? 50 : 50 + frac * 50, width = Math.abs(frac) * 50;
    const noise = p.noise ? `<div class="noise" style="left:${50 - 50 * NOISE / p.gaugeMax}%;width:${100 * NOISE / p.gaugeMax}%"></div>` : "";
    row.innerHTML = `<div class="name"><b style="color:${p.color}">${p.name}</b><small>${p.tech}<br>${p.band}</small></div>
      <div class="gauge">${noise}<div class="bar" style="left:${left}%;width:${width}%;background:${p.color}"></div><div class="mid"></div></div>
      <div class="num">${p.fmt(v)}</div><div class="verdict">${p.words(v)}</div>`;
    box.appendChild(row);
  });
  const R = ad.ratings || {}, rb = document.getElementById("ratings"); rb.innerHTML = "";
  [["liking", "Liking", "How much did you like this ad?"], ["purchase_intent", "Purchase intent", "How likely would you be to buy this product?"], ["rating", "Rating", "(older session: single 1–9 rating)"]]
    .forEach(([k, label, q]) => { if (R[k] !== undefined) rb.insertAdjacentHTML("beforeend", `<div class="rate"><b>${R[k]} / 9</b>${label}<div class="sub">${q}</div></div>`); });
  if (!rb.innerHTML && !isBase) rb.innerHTML = `<div class="sub">No ratings file found for this session.</div>`;
  if (isBase) { document.getElementById("story").innerHTML = baselineStory(S);
    document.querySelectorAll(".blk").forEach(b => b.classList.toggle("sel", b.dataset.ad === ad.name)); return; }
  // plain-language story
  const parts = [];
  const a = S.alpha_suppression_pct, e = S.engagement_change_pct, th = S.theta_change_db, f = S.faa_change;
  if (a !== undefined) parts.push(a >= 5 ? `the viewer's eyes were clearly engaged with the screen (back-of-head alpha ${a.toFixed(0)}% below rest)` : `visual attention was close to resting level (${a.toFixed(0)}%)`);
  if (e !== undefined) parts.push(Math.abs(e) >= 10 ? `they were ${Math.abs(e).toFixed(0)}% ${e > 0 ? "more" : "less"} alert than at rest` : `alertness stayed close to resting level (${e > 0 ? "+" : ""}${e.toFixed(0)}%)`);
  if (th !== undefined) parts.push(Math.abs(th) >= 1 ? `mental effort was ${th > 0 ? "a little higher" : "a little lower"} than at rest (${th > 0 ? "+" : ""}${th.toFixed(1)} dB)` : "mental effort was about the same as at rest");
  let story = parts.length ? "During this ad, " + parts.join(", ") + "." : "";
  if (f !== undefined) story += Math.abs(f) < NOISE ? ` Whether they <i>wanted</i> the product can't be told from one person: the asymmetry (${f > 0 ? "+" : ""}${f.toFixed(2)}) is inside the noise range.`
                                                 : ` The asymmetry (${f > 0 ? "+" : ""}${f.toFixed(2)}) ${f > 0 ? "leans toward wanting" : "leans away from wanting"}, beyond the one-person noise range.`;
  if (R.liking !== undefined && R.purchase_intent !== undefined) {
    story += ` In their own words they rated <b>liking ${R.liking}/9</b> and <b>purchase intent ${R.purchase_intent}/9</b>`;
    story += (R.liking - R.purchase_intent >= 3) ? ": they enjoyed the ad much more than they wanted the product." : ".";
  }
  document.getElementById("story").innerHTML = story || "No baseline in this session, so no conclusion can be drawn.";
  document.querySelectorAll(".blk").forEach(b => b.classList.toggle("sel", b.dataset.ad === ad.name));
}

function spread(key) {   // middle half of the values: how much this line wobbles
  const v = ad[key].filter(x => x !== null).sort((a, b) => a - b);
  if (!v.length) return null;
  return [v[Math.round(0.25 * (v.length - 1))], v[Math.round(0.75 * (v.length - 1))]];
}

function baselineStory(S) {
  if (ad.kind === "baseline_eo" || ad.kind === "baseline_neutral") {
    const w = PANELS.map(p => { const r = spread(p.key); return r ? `${p.name} ${p.fmt(r[0])} to ${p.fmt(r[1])}` : null; }).filter(Boolean);
    return `<b>This is the reference block.</b> The participant sat still, eyes open, looking at a cross. Brain rhythms are never perfectly steady, ` +
      `so each line wobbles above and below "rest". The <b>middle value (median) of each line over this block <i>is</i> the "rest" level</b>, ` +
      `which is why every summary number here is 0 by definition. For half of the time during rest the lines stayed within: ${w.join("; ")}. ` +
      `A change during an ad has to be clearly bigger than this normal wobble before it means anything.`;
  }
  const a = S.alpha_suppression_pct, cc = D.meta.cap_check;
  if (a === undefined || !cc) return "Eyes closed: posterior alpha was not available in this session.";
  const ratio = cc.ratio;
  return `<b>Cap check.</b> When the eyes close, back-of-head alpha should rise (the Berger effect), because the visual brain goes idle. ` +
    `Over the whole block, alpha on ${cc.channels.join(", ")} was <b>${ratio.toFixed(2)}×</b> the eyes-open level. ` +
    `(The graph shows the same rise window by window: the middle value is ${(-a).toFixed(0)}% above rest, which appears as ${a.toFixed(0)}% on the attention scale, i.e. <i>less</i> visual attention, as expected.) ` +
    (ratio >= 1.3 ? "That is above the 1.3× threshold, so the back electrodes are picking up real brain activity."
                  : "That is below the 1.3× threshold, so the back electrodes may not be touching the scalp properly.") +
    ` This block is only a contact check; it is not used as the reference.`;
}

function loadAd(i) {
  adIdx = i; ad = D.ads[i]; postStart = null; vStart = null; tNow = 0;
  const cross = document.getElementById("cross");
  if (ad.video) { cross.style.display = "none"; vid.style.display = "block"; vid.src = ad.video; vid.currentTime = 0; }
  else {
    vid.pause(); vid.removeAttribute("src"); vid.load(); vid.style.display = "none"; cross.style.display = "flex";
    const closed = ad.kind === "baseline_ec";
    document.getElementById("crossPlus").textContent = closed ? "( – – )" : "+";
    document.getElementById("crossText").textContent = closed
      ? `Eyes closed for ${ad.duration.toFixed(0)} s; a beep tells the participant to open them. Press Play to step through the readings.`
      : `The participant looked at this cross with eyes open for ${ad.duration.toFixed(0)} s. These readings define "rest". Press Play to step through them.`;
  }
  conclusion(); draw();
}
sel.addEventListener("change", () => loadAd(+sel.value));
vid.addEventListener("ended", () => { if (ad.total > ad.duration) postStart = performance.now(); });
vid.addEventListener("seeking", () => { if (!vid.paused) { postStart = null; } });
vid.addEventListener("play", () => { if (tNow >= ad.total - 0.05) { tNow = 0; vid.currentTime = 0; } postStart = null; });
document.getElementById("playBtn").addEventListener("click", () => {
  if (!ad.video) {
    if (vStart !== null) { vStart = null; return; }
    if (tNow >= ad.total - 0.05) tNow = 0;
    vStart = performance.now() - tNow * 1000; return;
  }
  if (postStart !== null) { postStart = null; return; }
  if (vid.paused) { if (tNow > ad.duration && tNow < ad.total) { postStart = performance.now() - (tNow - ad.duration) * 1000; } else vid.play(); }
  else vid.pause();
});
vid.addEventListener("error", () => { if (!ad.video) return; document.getElementById("story").innerHTML =
  `<b style="color:#dc2626">Couldn't load ${ad.video}.</b> Keep this page inside data/lab_session/ next to the ads/ folder.`; });
(function loop() { draw(); requestAnimationFrame(loop); })();
window.addEventListener("resize", draw);
const firstAd = Math.max(0, D.ads.findIndex(a => a.kind === "ad"));
sel.value = firstAd; loadAd(firstAd);
</script>
</body></html>
"""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="session recording, e.g. data/lab_session/test1_raw.fif")
    ap.add_argument("--bad", default="", help="extra dead electrodes, comma-separated")
    args = ap.parse_args()
    build(args.path, [b.strip() for b in args.bad.split(",") if b.strip()])
