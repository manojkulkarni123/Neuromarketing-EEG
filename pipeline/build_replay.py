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
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_mne import load_recording, clean_raw, quality_check
import extract_neuromarketing_features as ef

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "lab_session")


def _clean(arr):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), 4) for v in arr]


def build(path, extra_bad=()):
    raw = load_recording(path)
    raw.info["bads"] = sorted(set(quality_check(raw)) | set(ef.KNOWN_BAD) | set(extra_bad))
    cleaned = clean_raw(raw)
    tl = ef.sliding_window_features(cleaned)
    segs = ef.ad_segments(cleaned.annotations)
    if not segs:
        raise SystemExit("No AD_START/AD_END markers in this recording -- nothing to replay.")
    ref_kind, ref = ef.baseline_reference(tl, ef.baseline_segments(cleaned.annotations))
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
    ads = []
    for name, s, e in segs:
        # Include the post-ad gray screen (if recorded): the brain reacts to the
        # ending/logo a little late, so that reaction lands just after AD_END.
        end = post_ends.get(name, e)
        m = (t - half >= s) & (t + half <= end)
        sub = {k: tl[k][m] for k in ("faa", "theta", "post_alpha", "engagement")}
        center = lambda v: np.nanmedian(v) if np.any(np.isfinite(v)) else 0.0
        ads.append({
            "name": name,
            "video": f"ads/{name}",
            "duration": round(float(e - s), 2),          # video length on the recording clock
            "total": round(float(end - s), 2),           # video + post-ad gray screen
            "t": _clean(t[m] - s),
            "faa": _clean(series("faa", "faa", lambda v, b: v[m] - b, center)),
            "engagement": _clean(series("engagement", "engagement_log",
                                        lambda v, b: 100 * (10 ** (v[m] - b) - 1), center)),
            "theta": _clean(series("theta", "theta_log", lambda v, b: 10 * (v[m] - b), center)),
            # sign flipped so UP = alpha suppressed = more visual attention, like the other panels
            "alpha": _clean(series("post_alpha", "post_alpha_log", lambda v, b: -10 * (v[m] - b), center)),
        })

    meta = {
        "recording": os.path.basename(path),
        "baseline": ref_kind or "none (values relative to the session median instead)",
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
  body { font-family: system-ui, Segoe UI, sans-serif; margin: 0; background: #f5f5f4; color: #1c1c1c; }
  header { padding: 10px 18px; background: #fff; border-bottom: 1px solid #ddd; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; }
  header .meta { font-size: 12px; color: #555; }
  main { display: grid; grid-template-columns: minmax(320px, 46%) 1fr; gap: 14px; padding: 14px 18px; }
  video { width: 100%; background: #000; border-radius: 6px; }
  .readout { margin-top: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 12px; font-size: 13px; }
  .readout .t { font-size: 20px; font-weight: 600; }
  .readout table { width: 100%; border-collapse: collapse; margin-top: 6px; }
  .readout td { padding: 3px 0; }
  .readout td.v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .note { font-size: 12px; color: #666; margin-top: 10px; line-height: 1.45; }
  .panel { background: #fff; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 10px; padding: 6px 8px 2px; }
  .panel .title { font-size: 12px; font-weight: 600; display: flex; justify-content: space-between; }
  .panel .title span { font-weight: 400; color: #666; }
  canvas { width: 100%; height: 130px; display: block; cursor: crosshair; }
  select, button { font-size: 13px; padding: 4px 8px; }
</style></head>
<body>
<header>
  <h1>EEG Ad Replay</h1>
  <label>Ad: <select id="adSel"></select></label>
  <button id="playBtn">Play / Pause</button>
  <span class="meta" id="metaLine"></span>
</header>
<main>
  <div>
    <video id="vid" controls preload="auto"></video>
    <div class="readout">
      <div class="t" id="tNow">0.0 s</div>
      <div id="winLine" style="font-size:12px;color:#666"></div>
      <table id="vals"></table>
    </div>
    <div class="note" id="note"></div>
  </div>
  <div id="panels"></div>
</main>
<script>
const D = __DATA__;
const PANELS = [
  {key: "faa",        label: "Frontal asymmetry",        unit: "",   up: "up = approach (wants it)",       color: "#2563eb", fmt: v => v.toFixed(2)},
  {key: "engagement", label: "Engagement index",         unit: "%",  up: "up = more alert/engaged",        color: "#16a34a", fmt: v => (v>0?"+":"") + v.toFixed(0) + "%"},
  {key: "theta",      label: "Frontal theta",            unit: "dB", up: "up = more mental processing",    color: "#ea580c", fmt: v => (v>0?"+":"") + v.toFixed(1) + " dB"},
  {key: "alpha",      label: "Posterior alpha suppression", unit: "dB", up: "up = more visual attention",  color: "#9333ea", fmt: v => (v>0?"+":"") + v.toFixed(1) + " dB"},
];
const vid = document.getElementById("vid"), sel = document.getElementById("adSel");
const half = D.meta.window_s / 2;
let ad = null, canvases = {};
// Playhead on the timeline. Normally follows the video; after the video ends it
// keeps running through the post-ad gray screen (the video holds its last frame).
let tNow = 0, postStart = null;
function curTime() {
  if (postStart !== null) return Math.min(ad.total, ad.duration + (performance.now() - postStart) / 1000);
  return vid.currentTime;
}
function setTime(t) {
  postStart = null; vid.pause();
  tNow = Math.max(0, Math.min(ad.total, t));
  vid.currentTime = Math.min(tNow, Math.max(0, vid.duration ? vid.duration - 0.05 : ad.duration));
}

document.getElementById("metaLine").textContent =
  `${D.meta.recording} · baseline: ${D.meta.baseline} · ${D.meta.window_s}s windows · excluded electrodes: ${D.meta.bad.join(", ") || "none"}`;
document.getElementById("note").innerHTML =
  `<b>How to read this.</b> Each graph point summarizes ${D.meta.window_s} seconds of brain signal, so a spike at 14 s ` +
  `reflects roughly ${14-half}–${14+half} s (the shaded band). The brain also reacts a few hundred ms after something appears. ` +
  `Values are relative to this person's baseline (dashed line = 0). One person's graph is noisy; only patterns that ` +
  `repeat across participants are findings. The shaded "after ad" part is the gray screen shown after each ad: ` +
  `reactions to the ending and brand logo show up there, a little after they appear on screen.`;

D.ads.forEach((a, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `${a.name} (${a.duration.toFixed(0)} s)`; sel.appendChild(o); });

const panelsEl = document.getElementById("panels");
PANELS.forEach(p => {
  const div = document.createElement("div"); div.className = "panel";
  div.innerHTML = `<div class="title">${p.label} <span>${D.meta.channels[p.key]} · ${p.up}</span></div>`;
  const c = document.createElement("canvas"); div.appendChild(c); panelsEl.appendChild(div);
  canvases[p.key] = c;
  const seek = ev => { if (!ad) return; const r = c.getBoundingClientRect();
    const x = (ev.clientX - r.left - 44) / (r.width - 52);   // match drawPanel's L/R margins
    setTime(x * ad.total); draw(); };
  let dragging = false;
  c.addEventListener("mousedown", ev => { dragging = true; vid.pause(); seek(ev); });
  window.addEventListener("mousemove", ev => { if (dragging) seek(ev); });
  window.addEventListener("mouseup", () => dragging = false);
});

const vals = document.getElementById("vals");
PANELS.forEach(p => { const tr = document.createElement("tr"); tr.innerHTML = `<td>${p.label}</td><td class="v" id="v_${p.key}">–</td>`; vals.appendChild(tr); });

function valueAt(arr, t) {
  let best = -1, bd = 1e9;
  ad.t.forEach((tt, i) => { const d = Math.abs(tt - t); if (d < bd) { bd = d; best = i; } });
  return (best >= 0 && bd <= half) ? arr[best] : null;
}

function drawPanel(p) {
  const c = canvases[p.key], dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight; c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const L = 44, R = 8, T = 6, B = 18, pw = W - L - R, ph = H - T - B;
  const ys = ad[p.key].filter(v => v !== null);
  const tx = t => L + (t / ad.total) * pw;
  g.font = "11px system-ui"; g.fillStyle = "#666";
  if (!ys.length) { g.fillText("Unavailable for this session (no working electrodes)", L + 8, T + ph / 2); return; }
  let lo = Math.min(0, ...ys), hi = Math.max(0, ...ys); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  const ty = v => T + (1 - (v - lo) / (hi - lo)) * ph;
  if (ad.total > ad.duration) {   // post-ad gray screen
    g.fillStyle = "rgba(120,120,120,0.12)"; g.fillRect(tx(ad.duration), T, tx(ad.total) - tx(ad.duration), ph);
    g.fillStyle = "#777"; g.textAlign = "center"; g.fillText("after ad", (tx(ad.duration) + tx(ad.total)) / 2, T + 11);
  }
  g.fillStyle = "rgba(0,0,0,0.07)"; g.fillRect(tx(Math.max(0, tNow - half)), T, tx(Math.min(ad.total, tNow + half)) - tx(Math.max(0, tNow - half)), ph);
  g.strokeStyle = "#999"; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(L, ty(0)); g.lineTo(L + pw, ty(0)); g.stroke(); g.setLineDash([]);
  g.fillStyle = "#666"; g.textAlign = "right";
  [hi - pad, 0, lo + pad].forEach(v => g.fillText(p.fmt(v), L - 4, ty(v) + 4));
  g.textAlign = "center";
  for (let s = 0; s <= ad.total; s += (ad.total > 60 ? 10 : 5)) g.fillText(s + "s", tx(s), H - 4);
  g.strokeStyle = p.color; g.lineWidth = 1.8; g.beginPath(); let started = false;
  ad.t.forEach((t, i) => { const v = ad[p.key][i]; if (v === null) { started = false; return; }
    if (!started) { g.moveTo(tx(t), ty(v)); started = true; } else g.lineTo(tx(t), ty(v)); });
  g.stroke();
  g.strokeStyle = "#dc2626"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(tx(tNow), T); g.lineTo(tx(tNow), T + ph); g.stroke();
}

function draw() {
  if (!ad) return;
  if (!vid.paused || postStart !== null) tNow = curTime();
  PANELS.forEach(drawPanel);
  const t = tNow;
  document.getElementById("tNow").textContent = t.toFixed(1) + " s" + (t > ad.duration ? "  (gray screen after the ad)" : "");
  document.getElementById("winLine").textContent = `values here summarize ${Math.max(0, t - half).toFixed(1)}–${Math.min(ad.total, t + half).toFixed(1)} s`;
  PANELS.forEach(p => { const v = valueAt(ad[p.key], t); document.getElementById("v_" + p.key).textContent = v === null ? "–" : p.fmt(v); });
}

function loadAd(i) { ad = D.ads[i]; postStart = null; tNow = 0; vid.src = ad.video; vid.currentTime = 0; draw(); }
sel.addEventListener("change", () => loadAd(+sel.value));
vid.addEventListener("ended", () => { if (ad.total > ad.duration) postStart = performance.now(); });
vid.addEventListener("seeking", () => { if (!vid.paused) { postStart = null; } });
vid.addEventListener("play", () => { if (tNow >= ad.total - 0.05) { tNow = 0; vid.currentTime = 0; } postStart = null; });
document.getElementById("playBtn").addEventListener("click", () => {
  if (postStart !== null) { postStart = null; return; }             // pause during the gray-screen part
  if (vid.paused) { if (tNow > ad.duration && tNow < ad.total) { postStart = performance.now() - (tNow - ad.duration) * 1000; } else vid.play(); }
  else vid.pause();
});
vid.addEventListener("error", () => { document.getElementById("note").innerHTML =
  `<b style="color:#dc2626">Couldn't load ${ad.video}.</b> Keep this page inside data/lab_session/ next to the ads/ folder.`; });
(function loop() { draw(); requestAnimationFrame(loop); })();
window.addEventListener("resize", draw);
loadAd(0);
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
