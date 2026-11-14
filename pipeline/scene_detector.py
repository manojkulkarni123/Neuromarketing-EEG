"""
Frame-exact candidate shot-change detector for the ad videos (labeller step 2).

Role in the labeller: this script decides WHEN something may have changed, to
the frame. It deliberately over-detects; the vision model later decides WHAT
each candidate is (real cut vs motion, same story scene vs new one). Scene
boundaries are only ever allowed to fall on frames this detector proposes.

Three kinds of candidate:
  cut      -- an abrupt change between two consecutive frames
  gradual  -- a dissolve / wipe: little change frame-to-frame, but large
              change across ~0.5 s (a histogram detector alone misses these)
  fade     -- entering or leaving a (near-)black screen

Per-frame signals (on a small copy of each frame):
  hist  -- Bhattacharyya distance between hue/saturation histograms of frame
           i-1 and i (colour content changed)
  edge  -- edge change ratio (Zabih et al.): fraction of edges that appear or
           disappear, after tolerating small movement (structure changed)
  score -- mean of the two, 0..1

Motion vs cut: fast motion (a galloping horse, a whip pan) raises the score on
MANY consecutive frames; a cut raises it on ONE. So each frame's score is
divided by the median score of its neighbours (excluding the frames right
next to it) -- the "adaptive ratio". A cut has a high ratio; sustained motion
does not. This is the same idea as PySceneDetect's AdaptiveDetector.

Timing convention: a candidate's `frame` is the FIRST frame of the new shot;
`time_s` = frame / fps is when that frame starts. The script checks the video
really has a constant frame rate (per-frame timestamps) and refuses to guess
if it does not.

Usage:
  python pipeline/scene_detector.py data/lab_session/ads/ad04.mp4 [--sheet]
  python pipeline/scene_detector.py --all [--sheet]
Writes data/lab_session/scenes/<ad>.cuts.json (+ <ad>.cuts_review.jpg with --sheet).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADS_DIR = os.path.join(PROJ, "data", "lab_session", "ads")
OUT_DIR = os.path.join(PROJ, "data", "lab_session", "scenes")

PARAMS = {
    "version": 1,
    "small_size": [160, 90],      # analysis resolution (w, h)
    "hist_bins": [32, 16],        # hue x saturation
    "neigh_frames": 8,            # adaptive baseline uses frames i-8..i-2 and i+2..i+8
    "cut_ratio": 2.5,             # score must be >= this x the neighbours' median ...
    "cut_min_score": 0.20,        # ... and at least this large
    "cut_abs_score": 0.55,        # or simply this large (very different frames)
    "strong_ratio": 4.0,          # strength label only: strong = clear-cut candidate
    "strong_score": 0.35,
    "gradual_half": 6,            # compare frame i-6 with i+6 for dissolves/wipes
    "gradual_min": 0.45,
    "black_level": 0.06,          # mean brightness (0..1) below which a frame is "black"
    "min_gap_frames": 3,          # candidates closer than this are merged (keep strongest)
}


def _frame_features(frame, p):
    w, h = p["small_size"]
    small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, p["hist_bins"], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None).flatten()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 140) > 0
    return hist, edges, float(hsv[..., 2].mean()) / 255.0


def _edge_change_ratio(e_prev, e_cur):
    """Fraction of edges entering/leaving, tolerating ~2px movement."""
    k = np.ones((5, 5), np.uint8)
    d_prev = cv2.dilate(e_prev.astype(np.uint8), k) > 0
    d_cur = cv2.dilate(e_cur.astype(np.uint8), k) > 0
    n_prev, n_cur = e_prev.sum(), e_cur.sum()
    if n_prev < 20 and n_cur < 20:          # both nearly featureless (e.g. black)
        return 0.0
    entering = (e_cur & ~d_prev).sum() / max(n_cur, 1)
    leaving = (e_prev & ~d_cur).sum() / max(n_prev, 1)
    return float(max(entering, leaving))


def read_signals(path, p=PARAMS):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hists, bright, hist_d, edge_d, pts = [], [], [0.0], [0.0], []
    prev_edges = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        hist, edges, v = _frame_features(frame, p)
        if hists:
            hist_d.append(cv2.compareHist(hists[-1], hist, cv2.HISTCMP_BHATTACHARYYA))
            edge_d.append(_edge_change_ratio(prev_edges, edges))
        hists.append(hist)
        bright.append(v)
        prev_edges = edges
    cap.release()
    n = len(hists)
    if n < 2:
        raise RuntimeError(f"{path}: no frames decoded")
    pts = np.array(pts)
    # POS_MSEC is reported after the read, i.e. it can be the NEXT frame's
    # time depending on backend -- compare spacing, not absolute values.
    dt = np.diff(pts)
    cfr = bool(np.all(np.abs(dt - 1.0 / fps) < 0.25 / fps))
    return {
        "fps": fps, "n_frames": n, "size": [W, H], "constant_frame_rate": cfr,
        "hists": np.array(hists), "bright": np.array(bright),
        "hist_d": np.array(hist_d), "edge_d": np.array(edge_d),
    }


def find_candidates(sig, p=PARAMS):
    n, fps = sig["n_frames"], sig["fps"]
    hd, ed = sig["hist_d"], sig["edge_d"]
    score = (hd + ed) / 2.0
    k = p["neigh_frames"]
    cands = []

    # --- abrupt cuts -------------------------------------------------------
    for i in range(1, n):
        lo, hi = max(1, i - 2), min(n, i + 3)
        if score[i] < score[lo:hi].max():            # not the local peak
            continue
        neigh = np.r_[score[max(1, i - k):max(1, i - 1)], score[min(n, i + 2):min(n, i + k + 1)]]
        base = float(np.median(neigh)) if len(neigh) else 0.0
        ratio = score[i] / (base + 0.02)
        if (ratio >= p["cut_ratio"] and score[i] >= p["cut_min_score"]) or score[i] >= p["cut_abs_score"]:
            strong = ratio >= p["strong_ratio"] and score[i] >= p["strong_score"]
            cands.append({"frame": i, "kind": "cut", "strength": "strong" if strong else "weak",
                          "score": round(float(score[i]), 3), "ratio": round(float(ratio), 2),
                          "hist": round(float(hd[i]), 3), "edge": round(float(ed[i]), 3)})

    # --- gradual transitions (dissolve / wipe) -----------------------------
    g = p["gradual_half"]
    H = sig["hists"]
    glong = np.zeros(n)
    for i in range(g, n - g):
        glong[i] = cv2.compareHist(H[i - g], H[i + g], cv2.HISTCMP_BHATTACHARYYA)
    cut_frames = [c["frame"] for c in cands]
    for i in range(g, n - g):
        if glong[i] < p["gradual_min"] or glong[i] < glong[max(0, i - g):i + g + 1].max():
            continue
        if any(abs(i - c) <= g + 2 for c in cut_frames):      # explained by a hard cut
            continue
        cands.append({"frame": i, "kind": "gradual", "strength": "weak",
                      "score": round(float(glong[i]), 3), "span": [i - g, i + g]})

    # --- fades to / from black ---------------------------------------------
    black = sig["bright"] < p["black_level"]
    for i in range(1, n):
        if black[i] != black[i - 1]:
            cands.append({"frame": i, "kind": "fade", "strength": "strong",
                          "score": round(float(abs(sig["bright"][i] - sig["bright"][i - 1])), 3),
                          "into_black": bool(black[i])})

    # --- merge near-duplicates (keep the strongest) -------------------------
    rank = {"strong": 1, "weak": 0}
    cands.sort(key=lambda c: c["frame"])
    merged = []
    for c in cands:
        if merged and c["frame"] - merged[-1]["frame"] < p["min_gap_frames"]:
            a = merged[-1]
            if (rank[c["strength"]], c["score"]) > (rank[a["strength"]], a["score"]):
                merged[-1] = c
            continue
        merged.append(c)
    for c in merged:
        c["time_s"] = round(c["frame"] / fps, 3)
        if "span" in c:
            c["span_s"] = [round(c["span"][0] / fps, 3), round(c["span"][1] / fps, 3)]
    return merged, score


def refine_boundary(sig, t_draft, candidates, window_s=1.5, half=12):
    """Turn a model's rough boundary time (~1 s accurate) into a frame.

    1. If detector candidates lie within +-window_s, return them (nearest
       first) -- the caller / vision model picks one. method = "snap".
    2. Otherwise the change happened inside a continuous shot (a set change in
       a one-take ad, a slow dissolve, a camera move onto a new subject), which
       a frame-to-frame detector cannot see. Pick the frame where colour
       content changes most across +-half frames inside the window.
       method = "refined_no_cut" -> always flagged for human review.
    """
    fps, n = sig["fps"], sig["n_frames"]
    near = sorted((c for c in candidates if abs(c["time_s"] - t_draft) <= window_s),
                  key=lambda c: abs(c["time_s"] - t_draft))
    if near:
        return {"method": "snap", "options": [c["frame"] for c in near]}
    H = sig["hists"]
    lo = max(half, int((t_draft - window_s) * fps))
    hi = min(n - half - 1, int((t_draft + window_s) * fps))
    if hi <= lo:
        f = int(round(t_draft * fps))
        return {"method": "refined_no_cut", "frame": f, "change": 0.0}
    vals = [cv2.compareHist(H[i - half], H[i + half], cv2.HISTCMP_BHATTACHARYYA) for i in range(lo, hi + 1)]
    k = int(np.argmax(vals))
    return {"method": "refined_no_cut", "frame": lo + k, "change": round(float(vals[k]), 3)}


def detect(path, p=PARAMS, return_signals=False):
    sig = read_signals(path, p)
    if not sig["constant_frame_rate"]:
        raise RuntimeError(f"{os.path.basename(path)} has a variable frame rate -- frame/fps "
                           "times would be wrong. Re-encode it to constant fps first "
                           "(ffmpeg -i in.mp4 -vf fps=25 out.mp4).")
    cands, _ = find_candidates(sig, p)
    res = _result(path, sig, cands, p)
    return (res, sig) if return_signals else res


def _result(path, sig, cands, p):
    return {
        "video": os.path.basename(path),
        "fps": round(sig["fps"], 4),
        "n_frames": sig["n_frames"],
        "duration_s": round(sig["n_frames"] / sig["fps"], 3),
        "frame_size": sig["size"],
        "time_convention": "time_s = frame / fps = start of that frame; candidate frame = first frame of the new shot",
        "detector": p,
        "n_candidates": len(cands),
        "candidates": cands,
    }


def review_sheet(path, result, out_path, thumb=(192, 108)):
    """Before | after | +0.3 s thumbnails for every candidate, for a human or model to check."""
    from PIL import Image, ImageDraw
    cap = cv2.VideoCapture(path)
    fps, n = result["fps"], result["n_frames"]
    tw, th = thumb
    rows = result["candidates"]
    sheet = Image.new("RGB", (tw * 3 + 230, (th + 4) * max(1, len(rows))), "white")
    d = ImageDraw.Draw(sheet)

    def grab(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, i)))
        ok, fr = cap.read()
        return Image.fromarray(cv2.cvtColor(cv2.resize(fr, thumb), cv2.COLOR_BGR2RGB)) if ok else None

    for r, c in enumerate(rows):
        y = r * (th + 4)
        f = c["frame"]
        before = c["span"][0] if c["kind"] == "gradual" else f - 1
        after = c["span"][1] if c["kind"] == "gradual" else f
        for j, fi in enumerate((before, after, after + int(round(0.3 * fps)))):
            im = grab(fi)
            if im is not None:
                sheet.paste(im, (j * tw, y))
        d.text((tw * 3 + 6, y + 6), f"#{r}  {c['time_s']:.2f}s  f{f}", fill="black")
        d.text((tw * 3 + 6, y + 24), f"{c['kind']} / {c['strength']}", fill="black")
        d.text((tw * 3 + 6, y + 42), f"score {c['score']}" + (f"  ratio {c['ratio']}" if "ratio" in c else ""), fill="black")
    sheet.save(out_path, quality=80)
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--all", action="store_true", help="every .mp4 in data/lab_session/ads")
    ap.add_argument("--sheet", action="store_true", help="also write a before/after review image")
    a = ap.parse_args()
    vids = sorted(glob.glob(os.path.join(ADS_DIR, "*.mp4"))) if a.all else a.videos
    if not vids:
        ap.error("give video paths or --all")
    os.makedirs(OUT_DIR, exist_ok=True)
    for v in vids:
        res = detect(v)
        base = os.path.splitext(os.path.basename(v))[0]
        out = os.path.join(OUT_DIR, f"{base}.cuts.json")
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)
        kinds = {}
        for c in res["candidates"]:
            kinds[f"{c['kind']}/{c['strength']}"] = kinds.get(f"{c['kind']}/{c['strength']}", 0) + 1
        print(f"{base}: {res['duration_s']:.1f}s @ {res['fps']}fps -> {res['n_candidates']} candidates {kinds}  -> {out}")
        if a.sheet:
            review_sheet(v, res, os.path.join(OUT_DIR, f"{base}.cuts_review.jpg"))


if __name__ == "__main__":
    main()
