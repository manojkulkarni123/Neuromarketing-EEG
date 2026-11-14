"""
AI scene labeller for the ad videos: story-level scenes with frame-exact
boundaries, plus brand events. Runs ONCE per ad, before any recording; the
result is frozen after a human check and reused for every participant.

Division of labour (the core rule): the vision model decides MEANING, the
detector decides TIME. No scene boundary is ever placed at a time the model
typed in; every boundary is a detector frame (or, when the change happens
inside a continuous shot, a frame found by local refinement -- always flagged
for the human).

  Stage 0  audio        ad_audio.py: loudness curve + whisper transcript (word times)
  Stage 1  story pass   model sees ~1 frame/s + what is heard each second -> story summary, draft scenes
                        (label, description, story role) and draft brand
                        events, all with ROUGH times (~1 s)
  Stage 2  cut pass     scene_detector.py -> frame-exact candidate cuts
  Stage 3  snap         each draft boundary -> nearest detector candidates
                        within +-1.5 s; model picks between them from
                        before/after frames ("none" -> refine, flag). Draft
                        boundaries with no candidate -> refined, flagged.
  Stage 4  brand frames for each brand event, a strip of frames (~6/s) around
                        the rough start/end; model picks the first/last frame
                        on which the brand is visible
  Stage 5  human        <ad>.scenes_review.jpg + edit <ad>.scenes.json, then
                        `--approve` marks it reviewed (frozen)

What is sent to the API: only frames of the (public) ad video and text
prompts. No EEG, no participant data. store=False.

Usage:
  set OPENAI_API_KEY first (setx OPENAI_API_KEY "sk-...", new terminal)
  python pipeline/label_scenes.py ad04 [--model NAME] [--overwrite]
  python pipeline/label_scenes.py --all
  python pipeline/label_scenes.py ad04 --dry-run     # build everything, no API call
  python pipeline/label_scenes.py ad04 --approve     # after human review
Output: data/lab_session/scenes/<ad>.scenes.json (+ _review.jpg, raw model replies inside)
"""
import argparse
import base64
import datetime as dt
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_detector as SD  # noqa: E402
import ad_audio as AA  # noqa: E402

PROJ = SD.PROJ
ADS_DIR, OUT_DIR = SD.ADS_DIR, SD.OUT_DIR
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "")
STORY_ROLES = ["setup", "build_up", "climax", "resolution", "brand", "other"]
VISUAL_BRAND_KINDS = ["logo", "product", "brand_text", "tagline"]
BRAND_KINDS = VISUAL_BRAND_KINDS + ["spoken_brand"]       # spoken_brand comes from the transcript, not the model
SNAP_WINDOW_S = 1.5
MIN_SCENE_S = 3.0


# ----------------------------------------------------------------------------- frames
class Video:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frame(self, i, width=448):
        i = int(max(0, min(self.n - 1, i)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = self.cap.read()
        if not ok:
            return None
        h, w = fr.shape[:2]
        return cv2.resize(fr, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)

    def jpeg_b64(self, i, width=448, quality=80):
        fr = self.frame(i, width)
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode()


def _img(video, i, width=448, detail="low"):
    return {"type": "input_image", "detail": detail,
            "image_url": "data:image/jpeg;base64," + video.jpeg_b64(i, width)}


def _txt(s):
    return {"type": "input_text", "text": s}


# ----------------------------------------------------------------------------- model call
class Model:
    def __init__(self, name, dry_run=False, log=None):
        self.name, self.dry, self.log = name, dry_run, log if log is not None else []
        if not dry_run:
            from openai import OpenAI
            if not AA.load_key():
                sys.exit("OPENAI_API_KEY is not set. In a terminal: setx OPENAI_API_KEY \"sk-...\" "
                         "then open a NEW terminal / restart VS Code.")
            if not name:
                sys.exit("No model given. Pass --model NAME or set OPENAI_MODEL.")
            self.client = OpenAI()

    def ask(self, stage, instructions, content, schema, fallback):
        n_img = sum(1 for c in content if c["type"] == "input_image")
        if self.dry:
            self.log.append({"stage": stage, "dry_run": True, "n_images": n_img})
            print(f"  [dry-run] {stage}: {n_img} images, {len(instructions)} chars of instructions")
            return fallback()
        r = self.client.responses.create(
            model=self.name, instructions=instructions,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": stage, "schema": schema, "strict": True}},
            store=False,
        )
        out = json.loads(r.output_text)
        usage = getattr(r, "usage", None)
        self.log.append({"stage": stage, "n_images": n_img, "reply": out,
                         "usage": usage.model_dump() if usage is not None else None})
        return out


def _obj(props, required=None):
    return {"type": "object", "properties": props, "required": required or list(props),
            "additionalProperties": False}


# ----------------------------------------------------------------------------- stage 1
STORY_INSTRUCTIONS = """You are annotating a TV/online advertisement for a neuromarketing study.
You will see frames sampled about once per second. Before each frame you get its timestamp
and what is HEARD in that second: sound level in dB (0 = maximum; -60 = near silence; a jump
of 10+ dB is a sudden loud moment) and the words spoken or sung (from automatic speech
recognition; song lyrics are marked with the music sign and lines marked (uncertain) may be
wrong). Use sight AND sound together: jokes, voice-over, lyrics, music swells and silences
are often what a scene is about.

1. Understand the whole ad first: who/what is shown, where, what happens, the story arc,
   and which brand/product is advertised.
2. Split the ad into STORY scenes: a scene is a meaningful event or situation, usually
   several camera shots. Prefer scenes of 3 s or longer, but keep a shorter scene when it
   is an important beat on its own (above all the brand reveal / end card).
3. Give each scene a story role: setup, build_up, climax, resolution, brand (logo/end
   card/product hero shot), or other.
4. List brand events: moments when the brand is VISIBLE -- logo, the product itself,
   brand name as text, tagline. These can happen inside a scene (a logo on a cap).
   (Spoken brand mentions are found separately from the transcript -- just list, in
   brand_spoken_forms, how the brand/product name would be said aloud, e.g. "Old Spice",
   "Volkswagen", "VW".)
5. For each scene, say briefly what is heard (voice-over line, lyric, music, silence).

Timing: your times only need to be roughly right (+-1 s); exact frames are found later
by other tools. Scenes must start at 0, end at the given duration, and be contiguous
(each scene's start_s = the previous scene's end_s). Describe only what is visible --
do not guess at what the viewer feels."""


def _heard(per_sec, t, step):
    rows = [per_sec[s] for s in range(int(t), int(t + step)) if s in per_sec]
    if not rows:
        return "no audio"
    lvl = np.nanmean([r[0] for r in rows])
    words = " ".join(r[1] for r in rows if r[1]).strip()
    unc = any(r[2] for r in rows if r[1])
    out = f"sound {lvl:.0f} dB"
    if words:
        return out + " | heard: " + words + (" (uncertain)" if unc else "")
    return out + " | no words"


def stage1_story(model, video, duration, audio=None):
    step = 1.0 if duration <= 75 else duration / 75.0       # at most ~75 frames
    times = list(np.arange(0.0, duration - 0.05, step))
    per_sec = AA.per_second_summary(audio, duration) if audio else {}
    head = f"Ad duration: {duration:.2f} s."
    if audio and audio.get("has_audio"):
        jumps = [f"{e['t']:.1f} s (+{e['rise_db']:.0f} dB)" for e in audio["events"] if e["kind"] == "jump"]
        sil = [f"{e['start']:.1f}-{e['end']:.1f} s" for e in audio["events"] if e["kind"] == "silence"]
        head += (" Sudden loud moments: " + (", ".join(jumps) or "none") +
                 ". Silences: " + (", ".join(sil) or "none") + ".")
    else:
        head += " (No audio track.)"
    content = [_txt(head + " Frames follow.")]
    for t in times:
        content += [_txt(f"t = {t:.1f} s | {_heard(per_sec, t, step)}"),
                    _img(video, int(round(t * video.fps)), width=384)]
    schema = _obj({
        "brand": {"type": "string"},
        "product": {"type": "string"},
        "summary": {"type": "string"},
        "audio_summary": {"type": "string"},
        "brand_spoken_forms": {"type": "array", "items": {"type": "string"}},
        "scenes": {"type": "array", "items": _obj({
            "start_s": {"type": "number"}, "end_s": {"type": "number"},
            "label": {"type": "string"}, "description": {"type": "string"},
            "audio": {"type": "string"},
            "story_role": {"type": "string", "enum": STORY_ROLES}})},
        "brand_events": {"type": "array", "items": _obj({
            "start_s": {"type": "number"}, "end_s": {"type": "number"},
            "kind": {"type": "string", "enum": VISUAL_BRAND_KINDS}, "what": {"type": "string"}})},
    })

    def fake():   # dry run: equal 6 s scenes so later stages can be exercised
        edges = list(np.arange(0, duration, 6.0)) + [duration]
        return {"brand": "?", "product": "?", "summary": "(dry run)", "audio_summary": "",
                "brand_spoken_forms": [],
                "scenes": [{"start_s": float(a), "end_s": float(b), "label": f"scene {k}",
                            "description": "", "audio": "", "story_role": "other"}
                           for k, (a, b) in enumerate(zip(edges, edges[1:]))],
                "brand_events": [{"start_s": max(0.0, duration - 3), "end_s": duration,
                                  "kind": "logo", "what": "(dry run)"}]}
    return model.ask("story", STORY_INSTRUCTIONS, content, schema, fake)


# ----------------------------------------------------------------------------- stage 3
SNAP_INSTRUCTIONS = """You are placing scene boundaries in an advertisement exactly.
For each boundary you get: the scene before and after it (labels), and up to a few OPTIONS.
Each option is a candidate camera change shown as two frames: the last frame before it and
the first frame after it. Pick the option where the scene before ENDS and the scene after
BEGINS. Answer the option number (0, 1, ...), or -1 if no option is that boundary (e.g. they are motion inside one shot,
or the change happens elsewhere). Judge from the images, not from the times."""


def stage3_snap(model, video, draft_scenes, cands, sig):
    boundaries = []
    for k in range(1, len(draft_scenes)):
        t = draft_scenes[k]["start_s"]
        boundaries.append({"k": k, "t_draft": t,
                           "ref": SD.refine_boundary(sig, t, cands, window_s=SNAP_WINDOW_S)})
    content, asked = [], []
    for b in boundaries:
        if b["ref"]["method"] != "snap":
            continue
        opts = b["ref"]["options"][:4]
        k = b["k"]
        content.append(_txt(f"BOUNDARY {k}: between scene '{draft_scenes[k-1]['label']}' and "
                            f"scene '{draft_scenes[k]['label']}'. Options:"))
        for j, f in enumerate(opts):
            content += [_txt(f"  option {j}: before"), _img(video, f - 1, 320),
                        _txt(f"  option {j}: after"), _img(video, f, 320)]
        asked.append((k, opts))
    choices = {}
    if asked:
        schema = _obj({"choices": {"type": "array", "items": _obj({
            "boundary": {"type": "integer"},
            "option": {"type": "integer"},         # 0, 1, ... or -1 for none
            "reason": {"type": "string"}})}})
        reply = model.ask("snap", SNAP_INSTRUCTIONS, content, schema,
                          lambda: {"choices": [{"boundary": k, "option": 0, "reason": "dry run"}
                                               for k, _ in asked]})
        choices = {c["boundary"]: c for c in reply["choices"]}

    out = []
    for b in boundaries:
        k, ref = b["k"], b["ref"]
        if ref["method"] == "snap":
            opts = ref["options"][:4]
            c = choices.get(k, {"option": -1, "reason": "model gave no answer"})
            if 0 <= int(c["option"]) < len(opts):
                out.append({"k": k, "frame": opts[int(c["option"])], "method": "cut",
                            "reason": c["reason"], "flags": []})
                continue
            r2 = SD.refine_boundary(sig, b["t_draft"], [], window_s=SNAP_WINDOW_S)
            out.append({"k": k, "frame": r2["frame"], "method": "refined_no_cut",
                        "reason": c.get("reason", ""), "flags": ["model rejected all cut options"]})
        else:
            out.append({"k": k, "frame": ref["frame"], "method": "refined_no_cut", "reason": "",
                        "flags": ["no detector cut within 1.5 s (in-shot change?)"]})
    return out


# ----------------------------------------------------------------------------- stage 4
BRAND_INSTRUCTIONS = """You are timing brand appearances in an advertisement exactly.
For each brand event you get a strip of consecutive numbered frames around its rough START
and a strip around its rough END. Return the number of the FIRST frame on which the brand
item is clearly visible (in the start strip) and the LAST frame on which it is still visible
(in the end strip). Use -1 if it is visible in none of that strip's frames."""


def stage4_brand(model, video, draft_events, duration):
    n, fps = video.n, video.fps
    step = max(1, int(round(fps / 6)))                       # ~6 frames per second
    content, plan = [], []
    for e_idx, ev in enumerate(draft_events):
        s_strip = [f for f in range(int((ev["start_s"] - 1.5) * fps), int((ev["start_s"] + 1.5) * fps) + 1, step)
                   if 0 <= f < n]
        e_strip = [f for f in range(int((ev["end_s"] - 1.5) * fps), int((ev["end_s"] + 1.5) * fps) + 1, step)
                   if 0 <= f < n]
        content.append(_txt(f"EVENT {e_idx}: {ev['kind']} -- {ev['what']}. START strip:"))
        for f in s_strip:
            content += [_txt(f"frame {f}"), _img(video, f, 256)]
        content.append(_txt(f"EVENT {e_idx}: END strip:"))
        for f in e_strip:
            content += [_txt(f"frame {f}"), _img(video, f, 256)]
        plan.append((e_idx, s_strip, e_strip))
    if not plan:
        return []
    schema = _obj({"events": {"type": "array", "items": _obj({
        "event": {"type": "integer"}, "first_frame": {"type": "integer"},
        "last_frame": {"type": "integer"}})}})
    reply = model.ask("brand", BRAND_INSTRUCTIONS, content, schema,
                      lambda: {"events": [{"event": i, "first_frame": s[len(s) // 2] if s else -1,
                                           "last_frame": e[len(e) // 2] if e else -1} for i, s, e in plan]})
    got = {r["event"]: r for r in reply["events"]}
    out = []
    for e_idx, s_strip, e_strip in plan:
        ev, r = draft_events[e_idx], got.get(e_idx, {"first_frame": -1, "last_frame": -1})
        flags = []
        first, last = r["first_frame"], r["last_frame"]
        if first < 0:
            first, flags = int(round(ev["start_s"] * fps)), flags + ["start not found in strip (rough time kept)"]
        if last < 0:
            last, flags = min(n - 1, int(round(ev["end_s"] * fps))), flags + ["end not found in strip (rough time kept)"]
        if ev["end_s"] >= duration - 0.5 and r["last_frame"] >= 0 and e_strip and r["last_frame"] == e_strip[-1]:
            last = n - 1                                        # still visible at the end of the ad
        # strips are sampled every `step` frames: the true edge is within step-1 frames
        out.append({"start_frame": int(first), "end_frame": int(last) + 1, "kind": ev["kind"],
                    "what": ev["what"], "precision_frames": step - 1, "flags": flags,
                    "human_checked": False})
    return out


# ----------------------------------------------------------------------------- spoken brand
def _norm(w):
    return "".join(ch for ch in w.lower() if ch.isalnum())


def spoken_brand_events(audio, forms, fps, n):
    """Brand names said aloud: exact word matches in the transcript (no model timing)."""
    tr = (audio or {}).get("transcript") or {}
    words = tr.get("words") or []
    toks = [_norm(w["word"]) for w in words]
    out = []
    for form in sorted({f.strip() for f in forms if f and f.strip()}, key=len, reverse=True):
        ft = [_norm(x) for x in form.split() if _norm(x)]
        if not ft:
            continue
        for i in range(len(toks) - len(ft) + 1):
            if toks[i:i + len(ft)] == ft:
                a, z = words[i]["start"], words[i + len(ft) - 1]["end"]
                f0, f1 = max(0, int(a * fps)), min(n, int(np.ceil(z * fps)))
                if f1 <= f0 or any(e["start_frame"] <= f0 < e["end_frame"] for e in out):
                    continue
                flags = [] if all(w["reliable"] for w in words[i:i + len(ft)]) else ["transcript uncertain here"]
                said = " ".join(w["word"] for w in words[i:i + len(ft)])
                out.append({"start_frame": f0, "end_frame": f1, "kind": "spoken_brand",
                            "what": f'"{said}" said aloud',
                            "precision_frames": int(round(0.2 * fps)), "flags": flags,
                            "human_checked": False})
    return sorted(out, key=lambda e: e["start_frame"])


def scene_audio(audio, a_s, z_s):
    """Audio controls for one scene (for the EEG analysis and the copilot)."""
    if not audio or not audio.get("has_audio"):
        return None
    L = audio["loudness"]
    db, t0, hop = np.array(L["db"]), L["t0"], L["hop_s"]
    i0, i1 = int(max(0, (a_s - t0) / hop)), int(max(0, (z_s - t0) / hop))
    seg = db[i0:max(i1, i0 + 1)]
    words = [w for w in ((audio.get("transcript") or {}).get("words") or []) if a_s <= w["start"] < z_s]
    return {"loudness_db_mean": round(float(np.mean(seg)), 1), "loudness_db_max": round(float(np.max(seg)), 1),
            "loud_jumps": [e["t"] for e in audio["events"] if e["kind"] == "jump" and a_s <= e["t"] < z_s],
            "n_words": len(words), "text": " ".join(w["word"] for w in words)}


# ----------------------------------------------------------------------------- assemble
def assemble(story, snapped, brand, cands, fps, n, audio=None):
    starts = [0] + [b["frame"] for b in sorted(snapped, key=lambda b: b["k"])]
    methods = ["start"] + [b["method"] for b in sorted(snapped, key=lambda b: b["k"])]
    bflags = [[]] + [b["flags"] for b in sorted(snapped, key=lambda b: b["k"])]
    scenes, problems = [], []
    for k, ds in enumerate(story["scenes"]):
        a = starts[k]
        z = starts[k + 1] if k + 1 < len(starts) else n
        flags = list(bflags[k])
        if z <= a:
            problems.append(f"scene {k} '{ds['label']}' has no frames after snapping (boundaries crossed)")
            flags.append("EMPTY after snapping -- fix by hand")
        dur = (z - a) / fps
        if 0 < dur < MIN_SCENE_S and ds["story_role"] != "brand":
            flags.append(f"short ({dur:.2f} s < {MIN_SCENE_S:.0f} s): keep only if it is a real story beat")
        n_cuts = sum(1 for c in cands if a < c["frame"] < z and c["kind"] == "cut")
        scenes.append({
            "idx": k, "start_frame": a, "end_frame": z,               # end exclusive
            "start_s": round(a / fps, 3), "end_s": round(z / fps, 3), "dur_s": round(dur, 3),
            "label": ds["label"], "description": ds["description"], "audio": ds.get("audio", ""),
            "story_role": ds["story_role"], "audio_measures": scene_audio(audio, a / fps, z / fps),
            "boundary_method": methods[k], "n_cut_candidates_inside": n_cuts,
            "flags": flags, "human_checked": False})
    for b in brand:
        b["start_s"], b["end_s"] = round(b["start_frame"] / fps, 3), round(b["end_frame"] / fps, 3)
    return scenes, problems


def review_sheet(video, doc, out_path):
    from PIL import Image, ImageDraw
    tw, th = 200, int(200 * video.frame(0).shape[0] / video.frame(0).shape[1])
    rows = [("S", s) for s in doc["scenes"]] + [("B", b) for b in doc["brand_events"]]
    sheet = Image.new("RGB", (tw * 3 + 420, (th + 6) * len(rows)), "white")
    d = ImageDraw.Draw(sheet)
    for r, (kind, it) in enumerate(rows):
        y = r * (th + 6)
        a, z = it["start_frame"], it["end_frame"] - 1
        for j, f in enumerate((a, (a + z) // 2, z)):
            fr = video.frame(f, tw)
            if fr is not None:
                sheet.paste(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), (j * tw, y))
        x = tw * 3 + 8
        if kind == "S":
            d.text((x, y + 4), f"SCENE {it['idx']}  [{it['story_role']}]  {it['start_s']:.2f}-{it['end_s']:.2f}s "
                               f"(f{it['start_frame']}-{it['end_frame']})", fill="black")
            d.text((x, y + 22), it["label"][:60], fill="black")
            d.text((x, y + 40), f"boundary: {it['boundary_method']}", fill="black")
        else:
            d.text((x, y + 4), f"BRAND {it['kind']}  {it['start_s']:.2f}-{it['end_s']:.2f}s", fill="darkred")
            d.text((x, y + 22), it["what"][:60], fill="darkred")
        for q, fl in enumerate(it["flags"][:2]):
            d.text((x, y + 58 + 16 * q), "! " + fl[:62], fill="red")
    sheet.save(out_path, quality=80)


# ----------------------------------------------------------------------------- main
def label(ad, model_name, dry_run=False, overwrite=False, reuse_story=False, redo_audio=False, language="en"):
    path = os.path.join(ADS_DIR, ad + ".mp4")
    out_json = os.path.join(OUT_DIR, ad + (".scenes.dryrun.json" if dry_run else ".scenes.json"))
    if os.path.exists(out_json) and not overwrite and not dry_run:
        prev = json.load(open(out_json))
        if prev.get("status") == "reviewed":
            sys.exit(f"{out_json} is human-reviewed (frozen). Use --overwrite to relabel anyway.")
        sys.exit(f"{out_json} exists. Use --overwrite to relabel.")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{ad}: stage 2 (cut detector)...")
    det, sig = SD.detect(path, return_signals=True)
    with open(os.path.join(OUT_DIR, ad + ".cuts.json"), "w") as fh:
        json.dump(det, fh, indent=1)
    cands, fps, n = det["candidates"], det["fps"], det["n_frames"]
    duration = n / fps
    video, log = Video(path), []
    model = Model(model_name, dry_run, log)

    ap_ = os.path.join(OUT_DIR, ad + ".audio.json")
    if os.path.exists(ap_) and not redo_audio:
        print(f"{ad}: stage 0 (audio) -- reusing {os.path.basename(ap_)}")
        audio = json.load(open(ap_, encoding="utf-8"))
    else:
        print(f"{ad}: stage 0 (audio: loudness" + ("" if dry_run else " + transcript") + ")...")
        audio = AA.analyse(ad, do_transcript=not dry_run, language=language)
        if not dry_run:
            with open(ap_, "w", encoding="utf-8") as fh:
                json.dump(audio, fh, indent=1, ensure_ascii=False)

    prev_story = None
    if reuse_story and os.path.exists(out_json):
        prev = json.load(open(out_json))
        prev_story = next((l["reply"] for l in prev.get("model_log", []) if l["stage"] == "story" and "reply" in l), None)
    if prev_story:
        print(f"{ad}: stage 1 (story pass) -- reusing the previous reply")
        story = prev_story
        log.append({"stage": "story", "reused": True, "reply": story})
    else:
        print(f"{ad}: stage 1 (story pass)...")
        story = stage1_story(model, video, duration, audio)
    ds = story["scenes"]
    if not ds:
        sys.exit("model returned no scenes")
    ds.sort(key=lambda s: s["start_s"])
    ds[0]["start_s"] = 0.0                              # enforce coverage of the whole ad
    print(f"{ad}: stage 3 (snap {len(ds) - 1} boundaries)...")
    snapped = stage3_snap(model, video, ds, cands, sig)
    print(f"{ad}: stage 4 (brand frames, {len(story['brand_events'])} events)...")
    brand = stage4_brand(model, video, story["brand_events"], duration)
    forms = list(story.get("brand_spoken_forms", [])) + [story.get("brand", "")]
    brand = sorted(brand + spoken_brand_events(audio, forms, fps, n), key=lambda e: e["start_frame"])
    scenes, problems = assemble(story, snapped, brand, cands, fps, n, audio)

    doc = {
        "video": ad + ".mp4", "fps": fps, "n_frames": n, "duration_s": round(duration, 3),
        "status": "dry_run" if dry_run else "draft",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "model": None if dry_run else model_name,
        "time_convention": "frames are 0-based; end_frame is exclusive; seconds = frame / fps",
        "rule": "boundaries are detector frames (method 'cut') or locally refined frames "
                "(method 'refined_no_cut', always flagged); never model-typed times",
        "story": {k: story.get(k, "") for k in ("brand", "product", "summary", "audio_summary", "brand_spoken_forms")},
        "audio_file": ad + ".audio.json",
        "scenes": scenes, "brand_events": brand, "problems": problems,
        "model_log": log,
    }
    with open(out_json, "w") as fh:
        json.dump(doc, fh, indent=1)
    sheet = out_json.replace(".json", "_review.jpg")
    review_sheet(video, doc, sheet)
    n_flag = sum(1 for s in scenes if s["flags"]) + sum(1 for b in brand if b["flags"])
    print(f"{ad}: {len(scenes)} scenes, {len(brand)} brand events, {n_flag} flagged for review, "
          f"{len(problems)} problems -> {out_json}\n   review image: {sheet}")
    return doc


def check(doc):
    """Structural problems that block approval (empty list = OK)."""
    probs = []
    s, n = doc["scenes"], doc["n_frames"]
    if not s or s[0]["start_frame"] != 0 or s[-1]["end_frame"] != n:
        probs.append("scenes must start at frame 0 and end at the last frame")
    for a, b in zip(s, s[1:]):
        if a["end_frame"] != b["start_frame"]:
            probs.append(f"gap/overlap between scene {a['idx']} and {b['idx']}")
    for x in s:
        if x["end_frame"] <= x["start_frame"]:
            probs.append(f"scene {x['idx']} is empty")
        if x["story_role"] not in STORY_ROLES:
            probs.append(f"scene {x['idx']} has unknown story_role {x['story_role']}")
        if not str(x.get("label", "")).strip():
            probs.append(f"scene {x['idx']} has no label")
    for k, b in enumerate(doc.get("brand_events", [])):
        if not (0 <= b["start_frame"] < b["end_frame"] <= n):
            probs.append(f"brand event {k} has invalid frames {b['start_frame']}-{b['end_frame']}")
        if b["kind"] not in BRAND_KINDS:
            probs.append(f"brand event {k} has unknown kind {b['kind']}")
    return probs


def normalise(doc):
    """Recompute every derived field (seconds, durations, idx) from the frames."""
    fps = doc["fps"]
    for k, x in enumerate(doc["scenes"]):
        x["idx"] = k
        x["start_s"], x["end_s"] = round(x["start_frame"] / fps, 3), round(x["end_frame"] / fps, 3)
        x["dur_s"] = round(x["end_s"] - x["start_s"], 3)
    for b in doc.get("brand_events", []):
        b["start_s"], b["end_s"] = round(b["start_frame"] / fps, 3), round(b["end_frame"] / fps, 3)
    return doc


def approve_doc(doc):
    probs = check(doc)
    if probs:
        return probs
    normalise(doc)
    for x in doc["scenes"]:
        x["human_checked"] = True
    for b in doc["brand_events"]:
        b["human_checked"] = True
    doc["status"], doc["reviewed"] = "reviewed", dt.datetime.now().isoformat(timespec="seconds")
    return []


def approve(ad):
    p = os.path.join(OUT_DIR, ad + ".scenes.json")
    doc = json.load(open(p))
    probs = approve_doc(doc)
    if probs:
        sys.exit("Not approved:\n  " + "\n  ".join(probs))
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"{ad}: approved and frozen ({len(doc['scenes'])} scenes).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ads", nargs="*", help="ad names, e.g. ad04 (file in data/lab_session/ads)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="no API call; fake model replies")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--approve", action="store_true", help="validate + freeze after human review")
    ap.add_argument("--reuse-story", action="store_true", help="keep the previous stage-1 reply, redo stages 2-4")
    ap.add_argument("--redo-audio", action="store_true", help="recompute loudness + transcript")
    ap.add_argument("--language", default="en", help="spoken language for the transcript ('' = auto)")
    a = ap.parse_args()
    ads = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(ADS_DIR, "*.mp4")))] \
        if a.all else a.ads
    if not ads:
        ap.error("give ad names or --all")
    for ad in ads:
        if a.approve:
            approve(ad)
        else:
            label(ad, a.model, a.dry_run, a.overwrite, a.reuse_story, a.redo_audio, a.language or None)


if __name__ == "__main__":
    main()
