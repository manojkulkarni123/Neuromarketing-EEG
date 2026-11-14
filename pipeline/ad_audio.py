"""
Audio track of an ad: loudness curve + speech transcript with word timings
(labeller stage 0). Feeds the story pass (the model sees what is SAID next to
what is SHOWN) and gives the EEG analysis an audio control (a sudden loud
sound changes the EEG regardless of what is on screen, like a bright flash).

Timing: everything is in VIDEO time (frame k starts at k / fps, frame 0 =
0 s). MP4 audio and video tracks can start at different timestamps -- in our
ads the first video frame is stamped one frame (~40 ms) after the first audio
sample -- so audio times are shifted by (audio_start - video_start).

  loudness  -- computed here, no AI: RMS in 100 ms windows every 50 ms, in dB
               relative to digital full scale (dBFS). Exact timing.
  events    -- from the loudness curve, no AI:
               jump    : >= 12 dB above the median of the preceding second
               silence : >= 0.4 s below -50 dBFS
  transcript-- OpenAI whisper-1, verbose JSON with word timestamps (~0.1-0.3 s
               accuracy). Segments that look like non-speech (no_speech_prob
               > 0.6, avg_logprob < -1, or repetitive text) are kept in the
               file but marked unreliable -- song lyrics often land there.

Usage:
  python pipeline/ad_audio.py ad04 [--no-transcript]
Output: data/lab_session/scenes/<ad>.audio.json
"""
import argparse
import io
import json
import os
import sys
import wave

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ADS_DIR = os.path.join(PROJ, "data", "lab_session", "ads")
OUT_DIR = os.path.join(PROJ, "data", "lab_session", "scenes")
SR = 16000
HOP_S, WIN_S = 0.05, 0.10
JUMP_DB, JUMP_FLOOR_DB = 12.0, -35.0
SILENCE_DB, SILENCE_MIN_S = -50.0, 0.4


def decode(path):
    """Mono float32 audio at 16 kHz + the audio-to-video time offset (s)."""
    c = av.open(path)
    vs = c.streams.video[0]
    if not c.streams.audio:
        c.close()
        return None, 0.0
    a = c.streams.audio[0]
    t0 = lambda st: float(st.start_time * st.time_base) if st.start_time is not None else 0.0
    offset = t0(a) - t0(vs)               # audio sample 0 sits at this video time
    rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
    chunks = []
    for frame in c.decode(a):
        for f in rs.resample(frame):
            chunks.append(f.to_ndarray().reshape(-1))
    for f in rs.resample(None):
        chunks.append(f.to_ndarray().reshape(-1))
    c.close()
    return np.concatenate(chunks).astype(np.float32), offset


def loudness(x, offset):
    hop, win = int(HOP_S * SR), int(WIN_S * SR)
    n = max(1, 1 + (len(x) - win) // hop)
    db = np.empty(n)
    for i in range(n):
        seg = x[i * hop:i * hop + win]
        db[i] = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-6)
    t = offset + np.arange(n) * HOP_S + WIN_S / 2
    return t, db


def loud_events(t, db):
    ev = []
    back = int(round(1.0 / HOP_S))
    last = -1e9
    for i in range(back, len(db)):
        ref = np.median(db[i - back:i - 3])
        if db[i] >= JUMP_FLOOR_DB and db[i] - ref >= JUMP_DB and t[i] - last > 0.5:
            j = i + int(np.argmax(db[i:i + 4]))      # peak of the onset
            ev.append({"kind": "jump", "t": round(float(t[i]), 3), "db": round(float(db[j]), 1),
                       "rise_db": round(float(db[i] - ref), 1)})
            last = t[i]
    quiet = db < SILENCE_DB
    i = 0
    while i < len(db):
        if quiet[i]:
            j = i
            while j < len(db) and quiet[j]:
                j += 1
            if (j - i) * HOP_S >= SILENCE_MIN_S:
                ev.append({"kind": "silence", "start": round(float(t[i] - WIN_S / 2), 3),
                           "end": round(float(t[j - 1] + WIN_S / 2), 3)})
            i = j
        else:
            i += 1
    return sorted(ev, key=lambda e: e.get("t", e.get("start")))


def _wav_bytes(x):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    buf.name = "audio.wav"
    return buf


def load_key():
    """True if OPENAI_API_KEY is available. setx only reaches NEW processes, so fall
    back to the user's saved Windows variable (registry) -- never read from a file."""
    if os.environ.get("OPENAI_API_KEY"):
        return True
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                os.environ["OPENAI_API_KEY"] = winreg.QueryValueEx(k, "OPENAI_API_KEY")[0]
                return True
        except OSError:
            pass
    return False


def transcribe(x, offset, language="en"):
    from openai import OpenAI
    if not load_key():
        sys.exit("OPENAI_API_KEY is not set.")
    kw = {"language": language} if language else {}
    r = OpenAI().audio.transcriptions.create(
        model="whisper-1", file=_wav_bytes(x), response_format="verbose_json",
        timestamp_granularities=["word", "segment"], temperature=0, **kw)
    segs, seen = [], set()
    for s in r.segments or []:
        text = s.text.strip()
        if not text or s.end - s.start < 0.05:
            continue                                   # empty / zero-length: decoder noise
        unreliable = []
        # Whisper's own rule: silence only if BOTH likely-non-speech AND low confidence
        # (speech over a music bed often has a high no_speech_prob but is transcribed fine)
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
            unreliable.append("probably not speech")
        elif s.avg_logprob < -1.2:
            unreliable.append("low confidence")
        if s.compression_ratio > 2.4:
            unreliable.append("repetitive text (typical of hallucination)")
        repeat = text.lower() in seen          # a chorus repeats legitimately -- tag, don't reject
        seen.add(text.lower())
        segs.append({"start": round(s.start + offset, 3), "end": round(s.end + offset, 3),
                     "text": text, "reliable": not unreliable, "kind": "lyrics" if "♪" in text else "speech",
                     "repeat_of_earlier_line": repeat, "why_unreliable": unreliable,
                     "no_speech_prob": round(s.no_speech_prob, 3), "avg_logprob": round(s.avg_logprob, 3)})
    words = []
    for w in r.words or []:
        ws, we = w.start + offset, w.end + offset
        seg = next((s for s in segs if s["start"] - 0.05 <= ws <= s["end"] + 0.05), None)
        words.append({"word": w.word.strip(), "start": round(ws, 3), "end": round(we, 3),
                      "reliable": bool(seg and seg["reliable"])})
    return {"model": "whisper-1", "language": r.language, "text": r.text.strip(),
            "segments": segs, "words": words}


def analyse(ad, do_transcript=True, language="en"):
    path = os.path.join(ADS_DIR, ad + ".mp4")
    x, offset = decode(path)
    out = {"video": ad + ".mp4", "time_convention": "seconds in video time (frame k = k / fps)",
           "audio_offset_s": round(offset, 4)}
    if x is None:
        out.update({"has_audio": False, "loudness": None, "events": [], "transcript": None})
        return out
    t, db = loudness(x, offset)
    out.update({
        "has_audio": True,
        "loudness": {"hop_s": HOP_S, "window_s": WIN_S, "t0": round(float(t[0]), 4),
                     "db": [round(float(v), 1) for v in db],
                     "note": "dBFS: 0 = digital maximum; typical speech/music -30..-10"},
        "events": loud_events(t, db),
        "transcript": transcribe(x, offset, language) if do_transcript else None,
    })
    return out


def per_second_summary(audio, duration):
    """For the story prompt: {second: (loudness dB, words spoken in that second)}."""
    out = {}
    if not audio or not audio.get("has_audio"):
        return out
    L = audio["loudness"]
    db, t0, hop = np.array(L["db"]), L["t0"], L["hop_s"]
    tr = audio.get("transcript") or {}
    for s in range(int(np.ceil(duration))):
        i0, i1 = int(max(0, (s - t0) / hop)), int(max(0, (s + 1 - t0) / hop))
        lvl = float(np.mean(db[i0:i1])) if i1 > i0 else float("nan")
        words = [w["word"] for w in tr.get("words", []) if s <= w["start"] < s + 1]
        unrel = any(not w["reliable"] for w in tr.get("words", []) if s <= w["start"] < s + 1)
        out[s] = (lvl, " ".join(words), unrel)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ads", nargs="+")
    ap.add_argument("--no-transcript", action="store_true")
    ap.add_argument("--language", default="en", help="spoken language (ISO code, e.g. en, hi); '' = auto-detect")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ad in a.ads:
        res = analyse(ad, not a.no_transcript, a.language or None)
        p = os.path.join(OUT_DIR, ad + ".audio.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)
        tr = res.get("transcript") or {}
        print(f"{ad}: offset {res['audio_offset_s']*1000:.0f} ms, {len(res['events'])} loudness events, "
              f"{len(tr.get('words', []))} words ({sum(w['reliable'] for w in tr.get('words', []))} reliable) -> {p}")


if __name__ == "__main__":
    main()
