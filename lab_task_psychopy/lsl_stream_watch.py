"""
Scans for LSL streams every 2s and prints anything it finds, with full
metadata. Used to check whether EEGStudio's "LSL Outlet" is actually
broadcasting, and what it looks like (name, type, channel count, rate).

Usage:
    .venv\\Scripts\\python.exe lab_task_psychopy\\lsl_stream_watch.py [seconds]
"""
import sys
import time

from pylsl import resolve_streams, StreamInlet

DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 300


def main():
    seen = set()
    t0 = time.time()
    print(f"Watching for LSL streams for {DURATION_S:.0f}s...", flush=True)
    while time.time() - t0 < DURATION_S:
        for s in resolve_streams(wait_time=2.0):
            key = (s.name(), s.source_id(), s.hostname())
            if key in seen:
                continue
            seen.add(key)
            print(f"\nFOUND: name={s.name()!r} type={s.type()!r} channels={s.channel_count()} "
                  f"rate={s.nominal_srate()}Hz format={s.channel_format()} host={s.hostname()!r} "
                  f"source_id={s.source_id()!r}", flush=True)
            try:
                inlet = StreamInlet(s)
                full = inlet.info(timeout=5)
                ch = full.desc().child("channels").child("channel")
                labels = []
                while not ch.empty():
                    labels.append(ch.child_value("label"))
                    ch = ch.next_sibling()
                print(f"  channel labels: {labels if labels else '(none in metadata)'}", flush=True)
                chunk, ts = inlet.pull_chunk(timeout=2.0)
                print(f"  data flowing: {len(chunk)} samples pulled in 2s", flush=True)
                if chunk:
                    print(f"  first sample: {[round(v, 2) for v in chunk[0][:6]]}...", flush=True)
            except Exception as e:
                print(f"  could not inspect: {e}", flush=True)
    print(f"\nDone. Total distinct streams seen: {len(seen)}", flush=True)


if __name__ == "__main__":
    main()
