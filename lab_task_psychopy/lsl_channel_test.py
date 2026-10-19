"""
Markers as a continuous extra channel -- fallback for when EEGStudio's LSL
Inlet rejects irregular-rate "Markers" streams (tried text and int32 marker
streams; both failed to connect). Sends a regular-rate 1-channel float32
stream that reads 0 continuously and jumps to a code for a short pulse at
each event, so the markers get recorded as an extra channel alongside the EEG.

Usage (from the venv):
    .venv\\Scripts\\python.exe lab_task_psychopy\\lsl_channel_test.py
    .venv\\Scripts\\python.exe lab_task_psychopy\\lsl_channel_test.py --type Markers
"""
import sys
import time

from pylsl import StreamInfo, StreamOutlet, local_clock

STREAM_NAME = "PsychoPyMarkers"
SRATE = 500
CHUNK_S = 0.02
PULSE_SAMPLES = 25  # 50ms at 500Hz -- long enough to be unmissable in the trace
EVENT_EVERY_S = 2.0


def main():
    stream_type = sys.argv[sys.argv.index("--type") + 1] if "--type" in sys.argv else "EEG"
    info = StreamInfo(name=STREAM_NAME, type=stream_type, channel_count=1,
                       nominal_srate=SRATE, channel_format="float32", source_id="lab_task_channel")
    info.desc().append_child("channels").append_child("channel").append_child_value("label", "MARKER")
    outlet = StreamOutlet(info)
    print(f"LSL outlet '{STREAM_NAME}' (type='{stream_type}', {SRATE}Hz, 1 float32 channel 'MARKER') is live.")
    print(f"Channel reads 0, pulses to 1,2,3... every {EVENT_EVERY_S}s. Ctrl+C to stop.\n")

    chunk_n = int(SRATE * CHUNK_S)
    pulse_left, code, next_event = 0, 0, local_clock() + EVENT_EVERY_S
    t_next = local_clock()
    try:
        while True:
            now = local_clock()
            if now >= next_event:
                code = code % 250 + 1
                pulse_left = PULSE_SAMPLES
                next_event += EVENT_EVERY_S
                print(f"Pulse: {code}  (check EEGStudio now)")
            chunk = []
            for _ in range(chunk_n):
                chunk.append([float(code) if pulse_left > 0 else 0.0])
                pulse_left = max(0, pulse_left - 1)
            outlet.push_chunk(chunk)
            t_next += CHUNK_S
            time.sleep(max(0, t_next - local_clock()))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
