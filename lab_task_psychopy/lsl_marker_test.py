"""
Minimal LSL marker broadcaster -- purely for testing the connection to
EEGStudio's "LSL Inlet" live, before any real task exists. Run this, then
watch EEGStudio to confirm it sees the stream and the markers land in its
recording. Ctrl+C to stop.

Usage (from the venv):
    .venv\\Scripts\\python.exe lab_task_psychopy\\lsl_marker_test.py          (text markers)
    .venv\\Scripts\\python.exe lab_task_psychopy\\lsl_marker_test.py --int    (numeric markers)

Use --int if EEGStudio's Inlet refuses text markers -- many acquisition
programs only accept numeric (int32) marker codes.
"""
import sys
import time

from pylsl import StreamInfo, StreamOutlet

STREAM_NAME = "PsychoPyMarkers"
STREAM_TYPE = "Markers"


def main():
    numeric = "--int" in sys.argv
    info = StreamInfo(name=STREAM_NAME, type=STREAM_TYPE, channel_count=1, nominal_srate=0,
                       channel_format="int32" if numeric else "string", source_id="lab_task_test")
    outlet = StreamOutlet(info)
    print(f"LSL outlet '{STREAM_NAME}' (type='{STREAM_TYPE}', "
          f"format={'int32' if numeric else 'string'}) is live.")
    print("In EEGStudio: open LSL Inlet, it should now be able to discover this stream by name.")
    print("Sending a test marker every 2 seconds. Press Ctrl+C to stop.\n")

    i = 0
    try:
        while True:
            label = (i % 250) + 1 if numeric else f"test_marker_{i}"
            outlet.push_sample([label])
            print(f"Sent: {label}  (check EEGStudio now)")
            i += 1
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
