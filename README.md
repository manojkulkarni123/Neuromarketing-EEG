# EEG Neuromarketing — Ad Engagement

Records EEG while participants watch video ads, then produces a
second-by-second engagement timeline (frontal alpha asymmetry + frontal theta)
and a 1–9 self-report rating per ad.

Part of the BMS College of Engineering CSE major project (guide: Prof. Amit
Pradhan; Siddhant Sawhney, Manoj Kulkarni, Pranav Easwar, Padma Deepak).

## Hardware / software
- Mitsar amplifier + SmartDRY 19-channel **dry-electrode** cap
- EEGStudio Acquisition, streaming over **LSL Outlet**
  (Settings → Plug-ins → *Data type to transmit* = **Raw data (in uV)**)
- Windows, Python 3.11

## Setup
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running a session
1. In EEGStudio: start acquisition, turn **LSL Outlet** on, check electrode contact.
2. Put ad videos in `data/lab_session/ads/` (simple names like `ad01.mp4`: spaces
   or brackets in filenames break PsychoPy's video playback).
3. Record (a new `--id` per participant; it refuses to overwrite an existing session):
   ```
   .venv\Scripts\python.exe lab_task_psychopy\ad_viewing_task.py --id p01        # all ads, shuffled per participant
   .venv\Scripts\python.exe lab_task_psychopy\ad_viewing_task.py --id test1 --one  # quick test, one ad
   ```
   Saves `data/lab_session/<id>_raw.fif` (EEG + AD_START/AD_END markers on one
   LSL clock) and `<id>_ratings.csv`.
4. Analyze:
   ```
   .venv\Scripts\python.exe pipeline\extract_neuromarketing_features.py data\lab_session\<id>_raw.fif
   ```
   Runs an electrode quality check (auto-flags dead channels), ICA blink
   removal, then FAA/theta per ad, and saves `<id>_raw_engagement_timeline.png`.

## Layout
| Path | Purpose |
|---|---|
| `lab_task_psychopy/ad_viewing_task.py` | PsychoPy ad playback + rating, records EEG |
| `lab_task_psychopy/lsl_recorder.py` | Records EEGStudio's LSL stream, aligns markers, saves `.fif` |
| `lab_task_psychopy/calibrate_channels.py`, `channel_map.json` | Recovers channel labels (the stream has none) |
| `lab_task_psychopy/blink_test.py`, `tap_test.py`, `lsl_stream_watch.py` | Setup/diagnostic tools |
| `pipeline/preprocess_mne.py` | Loading, quality check, filtering, ICA |
| `pipeline/extract_neuromarketing_features.py` | Sliding-window FAA + theta, plots, per-ad summaries |
| `lab_task/` | Older manual-marker fallback tasks (pygame) |

## Known limitations
- **No channel labels in the LSL stream**: order recovered by calibration
  against EEGStudio's own EDF export; rerun `calibrate_channels.py` if the
  EEGStudio data mode changes.
- **F3 and F8 are dead on this cap**, so FAA uses **Fp1/Fp2** (frontal-pole
  asymmetry), which is more exposed to eye artifacts. Disclose this in any write-up.
- The delivered sample rate differs from the nominal 250 Hz; the recorder uses
  the measured rate and per-sample LSL timestamps.
- EEGStudio's LSL **Inlet** never worked, which is why we record its Outlet instead.

## Not in this repo
Participant EEG recordings, ad video files (copyright unknown;
`data/lab_session/ads/ad_name_map.csv` maps names to originals), and the
separate dataset-reconstruction work.
