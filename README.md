# EEG Neuromarketing — Ad Engagement

Records EEG while participants watch real video ads, then produces a
second-by-second timeline of four EEG measures, per-ad numbers relative to the
person's own resting baseline, and an offline replay page that plays the ad in
sync with the graphs. After every ad the participant gives two 1–9 ratings
(liking, purchase intent), which the EEG measures are compared against.

Part of the BMS College of Engineering CSE major project (guide: Prof. Amit
Pradhan; Siddhant Sawhney, Manoj Kulkarni, Pranav Easwar, Padma Deepak). It
continues Review 1 (EEG decision decoding on the public OpenNeuro ds002739
dataset) by applying EEG decision research to a real-world decision with our
own recordings. Review 2 slides: `Review2_Neuromarketing_EEG.pptx`; team guide:
`NEUROMARKETING_PROJECT_GUIDE.md`.

**Status (Oct 2026):** the full chain (record → clean → measures → plot →
replay) has run on the real cap for one participant (`test1`, ad04). Every
analysis step was validated on synthetic data. Group data collection is next.

## Hardware / software
- Mitsar amplifier + SmartDRY 19-channel **dry-electrode** cap
- EEGStudio Acquisition, streaming over **LSL Outlet**
  (Settings → Plug-ins → *Data type to transmit* = **Raw data (in uV)**;
  27 channels at a measured ~252.6 Hz)
- Windows, Python 3.11 (PsychoPy, pylsl, MNE, NumPy/SciPy, Matplotlib)

## Setup
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Session protocol
| Step | Duration | Marker |
|---|---|---|
| Eyes-open baseline (fixation cross) | 60 s | `BASELINE_EO_START/END` |
| Eyes-closed baseline (beep to open) | 60 s | `BASELINE_EC_START/END` |
| Per ad, random order per participant: fixation cross | 4 s | `PRE_AD_START:<ad>` |
| Ad | 32–90 s | `AD_START:<ad>` / `AD_END:<ad>` |
| Mid-gray screen with cross | 3 s | `POST_AD_END:<ad>` |
| Ratings: "How much did you like this ad?", "How likely would you be to buy this product?" (1–9) | self-paced | — |

The eyes-open block is the reference for every measure; eyes-closed checks cap
contact (posterior alpha should rise ≥1.3×). The fixation cross hides the
movement artefact from the previous key press; the gray screen catches the
delayed reaction to the ending/logo. Per-ad numbers use only AD_START..AD_END.

## Measures
| Measure | Electrodes | Computed as | Higher = | Reported vs baseline |
|---|---|---|---|---|
| Frontal alpha asymmetry | Fp1 / Fp2 | ln(α right) − ln(α left), α 8–13 Hz | approach / wanting | `faa_change` |
| Engagement index | Cz, Fz | log10(β/α), β 13–25 Hz | more alert / engaged | `engagement_change_pct` |
| Frontal theta | Fz | log10 power, 4–8 Hz | more mental processing | `theta_change_db` |
| Posterior alpha suppression | O1/O2/Pz (fallback P3/P4) | drop in 8–13 Hz power | more visual attention | `alpha_suppression_pct` |

3 s windows in 1 s steps (Welch, 1 s segments, 50% overlap); per-ad value =
median over windows inside the ad.

## Running a session
1. In EEGStudio: start acquisition, turn **LSL Outlet** on, check electrode contact.
2. Put ad videos in `data/lab_session/ads/` (simple names like `ad01.mp4`: spaces
   or brackets in filenames break PsychoPy's video playback).
3. Record (a new `--id` per participant; it refuses to overwrite an existing
   session unless `--overwrite` is given):
   ```
   .venv\Scripts\python.exe lab_task_psychopy\ad_viewing_task.py --id p01          # baselines + all ads
   .venv\Scripts\python.exe lab_task_psychopy\ad_viewing_task.py --id test1 --one  # quick test, one ad
   ```
   `--no-baseline` skips the baseline block. Saves
   `data/lab_session/<id>_raw.fif` (EEG + markers on one LSL clock) and
   `<id>_ratings.csv` (order, ad, duration_s, liking, liking_rt_s,
   purchase_intent, purchase_intent_rt_s), even if the session is stopped with ESC.
4. Analyze:
   ```
   .venv\Scripts\python.exe pipeline\extract_neuromarketing_features.py data\lab_session\<id>_raw.fif
   ```
   Electrode quality check (flags dead channels: |r| > 0.8 with an unconnected
   input, or > 150 µV), 1–40 Hz filter, ICA blink removal, the four measures,
   the cap check, per-ad numbers vs baseline, and
   `<id>_raw_engagement_timeline.png`.
5. Replay viewer:
   ```
   .venv\Scripts\python.exe pipeline\build_replay.py data\lab_session\<id>_raw.fif
   ```
   Writes `data/lab_session/<id>_raw_replay.html` (offline; keep it in that
   folder so it finds `ads/`). Pick an ad, play it next to the four graphs,
   click a graph to seek.

## Layout
| Path | Purpose |
|---|---|
| `lab_task_psychopy/ad_viewing_task.py` | PsychoPy session: baselines, ads, buffers, ratings, markers |
| `lab_task_psychopy/lsl_recorder.py` | Records EEGStudio's LSL stream, applies channel map, saves `.fif` |
| `lab_task_psychopy/calibrate_channels.py`, `channel_map.json` | Recovers channel labels (the stream has none) |
| `lab_task_psychopy/blink_test.py`, `tap_test.py`, `lsl_stream_watch.py` | Setup/diagnostic tools |
| `pipeline/preprocess_mne.py` | Loading, quality check, filtering, ICA |
| `pipeline/extract_neuromarketing_features.py` | Four measures, baseline comparison, plots, per-ad summaries |
| `pipeline/build_replay.py` | Offline HTML replay: ad video + four graphs in sync |
| `data/lab_session/test1_*` | First full test run: ratings + timeline plot |
| `lab_task/` | Older manual-marker fallback tasks (pygame) |

## Results so far (test1, ad04, one participant)
- Synthetic checks: blink 127 → 21 µV; alpha suppression 50% → 49.4%;
  theta +3 dB → +2.87 dB; β doubled → +92% engagement.
- Markers within ~0.1 s (ad marked 59.8 s vs 60.1 s video).
- 10 of 19 electrodes usable; cap check 1.35× on P3 (OK).
- ad04 vs baseline: alpha suppression 45.4%, engagement +14.5%, theta −1.1 dB,
  asymmetry +0.06 (inside the noise floor); ratings liking 8, purchase intent 4.

## Known limitations
- **No channel labels in the LSL stream**: order recovered by calibration
  against EEGStudio's own EDF export; rerun `calibrate_channels.py` if the
  EEGStudio data mode changes.
- **F3 and F8 are dead on this cap**, so asymmetry uses **Fp1/Fp2**
  (frontal-pole asymmetry). Disclose this in any write-up.
- **Posterior contact is weak** (O1/O2/Pz often rejected), so posterior alpha
  may fall back to P3 alone.
- **Asymmetry noise floor**: on ~30 s of data, noise alone gives changes of
  SD ≈ 0.18, so asymmetry is only meaningful as a group average.
- The delivered sample rate differs from the nominal 250 Hz; the recorder uses
  the measured rate and per-sample LSL timestamps.
- EEGStudio's LSL **Inlet** never worked, which is why we record its Outlet instead.

## Next steps
Full 5-ad session (test2), fix posterior contact, consent forms, ~20
participants, group statistics (do the measures track liking / purchase
intent?), opening-seconds "hook" score, inter-subject correlation.

## Not in this repo
Participant EEG recordings, ad video files (copyright unknown;
`data/lab_session/ads/ad_name_map.csv` maps names to originals), and the
separate dataset-reconstruction work.
