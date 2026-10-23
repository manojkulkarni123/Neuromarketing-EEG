# EEG Neuromarketing — Ad Engagement

BMS College of Engineering, CSE major project. Guide: Prof. Amit Pradhan.
Team: Siddhant Sawhney, Manoj Kulkarni, Pranav Easwar, Padma Deepak.

## What we did, in one paragraph

We record a person's brain activity (EEG) while they watch real TV ads, and
turn it into a second-by-second picture of how engaged they were. After each
ad they also tell us how much they liked it and how likely they are to buy
the product. The research question is whether the brain signals track those
answers, and where in the ad engagement rises or falls, which a questionnaire
alone cannot show. Everything runs on a low-cost 19-channel dry-electrode cap
and our own Python software.

The stimuli are **five existing real TV ads** (32–90 s). We did not use any
AI-generated ads, A/B ad versions, or end-of-session memory/choice tests.

## How we got here

1. **Review 1 — decoding decisions.** We rebuilt a published EEG study
   (OpenNeuro ds002739, 24 people choosing the direction of moving dots).
   Brain activity just before the response predicted the upcoming choice
   (AUC 0.563 in our rebuild vs 0.558 in the paper draft). A deep network
   (EEGNet) did *worse* than a simple linear decoder (0.544 vs 0.563,
   p = 0.019): with ~150 trials per person, extra model capacity overfits.
2. **Lesson carried forward.** EEG does contain decision information, but
   with small lab datasets simple, well-understood measures beat complex models.
3. **Review 2 — a real-world decision with our own data.** Instead of dots,
   real ads; instead of a public dataset, our own recordings; instead of a
   black-box decoder, four established EEG measures.

## The core concepts

### 1. What the brain signals mean
EEG measures tiny voltage rhythms on the scalp. Different rhythms (frequency
bands) go up or down with different mental states. We track four:

| Measure | Where on the head | Idea | Higher means |
|---|---|---|---|
| **Frontal alpha asymmetry** | Fp1 vs Fp2 (front, left vs right) | Alpha is an "idling" rhythm, so *less* alpha = *more* activity. Relatively more left-frontal activity is linked to approach / wanting (Davidson). | Wanting / approach |
| **Engagement index** | Cz, Fz (top / front centre) | Beta (13–25 Hz) rises with focus, alpha with idling, so beta ÷ alpha tracks how switched-on someone is. | More alert, engaged |
| **Frontal theta** | Fz | Theta (4–8 Hz) at the front midline rises with mental effort and memory encoding. | More processing |
| **Posterior alpha suppression** | Back of head (O1/O2/Pz, else P3/P4) | When the eyes take in something interesting, back-of-head alpha drops (the reverse of closing your eyes). | More visual attention |

Asymmetry and engagement are different things: asymmetry is *direction*
(want vs avoid), engagement is *intensity* (how switched-on). A disturbing ad
can be highly engaging and still produce avoidance.

### 2. Why we compare against a baseline
Raw EEG power differs hugely between people (skull, hair, electrode contact).
So we never interpret a raw number. Each session starts with 60 s of resting
with eyes open, and every measure is reported as a change from that person's
own rest. Theta is given in decibels: 10 × log10(ad ÷ rest), so +3 dB ≈ double,
−3 dB ≈ half. A ratio makes different people comparable.

A second 60 s block with eyes **closed** checks the cap: closing the eyes
should raise back-of-head alpha by at least 1.3×. If it doesn't, those
electrodes aren't really touching the scalp.

### 3. Why the session has buffers around each ad
Each ad is shown as: **fixation cross (4 s) → ad → gray screen (3 s) → two ratings.**
- The **cross** gives the head time to settle after the last key press. In our
  first test the first ~5 s of the ad were ruined by that movement.
- The **gray screen** catches the brain's delayed reaction to the ending and logo.
- Buffers are padding only. The numbers for an ad come only from while it plays.

We decided against a "neutral video" baseline because no clip is truly
neutral. The difference between resting on a cross and watching any video is
the same for every ad, so it cancels when ads are compared with each other.

### 4. How brain data and ad timing line up
EEGStudio streams the EEG over Lab Streaming Layer (LSL); our task stamps
"ad started / ad ended" markers on the **same LSL clock**. Both live in one
file, so "the ad started at 191.3 s" and "this EEG sample is at 191.3 s" mean
the same moment, with no manual alignment needed. Measured accuracy is ~0.1 s.
We also use the amplifier's *measured* rate (~252.6 Hz), not the nominal
250 Hz, which would drift markers by ~6%.

### 5. How a graph point is computed
The brain signal is cut into **3-second windows, moved 1 s at a time**. Each
window's frequency content gives one point on each graph, so a point at 14 s
summarises about 12.5–15.5 s. Shorter windows were too noisy (±0.8 dB at 1 s
vs ±0.35 dB at 3 s); published work uses 2–4 s. One number per ad is the
**median** of the windows inside the ad, so a short artefact (a cough, a jaw
clench) barely moves it (<1% for a 2 s burst).

### 6. Cleaning the signal
- **Dead electrodes are detected automatically.** An electrode is rejected if
  it copies an unconnected input (it's picking up room noise) or its signal is
  implausibly large (>150 µV). On our cap, F3 and F8 are permanently dead,
  which is why asymmetry uses Fp1/Fp2.
- **Blinks are removed with ICA**, which splits the signal into independent
  sources and drops the one that looks like a blink. We match on the slow
  (<4 Hz) band only, because blinks are slow and alpha is 8–13 Hz. That way
  we remove blinks without removing the alpha we want to measure at Fp1/Fp2.

### 7. How we know the software is right
Before trusting real data, we fed the pipeline synthetic EEG with known
effects planted in it:

| Planted | Pipeline found |
|---|---|
| Blink of 127 µV | Reduced to 21 µV, asymmetry left intact |
| 50% alpha drop | 49.4% |
| +3.0 dB theta | +2.87 dB |
| Beta doubled (+100%) | +92% engagement |

The same tests exposed a real limit: with ~30 s of data, noise alone moves
asymmetry by about ±0.18. Real effects are 0.1–0.3. So asymmetry is only
meaningful averaged over a group, never for one person and one ad.

### 8. What the first real recording showed
Test run `test1` (19-10-2026, one participant, ad04, Budweiser):
10 of 19 electrodes usable; cap check passed (1.35×). During the ad, compared
with rest: posterior alpha suppression 45%, engagement +14.5%, theta −1.1 dB,
asymmetry +0.06 (inside the noise range). Ratings: liking 8, purchase intent 4,
a reminder that liking an ad and wanting the product are different, which is
why we ask both. One person proves the system works, not a scientific claim.

### 9. What comes next
A full five-ad session, better contact on the back electrodes, consent forms,
about 20 participants, then group statistics: do the four measures rise and
fall with liking and purchase intent across people? Later: an
opening-seconds "hook" score and inter-subject correlation (how similarly
different brains respond to the same ad).

## Using the code

**Hardware:** Mitsar amplifier + SmartDRY 19-channel dry cap; EEGStudio with
LSL Outlet on and *Settings → Plug-ins → Data type to transmit* = **Raw data (in uV)**.

**Setup** (Windows, Python 3.11):
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Run a session** (ads go in `data/lab_session/ads/` with simple names like `ad01.mp4`):
```
.venv\Scripts\python.exe lab_task_psychopy\ad_viewing_task.py --id p01          # baselines + all ads
.venv\Scripts\python.exe lab_task_psychopy\ad_viewing_task.py --id test1 --one  # quick test, one ad
```
Options: `--no-baseline`, `--overwrite`. Saves `data/lab_session/<id>_raw.fif`
(EEG + markers) and `<id>_ratings.csv`, even if stopped with ESC.

**Analyse:**
```
.venv\Scripts\python.exe pipeline\extract_neuromarketing_features.py data\lab_session\<id>_raw.fif
.venv\Scripts\python.exe pipeline\build_replay.py data\lab_session\<id>_raw.fif
```
The first prints the quality check and per-ad numbers and saves the timeline
plot. The second writes `<id>_raw_replay.html`: the ad plays next to the four
graphs with a moving cursor, and clicking a graph seeks the video.

| Path | Purpose |
|---|---|
| `lab_task_psychopy/ad_viewing_task.py` | The session: baselines, ads, buffers, ratings, markers |
| `lab_task_psychopy/lsl_recorder.py` | Records the LSL stream, applies channel map, saves `.fif` |
| `lab_task_psychopy/calibrate_channels.py`, `channel_map.json` | Recovers channel labels (the stream has none) |
| `lab_task_psychopy/blink_test.py`, `tap_test.py`, `lsl_stream_watch.py` | Lab diagnostics |
| `pipeline/preprocess_mne.py` | Loading, quality check, filtering, ICA |
| `pipeline/extract_neuromarketing_features.py` | Four measures, baseline comparison, plots |
| `pipeline/build_replay.py` | Offline video + graphs replay page |
| `lab_task/` | Older manual-marker fallback tasks |
| `Review2_Neuromarketing_EEG.pptx`, `NEUROMARKETING_PROJECT_GUIDE.md` | Review 2 slides, team guide |

## Known limitations
- The LSL stream has no channel labels; order was recovered by calibrating
  against EEGStudio's labelled EDF export (rerun if the data mode changes).
- F3 and F8 are dead, so asymmetry is frontal-pole (Fp1/Fp2); disclose this.
- Back-of-head contact is weak, so posterior alpha may rely on P3 alone.
- EEGStudio's LSL Inlet never worked; we record its Outlet instead.
- Real ads differ in length (32–90 s) and some may be familiar to participants.

## Not in this repo
Participant EEG recordings, ad videos (copyright; `ads/ad_name_map.csv` maps
names to originals), and the Review 1 dataset-reconstruction code.
