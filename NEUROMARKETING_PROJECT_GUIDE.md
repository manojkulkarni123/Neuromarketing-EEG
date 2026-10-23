# EEG Neuromarketing Project: Team Guide

*BMS College of Engineering, Dept. of CSE, Major Project*
*Guide: Prof. Amit Pradhan · Team: Siddhant Sawhney, Manoj Kulkarni, Pranav Easwar, Padma Deepak*
*Last updated: 23 Oct 2026*

This document covers the project as it actually stands: why we're doing it,
how the system works, what went wrong and how we fixed it, how to run a
session, how to read the results, and what's left. Read it before going to the lab.

> **Status legend:** ✅ built and tested · 🟡 built, not yet tested on real data · ⬜ planned, not built

---

## Contents
1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [How we got here](#2-how-we-got-here)
3. [Research question](#3-research-question)
4. [What others have done (literature)](#4-what-others-have-done-literature)
5. [Hardware and software](#5-hardware-and-software)
6. [How the system works](#6-how-the-system-works)
7. [Setup problems we hit, and how we solved them](#7-setup-problems-we-hit-and-how-we-solved-them)
8. [Experiment design](#8-experiment-design)
9. [What we measure from the brain, explained](#9-what-we-measure-from-the-brain-explained)
10. [Analysis plan](#10-analysis-plan)
11. [Running a session, step by step](#11-running-a-session-step-by-step)
12. [Reading the output](#12-reading-the-output)
13. [Results so far](#13-results-so-far)
14. [Limitations to state honestly](#14-limitations-to-state-honestly)
15. [What's built vs. what's left](#15-whats-built-vs-whats-left)
16. [Code map](#16-code-map)
17. [Glossary](#17-glossary)

---

## 1. The project in one paragraph

We show people **five existing, real TV ads** while recording their brain
activity with a **19-channel dry EEG cap**. After each ad they rate how much
they **liked** it and how likely they'd be to **buy** the product (1–9). The
software turns the EEG into a **second-by-second engagement timeline** of four
brain measures, plus one number per ad compared with the person's own resting
baseline. The question is whether those brain measures **track liking and
purchase intent**, and **where in the ad** engagement rises or drops, which a
questionnaire alone cannot show.

---

## 2. How we got here

**Phase 1: Reanalysis of a public dataset (done, Review 1).**
Our earlier work used a public EEG dataset (OpenNeuro ds002739, Gherman &
Philiastides 2018; 24 people doing a moving-dots left/right decision task). We
tested whether brain activity just before a response predicts the choice.
- We rebuilt the whole pipeline locally and reproduced the original numbers:
  choice AUC **0.563** vs. 0.558 reported, accuracy AUC **0.542** vs. 0.537.
- We tested a deep-learning model (**EEGNet**) against the simpler linear model
  (**LDA**). LDA won significantly for choice (p = 0.019). With only ~150 trials
  per person, the bigger model overfits. This is an honest negative result.

**Why we moved on:** reproducing someone else's analysis isn't a strong enough
contribution on its own. We have access to a real EEG cap, so we moved to
collecting our own data on an applied decision-making problem:
**neuromarketing**. Responding to an ad is a real-world decision, so this
stays connected to Phase 1.

**Ideas we considered and dropped:**
- **AI-generated ads with A/B versions.** Considered for a controlled
  manipulation, but the team decided to keep the scope to the **5 existing
  real ads**: no AI ads, no counterbalanced lists, no end-of-session recall,
  recognition or choice tests.
- **A neutral video as the baseline.** No clip is truly neutral, so we use
  eyes-open rest instead (§8).
- **Meta's TRIBE v2** (an AI model that predicts fMRI responses from video):
  predicts fMRI rather than EEG, needs a GPU; dropped to keep focus.
- **P300 / LPP responses:** need the same event repeated dozens of times; each
  ad is shown once, so they aren't measurable here.

---

## 3. Research question

> **Q1 (system):** Can a low-cost dry-electrode cap and our own pipeline
> produce a reliable second-by-second engagement timeline for each ad?
>
> **Q2 (science):** Do four EEG measures (frontal asymmetry, engagement index,
> frontal theta, posterior alpha suppression) track viewers' **liking** and
> **purchase-intent** ratings?

Q1 is answered for one participant (yes, with electrode-contact caveats). Q2
needs group data (~20 people × 5 ads) and is the main goal of the next phase.

**What the project contributes:**
- A complete, validated, open Python pipeline for a **consumer-grade dry cap**
  with broken electrodes, which is close to real-world use. Bad electrodes are
  detected automatically and the analysis adapts.
- A **replay viewer** that shows the ad and the brain measures side by side,
  so engagement changes can be matched to specific moments in the ad.
- A within-person test of whether simple EEG measures follow what people say
  about real ads.

---

## 4. What others have done (literature)

| Study | What they measured | Outcome | Result | What we take from it |
|---|---|---|---|---|
| **Hakim et al. 2021** (Int. J. Research in Marketing) | Band power at Fp1, Fp2, Fpz, F7/F8 asymmetry, inter-subject correlation | Product preference | EEG improved preference prediction over questionnaires alone | Uses **the same frontal-pole electrodes that work on our cap**. Main methods reference. |
| **Boksem & Smidts 2015** (J. Marketing Research) | Frontal-midline beta, gamma | Willingness to pay; box-office revenue | Brain measures beat stated willingness to pay | Beta is useful. **Skip gamma**: on a dry cap it's mostly muscle noise. |
| **Venkatraman et al. 2015** (J. Marketing Research) | Occipital alpha, frontal asymmetry (plus fMRI, eye tracking) | Real ad-driven sales | These EEG measures added nothing beyond questionnaires; only fMRI did | A known challenge; expect reviewers to cite it. |
| **Dmochowski et al. 2014** (Nature Communications) | Inter-subject correlation (ISC) of EEG | TV ratings, Super Bowl ad rankings | A small lab group's synchrony predicted what millions preferred | Planned **later** extension (needs group data). |
| **Davidson 1992**; **Allen et al. 2004** | Frontal alpha asymmetry | Emotion / motivation | Relatively more left-frontal activity = approach | Basis of our asymmetry measure and sign convention. |
| **Pope et al. 1995** | Beta / alpha (+theta) ratio | Operator engagement | Ratio tracks engagement | Basis of our engagement index. |
| **Klimesch 1999** | Alpha and theta | Attention, memory | Alpha drops with attention; theta rises with memory load | Basis of posterior alpha suppression and frontal theta. |

*Verify each reference against the original paper before citing.*

**Is this used commercially?** Yes: Nielsen ran EEG ad-testing labs; others
include Neurons Inc, iMotions, Emotiv and Neuro-Insight. Their accuracy claims
are mostly unpublished, company-run analyses, so careful academic work is useful.

---

## 5. Hardware and software

### Hardware
- **Amplifier:** Mitsar
- **Cap:** **SmartDRY 19-channel dry-electrode cap**. No gel: electrodes press
  through the hair onto the scalp. Quick to set up, but noisier than gel, and
  contact is the main source of trouble.
- **Electrodes:** standard 10–20 layout: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3,
  Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2. **F3 and F8 are permanently dead.**

### Software
- **EEGStudio Acquisition** (Mitsar's recording software)
- **Our code** (GitHub: `manojkulkarni123/Neuromarketing-EEG`, Python 3.11),
  run inside an isolated virtual environment `.venv` with PsychoPy (stimulus
  display), pylsl (data streaming), MNE (EEG processing), NumPy/SciPy, matplotlib.

### First-time setup on a new laptop
```
git clone https://github.com/manojkulkarni123/Neuromarketing-EEG.git
cd Neuromarketing-EEG
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Always run scripts with `.venv\Scripts\python.exe`, not plain `python`.

---

## 6. How the system works

```
 ┌─────────────────┐   LSL stream    ┌──────────────────────────────┐
 │  EEG cap → amp  │ ──────────────► │  Our task (ad_viewing_task)  │
 │  → EEGStudio    │  (raw EEG,      │  • baselines, cross, ads,    │
 │  "LSL Outlet"   │   27 channels,  │    gray screen, ratings      │
 └─────────────────┘   ~252.6 Hz)    │  • records the EEG stream    │
                                     │  • stamps markers on the     │
                                     │    SAME clock                │
                                     └──────────────┬───────────────┘
                                                    │ saves
                                      <id>_raw.fif + <id>_ratings.csv
                                                    │
                                     ┌──────────────▼───────────────┐
                                     │  Analysis                    │
                                     │  1. electrode quality check  │
                                     │  2. filter + remove blinks   │
                                     │  3. four measures, 3 s       │
                                     │     windows, 1 s steps       │
                                     │  4. relative to baseline     │
                                     │  5. per-ad numbers, timeline │
                                     │     plot, replay viewer      │
                                     └──────────────────────────────┘
```

**LSL (Lab Streaming Layer)** streams data between programs with precise
shared timestamps. EEGStudio *broadcasts* the EEG; our task *receives* it and
records exactly when each ad started and ended, all on one clock. That's how
we know which brain signal belongs to which second of which ad (accurate to
about 0.1 s).

---

## 7. Setup problems we hit, and how we solved them

| Problem | What we found | Fix |
|---|---|---|
| **EEGStudio's "LSL Inlet" wouldn't connect** (the plan was for EEGStudio to *receive* our markers) | Tried text, numeric and continuous-signal markers, firewall off, network config files. Other programs could see our stream, so the fault is on EEGStudio's side. | **Reversed the direction:** EEGStudio's **LSL Outlet** sends EEG to us, and we record it plus markers ourselves. |
| **Unstable sample rate** | "Processed data" mode delivered 236–271 Hz instead of 250 Hz. | Use **"Raw data (in uV)"** mode (steady ~252.6 Hz); the code always uses the **measured** rate and real timestamps (nominal 250 Hz would drift markers ~6%). |
| **The stream has no channel names** | 27 unlabeled channels. A first guess from EEGStudio's settings list was **wrong** (tapping O1 showed up as "Cz"). | **Calibration:** recorded our stream and EEGStudio's labeled EDF export together, matched channels by correlation, confirmed with a tap test. Saved in `channel_map.json`. |
| **Dead electrodes** | **F3 and F8 can't make contact** (same noise as unconnected inputs). Back electrodes often have poor contact. | **Automatic quality check** every session; analysis switches to working electrodes and refuses to compute a measure rather than fake it. |
| **Quality limits too lenient** | In test1, Pz (466 µV) and O2 (404 µV) passed the old 500 µV / r 0.9 limits but showed no real brain signal. | Tightened to **150 µV / r 0.8**. |
| **Video playback crashes** | PsychoPy failed on filenames with spaces/brackets, and sometimes on the first frame. | Ads renamed `ad01.mp4`…; the code **retries** the start, and the "ad started" marker is sent only once the first frame is on screen. |
| **Key-press artifact at ad start** | In test1, the first ~5 s of the ad were ruined by movement from the rating key press. | Added a **4 s fixation cross** before every ad. |
| **Logo reaction was cut off** | The brain reacts a moment after the ending/logo, after the video stops. | Added a **3 s gray screen** after every ad; the timeline continues through it. |
| **Blink removal could erase the signal we measure** | Fp1/Fp2 are both our asymmetry pair and our blink detector. | Blink detection uses only the **slow (<4 Hz)** band; blinks are slow, alpha is 8–13 Hz. Tested: blinks removed, alpha pattern preserved. |
| **Backwards label** | The code labeled negative asymmetry as "approach". | Fixed. **Positive = approach.** |
| **EEGStudio export unreadable** | Default export is a proprietary `.eeg` format. | Export as **EDF** when an EEGStudio file is needed. |

---

## 8. Experiment design

### Stimuli: five existing real TV ads ✅
- `ad01.mp4` … `ad05.mp4`, **32–90 s** long (original names in
  `ads/ad_name_map.csv`). Examples: ad02 = Volkswagen "The Force" (61.6 s),
  ad04 = Budweiser puppy/Clydesdale (60.1 s).
- Shown in **random order, shuffled differently per participant ID**.
- **No AI-generated ads, no A/B versions, no fictional brands.**

### Session

| Step | Time | Marker | Purpose | Status |
|---|---|---|---|---|
| Consent, cap fitting, contact check | ~15 min | — | Don't start until Fp1/Fp2/Fz are good | ✅ quality check in analysis; ⬜ check *before* recording |
| Eyes open, look at a cross | 60 s | `BASELINE_EO_START/END` | Reference level for every measure | ✅ |
| Eyes closed (beep to open) | 60 s | `BASELINE_EC_START/END` | Cap check: back-of-head alpha should rise | ✅ |
| **For each of the 5 ads:** fixation cross | 4 s | `PRE_AD_START:<ad>` | Lets the head settle after the last key press | ✅ |
| Ad | 32–90 s | `AD_START` / `AD_END:<ad>` | The only part used for per-ad numbers | ✅ |
| Mid-gray screen with cross | 3 s | `POST_AD_END:<ad>` | Catches the delayed reaction to the ending/logo | ✅ |
| Two ratings (1–9) | self-paced | (CSV) | What the EEG is compared against | ✅ |

**Rating questions** (same wording and order every time; keys pressed during
the ad are cleared first):
1. Liking: *"How much did you like this ad?"* (1 = not at all … 9 = very much)
2. Purchase intent: *"How likely would you be to buy this product?"* (1 = very unlikely … 9 = very likely)

**Why eyes-open rest and not a neutral video as the baseline:** no clip is
truly neutral. The difference between resting on a cross and watching any
video is the same for every ad, so it cancels when ads are compared with each
other. The buffers are padding only, never the reference.

### Participants
- **Recruit ~24, aiming for 20 usable** (some won't get good contact on a dry cap).
- **Exclusion rule, fixed in advance:** exclude anyone with bad Fp1, Fp2 or Fz,
  or more than 5 bad channels.
- **Written consent** for everyone; check whether college ethics approval is needed.

**Open decisions:** add a "have you seen this ad before?" question (real ads
can be familiar); trim ads to equal length.

---

## 9. What we measure from the brain, explained

### Background: EEG frequency bands

| Band | Speed | Rough meaning |
|---|---|---|
| Delta | 1–4 Hz | Sleep; slow eye movements (artifact) |
| **Theta** | 4–8 Hz | Mental effort, working memory |
| **Alpha** | 8–13 Hz | **"Idling": high when an area rests, drops when it works** |
| **Beta** | 13–30 Hz | Alertness, active thinking (we cap at 25 Hz) |
| Gamma | 30+ Hz | On a dry cap, mostly muscle noise (skipped) |

**Key point: alpha is backwards.** More alpha = *less* active. Alpha dropping
("suppression") is a sign of engagement.

We compute each measure in **3-second windows, sliding 1 second at a time**
(Welch spectrum, 1 s segments, 50% overlap). A point at 14 s summarises about
12.5–15.5 s. One number per ad = the **median** over windows inside the ad.

### The four measures

| # | Measure | Electrodes | What it tells us | How to read it | Status |
|---|---|---|---|---|---|
| 1 | **Frontal (pole) alpha asymmetry** | Fp1, Fp2 | Does the reaction lean **toward (approach)** or **away (avoid)**? | ln(right α) − ln(left α). **Positive = approach ("I want this")**. Reported as change from baseline. | ✅ |
| 2 | **Engagement index (β ÷ α)** | Cz, Fz | How alert and engaged the brain is | log10(β/α); % change from baseline; higher = more engaged | ✅ |
| 3 | **Frontal theta** | Fz | How hard the brain is processing | Change in dB from baseline; moderate rise = processing; very large rise can mean confusion | ✅ |
| 4 | **Posterior alpha suppression** | O1, O2, Pz (fallback P3/P4) | Are the eyes and visual brain locked on the screen? | % alpha drop vs. baseline = visual attention (says *watched*, not *liked*) | ✅ |

**How they fit together:** posterior alpha → did they look? · engagement →
were they alert? · theta → were they processing? · asymmetry → did they want it?
Asymmetry is *direction*, engagement is *intensity*: a disturbing ad can be
very engaging and still produce avoidance.

### Three rules for interpreting any number
1. **Always relative to that person's own baseline.** Raw values on a dry cap
   can't be compared between people. Theta is in dB: +3 dB ≈ double, −3 dB ≈ half.
2. **One person × one ad is too noisy for asymmetry.** Over ~30 s, pure noise
   produces asymmetry changes of SD ≈ 0.18, while real effects are ~0.1–0.3.
   **Asymmetry conclusions come from group averages only.**
3. **A single person's timeline is a hypothesis, not a finding.** Matching a
   spike to an ad moment needs the same pattern across people.

**Caveats specific to our cap:**
- Classic asymmetry uses **F3/F4**; F3 is dead, so we use **Fp1/Fp2**. Call it
  **"frontal-pole asymmetry"** in the paper; it is more exposed to eye artifacts.
- **Posterior alpha needs good contact at the back of the head**, which is often weak.

---

## 10. Analysis plan

1. **Quality check** ✅: flag dead electrodes (|r| > 0.8 with an unconnected
   input, or > 150 µV after 1–40 Hz filtering).
2. **Cleaning** ✅: 1–40 Hz filter, then ICA to remove blinks.
3. **Features per ad** ✅: the four measures relative to baseline, plus a
   timeline plot and the replay viewer.
4. **Batch table** ⬜: one row per participant × ad: four measures + two ratings.
5. **Main test** ⬜: within each person, do the measures rise and fall with
   liking and purchase intent across the 5 ads? (Mixed-effects / repeated-measures
   statistics across ~20 people.)
6. **Hook score** ⬜: engagement in the first 5 s of each ad.
7. **ISC** ⬜ (later): how synchronized viewers' brains are for each ad.
8. **Prediction model** ⬜ (only after group data): predict ratings from EEG,
   tested on people the model never saw (leave one participant out).

**Keep models simple.** ~20 people × 5 ads = ~100 data points. Small models
beat deep ones at this size, which is exactly what Phase 1 showed (LDA > EEGNet).

---

## 11. Running a session, step by step

### Before the participant arrives
- Laptop charged; repo up to date (`git pull`).
- Ads in `data/lab_session/ads/` named `ad01.mp4` … `ad05.mp4` (**no spaces or brackets**).
- Choose the participant's ID (e.g. `p01`) and pass it with `--id`. It sets
  the file names **and** the ad order. The script refuses to overwrite an
  existing ID (`--overwrite` only if you mean it).

### EEGStudio
1. Start acquisition.
2. **Settings → Plug-ins → "Data type to transmit" = "Raw data (in uV)"**.
3. **Add-on tab → turn on "LSL Outlet"**.

### Cap fitting
- Press each electrode through the hair to the scalp. Part the hair under the
  back electrodes (O1, O2, P3, P4, T5); they often fail.
- **Fp1 and Fp2 are critical** (our only usable asymmetry pair). Press them
  firmly; a small drop of saline helps dry electrodes.
- Ignore F3 and F8: they're known dead.

### Record
```
.venv\Scripts\python.exe lab_task_psychopy\ad_viewing_task.py --id p01
```
- Check the terminal shows `[recorder] found 'Mitsar': 27 ch …`. If not,
  toggle LSL Outlet off/on in EEGStudio and retry.
- The participant follows the on-screen instructions: baselines, then for each
  ad a cross, the ad, a gray screen, and two ratings.
- **ESC** quits early; the data recorded so far is still saved.
- Quick tests: `--one` (just one ad), `--no-baseline` (skip baselines).

### Analyze
```
.venv\Scripts\python.exe pipeline\extract_neuromarketing_features.py data\lab_session\p01_raw.fif
.venv\Scripts\python.exe pipeline\build_replay.py data\lab_session\p01_raw.fif
```
The second command writes `data/lab_session/p01_raw_replay.html`; keep it in
that folder so it can find the ad videos.

### After the session
- **Back up `data/lab_session/` to a drive.** Recordings are **not** on GitHub,
  on purpose: they're participant data.
- Note anything unusual (participant moved a lot, electrode fell off).

---

## 12. Reading the output

The analysis prints:
1. **Electrode quality table:** each electrode `ok` or `BAD` with the reason.
   If Fp1/Fp2/Fz are BAD, the session can't be used for those measures.
2. **Cap check:** eyes-closed ÷ eyes-open alpha at the back of the head.
   **≥1.3 = OK**; lower means poor contact at the back.
3. **Which electrodes were used** for each measure.
4. **Per ad:** raw values plus **"vs baseline"**: `faa_change`,
   `engagement_change_pct`, `theta_change_db`, `alpha_suppression_pct`.

The plot (`<id>_raw_engagement_timeline.png`) has four panels:
- **Green shading** = ad playing; **hatched** = gray screen after the ad;
  **gray** = baseline periods; **dashed line** = the person's baseline level.
- Posterior alpha is drawn with an **inverted axis**, so up = more attention.
- If no working electrode pair exists for asymmetry, it says "unavailable"
  instead of drawing a fake line.

The **replay viewer** (`<id>_raw_replay.html`): choose an ad, press play; a red
cursor moves along the four graphs in time with the video, the shaded band
shows the 3 s window behind each point, and clicking or dragging on a graph
jumps the video to that moment.

---

## 13. Results so far

### Synthetic validation ✅
| Planted in fake EEG | Pipeline found |
|---|---|
| Blink of 127 µV on Fp1 | Reduced to 21 µV, asymmetry pattern intact |
| 50% alpha drop | 49.4% |
| +3.0 dB theta | +2.87 dB |
| Beta doubled (+100%) | +92% engagement |
| 2 s, 150 µV artifact burst | Per-ad result changed < 1% |

### Pilot session `pilot01` (29 Sep 2026, ad02)
- ✅ Recording and markers worked; marked ad length matched the video exactly.
- ❌ **8 of 19 electrodes had no real contact** (Fp1, F3, F8, T5, P3, P4, O1, O2),
  so asymmetry couldn't be measured. Theta (Fz) was valid.
- Lesson: **contact quality, not software, is the bottleneck.**

### Test run `test1` (19 Oct 2026, ad04, one participant)
- ✅ Whole chain worked: record → clean → measures → plot → replay. Ad marked
  as 59.8 s vs 60.1 s video.
- **10 of 19 electrodes usable** (Fp1, Fp2, F7, Fz, F4, C3, Cz, C4, T5, P3).
  ICA removed 1 blink component. Cap check 1.35× on P3: OK.
- **ad04 vs baseline:** posterior alpha suppression **45.4%**, engagement
  **+14.5%**, theta **−1.1 dB**, asymmetry **+0.06** (inside the noise floor).
- **Ratings:** liking 8, purchase intent 4. Liking an ad and wanting the
  product are different, which is why we ask both.
- This shows the system works; it is **not** a scientific result (n = 1).

---

## 14. Limitations to state honestly

- **Pilot scale:** ~20 people × 5 ads. Present it as a proof of concept.
- **Dry-electrode cap**, noisier than gel; **F3 and F8 dead**, so we use
  frontal-pole asymmetry (Fp1/Fp2) instead of the classic F3/F4.
- **Weak posterior contact**; posterior alpha may rely on P3 alone.
- **Single-person asymmetry is noise-limited** (see §9).
- **Real ads differ** in length (32–90 s), content and brand, and some may be
  familiar, so differences between ads can't be pinned to one cause.
- **Baseline is a static cross**, so alpha suppression includes a generic
  "watching video" effect (equal for every ad).
- **Ratings, not real purchases**, are the outcome.
- **EEGStudio's stream has no channel labels**; mapping recovered by
  calibration (redo it if the EEGStudio data mode changes).
- Venkatraman 2015 found classic EEG measures added little beyond
  questionnaires for predicting sales; a known challenge in the field.

---

## 15. What's built vs. what's left

### Built ✅
- Ad playback with per-participant shuffling and uniform screen size
- EEG recording over LSL with precise markers, saved even if the session is aborted
- Channel calibration and the saved channel map
- Baseline block (eyes open / eyes closed), fixation cross and gray-screen buffers
- Two ratings after every ad (liking, purchase intent)
- Automatic dead-electrode detection
- Blink removal (ICA) that protects frontal alpha
- Four measures (asymmetry, engagement, theta, posterior alpha) relative to
  baseline, with timeline plots and per-ad medians
- Offline replay viewer (video + graphs in sync)
- Synthetic validation of every analysis step
- Setup/diagnostic tools (`blink_test`, `tap_test`, `lsl_stream_watch`)
- Review 2 slides (`Review2_Neuromarketing_EEG.pptx`)

### To do ⬜, roughly in order
1. **test2:** full 5-ad session; check video playback stays stable.
2. **Fix posterior electrode contact** (O1/O2/Pz).
3. **Decide one laptop vs. two** for the session.
4. **Ethics/consent forms.**
5. **Electrode check before recording starts** (not just in analysis).
6. **Collect ~20 participants.**
7. **Batch results table** and **group statistics** (§10).
8. **Hook score**, then **ISC**, then a prediction model.
9. **Decide** on a "seen this ad before?" question and equal-length ads.
10. **Write up**, targeting MDPI *Sensors* / *Brain Sciences*, IEEE Access, or Frontiers.

---

## 16. Code map

| File | Purpose |
|---|---|
| `lab_task_psychopy/ad_viewing_task.py` | **The session**: baselines, cross, ads, gray screen, ratings, recording |
| `lab_task_psychopy/lsl_recorder.py` | Records EEGStudio's stream, stamps markers, saves `.fif` |
| `lab_task_psychopy/calibrate_channels.py` + `channel_map.json` | Recovers channel names (the stream has none) |
| `lab_task_psychopy/blink_test.py` | Quick rate / amplitude / physiology check |
| `lab_task_psychopy/tap_test.py` | Tap an electrode, see which channel lights up |
| `lab_task_psychopy/lsl_stream_watch.py` | Lists visible LSL streams |
| `lab_task_psychopy/lsl_marker_test.py`, `lsl_channel_test.py` | Leftovers from the failed LSL Inlet attempt |
| `pipeline/preprocess_mne.py` | Loading, quality check, filtering, ICA |
| `pipeline/extract_neuromarketing_features.py` | **Analysis**: four measures, baseline, plots, per-ad summary |
| `pipeline/build_replay.py` | Offline replay page: ad video + four graphs in sync |
| `pipeline/edf_reader.py`, `analyze_alpha_blocking.py` | EDF reading + simple cap check (older fallback) |
| `lab_task/` | Older manual-marker tasks (fallback if LSL ever fails) |
| `data/lab_session/ads/` | Ad videos (**not on GitHub**) |

---

## 17. Glossary

- **EEG**: electroencephalography; recording the brain's electrical activity from the scalp.
- **Dry electrode**: touches the scalp directly without conductive gel.
- **10–20 system**: standard electrode positions (Fp = frontal pole, F = frontal,
  C = central, T = temporal, P = parietal, O = occipital; odd numbers = left,
  even = right, z = midline).
- **LSL (Lab Streaming Layer)**: software for streaming data between programs on a shared clock.
- **Marker**: a timestamped label in the recording, e.g. `AD_START:ad03.mp4`.
- **Band power**: how much of a frequency rhythm (theta, alpha, …) is present.
- **FAA / asymmetry**: difference in alpha between right and left; positive = approach.
- **Engagement index**: beta ÷ alpha; higher = more alert.
- **ISC (inter-subject correlation)**: how synchronized different viewers' brain signals are.
- **ICA**: a method that splits EEG into components so artifacts like blinks can be removed.
- **Baseline**: a resting recording used as each person's reference point.
- **dB (decibel)**: log scale for power ratios; +3 dB ≈ double, −3 dB ≈ half.
- **Median**: the middle value; unlike the mean, barely moved by a short artifact.
- **AUC**: prediction accuracy score; 0.5 = chance, 1.0 = perfect.
- **Leave-one-participant-out**: train on everyone except one person, test on that person, repeat for each.
