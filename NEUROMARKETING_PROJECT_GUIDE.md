# EEG Neuromarketing Project: Team Guide

*BMS College of Engineering, Dept. of CSE, Major Project*
*Guide: Prof. Amit Pradhan · Team: Siddhant Sawhney, Manoj Kulkarni, Pranav Easwar, Padma Deepak*
*Last updated: 19 Oct 2026*

This document covers everything about the project: why we're doing it, how the
system works, what went wrong along the way and how we fixed it, how to run a
session, how to read the results, and what's left to do. Read it before going
to the lab.

> **Status legend:** ✅ built and tested · 🟡 built, not yet tested on real data · ⬜ planned, not built

---

## Contents
1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [How we got here](#2-how-we-got-here)
3. [Research question and what makes it novel](#3-research-question-and-what-makes-it-novel)
4. [What others have done (literature)](#4-what-others-have-done-literature)
5. [Hardware and software](#5-hardware-and-software)
6. [How the system works](#6-how-the-system-works)
7. [Setup problems we hit, and how we solved them](#7-setup-problems-we-hit-and-how-we-solved-them)
8. [Experiment design](#8-experiment-design)
9. [What we measure from the brain, explained](#9-what-we-measure-from-the-brain-explained)
10. [Analysis plan](#10-analysis-plan)
11. [Running a session, step by step](#11-running-a-session-step-by-step)
12. [Reading the output](#12-reading-the-output)
13. [Results so far (pilot)](#13-results-so-far-pilot)
14. [Limitations to state honestly](#14-limitations-to-state-honestly)
15. [What's built vs. what's left](#15-whats-built-vs-whats-left)
16. [Code map](#16-code-map)
17. [Glossary](#17-glossary)

---

## 1. The project in one paragraph

We show people short **AI-generated video ads** while recording their brain
activity with an **EEG cap**. Afterwards we test whether they **remember** the
products and which product they'd **choose**. The question is whether brain
activity recorded *during* an ad can predict whether that ad "works", meaning
whether it's remembered and chosen, **better than simply asking people
whether they liked it**.

---

## 2. How we got here

**Phase 1: Reanalysis of a public dataset (done).**
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
collecting our own data, on an applied decision-making problem:
**neuromarketing**. Choosing a product is a decision, so this stays connected
to Phase 1.

**Ideas we considered and dropped:**
- **Meta's TRIBE v2** (an AI model that predicts fMRI brain responses from video).
  Interesting, but it predicts fMRI rather than EEG, needs a GPU, and recent
  preprints show it doesn't reliably predict viewer engagement. Dropped to keep
  the scope focused.
- **P300 / LPP brain responses.** These need the same event repeated dozens of
  times to average out noise. Our design shows each ad once, so they aren't
  measurable here.

---

## 3. Research question and what makes it novel

**Main question:**
> Does EEG recorded during an ad predict whether people later **remember** and
> **choose** the product, beyond what their own rating says?

**Hypotheses:**
1. Ads that make viewers' brains respond *in sync* (high inter-subject
   correlation) are remembered and chosen more often.
2. A model using **rating + EEG** predicts choice better than **rating alone**.
3. The one element we vary between ad versions (see §8) changes both the brain
   response and the outcomes.

**Why it's novel:**
- **AI-generated ads are barely studied with EEG.** The only large EEG result
  we found is a NielsenIQ industry study (CES 2025; press release, not
  peer-reviewed). It reported *weaker memory-related brain activity* for AI ads,
  even high-quality ones. Nobody has properly tested whether EEG can predict
  which AI ads succeed.
- **AI ads let us run a controlled experiment.** With real ads, everything
  differs between ads, so you can't tell *why* one worked. With AI we make two
  versions of the same ad that differ in **one thing only**, which lets us talk
  about cause, not just correlation.
- **Fictional brands** remove a common confound: nobody walks in with an
  existing opinion of a brand that doesn't exist.
- **A realistic, low-cost setup:** a dry-electrode cap with some broken
  electrodes is close to how this would be used in practice. Our pipeline
  detects bad electrodes automatically and adapts.

---

## 4. What others have done (literature)

| Study | What they measured | Outcome | Result | What we take from it |
|---|---|---|---|---|
| **Hakim et al.** (bioRxiv, Tel Aviv University) | Band power at **Fp1, Fp2, Fpz** (frontal pole only), F7/F8 asymmetry, inter-subject correlation | Product preference; YouTube metrics | 68.5% accuracy (most vs. least preferred); **EEG improved on questionnaires alone** | Uses **the same electrodes that work on our cap**. Our main methods reference. |
| **Boksem & Smidts 2015** (J. Marketing Research) | Frontal-midline beta, gamma | Willingness to pay; **box-office revenue** | Brain measures beat stated willingness to pay | Use frontal beta. **Skip gamma**: on a dry cap it's mostly muscle noise. |
| **Venkatraman et al. 2015** (J. Marketing Research) | Occipital alpha, frontal asymmetry (and fMRI, eye tracking, etc.) | Real ad-driven **sales** | **These EEG measures added nothing beyond questionnaires**; only fMRI did | Reviewers may cite this against us. Alpha and asymmetry alone aren't enough, so we add synchrony. |
| **Dmochowski et al. 2014** (Nature Communications) | How *synchronized* viewers' EEG is (inter-subject correlation) | TV ratings, Twitter volume, Super Bowl ad rankings | A small lab group's synchrony predicted what millions preferred | **Our main measure.** (Verify details in the paper before citing.) |
| **Kislov et al. 2023** (MDPI Brain Sciences) | Central **beta/alpha ratio** | Real banner-ad performance across **291,301 users** | Significant correlation; EEG + eye tracking + self-report beat any single one | Add the beta/alpha engagement index. |
| **Eijlers, Boksem & Smidts 2020** | EEG arousal | Ad memorability and liking | Arousal → more memorable but sometimes **less liked** | "Works" must mean memory/choice, not just liking. |
| **NielsenIQ 2025** (industry) | EEG on AI ads (~150 people) | Memory activation | Weaker memory response to AI ads | The gap our project addresses. Press release only. |

**Is this used commercially?** Yes. Nielsen ran EEG ad-testing labs worldwide
from 2011 (scaled back during the pandemic); other players include Neurons
Inc, iMotions, Emotiv and Neuro-Insight. Industry accuracy claims (Nielsen
cites ~77%) are mostly **unpublished, company-run analyses**, and independent
replication is thin. That's an opening for careful academic work.

**Read before writing:** *Reliability of EEG Metrics for Assessing Video
Advertisements* (2024) and *Is EEG Suitable for Marketing Research? A
Systematic Review* (PMC7779633).

---

## 5. Hardware and software

### Hardware
- **Amplifier:** Mitsar
- **Cap:** **SmartDRY 19-channel dry-electrode cap**. No gel: electrodes press
  through the hair onto the scalp. Quick to set up, but noisier than gel, and
  contact is the main source of trouble.
- **Electrodes:** standard 10–20 layout: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3,
  Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2.

### Software
- **EEGStudio Acquisition** (Mitsar's recording software)
- **Our code** (GitHub: `manojkulkarni123/Neuromarketing-EEG`, Python 3.11),
  run inside an isolated virtual environment `.venv` with: PsychoPy (stimulus
  display), pylsl (data streaming), MNE (EEG processing), scikit-learn,
  OpenCV, matplotlib, pandas.

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
 │  → EEGStudio    │  (raw EEG,      │  • shows baseline + ads      │
 │  "LSL Outlet"   │   27 channels)  │  • records the EEG stream    │
 └─────────────────┘                 │  • stamps markers (ad start/ │
                                     │    end) on the SAME clock    │
                                     │  • collects ratings          │
                                     └──────────────┬───────────────┘
                                                    │ saves
                                      <id>_raw.fif + <id>_ratings.csv
                                                    │
                                     ┌──────────────▼───────────────┐
                                     │  Analysis                    │
                                     │  1. electrode quality check  │
                                     │  2. filter + remove blinks   │
                                     │  3. brain measures per ad    │
                                     │  4. relative to baseline     │
                                     │  5. timeline plot + summary  │
                                     └──────────────────────────────┘
```

**LSL (Lab Streaming Layer)** is a standard way of streaming data between
programs with precise shared timestamps. EEGStudio *broadcasts* the EEG, and
our task *receives* it and records exactly when each ad started and ended, all
on one clock. That's how we know which brain signal belongs to which second of
which ad (accurate to about 0.1 s).

---

## 7. Setup problems we hit, and how we solved them

Worth knowing so nobody repeats them.

| Problem | What we found | Fix |
|---|---|---|
| **EEGStudio's "LSL Inlet" wouldn't connect** (the plan was for EEGStudio to *receive* our markers) | Tried text, numeric and continuous-signal markers, firewall off, network config files. Other programs could see our stream, so the fault is on EEGStudio's side. | **Reversed the direction:** EEGStudio's **LSL Outlet** sends EEG to us, and we record it plus markers ourselves. Works reliably. |
| **Unstable sample rate** | "Processed data" mode delivered 236–271 Hz instead of 250 Hz. Assuming 250 would have shifted markers by seconds. | Use **"Raw data (in uV)"** mode (steady ~252 Hz), and the code always uses the **measured** rate and real timestamps. |
| **The stream has no channel names** | EEGStudio sends 27 unlabeled channels. A first guess from EEGStudio's settings list was **wrong** (tapping O1 showed up as "Cz"). | **Calibration:** recorded our stream and EEGStudio's own labeled export at the same time, then matched channels by correlation. The map is saved in `channel_map.json` and loads automatically. |
| **Dead electrodes** | **F3 and F8 can't make contact** (they record the same noise as unconnected inputs). In the pilot, Fp1 and the back electrodes also had poor contact. | An **automatic quality check** flags dead electrodes every session. Analysis switches to working electrode pairs, and refuses to compute a measure rather than fake it. |
| **Video playback crashes** | PsychoPy failed on files with spaces/brackets in the name, and randomly failed to start videos ~50% of the time. | Ads renamed `ad01.mp4`…; the code **retries** the start, and the "ad started" marker is only sent once the first frame is actually on screen. |
| **Blink removal could erase the signal we measure** | We use Fp1/Fp2 for asymmetry, and they're also our blink detector. | Blink detection uses only the **slow (<4 Hz)** part of the signal; blinks are slow, alpha is 8–13 Hz. Tested: blinks removed, alpha pattern preserved. |
| **EEGStudio export unreadable** | Default export is a proprietary `.eeg` format. | Explicitly export as **EDF** when an EEGStudio file is needed. |
| **Backwards label** | The code labeled negative asymmetry as "approach"; the standard convention is the opposite. | Fixed. **Positive = approach.** |

---

## 8. Experiment design

### Stimuli: AI-generated ads ⬜
- **8 fictional products × 2 versions = 16 ads**, each **~30 s**.
- The two versions differ in **one factor only** (to decide as a team), e.g.:
  - emotional story with a person **vs.** product-only showcase, or
  - brand logo early **vs.** only at the end.
- Keep identical: length, voice, music volume, and the same 3-second end card
  (brand name + logo).
- **Each participant sees only one version of each product** (8 ads): two
  lists (A/B), half the participants each. Seeing both would make the second
  feel familiar.
- **Realism pretest:** ~15 people online rate each ad for "how real/professional
  does it look." Regenerate anything with obvious AI artifacts (odd hands,
  flicker); noticing them distorts attention.
- *(Optional but valuable)* add 1–2 **real ads** to compare against NielsenIQ's
  "AI ads are less memorable" claim.

### Participants
- **Recruit 24, aiming for 20 usable** (some won't get good contact on a dry cap).
- **Exclusion rule, fixed in advance:** exclude anyone with bad Fp1, Fp2 or Fz,
  or more than 5 bad channels.
- **Written consent** for everyone; check whether college ethics approval is needed.

### Session (~45 min)

| Step | Time | Purpose | Status |
|---|---|---|---|
| Consent, cap fitting, contact check | 15 min | Don't start until Fp1/Fp2/Fz are good | ✅ quality check in analysis; ⬜ check *before* recording |
| Eyes open, 60 s (look at a cross) | 1 min | Baseline | ✅ |
| Eyes closed, 60 s (beep to open) | 1 min | Baseline + cap check | ✅ |
| Neutral calm video, 30–60 s | 1 min | Best baseline for "watching video" | ✅ (needs a clip in `data/lab_session/baseline/`) |
| **8 ads**, random order; after each: liking (1–9) + purchase intent (1–9) | ~6 min | Main recording | ✅ |
| Distractor puzzles | 5 min | Clears short-term memory | ⬜ |
| **Free recall:** "type every brand you remember" | 2 min | Memory outcome | ⬜ |
| **Recognition:** 8 real + 8 fake logos, "seen it?" | 2 min | Memory outcome | ⬜ |
| **Choice:** product pairs, "which would you take?" (one honored with a small real reward) | 3 min | Behavior outcome with real stakes | ⬜ |

**What "the ad works" means:** it is **remembered and chosen**. The rating is
secondary, because people tend to give polite ratings.

---

## 9. What we measure from the brain, explained

### Background: EEG frequency bands
EEG is the brain's tiny voltage at the scalp. It mixes rhythms at different speeds:

| Band | Speed | Rough meaning |
|---|---|---|
| Delta | 1–4 Hz | Sleep; slow eye movements (artifact) |
| **Theta** | 4–8 Hz | Mental effort, focus, working memory |
| **Alpha** | 8–13 Hz | **"Idling": high when an area rests, drops when it works** |
| **Beta** | 13–30 Hz | Alertness, active thinking |
| Gamma | 30+ Hz | On a dry cap, mostly muscle noise (skipped) |

**Key point: alpha is backwards.** More alpha = *less* active. Alpha dropping
("suppression") is a sign of engagement.

We compute each measure in **3-second windows, sliding 1 second at a time**,
which gives a value every second (the timeline) plus one summary number per ad.

### The five measures

| # | Measure | Electrodes | What it tells us | How to read it | Status |
|---|---|---|---|---|---|
| 1 | **Inter-subject correlation (ISC)**, *main measure* | All good channels | Did the ad grab **everyone** the same way? Engaged viewers' brains rise and fall together; bored viewers drift apart. | Higher = the ad captured people collectively | ⬜ |
| 2 | **Frontal-pole band power + asymmetry** | Fp1, Fp2 | Front-of-brain activity, and whether the reaction leans **toward (approach)** or **away (avoid)** | Asymmetry = ln(right alpha) − ln(left alpha). **Positive = approach ("I want this"), negative = avoid.** | ✅ asymmetry · ⬜ band power |
| 3 | **Engagement index (beta ÷ alpha)** | Cz, Fz | How alert and engaged the brain is right now | Higher than baseline = more engaged | ✅ |
| 4 | **Frontal theta** | Fz | How hard the brain is processing | Moderate rise = processing (good for memory); very large rise can mean confusion, so read it with the others | ✅ |
| 5 | **Posterior alpha suppression** | O1, O2, Pz | Are the eyes and visual brain locked on the screen? | Alpha drop vs. baseline = visual attention (doesn't say *liked*, only *watched*) | ✅ |

**How they fit together:**
- ISC → did it hold everyone's attention?
- Posterior alpha → did they look?
- Beta/alpha → were they alert?
- Theta → were they processing?
- Asymmetry → did they want it?

### Three rules for interpreting any number
1. **Always relative to that person's own baseline.** Raw values on a dry cap
   can't be compared between people. The pipeline already reports FAA as a
   change from baseline and theta as a change in decibels.
2. **One person × one ad is too noisy for asymmetry.** We tested this: over
   30 s, pure noise produces apparent asymmetry changes of about ±0.2, while
   real effects are only ~0.1–0.3. **Asymmetry conclusions must come from the
   group average, never from one person.** This is a limit of the data, not a
   bug.
3. **Per ad, keep it to a few numbers:** the average over the ad, the first
   5 seconds (the "hook"), and the timeline for plots.

**Caveats specific to our cap:**
- Classic asymmetry uses **F3/F4**, which is dead on our cap, so we use
  **Fp1/Fp2**. Call it **"frontal-pole asymmetry"** in the paper, and note it's
  more exposed to eye artifacts.
- The forehead electrodes pick up eye movements, so **frontal-pole theta is the
  least trustworthy number** in the set.
- **Posterior alpha needs good contact at the back of the head**, which failed
  in the pilot.

---

## 10. Analysis plan

1. **Quality check** ✅: flag dead electrodes automatically (same as an
   unconnected input, or amplitude far above real EEG).
2. **Cleaning** ✅: 1–40 Hz filter, then ICA to remove blinks.
3. **Features per ad** (partly ✅): the five measures above, relative to baseline.
4. **Main test** ⬜: three models predicting *chosen* and *recalled*:
   - **rating only**,
   - **EEG only**,
   - **rating + EEG**.

   Always tested on **people the model never saw** (leave one participant out).
   **If rating + EEG beats rating only, that's our main result.** It's also the
   exact test that EEG failed in Venkatraman 2015, which makes it a strong framing.
5. **Ad-level test** ⬜: across all 16 ads, does an ad's ISC predict how often
   it's remembered/chosen? This is the "will this ad work" question.
6. **Manipulation test** ⬜: version A vs. B, on brain measures and outcomes.
   This tells us *why* ads work.
7. **Timelines** ✅: second-by-second plots per ad showing where engagement drops.

**Keep models simple.** ~20 people × 8 ads = ~160 data points. Small models
beat deep ones at this size; that's exactly what Phase 1 showed (LDA > EEGNet).

---

## 11. Running a session, step by step

### Before the participant arrives
- Laptop charged; repo up to date (`git pull`).
- Ads in `data/lab_session/ads/` named `ad01.mp4`, `ad02.mp4`… (**no spaces or brackets**).
- Neutral clip in `data/lab_session/baseline/`.
- Choose the participant's ID (e.g. `p01`) and pass it with `--id`. It controls
  the file names **and** the ad order, which is shuffled differently per
  participant. The script refuses to overwrite an existing ID (`--overwrite`
  only if you mean it).

### EEGStudio
1. Start acquisition.
2. **Settings → Plug-ins → "Data type to transmit" = "Raw data (in uV)"**.
3. **Add-on tab → turn on "LSL Outlet"**.

### Cap fitting
- Press each electrode through the hair to the scalp. Part the hair under the
  back electrodes (O1, O2, P3, P4, T5); they failed in the pilot.
- **Fp1 and Fp2 are critical** (they're our only usable asymmetry pair). They
  sit on the forehead with no hair, so press them firmly; a small drop of
  saline helps dry electrodes.
- Ignore F3 and F8: they're known dead.

### Record
```
.venv\Scripts\python.exe lab_task_psychopy\ad_viewing_task.py --id p01
```
- First check the terminal shows `[recorder] found 'Mitsar': 27 ch …`. If not,
  toggle LSL Outlet off/on in EEGStudio and retry.
- The participant follows the on-screen instructions: baseline, then ads, then a 1–9 rating after each.
- **ESC** quits early. The data recorded so far is still saved.
- Quick tests: `--one` (just one ad), `--no-baseline` (skip baseline).

### Analyze
```
.venv\Scripts\python.exe pipeline\extract_neuromarketing_features.py data\lab_session\p01_raw.fif
```
Optionally add `--bad F3,F8` to mark known-dead electrodes. They're also
detected automatically.

### After the session
- **Back up `data/lab_session/` to a drive.** Recordings are **not** on GitHub,
  on purpose: they're participant data.
- Note anything unusual (participant moved a lot, electrode fell off).

---

## 12. Reading the output

The analysis prints:
1. **Electrode quality table:** each electrode `ok` or `BAD` with the reason.
   If Fp1/Fp2/Fz are BAD, the session can't be used for those measures.
2. **Cap check:** eyes-closed ÷ eyes-open alpha at the back of the head. **≥1.3 = OK**;
   lower means poor contact at the back.
3. **Which electrodes were used** for asymmetry and theta.
4. **Per ad:** raw values plus **"vs baseline"**: asymmetry change, and theta
   change in dB (+3 dB means theta power doubled).

The plot (`<id>_raw_engagement_timeline.png`):
- **Green shading** = ad playing; **gray** = baseline periods.
- **Dashed line** = the person's baseline level.
- Top panel: asymmetry (**above baseline = more approach**). If no working
  electrode pair exists, it says "FAA unavailable" instead of drawing a fake line.
- Bottom panel: frontal theta.

---

## 13. Results so far (pilot)

**One pilot session (29 Sep 2026), one ad (ad02, 61.6 s), rated 9/9.**
- ✅ Recording, markers and timing all worked. The marked ad length matched the
  video exactly.
- ❌ **8 of 19 electrodes had no real contact:**
  - Fp1;
  - F3 and F8 (permanently dead);
  - T5, P3, P4, O1, O2 (readings of 3,000–10,000 µV; real EEG is ~15–60 µV).

  So **asymmetry couldn't be measured** for that session.
- ✅ **Theta (Fz) was valid.** One clear peak ~37 s into the ad, worth checking
  against what happens on screen at that moment. An earlier spike came before
  the ad started (the participant settling in).
- Lesson: **contact quality, not software, is now the bottleneck.**

---

## 14. Limitations to state honestly

Reviewers accept disclosed limitations. Hidden ones get papers rejected.
- **Pilot scale:** ~20 people, 16 ads. Present it as a proof of concept.
- **Dry-electrode cap**, noisier than gel; **F3 and F8 dead**, so we use
  frontal-pole asymmetry (Fp1/Fp2) instead of the classic F3/F4.
- **Single-person asymmetry is noise-limited** (see §9); asymmetry conclusions
  are group-level only.
- **AI ads may look artificial**: mitigated by the realism pretest.
- **The choice task uses fictional products** with a small token reward, not
  real purchases.
- **EEGStudio's stream has no channel labels**; mapping recovered by
  calibration (redo it if the EEGStudio data mode changes).
- Venkatraman 2015 found classic EEG measures don't predict sales. We address
  this with ISC and the "beyond questionnaire" test, but it's a known challenge
  in the field.

---

## 15. What's built vs. what's left

### Built ✅
- Ad playback with per-participant shuffling and uniform screen size
- EEG recording over LSL with precise markers, saved even if the session is aborted
- Channel calibration and the saved channel map
- Baseline block (eyes open / closed / neutral video)
- Automatic dead-electrode detection
- Blink removal (ICA) that protects frontal alpha
- Frontal-pole asymmetry + frontal theta timelines, relative to baseline, with plots
- Setup/diagnostic tools (`blink_test`, `tap_test`, `lsl_stream_watch`)

### To do ⬜, roughly in order
1. **Decide the manipulated factor** and the 8 products (team decision).
2. **Generate the 16 AI ads** and run the realism pretest.
3. **Pick a neutral baseline clip.**
4. **Task additions:** two counterbalanced lists, purchase-intent rating,
   distractor, free recall, recognition test, choice task, and an electrode
   check *before* recording starts.
5. **Analysis additions:** ISC, beta/alpha index, posterior alpha, frontal-pole
   band power, the three-model comparison, and the ad-level and manipulation tests.
6. **Ethics/consent forms.**
7. **Pilot 3 people**, fix issues, then collect the full 20.
8. **Write up**, targeting MDPI *Sensors* / *Brain Sciences*, IEEE Access, or
   Frontiers (realistic venues for applied pilot EEG work).

---

## 16. Code map

| File | Purpose |
|---|---|
| `lab_task_psychopy/ad_viewing_task.py` | **The session**: baseline, ads, ratings, recording |
| `lab_task_psychopy/lsl_recorder.py` | Records EEGStudio's stream, stamps markers, saves `.fif` |
| `lab_task_psychopy/calibrate_channels.py` + `channel_map.json` | Recovers channel names (the stream has none) |
| `lab_task_psychopy/blink_test.py` | Quick rate / amplitude / physiology check |
| `lab_task_psychopy/tap_test.py` | Tap an electrode, see which channel lights up |
| `lab_task_psychopy/lsl_stream_watch.py` | Lists visible LSL streams |
| `lab_task_psychopy/lsl_marker_test.py`, `lsl_channel_test.py` | Leftovers from the failed LSL Inlet attempt |
| `pipeline/preprocess_mne.py` | Loading, quality check, filtering, ICA |
| `pipeline/extract_neuromarketing_features.py` | **Analysis**: measures, baseline, plots, per-ad summary |
| `pipeline/edf_reader.py`, `analyze_alpha_blocking.py` | EDF reading + simple cap check (older fallback) |
| `lab_task/` | Older manual-marker tasks (fallback if LSL ever fails) |
| `data/lab_session/ads/`, `baseline/` | Ad videos and the neutral clip (**not on GitHub**) |

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
- **ISC (inter-subject correlation)**: how synchronized different viewers' brain signals are.
- **ICA**: a method that splits EEG into components so artifacts like blinks can be removed.
- **Baseline**: a resting recording used as each person's reference point.
- **dB (decibel)**: log scale for power ratios; +3 dB ≈ double, −3 dB ≈ half.
- **AUC**: prediction accuracy score; 0.5 = chance, 1.0 = perfect.
- **Leave-one-participant-out**: train on everyone except one person, test on that person, repeat for each.
- **Counterbalancing**: splitting participants across versions/orders so no version is systematically favored.
