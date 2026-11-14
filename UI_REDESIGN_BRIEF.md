# UI redesign brief (for the next session)

Goal: make the app look and work like a professional ad-testing product, and make the
EEG + video insights easy to find. Audience: marketers/product people (SAP and Bosch
panelists at the 14 Nov review) and the research team.

## What's wrong today
- No hierarchy: the answer ("what happened in this ad") is not at the top; six equally loud
  tiles with coloured tags (primary / exploratory / hypothesis / draft).
- Inconsistent: report renders dark, review light; each page has its own toolbar; generic gradient logo.
- Video is a black box until played (no poster frame); scene names truncated in the strip.
- Wasted width: ad thumbnails column + always-open assistant column.
- Jargon: "2 viewers · hypothesis", "vs rest", "pts", "Approach –".
- Scene review: wall of buttons + paragraph of keyboard help.

## Design rules
1. Core answer fits one screen; detail below it (progressive disclosure).
2. Grey first, ONE accent colour for what matters (selected measure, brand moments). No gradients, glow, emoji, gauges, donuts.
3. Every number has context (vs rest, vs this ad's median, n viewers). No number without it.
4. Trace directly under the video, same width, shared time axis, playhead, click-to-seek; scene bands + brand ticks as thin lanes; label lines directly.
5. Show n viewers next to every aggregate; withhold a verdict when data is noisy or n is small (say why).
6. Same layout for every ad; one left navigation for the whole app; light theme default (dark optional).
7. Empty / loading / error states say why and what to do.

## Features, in priority order
| # | Feature | Notes |
|---|---|---|
| 1 | Key takeaways + next steps card at the top of each ad report | Generated from existing data; each marked "confirmed for this viewer" or "hypothesis" |
| 2 | Peak/dip markers on the trace linked to the scene, with what was on screen and heard | Data exists (scenes, transcript, scene_values) |
| 3 | Brand-moment scorecard | brand_test table exists |
| 4 | All-viewers view per ad (median trace + each viewer + agreement) | Today only one viewer at a time |
| 5 | n and confidence everywhere | |
| 6 | Export: print-to-PDF report, CSV of timeline/scenes | Print CSS + Blob download |
| 7 | Cap-contact check inside Setup (head diagram, green/amber/red per electrode) | Would have saved p03; logic in scratch contact_check (notch + 1–40 Hz std, r with AUX) |
| 8 | Ad library: add video, label, edit brand / recall options / objectives in a form | Today via pipeline/make_ad_profiles.py |
| 9 | Viewer management: exclude a recording (p03), notes | |
| 10 | Ad comparison (small multiples) | Careful: SAP and Bosch panelists both attend -- no rankings between their ads |

## Must not break
- Server: `app/server.py` (stdlib, 127.0.0.1:8765, offline -- no CDN; bundle any font locally).
- APIs used by the pages: /api/setup, /api/session(+status), /api/eeg_check, /api/questions,
  /api/sessions, /api/process/<pid>(+status), /api/label(+status), /api/scenes/<ad>(+approve),
  /api/analysis/<ad>?session=, /api/thumb/<ad>?t=&w=, /media/<file>, /api/copilot, /api/copilot/brief/<ad>.
- Pages: app/static/setup.html, review.html, analysis.html, app.css.
- Measure tiers: attention primary, alertness supporting, processing exploratory, approach experimental (never a conclusion).
- Honesty rules: 1-2 viewers = hypothesis; noisy = "too noisy to judge"; never invent numbers.

Research sources (vendors and design guidance): iMotions, System1, Kantar LINK, Realeyes, Neurons,
NIQ BASES Ad360, Braingineers; NN/g, IBM Carbon, Datawrapper, Few.
