# NFL Model

Independent NFL score/spread/total/player-prop projections (EPA-based) and college
football score/spread/total projections (score-based), compared against live
FanDuel/DraftKings lines from The Odds API.

## How it works

1. **Ratings** (`models/ratings.py`) — pulls the last 5 seasons of play-by-play
   from nflverse and builds opponent-adjusted offensive/defensive EPA-per-play
   ratings (split pass vs. rush), with recency weighting across weeks and
   seasons and shrinkage toward league average for small samples.
2. **Game model** (`models/game_model.py`) — converts those ratings into a
   projected final score for a specific matchup, using a conversion factor
   from EPA to points that's calibrated from real historical games (not a
   guessed constant). Adjusts for home field, rest advantage, divisional
   familiarity, and weather. Compares to the market spread/total to find edges.
3. **Props model** (`models/props_model.py`) — projects player volume (targets/
   carries via recency-weighted usage share) × efficiency (yards/target,
   yards/carry, regressed toward a position baseline), adjusted for the
   opponent's defensive rating. Compares to the market prop line.
4. **Pick stability** (`storage/db.py`) — a pick only changes if the edge moves
   by more than `PICK_STABILITY_THRESHOLD` points, or fresh injury news shows
   up. This is what stops the dashboard from flip-flopping picks on normal
   line noise as kickoff approaches.
5. **Live tracking** (`pipeline/track_picks.py`) — every strong-confidence
   pick gets permanently snapshotted into `tracked_picks` the moment its game
   kicks off (so it can never be edited after the fact), then graded win/loss/
   push once the game's final — scores from The Odds API for spread/total,
   actual player stats from fresh play-by-play for props. This is the real,
   live record, distinct from the historical `pipeline/backtest.py`.
6. **Closing Line Value (CLV)** — also computed in `track_picks.py`, and
   available immediately at lock time (no need to wait for the game to
   finish). Compares the market line from when a pick FIRST went strong to
   the line at kickoff: positive CLV means betting the moment the pick fired
   would have beaten the closing number — the standard sharp-money signal,
   independent of whether the individual game wins or loses. Over enough
   picks, consistently positive CLV is a faster, lower-variance read on
   whether the model's signal is real than win rate alone.
7. **Dashboard** (`app.py`) — Streamlit. Top Plays (strong-confidence only),
   Track Record (results + CLV), Game Projections, Spreads & Totals, Player Props.

## Setup

```bash
pip install -r requirements.txt
```

Your Odds API key already lives in `.env` (gitignored). To rotate it, edit
`.env` — never commit that file.

## Running it

```bash
# 1. Build/refresh team ratings (run daily, or whenever you want fresher data)
python -m pipeline.build_ratings

# 2. Pull live odds + generate picks
python -m pipeline.generate_picks

# 3. Launch the dashboard
streamlit run app.py
```

The dashboard's sidebar also has a "Refresh picks now" button that re-runs
step 2 for you — which also locks/grades tracked picks (`pipeline/track_picks.py`
runs automatically at the end of `generate_picks`, no separate step needed).

## College football (spreads, totals, score predictions)

Added as a second league (toggle in the sidebar). Spreads and totals only — no
player props for college.

```bash
python -m pipeline.cfb_picks      # projections + picks for the current college week
python -m pipeline.cfb_backtest   # walk-forward test against 2022-2025 closing lines
```

**How it differs from the NFL model.** There is no free, current college
play-by-play (the public cfbfastR repo stops at 2021), so there's no EPA. The
college model is a ridge regression on final scores: each team gets an offense
and defense rating plus a league-wide home-field edge, games are weighted by an
exponential recency decay (400-day half-life), blowouts are capped at 40, and
every non-FBS opponent is lumped into one "generic FCS" team. Game results come
from ESPN's public scoreboard; historical closing lines (backtest only) come
from the cfbfastR-data repo. Score predictions come from the same ratings.

**Backtest (walk-forward, 3,000 FBS-vs-FBS games, real consensus closing lines;
parameters chosen on 2022-23, reported on held-out 2024-25):**

| | Result |
|---|---|
| Margin error (MAE) | model 13.2 pts vs. market 12.0 pts |
| Spread picks vs. the line | ~50% at every edge size from 0 to 10+ points |
| Total picks vs. the line | 2022-23: ~50-51%; 2024-25: 54-57% at edges >= 4 pts |

Read that plainly: **the college spread model has no demonstrated edge**, and
its picks are worse-informed than the market's own number. The 2024-25 totals
result is suggestive but did not show up in the tune seasons, so it isn't
established either. Treat college score predictions as a second opinion. The
college picks are tracked in their own Track Record (sport-filtered) so live
results, not this README, decide whether any tier is worth betting.

**College-specific notes**
- Edge thresholds are much wider than the NFL's (`config.CFB_EDGE_THRESHOLDS`:
  strong = 10 pts spread / 6 pts total) because the model is noisier. At these
  levels a meaningful share of games can clear the bar, so use the "plays per
  category" slider rather than reading "strong" as rare.
- The Odds API returns ~40 college games for FanDuel/DraftKings, and only games
  that haven't kicked off yet get picks. A first run mid-Saturday misses the
  games already underway.
- Team names are matched between the Odds API and ESPN with aliases plus a
  conservative fuzzy match. A game whose teams can't be matched is skipped and
  logged rather than guessed.
- Graded picks need a refresh within ~3 days of the game finishing (the Odds
  API scores endpoint only looks back 3 days).

## Backtest results (read this before betting anything)

`python -m pipeline.backtest` runs a walk-forward test: for every game, ratings
are rebuilt using ONLY play-by-play from strictly before that game (no
lookahead), graded against nflverse's real historical closing lines. Every
hyperparameter below is swept and selected on the 2024–2025 tune set only,
then validated on 2026 holdout data the selection process never saw — this
matters, because tuning a knob and reporting its score on the same games it
was tuned on will always look better than reality.

**Two structural fixes were hypothesized, implemented, and honestly tested.
Neither improved performance. Both were left at their original values because
the data didn't support changing them:**

1. **Shrink the model's margin toward the market's own line** (`BLEND_WEIGHT`,
   `final = BLEND_WEIGHT * model + (1-BLEND_WEIGHT) * market`). Hypothesis: the
   model's raw disagreements with Vegas are too extreme to trust at face value.
   Swept 1.0 (no shrinkage) down to 0.0 (pure market) on the tune set: **1.0
   was already the best option** (49.7% spread win rate on 447 decisions);
   every weaker value did worse. `config.BLEND_WEIGHT` stays at 1.0.
2. **Strengthen the ratings' shrinkage toward league average**
   (`PRIOR_GAMES`, in effective recency-weighted games). This started from a
   real bug: shrinkage was comparing against *raw* game count (~69 games
   across a 5-season window) rather than *effective* weighted sample size, so
   `PRIOR_GAMES=4` was shrinking almost nothing (weight ≈ 69/73 = 94.5% raw
   signal). **That bug is fixed** — shrinkage now correctly uses the
   recency-weighted effective sample size. But sweeping the shrinkage strength
   itself (4 up to 60 effective games) on the tune set found **4 — the
   original default — still tied-best** (49.7%); stronger shrinkage flattened
   out around 49% regardless of strength. `config.PRIOR_GAMES` stays at 4.

**What this means: the model's ATS win rate holds steady at ~48–50% across
every variant tested — a fix to the confidence-magnitude, and a fix to the
ratings' noise-tolerance, both land in the same place.** That's evidence the
limiting factor isn't miscalibration or overfitting-to-noise (both of those
would respond to shrinkage) — it's that the underlying signal (opponent-adjusted
EPA/play, home field, rest, divisional dampening) doesn't yet carry enough
independent information to beat a market that's already pricing in all of
that plus a lot more (injury-specific lineup impact, situational tendencies,
line movement itself). Totals fared better throughout (~52–56%, closer to or
above breakeven) and weren't implicated in either failed hypothesis.

**Real next steps, if you want to keep pushing on this, roughly in order of
expected impact:** (1) add signal the market has that the model doesn't yet
use directly — most of all, real injury severity/lineup-impact rather than
just a same-week names list; (2) replace the iterative opponent-adjustment
with a proper ridge regression (more standard, better handles small/uneven
schedules); (3) consider that a single power-rating number per team may be
inherently capped — most public models that do beat closing lines blend
several independent rating systems (EPA, DVOA-style, market-implied) rather
than trusting one. None of these are guaranteed to work either — the honest
answer after this pass is that a solid, bug-free implementation of "opponent-
adjusted EPA power ratings" is table stakes, not an edge, against a market
this efficient.

`python -m pipeline.props_backtest` checks the props model's method — was
recency-decay-weighting (recent games count more) earning its complexity? —
against three baselines: last game only, a plain season average, and a
trailing-8-game equal-weighted average, across 4 seasons of real data (e.g.
rushing yards, n=2,328). Season average won or tied-won on every stat tested;
decay-weighting was a consistent (small) step backward. **Fixed:**
`models/props_model.py` now uses equal weighting over a ~season-length window
instead of exponential recency decay — simpler and, per this test, at least as
accurate.

### What this means practically

- **Don't trust the "strong" spread tier's confidence label as "better bet."**
  Confidence tiers are based on edge size, and edge size has not been shown to
  predict accuracy in this model — the largest honestly-tested sample put the
  raw model's overall ATS win rate at ~48–50%, i.e. no better than a coin flip
  and below the 52.4% needed to beat -110 vig.
- **Two rounds of honest tuning (market-blending, ratings shrinkage strength)
  both failed to move that number.** That's a real result, not a dead end to
  paper over — it points at the underlying signal being the limiting factor,
  not calibration. See "Real next steps" above for what would actually need
  to change.
- Totals performed closer to (and sometimes above) breakeven throughout and
  weren't implicated in either failed hypothesis — still not proven, but not
  showing the same problem spreads are.
- The props model's projection *method* tests as sound (beats naive
  baselines) — that's a check of relative accuracy, not of betting edge, since
  there's no free historical prop-odds feed to grade actual bets against.
- `config.BLEND_WEIGHT=1.0` and `config.PRIOR_GAMES=4` are both the
  empirically-best values found so far, not just untouched defaults — but
  "best of the options tried" isn't the same as "proven to win." Don't read
  any pick's confidence label as "bet with confidence" until a future backtest
  shows a tier clearing breakeven on a larger, still-honestly-held-out sample.

## Other known limitations

- **Matchup adjustment for props is team-wide, not position-specific.** A
  cornerback matchup for a #1 WR uses the whole team's pass defense rating,
  not a CB1-specific or slot-specific split. Real sharp models split this out.
- **No live in-week injury scraping.** Injury notes come from the official
  weekly report via nflverse, which updates during the week but won't catch a
  Sunday-morning surprise scratch.
- **Backtest weather is not reconstructed** — historical games in the
  backtest are projected with `weather=None` since past forecasts aren't
  available after the fact, so the backtest slightly understates what the
  live model does for outdoor games.

## Project layout

```
config.py              constants, thresholds, paths
data/                   all external data access (odds, weather, nflverse, stadiums)
models/                 ratings engine, game model, props model
storage/                SQLite schema + access layer
pipeline/               build_ratings, generate_picks, track_picks, backtest, props_backtest,
                        cfb_picks, cfb_backtest
app.py                  Streamlit dashboard
```


## Position-specific prop matchups — tested, and rejected (with a bigger finding underneath)

Checked via `pipeline/props_matchup_check.py`: does splitting a defense's pass-EPA-allowed
by receiver position (WR vs TE vs RB) predict a player's next-game receiving yards better
than one team-wide number? Walk-forward, tuned on 2022-23, checked on 2024-25 (~6,000
player-games each), holding the player's own volume/efficiency baseline fixed so only the
matchup signal changes.

| Matchup signal | Tune MAE | Holdout MAE |
|---|---|---|
| None at all | 15.49 | **15.23** |
| Team-wide (current live model) | 15.65 | 15.31 |
| Position-specific (WR/TE/RB split) | 15.68 | 15.37 |

**Position-specific lost on every position group in the holdout (RB, TE, WR) — rejected,
not implemented.** The bigger, unexpected finding underneath it: *no* matchup adjustment at
all beat both alternatives, consistently in both the tune and holdout windows. That's a
real, evidence-based result, not just "we tried and it didn't help" — it suggests the
live model's existing team-wide matchup nudge (the small +/-10% clip in
`models/props_model.py`) may itself not be earning its keep for receiving yards, at least
with this EPA-allowed-per-target signal.

That said, this test used a simplified stand-in for the live model's efficiency baseline
(a plain cumulative yards/target average, not the full pooled-and-shrunk version
`project_receiving_rushing` actually uses), so it's evidence to investigate further, not
grounds to rip the live adjustment out on its own — a follow-up worth doing is running this
same comparison against the exact live formula before touching production code.

## Player-prop model fixes (from reviewing obviously-wrong projections)

A reviewer flagged props like "a RB projected for 22 rush yards / 5 carries" and "a QB
projected for 13 pass attempts as a 14-point underdog". Root causes, each tested on
real history in `pipeline/props_volume_check.py` (tuned on 2022-23, checked on 2024-25):

- **Volume was shrunk toward a generic 3-per-game prior.** With one game of data that
  put 75% of the weight on "3 carries" — a rookie with 10 carries was projected for ~5.
  Role players average ~10.8 carries, and usage is persistent, so volume now uses
  position-specific priors and light shrinkage (RB carries error 5.67 -> 4.24).
- **Relief appearances diluted QB volume.** A backup's history is mostly 1-6 attempt
  cameos; only games with 15+ attempts now count for a QB's passing *and* rushing.
- **Efficiency averaged per-game ratios**, so a 1-carry game counted like a 20-carry
  game. Yards per carry/target and catch rate are now pooled and shrunk by opportunities.
- **Matchup scaled volume by up to +/-25%** (Mahomes at 45 attempts). A weak defense
  changes yards per touch, not touches, so matchup now only scales yardage/TDs, capped at 10%.
- **Guardrails:** no "strong" pick on fewer than 3 games of data, and no pick at all when
  the model is more than 40% away from the market line. There's no free prop-line history to
  validate that exact number; it's a sanity limit, and the reasoning is that in an efficient
  market a huge gap is usually the model missing something (the game backtest found the
  same: the biggest model-vs-market gaps did worst).

Still unmodeled: role changes from injuries to teammates. That is exactly when a market
line sits far above a player's history, so treat any prop whose line looks "too high" for the
player as a signal the model may be missing news, not as an edge.

## Lines are real, single-book numbers

Picks used to average FanDuel and DraftKings (giving lines like +2.8 that don't exist).
Every spread, total, and prop now uses one actual line — FanDuel, falling back to
DraftKings — and the card shows which book ("FD"/"DK"). A moved line is saved as a new
version of the game without flipping the pick, so the opening line used for closing-line
value is preserved.
