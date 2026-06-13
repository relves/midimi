# Scale-Patch Drill — Slice 2

**Goal:** Build on Slice 1's daily scale-spelling drill with the features that make it land for
*this* learner specifically — play the scale on a real keyboard and have midimi check it, see
progress as a circle-of-fifths map, and flip the drill into ear-training (hear a scale, name the
key). Depends on Slice 1 (`docs/plans/scale-patch-drill-slice1.md`) being shipped.

**Why now (after Slice 1):** Slice 1 proves the daily loop with type/click input. Slice 2 adds
the three things the plan and the learner profile most reward:
- **"Play it" grading** — the plan's actual instruction is "play the scale slowly, one hand,
  while saying each note name." This learner has Grade-4 keyboard facility and plays by ear, so
  playing > typing. midimi already records MIDI input; we just need to grade it.
- **Mastery grid** — turns abstract progress into the plan's own mental model (the circle of
  fifths) and doubles as the Week 0 / Checkpoint self-test surface.
- **Ear→key direction** — the plan's ear-training half. Hearing a scale and naming the key uses
  the same engine in reverse and is a natural fit for an ear player.

## Scope of Slice 2

**In:** MIDI "play it" answer mode + grading, circle-of-fifths mastery grid, ear→key reverse
drill direction, and the per-direction state needed to support it.

**Out (Slice 3+):** macOS notifications / any background scheduler, minor & modal scales,
extended/seventh-chord drills, multi-octave spelling, microphone (sung) input.

## Decisions (settled, carried from Slice 1 + extended)

- **Nudge stays badge-on-open.** No notifications in Slice 2 either.
- **Spelling strictness in typed/clicked modes stays exact** (per-key correct enharmonic).
- **NEW — "play it" mode is pitch-class tolerant.** A MIDI keyboard sends note numbers, not
  spellings; it cannot convey whether the player "meant" F♯ or G♭. So play-it grading compares
  *pitch-class sets/sequences*, and the result card still **shows** the correct spelling as the
  teaching reinforcement. Spelling rigor lives in the type/click modes; play-it rewards the
  physical/aural pattern.
- **NEW — ear→key is multiple-choice, not free spell.** Naming the key from sound is the skill;
  re-spelling it is Slice 1's job. Present 3–4 key choices to keep it a fast ear rep.

## Changes

### 1. "Play it" answer mode — grading recorded MIDI

**Reuse the existing input recording path.** `server.py` already has `engine.arm_recording()`,
`engine.stop_recording()` (returns `raw_events`), and `engine.current_input_port()`, exposed via
`POST /record/start` and `POST /record/stop`. The drill does **not** need quantization or tempo
estimation — only the pitches played — so it must *not* go through `quantize_recording`.

Add to `server.py`:

```
POST /drill/record/start   -> arms input (400 if no MIDI input port selected)
POST /drill/grade_played   body: {key}
       -> stops recording, extracts note-on pitches in order, grades, returns the
          same shape as /drill/grade plus {played_midi:[int], played_names:[str]}
```

Grading helper in `sequencer/drill.py` (pure, testable — takes raw events, no engine):

```python
def grade_played(events: list[dict], key: str) -> dict:
    """Extract note-on pitch classes in played order; compare to the 7 PCs of `key` major.
    Order-tolerant by default (ascending or any order accepted) but de-duplicates octaves so a
    two-octave run still reads as the 7-note set. correct iff the played PC set == expected PC
    set AND every played note belongs to the scale (no wrong notes)."""
```

- Display: `played_names` rendered with the key's *correct* spelling (map each played pitch class
  through `major_scale_notes(key)`), so a right answer visually reinforces the spelling even
  though grading was pitch-class based.
- Wrong notes highlighted; missing scale degrees listed. Same box/streak update via
  `schedule_after` as the other modes.
- **No MIDI input port?** The play-it toggle is disabled with a hint pointing at Settings
  (mirrors the existing `/record/start` 400 behavior). Type/click remain available.

### 2. Circle-of-fifths mastery grid

A read-only visualization driven by the existing `GET /drill/status` (extend its `keys[]` with
whatever the grid needs — `box`, `active`, `last_result`, `streak`).

- **Layout:** the 12 keys arranged as a clock/circle of fifths (C top, sharps clockwise, flats
  counter-clockwise) — the same map the plan teaches in Week 0. Each key is a segment/node.
- **Coloring:** by Leitner `box` (1→5) on a cold→warm ramp; dormant (not-yet-unlocked) keys are
  greyed; `due today` keys get a ring. Hover/tap shows last result + next-due.
- **Placement:** inside the drill panel, above or beside the prompt card; also serves as the
  Checkpoint self-test view ("recite C, G, D, A, F, B♭, E♭" → see them all at box ≥ 4).
- Pure-frontend in `static/index.html`; SVG or CSS-grid, matching current vanilla style. No new
  endpoint beyond the `status` extension.

### 3. Ear→key reverse direction

The drill gains a `direction` field; Slice 1 was implicitly `spell` (key→notes). Add `ear`
(sound→key).

- **State:** track scheduling per (key, direction). Cleanest is a `direction` column on
  `scale_drill` with PK `(key, direction)`; migrate via the `try/except ALTER`/rebuild pattern
  used in `init_db()`. A new key's `ear` row unlocks only once its `spell` row reaches box ≥ 3
  (you should be able to spell it before you're asked to recognize it by ear).
- **Endpoints:** `/drill/next` already returns `direction`; for `ear` it also returns
  `choices:[key,...]` (the answer plus 2–3 plausible distractors — neighbors on the circle of
  fifths make good distractors). Add:

```
POST /drill/play_prompt  body: {key}
       -> plays the scale ascending via the existing playback engine (server-side
          play_in_background), so the learner hears it without seeing the notes
GET  /drill/next          -> may now return {direction:"ear", key, choices:[...]}
POST /drill/grade         -> already accepts {key, answer}; for ear, answer is the chosen key
```

- **UI:** ear prompt shows a ▶ (replayable) and the choice buttons; no notes visible until after
  answering, then reveal the spelling + a ▶ to compare. Grading is exact key match; box/streak
  via the same `schedule_after`.
- **Mix policy:** `pick_next` interleaves due `spell` and `ear` items (e.g. spell-biased early in
  a key's life, ear once unlocked). Keep it in `drill.py`, unit-tested.

## Reuse map (beyond Slice 1)

| Need | Reused asset |
|---|---|
| Capture played notes | `engine.arm_recording` / `stop_recording` / `current_input_port` (existing) |
| "No input port" UX | existing `/record/start` 400 path + Settings MIDI-input selector |
| Hear-the-scale prompt | existing `play_in_background` (server-side playback) |
| Grid data | extended `GET /drill/status` |
| Scheduling/box logic | Slice 1's `sequencer/drill.py` (extended, not replaced) |

## New / touched files

- **edit** `sequencer/drill.py` — `grade_played`, ear distractor selection, direction-aware
  `pick_next`/unlock.
- **edit** `server.py` — `/drill/record/start`, `/drill/grade_played`, `/drill/play_prompt`;
  `direction` column + migration; `status` extension.
- **edit** `static/index.html` — play-it toggle + record/stop flow, mastery grid, ear-mode UI.
- **edit** `tests/test_drill.py` — `grade_played` (octave dedupe, wrong-note rejection, order
  tolerance), ear distractor sanity, direction-aware scheduling/unlock.

## Tests

Unit (pure, no engine/audio):
- `grade_played`: correct two-octave run → pass; one wrong note → fail with the offender flagged;
  6-of-7 → fail listing the missing degree; order-tolerant set match.
- ear distractors: returns the true key + 2–3 distinct in-circle neighbors, never the answer
  twice.
- direction scheduling: `ear` unlocks only at `spell` box ≥ 3; `pick_next` interleaving.

Endpoint (`TestClient`, temp DB; MIDI engine mocked/stubbed for record paths):
- `/drill/grade_played` with synthetic `raw_events` updates box/streak and returns
  `played_names` in the key's spelling.
- `/drill/play_prompt` returns ok and (mock) invokes playback.
- `/drill/next` yields an `ear` item with `choices` once a key qualifies.

Manual smoke: select a MIDI input in Settings → play-it toggle enabled → play C major up two
octaves → graded correct, spelling shown → grid shows C warming → later an ear rep for C plays
the scale and offers C/G/F choices.

## Risks / notes

- **Octave/duplicate handling in `grade_played`** is the subtle part — collapse to pitch classes
  and de-dupe before comparing, or a held/repeated note reads as wrong. Cover with tests first.
- **Direction migration** on `scale_drill`: changing the PK to `(key, direction)` on an existing
  DB needs a table rebuild, not a bare `ALTER`. Do it idempotently in `init_db()` guarded by a
  schema check, preserving Slice 1 rows as `direction="spell"`.
- Keep the daily "done" definition coherent across directions — the streak should count a day
  where the due queue (both directions) was cleared, not each direction separately.
