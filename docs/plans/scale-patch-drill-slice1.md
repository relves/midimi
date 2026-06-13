# Scale-Patch Drill — Slice 1 (MVP)

**Goal:** Give midimi a dedicated *daily scale-spelling drill* so the learner can practice the
"scale patch" — the most important addition in the 8-week Functional Harmony plan — inside
midimi instead of bouncing to musictheory.net. One prompt at a time ("Spell D major"), graded
against correctly-spelled notes, with confirmation audio and a daily streak.

**Why now:** midimi today is an *explain-and-play* chatbot. The study plan
(`Functional_Harmony_Study_Plan_Personalized.md`) is built on *daily drilling and active recall*
— "5 minutes daily, every day, for the entire 8 weeks," with a recall test ("say the notes of D
major") and a fixed circle-of-fifths rotation. A conversational explainer can't verify recall or
hold a daily habit. This slice closes that gap with mostly-reused infrastructure.

**Pedagogical anchors (from the plan, §Week 0 + Checkpoint):**
- Spelling, *not* technique — "know A major contains F♯, C♯, G♯ without thinking."
- Daily ritual — cadence is the point; this gets its own surface, not a chat turn.
- Fixed rotation — C, G, F, D, B♭, A, E♭ first, then expand outward along the circle of fifths.
- Recall direction — prompt is "Spell D major" (no audio); audio is *confirmation*, not the cue.

## Scope of Slice 1

**In:** spelling helper, drill-state table with Leitner scheduling, `/drill/*` endpoints,
**type-only** answer input, a drill panel with a badge-on-open daily nudge and a streak.

**Out (deferred to Slice 2):** click-piano / MIDI keyboard answer input, MIDI "play it"
grading, the circle-of-fifths mastery grid, the ear→key reverse direction, macOS
notifications. Badge-on-open is the agreed nudge for Slice 1.

**Why type-only:** Slice 1 grades *strict enharmonic spelling* (C♯ ≠ D♭). Any pitch-based
input — on-screen piano *or* a MIDI keyboard — emits pitch classes only and cannot express
intended spelling (the black key is both C♯ and D♭). Pitch-class input therefore belongs to
Slice 2's enharmonic-tolerant "play it" mode. The text field is the only input that conveys
spelling, so it is the sole answer mode in Slice 1.

## Decisions (settled)

- **Answer input:** type-only in Slice 1 (see "Why type-only" above).
- **DB test seam:** `DB_PATH` honors a `MIDIMI_DB` env var so `TestClient` tests run against a
  temp DB and never touch the real `midimi.db`. Default unchanged when the var is unset.
- **Wrong-answer re-drill:** immediate. `schedule_after` resets to box 1, `due_at = now`;
  `pick_next` (most-overdue) returns it next, so the missed key is re-prompted right away.
- **Daily streak:** server-local calendar day. On the first correct that empties the due-queue
  for the day, advance `drill_streak_days` and set `drill_last_completed_date`. Reset streak to 1
  if `last_completed` is older than yesterday (a skipped day breaks the streak).
- **Nudge:** badge-on-open only. No notifications, no background scheduler.
- **Spelling strictness:** enforce *correct* spelling per key. In C major, `C#` is wrong where
  `Db` is expected and vice-versa — this is the lesson (cf. plan's "E♭–G♭–B♭ not E♭–F♯–B♭").
  Input is accidental-tolerant only on unicode vs ascii (`F#` == `F♯`) and case (`bb` == `Bb`),
  never on enharmonic substitution. (Enharmonic tolerance belongs to Slice 2's "play it" mode,
  where a keyboard can't convey intended spelling.)

## Changes

### 1. Spelling source of truth — `sequencer/theory.py`

Add a small helper wrapping music21 (already a dependency):

```python
def major_scale_notes(key: str) -> list[str]:
    """Correctly-spelled note names of `key` major, tonic first, no octave.
    e.g. "D" -> ["D","E","F#","G","A","B","C#"]; "Gb" -> ["Gb","Ab","Bb","Cb","Db","Eb","F"]."""
```

Implementation: `music21.scale.MajorScale(key).getPitches(...)` for one octave, take the 7
distinct pitch classes, render names with `#`/`b` (strip the trailing octave digit). music21
gives correct enharmonics for free (F♯ major → E♯, G♭ major → C♭), which is exactly the
spelling rigor the plan cares about. Add a unit test asserting a handful of keys incl. the
tricky ones (F#, Gb, Cb-free set, B, Db).

Normalization helper for grading input (also in theory.py, small + tested):

```python
def normalize_note_name(s: str) -> str:
    """'f#' / 'F♯' -> 'F#'; 'bb' -> 'Bb'. Letter upper, accidental ascii. No enharmonic remap."""
```

### 2. Drill-state model — new table in `server.py` `init_db()`

Follow the existing `CREATE TABLE IF NOT EXISTS` + `try/except ALTER` convention.

```sql
CREATE TABLE IF NOT EXISTS scale_drill (
    key         TEXT PRIMARY KEY,   -- "C","G","F","D","Bb","A","Eb",...
    box         INTEGER DEFAULT 1,  -- Leitner box 1..5
    due_at      INTEGER,            -- next-due epoch seconds
    streak      INTEGER DEFAULT 0,  -- consecutive correct for this key
    last_result TEXT,               -- "correct" | "incorrect" | NULL
    last_seen   INTEGER
);
```

Seeding (idempotent, on init): insert the 7 starter keys (C, G, F, D, B♭, A, E♭) as
**active** with `due_at = now` so they're due on first open. The remaining keys stay dormant
(absent from the table) until unlocked — see scheduling below.

Daily-streak counter (distinct from per-key streak) lives in the existing `settings` table via
`db_get_setting`/`db_set_setting`: keys `drill_streak_days` and `drill_last_completed_date`.

### 3. Scheduling logic — new `sequencer/drill.py` (pure, unit-tested)

Keep policy out of the route handlers.

```python
ROTATION = ["C","G","F","D","Bb","A","Eb","B","Db","F#","Gb","Cb","C#"]  # circle-of-fifths order
BOX_INTERVALS_DAYS = {1: 0, 2: 1, 3: 2, 4: 4, 5: 8}  # days added on correct, by NEW box

def schedule_after(box: int, correct: bool, now: int) -> tuple[int, int]:
    """Return (new_box, new_due_at). Correct -> promote (max 5), push due out.
    Wrong -> reset to box 1, due now (re-drill same session)."""

def pick_next(rows: list[dict], now: int) -> dict | None:
    """Most-overdue active key whose due_at <= now; None if nothing due today."""

def maybe_unlock(rows: list[dict], now: int) -> str | None:
    """If all currently-active keys are at box >= 3, return the next ROTATION key to seed."""
```

Unlock rule keeps the plan's "start with these, expand outward" pacing automatic: dormant keys
join only once the starter set is reasonably internalized.

### 4. Endpoints — `server.py`

```
GET  /drill/next
       -> {key, prompt:"Spell D major", due_today:int, streak_days:int} | {due:false,...}
POST /drill/grade   body: {key, answer:[str]}   # type-only in Slice 1
       -> {correct:bool, expected:[str], normalized_answer:[str],
           per_note:[{got,expected,ok}], box, due_today_remaining, streak_days}
GET  /drill/status
       -> {keys:[{key,box,active,last_result}], due_today, streak_days, done_today}
```

- **Grading:** order-strict, length-7, per-note compare of `normalize_note_name(answer[i])`
  vs `major_scale_notes(key)[i]`. Correct iff all 7 match. `per_note` drives ✓/✗ highlighting.
- On grade: update `scale_drill` row via `schedule_after`; bump per-key `streak`; on the first
  correct of a new calendar day that empties the due-queue, advance `drill_streak_days`; call
  `maybe_unlock` and seed a new key if returned.
- **Confirmation audio** reuses the existing `POST /play_midi/{note}` endpoint and the existing
  `playMidiNote(midiNum)` client helper (`static/index.html`). Note: `/play_midi` takes a MIDI
  **int**, and there is currently **no** client-side note-name→MIDI map (chips are rendered from
  server-supplied MIDI numbers; the only name conversion present is name→ABC). So a small
  name→MIDI helper (octave 4, ascii accidentals) is **net-new** frontend code — minor, but not a
  reuse.

### 5. UI — `static/index.html` (single file, vanilla, matches current style)

- **Sidebar:** a **"Daily Scale Patch"** entry pinned above the conversation list, showing a
  badge — `N due` (or a 🔥 streak count when 0 due). Badge fetched from `/drill/status` on page
  load = the badge-on-open nudge. Clicking opens the drill panel in the main pane (chat hidden).
- **Drill panel (one card, one prompt at a time):**
  - Prompt line: **"Spell D major"** + small "▶ hear after answering" affordance.
  - Answer input — **type only:** a single text field accepting space/comma-separated names
    (`D E F# G A B C#`). (Click-piano / MIDI input deferred to Slice 2; see "Why type-only".)
  - **Submit** → POST `/drill/grade` → render per-note ✓/✗, show the correctly-spelled answer,
    a ▶ to hear the scale ascending, and **Next** (auto-advance on correct after the playback).
  - Wrong answer → key resets to box 1, re-queued this session (immediate second chance later).
  - **Done state:** when `/drill/next` returns `due:false` → "✅ Patch done — 🔥 {streak} day
    streak. Come back tomorrow." Keep it a clean stopping point (the plan's "5 minutes").

## Reuse map

| Need | Reused asset |
|---|---|
| Correct scale spelling | `music21.scale.MajorScale` (already a dep via `theory.py`) |
| Confirmation playback | existing `POST /play_midi/{note}` + `playMidiNote()` (name→MIDI helper is net-new) |
| State persistence | existing `sqlite3` helpers + `init_db()` ALTER pattern |
| Daily-streak storage | existing `settings` table (`db_get/set_setting`) |
| UI shell, sidebar, fetch patterns | existing `static/index.html` |

## New files / touched files

- **new** `sequencer/drill.py` — scheduling policy (pure).
- **new** `tests/test_drill.py` — `major_scale_notes`, `normalize_note_name`, `schedule_after`,
  `pick_next`, `maybe_unlock`.
- **edit** `sequencer/theory.py` — `major_scale_notes`, `normalize_note_name`.
- **edit** `server.py` — `scale_drill` table + seed in `init_db()`; `/drill/next|grade|status`.
- **edit** `static/index.html` — sidebar badge + drill panel.

## Tests

Unit (no audio, no server):
- `major_scale_notes` for C, G, F, D, Bb, A, Eb, B, Db, F# (E♯), Gb (Cb) — exact spelling.
- `normalize_note_name` ascii/unicode/case; rejects enharmonic remap (leaves `Db` as `Db`).
- `schedule_after` promotion/reset + due math; `pick_next` most-overdue selection;
  `maybe_unlock` gating at box ≥ 3.

Endpoint (FastAPI `TestClient`, in-memory or temp DB):
- `/drill/next` returns a starter key on fresh DB; `/drill/grade` correct vs wrong updates box
  and streak; `/drill/status` counts due/done.

Manual smoke: open app → badge shows `7 due` → spell C correctly → hear it → advance → wrong
spelling on G re-queues it → finish → streak shows `1`.

## Out of scope / explicit non-goals

Modes, ear→key direction, MIDI "play it" grading, mastery grid, notifications, multi-octave
spelling, minor/other scales. All Slice 2+.
