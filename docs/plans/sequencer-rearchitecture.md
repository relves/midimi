# Sequencer Re-architecture Plan

**Goal:** The agent reliably produces correct MIDI sequences on request (e.g. "play the first
6 bars of *Somewhere Over the Rainbow*"), and the user can play a MIDI controller into the
same sequencer so the agent can see, critique, and replay the performance.

**Why it fails today:** (1) melodies of named songs are reconstructed from LLM memory, which
is unreliable at the pitch/rhythm level; (2) the agent emits one-shot flat JSON event lists
(`at_beat` floats) with no readable result to verify against — beat arithmetic errors and
wrong notes go undetected. `validate_sequence` only flags gaps/overlaps/meter remainders, not
wrong content.

Phases are ordered by leverage. Each is independently shippable; later phases build on earlier
ones. Current code references are to `server.py` (1767 lines) as of branch `sequencer`.

---

## Phase 1 — ABC notation as the agent-facing format, with per-bar validation feedback

Highest-leverage change. LLMs are far better at ABC notation (heavily present in training
data) than at beat-offset JSON, and ABC's explicit barlines (`|`) make timing errors
mechanically detectable per bar instead of as a vague "remainder" warning at the end.

### 1.1 ABC parser/serializer module

New file `sequencer/abc.py` (this phase also starts the package split — see Phase 3.4):

- `parse_abc(text: str) -> Sequence` — supports the subset we need: `X:`/`T:`/`M:`/`L:`/`Q:`/`K:`
  headers, notes with octave marks (`C c c'`), accidentals (`^ _ =`), duration multipliers
  (`C2`, `C/2`, `C3/2`), rests (`z`), ties (`-`), chords (`[CEG]`), chord symbols (`"Cmaj7"`),
  barlines, and repeats (`|: :|` — expand on parse). Reject anything else with a precise
  error message naming the bar and token.
- `to_abc(sequence) -> str` — deterministic serialization, one bar per `|`, 4 bars per line.
- Internal target representation: the existing normalized sequence dict produced by
  `build_sequence()` (`server.py:724`) so playback, MIDI export (`write_sequence_midi`,
  `server.py:850`), and the UI pill pipeline are untouched.

Use a hand-rolled parser (the subset is small) rather than pulling in `music21` here —
`music21` arrives in Phase 2 and can be a cross-check in tests, but the hot path should be
dependency-light and give bar-precise errors.

### 1.2 Per-bar validation in the report

Extend `_sequence_warnings` / `_sequence_report` (`server.py:648,683`):

- Per-bar beat accounting: "bar 3 contains 4.5 beats; meter 4/4 expects 4" (error, not warning).
- Key analysis: given a `K:` key, list accidentals outside the key per bar ("bar 2: F♮ where
  key D major expects F♯") — informational, the agent decides if it's intentional.
- Echo back the normalized ABC (`to_abc`) in every tool result so the agent always reads what
  was actually stored, not what it thinks it sent.

### 1.3 Tool changes

- Replace `play_melody` and `play_sequence` inputs with a single `abc` string parameter
  (keep `title`; tempo/meter/key come from ABC headers). Keep `play_notes` as-is — root+quality
  works well for isolated chords and the chord-construction code (`build_chord`,
  `chord_note_names`) is solid.
- `validate_sequence` becomes `check_abc`: parses, returns normalized ABC + per-bar report +
  errors, plays nothing.
- Tool results for play tools include the same normalized ABC + report, so even an unverified
  play gives the agent a chance to self-correct on the next turn.
- Rewrite the relevant system-prompt sections (`server.py:248-272`): instruct the agent to
  write ABC, run `check_abc` until clean for anything beyond a couple of bars, and compare the
  echoed notation against intent before playing.

### 1.4 Tests

New `tests/test_abc.py` (pytest, run via `.venv/bin/python -m pytest`):

- Round-trip: `parse_abc(to_abc(seq)) == seq` for melodies, chords, dotted rhythms, ties,
  rests, 3/4 and 6/8 meters, repeats.
- Bar-accounting errors fire on over/underfull bars with correct bar numbers.
- Golden test: a known public-domain tune in ABC → MIDI tick-for-tick against a checked-in
  `.mid` fixture.

**Exit criteria:** agent-authored ABC with a deliberate bar-length error is rejected with a
bar-precise message; clean ABC plays and the pill/MIDI download still work.

---

## Phase 2 — Ground known songs in real data (import/lookup)

Fixes wrong *notes* for named pieces. Rule: **for any existing named piece, import — never
reconstruct from memory.**

### 2.1 Dependencies

Add `music21` to the venv (`.venv/bin/pip install music21`). It parses ABC, MusicXML, and MIDI,
and ships a public-domain corpus (Bach chorales, folk tunes, etc.).

### 2.2 Import tools

- `import_file` tool + `/import` upload endpoint: accept `.mid`, `.musicxml`/`.mxl`, `.abc`.
  Convert via `music21` into the internal sequence representation; return normalized ABC so the
  agent can excerpt/discuss it. Cap length (e.g. 64 bars) and flatten to the melody +
  chord-symbol subset our model supports, reporting what was dropped (extra staves, dynamics).
- `search_corpus` tool: query the bundled `music21` corpus by title; return matches and load
  one as ABC. This covers a lot of classical/folk material offline.
- Optional (flag-gated, off by default since this is a local-only app): `fetch_abc` tool that
  retrieves ABC from a user-pasted URL (thesession.org, abcnotation.com). No auto-searching
  the web — the user supplies the link.
- Keep the existing sheet-music-image path (system prompt line 255) but route its output
  through `check_abc` before playback.

Copyright note: *Somewhere Over the Rainbow* itself is not public domain; the supported answer
for in-copyright works is "user supplies a MIDI/MusicXML file or sheet-music image." The plan
makes that path first-class rather than pretending the model can recall it.

### 2.3 System prompt

Add the grounding rule: named existing piece → `search_corpus` / ask the user for a file or
image; only freely compose for original examples and generic theory demonstrations.

### 2.4 Replace the hand-rolled theory layer with music21

`server.py:26-233` (NOTE_NAMES, CHORD_INTERVALS, `_QUALITY_ALIASES`, `_INTERVAL_DIATONIC`,
`_DYAD_DIATONIC`, `_note_name_for_interval`, `prefer_flats_for`) reimplements chord-symbol
parsing and enharmonic spelling by hand. The override tables are the symptom: spelling from a
bare semitone interval is fundamentally ambiguous (6 semitones = A4 or d5), so exceptions
accumulate per quality, and `_QUALITY_ALIASES` grows one entry per shorthand the model invents.
Since this phase adds `music21` anyway, rebase the layer on it:

- `normalize_chord_quality` → `music21.harmony.ChordSymbol` parsing (handles "min7", "Δ", "ø",
  "C7b9" natively). Keep the interval-dyad qualities (m2…M7) as a small explicit map onto
  `music21.interval.Interval`, since dyads aren't chord symbols.
- `chord_note_names` / spelling → `Pitch.transpose(Interval(...))`, which spells correctly by
  construction; delete `_INTERVAL_DIATONIC`, `_DYAD_DIATONIC`, `_note_name_for_interval`, and
  the `prefer_flats_for` heuristic (key context comes from the ABC `K:` header where available).
- **Migration safety:** before swapping, snapshot current `build_chord` + `chord_note_names`
  output for every quality × representative roots (incl. flat/sharp roots) as golden tests.
  The current code's output is mostly correct — the goal is behavior-preserving replacement;
  any music21 disagreement gets reviewed, not silently adopted.
- **Containment:** all music21 usage stays behind `sequencer/theory.py` exposing the same
  three functions; cache `ChordSymbol` parses. Playback/engine code never sees music21 objects
  (it's a heavy import with its own object model).

### 2.5 Tests

- Fixture `.mid` and `.musicxml` files import to expected ABC.
- A corpus search for a known title returns the right piece and round-trips to playable MIDI.
- Golden spelling tests from 2.4 pass against the music21-backed `theory.py`.

**Exit criteria:** "play the first 6 bars of <corpus tune>" produces note-for-note correct
playback because the notes came from the corpus, not from recall.

---

## Phase 3 — Persistent, editable Sequence entity (the sequencer core)

Makes sequences first-class instead of ephemeral play-call byproducts, enabling incremental
construction, revision ("transpose bar 3 up a step"), and Phase 4 recording.

### 3.1 Data model

New tables in `init_db()` (`server.py:956`):

```sql
CREATE TABLE sequences (
    id TEXT PRIMARY KEY,           -- short uuid, same scheme as today
    session_id TEXT,               -- owning chat session (nullable for imports)
    title TEXT, abc TEXT,          -- ABC is the source of truth
    tempo_bpm REAL, time_signature TEXT, key TEXT,
    source TEXT,                   -- 'agent' | 'import' | 'recording'
    created_at INTEGER, modified_at INTEGER
);
CREATE TABLE sequence_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id TEXT, abc TEXT, created_at INTEGER
);
```

Store ABC as canonical; derive events/MIDI on demand (cached). Every edit appends a revision —
cheap undo and lets the agent diff "what changed."

### 3.2 Tools (replace the in-memory `_sequence_registry`, `server.py:584`)

- `create_sequence(title, abc)` → id + normalized ABC + report
- `read_sequence(id, bars="1-6")` → ABC excerpt + per-bar report
- `update_sequence(id, abc | bar_edits)` — `bar_edits` replaces specific bars, so the agent
  fixes bar 3 without re-emitting 32 bars (re-emission is where transcription errors creep in)
- `play(id, bars="1-6")` — bar-range playback; replaces the play side of `play_sequence`/
  `play_melody`, which become thin wrappers (create + play) or are dropped from TOOLS
- `list_sequences(session_id)`

### 3.3 Persistence migration

- `play_sequence_in_background` (`server.py:888`) and the `/sequence/{id}` endpoints read from
  the DB instead of `_sequence_registry`. Sequence pills in old chat histories regain replay
  after restart (today they silently die with the in-memory registry).
- Migrate `generated/orchestrations/*.mid` lazily: keep serving existing files; new sequences
  write MIDI on first download/play request.

### 3.4 Module split

Carve `server.py` into a package as part of this phase (it's the natural moment — the
sequencer gets its own state):

```
sequencer/
  model.py      # Sequence dataclass, DB access, revisions
  abc.py        # Phase 1 parser/serializer
  theory.py     # build_chord, parse_pitch, chord_note_names, normalize_chord_quality
  engine.py     # FluidSynth/MIDI-out player, transport (play/stop, bar-range scheduling)
  midi_io.py    # write_sequence_midi, Phase 2 importers, Phase 4 capture
tools.py        # TOOLS schemas + dispatch (agent-facing layer)
server.py       # FastAPI app, chat loop, SSE/UI rendering only
```

`engine.py` should also fix the transport while it moves: today playback holds `_play_lock`
for the whole sequence and sleeps between events; restructure as a single scheduler thread with
a stop flag and bar-range support (`bars="3-6"` offsets the schedule). Tick math stays as-is
(`MIDI_TICKS_PER_BEAT = 480`).

### 3.5 Tests

- CRUD + revision round-trips; bar-range read/play extracts the right events.
- `bar_edits` on bar N leaves all other bars byte-identical in ABC.
- Engine test with a fake `note_on/note_off` recorder asserting event order and bar-range
  offsets (no audio needed in CI).

**Exit criteria:** agent builds a 16-bar piece across multiple turns, fixes one bar without
touching the rest, plays bars 1–6 only, and the sequence survives a server restart.

---

## Phase 4 — MIDI controller input → same sequence model

### 4.1 Capture path

Server-side capture via `mido.open_input()` (mirrors the existing `mido` output handling in
`_init_player`, `server.py:511`):

- Settings: MIDI input port picker next to the existing MIDI-out picker in `/config`.
- Endpoints: `POST /record/start` (arm; optional count-in click through the engine),
  `POST /record/stop` → captures note_on/off with `time.monotonic()` timestamps.
- On stop: estimate or accept a tempo (UI lets the user set BPM before recording; tap-tempo
  later), quantize onsets/durations to a grid (default 1/16, configurable), convert to the
  internal representation, store as a `Sequence` with `source='recording'` plus a
  `raw_events` sidecar (unquantized timestamps, kept for timing critique).
- UI: a record button + "recorded: <pill>" message injected into the chat, same pill component
  as generated sequences (`sequence_pill`, `server.py:1131`).

WebMIDI in the browser is a fallback if server-side port handling proves flaky on macOS, but
server-side keeps one capture path for both UI and any future headless use.

### 4.2 Agent visibility and critique

- `read_sequence` works on recordings unchanged — the agent sees the quantized ABC.
- Add a timing report to `read_sequence` for `source='recording'`: per-note deviation from the
  quantized grid in ms (from `raw_events`), so the agent can say "bar 2 beat 3 was 40 ms early"
  rather than only critiquing the quantized result.
- Playback of recordings goes through the same engine — guaranteed faithful replay (quantized,
  or `quantize=false` to replay raw timestamps verbatim).
- System prompt: when a recording appears, read it before commenting; critique notes against
  the stated target (key/scale/tune) and timing against the grid report.

### 4.3 Tests

- Synthetic capture stream (recorded `mido` message list with timestamps) → quantizer →
  expected ABC, including swing/rubato edge cases near grid boundaries.
- Raw-replay path reproduces input timestamps within 1 tick.

**Exit criteria:** user plays a phrase on a controller, sees it as a pill, replays it
faithfully, and the agent critiques both note choice and timing from the same data.

---

## Cross-cutting

- **Test harness:** introduce pytest in Phase 1 (`tests/`, `.venv/bin/pip install pytest`);
  every phase adds to it. The audio layer is always behind the `engine.py` interface so tests
  inject a recorder instead of FluidSynth.
- **Sessions/chat schema:** unchanged. Existing chat histories keep rendering; old
  `play_sequence` tool-result messages in history remain valid JSON blobs.
- **Frontend:** Phase 1 needs no UI change (pills unchanged). Phase 3 adds nothing required.
  Phase 4 adds the record button and input-port setting.
- **Out of scope:** notation image rendering (via `music21`/verovio for visual verification),
  velocity curves/humanization. Multi-voice arrangements, voicing helpers, and per-note
  articulation/tempo are now planned — see `sequencer-phase5-arranging.md`.

## Order and rough effort

| Phase | What | Effort |
|---|---|---|
| 1 | ABC format + per-bar validation loop | ~2–3 sessions |
| 2 | Import/lookup grounding (`music21`) | ~1–2 sessions |
| 3 | Persistent Sequence entity + module split + transport | ~2–3 sessions |
| 4 | MIDI capture, quantize, critique | ~2–3 sessions |
