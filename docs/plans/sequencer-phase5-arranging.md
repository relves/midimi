# Sequencer Phase 5 — Arranging: multi-voice ABC, voicing helper, expressive timing

**Goal:** The agent can arrange a small piece — independent melody, inner-voice chord
voicings, and bass — and have it play back cleanly with musical phrasing (fermatas, dynamics,
tempo breathing), instead of mangling everything into one monophonic line.

**Why now:** Field test of Phases 1–4: asked to arrange a short ballad, the agent produced a
mangled result. Asked what it was missing, it named three gaps, all real limitations of the
current subset:

1. **Multi-voice ABC** — `V:` voices so melody, voicings, and bass are independent lines.
   Today `parse_abc` (`sequencer/abc.py`) is strictly single-voice; the only polyphony is a
   block chord `[CEG]` sharing one duration, so any arrangement must be vertically sliced into
   simultaneous-attack chords. That's the mangling.
2. **A voicing helper** — `voice_chord(root, quality, melody_note, register)` returning a
   sensible voicing under a melody note. LLMs know which chord to play but are unreliable at
   register/doubling/low-interval-limit details; this is exactly the kind of deterministic
   theory the music21-backed `theory.py` should own.
3. **Per-note tempo and articulation** — fermatas, dynamics, mid-piece tempo changes. Today
   every note plays at fixed velocity and full duration at one global tempo.

This phase supersedes the "multi-track arrangements … velocity curves" out-of-scope note in
`sequencer-rearchitecture.md` (Cross-cutting). Notation-image rendering remains out of scope.

Code references are to the post-Phase-4 layout: `sequencer/abc.py` (942 lines),
`sequencer/engine.py`, `sequencer/theory.py`, `tools.py`, `server.py`.

---

## 5.1 Multi-voice ABC (`V:`)

### Syntax subset

Standard ABC voices, the simple "stacked bodies" form (no interleaved inline `[V:1]`
mid-line switching — reject that with a precise error suggesting the stacked form):

```abc
X:1
T:Example
M:4/4
L:1/8
Q:1/4=72
K:Eb
V:1 name="melody"
V:2 name="harmony"
V:3 name="bass" octave=-1
[V:1] E2 G2 B2 g2 | f6 z2 |
[V:2] [GBe]4 [Bdg]4 | [Acf]8 |
[V:3] E,4 B,,4 | F,8 |
```

- `V:` header lines declare voices (id required; `name=` optional, `octave=` shift optional —
  these are the only attributes accepted, others rejected by name).
- Body lines begin with `[V:<id>]` and contain whole bars for that voice. Consecutive lines
  for the same voice concatenate, so long pieces interleave voices in 4-bar systems.
- A tune with no `V:` lines parses exactly as today — fully backward compatible; `to_abc`
  emits the single-voice form when there is one voice.

### Internal representation

- Each event in the sequence dict gains `voice: str` (default `"1"`) and the sequence gains
  `voices: [{id, name, octave_shift, channel, program}]`. Channels assigned in declaration
  order (0, 1, 2 …, skipping 9/percussion). `program` defaults to the current piano; expose
  later if needed.
- Existing consumers that ignore `voice` (pill rendering, bar accounting) keep working because
  events remain one flat `at_beat`-sorted list.

### Validation (extends the per-bar report)

- Per-voice per-bar beat accounting: "voice bass, bar 3: 4.5 beats; meter 4/4 expects 4".
- Voice length mismatch is an **error**: "voice harmony has 7 bars; melody has 8".
- Informational: voice crossing (a lower-declared voice sounding above a higher one) and
  unisons between voices, since both are common arrangement bugs the agent should see.

### Playback and export

- `engine._run_sequence` needs no structural change — it already plays a flat timed event
  list; it just sends each event on its voice's channel. Bar-range playback (`bars="1-6"`)
  filters all voices by the same bar window.
- `write_sequence_midi` (`sequencer/midi_io.py`) writes one track per voice (type-1 file),
  named from `name=`.
- `import_file` / corpus import: where Phase 2 flattened multi-staff sources to melody-only,
  now map up to 4 staves/parts onto voices instead of dropping them; keep reporting anything
  still dropped.

---

## 5.2 Voicing helper tools (`sequencer/theory.py`)

### `voice_chord(root, quality, melody_note=None, register="mid", style="close", omit_root=False)`

Returns a concrete voicing as note names + MIDI numbers + a ready-to-paste ABC chord token
(e.g. `[G,Bdf]`). Deterministic rules, implemented on music21 pitches:

- **Below the melody:** if `melody_note` given, top voicing note sits a 3rd–6th below it and
  never doubles it.
- **Low interval limits:** enforce the standard LIL table (no m3 below E3, no M3 below C3,
  no P4 below B♭2, etc.); violations push tones up an octave.
- **Guide tones first:** 3rd and 7th always present; 5th and root are droppable, root first
  when `omit_root=True` (the bass voice has it).
- **Styles:** `"close"` (block under melody), `"drop2"`, `"shell"` (root/3rd/7th or 3rd/7th
  only), `"spread"` (root low, guide tones mid). That's enough vocabulary for ballads and
  lead-sheet comping without becoming an arranging engine.
- **Register:** `"low" | "mid" | "high"` centers the voicing when there's no melody note.

### `voice_progression(chords, melody=None, style=...)`

Same rules applied across a chord list with minimal-motion voice leading: each voicing chosen
to minimize total semitone movement from the previous one (resolve ties toward keeping common
tones). Input: `[{symbol: "Cmaj7", beats: 4, melody_note: "e'"}, ...]`. Output: one ABC line
per chord plus the note-name breakdown, suitable for pasting into a `[V:2]` voice.

Both are read-only theory tools (no playback) so the agent can iterate cheaply, then assemble
the ABC itself. Cache nothing; they're fast.

---

## 5.3 Expressive timing and articulation

Smallest useful subset of standard ABC decorations and inline fields — all standard notation,
so the LLM already knows the syntax:

| ABC | Effect in playback |
|---|---|
| `!fermata!C4` or `HC4` | duration × fermata factor (default 1.8, header-overridable) |
| `.C` (staccato) | gate 50% of written duration |
| `!tenuto!C` | gate 100% (default gate is 90%) |
| `!accent!C` / `LC` | velocity +18 |
| `!p! !mp! !mf! !f!` | set running velocity (48/64/80/96) until next dynamic |
| `[Q:1/4=60]` inline | tempo change from that point (per-voice positions stay aligned because tempo is global) |

- Parser: decorations attach to the next note event as `articulation`, `dynamic`, `fermata`
  flags; inline `[Q:]` becomes a tempo-change event at its beat position. Unknown `!…!`
  decorations are an error naming bar and token, per the Phase 1 convention.
- `to_abc` round-trips all of the above; the per-bar report ignores them for beat accounting
  (a fermata stretches *performed* time, not written time — bar math stays on written values).
- Engine: gate length and velocity are per-event already; tempo changes make the scheduler
  convert beats→seconds piecewise instead of with one constant. Fermata stretch applies at
  schedule time so MIDI export can choose written (`expressive=false`) or performed timing.
- Explicitly **not** doing: continuous rubato curves, crescendo hairpins, humanization. The
  fermata + dynamics + inline tempo combination covers "ballad breathing" the agent asked for.

---

## 5.4 Tool and prompt changes

- `check_abc`, `play_abc`, `create_sequence`, `update_sequence`, `read_sequence`, `play`
  (`tools.py`) all accept multi-voice ABC automatically once the parser does — their report
  text gains the per-voice sections from 5.1.
- New tools: `voice_chord`, `voice_progression` (5.2). No new endpoints; no UI changes —
  pills and the MIDI download work as before (download just becomes multi-track).
- `update_sequence` `bar_edits` becomes voice-aware: `{"voice": "2", "bar": 3, "abc": "..."}`
  so the agent fixes one voice's bar without re-emitting the system.
- System prompt: add an arranging recipe — (1) write/import the melody as voice 1 and verify
  with `check_abc`; (2) decide the harmonic rhythm and call `voice_progression` for voice 2;
  (3) write the bass last; (4) `check_abc` until clean, reading the per-voice report; (5) add
  dynamics/fermatas only after pitches are right. Emphasize: never hand-construct voicings
  below the melody — use the helper.

---

## 5.5 Tests

- `tests/test_abc.py`: multi-voice round-trip (`parse_abc(to_abc(seq)) == seq`); voice
  bar-count mismatch and per-voice bar-length errors with correct voice+bar in the message;
  single-voice ABC unchanged byte-for-byte (regression).
- `tests/test_theory.py`: `voice_chord` golden tests per quality × style × melody note,
  asserting LIL compliance and no melody doubling; `voice_progression` total-motion bound
  (e.g. ii–V–I in close style moves ≤ 2 semitones per voice between chords).
- Engine recorder tests: three-voice sequence emits on three channels in correct order;
  staccato/tenuto gate lengths; inline tempo change shifts subsequent event times; fermata
  stretch present in performed schedule, absent from written-timing MIDI export.
- Golden MIDI: a 4-bar, 3-voice arrangement fixture, tick-for-tick.

---

## Exit criteria

The agent, asked to "arrange the first 8 bars of <corpus tune> as a ballad," imports the
melody, builds a 3-voice arrangement using `voice_progression`, passes `check_abc` with a
clean per-voice report, plays it with a fermata at the phrase end and a dynamic shape — and
the downloaded MIDI opens in a DAW as three named tracks.

## Order and rough effort

| Step | What | Effort |
|---|---|---|
| 5.1 | Multi-voice parse/serialize/validate/play/export | ~1–2 sessions |
| 5.2 | `voice_chord` / `voice_progression` | ~1 session |
| 5.3 | Decorations, dynamics, inline tempo | ~1 session |
| 5.4–5.5 | Tools, prompt, tests | folded into the above |

5.1 ships alone usefully (the agent can already hand-write voicings into voice 2); 5.2 and
5.3 are independent of each other and can land in either order after it.
