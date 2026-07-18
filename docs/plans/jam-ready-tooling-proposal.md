# Jam-Ready Tooling — What midimi Should Build

**Status:** proposal (no code yet). Sizing is relative, not estimated in hours.

**Source plan:** the Jam-Ready Edition of the 6–8 week Functional Harmony path. It is a
*superset* of `Functional_Harmony_Study_Plan_Personalized.md` already in this repo — same
harmony spine, plus a second thread aimed at playing with other people.

## The one-paragraph answer

The study plan has two spines, and midimi is currently built for only one of them.

The **harmony spine** (Intervals → Chords → Function → Motion) is recall work, and midimi
already has the right machine for it: `sequencer/drill.py` gives us Leitner scheduling, a
circle-of-fifths rotation, spell/ear directions, unlock gating, and MIDI-in grading, all
behind `/drill/*`. Supporting Weeks 1–6 is mostly **new card types on the existing
scheduler** — real work, but no new architecture.

The **jam spine** (Play in time → with a track → a form → a role → with people) is real-time
work, and *none of it exists*. There is no metronome, no backing track, no looping playback,
no drums, no swing feel, and no concept of "where am I in the form." That is the actual
build. It needs exactly one new primitive — a **looping comp engine with a bar/beat clock** —
after which every jam-thread exercise in the plan becomes a configuration of that one thing.

So: build the loop clock once, then spend the rest of the effort on drill card types.

## What already exists (and carries)

| Asset | Where | What it buys us |
|---|---|---|
| Leitner scheduler | `sequencer/drill.py` | Boxes, due dates, streak, unlock gating — reusable for *any* card type |
| Drill routes + UI | `server.py` `/drill/*`, `static/index.html` | Prompt → answer → grade → confirm-audio loop, mastery grid |
| Live MIDI input | `sequencer/engine.py` | Note-on capture; `grade_played` already grades played answers |
| Recording + quantize | `midi_io.quantize_recording` | Pairs note on/off against a tempo grid — **this is timing analysis already** |
| Playback engine | `sequencer/engine.py` | FluidSynth, scheduler thread, bar-range playback, tempo |
| Theory helpers | `sequencer/theory.py` | Scale/chord spelling, voicing, harmonization |
| Staff rendering | `sequencer/abc.py` | Visual confirmation of any prompt |

The quantizer is the quiet win. Grading *time-feel* — the thing the plan says jam hosts
notice first — mostly means comparing captured note-on timestamps to an expected grid, and
we already pair and quantize note events against a tempo.

## What's missing

Verified absent from the codebase: metronome, backing tracks, looping playback, drum parts,
swing/shuffle feel, form position, chord-chart data model, repertoire tracking.

---

## Slice A — The Loop Clock (the unlock)

**Status: shipped** — `sequencer/loop.py`, `/loop/*` routes, `tests/test_loop.py`. See the
Loop transport section of the README for the API.

Everything in the jam spine depends on this, so it ships first and alone.

A looping transport that plays a chord progression in time, forever, until stopped:

- **Bar/beat clock** with a tempo, time signature, and a straight/shuffle feel flag.
- **A comp track** generated from chord symbols via the existing `voice_progression` /
  `voice_chord` helpers — piano comp, plus a root-note bass so the learner can practise
  rootless voicings against a "bass player."
- **A click**, toggleable, with an accented downbeat.
- **A position feed** the UI can subscribe to: current bar, current beat, current chord, next
  chord. This is what makes "keeping your place in the form" visible.
- **Count-in** before the loop starts.

Why the position feed matters more than it sounds: the plan's recurring failure mode isn't
playing wrong notes, it's *losing the form*. A visible bar cursor turns an invisible skill
into a legible one.

**Serves:** Wk1 metronome habit, Wk2 root-through-the-form, Wk3 comping, Wk4 in-time
voicings, Wk5 rhythm and space, Wk6 rootless comping. Six of the eight weeks.

## Slice B — Charts & the 12-Bar Blues

A chord-chart data model — bars, chord symbols, repeats, key — that feeds Slice A.

- Ship the 12-bar blues as a built-in, transposable to any key (the plan calls out F, C, B♭).
- Triad mode vs dominant-7 mode, because Wk2 wants triads and Wk3 wants the all-dominant
  blues over the *same* form.
- Roman-numeral overlay toggle, so the Wk4 "identify the I, IV, V" lens is one click.
- Let the chat write charts: the LLM already emits playable pills, so *"give me a slow blues
  in F"* should produce a loopable chart, not just a one-shot playback.

**Serves:** Wk2 form introduction, Wk3 blues in F and C, Wk4 lead-sheet reading, Wk7–8
repertoire.

## Slice C — Harmony Drill Card Types

New prompt/grade pairs on the existing scheduler. Each is small; the value is cumulative.

| Week | Card type | Prompt → answer |
|---|---|---|
| 1 | Interval spelling | "Major 3rd above E♭" → play or type |
| 1 | Interval ear | hear it → name it |
| 2 | Triad spelling | "F♯ diminished" → play; inversions as a variant |
| 3 | Seventh chords | "Cm7♭5" → play; and the reverse, hear → name |
| 4 | Diatonic harmony | "iv7 in A♭" → play; "list the diatonic 7ths in E" |
| 5 | Function | given a progression, label T / PD / D; predict the resolution |
| 6 | Guide tones | "ii–V–I in D, 3rds and 7ths only, rootless" |

These reuse `grade_played` almost verbatim — it is already pitch-class tolerant, which is
exactly right for played chord answers where enharmonic intent can't be expressed.

Gating should mirror the existing `ear_unlocks` idea: you must be able to spell a thing
before you're asked to recognise it, and the plan's week order gives us the unlock graph for
free.

## Slice D — Playing in Time (grading the jam spine)

This is where Slice A and the quantizer meet.

- **Timing grade**: capture MIDI while the loop runs, report how close hits landed to the
  grid, per bar. Not pass/fail — a drift readout, because the plan's criterion is "without
  stopping," not "perfectly."
- **Form-integrity check**: did the learner play *something* in every bar they intended to,
  and did they change chords where the chart changes? This detects the actual failure
  ("lost the form") rather than wrong notes.
- **Space discipline** (Wk5): mark bars as rest-bars and verify they stayed silent while
  still landing the next entrance in time.
- **Chord-tone solo check** (Wk6): over a running chart, flag notes outside the current
  chord's tones. Confidence-building by construction, since the plan wants "every note works."

## Slice E — Repertoire & Milestones

Thin, but it's what makes the eight weeks feel like a path instead of a pile of drills.

- A **jam-readiness checklist** mirroring the plan's eight milestones, manually checkable,
  with drill evidence auto-suggesting a tick where we can measure it.
- A **three-tune pocket repertoire** shelf: charts marked as memorized, with a "test me"
  mode that hides the chart and runs the loop, so memorization is verified rather than
  claimed.

---

## Recommended order

**A → B → C → D → E.** A and B together are the smallest thing that changes daily practice:
a blues in F that loops, in time, with a click and a visible bar cursor. That alone covers
the Wk1–Wk3 jam thread. C can proceed in parallel with anything since it touches a different
subsystem. D is worthless before A exists. E is last and cheapest.

If only one slice ever ships, ship A.

## Open questions

1. **Two plan docs, one repo.** The personalized edition is in the repo root and the shipped
   drill was built against it; the Jam-Ready edition arrived via this task. One should be
   marked canonical before someone builds against the wrong spine.
2. **Drums.** Slice A assumes piano comp plus bass. A real drum loop is more motivating but
   depends on the soundfont's percussion bank being usable — worth a spike, not an
   assumption.
3. **Scope discipline.** The plan explicitly rules out modes, extended harmony, bebop
   vocabulary, and advanced rootless voicings. midimi's chat can already discuss all of
   these; the *drill* surface should honour the exclusions so practice stays on-plan.
