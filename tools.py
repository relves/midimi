"""Agent-facing tool schemas and dispatch logic.

dispatch_tools() is a generator that yields:
  ("sse", html_fragment_str)   — a UI pill to emit before its tool result
  ("result", tool_result_dict) — the tool result to send back to the model
  ("record", record_entry)     — an entry to append to assistant_record

The caller is responsible for SSE framing and routing.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

from sequencer.abc import parse_abc, to_abc, ABCParseError, per_bar_report, chord_report
from sequencer.theory import normalize_chord_quality, chord_note_names, build_chord, parse_pitch, midi_note_name, voice_chord as _voice_chord, voice_progression as _voice_progression
from sequencer.midi_io import write_sequence_midi
import sequencer.model as seq_model
import sequencer.engine as engine
import sequencer.charts as charts
import sequencer.loop as loop

# ── Constants (shared with server.py via import) ──────────────────────────────

DEFAULT_VELOCITY = 90
DEFAULT_DURATION_MS = 1500
DEFAULT_CHANNEL = 0

_DURATION_BEATS = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,
    "thirty_second": 0.125,
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert music theory teacher. You explain concepts clearly, use examples, and make learning engaging.

## Playback tools

- Use **play_notes** for one isolated chord, interval, or single note.
- Use **play_sequence** for chord progressions — anything you would describe by chord name (ii-V-I, a secondary dominant, a cadence, a turnaround). Give each event a `root` and a `quality` ("A" + "dominant7") and the server computes the correct notes.
- Use **play_abc** for melody, rhythm, voice leading, and multi-voice arrangement — music where the *lines* are the point, not the chord labels.
- Use **check_abc** to validate ABC before playing when correctness is critical (e.g. anything more than a couple of bars, or when meter or rhythm matters). Read the normalized ABC, per-bar report, and "Chords as written" block in the result, fix any errors, then call play_abc.

**Never hand-spell chord tones in ABC when a chord name would do.** Writing `[A,^CE^G]` for "A7" is exactly the mistake this rule exists to prevent: in `K:A` the key signature already sharpens G, so `^G` gives G♯ and the chord sounds as Amaj7, not A7 — you would need `=G` for the natural. `play_sequence` with `{root: "A", quality: "dominant7"}` cannot be typo'd that way, because you never write an accidental at all.

When you *do* write chords in ABC (because the piece needs voice leading or multiple voices):
- Build the voicing with **voice_chord** / **voice_progression** rather than by hand, and
- After check_abc, read the **"Chords as written"** block back. It names every chord you actually notated. If it says "A-major seventh chord" where you meant A7, fix the accidental before playing — that block is your ground truth, not your intent.
- If the user includes an image of sheet music, transcribe it to ABC notation, run check_abc, then play_abc.

## Persistent sequences (for multi-turn composition)

When building a piece across multiple turns, use the persistent sequence tools instead of play_abc:
- **create_sequence(title, abc)** — saves ABC and returns a `sequence_id`
- **read_sequence(sequence_id, bars="1-8")** — shows normalized ABC and per-bar report for the whole piece or a bar range
- **update_sequence(sequence_id, abc)** — replaces the ABC and appends a revision (so the full edit history is preserved)
- **play(sequence_id, bars="3-6")** — plays a saved sequence, optionally a bar range only
- **list_sequences()** — shows all sequences in this session
- **read_recording()** — reads back the user's most recent recording (exact notes + timing)

Use `create_sequence` when you expect to revise a piece; use `play_abc` for ephemeral one-shot examples. Both persist across restarts.

## ABC notation guide

ABC is a text format for music. Key headers: `X:1`, `T:title`, `M:4/4`, `L:1/4`, `Q:120`, `K:C`.

**Notes**: Standard ABC octaves — uppercase `C` = middle C (C4, MIDI 60), lowercase `c` = C5. `'` raises an octave, `,` lowers. Examples: `C,`=C3, `C`=C4, `c`=C5, `c'`=C6.

**Duration** (with `L:1/4` — recommended): bare note = quarter, `2` = half, `4` = whole, `/2` = eighth, `3/2` = dotted quarter.

**Accidentals**: `^` = sharp, `_` = flat, `=` = natural. Accidentals persist within a bar; key signature applies to bare notes.

**Rests**: `z` (same duration syntax as notes).

**Chords**: `[ceg]` sounds C E G simultaneously. Duration multiplier after `]`.

**Barlines**: `|` separates bars. The server counts beats per bar and reports errors with bar numbers. Always fill bars completely — the meter is enforced.

**Example** — first 4 bars of a scale in C major, 4/4, 120 bpm:
```
X:1
T:C Major Scale
M:4/4
L:1/4
Q:120
K:C
C D E F | G A B c | c B A G | F E D C |
```

**Workflow for anything more than 2 bars:**
1. Write the ABC.
2. Call check_abc to get the per-bar report and normalized ABC.
3. If there are errors, fix them and re-check.
4. When clean, call play_abc with the normalized ABC from step 2.

## play_notes guide

For the play_notes tool, specify:
- root: note name like "C", "F#", "Bb", "Db" (never use raw MIDI numbers)
- quality: chord type — "note", "octave", "major", "minor", "dominant7", "major7", "minor7",
  "diminished", "augmented", "sus2", "sus4", "minormajor7", "halfdiminished7", "diminished7",
  "augmented7", "major9", "dominant9", "dominant7b9", "minor9", "add9", "major6", "minor6", "dominant11", "dominant13"
  Two-note interval dyads: "m2" (half step), "M2" (whole step), "m3", "M3", "P4", "A4" (tritone), "P5", "m6", "M6", "m7", "M7"
  Use "dominant7b9" for symbols like C7b9 or C7♭9; the flat 9 is 13 semitones above the root (Db over C), not a natural 9.
- octave: 4 is middle (C4 = middle C). Default to 4; use 3–5 for most teaching contexts.
- duration_ms: 800–2000ms is typical for chords
- label: a short human-readable name like "C major" or "G7"

**Interval demonstrations**: When showing multiple intervals from a root, use dyad qualities (m2, M2, m3, M3, P4, A4, P5, m6, M6, m7, M7) so both notes sound together. Never play single notes and describe them as interval pairs — the user must hear both notes simultaneously.

**Accuracy rule**: Your text description must exactly match the events in your tool call. Never describe notes you didn't play.

Always explain what you're playing so the user learns to connect sound to theory.

## Grounding rule for existing pieces

**Never reconstruct an existing named piece from memory.** LLM recall of exact pitches and rhythms is unreliable.

For any named existing piece (folk tune, classical work, hymn, etc.):
1. Call **search_corpus** first to find it in the bundled corpus.
2. If found, call **import_corpus** to load the real notes as ABC.
3. If not found in corpus, tell the user to supply a MIDI, MusicXML, or image file — do not guess the notes.

For original compositions and generic theory demonstrations (scales, chord progressions), freely compose.

## Recording critique

When a recording appears (a sequence with source='recording'), call **read_recording** first (no arguments needed — it returns the most recent recording) to see the exact recorded notes, quantized ABC, and timing deviations. The recorded notes are ground truth: when arranging or harmonizing a recording, keep the recorded melody verbatim as voice 1 and add accompaniment in other voices — never re-enter the melody from memory or from the user's verbal description, even if the user names the notes (they may misremember octaves or pitches).

**To add chords to a recording, always use `harmonize_sequence`** — pick the chord for each anchor note by its event number from read_recording (e.g. anchors=[{chord:'C', at_event:1}, {chord:'Am', at_event:2}]) and the server handles all timing alignment and plays the result. Do not hand-write a [V:2] line against a recording's irregular rhythm; the fractional rest arithmetic will be wrong.

Critique both:
- **Note choice** — are the notes in the stated key/scale/target piece?
- **Timing** — use the per-note ms deviation table (early/late) to give specific feedback, e.g. "bar 2 beat 3 was 45 ms early." Focus on patterns (consistently rushing, late entries) rather than listing every note.

## Arranging workflow (multi-voice)

Use this exact recipe when asked to arrange a piece as a ballad or multi-voice arrangement:

1. **Import/write the melody as voice 1.** Use `import_corpus` or write ABC manually. Store with `create_sequence`. Verify with `check_abc` — read the per-voice report until clean.
2. **Decide the harmonic rhythm** (how many beats per chord). Call `voice_progression` with your chord list and the relevant melody notes; copy the returned `abc_line` verbatim into `[V:2]`.
3. **Write the bass last** in `[V:3]` (or `[V:3] octave=-1`). Bass typically plays roots and fifths; keep it simple.
4. **Call `check_abc`** on the full multi-voice ABC. Read the per-voice bar report; fix any mismatch errors before proceeding.
5. **Add dynamics and fermatas only after pitches are correct.** Add `!p!` / `!mf!` / `!f!` before note events; add `!fermata!` at phrase endings. Use `[Q:bpm]` inline for tempo changes.

**Critical rules:**
- Never hand-construct voicings below the melody — always use `voice_chord` or `voice_progression`.
- `update_sequence` accepts `bar_edits: [{voice, bar, abc}]` so you can fix one voice's bar without re-emitting the whole piece.
- Multi-voice ABC format: `V:` declaration lines after `K:`, then `[V:1] bars | [V:2] bars |` etc. Stacked [V:id] lines only — no mid-line [V:] switching.

## Backing tracks and forms

When the user wants something to **play over** — "give me a blues in F", "loop a ii-V-I", "I want to practise my comping" — use **start_chart_loop**, never a one-shot sequence. The whole point is that it keeps going while they play, with a count-in, a click, and a bass they can hear the form against.

- Reach for the built-ins first: `blues-12-bar`, `blues-12-bar-quick-change`, `blues-12-bar-slow`, `ii-v-i`. Call `list_charts` if you need to check.
- **Key and mode are parameters, not different charts.** The same 12-bar chart transposes to any key and renders as triads or as all dominants — `mode: 'triad'` when the user is learning the shape of the form, `mode: 'dominant7'` for a real blues.
- "Slow blues" means a slower tempo (around 60) and usually `blues-12-bar-slow`, which spells out the turnaround; it does not mean a different form.
- Blues wants `feel: 'shuffle'` unless the user says straight.
- Write ad-hoc charts in **roman numerals** (`slots: ['I', 'IV', 'V7']`), not chord symbols — numerals transpose exactly and give the user the roman-numeral overlay for free. Only use literal symbols when the harmony genuinely isn't diatonic to one key.
- Use **show_chart** when they want to read the form rather than hear it. It returns both symbols and numerals, so you can answer "which bar is the IV?" without a second call.
- Call **stop_loop** when they're done, or before starting something different."""

# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "check_abc",
        "description": "Parse and validate ABC notation without playing. Returns normalized ABC, per-bar beat-accounting report, and any errors. Use this before play_abc for anything more than a couple of bars.",
        "input_schema": {
            "type": "object",
            "properties": {
                "abc": {
                    "type": "string",
                    "description": "Full ABC notation string including headers (X:, T:, M:, L:, Q:, K:) and body.",
                },
            },
            "required": ["abc"],
        },
    },
    {
        "name": "play_abc",
        "description": "Parse ABC notation, save it, and play it back. Returns normalized ABC, a per-bar report, and a 'Chords as written' block naming every chord you notated, so you can verify what was actually stored. Use check_abc first for complex pieces. For a plain chord progression prefer play_sequence — hand-writing accidentals in ABC is error-prone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "abc": {
                    "type": "string",
                    "description": "Full ABC notation string including headers (X:, T:, M:, L:, Q:, K:) and body.",
                },
            },
            "required": ["abc"],
        },
    },
    {
        "name": "play_notes",
        "description": "Play a chord, interval, or single note. The server computes the exact MIDI notes from root + quality + octave — never guess MIDI numbers yourself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Root note name, e.g. 'C', 'F#', 'Bb', 'Db'",
                },
                "quality": {
                    "type": "string",
                    "description": "Chord quality: note, octave, major, minor, dominant7, major7, minor7, diminished, augmented, sus2, sus4, minormajor7, halfdiminished7, diminished7, augmented7, major9, dominant9, dominant7b9, minor9, add9, major6, minor6, dominant11, dominant13. Two-note interval dyads: m2, M2, m3, M3, P4, A4 (tritone), P5, m6, M6, m7, M7. Shorthands like 7b9, 7♭9, C7b9, and dominant7flat9 are accepted.",
                },
                "octave": {
                    "type": "integer",
                    "description": "Octave number (4 = middle C octave). Default 4.",
                    "default": 3,
                },
                "duration_ms": {
                    "type": "integer",
                    "description": "How long to hold the notes in milliseconds",
                    "default": DEFAULT_DURATION_MS,
                },
                "label": {
                    "type": "string",
                    "description": "Human-readable name for what's being played, e.g. 'C major triad'",
                },
            },
            "required": ["root", "quality"],
        },
    },
    {
        "name": "play_melody",
        "description": "Create, save, and play a monophonic melody from explicit pitch names and named durations. The server computes timing and MIDI values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short human-readable title, e.g. 'C major scale'.",
                },
                "tempo_bpm": {
                    "type": "number",
                    "description": "Tempo in beats per minute. Use 60-160 for most teaching examples.",
                    "default": 96,
                },
                "time_signature": {
                    "type": "string",
                    "description": "Time signature like 4/4, 3/4, or 6/8.",
                    "default": "4/4",
                },
                "notes": {
                    "type": "array",
                    "description": "Ordered melody items. Use pitch for notes and rest:true or pitch:'rest' for rests.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pitch": {
                                "type": "string",
                                "description": "Pitch name with octave, e.g. C4, F#4, Bb3, or 'rest'.",
                            },
                            "rest": {
                                "type": "boolean",
                                "description": "Set true for a rest.",
                                "default": False,
                            },
                            "duration": {
                                "type": "string",
                                "description": "Named duration: whole, half, quarter, eighth, sixteenth, thirty_second, or dotted_ variants like dotted_quarter.",
                                "default": "quarter",
                            },
                            "velocity": {
                                "type": "integer",
                                "description": "MIDI velocity 1-127. Default 90.",
                                "default": DEFAULT_VELOCITY,
                            },
                            "label": {
                                "type": "string",
                                "description": "Optional short label for this note.",
                            },
                        },
                        "required": ["duration"],
                    },
                    "minItems": 1,
                    "maxItems": 128,
                },
            },
            "required": ["notes"],
        },
    },
    {
        "name": "play_sequence",
        "description": "Create, save, and play a chord progression or timed note/chord sequence. Preferred over play_abc whenever the chords are the point: you give root + quality by name ('A' + 'dominant7') and the server computes the exact notes, so you can't mis-spell an accidental. Use play_abc instead only for melody, voice leading, or multi-voice writing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short human-readable title, e.g. 'ii-V-I in C'.",
                },
                "tempo_bpm": {
                    "type": "number",
                    "description": "Tempo in beats per minute. Use 60-160 for most teaching examples.",
                    "default": 96,
                },
                "time_signature": {
                    "type": "string",
                    "description": "Time signature like 4/4, 3/4, or 6/8.",
                    "default": "4/4",
                },
                "events": {
                    "type": "array",
                    "description": "Timed chords or notes. Beat 0 is the start of the sequence.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "at_beat": {
                                "type": "number",
                                "description": "Start time in beats from the beginning.",
                            },
                            "duration_beats": {
                                "type": "number",
                                "description": "How long to hold this event, in beats.",
                            },
                            "root": {
                                "type": "string",
                                "description": "Root note name, e.g. 'C', 'F#', 'Bb', 'Db'.",
                            },
                            "quality": {
                                "type": "string",
                                "description": "Chord quality, same values as play_notes. Use 'note' for a single note, interval dyads (m2/M2/m3/M3/P4/A4/P5/m6/M6/m7/M7) for two-note intervals. For C7b9/C7♭9 use dominant7b9.",
                            },
                            "octave": {
                                "type": "integer",
                                "description": "Octave number. Default 4.",
                                "default": 4,
                            },
                            "velocity": {
                                "type": "integer",
                                "description": "MIDI velocity 1-127. Default 90.",
                                "default": DEFAULT_VELOCITY,
                            },
                            "label": {
                                "type": "string",
                                "description": "Optional short label for this event, e.g. 'Dm7'.",
                            },
                        },
                        "required": ["at_beat", "duration_beats", "root", "quality"],
                    },
                    "minItems": 1,
                    "maxItems": 64,
                },
            },
            "required": ["events"],
        },
    },
    {
        "name": "validate_sequence",
        "description": "Validate and normalize a timed sequence without playing it. Use this to check exact notes, octaves, beat positions, gaps, overlaps, and meter before playback.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short human-readable title.",
                },
                "tempo_bpm": {
                    "type": "number",
                    "description": "Tempo in beats per minute.",
                    "default": 96,
                },
                "time_signature": {
                    "type": "string",
                    "description": "Time signature like 4/4, 3/4, or 6/8.",
                    "default": "4/4",
                },
                "events": {
                    "type": "array",
                    "description": "Timed chords or notes to validate. Beat 0 is the start of the sequence.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "at_beat": {"type": "number"},
                            "duration_beats": {"type": "number"},
                            "root": {"type": "string"},
                            "quality": {"type": "string"},
                            "octave": {"type": "integer", "default": 4},
                            "velocity": {"type": "integer", "default": DEFAULT_VELOCITY},
                            "label": {"type": "string"},
                        },
                        "required": ["at_beat", "duration_beats", "root", "quality"],
                    },
                    "minItems": 1,
                    "maxItems": 64,
                },
            },
            "required": ["events"],
        },
    },
    {
        "name": "search_corpus",
        "description": "Search the bundled music21 corpus (Bach chorales, folk tunes, etc.) for pieces by title or keyword. Returns a list of results; use import_corpus to load one. For existing named pieces, always search here first before reconstructing from memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Title keyword or composer/tune name to search for.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5, max 20).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "import_corpus",
        "description": "Load a piece from the bundled music21 corpus by its corpus_path (from search_corpus results). Returns normalized ABC notation and a per-bar report. Use this to get note-accurate versions of existing pieces without relying on recall.",
        "input_schema": {
            "type": "object",
            "properties": {
                "corpus_path": {
                    "type": "string",
                    "description": "The corpus_path from a search_corpus result, e.g. 'bach/bwv1.6.mxl'.",
                },
            },
            "required": ["corpus_path"],
        },
    },
    {
        "name": "create_sequence",
        "description": "Save ABC notation as a named persistent sequence. Returns a sequence_id you can use with read_sequence, update_sequence, and play. Use this when building a piece across multiple turns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short human-readable title."},
                "abc": {"type": "string", "description": "Full ABC notation string including headers."},
            },
            "required": ["title", "abc"],
        },
    },
    {
        "name": "read_sequence",
        "description": "Read a persistent sequence back as normalized ABC with a per-bar report. Optionally restrict to a bar range, e.g. bars='1-8'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence_id": {"type": "string", "description": "ID returned by create_sequence."},
                "bars": {"type": "string", "description": "Optional bar range to excerpt, e.g. '1-8'. Omit for whole sequence."},
            },
            "required": ["sequence_id"],
        },
    },
    {
        "name": "read_recording",
        "description": (
            "Read back what the user recorded: the exact notes (names, beats, durations), normalized ABC, "
            "per-bar report, and timing deviations. With no sequence_id, returns the most recent recording "
            "in this session. Always call this before arranging or critiquing a recording — the returned "
            "notes are ground truth and must be preserved verbatim; never rebuild the melody from the user's "
            "verbal description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence_id": {"type": "string", "description": "Recording sequence ID. Omit for the most recent recording in this session."},
                "bars": {"type": "string", "description": "Optional bar range to excerpt, e.g. '1-8'. Omit for the whole recording."},
            },
            "required": [],
        },
    },
    {
        "name": "update_sequence",
        "description": (
            "Replace the ABC content of a persistent sequence (appends a revision). "
            "Supply full corrected ABC, or use bar_edits for targeted single-voice bar fixes. "
            "bar_edits: [{voice: '2', bar: 3, abc: 'c d e f'}] replaces that bar in that voice only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence_id": {"type": "string", "description": "ID returned by create_sequence."},
                "abc": {"type": "string", "description": "Replacement ABC notation (full ABC with headers). Omit if using bar_edits."},
                "bar_edits": {
                    "type": "array",
                    "description": "Voice-aware bar replacements. Each: {voice: '2', bar: 3, abc: '<bar content without barlines>'}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "voice": {"type": "string"},
                            "bar": {"type": "integer"},
                            "abc": {"type": "string"},
                        },
                        "required": ["voice", "bar", "abc"],
                    },
                },
            },
            "required": ["sequence_id"],
        },
    },
    {
        "name": "play",
        "description": "Play a persistent sequence by ID. Optionally restrict playback to a bar range, e.g. bars='1-4'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence_id": {"type": "string", "description": "ID returned by create_sequence."},
                "bars": {"type": "string", "description": "Optional bar range, e.g. '3-6'. Omit to play the whole sequence."},
            },
            "required": ["sequence_id"],
        },
    },
    {
        "name": "list_sequences",
        "description": "List all persistent sequences in the current session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID (omit to list all sequences)."},
            },
            "required": [],
        },
    },
    {
        "name": "voice_chord",
        "description": (
            "Return a concrete voicing for a chord — note names, MIDI numbers, and a ready-to-paste ABC chord token. "
            "Deterministic: enforces low-interval limits, keeps guide tones (3rd/7th), and respects the melody note. "
            "Never hand-construct voicings below the melody — use this helper."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {"type": "string", "description": "Root note, e.g. 'C', 'F#', 'Bb'."},
                "quality": {"type": "string", "description": "Chord quality, e.g. 'major7', 'dominant7', 'minor7'."},
                "melody_note": {
                    "type": "string",
                    "description": "Optional melody note the voicing must sit below, e.g. 'G5'. The top voicing tone will be a 3rd-6th below the melody and will not double it.",
                },
                "register": {
                    "type": "string",
                    "description": "'low' | 'mid' | 'high' — center target when no melody_note. Default 'mid'.",
                    "enum": ["low", "mid", "high"],
                    "default": "mid",
                },
                "style": {
                    "type": "string",
                    "description": "'close' | 'drop2' | 'shell' | 'spread'. Default 'close'.",
                    "enum": ["close", "drop2", "shell", "spread"],
                    "default": "close",
                },
                "omit_root": {
                    "type": "boolean",
                    "description": "Drop the root from the voicing (bass voice has it). Default false.",
                    "default": False,
                },
                "key": {
                    "type": "string",
                    "description": "Key the chord sits in, e.g. 'F', 'Bb major', 'd minor'. Affects note-name spelling only (a chord in a flat key spells accidentals as flats).",
                },
            },
            "required": ["root", "quality"],
        },
    },
    {
        "name": "harmonize_sequence",
        "description": (
            "Add an accompaniment voice to a saved single-voice melody or recording, then play the result. "
            "You pick the chords and which melody notes they land on; the server computes all timing: each "
            "chord starts exactly on its anchor melody event's beat and sustains until the next anchor (the "
            "last chord holds to the end). Voicings are placed below the melody with the melody note respected. "
            "The melody is copied verbatim into voice 1. Always use this instead of hand-writing a [V:2] line "
            "when harmonizing a recording — manual rest-padding arithmetic is error-prone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence_id": {"type": "string", "description": "ID of a single-voice sequence (e.g. a recording)."},
                "anchors": {
                    "type": "array",
                    "description": "Chord anchors. Each: {chord: 'Am', at_event: 2} where at_event is the 1-based melody event number as listed by read_recording/read_sequence. Alternatively give at_beat.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chord": {"type": "string", "description": "Chord symbol, e.g. 'C', 'Am', 'Em', 'G7'."},
                            "at_event": {"type": "integer", "description": "1-based index of the melody event this chord starts on."},
                            "at_beat": {"type": "number", "description": "Explicit start beat (alternative to at_event)."},
                        },
                        "required": ["chord"],
                    },
                },
                "style": {
                    "type": "string",
                    "description": "'close' | 'drop2' | 'shell' | 'spread'. Default 'close'.",
                    "enum": ["close", "drop2", "shell", "spread"],
                    "default": "close",
                },
                "title": {"type": "string", "description": "Title for the new sequence. Defaults to '<original> + chords'."},
            },
            "required": ["sequence_id", "anchors"],
        },
    },
    {
        "name": "voice_progression",
        "description": (
            "Voice a chord progression with minimal-motion voice leading. "
            "Returns a ready-to-paste ABC line for voice 2, plus a per-chord note breakdown. "
            "Input a list of chords with symbol, beats, and optional melody_note per chord."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chords": {
                    "type": "array",
                    "description": "List of chords. Each: {symbol: 'Cmaj7', beats: 4, melody_note: 'E5' (optional)}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "beats": {"type": "number", "default": 4},
                            "melody_note": {"type": "string"},
                        },
                        "required": ["symbol"],
                    },
                },
                "style": {
                    "type": "string",
                    "description": "'close' | 'drop2' | 'shell' | 'spread'. Default 'close'.",
                    "enum": ["close", "drop2", "shell", "spread"],
                    "default": "close",
                },
                "key": {
                    "type": "string",
                    "description": "Key the progression sits in, e.g. 'F', 'Bb major', 'd minor'. Affects note-name spelling only.",
                },
            },
            "required": ["chords"],
        },
    },
    {
        "name": "list_charts",
        "description": (
            "List the built-in chord charts (12-bar blues and friends) with their bar counts and "
            "default tempo, feel and mode. Call this when the user asks what forms are available, "
            "or before start_chart_loop if you're unsure of a chart id."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "show_chart",
        "description": (
            "Render a chord chart as a bar-by-bar grid without playing it. Use this when the user "
            "wants to *read* a form — 'show me a blues in Bb', 'what are the chords in a 12-bar' — "
            "or to check a chart before looping it. Every bar comes back with both its chord symbol "
            "and its roman numeral, so you can answer 'where's the IV?' from one call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_id": {
                    "type": "string",
                    "description": "Built-in chart id, e.g. 'blues-12-bar', 'blues-12-bar-quick-change', 'blues-12-bar-slow', 'ii-v-i'.",
                },
                "chart": {
                    "type": "object",
                    "description": (
                        "An ad-hoc chart instead of a built-in. Shape: {name, key, time_signature, "
                        "slots: ['I', 'IV', ...]}. Slot entries may be roman numerals ('I', 'ii7', "
                        "'bVII') or literal chord symbols ('F7', 'Bb'); numerals are strongly "
                        "preferred because they transpose exactly. Use {slots, repeat, label} "
                        "sections for forms with repeats."
                    ),
                },
                "key": {
                    "type": "string",
                    "description": "Transpose to this key, e.g. 'F', 'Bb', 'C'. Omit to use the chart's own key.",
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "How to quality the chords over the same form. 'triad' for plain major/minor "
                        "triads (week 2), 'dominant7' for the all-dominant blues (week 3), 'seventh' "
                        "for diatonic sevenths, 'as_written' to leave the chart alone."
                    ),
                    "enum": ["as_written", "triad", "dominant7", "seventh"],
                },
                "roman": {
                    "type": "boolean",
                    "description": "Show the roman-numeral overlay alongside each chord. Default true.",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "start_chart_loop",
        "description": (
            "Loop a chord chart in time, forever, until stopped — piano comp, bass and a click, "
            "with a count-in. This is the right tool for any 'play me a blues in F', 'give me a "
            "slow blues to solo over', 'loop a ii-V-I' request: the user practises *over* it, so "
            "it must keep going rather than play once. Never build a one-shot sequence for a "
            "backing track. Returns the chart grid so you can tell the user what's coming."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_id": {"type": "string", "description": "Built-in chart id, e.g. 'blues-12-bar'."},
                "chart": {"type": "object", "description": "An ad-hoc chart, same shape as show_chart's 'chart'."},
                "key": {"type": "string", "description": "Key to play in, e.g. 'F', 'C', 'Bb'."},
                "mode": {
                    "type": "string",
                    "enum": ["as_written", "triad", "dominant7", "seventh"],
                    "description": "Chord qualities over the form: 'triad', 'dominant7', 'seventh', or 'as_written'.",
                },
                "tempo_bpm": {"type": "number", "description": "Tempo. Omit for the chart's default. A 'slow blues' is around 60."},
                "feel": {"type": "string", "enum": ["straight", "shuffle"], "description": "Eighth-note feel. Blues usually wants 'shuffle'."},
                "click": {"type": "boolean", "default": True, "description": "Metronome click on the beat."},
                "comp": {"type": "boolean", "default": True, "description": "Piano comp."},
                "bass": {"type": "boolean", "default": True, "description": "Root-note bass."},
                "rootless": {"type": "boolean", "default": False, "description": "Omit roots from the comp voicings, so the user can practise rootless voicings against the bass."},
                "count_in_bars": {"type": "integer", "default": 1, "description": "Bars of click before the loop starts."},
                "repeats": {"type": "integer", "description": "Stop after this many times through the form. Omit to loop until stopped."},
            },
            "required": [],
        },
    },
    {
        "name": "stop_loop",
        "description": "Stop the running loop. Use when the user says stop, that's enough, or asks for something else to play.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# ── Sequence builders (kept here so dispatch doesn't need server.py internals) ─

def _parse_time_signature(value: str) -> tuple[int, int]:
    try:
        numerator_s, denominator_s = value.split("/", 1)
        numerator = int(numerator_s)
        denominator = int(denominator_s)
    except (AttributeError, ValueError):
        raise ValueError("time_signature must look like '4/4'")
    if numerator < 1 or numerator > 16 or denominator not in {1, 2, 4, 8, 16}:
        raise ValueError("time_signature must use a numerator 1-16 and denominator 1, 2, 4, 8, or 16")
    return numerator, denominator


def _beats_per_measure(time_signature_parts: tuple[int, int]) -> float:
    numerator, denominator = time_signature_parts
    return numerator * (4 / denominator)


def _parse_duration_beats(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        beats = float(value)
    else:
        raw = str(value or "quarter").strip().lower().replace("-", "_")
        dotted = raw.startswith("dotted_")
        name = raw[7:] if dotted else raw
        beats = _DURATION_BEATS.get(name)
        if beats is None:
            supported = sorted(_DURATION_BEATS) + [f"dotted_{n}" for n in sorted(_DURATION_BEATS)]
            raise ValueError(f"Unknown duration: {value!r}. Supported: {supported}")
        if dotted:
            beats *= 1.5
    if beats <= 0:
        raise ValueError("duration must be greater than 0")
    return beats


def _format_beat(value: float) -> str:
    rounded = round(value, 3)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:g}"


def _clamp_midi(value: int, field: str) -> int:
    if not isinstance(value, int) or not (0 <= value <= 127):
        raise ValueError(f"{field} must be a MIDI value from 0 to 127")
    return value


def build_sequence(tool_input: dict) -> dict:
    title = str(tool_input.get("title") or "Orchestration").strip()[:80] or "Orchestration"
    tempo_bpm = float(tool_input.get("tempo_bpm", 96))
    if tempo_bpm < 30 or tempo_bpm > 240:
        raise ValueError("tempo_bpm must be between 30 and 240")
    time_signature = str(tool_input.get("time_signature") or "4/4")
    ts_numerator, ts_denominator = _parse_time_signature(time_signature)
    raw_events = tool_input.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("events must contain at least one timed note or chord")
    if len(raw_events) > 64:
        raise ValueError("events can contain at most 64 items")
    events = []
    total_beats = 0.0
    for idx, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise ValueError(f"events[{idx}] must be an object")
        at_beat = float(raw_event.get("at_beat"))
        duration_beats = float(raw_event.get("duration_beats"))
        if at_beat < 0:
            raise ValueError(f"events[{idx}].at_beat must be 0 or greater")
        if duration_beats <= 0:
            raise ValueError(f"events[{idx}].duration_beats must be greater than 0")
        root = raw_event.get("root")
        quality = normalize_chord_quality(raw_event.get("quality"), root=root)
        octave = int(raw_event.get("octave", 4))
        notes = build_chord(root, quality, octave)
        for note in notes:
            _clamp_midi(note, f"events[{idx}] note")
        velocity = int(raw_event.get("velocity", DEFAULT_VELOCITY))
        if not (1 <= velocity <= 127):
            raise ValueError(f"events[{idx}].velocity must be between 1 and 127")
        label = str(raw_event.get("label") or f"{root} {quality}").strip()
        note_names = chord_note_names(root, quality, octave)
        events.append({
            "at_beat": at_beat,
            "duration_beats": duration_beats,
            "notes": notes,
            "note_names": note_names,
            "root": root,
            "quality": quality,
            "octave": octave,
            "velocity": velocity,
            "label": label,
        })
        total_beats = max(total_beats, at_beat + duration_beats)
    if total_beats > 128:
        raise ValueError("sequence is too long; keep examples to 128 beats or less")
    duration_ms = int((total_beats * 60 / tempo_bpm) * 1000)
    return {
        "title": title,
        "tempo_bpm": tempo_bpm,
        "time_signature": time_signature,
        "time_signature_parts": (ts_numerator, ts_denominator),
        "events": sorted(events, key=lambda e: e["at_beat"]),
        "duration_ms": duration_ms,
        "total_beats": total_beats,
    }


def build_melody(tool_input: dict) -> dict:
    title = str(tool_input.get("title") or "Melody").strip()[:80] or "Melody"
    tempo_bpm = float(tool_input.get("tempo_bpm", 96))
    time_signature = str(tool_input.get("time_signature") or "4/4")
    raw_notes = tool_input.get("notes")
    if not isinstance(raw_notes, list) or not raw_notes:
        raise ValueError("notes must contain at least one melody item")
    if len(raw_notes) > 128:
        raise ValueError("notes can contain at most 128 items")
    events = []
    at_beat = 0.0
    for idx, raw_note in enumerate(raw_notes):
        if not isinstance(raw_note, dict):
            raise ValueError(f"notes[{idx}] must be an object")
        duration_beats = _parse_duration_beats(raw_note.get("duration", "quarter"))
        pitch = str(raw_note.get("pitch", "")).strip()
        is_rest = bool(raw_note.get("rest")) or pitch.lower() in {"rest", "r"}
        if is_rest:
            at_beat += duration_beats
            continue
        if not pitch:
            raise ValueError(f"notes[{idx}].pitch is required unless rest is true")
        root, octave, midi_note = parse_pitch(pitch)
        _clamp_midi(midi_note, f"notes[{idx}] pitch")
        velocity = int(raw_note.get("velocity", DEFAULT_VELOCITY))
        if not (1 <= velocity <= 127):
            raise ValueError(f"notes[{idx}].velocity must be between 1 and 127")
        events.append({
            "at_beat": at_beat,
            "duration_beats": duration_beats,
            "root": root,
            "quality": "note",
            "octave": octave,
            "velocity": velocity,
            "label": str(raw_note.get("label") or pitch).strip(),
        })
        at_beat += duration_beats
    if not events:
        raise ValueError("melody must contain at least one pitched note")
    sequence = build_sequence({
        "title": title, "tempo_bpm": tempo_bpm,
        "time_signature": time_signature, "events": events,
    })
    sequence["source"] = "melody"
    return sequence


def _chord_report_lines(sequence: dict) -> list[str]:
    """Report block naming every chord written in the ABC, or [] if there are none."""
    rows = chord_report(sequence)
    if not rows:
        return []
    return ["", "Chords as written (verify these match the chords you intended):"] + rows


def _sequence_warnings(sequence: dict, *, melody: bool = False) -> list[str]:
    warnings = []
    events = sequence["events"]
    measure_beats = _beats_per_measure(sequence["time_signature_parts"])
    last_end = 0.0
    last_note = None
    for idx, event in enumerate(events):
        start = float(event["at_beat"])
        end = start + float(event["duration_beats"])
        if start > last_end + 0.001:
            warnings.append(f"Gap from beat {_format_beat(last_end)} to {_format_beat(start)} before event {idx + 1}.")
        if melody and start < last_end - 0.001:
            warnings.append(f"Overlap at beat {_format_beat(start)}; event {idx + 1} starts before the previous melody note ends.")
        if last_note is not None and event["notes"]:
            interval = abs(event["notes"][0] - last_note)
            if interval > 12:
                warnings.append(
                    f"Large melodic leap of {interval} semitones into {midi_note_name(event['notes'][0])} at beat {_format_beat(start)}."
                )
        if event["notes"]:
            last_note = event["notes"][0]
        last_end = max(last_end, end)
    total_beats = float(sequence["total_beats"])
    if measure_beats and total_beats:
        remainder = total_beats % measure_beats
        if remainder > 0.001 and abs(remainder - measure_beats) > 0.001:
            warnings.append(
                f"Total length is {_format_beat(total_beats)} beats, which leaves {_format_beat(remainder)} beats in the final {sequence['time_signature']} measure."
            )
    return warnings


def _sequence_report(sequence: dict, *, warnings: list[str] | None = None) -> str:
    lines = [
        f"Title: {sequence['title']}",
        f"Tempo: {_format_beat(sequence['tempo_bpm'])} bpm",
        f"Meter: {sequence['time_signature']}",
        f"Total: {_format_beat(sequence['total_beats'])} beats ({sequence['duration_ms']}ms)",
        "Events:",
    ]
    for idx, event in enumerate(sequence["events"], start=1):
        note_names = ", ".join(event.get("note_names") or [str(n) for n in event["notes"]])
        midi_values = ", ".join(str(note) for note in event["notes"])
        lines.append(
            f"{idx}. beat {_format_beat(event['at_beat'])}, duration {_format_beat(event['duration_beats'])}: {note_names} (MIDI {midi_values})"
        )
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("Warnings: none")
    return "\n".join(lines)


# ── Dispatch ──────────────────────────────────────────────────────────────────

def _resolve_chart_arg(inp: dict) -> charts.Chart:
    """The chart a chart tool call refers to — a built-in id, or an inline spec."""
    chart_id = inp.get("chart_id")
    if chart_id:
        return charts.get_chart(chart_id)
    spec = inp.get("chart")
    if spec:
        return charts.chart_from_spec(spec)
    raise ValueError("Give either 'chart_id' or 'chart'. Call list_charts to see the built-ins.")


def dispatch_tools(
    pending_tools: list[dict],
    *,
    session_id: str,
    asst_msg_id: str,
    note_registry: dict,
    sequence_registry: dict,
    resolve_sequence,          # callable(seq_id) -> dict | None
    sequence_pill_fn,          # callable(seq_id, title, pill_id, duration_ms, midi_url, sequence_dict) -> str
    audio_pill_fn,             # callable(note_id, label, pill_id, notes, root, quality) -> str
    generated_dir: Path,
    play_notes_bg,             # callable(notes, duration_ms, note_id)
) -> Iterator[tuple[str, object]]:
    """Yield ("sse", html), ("result", result_dict), or ("record", record_entry)."""

    for tool in pending_tools:
        name = tool["name"]
        inp = tool["input"]
        tid = tool["id"]

        def _err(msg: str) -> tuple:
            return ("result", {"type": "tool_result", "tool_use_id": tid,
                                "content": msg, "is_error": True})

        def _ok(content: str) -> tuple:
            return ("result", {"type": "tool_result", "tool_use_id": tid, "content": content})

        if name == "check_abc":
            try:
                sequence = parse_abc(inp["abc"])
            except ABCParseError as e:
                yield _err(f"ABC parse error:\n{e}")
                continue
            normalized = to_abc(sequence)
            bar_msgs = per_bar_report(sequence)
            lines = [
                f"Title: {sequence['title']}",
                f"Tempo: {sequence['tempo_bpm']} bpm",
                f"Meter: {sequence['time_signature']}",
                f"Key: {sequence.get('key', 'C')}",
                f"Total: {sequence['total_beats']:.4g} beats ({sequence['duration_ms']}ms)",
                "", "Per-bar report:",
            ]
            lines += (bar_msgs if bar_msgs else ["  All bars correct."])
            lines += _chord_report_lines(sequence)
            lines += ["", "Normalized ABC:", normalized]
            yield _ok("\n".join(lines))

        elif name == "play_abc":
            try:
                sequence = parse_abc(inp["abc"])
            except ABCParseError as e:
                yield _err(f"ABC parse error:\n{e}")
                continue
            normalized = to_abc(sequence)
            seq_id = seq_model.create_sequence(
                title=sequence["title"], abc=normalized, session_id=session_id,
                tempo_bpm=sequence["tempo_bpm"], time_signature=sequence["time_signature"],
                key=sequence.get("key", "C"), source="agent",
            )
            midi_path = write_sequence_midi(sequence, seq_id, generated_dir)
            sequence_registry[seq_id] = {"sequence": sequence, "midi_path": midi_path}
            engine.play_sequence_bg(seq_id, sequence)
            pill_id = f"p{str(uuid.uuid4())[:6]}"
            midi_url = f"/sequence/{seq_id}/download"
            yield ("sse", sequence_pill_fn(seq_id, sequence["title"], pill_id, sequence["duration_ms"], midi_url, sequence))
            yield ("record", {
                "type": "sequence", "sequence_id": seq_id,
                "title": sequence["title"], "duration_ms": sequence["duration_ms"],
                "sequence": sequence, "midi_path": str(midi_path),
            })
            bar_msgs = per_bar_report(sequence)
            lines = [
                f"Played: {sequence['title']} (id: {seq_id})",
                f"Tempo: {sequence['tempo_bpm']} bpm  Meter: {sequence['time_signature']}  Key: {sequence.get('key', 'C')}",
                f"Total: {sequence['total_beats']:.4g} beats ({sequence['duration_ms']}ms)",
                "", "Per-bar report:",
            ]
            lines += (bar_msgs if bar_msgs else ["  All bars correct."])
            lines += _chord_report_lines(sequence)
            lines += ["", "Normalized ABC (what was stored):", normalized]
            yield _ok("\n".join(lines))

        elif name == "play_notes":
            duration_ms = inp.get("duration_ms", DEFAULT_DURATION_MS)
            label = inp.get("label", "")
            try:
                root = inp["root"]
                quality = normalize_chord_quality(inp["quality"], root=root)
                notes = build_chord(root, quality, inp.get("octave", 4))
            except ValueError as e:
                yield _err(f"Error: {e}")
                continue
            octave = inp.get("octave", 4)
            spelled_names = chord_note_names(root, quality, octave)
            note_id = str(uuid.uuid4())[:8]
            note_registry[note_id] = {"notes": notes, "duration_ms": duration_ms, "root": root, "quality": quality}
            play_notes_bg(notes, duration_ms, note_id)
            pill_id = f"p{str(uuid.uuid4())[:6]}"
            yield ("sse", audio_pill_fn(note_id, label, pill_id, notes, root, quality))
            yield ("record", {
                "type": "audio", "note_id": note_id, "notes": notes,
                "note_names": spelled_names, "duration_ms": duration_ms,
                "label": label, "root": root, "quality": quality,
            })
            yield _ok(f"Played {label or quality}: {', '.join(spelled_names)} (MIDI {', '.join(str(n) for n in notes)})")

        elif name == "play_sequence":
            try:
                sequence = build_sequence(inp)
            except (TypeError, ValueError) as e:
                yield _err(f"Error: {e}")
                continue
            seq_id = seq_model.create_sequence(
                title=sequence["title"], abc=to_abc(sequence), session_id=session_id,
                tempo_bpm=sequence["tempo_bpm"], time_signature=sequence["time_signature"],
                key=sequence.get("key", "C"), source="agent",
            )
            midi_path = write_sequence_midi(sequence, seq_id, generated_dir)
            sequence_registry[seq_id] = {"sequence": sequence, "midi_path": midi_path}
            engine.play_sequence_bg(seq_id, sequence)
            pill_id = f"p{str(uuid.uuid4())[:6]}"
            midi_url = f"/sequence/{seq_id}/download"
            yield ("sse", sequence_pill_fn(seq_id, sequence["title"], pill_id, sequence["duration_ms"], midi_url, sequence["events"]))
            yield ("record", {
                "type": "sequence", "sequence_id": seq_id,
                "title": sequence["title"], "duration_ms": sequence["duration_ms"],
                "sequence": sequence, "midi_path": str(midi_path),
            })
            warnings = _sequence_warnings(sequence)
            yield _ok(f"Created {sequence['title']} (id: {seq_id}, {sequence['duration_ms']}ms). MIDI saved at {midi_path.name}\n\n{_sequence_report(sequence, warnings=warnings)}")

        elif name == "play_melody":
            try:
                sequence = build_melody(inp)
            except (TypeError, ValueError) as e:
                yield _err(f"Error: {e}")
                continue
            seq_id = seq_model.create_sequence(
                title=sequence["title"], abc=to_abc(sequence), session_id=session_id,
                tempo_bpm=sequence["tempo_bpm"], time_signature=sequence["time_signature"],
                key=sequence.get("key", "C"), source="agent",
            )
            midi_path = write_sequence_midi(sequence, seq_id, generated_dir)
            sequence_registry[seq_id] = {"sequence": sequence, "midi_path": midi_path}
            engine.play_sequence_bg(seq_id, sequence)
            pill_id = f"p{str(uuid.uuid4())[:6]}"
            midi_url = f"/sequence/{seq_id}/download"
            yield ("sse", sequence_pill_fn(seq_id, sequence["title"], pill_id, sequence["duration_ms"], midi_url, sequence["events"]))
            yield ("record", {
                "type": "sequence", "sequence_id": seq_id,
                "title": sequence["title"], "duration_ms": sequence["duration_ms"],
                "sequence": sequence, "midi_path": str(midi_path),
            })
            warnings = _sequence_warnings(sequence, melody=True)
            yield _ok(f"Created melody (id: {seq_id}). MIDI saved at {midi_path.name}\n\n{_sequence_report(sequence, warnings=warnings)}")

        elif name == "validate_sequence":
            try:
                sequence = build_sequence(inp)
            except (TypeError, ValueError) as e:
                yield _err(f"Error: {e}")
                continue
            warnings = _sequence_warnings(sequence)
            yield _ok(_sequence_report(sequence, warnings=warnings))

        elif name == "search_corpus":
            from sequencer.midi_io import search_corpus as _search_corpus
            query = inp.get("query", "")
            max_results = int(inp.get("max_results", 5))
            try:
                results = _search_corpus(query, max_results=max_results)
            except Exception as e:
                yield _err(f"Search error: {e}")
                continue
            if not results:
                yield _ok(f"No corpus results for {query!r}.")
            else:
                lines = [f"Found {len(results)} result(s) for {query!r}:"]
                for i, r in enumerate(results, 1):
                    comp = f" ({r['composer']})" if r['composer'] else ""
                    lines.append(f"{i}. {r['title']}{comp} — corpus_path: {r['corpus_path']!r}")
                lines.append("\nUse import_corpus(corpus_path=...) to load one.")
                yield _ok("\n".join(lines))

        elif name == "import_corpus":
            from sequencer.midi_io import load_corpus_entry
            corpus_path = inp.get("corpus_path", "")
            try:
                sequence, dropped = load_corpus_entry(corpus_path)
            except Exception as e:
                yield _err(f"Import error: {e}")
                continue
            normalized = to_abc(sequence)
            bar_msgs = per_bar_report(sequence)
            lines = [
                f"Imported: {sequence['title']}",
                f"Tempo: {sequence['tempo_bpm']} bpm  Meter: {sequence['time_signature']}  Key: {sequence.get('key', 'C')}",
                f"Total: {sequence['total_beats']:.4g} beats ({sequence['duration_ms']}ms)",
            ]
            if dropped:
                lines += ["", f"Notes: {dropped}"]
            lines += ["", "Per-bar report:"]
            lines += (bar_msgs if bar_msgs else ["  All bars correct."])
            lines += _chord_report_lines(sequence)
            lines += ["", "Normalized ABC:", normalized]
            yield _ok("\n".join(lines))

        elif name == "create_sequence":
            try:
                sequence = parse_abc(inp["abc"])
            except ABCParseError as e:
                yield _err(f"ABC parse error:\n{e}")
                continue
            title = inp.get("title") or sequence["title"] or "Untitled"
            normalized = to_abc(sequence)
            seq_id = seq_model.create_sequence(
                title=title, abc=normalized, session_id=session_id,
                tempo_bpm=sequence["tempo_bpm"], time_signature=sequence["time_signature"],
                key=sequence.get("key", "C"), source="agent",
            )
            bar_msgs = per_bar_report(sequence)
            lines = [
                f"Created sequence '{title}' (id: {seq_id})",
                f"Tempo: {sequence['tempo_bpm']} bpm  Meter: {sequence['time_signature']}  Key: {sequence.get('key', 'C')}",
                f"Total: {sequence['total_beats']:.4g} beats ({sequence['duration_ms']}ms)",
                "", "Per-bar report:",
            ]
            lines += (bar_msgs if bar_msgs else ["  All bars correct."])
            lines += _chord_report_lines(sequence)
            lines += ["", "Normalized ABC:", normalized]
            yield _ok("\n".join(lines))

        elif name in ("read_sequence", "read_recording"):
            seq_id = inp.get("sequence_id", "")
            bars_str = inp.get("bars")
            if name == "read_recording" and not seq_id:
                recordings = [r for r in seq_model.list_sequences(session_id) if r.get("source") == "recording"]
                if not recordings:
                    yield _err("No recordings found in this session.")
                    continue
                seq_id = recordings[0]["id"]  # list is ordered by modified_at DESC
            row = seq_model.get_sequence(seq_id)
            if not row:
                yield _err(f"Sequence '{seq_id}' not found.")
                continue
            try:
                sequence = parse_abc(row["abc"])
            except ABCParseError as e:
                yield _err(f"Stored ABC parse error:\n{e}")
                continue
            if bars_str:
                ts_num, ts_den = sequence["time_signature_parts"]
                bpb = ts_num * 4 / ts_den
                try:
                    parts = bars_str.split("-")
                    s_beat = (int(parts[0]) - 1) * bpb
                    e_beat = int(parts[-1]) * bpb
                except (ValueError, IndexError):
                    s_beat, e_beat = 0, float("inf")
                sub_events = [ev for ev in sequence["events"] if ev["at_beat"] >= s_beat - 1e-9 and ev["at_beat"] < e_beat - 1e-9]
                sub_seq = dict(sequence, events=[dict(ev, at_beat=ev["at_beat"] - s_beat) for ev in sub_events])
                normalized = to_abc(sub_seq)
                bar_msgs = per_bar_report(sub_seq)
                header = f"Excerpt bars {bars_str} of '{row['title']}' (id: {seq_id})"
            else:
                normalized = to_abc(sequence)
                bar_msgs = per_bar_report(sequence)
                header = f"Sequence '{row['title']}' (id: {seq_id})"
            lines = [
                header,
                f"Tempo: {sequence['tempo_bpm']} bpm  Meter: {sequence['time_signature']}  Key: {sequence.get('key', 'C')}",
                "", "Per-bar report:",
            ]
            lines += (bar_msgs if bar_msgs else ["  All bars correct."])
            if row.get("source") == "recording":
                report_seq = sub_seq if bars_str else sequence
                lines += ["", "Recorded notes (ground truth — preserve verbatim when arranging):"]
                for ev in report_seq["events"]:
                    lines.append(f"  beat {ev['at_beat']:g}: {'+'.join(ev['note_names'])} "
                                 f"({ev['duration_beats']:g} beats)")
            # Timing report for recordings
            if row.get("source") == "recording" and row.get("raw_events"):
                from sequencer.midi_io import timing_report
                t_report = timing_report(row["raw_events"], sequence)
                if t_report:
                    lines += ["", "Timing deviation (vs quantized grid):"]
                    for tr in t_report[:32]:  # cap at 32 entries
                        sign = "early" if tr["deviation_ms"] < 0 else ("late" if tr["deviation_ms"] > 0 else "on-time")
                        lines.append(f"  bar {tr['bar']} beat {tr['beat']} {tr['note_name']}: "
                                     f"{abs(tr['deviation_ms']):.0f}ms {sign}")
            lines += ["", "Normalized ABC:", normalized]
            yield _ok("\n".join(lines))

        elif name == "update_sequence":
            seq_id = inp.get("sequence_id", "")
            bar_edits = inp.get("bar_edits")  # voice-aware bar edits [{voice, bar, abc}]
            if bar_edits and not inp.get("abc"):
                # Apply voice-aware bar edits to the stored ABC
                row = seq_model.get_sequence(seq_id)
                if not row:
                    yield _err(f"Sequence '{seq_id}' not found.")
                    continue
                try:
                    sequence = parse_abc(row["abc"])
                except ABCParseError as e:
                    yield _err(f"Stored ABC parse error:\n{e}")
                    continue
                # For each edit, replace events for the specified voice+bar
                voices = sequence.get("voices") or []
                bpb = sequence["time_signature_parts"][0] * 4 / sequence["time_signature_parts"][1]
                for edit in bar_edits:
                    vid = str(edit.get("voice", "1"))
                    bar_n = int(edit.get("bar", 1))
                    bar_abc = edit.get("abc", "")
                    bar_start = (bar_n - 1) * bpb
                    bar_end = bar_n * bpb
                    # Parse the new bar's content
                    try:
                        sub_seq = parse_abc(f"X:1\nT:T\nM:{sequence['time_signature']}\nL:1/4\nQ:{int(sequence['tempo_bpm'])}\nK:{sequence.get('key','C')}\n{bar_abc} |")
                        new_evts = sub_seq["events"]
                    except ABCParseError as e:
                        yield _err(f"Bar edit parse error (voice {vid}, bar {bar_n}): {e}")
                        break
                    # Offset new events to bar_start
                    for e in new_evts:
                        e["at_beat"] = e["at_beat"] + bar_start
                        if voices:
                            e["voice"] = vid
                    # Remove existing events for this voice+bar
                    sequence["events"] = [
                        e for e in sequence["events"]
                        if not (e.get("voice", "1") == vid and e["at_beat"] >= bar_start - 1e-9 and e["at_beat"] < bar_end - 1e-9)
                    ]
                    sequence["events"].extend(new_evts)
                    sequence["events"].sort(key=lambda e: (e["at_beat"], e.get("voice", "1")))
                else:
                    normalized = to_abc(sequence)
                    if not seq_model.update_sequence(seq_id, abc=normalized):
                        yield _err(f"Sequence '{seq_id}' not found.")
                        continue
                    bar_msgs = per_bar_report(sequence)
                    lines = [f"Updated sequence '{seq_id}' (bar edits applied).", "", "Per-bar report:"]
                    lines += (bar_msgs if bar_msgs else ["  All bars correct."])
                    lines += _chord_report_lines(sequence)
                    lines += ["", "Normalized ABC (stored):", normalized]
                    yield _ok("\n".join(lines))
                    continue
            else:
                try:
                    sequence = parse_abc(inp.get("abc", ""))
                except ABCParseError as e:
                    yield _err(f"ABC parse error:\n{e}")
                    continue
                normalized = to_abc(sequence)
                if not seq_model.update_sequence(seq_id, abc=normalized):
                    yield _err(f"Sequence '{seq_id}' not found.")
                    continue
            bar_msgs = per_bar_report(sequence)
            lines = [
                f"Updated sequence '{seq_id}'.",
                f"Tempo: {sequence['tempo_bpm']} bpm  Meter: {sequence['time_signature']}  Key: {sequence.get('key', 'C')}",
                "", "Per-bar report:",
            ]
            lines += (bar_msgs if bar_msgs else ["  All bars correct."])
            lines += _chord_report_lines(sequence)
            lines += ["", "Normalized ABC (stored):", normalized]
            yield _ok("\n".join(lines))

        elif name == "play":
            seq_id = inp.get("sequence_id", "")
            bars_str = inp.get("bars")
            seq_dict = resolve_sequence(seq_id)
            if seq_dict is None:
                yield _err(f"Sequence '{seq_id}' not found.")
                continue
            engine.play_sequence_bg(seq_id, seq_dict, bars=bars_str)
            midi_path = generated_dir / f"{seq_id}.mid"
            if not midi_path.exists():
                midi_path = write_sequence_midi(seq_dict, seq_id, generated_dir)
            row = seq_model.get_sequence(seq_id)
            title = row["title"] if row else seq_id
            pill_id = f"p{str(uuid.uuid4())[:6]}"
            midi_url = f"/sequence/{seq_id}/download"
            yield ("sse", sequence_pill_fn(seq_id, title, pill_id, seq_dict["duration_ms"], midi_url, seq_dict))
            bars_note = f" (bars {bars_str})" if bars_str else ""
            yield _ok(f"Playing '{title}'{bars_note}.")

        elif name == "list_sequences":
            sid = inp.get("session_id") or session_id
            seqs = seq_model.list_sequences(session_id=sid)
            if not seqs:
                yield _ok("No saved sequences found.")
            else:
                lines = [f"{len(seqs)} sequence(s):"]
                for s in seqs:
                    lines.append(f"  {s['id']}  {s['title']}  ({s['time_signature']}, {s['tempo_bpm']} bpm)")
                yield _ok("\n".join(lines))

        elif name == "voice_chord":
            root = inp.get("root", "C")
            quality = inp.get("quality", "major")
            melody_note = inp.get("melody_note")
            register = inp.get("register", "mid")
            style = inp.get("style", "close")
            omit_root = bool(inp.get("omit_root", False))
            key = inp.get("key")
            try:
                result = _voice_chord(root, quality, melody_note=melody_note,
                                      register=register, style=style, omit_root=omit_root,
                                      key=key)
            except Exception as e:
                yield _err(f"voice_chord error: {e}")
                continue
            lines = [
                f"Voicing: {root} {quality} (style={style})",
                f"Notes (bottom to top): {', '.join(result['notes'])}",
                f"MIDI: {result['midi']}",
                f"ABC chord token: {result['abc']}",
            ]
            if melody_note:
                lines.append(f"Sits below melody note: {melody_note}")
            yield _ok("\n".join(lines))

        elif name == "harmonize_sequence":
            seq_id = inp.get("sequence_id", "")
            anchors_in = inp.get("anchors") or []
            style = inp.get("style", "close")
            row = seq_model.get_sequence(seq_id)
            if not row:
                yield _err(f"Sequence '{seq_id}' not found.")
                continue
            if not anchors_in:
                yield _err("anchors is required: e.g. [{chord: 'C', at_event: 1}, {chord: 'Am', at_event: 2}]")
                continue
            try:
                melody_seq = parse_abc(row["abc"])
            except ABCParseError as e:
                yield _err(f"Stored ABC parse error:\n{e}")
                continue
            if melody_seq.get("voices") and len(melody_seq["voices"]) > 1:
                yield _err("Sequence is already multi-voice. Harmonize the original single-voice melody or recording instead.")
                continue
            mel_events = sorted(melody_seq["events"], key=lambda e: e["at_beat"])
            total_beats = melody_seq["total_beats"]

            resolved = []
            anchor_error = None
            for a in anchors_in:
                symbol = a.get("chord") or a.get("symbol") or ""
                if not symbol:
                    anchor_error = "Each anchor needs a chord symbol."
                    break
                if a.get("at_event") is not None:
                    idx = int(a["at_event"]) - 1
                    if idx < 0 or idx >= len(mel_events):
                        anchor_error = f"at_event {a['at_event']} out of range (melody has {len(mel_events)} events)."
                        break
                    onset = mel_events[idx]["at_beat"]
                    mel_ev = mel_events[idx]
                elif a.get("at_beat") is not None:
                    onset = float(a["at_beat"])
                    mel_ev = next((e for e in reversed(mel_events) if e["at_beat"] <= onset + 1e-9), mel_events[0])
                else:
                    anchor_error = f"Anchor for '{symbol}' needs at_event or at_beat."
                    break
                resolved.append({"symbol": symbol, "onset": onset, "melody_note": mel_ev["note_names"][-1]})
            if anchor_error:
                yield _err(anchor_error)
                continue

            resolved.sort(key=lambda r: r["onset"])
            chords_arg = []
            for i, r in enumerate(resolved):
                end = resolved[i + 1]["onset"] if i + 1 < len(resolved) else total_beats
                if end - r["onset"] <= 1e-9:
                    anchor_error = f"Anchor '{r['symbol']}' at beat {r['onset']:g} has zero duration (same onset as the next anchor)."
                    break
                chords_arg.append({"symbol": r["symbol"], "beats": end - r["onset"], "melody_note": r["melody_note"]})
            if anchor_error:
                yield _err(anchor_error)
                continue
            try:
                vp = _voice_progression(chords_arg, style=style,
                                        key=melody_seq.get("key"))
            except Exception as e:
                yield _err(f"Voicing error: {e}")
                continue

            chord_events = []
            for r, v in zip(resolved, vp["voicings"]):
                chord_events.append({
                    "at_beat": float(r["onset"]),
                    "duration_beats": float(v["beats"]),
                    "notes": list(v["midi"]),
                    "note_names": list(v["notes"]),
                    "root": r["symbol"],
                    "quality": "note",
                    "octave": v["midi"][0] // 12 - 1,
                    "velocity": 80,
                    "label": "+".join(v["notes"]),
                    "voice": "2",
                })
            events = [dict(e, voice="1") for e in mel_events] + chord_events
            events.sort(key=lambda e: (e["at_beat"], e["voice"]))
            title = (inp.get("title") or f"{row['title']} + chords").strip()[:80]
            new_seq = {
                "title": title,
                "tempo_bpm": melody_seq["tempo_bpm"],
                "time_signature": melody_seq["time_signature"],
                "time_signature_parts": melody_seq["time_signature_parts"],
                "key": melody_seq.get("key", "C"),
                "voices": [
                    {"id": "1", "name": "Melody", "octave_shift": 0},
                    {"id": "2", "name": "Chords", "octave_shift": 0},
                ],
                "events": events,
                "total_beats": total_beats,
                "duration_ms": int(total_beats * 60 / melody_seq["tempo_bpm"] * 1000),
            }
            abc_text = to_abc(new_seq)
            try:
                final_seq = parse_abc(abc_text)
            except ABCParseError as e:
                yield _err(f"Internal error: harmonized ABC failed to validate:\n{e}")
                continue
            new_id = seq_model.create_sequence(
                title=title, abc=abc_text, session_id=session_id,
                tempo_bpm=final_seq["tempo_bpm"], time_signature=final_seq["time_signature"],
                key=final_seq.get("key", "C"), source="agent",
            )
            midi_path = write_sequence_midi(final_seq, new_id, generated_dir)
            sequence_registry[new_id] = {"sequence": final_seq, "midi_path": midi_path}
            engine.play_sequence_bg(new_id, final_seq)
            pill_id = f"p{str(uuid.uuid4())[:6]}"
            midi_url = f"/sequence/{new_id}/download"
            yield ("sse", sequence_pill_fn(new_id, title, pill_id, final_seq["duration_ms"], midi_url, final_seq))
            yield ("record", {
                "type": "sequence", "sequence_id": new_id,
                "title": title, "duration_ms": final_seq["duration_ms"],
                "sequence": final_seq, "midi_path": str(midi_path),
            })
            lines = [f"Harmonized '{row['title']}' → '{title}' (id: {new_id}). Melody copied verbatim to voice 1.", "", "Chord alignment:"]
            for c, ce in zip(chords_arg, chord_events):
                lines.append(f"  {ce['root']}: beats {ce['at_beat']:g}–{ce['at_beat'] + ce['duration_beats']:g} "
                             f"under melody {c['melody_note']}  →  {ce['label']}")
            lines += ["", "ABC:", abc_text]
            yield _ok("\n".join(lines))

        elif name == "voice_progression":
            chords = inp.get("chords", [])
            style = inp.get("style", "close")
            key = inp.get("key")
            try:
                result = _voice_progression(chords, style=style, key=key)
            except Exception as e:
                yield _err(f"voice_progression error: {e}")
                continue
            lines = ["Voiced progression:"]
            for v in result["voicings"]:
                lines.append(f"  {v['symbol']} ({v['beats']} beats): {', '.join(v['notes'])}  →  {v['abc']}")
            lines += ["", f"ABC line for [V:2]:", result["abc_line"]]
            yield _ok("\n".join(lines))

        elif name == "list_charts":
            lines = ["Built-in charts:"]
            for entry in charts.list_charts():
                lines.append(
                    f"  {entry['id']} — {entry['name']}, {entry['bars']} bars in "
                    f"{entry['time_signature']}, default key {entry['key']}, "
                    f"{entry['default_tempo_bpm']:g} bpm {entry['default_feel']}, "
                    f"mode {entry['default_mode']}"
                )
                if entry["description"]:
                    lines.append(f"      {entry['description']}")
            lines += ["", f"Modes: {', '.join(charts.MODES)}"]
            yield _ok("\n".join(lines))

        elif name in ("show_chart", "start_chart_loop"):
            try:
                chart = _resolve_chart_arg(inp)
                rendered = charts.render_chart(chart, key=inp.get("key"), mode=inp.get("mode"))
            except (KeyError, ValueError) as e:
                yield _err(f"Chart error: {e}")
                continue

            grid = charts.chart_text(rendered, roman=bool(inp.get("roman", True)))

            if name == "show_chart":
                yield _ok(grid)
                continue

            config = loop.LoopConfig(
                chords=charts.to_loop_chords(rendered),
                tempo_bpm=float(inp.get("tempo_bpm") or rendered["tempo_bpm"]),
                time_signature=rendered["time_signature"],
                feel=inp.get("feel") or rendered["feel"],
                click=bool(inp.get("click", True)),
                comp=bool(inp.get("comp", True)),
                bass=bool(inp.get("bass", True)),
                count_in_bars=int(inp.get("count_in_bars", 1)),
                rootless=bool(inp.get("rootless", False)),
                repeats=inp.get("repeats"),
                key=rendered["key"],
            )
            try:
                loop.start(config)
            except (ValueError, RuntimeError) as e:
                yield _err(f"Could not start the loop: {e}")
                continue

            tail = "looping until stopped" if config.repeats is None else f"{config.repeats}x through"
            yield _ok(
                f"Loop running — {config.tempo_bpm:g} bpm, {config.feel} feel, {tail}.\n\n{grid}"
            )

        elif name == "stop_loop":
            loop.stop()
            yield _ok("Loop stopped.")
