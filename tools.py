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

from sequencer.abc import parse_abc, to_abc, ABCParseError, per_bar_report
from sequencer.theory import normalize_chord_quality, chord_note_names, build_chord, parse_pitch, midi_note_name
from sequencer.midi_io import write_sequence_midi
import sequencer.model as seq_model
import sequencer.engine as engine

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
- Use **play_abc** for any melody, progression, or rhythmic example longer than a single chord. This is the primary tool for one-shot playback — it accepts ABC notation and gives you per-bar feedback.
- Use **check_abc** to validate ABC before playing when correctness is critical (e.g. anything more than a couple of bars, or when meter or rhythm matters). Read the normalized ABC and per-bar report in the result, fix any errors, then call play_abc.
- If the user includes an image of sheet music, transcribe it to ABC notation, run check_abc, then play_abc.

## Persistent sequences (for multi-turn composition)

When building a piece across multiple turns, use the persistent sequence tools instead of play_abc:
- **create_sequence(title, abc)** — saves ABC and returns a `sequence_id`
- **read_sequence(sequence_id, bars="1-8")** — shows normalized ABC and per-bar report for the whole piece or a bar range
- **update_sequence(sequence_id, abc)** — replaces the ABC and appends a revision (so the full edit history is preserved)
- **play(sequence_id, bars="3-6")** — plays a saved sequence, optionally a bar range only
- **list_sequences()** — shows all sequences in this session

Use `create_sequence` when you expect to revise a piece; use `play_abc` for ephemeral one-shot examples. Both persist across restarts.

## ABC notation guide

ABC is a text format for music. Key headers: `X:1`, `T:title`, `M:4/4`, `L:1/4`, `Q:120`, `K:C`.

**Notes**: Uppercase letters = octave 3 (C=C3, middle C is `c` lowercase). Lowercase = octave 4. `'` raises an octave, `,` lowers. Examples: `C`=C3, `c`=C4, `c'`=C5, `C,`=C2.

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
c d e f | g a b c' | c' b a g | f e d c |
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

For original compositions and generic theory demonstrations (scales, chord progressions), freely compose."""

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
        "description": "Parse ABC notation, save it, and play it back. Returns normalized ABC and a per-bar report so you can verify what was actually stored. Use check_abc first for complex pieces.",
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
        "description": "Create, save, and play a short timed MIDI orchestration. Use this for chord progressions or note/chord sequences where timing matters.",
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
        "name": "update_sequence",
        "description": "Replace the ABC content of a persistent sequence (appends a revision). Supply full corrected ABC.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence_id": {"type": "string", "description": "ID returned by create_sequence."},
                "abc": {"type": "string", "description": "Replacement ABC notation (full ABC with headers)."},
            },
            "required": ["sequence_id", "abc"],
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

def dispatch_tools(
    pending_tools: list[dict],
    *,
    session_id: str,
    asst_msg_id: str,
    note_registry: dict,
    sequence_registry: dict,
    resolve_sequence,          # callable(seq_id) -> dict | None
    sequence_pill_fn,          # callable(seq_id, title, pill_id, duration_ms, midi_url, events) -> str
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
            yield ("sse", sequence_pill_fn(seq_id, sequence["title"], pill_id, sequence["duration_ms"], midi_url, sequence["events"]))
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
            lines += ["", "Normalized ABC:", normalized]
            yield _ok("\n".join(lines))

        elif name == "read_sequence":
            seq_id = inp.get("sequence_id", "")
            bars_str = inp.get("bars")
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
            lines += ["", "Normalized ABC:", normalized]
            yield _ok("\n".join(lines))

        elif name == "update_sequence":
            seq_id = inp.get("sequence_id", "")
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
            yield ("sse", sequence_pill_fn(seq_id, title, pill_id, seq_dict["duration_ms"], midi_url, seq_dict["events"]))
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
