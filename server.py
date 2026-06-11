#!/usr/bin/env python3
"""midimi web server — FastAPI + Datastar SSE, server-side playback via FluidSynth or MIDI out"""

import base64
import html
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import anthropic
import fluidsynth
import mido
from sequencer.abc import parse_abc, to_abc, ABCParseError, per_bar_report
from sequencer.theory import (
    normalize_chord_quality, chord_note_names, build_chord, parse_pitch,
    NOTE_NAMES, CHORD_INTERVALS,
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


_DURATION_BEATS = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,
    "thirty_second": 0.125,
}

_SHARP_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
_FLAT_NAMES  = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]


def midi_note_name(n: int, prefer_flats: bool = False) -> str:
    names = _FLAT_NAMES if prefer_flats else _SHARP_NAMES
    return f"{names[n % 12]}{n // 12 - 1}"


SOUNDFONT = os.environ.get("SOUNDFONT", str(Path.home() / "Music" / "GeneralUser-GS.sf2"))
DB_PATH = Path(__file__).parent / "midimi.db"
STATIC_DIR = Path(__file__).parent / "static"
GENERATED_DIR = Path(__file__).parent / "generated" / "orchestrations"
DEFAULT_INSTRUMENT = 0
DEFAULT_CHANNEL = 0
DEFAULT_VELOCITY = 90
DEFAULT_DURATION_MS = 1500
MIDI_TICKS_PER_BEAT = 480
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

SYSTEM_PROMPT = """You are an expert music theory teacher. You explain concepts clearly, use examples, and make learning engaging.

## Playback tools

- Use **play_notes** for one isolated chord, interval, or single note.
- Use **play_abc** for any melody, progression, or rhythmic example longer than a single chord. This is the primary tool for sequences — it accepts ABC notation and gives you per-bar feedback.
- Use **check_abc** to validate ABC before playing when correctness is critical (e.g. anything more than a couple of bars, or when meter or rhythm matters). Read the normalized ABC and per-bar report in the result, fix any errors, then call play_abc.
- If the user includes an image of sheet music, transcribe it to ABC notation, run check_abc, then play_abc.

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
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
_anthropic_client: anthropic.Anthropic | None = None


def get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = db_get_setting("api_key")
        if not api_key:
            raise HTTPException(400, "Anthropic API key not configured. Open Settings (⚙) to add it.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Player (FluidSynth or MIDI out) ──────────────────────────────────────────

_play_lock = threading.Lock()
_stop_event = threading.Event()
_note_on = None
_note_off = None
_current_fs = None  # active FluidSynth instance, kept so it can be cleaned up on switch


def _init_player():
    midi_out = (db_get_setting("midi_out") or "").strip()
    if midi_out:
        available = mido.get_output_names()
        matches = [p for p in available if midi_out.lower() in p.lower()]
        if not matches:
            print(f"Warning: saved MIDI_OUT='{midi_out}' matched no ports. Falling back to FluidSynth.")
            return _init_fluidsynth()
        port = mido.open_output(matches[0])
        print(f"MIDI out: {matches[0]}")

        def play(notes: list[int], duration_ms: int) -> None:
            print(f"[midi-out:{matches[0]}] {notes}")
            with _play_lock:
                _stop_event.clear()
                for n in notes:
                    note_on(n, DEFAULT_VELOCITY, DEFAULT_CHANNEL)
                _stop_event.wait(duration_ms / 1000)
                for n in notes:
                    note_off(n, DEFAULT_CHANNEL)

        def note_on(note: int, velocity: int = DEFAULT_VELOCITY, channel: int = DEFAULT_CHANNEL) -> None:
            port.send(mido.Message("note_on", channel=channel, note=note, velocity=velocity))

        def note_off(note: int, channel: int = DEFAULT_CHANNEL) -> None:
            port.send(mido.Message("note_off", channel=channel, note=note, velocity=0))

        return play, matches[0], note_on, note_off
    else:
        return _init_fluidsynth()


def _init_fluidsynth():
    global _current_fs
    if _current_fs is not None:
        try:
            _current_fs.delete()
        except Exception:
            pass
        _current_fs = None

    fs = fluidsynth.Synth(gain=0.5)
    fs.start(driver="coreaudio")
    _current_fs = fs
    sfid = fs.sfload(SOUNDFONT)
    if sfid == -1:
        raise RuntimeError(f"Could not load soundfont: {SOUNDFONT}")
    fs.program_select(DEFAULT_CHANNEL, sfid, 0, DEFAULT_INSTRUMENT)
    fs.set_reverb(roomsize=0.5, damping=0.3, width=0.8, level=0.7)
    fs.set_chorus(nr=4, level=0.55, speed=0.36, depth=3.6, type=0)
    print("Audio out: FluidSynth")

    def play(notes: list[int], duration_ms: int) -> None:
        print(f"[fluidsynth] {notes}")
        with _play_lock:
            _stop_event.clear()
            for n in notes:
                note_on(n, DEFAULT_VELOCITY, DEFAULT_CHANNEL)
            _stop_event.wait(duration_ms / 1000)
            for n in notes:
                note_off(n, DEFAULT_CHANNEL)

    def note_on(note: int, velocity: int = DEFAULT_VELOCITY, channel: int = DEFAULT_CHANNEL) -> None:
        fs.noteon(channel, note, velocity)

    def note_off(note: int, channel: int = DEFAULT_CHANNEL) -> None:
        fs.noteoff(channel, note)

    return play, None, note_on, note_off


# In-memory registry: note_id → {notes, duration_ms} for replay
_note_registry: dict[str, dict] = {}
_sequence_registry: dict[str, dict] = {}

# Set of note_ids currently being played
_currently_playing: set[str] = set()
_currently_playing_lock = threading.Lock()


def play_in_background(notes: list[int], duration_ms: int, note_id: str | None = None) -> None:
    def _run():
        if note_id:
            with _currently_playing_lock:
                _currently_playing.add(note_id)
        try:
            _play(notes, duration_ms)
        finally:
            if note_id:
                with _currently_playing_lock:
                    _currently_playing.discard(note_id)

    threading.Thread(target=_run, daemon=True).start()


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
            supported = sorted(_DURATION_BEATS) + [f"dotted_{name}" for name in sorted(_DURATION_BEATS)]
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


def _clamp_midi_value(value: int, field: str) -> int:
    if not isinstance(value, int) or not (0 <= value <= 127):
        raise ValueError(f"{field} must be a MIDI value from 0 to 127")
    return value


def _midi_meta_text(value: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    text = "".join(replacements.get(ch, ch) for ch in str(value))
    return text.encode("latin-1", "replace").decode("latin-1")


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
            _clamp_midi_value(note, f"events[{idx}] note")

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
        "events": sorted(events, key=lambda event: event["at_beat"]),
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
        _clamp_midi_value(midi_note, f"notes[{idx}] pitch")

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
        "title": title,
        "tempo_bpm": tempo_bpm,
        "time_signature": time_signature,
        "events": events,
    })
    sequence["source"] = "melody"
    return sequence


def write_sequence_midi(sequence: dict, sequence_id: str) -> Path:
    midi_path = GENERATED_DIR / f"{sequence_id}.mid"
    mid = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    ts_numerator, ts_denominator = sequence["time_signature_parts"]
    track.append(mido.MetaMessage("track_name", name=_midi_meta_text(sequence["title"]), time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(sequence["tempo_bpm"]), time=0))
    track.append(mido.MetaMessage(
        "time_signature",
        numerator=ts_numerator,
        denominator=ts_denominator,
        time=0,
    ))
    track.append(mido.Message("program_change", channel=DEFAULT_CHANNEL, program=DEFAULT_INSTRUMENT, time=0))

    midi_events = []
    for event in sequence["events"]:
        start_tick = int(round(event["at_beat"] * MIDI_TICKS_PER_BEAT))
        end_tick = int(round((event["at_beat"] + event["duration_beats"]) * MIDI_TICKS_PER_BEAT))
        for note in event["notes"]:
            midi_events.append((start_tick, 1, note, event["velocity"]))
            midi_events.append((end_tick, 0, note, 0))

    midi_events.sort(key=lambda item: (item[0], item[1]))
    last_tick = 0
    for tick, kind, note, velocity in midi_events:
        delta = max(0, tick - last_tick)
        msg_type = "note_on" if kind else "note_off"
        track.append(mido.Message(msg_type, channel=DEFAULT_CHANNEL, note=note, velocity=velocity, time=delta))
        last_tick = tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(midi_path)
    return midi_path


def play_sequence_in_background(sequence_id: str) -> None:
    def _run():
        entry = _sequence_registry.get(sequence_id)
        if not entry:
            return
        sequence = entry["sequence"]
        seconds_per_beat = 60 / sequence["tempo_bpm"]
        actions = []
        for event in sequence["events"]:
            start = event["at_beat"] * seconds_per_beat
            end = (event["at_beat"] + event["duration_beats"]) * seconds_per_beat
            for note in event["notes"]:
                actions.append((start, 1, note, event["velocity"]))
                actions.append((end, 0, note, 0))
        actions.sort(key=lambda item: (item[0], item[1]))

        with _currently_playing_lock:
            _currently_playing.add(sequence_id)
        try:
            with _play_lock:
                _stop_event.clear()
                start_time = time.monotonic()
                sounding = []
                try:
                    for action_time, kind, note, velocity in actions:
                        if _stop_event.is_set():
                            break
                        sleep_for = start_time + action_time - time.monotonic()
                        if sleep_for > 0:
                            _stop_event.wait(sleep_for)
                        if _stop_event.is_set():
                            break
                        if kind:
                            _note_on(note, velocity, DEFAULT_CHANNEL)
                            sounding.append(note)
                        else:
                            _note_off(note, DEFAULT_CHANNEL)
                            if note in sounding:
                                sounding.remove(note)
                finally:
                    for note in sounding:
                        _note_off(note, DEFAULT_CHANNEL)
        finally:
            with _currently_playing_lock:
                _currently_playing.discard(sequence_id)

    threading.Thread(target=_run, daemon=True).start()


# ── Database ──────────────────────────────────────────────────────────────────

def db_get_setting(key: str) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def db_set_setting(key: str, value: str | None):
    conn = sqlite3.connect(DB_PATH)
    if value is None:
        conn.execute("DELETE FROM settings WHERE key=?", (key,))
    else:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, value))
    conn.commit()
    conn.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at INTEGER,
            title TEXT,
            modified_at INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at INTEGER
        )
    """)
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN starred INTEGER DEFAULT 0")
    except Exception:
        pass
    conn.commit()
    conn.close()


init_db()
_play, _current_port, _note_on, _note_off = _init_player()


def db_update_session_modified(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE sessions SET modified_at=? WHERE id=?", (int(time.time()), session_id))
    conn.commit()
    conn.close()


def generate_title_async(session_id: str, first_user_message: str):
    def _gen():
        try:
            resp = get_anthropic_client().messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": (
                    f'Give a 3-5 word title for a music theory conversation that starts with: '
                    f'"{first_user_message}". Reply with just the title, no quotes or punctuation.'
                )}],
            )
            title = resp.content[0].text.strip()
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
            conn.commit()
            conn.close()
        except Exception:
            pass
    threading.Thread(target=_gen, daemon=True).start()


def db_save_message(session_id: str, role: str, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
        (session_id, role, json.dumps(content), int(time.time())),
    )
    conn.commit()
    conn.close()


def db_get_history(session_id: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": json.loads(r[1])} for r in rows]


# ── Datastar SSE helpers ──────────────────────────────────────────────────────

def ds_merge_fragment(fragment: str, selector: str = "", merge_mode: str = "append") -> str:
    lines = ["event: datastar-merge-fragments"]
    if selector:
        lines.append(f"data: selector {selector}")
    lines.append(f"data: mergeMode {merge_mode}")
    for line in fragment.splitlines():
        lines.append(f"data: fragments {line}")
    return "\n".join(lines) + "\n\n"


def ds_merge_signals(signals: dict) -> str:
    return f"event: datastar-merge-signals\ndata: signals {json.dumps(signals)}\n\n"


# ── HTML fragments ────────────────────────────────────────────────────────────

def user_bubble(text: str, msg_id: str, images: list[dict] | None = None) -> str:
    safe = html.escape(text)
    image_html = ""
    if images:
        image_html = '<div class="message-images">' + "".join(
            f'<img src="data:{html.escape(image["media_type"])};base64,{html.escape(image["data"])}" alt="Pasted sheet music" />'
            for image in images
        ) + "</div>"
    return (
        f'<div id="{msg_id}" class="msg msg-user">'
        f'<div class="bubble">{image_html}{safe}</div>'
        f'</div>'
    )


def assistant_bubble_open(msg_id: str) -> str:
    return (
        f'<div id="{msg_id}" class="msg msg-assistant">'
        f'<div class="bubble">'
        f'<span id="{msg_id}-spinner" class="thinking-spinner"></span>'
        f'</div>'
        f'</div>'
    )


def audio_pill(
    note_id: str,
    label: str,
    pill_id: str,
    notes: list[int] | None = None,
    root: str = "",
    quality: str = "",
) -> str:
    import json as _json
    safe_label = html.escape(label or "Play")
    chips_html = ""
    staff_btn = ""
    staff_panel = ""
    if notes:
        if root and quality:
            octave = notes[0] // 12 - 1
            names = chord_note_names(root, quality, octave)
        else:
            prefer_flats = prefer_flats_for(root, quality)
            names = [midi_note_name(n, prefer_flats) for n in notes]
        chips = "".join(
            f'<button class="note-chip" onclick="playMidiNote({n}, this)" title="{name}">'
            f'{html.escape(name)}</button>'
            for n, name in zip(notes, names)
        )
        chips_html = f'<div class="pill-notes">{chips}</div>'
        staff_panel_id = f"{pill_id}-staff"
        note_names_json = html.escape(_json.dumps(names), quote=True)
        staff_btn = (
            f'<button class="staff-btn" '
            f"onclick=\"toggleStaff('{staff_panel_id}', this, {note_names_json})\" "
            f'title="Show on staff">\U0001D11E</button>'
        )
        staff_panel = f'<div id="{staff_panel_id}" class="staff-panel"></div>'
    return (
        f'<div class="pill-wrap">'
        f'  <div id="{pill_id}" class="audio-pill">'
        f'    <button class="play-btn" onclick="replayNote(\'{note_id}\')" title="Replay">▶</button>'
        f'    <span class="pill-label">♪ {safe_label}</span>'
        f'    {chips_html}'
        f'    {staff_btn}'
        f'  </div>'
        f'  {staff_panel}'
        f'</div>'
    )


def sequence_pill(sequence_id: str, title: str, pill_id: str, duration_ms: int, midi_url: str, events: list[dict]) -> str:
    import json as _json
    safe_title = html.escape(title or "Orchestration")
    seconds = max(1, round(duration_ms / 1000))
    staff_panel_id = f"{pill_id}-staff"
    events_json = html.escape(_json.dumps(events), quote=True)
    return (
        f'<div class="pill-wrap">'
        f'  <div id="{pill_id}" class="audio-pill sequence-pill">'
        f'    <button class="play-btn" onclick="replaySequence(\'{sequence_id}\')" title="Replay sequence">▶</button>'
        f'    <span class="pill-label">♫ {safe_title}</span>'
        f'    <span class="sequence-meta">{seconds}s</span>'
        f'    <a class="download-btn" href="{html.escape(midi_url)}" title="Download MIDI">MIDI</a>'
        f'    <button class="staff-btn" onclick="toggleSequenceStaff(\'{staff_panel_id}\', this, {events_json})" title="Show on staff">\U0001D11E</button>'
        f'  </div>'
        f'  <div id="{staff_panel_id}" class="staff-panel sequence-staff-panel"></div>'
        f'</div>'
    )


# ── Config routes ─────────────────────────────────────────────────────────────

class ConfigRequest(BaseModel):
    port: str | None  # None = FluidSynth


@app.get("/config")
def get_config():
    return {
        "current": _current_port,
        "ports": mido.get_output_names(),
    }


@app.post("/config")
def set_config(req: ConfigRequest):
    global _play, _current_port, _note_on, _note_off

    if req.port is None:
        _play, _current_port, _note_on, _note_off = _init_fluidsynth()
        db_set_setting("midi_out", None)
        return {"ok": True, "current": None}

    available = mido.get_output_names()
    if req.port not in available:
        raise HTTPException(400, f"Port '{req.port}' not found. Available: {available}")

    port = mido.open_output(req.port)

    def play(notes: list[int], duration_ms: int) -> None:
        print(f"[midi-out:{req.port}] {notes}")
        with _play_lock:
            _stop_event.clear()
            for n in notes:
                note_on(n, DEFAULT_VELOCITY, DEFAULT_CHANNEL)
            _stop_event.wait(duration_ms / 1000)
            for n in notes:
                note_off(n, DEFAULT_CHANNEL)

    def note_on(note: int, velocity: int = DEFAULT_VELOCITY, channel: int = DEFAULT_CHANNEL) -> None:
        port.send(mido.Message("note_on", channel=channel, note=note, velocity=velocity))

    def note_off(note: int, channel: int = DEFAULT_CHANNEL) -> None:
        port.send(mido.Message("note_off", channel=channel, note=note, velocity=0))

    _play = play
    _current_port = req.port
    _note_on = note_on
    _note_off = note_off
    db_set_setting("midi_out", req.port)
    return {"ok": True, "current": req.port}


class TestRequest(BaseModel):
    port: str | None


@app.post("/config/test")
def test_output(req: TestRequest):
    """Switch to the given port (without saving) and send a test chord."""
    global _play, _current_port
    # Temporarily apply the selected port so the test uses it
    result = set_config(ConfigRequest(port=req.port))
    play_in_background([60, 64, 67], 800)
    return result


class SettingsRequest(BaseModel):
    api_key: str | None = None
    port: str | None = None


@app.get("/settings")
def get_settings():
    api_key = db_get_setting("api_key")
    return {
        "api_key_set": bool(api_key),
        "current_port": _current_port,
        "ports": mido.get_output_names(),
    }


@app.post("/settings")
def save_settings(req: SettingsRequest):
    global _anthropic_client
    if req.api_key is not None:
        db_set_setting("api_key", req.api_key or None)
        _anthropic_client = None  # force re-init with new key
    if "port" in req.model_fields_set:
        set_config(ConfigRequest(port=req.port))
    return {"ok": True, "api_key_set": bool(db_get_setting("api_key"))}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/sessions")
def list_sessions():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title, modified_at, COALESCE(starred, 0) FROM sessions "
        "ORDER BY COALESCE(starred, 0) DESC, COALESCE(modified_at, created_at) DESC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1] or "New session", "modified_at": r[2], "starred": bool(r[3])} for r in rows]


@app.post("/session/{session_id}/star")
def toggle_star(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE sessions SET starred = CASE WHEN COALESCE(starred,0)=1 THEN 0 ELSE 1 END WHERE id=?",
        (session_id,)
    )
    conn.commit()
    starred = conn.execute("SELECT COALESCE(starred,0) FROM sessions WHERE id=?", (session_id,)).fetchone()[0]
    conn.close()
    return {"starred": bool(starred)}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/session/new")
def new_session():
    sid = str(uuid.uuid4())[:8]
    now = int(time.time())
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO sessions (id, created_at, title, modified_at) VALUES (?,?,?,?)", (sid, now, "New session", now))
    conn.commit()
    conn.close()
    return {"id": sid}


@app.get("/session/{session_id}/history")
def session_history(session_id: str):
    history = db_get_history(session_id)
    result = []
    for msg in history:
        if msg["role"] == "user":
            content = msg["content"]
            result.append({
                "role": "user",
                "text": user_content_text(content),
                "images": user_content_images(content),
            })
        elif msg["role"] == "assistant":
            parts = msg["content"] if isinstance(msg["content"], list) else []
            items = []
            for p in parts:
                if p.get("type") == "text":
                    items.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "audio":
                    nid = p["note_id"]
                    _note_registry[nid] = {
                        "notes": p["notes"],
                        "duration_ms": p["duration_ms"],
                        "root": p.get("root", ""),
                        "quality": p.get("quality", ""),
                    }
                    items.append({
                        "type": "audio",
                        "note_id": nid,
                        "label": p.get("label", ""),
                        "notes": p["notes"],
                        "root": p.get("root", ""),
                        "quality": p.get("quality", ""),
                        "prefer_flats": p.get("prefer_flats", prefer_flats_for(p.get("root", ""), p.get("quality", ""))),
                    })
                elif p.get("type") == "sequence":
                    sequence_id = p["sequence_id"]
                    midi_path = Path(p.get("midi_path", GENERATED_DIR / f"{sequence_id}.mid"))
                    if "sequence" in p:
                        _sequence_registry[sequence_id] = {"sequence": p["sequence"], "midi_path": midi_path}
                    items.append({
                        "type": "sequence",
                        "sequence_id": sequence_id,
                        "title": p.get("title", "Orchestration"),
                        "duration_ms": p.get("duration_ms", 1000),
                        "events": p.get("sequence", {}).get("events", []),
                        "midi_url": f"/sequence/{sequence_id}/download",
                    })
            if items:
                result.append({"role": "assistant", "items": items})
    return result


@app.post("/play/{note_id}")
def replay(note_id: str):
    entry = _note_registry.get(note_id)
    if not entry:
        raise HTTPException(404, "Note not found")
    play_in_background(entry["notes"], entry["duration_ms"], note_id)
    return {"ok": True}


@app.post("/sequence/{sequence_id}")
def replay_sequence(sequence_id: str):
    entry = _sequence_registry.get(sequence_id)
    if not entry:
        raise HTTPException(404, "Sequence not found")
    play_sequence_in_background(sequence_id)
    return {"ok": True}


@app.get("/sequence/{sequence_id}/download")
def download_sequence(sequence_id: str):
    entry = _sequence_registry.get(sequence_id)
    midi_path = Path(entry["midi_path"]) if entry else GENERATED_DIR / f"{sequence_id}.mid"
    if not midi_path.exists() or midi_path.parent != GENERATED_DIR:
        raise HTTPException(404, "MIDI file not found")
    return FileResponse(midi_path, media_type="audio/midi", filename=f"{sequence_id}.mid")


@app.post("/play_midi/{note}")
def play_single_midi(note: int):
    if not (0 <= note <= 127):
        raise HTTPException(400, "MIDI note must be 0–127")
    play_in_background([note], 800)
    return {"ok": True}


@app.get("/playing")
def get_playing():
    with _currently_playing_lock:
        return list(_currently_playing)


@app.post("/stop")
def stop_playback():
    _stop_event.set()
    return {"ok": True}


ALLOWED_MODELS = {
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
}


class ImageAttachment(BaseModel):
    media_type: str
    data: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "claude-haiku-4-5-20251001"
    images: list[ImageAttachment] = Field(default_factory=list)


def normalize_image_attachments(images: list[ImageAttachment]) -> list[dict]:
    if len(images) > 3:
        raise HTTPException(400, "Attach at most 3 images per message.")

    normalized = []
    for idx, image in enumerate(images):
        media_type = image.media_type.strip().lower()
        if media_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, f"Unsupported image type for attachment {idx + 1}: {media_type}")
        try:
            raw = base64.b64decode(image.data, validate=True)
        except Exception:
            raise HTTPException(400, f"Attachment {idx + 1} is not valid base64.")
        if not raw:
            raise HTTPException(400, f"Attachment {idx + 1} is empty.")
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(400, f"Attachment {idx + 1} is larger than 5 MB.")
        normalized.append({"media_type": media_type, "data": image.data})
    return normalized


def user_content_blocks(message: str, images: list[dict]) -> str | list[dict]:
    text = message.strip()
    if not images:
        return text

    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    else:
        blocks.append({"type": "text", "text": "Please transcribe this sheet music and play it back as MIDI."})
    blocks.extend({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image["media_type"],
            "data": image["data"],
        },
    } for image in images)
    return blocks


def user_content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""


def user_content_images(content) -> list[dict]:
    if not isinstance(content, list):
        return []
    images = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image":
            continue
        source = part.get("source") or {}
        if source.get("type") != "base64":
            continue
        images.append({
            "media_type": source.get("media_type", "image/png"),
            "data": source.get("data", ""),
        })
    return images


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if req.model not in ALLOWED_MODELS:
        raise HTTPException(400, f"Unknown model: {req.model}")
    images = normalize_image_attachments(req.images)
    message_text = req.message.strip()
    if not message_text and not images:
        raise HTTPException(400, "Message or image required.")
    user_content = user_content_blocks(message_text, images)
    history = db_get_history(req.session_id)

    anthropic_history = []
    for msg in history:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                anthropic_history.append({"role": "user", "content": content})
            else:
                anthropic_history.append({"role": "user", "content": str(content)})
        elif msg["role"] == "assistant":
            parts = msg["content"]
            if isinstance(parts, list):
                api_parts = [p for p in parts if p.get("type") in ("text", "tool_use")]
                if api_parts:
                    anthropic_history.append({"role": "assistant", "content": api_parts})

    user_msg_id = f"u{str(uuid.uuid4())[:6]}"
    asst_msg_id = f"a{str(uuid.uuid4())[:6]}"

    db_save_message(req.session_id, "user", user_content)
    anthropic_history.append({"role": "user", "content": user_content})

    def generate():
        yield ds_merge_fragment(user_bubble(message_text, user_msg_id, images), selector="#transcript")
        yield ds_merge_fragment(assistant_bubble_open(asst_msg_id), selector="#transcript")
        yield ds_merge_signals({"loading": True})

        current_history = list(anthropic_history)
        assistant_record = []
        seg_idx = 0

        # Open first text segment inside the bubble
        seg_id = f"{asst_msg_id}-t{seg_idx}"
        yield ds_merge_fragment(
            f'<span id="{seg_id}"></span>',
            selector=f"#{asst_msg_id} .bubble",
        )
        seg_text = ""

        while True:
            api_content = []        # what we'll append to history as assistant turn
            pending_tools = []      # tool calls completed this turn

            # Per-block streaming state
            in_tool = False
            tool_id = tool_name = ""
            tool_input_buf = ""

            with get_anthropic_client().messages.stream(
                model=req.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=current_history,
            ) as stream:
                for event in stream:
                    etype = type(event).__name__

                    if etype == "RawContentBlockStartEvent":
                        blk = event.content_block
                        if blk.type == "tool_use":
                            in_tool = True
                            tool_id = blk.id
                            tool_name = blk.name
                            tool_input_buf = ""
                        else:
                            in_tool = False

                    elif etype == "RawContentBlockDeltaEvent":
                        delta = event.delta
                        if delta.type == "text_delta":
                            seg_text += delta.text
                            yield ds_merge_signals({"textUpdate": {"segId": seg_id, "raw": seg_text}})
                        elif delta.type == "input_json_delta":
                            tool_input_buf += delta.partial_json

                    elif etype == "ParsedContentBlockStopEvent" and in_tool:
                        inp = json.loads(tool_input_buf) if tool_input_buf else {}
                        api_content.append({
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": inp,
                        })
                        pending_tools.append({"id": tool_id, "name": tool_name, "input": inp})
                        in_tool = False

                final = stream.get_final_message()

            # Harvest text blocks from the final message
            for blk in final.content:
                if blk.type == "text":
                    api_content.insert(0, {"type": "text", "text": blk.text})

            current_history.append({"role": "assistant", "content": api_content})

            if final.stop_reason != "tool_use" or not pending_tools:
                for blk in final.content:
                    if blk.type == "text":
                        assistant_record.append({"type": "text", "text": blk.text})
                break

            # Save pre-tool text so it appears before the pills on restore
            for blk in final.content:
                if blk.type == "text" and blk.text.strip():
                    assistant_record.append({"type": "text", "text": blk.text})

            # Execute tools and emit pills inline
            tool_results = []
            for tool in pending_tools:
                if tool["name"] == "check_abc":
                    inp = tool["input"]
                    try:
                        sequence = parse_abc(inp["abc"])
                    except ABCParseError as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": f"ABC parse error:\n{e}",
                            "is_error": True,
                        })
                        continue
                    normalized = to_abc(sequence)
                    bar_msgs = per_bar_report(sequence)
                    report_lines = [
                        f"Title: {sequence['title']}",
                        f"Tempo: {sequence['tempo_bpm']} bpm",
                        f"Meter: {sequence['time_signature']}",
                        f"Key: {sequence.get('key', 'C')}",
                        f"Total: {sequence['total_beats']:.4g} beats ({sequence['duration_ms']}ms)",
                        "",
                        "Per-bar report:",
                    ]
                    report_lines += (bar_msgs if bar_msgs else ["  All bars correct."])
                    report_lines += ["", "Normalized ABC:", normalized]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": "\n".join(report_lines),
                    })
                    continue

                elif tool["name"] == "play_abc":
                    inp = tool["input"]
                    try:
                        sequence = parse_abc(inp["abc"])
                    except ABCParseError as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": f"ABC parse error:\n{e}",
                            "is_error": True,
                        })
                        continue

                    sequence_id = str(uuid.uuid4())[:8]
                    midi_path = write_sequence_midi(sequence, sequence_id)
                    _sequence_registry[sequence_id] = {"sequence": sequence, "midi_path": midi_path}
                    play_sequence_in_background(sequence_id)

                    pill_id = f"p{str(uuid.uuid4())[:6]}"
                    midi_url = f"/sequence/{sequence_id}/download"
                    yield ds_merge_fragment(
                        sequence_pill(sequence_id, sequence["title"], pill_id, sequence["duration_ms"], midi_url, sequence["events"]),
                        selector=f"#{asst_msg_id} .bubble",
                    )

                    assistant_record.append({
                        "type": "sequence",
                        "sequence_id": sequence_id,
                        "title": sequence["title"],
                        "duration_ms": sequence["duration_ms"],
                        "sequence": sequence,
                        "midi_path": str(midi_path),
                    })

                    normalized = to_abc(sequence)
                    bar_msgs = per_bar_report(sequence)
                    report_lines = [
                        f"Played: {sequence['title']}",
                        f"Tempo: {sequence['tempo_bpm']} bpm  Meter: {sequence['time_signature']}  Key: {sequence.get('key', 'C')}",
                        f"Total: {sequence['total_beats']:.4g} beats ({sequence['duration_ms']}ms)",
                        "",
                        "Per-bar report:",
                    ]
                    report_lines += (bar_msgs if bar_msgs else ["  All bars correct."])
                    report_lines += ["", "Normalized ABC (what was stored):", normalized]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": "\n".join(report_lines),
                    })
                    continue

                elif tool["name"] == "play_notes":
                    inp = tool["input"]
                    duration_ms = inp.get("duration_ms", DEFAULT_DURATION_MS)
                    label = inp.get("label", "")
                    try:
                        root = inp["root"]
                        quality = normalize_chord_quality(inp["quality"], root=root)
                        notes = build_chord(root, quality, inp.get("octave", 4))
                    except ValueError as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": f"Error: {e}",
                            "is_error": True,
                        })
                        continue

                    octave = inp.get("octave", 4)
                    spelled_names = chord_note_names(root, quality, octave)
                    note_names_str = ", ".join(spelled_names)
                    note_id = str(uuid.uuid4())[:8]
                    _note_registry[note_id] = {
                        "notes": notes,
                        "duration_ms": duration_ms,
                        "root": root,
                        "quality": quality,
                    }
                    pill_id = f"p{str(uuid.uuid4())[:6]}"
                    yield ds_merge_fragment(
                        audio_pill(note_id, label, pill_id, notes, root, quality),
                        selector=f"#{asst_msg_id} .bubble",
                    )

                    assistant_record.append({
                        "type": "audio",
                        "note_id": note_id,
                        "notes": notes,
                        "note_names": spelled_names,
                        "duration_ms": duration_ms,
                        "label": label,
                        "root": root,
                        "quality": quality,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": f"Played {label or quality}: {note_names_str} (MIDI {', '.join(str(note) for note in notes)})",
                    })
                elif tool["name"] == "play_sequence":
                    inp = tool["input"]
                    try:
                        sequence = build_sequence(inp)
                    except (TypeError, ValueError) as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": f"Error: {e}",
                            "is_error": True,
                        })
                        continue

                    sequence_id = str(uuid.uuid4())[:8]
                    midi_path = write_sequence_midi(sequence, sequence_id)
                    _sequence_registry[sequence_id] = {"sequence": sequence, "midi_path": midi_path}

                    pill_id = f"p{str(uuid.uuid4())[:6]}"
                    midi_url = f"/sequence/{sequence_id}/download"
                    yield ds_merge_fragment(
                        sequence_pill(sequence_id, sequence["title"], pill_id, sequence["duration_ms"], midi_url, sequence["events"]),
                        selector=f"#{asst_msg_id} .bubble",
                    )

                    assistant_record.append({
                        "type": "sequence",
                        "sequence_id": sequence_id,
                        "title": sequence["title"],
                        "duration_ms": sequence["duration_ms"],
                        "sequence": sequence,
                        "midi_path": str(midi_path),
                    })
                    warnings = _sequence_warnings(sequence)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": f"Created {sequence['title']} ({sequence['duration_ms']}ms). MIDI saved at {midi_path.name}\n\n{_sequence_report(sequence, warnings=warnings)}",
                    })
                elif tool["name"] == "play_melody":
                    inp = tool["input"]
                    try:
                        sequence = build_melody(inp)
                    except (TypeError, ValueError) as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": f"Error: {e}",
                            "is_error": True,
                        })
                        continue

                    sequence_id = str(uuid.uuid4())[:8]
                    midi_path = write_sequence_midi(sequence, sequence_id)
                    _sequence_registry[sequence_id] = {"sequence": sequence, "midi_path": midi_path}

                    pill_id = f"p{str(uuid.uuid4())[:6]}"
                    midi_url = f"/sequence/{sequence_id}/download"
                    yield ds_merge_fragment(
                        sequence_pill(sequence_id, sequence["title"], pill_id, sequence["duration_ms"], midi_url, sequence["events"]),
                        selector=f"#{asst_msg_id} .bubble",
                    )

                    assistant_record.append({
                        "type": "sequence",
                        "sequence_id": sequence_id,
                        "title": sequence["title"],
                        "duration_ms": sequence["duration_ms"],
                        "sequence": sequence,
                        "midi_path": str(midi_path),
                    })
                    warnings = _sequence_warnings(sequence, melody=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": f"Created melody. MIDI saved at {midi_path.name}\n\n{_sequence_report(sequence, warnings=warnings)}",
                    })
                elif tool["name"] == "validate_sequence":
                    inp = tool["input"]
                    try:
                        sequence = build_sequence(inp)
                    except (TypeError, ValueError) as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": f"Error: {e}",
                            "is_error": True,
                        })
                        continue

                    warnings = _sequence_warnings(sequence)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": _sequence_report(sequence, warnings=warnings),
                    })

                elif tool["name"] == "search_corpus":
                    from sequencer.midi_io import search_corpus as _search_corpus
                    inp = tool["input"]
                    query = inp.get("query", "")
                    max_results = int(inp.get("max_results", 5))
                    try:
                        results = _search_corpus(query, max_results=max_results)
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": f"Search error: {e}",
                            "is_error": True,
                        })
                        continue
                    if not results:
                        content = f"No corpus results for {query!r}."
                    else:
                        lines = [f"Found {len(results)} result(s) for {query!r}:"]
                        for i, r in enumerate(results, 1):
                            comp = f" ({r['composer']})" if r['composer'] else ""
                            lines.append(f"{i}. {r['title']}{comp} — corpus_path: {r['corpus_path']!r}")
                        lines.append("\nUse import_corpus(corpus_path=...) to load one.")
                        content = "\n".join(lines)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": content,
                    })

                elif tool["name"] == "import_corpus":
                    from sequencer.midi_io import load_corpus_entry
                    from sequencer.abc import to_abc, per_bar_report
                    inp = tool["input"]
                    corpus_path = inp.get("corpus_path", "")
                    try:
                        sequence, dropped = load_corpus_entry(corpus_path)
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": f"Import error: {e}",
                            "is_error": True,
                        })
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
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": "\n".join(lines),
                    })

            current_history.append({"role": "user", "content": tool_results})

            # New text segment after the pill(s)
            seg_idx += 1
            seg_id = f"{asst_msg_id}-t{seg_idx}"
            seg_text = ""
            yield ds_merge_fragment(
                f'<span id="{seg_id}"></span>',
                selector=f"#{asst_msg_id} .bubble",
            )

        db_save_message(req.session_id, "assistant", assistant_record)
        db_update_session_modified(req.session_id)
        if not history:  # first exchange
            generate_title_async(req.session_id, message_text or "Pasted sheet music")
        yield ds_merge_signals({"loading": False})

    return StreamingResponse(generate(), media_type="text/event-stream")
