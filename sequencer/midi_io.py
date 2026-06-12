"""MIDI and score import utilities (Phase 2).

Functions:
  import_score(path_or_bytes, filename) -> (sequence_dict, dropped_report)
  search_corpus(query, max_results=5) -> list[dict]  (title, path, composer)
  load_corpus_entry(path) -> (sequence_dict, dropped_report)
  fetch_abc_url(url) -> str  (flag-gated; raises ImportError if disabled)

Internal target format: the normalized sequence dict produced by parse_abc()
(same shape as build_sequence output).  music21 objects never leave this module.
"""

from __future__ import annotations

import os
import io
from pathlib import Path
from fractions import Fraction

# music21 is contained here; callers never see m21 objects
from music21 import corpus, converter, stream, note as m21note, chord as m21chord
from music21 import tempo as m21tempo, meter as m21meter, key as m21key

MAX_BARS = 64


# ── Public API ────────────────────────────────────────────────────────────────

def import_score(source: str | bytes | Path, filename: str = "") -> tuple[dict, str]:
    """Import a MIDI, MusicXML, or ABC file and return (sequence_dict, dropped_report).

    source: file path (str/Path) or raw bytes.
    filename: used for format detection when source is bytes.
    """
    if isinstance(source, (str, Path)):
        s = _load_path(Path(source))
    else:
        s = _load_bytes(source, filename)
    return _score_to_sequence(s)


def search_corpus(query: str, max_results: int = 5) -> list[dict]:
    """Search the bundled music21 corpus.

    Returns list of dicts with keys: title, path, composer.
    """
    results = corpus.search(query)
    out = []
    for entry in results:
        meta = entry.metadata
        title = meta.title if meta and meta.title else str(entry.sourcePath)
        composer = meta.composer if meta and hasattr(meta, 'composer') else ""
        out.append({
            "title": title,
            "composer": composer or "",
            "corpus_path": str(entry.sourcePath),
        })
        if len(out) >= max_results:
            break
    return out


def load_corpus_entry(corpus_path: str) -> tuple[dict, str]:
    """Load a corpus entry by its sourcePath and return (sequence_dict, dropped_report)."""
    results = corpus.search(corpus_path)
    entries = list(results)
    if not entries:
        raise ValueError(f"Corpus entry not found: {corpus_path!r}")
    # Find exact match or first with path containing the query
    entry = None
    for e in entries:
        if corpus_path in str(e.sourcePath):
            entry = e
            break
    if entry is None:
        entry = entries[0]
    s = entry.parse()
    return _score_to_sequence(s)


ENABLE_FETCH_ABC = os.environ.get("MIDIMI_ENABLE_FETCH_ABC", "0").lower() in ("1", "true", "yes")


def fetch_abc_url(url: str) -> str:
    """Fetch raw ABC notation from a user-supplied URL (flag-gated off by default).

    Raises RuntimeError if MIDIMI_ENABLE_FETCH_ABC is not set.
    """
    if not ENABLE_FETCH_ABC:
        raise RuntimeError(
            "fetch_abc is disabled. Set MIDIMI_ENABLE_FETCH_ABC=1 to enable user-supplied URL fetching."
        )
    import urllib.request
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ── Internal ──────────────────────────────────────────────────────────────────

def _load_path(path: Path) -> stream.Score:
    return converter.parse(str(path))


def _load_bytes(data: bytes, filename: str) -> stream.Score:
    fmt = None
    name = filename.lower()
    if name.endswith(".mid") or name.endswith(".midi"):
        fmt = "midi"
    elif name.endswith(".xml") or name.endswith(".musicxml"):
        fmt = "musicxml"
    elif name.endswith(".mxl"):
        fmt = "musicxml"
    elif name.endswith(".abc"):
        fmt = "abc"
    s = converter.parse(data, format=fmt)
    return s


def _score_to_sequence(s) -> tuple[dict, str]:
    """Convert a music21 Score/Part/Stream to our internal sequence dict."""
    dropped: list[str] = []

    # Flatten to a single part (melody) — take the top part of a score
    if hasattr(s, 'parts') and len(s.parts) > 1:
        dropped.append(f"Flattened {len(s.parts)} parts to melody (top part only).")
        part = s.parts[0]
    elif hasattr(s, 'parts') and len(s.parts) == 1:
        part = s.parts[0]
    else:
        part = s

    flat = part.flatten()

    # Extract metadata
    title = "Untitled"
    if hasattr(s, 'metadata') and s.metadata and s.metadata.title:
        title = s.metadata.title

    # Tempo
    tempo_bpm = 96.0
    for mm in flat.getElementsByClass(m21tempo.MetronomeMark):
        if mm.number:
            tempo_bpm = float(mm.number)
            break

    # Time signature
    ts_num, ts_den = 4, 4
    for ts in flat.getElementsByClass(m21meter.TimeSignature):
        ts_num = ts.numerator
        ts_den = ts.denominator
        break
    time_signature = f"{ts_num}/{ts_den}"
    beats_per_bar = ts_num * 4 / ts_den

    # Key
    key_str = "C"
    for ks in flat.getElementsByClass(m21key.KeySignature):
        key_str = ks.asKey().tonic.name + ("m" if ks.asKey().mode == "minor" else "")
        break

    # Events: walk notes/chords, cap at MAX_BARS bars
    events: list[dict] = []
    max_offset = MAX_BARS * beats_per_bar  # in quarter-note beats

    for element in flat.notesAndRests:
        offset_beats = float(element.offset)  # music21 offset in quarter notes
        if offset_beats >= max_offset:
            dropped.append(f"Truncated at {MAX_BARS} bars.")
            break
        dur_beats = float(element.quarterLength)
        if dur_beats <= 0:
            continue

        if isinstance(element, m21note.Rest):
            continue  # rests are implicit gaps between events

        if isinstance(element, m21note.Note):
            midi = element.pitch.midi
            name = _pitch_name(element.pitch)
            events.append(_make_event(offset_beats, dur_beats, [midi], [name]))

        elif isinstance(element, m21chord.Chord):
            midis = [p.midi for p in element.pitches]
            names = [_pitch_name(p) for p in element.pitches]
            events.append(_make_event(offset_beats, dur_beats, midis, names))

    if not events:
        raise ValueError("No notes found in score.")

    # Cap per dynamics/etc report
    total_beats = max(e["at_beat"] + e["duration_beats"] for e in events)
    duration_ms = int(total_beats * 60 / tempo_bpm * 1000)

    seq = {
        "title": title,
        "tempo_bpm": tempo_bpm,
        "time_signature": time_signature,
        "time_signature_parts": (ts_num, ts_den),
        "key": key_str,
        "events": events,
        "total_beats": total_beats,
        "duration_ms": duration_ms,
        "abc_errors": [],
    }
    return seq, "\n".join(dropped) if dropped else ""


def _pitch_name(pitch) -> str:
    """Convert a music21 Pitch to our note-name-with-octave format (e.g. 'C#4')."""
    letter = pitch.step
    acc = pitch.accidental
    acc_str = ""
    if acc:
        modifier = acc.modifier  # '#', '##', '-', '--', ''
        acc_str = modifier.replace("-", "b")
    octave = pitch.octave if pitch.octave is not None else 4
    return f"{letter}{acc_str}{octave}"


# ── MIDI export ───────────────────────────────────────────────────────────────

def write_sequence_midi(sequence: dict, sequence_id: str, dest_dir: Path | None = None) -> Path:
    """Serialize a sequence dict to a MIDI file. Returns the Path written."""
    import mido as _mido

    DEFAULT_CHANNEL = 0
    DEFAULT_INSTRUMENT = 0
    MIDI_TICKS_PER_BEAT = 480

    if dest_dir is None:
        dest_dir = Path(__file__).parent.parent / "generated" / "orchestrations"
    dest_dir.mkdir(parents=True, exist_ok=True)
    midi_path = dest_dir / f"{sequence_id}.mid"

    def _midi_meta_text(value: str) -> str:
        replacements = {"–": "-", "—": "-", "‘": "'",
                        "’": "'", "“": '"', "”": '"'}
        text = "".join(replacements.get(ch, ch) for ch in str(value))
        return text.encode("latin-1", "replace").decode("latin-1")

    mid = _mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    track = _mido.MidiTrack()
    mid.tracks.append(track)

    ts_numerator, ts_denominator = sequence["time_signature_parts"]
    track.append(_mido.MetaMessage("track_name", name=_midi_meta_text(sequence["title"]), time=0))
    track.append(_mido.MetaMessage("set_tempo", tempo=_mido.bpm2tempo(sequence["tempo_bpm"]), time=0))
    track.append(_mido.MetaMessage(
        "time_signature",
        numerator=ts_numerator,
        denominator=ts_denominator,
        time=0,
    ))
    track.append(_mido.Message("program_change", channel=DEFAULT_CHANNEL, program=DEFAULT_INSTRUMENT, time=0))

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
        track.append(_mido.Message(msg_type, channel=DEFAULT_CHANNEL, note=note, velocity=velocity, time=delta))
        last_tick = tick

    track.append(_mido.MetaMessage("end_of_track", time=0))
    mid.save(midi_path)
    return midi_path


# ── Recording quantizer ───────────────────────────────────────────────────────

def quantize_recording(
    raw_events: list[dict],
    tempo_bpm: float = 120.0,
    time_signature: str = "4/4",
    grid: float = 0.25,  # grid in beats (0.25 = 1/16th note)
) -> dict:
    """Convert raw MIDI capture events to a sequence dict.

    raw_events: list of {note, on (bool), t (float, monotonic seconds), velocity}
    Returns a sequence dict in the same shape as parse_abc(), plus a 'raw_events' key.

    Algorithm:
    - Pair note_on with note_off (velocity-0 note_on treated as note_off)
    - t0 = first note_on timestamp (so piece starts at beat 0)
    - Convert timestamps to beats using tempo_bpm
    - Snap onset and duration to nearest grid cell
    - Group simultaneous onsets (same grid cell) into chord events
    """
    if not raw_events:
        raise ValueError("No events to quantize")

    sec_per_beat = 60.0 / tempo_bpm

    # Separate on/off, treating velocity-0 as off
    ons: dict[int, list[tuple[float, int]]] = {}   # note -> [(t, velocity), ...]
    offs: dict[int, list[float]] = {}               # note -> [t, ...]

    for ev in raw_events:
        note = ev["note"]
        t = ev["t"]
        is_on = ev.get("on", True) and ev.get("velocity", 1) > 0
        if is_on:
            ons.setdefault(note, []).append((t, ev.get("velocity", 90)))
        else:
            offs.setdefault(note, []).append(t)

    # Find t0 from first note_on
    all_on_times = [t for times in ons.values() for t, _ in times]
    if not all_on_times:
        raise ValueError("No note_on events found")
    t0 = min(all_on_times)

    def to_beat(t: float) -> float:
        return (t - t0) / sec_per_beat

    def snap(beat: float) -> float:
        return round(beat / grid) * grid

    # Build note instances: (on_beat_raw, off_beat_raw, note, velocity)
    instances: list[tuple[float, float, int, int]] = []
    for note, on_list in sorted(ons.items()):
        off_list = sorted(offs.get(note, []))
        for i, (ton, vel) in enumerate(sorted(on_list, key=lambda x: x[0])):
            # Find first off after this on
            toff = None
            for tof in off_list:
                if tof > ton:
                    toff = tof
                    off_list = [x for x in off_list if x != toff or off_list.index(x) != off_list.index(toff)]
                    break
            if toff is None:
                toff = ton + sec_per_beat  # default 1 beat if no off found
            instances.append((to_beat(ton), to_beat(toff), note, vel))

    if not instances:
        raise ValueError("No complete note instances found")

    # Snap to grid; minimum duration = 1 grid cell
    snapped: list[tuple[float, float, int, int]] = []
    for on_b, off_b, note, vel in instances:
        s_on = snap(on_b)
        dur = max(grid, snap(off_b - on_b))
        snapped.append((s_on, dur, note, vel))

    # Group by onset into events (chords share the same onset beat)
    from collections import defaultdict
    onset_groups: dict[float, list[tuple[float, int, int]]] = defaultdict(list)
    for s_on, dur, note, vel in snapped:
        onset_groups[s_on].append((dur, note, vel))

    ts_num, ts_den = (int(x) for x in time_signature.split("/"))

    events: list[dict] = []
    for onset in sorted(onset_groups):
        group = onset_groups[onset]
        # All share the same onset; use median duration for the group
        dur = sorted(d for d, _, _ in group)[len(group) // 2]
        notes_list = sorted(set(n for _, n, _ in group))
        vel = max(v for _, _, v in group)
        from sequencer.theory import midi_note_name as _mnn
        note_names = [_mnn(n, False) for n in notes_list]
        events.append(_make_event(onset, dur, notes_list, note_names, velocity=vel))

    if not events:
        raise ValueError("Quantization produced no events")

    total_beats = max(e["at_beat"] + e["duration_beats"] for e in events)
    duration_ms = int(total_beats * sec_per_beat * 1000)

    return {
        "title": "Recording",
        "tempo_bpm": float(tempo_bpm),
        "time_signature": time_signature,
        "time_signature_parts": (ts_num, ts_den),
        "key": "C",
        "events": events,
        "total_beats": total_beats,
        "duration_ms": duration_ms,
        "abc_errors": [],
        "raw_events": raw_events,
    }


def timing_report(raw_events: list[dict], sequence: dict) -> list[dict]:
    """Generate per-note timing deviation report (quantized vs raw).

    Returns list of {note_name, bar, beat, deviation_ms, early_or_late}.
    """
    if not raw_events or not sequence.get("events"):
        return []

    tempo_bpm = sequence["tempo_bpm"]
    sec_per_beat = 60.0 / tempo_bpm
    ts_num, ts_den = sequence["time_signature_parts"]
    beats_per_bar = ts_num * 4 / ts_den

    ons: list[tuple[float, int, int]] = []
    for ev in raw_events:
        if ev.get("on", True) and ev.get("velocity", 1) > 0:
            ons.append((ev["t"], ev["note"], ev.get("velocity", 90)))

    if not ons:
        return []

    t0 = min(t for t, _, _ in ons)

    report = []
    for ev in sequence["events"]:
        grid_beat = ev["at_beat"]
        for midi_note in ev["notes"]:
            # Find closest raw on for this note
            candidates = [(t, n, v) for t, n, v in ons if n == midi_note]
            if not candidates:
                continue
            raw_t = min(candidates, key=lambda x: abs((x[0] - t0) / sec_per_beat - grid_beat))[0]
            raw_beat = (raw_t - t0) / sec_per_beat
            dev_ms = (raw_beat - grid_beat) * sec_per_beat * 1000
            bar = int(grid_beat / beats_per_bar) + 1
            beat_in_bar = (grid_beat % beats_per_bar) + 1

            from sequencer.theory import midi_note_name as _mnn
            report.append({
                "note_name": _mnn(midi_note, False),
                "bar": bar,
                "beat": round(beat_in_bar, 2),
                "deviation_ms": round(dev_ms, 1),
                "early_or_late": "early" if dev_ms < 0 else ("late" if dev_ms > 0 else "on-time"),
            })

    return report


def _make_event(at_beat: float, duration_beats: float, notes: list[int], note_names: list[str], velocity: int = 90) -> dict:
    root = note_names[0][:-1] if note_names and note_names[0][-1:].isdigit() else (note_names[0] if note_names else "C")
    octave = notes[0] // 12 - 1 if notes else 4
    return {
        "at_beat": at_beat,
        "duration_beats": duration_beats,
        "notes": notes,
        "note_names": note_names,
        "root": root,
        "quality": "note" if len(notes) == 1 else "chord",
        "octave": octave,
        "velocity": velocity,
        "label": note_names[0] if note_names else "",
    }
