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


MAX_IMPORT_VOICES = 4


def _score_to_sequence(s) -> tuple[dict, str]:
    """Convert a music21 Score/Part/Stream to our internal sequence dict.

    Maps up to MAX_IMPORT_VOICES (4) parts onto voices; reports any dropped.
    """
    dropped: list[str] = []

    # Collect parts (up to MAX_IMPORT_VOICES)
    if hasattr(s, 'parts') and len(s.parts) >= 1:
        all_parts = list(s.parts)
    else:
        all_parts = [s]

    if len(all_parts) > MAX_IMPORT_VOICES:
        n_dropped = len(all_parts) - MAX_IMPORT_VOICES
        dropped.append(f"Dropped {n_dropped} part(s) beyond the {MAX_IMPORT_VOICES}-voice limit.")
        all_parts = all_parts[:MAX_IMPORT_VOICES]

    use_voices = len(all_parts) > 1

    # Extract metadata from score
    title = "Untitled"
    if hasattr(s, 'metadata') and s.metadata and s.metadata.title:
        title = s.metadata.title

    # Tempo and time signature from the first (top) part
    top_flat = all_parts[0].flatten()

    tempo_bpm = 96.0
    for mm in top_flat.getElementsByClass(m21tempo.MetronomeMark):
        if mm.number:
            tempo_bpm = float(mm.number)
            break

    ts_num, ts_den = 4, 4
    for ts in top_flat.getElementsByClass(m21meter.TimeSignature):
        ts_num = ts.numerator
        ts_den = ts.denominator
        break
    time_signature = f"{ts_num}/{ts_den}"
    beats_per_bar = ts_num * 4 / ts_den

    key_str = "C"
    for ks in top_flat.getElementsByClass(m21key.KeySignature):
        key_str = ks.asKey().tonic.name + ("m" if ks.asKey().mode == "minor" else "")
        break

    max_offset = MAX_BARS * beats_per_bar

    all_events: list[dict] = []
    voice_decls: list[dict] = []

    # Assign channels (skip 9)
    ch = 0
    for idx, part in enumerate(all_parts):
        vid = str(idx + 1)
        # Try to get a part name
        part_name = ""
        if hasattr(part, 'partName') and part.partName:
            part_name = part.partName
        elif hasattr(part, 'id') and part.id:
            part_name = str(part.id)
        else:
            part_name = f"voice{vid}"

        if ch == 9:
            ch += 1
        voice_decls.append({
            'id': vid,
            'name': part_name,
            'octave_shift': 0,
            'channel': ch,
            'program': 0,
        })
        ch += 1

        flat = part.flatten()
        truncated = False
        for element in flat.notesAndRests:
            offset_beats = float(element.offset)
            if offset_beats >= max_offset:
                if not truncated:
                    dropped.append(f"Part {part_name}: truncated at {MAX_BARS} bars.")
                    truncated = True
                break
            dur_beats = float(element.quarterLength)
            if dur_beats <= 0:
                continue
            if isinstance(element, m21note.Rest):
                continue
            if isinstance(element, m21note.Note):
                midi = element.pitch.midi
                name = _pitch_name(element.pitch)
                evt = _make_event(offset_beats, dur_beats, [midi], [name])
            elif isinstance(element, m21chord.Chord):
                midis = [p.midi for p in element.pitches]
                names = [_pitch_name(p) for p in element.pitches]
                evt = _make_event(offset_beats, dur_beats, midis, names)
            else:
                continue
            if use_voices:
                evt['voice'] = vid
            all_events.append(evt)

    if not all_events:
        raise ValueError("No notes found in score.")

    total_beats = max(e["at_beat"] + e["duration_beats"] for e in all_events)
    duration_ms = int(total_beats * 60 / tempo_bpm * 1000)

    seq: dict = {
        "title": title,
        "tempo_bpm": tempo_bpm,
        "time_signature": time_signature,
        "time_signature_parts": (ts_num, ts_den),
        "key": key_str,
        "events": all_events,
        "total_beats": total_beats,
        "duration_ms": duration_ms,
        "abc_errors": [],
    }
    if use_voices:
        seq["voices"] = voice_decls

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

def write_sequence_midi(
    sequence: dict,
    sequence_id: str,
    dest_dir: Path | None = None,
    expressive: bool = True,
) -> Path:
    """Serialize a sequence dict to a MIDI file. Returns the Path written.

    Multi-voice sequences write a type-1 file with one named track per voice.
    Single-voice sequences write a type-0 file (unchanged behavior).

    expressive=True uses performed timing (fermata stretch etc.); False uses
    written timing (not yet implemented -- reserved for Phase 5.3).
    """
    import mido as _mido

    _DEFAULT_CHANNEL = 0
    _DEFAULT_INSTRUMENT = 0
    MIDI_TICKS_PER_BEAT = 480

    if dest_dir is None:
        dest_dir = Path(__file__).parent.parent / "generated" / "orchestrations"
    dest_dir.mkdir(parents=True, exist_ok=True)
    midi_path = dest_dir / f"{sequence_id}.mid"

    def _midi_meta_text(value: str) -> str:
        replacements = {"–": "-", "—": "-", "'": "'",
                        "'": "'", """: '"', """: '"'}
        text = "".join(replacements.get(ch, ch) for ch in str(value))
        return text.encode("latin-1", "replace").decode("latin-1")

    voices = sequence.get("voices")

    if voices and len(voices) > 1:
        # ── Type-1 MIDI: one track per voice ──────────────────────────────────
        mid = _mido.MidiFile(type=1, ticks_per_beat=MIDI_TICKS_PER_BEAT)

        # Tempo track
        tempo_track = _mido.MidiTrack()
        mid.tracks.append(tempo_track)
        ts_numerator, ts_denominator = sequence["time_signature_parts"]
        tempo_track.append(_mido.MetaMessage("track_name", name=_midi_meta_text(sequence["title"]), time=0))
        tempo_track.append(_mido.MetaMessage("set_tempo", tempo=_mido.bpm2tempo(sequence["tempo_bpm"]), time=0))
        tempo_track.append(_mido.MetaMessage(
            "time_signature",
            numerator=ts_numerator,
            denominator=ts_denominator,
            time=0,
        ))
        tempo_track.append(_mido.MetaMessage("end_of_track", time=0))

        # Group events by voice
        voice_channel_map = {v["id"]: v.get("channel", i) for i, v in enumerate(voices)}
        voice_name_map = {v["id"]: v.get("name", v["id"]) for v in voices}
        events_by_voice: dict[str, list[dict]] = {v["id"]: [] for v in voices}
        for event in sequence["events"]:
            vid = event.get("voice", voices[0]["id"])
            if vid in events_by_voice:
                events_by_voice[vid].append(event)

        for v in voices:
            vid = v["id"]
            channel = voice_channel_map[vid]
            program = v.get("program", 0)
            vtrack = _mido.MidiTrack()
            mid.tracks.append(vtrack)
            vtrack.append(_mido.MetaMessage(
                "track_name", name=_midi_meta_text(voice_name_map[vid]), time=0
            ))
            vtrack.append(_mido.Message(
                "program_change", channel=channel, program=program, time=0
            ))
            midi_events = []
            for event in events_by_voice[vid]:
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
                vtrack.append(_mido.Message(
                    msg_type, channel=channel, note=note, velocity=velocity, time=delta
                ))
                last_tick = tick
            vtrack.append(_mido.MetaMessage("end_of_track", time=0))

    else:
        # ── Type-0 MIDI: single track (unchanged behavior) ────────────────────
        mid = _mido.MidiFile(type=0, ticks_per_beat=MIDI_TICKS_PER_BEAT)
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
        track.append(_mido.Message("program_change", channel=_DEFAULT_CHANNEL, program=_DEFAULT_INSTRUMENT, time=0))

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
            track.append(_mido.Message(msg_type, channel=_DEFAULT_CHANNEL, note=note, velocity=velocity, time=delta))
            last_tick = tick

        track.append(_mido.MetaMessage("end_of_track", time=0))

    mid.save(midi_path)
    return midi_path


# ── Recording quantizer ───────────────────────────────────────────────────────

# Candidate note durations (beats) with complexity penalties: simpler = cheaper.
_TEMPO_DURATIONS = {0.5: 0.0, 1.0: 0.0, 2.0: 0.0, 4.0: 0.0, 1.5: 0.01, 3.0: 0.02,
                    0.75: 0.03, 0.25: 0.04, 1.25: 0.05, 2.5: 0.05}


def estimate_tempo(raw_events: list[dict], lo: float = 50.0, hi: float = 160.0) -> float:
    """Estimate the performed tempo from raw capture events.

    Scores candidate tempos by how closely inter-onset intervals land on simple
    note durations (weighted by interval length, so long notes dominate), with a
    mild prior toward moderate tempos to resolve halving/doubling ambiguity.
    Returns the tempo in bpm, or 120.0 if there are too few onsets to estimate.
    """
    import math

    onsets = sorted(ev["t"] for ev in raw_events
                    if ev.get("on", True) and ev.get("velocity", 1) > 0)
    iois = [b - a for a, b in zip(onsets, onsets[1:]) if b - a > 1e-3]
    if len(iois) < 2:
        return 120.0

    def score(bpm: float) -> float:
        spb = 60.0 / bpm
        total = sum(
            min(abs(ioi / spb - d) / max(ioi / spb, 1e-9) + pen
                for d, pen in _TEMPO_DURATIONS.items()) * ioi
            for ioi in iois
        )
        return total / sum(iois) + 0.02 * abs(math.log2(bpm / 90.0))

    best = min((score(b / 2), b / 2) for b in range(int(lo * 2), int(hi * 2) + 1))
    return round(best[1])


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

    # Clip overlaps: legato playing releases a note after the next one starts.
    # Truncate at the next onset so serialized bars stay beat-valid.
    events.sort(key=lambda e: e["at_beat"])
    for cur, nxt in zip(events, events[1:]):
        gap = nxt["at_beat"] - cur["at_beat"]
        if cur["duration_beats"] > gap:
            cur["duration_beats"] = gap

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
