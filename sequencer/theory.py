"""Music theory helpers backed by music21.

Exposes the same surface as server.py's hand-rolled theory layer:
  - normalize_chord_quality(quality, root=None) -> str
  - chord_note_names(root, quality, octave=4) -> list[str]
  - build_chord(root, quality, octave=4) -> list[int]
  - parse_pitch(value) -> (root, octave, midi)

All music21 objects are contained here — callers never see them.
ChordSymbol parses are cached to avoid repeated heavy object construction.
"""

import re
from functools import lru_cache

# music21 is a heavy import; keep it isolated here.
from music21 import harmony, pitch as m21pitch, interval as m21interval

# ── Canonical interval table (same as server.py) ─────────────────────────────
# This is the source of truth for which qualities exist and their intervals.
# music21 handles parsing; we still resolve to these semitone lists.

CHORD_INTERVALS: dict[str, list[int]] = {
    "note":        [0],
    "octave":      [0, 12],
    "m2":          [0, 1],
    "M2":          [0, 2],
    "m3":          [0, 3],
    "M3":          [0, 4],
    "P4":          [0, 5],
    "A4":          [0, 6],
    "P5":          [0, 7],
    "m6":          [0, 8],
    "M6":          [0, 9],
    "m7":          [0, 10],
    "M7":          [0, 11],
    "major":       [0, 4, 7],
    "minor":       [0, 3, 7],
    "diminished":  [0, 3, 6],
    "augmented":   [0, 4, 8],
    "sus2":        [0, 2, 7],
    "sus4":        [0, 5, 7],
    "major7":      [0, 4, 7, 11],
    "dominant7":   [0, 4, 7, 10],
    "minor7":      [0, 3, 7, 10],
    "minormajor7": [0, 3, 7, 11],
    "halfdiminished7": [0, 3, 6, 10],
    "diminished7": [0, 3, 6, 9],
    "augmented7":  [0, 4, 8, 10],
    "major9":      [0, 4, 7, 11, 14],
    "dominant9":   [0, 4, 7, 10, 14],
    "dominant7b9": [0, 4, 7, 10, 13],
    "minor9":      [0, 3, 7, 10, 14],
    "add9":        [0, 4, 7, 14],
    "major6":      [0, 4, 7, 9],
    "minor6":      [0, 3, 7, 9],
    "dominant11":  [0, 4, 7, 10, 14, 17],
    "dominant13":  [0, 4, 7, 10, 14, 17, 21],
}

_FLAT_ROOTS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb"}

NOTE_NAMES: dict[str, int] = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}

# Interval dyad qualities map to music21 interval specs.
# These aren't chord symbols so we handle them separately.
_DYAD_INTERVALS: dict[str, str] = {
    "m2": "m2", "M2": "M2", "m3": "m3", "M3": "M3",
    "P4": "P4", "A4": "A4", "P5": "P5", "m6": "m6",
    "M6": "M6", "m7": "m7", "M7": "M7",
}

# music21 chord symbol type strings → our canonical quality names.
# music21 uses its own type strings from harmony.ChordSymbol.chordKind.
_M21_KIND_TO_QUALITY: dict[str, str] = {
    "major":                        "major",
    "minor":                        "minor",
    "diminished":                   "diminished",
    "augmented":                    "augmented",
    "suspended-second":             "sus2",
    "suspended-fourth":             "sus4",
    "major-seventh":                "major7",
    "dominant":                     "dominant7",   # dominant-seventh
    "minor-seventh":                "minor7",
    "minor-major-seventh":          "minormajor7",
    "half-diminished":              "halfdiminished7",
    "diminished-seventh":           "diminished7",
    "augmented-seventh":            "augmented7",
    "major-ninth":                  "major9",
    "dominant-ninth":               "dominant9",
    "minor-ninth":                  "minor9",
    "major-sixth":                  "major6",
    "minor-sixth":                  "minor6",
    "dominant-11th":                "dominant11",
    "dominant-13th":                "dominant13",
    "add-ninth":                    "add9",
}

# Simple alias normalization for inputs the agent or old server code might send.
_QUALITY_ALIASES: dict[str, str] = {
    "7": "dominant7", "dom7": "dominant7", "dominant": "dominant7",
    "9": "dominant9", "dom9": "dominant9",
    "7b9": "dominant7b9", "7flat9": "dominant7b9", "dom7b9": "dominant7b9",
    "dom7flat9": "dominant7b9", "dominant7flat9": "dominant7b9",
    "minor2": "m2", "halfstep": "m2", "half": "m2",
    "major2": "M2", "wholestep": "M2", "whole": "M2",
    "minor3": "m3", "major3": "M3",
    "perfect4": "P4", "perf4": "P4",
    "tritone": "A4", "aug4": "A4", "dim5": "A4",
    "perfect5": "P5", "perf5": "P5",
    "minor6": "m6", "major6dyad": "M6",
    "minor7dyad": "m7", "major7dyad": "M7",
}

# Map from semitone count to the canonical diatonic interval name (music21 notation).
# These match the standard chord-tone spellings used in server.py's _INTERVAL_DIATONIC.
# Context-dependent cases (6=A4 vs d5, 8=m6 vs A5) use the most common chord spelling.
# For qualities that need the exception (augmented uses A5 for semitone 8), we override
# via _QUALITY_INTERVAL_OVERRIDE below.
_SEMITONE_TO_INTERVAL_NAME: dict[int, str] = {
    0:  "P1",
    1:  "m2",
    2:  "M2",
    3:  "m3",
    4:  "M3",
    5:  "P4",
    6:  "A4",   # augmented 4th (tritone spelled as A4)
    7:  "P5",
    8:  "m6",   # default; augmented chords override to A5
    9:  "M6",
    10: "m7",
    11: "M7",
    12: "P8",
    13: "m9",
    14: "M9",
    17: "P11",
    21: "M13",
}

# Per-quality overrides: {quality: {semitone: interval_name}}
_QUALITY_INTERVAL_OVERRIDE: dict[str, dict[int, str]] = {
    "diminished":      {6: "d5"},
    "halfdiminished7": {6: "d5"},
    "diminished7":     {6: "d5", 9: "M6"},  # spelled as M6 for readability (old code convention)
    "augmented":       {8: "A5"},
    "augmented7":      {8: "A5"},
}

_PITCH_RE = re.compile(r"^([A-Ga-g])([#b♯♭]?)(-?\d+)$")


def normalize_chord_quality(quality: str, root: str | None = None) -> str:
    """Normalize chord shorthand to canonical quality names."""
    raw = str(quality or "").strip()
    if raw in CHORD_INTERVALS:
        return raw

    normalized = (
        raw.lower()
        .replace("♭", "b").replace("♯", "#")
        .replace(" ", "").replace("-", "").replace("_", "")
    )
    if root:
        normalized_root = root.lower().replace("♭", "b").replace("♯", "#")
        if normalized.startswith(normalized_root):
            normalized = normalized[len(normalized_root):]

    alias = _QUALITY_ALIASES.get(normalized)
    if alias:
        return alias

    # Try music21 ChordSymbol parsing for unrecognized quality strings.
    try:
        cs = _parse_chord_symbol(f"{root or 'C'}{raw}")
        kind = cs.chordKind
        canonical = _M21_KIND_TO_QUALITY.get(kind)
        if canonical:
            return canonical
    except Exception:
        pass

    return normalized


@lru_cache(maxsize=256)
def _parse_chord_symbol(symbol: str) -> harmony.ChordSymbol:
    return harmony.ChordSymbol(symbol)


def chord_note_names(root: str, quality: str, octave: int = 4) -> list[str]:
    """Return correctly spelled note names for a chord (e.g. ['C4', 'E4', 'G4'])."""
    quality = normalize_chord_quality(quality, root=root)

    if quality in ("note", "octave"):
        midi_notes = build_chord(root, quality, octave)
        prefer_flats = root in _FLAT_ROOTS
        return [_midi_to_name(m, prefer_flats) for m in midi_notes]

    if quality in _DYAD_INTERVALS:
        return _dyad_note_names(root, quality, octave)

    intervals = CHORD_INTERVALS.get(quality)
    if intervals is None:
        raise ValueError(f"Unknown chord quality: {quality!r}")

    overrides = _QUALITY_INTERVAL_OVERRIDE.get(quality, {})
    root_midi = _root_midi(root, octave)
    result = []
    for semitones in intervals:
        target_midi = root_midi + semitones
        if semitones == 0:
            result.append(f"{root}{octave}")
            continue
        interval_name = overrides.get(semitones, _SEMITONE_TO_INTERVAL_NAME.get(semitones))
        name = _spell_interval_named(root, interval_name, target_midi)
        result.append(name)
    return result


def _spell_interval_named(root: str, interval_name: str | None, target_midi: int) -> str:
    """Spell a note above `root` using a named music21 interval (e.g. 'M3', 'd5')."""
    if interval_name is None:
        return _midi_to_name(target_midi)
    root_pitch = m21pitch.Pitch(root)
    try:
        iobj = m21interval.Interval(interval_name)
        target_pitch = root_pitch.transpose(iobj)
    except Exception:
        return _midi_to_name(target_midi)
    letter = target_pitch.step
    acc = target_pitch.accidental
    acc_str = _normalize_acc(acc.modifier if acc else "")
    # Compute octave using the same formula as the original hand-rolled code:
    # octave = (target_midi - named_pc_value) // 12 - 1
    # This gives the correct octave for enharmonics like B#4 and Cb5.
    _LETTER_PC_LOCAL = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    acc_offset = acc.alter if acc else 0
    natural_pc = _LETTER_PC_LOCAL.get(letter, 0)
    named_pc_value = natural_pc + int(acc_offset)
    oct_actual = (target_midi - named_pc_value) // 12 - 1
    return f"{letter}{acc_str}{oct_actual}"


def _dyad_note_names(root: str, quality: str, octave: int) -> list[str]:
    """Return two note names for an interval dyad using music21 interval spelling."""
    root_midi = _root_midi(root, octave)
    semitones = CHORD_INTERVALS[quality][1]
    root_name = f"{root}{octave}"
    interval_name = _SEMITONE_TO_INTERVAL_NAME.get(semitones)
    upper_name = _spell_interval_named(root, interval_name, root_midi + semitones)
    return [root_name, upper_name]


def build_chord(root: str, quality: str, octave: int = 4) -> list[int]:
    """Return MIDI note numbers for a chord/note/interval."""
    pc = NOTE_NAMES.get(root)
    if pc is None:
        raise ValueError(f"Unknown root note: {root!r}. Use names like C, D#, Gb, Bb.")
    quality = normalize_chord_quality(quality, root=root)
    intervals = CHORD_INTERVALS.get(quality)
    if intervals is None:
        raise ValueError(f"Unknown chord quality: {quality!r}. Supported: {sorted(CHORD_INTERVALS)}")
    midi_root = 12 * (octave + 1) + pc
    return [midi_root + i for i in intervals]


def parse_pitch(value: str) -> tuple[str, int, int]:
    """Parse 'C4', 'F#3', 'Bb5' → (root, octave, midi)."""
    match = _PITCH_RE.match(str(value).strip())
    if not match:
        raise ValueError(f"Unknown pitch: {value!r}. Use names like C4, F#3, or Bb5.")
    letter, accidental, octave_s = match.groups()
    accidental = accidental.replace("♯", "#").replace("♭", "b")
    root = letter.upper() + accidental
    octave = int(octave_s)
    return root, octave, build_chord(root, "note", octave)[0]


def _root_midi(root: str, octave: int) -> int:
    pc = NOTE_NAMES.get(root)
    if pc is None:
        raise ValueError(f"Unknown root note: {root!r}")
    return 12 * (octave + 1) + pc


def _normalize_acc(modifier: str) -> str:
    """Convert music21 accidental modifier to string form (- → b, # → #, -- → bb)."""
    return modifier.replace("-", "b")


def _midi_to_name(midi: int, prefer_flats: bool = False) -> str:
    sharps = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    flats  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    pc = midi % 12
    oct_ = midi // 12 - 1
    return f"{(flats if prefer_flats else sharps)[pc]}{oct_}"
