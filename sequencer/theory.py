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
from music21 import harmony, pitch as m21pitch, interval as m21interval, scale as m21scale

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
    "minor11":     [0, 3, 7, 10, 14, 17],
    # Altered / extended qualities. music21 reports these as a plain kind plus a
    # chordStepModification, so we resolve them by suffix instead (see
    # _QUALITY_ALIASES) rather than letting the alteration get dropped.
    "dominant7#5":  [0, 4, 8, 10],
    "dominant7#9":  [0, 4, 7, 10, 15],
    "dominant7b13": [0, 4, 7, 10, 20],
    "dominant7alt": [0, 4, 10, 15, 20],   # R 3 b7 #9 b13; 5th omitted
    "dominant7sus4": [0, 5, 7, 10],
    "major7#11":   [0, 4, 7, 11, 18],
    "major69":     [0, 4, 7, 9, 14],
}

_FLAT_ROOTS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb"}

# Enharmonic partners for the five black keys, used to respell a chord root so it
# agrees with the surrounding key signature (A#7 in a blues in F -> Bb7).
_ENHARMONIC_PARTNER: dict[str, str] = {
    "A#": "Bb", "Bb": "A#", "C#": "Db", "Db": "C#", "D#": "Eb",
    "Eb": "D#", "F#": "Gb", "Gb": "F#", "G#": "Ab", "Ab": "G#",
}

_LETTER_PC: dict[str, int] = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

NOTE_NAMES: dict[str, int] = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}

# Double accidentals. `chord_note_names` spells intervals by letter distance, so
# it legitimately produces these (a minor 2nd above Db is Ebb, not D) — without
# them any lookup of such a name raises KeyError instead of returning a pitch class.
for _letter, _pc in _LETTER_PC.items():
    NOTE_NAMES.setdefault(_letter + "##", (_pc + 2) % 12)
    NOTE_NAMES.setdefault(_letter + "bb", (_pc - 2) % 12)
del _letter, _pc

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

    # Altered / extended chord-symbol suffixes. These are resolved here rather
    # than by music21, which reports them as a base chordKind plus a
    # chordStepModification and so loses the alteration when we read only the
    # kind. Keys are post-normalization: lower-cased, with spaces, "-" and "_"
    # already stripped (so "m7-5" arrives as "m75").
    "m7b5": "halfdiminished7", "m7(b5)": "halfdiminished7",
    "min7b5": "halfdiminished7", "mi7b5": "halfdiminished7",
    "minor7b5": "halfdiminished7", "m7flat5": "halfdiminished7",
    "min7flat5": "halfdiminished7", "m75": "halfdiminished7",
    "halfdiminished": "halfdiminished7", "halfdim": "halfdiminished7",
    "halfdiminished7": "halfdiminished7", "hdim7": "halfdiminished7",
    "ø": "halfdiminished7", "ø7": "halfdiminished7",

    "7#5": "dominant7#5", "7sharp5": "dominant7#5", "dom7#5": "dominant7#5",
    "7+5": "dominant7#5", "7+": "dominant7#5",
    "7#9": "dominant7#9", "7sharp9": "dominant7#9", "dom7#9": "dominant7#9",
    "7b13": "dominant7b13", "7flat13": "dominant7b13",
    "7alt": "dominant7alt", "alt": "dominant7alt", "7altered": "dominant7alt",
    "7sus4": "dominant7sus4", "7sus": "dominant7sus4",
    "dom7sus4": "dominant7sus4",

    "maj7#11": "major7#11", "ma7#11": "major7#11", "major7#11": "major7#11",
    "6/9": "major69", "69": "major69", "maj69": "major69",
    "maj6/9": "major69", "major6/9": "major69",

    "mmaj7": "minormajor7", "minmaj7": "minormajor7", "mimaj7": "minormajor7",
    "m(maj7)": "minormajor7", "minormaj7": "minormajor7",
    "maj9": "major9", "ma9": "major9",
    "sus": "sus4", "5": "P5",
    "m11": "minor11", "min11": "minor11", "minor11": "minor11",
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
    15: "A9",   # #9 (spelled as augmented 9th in altered dominants)
    17: "P11",
    18: "A11",  # #11
    20: "m13",  # b13
    21: "M13",
}

# Per-quality overrides: {quality: {semitone: interval_name}}
_QUALITY_INTERVAL_OVERRIDE: dict[str, dict[int, str]] = {
    "diminished":      {6: "d5"},
    "halfdiminished7": {6: "d5"},
    "diminished7":     {6: "d5", 9: "M6"},  # spelled as M6 for readability (old code convention)
    "augmented":       {8: "A5"},
    "augmented7":      {8: "A5"},
    "dominant7#5":     {8: "A5"},
    "dominant7alt":    {8: "A5"},
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
    # `quality` is sometimes handed to us as a whole chord symbol ("Cm7"), so we
    # also consider the string with a leading root stripped. Try the unstripped
    # form first: suffixes like "alt" or "b13" begin with a note letter and would
    # otherwise be mangled into "lt" / "13" when the root happens to be A or B.
    # `quality` is sometimes handed to us as a whole chord symbol ("Cm7"), so we
    # also consider the string with a leading root stripped. Both forms are tried
    # everywhere, unstripped first, because stripping is ambiguous in both
    # directions: "alt"/"b13" begin with a note letter and would be mangled into
    # "lt"/"13" under root A or B, while "dim7" under root D would become "im7".
    candidates = [normalized]
    m21_suffixes = [raw]
    if root:
        normalized_root = root.lower().replace("♭", "b").replace("♯", "#")
        if normalized.startswith(normalized_root):
            candidates.append(normalized[len(normalized_root):])
        if raw.lower().startswith(normalized_root):
            m21_suffixes.append(raw[len(normalized_root):])

    for candidate in candidates:
        if candidate in CHORD_INTERVALS:
            return candidate
        alias = _QUALITY_ALIASES.get(candidate)
        if alias:
            return alias

    # Try music21 ChordSymbol parsing for unrecognized quality strings. Parse
    # against a neutral C root: music21 reads a flat root followed by a degree
    # ("Bb13") as B plus a b13 alteration, which corrupts both root and quality.
    for suffix in m21_suffixes:
        try:
            cs = _parse_chord_symbol(f"C{suffix}")
        except Exception:
            continue
        canonical = _M21_KIND_TO_QUALITY.get(cs.chordKind)
        # chordStepModifications hold alterations (b5, #9, b13, ...) that the
        # chordKind alone does not express. If we have any left over, the kind is
        # a lossy answer -- reject it and let the caller fail loudly on an
        # unknown quality rather than silently voicing the wrong chord.
        if canonical and not cs.chordStepModifications:
            return canonical

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


_NOTE_NAME_RE = re.compile(r"^([A-Ga-g])([#b♯♭]{0,2})$")


def normalize_note_name(s: str) -> str:
    """Normalize a bare note name for grading: letter upper-cased, accidental ascii.

    'f#' / 'F♯' -> 'F#';  'bb' -> 'Bb';  'cB' -> 'Cb'.  Does NOT remap enharmonics
    ('Db' stays 'Db', never 'C#'). Raises ValueError on unparseable input.
    """
    raw = str(s or "").strip().replace("♯", "#").replace("♭", "b")
    m = _NOTE_NAME_RE.match(raw)
    if not m:
        raise ValueError(f"Unknown note name: {s!r}. Use names like C, F#, Bb.")
    letter, accidental = m.groups()
    return letter.upper() + accidental


@lru_cache(maxsize=64)
def major_scale_notes(key: str) -> list[str]:
    """Correctly-spelled note names of `key` major, tonic first, no octave.

    e.g. "D" -> ["D","E","F#","G","A","B","C#"];
         "Gb" -> ["Gb","Ab","Bb","Cb","Db","Eb","F"].
    music21 supplies correct enharmonics (F# major -> E#, Cb major -> all flats).
    """
    tonic = normalize_note_name(key)
    sc = m21scale.MajorScale(tonic)
    # getPitches over one octave returns 8 pitches (tonic repeated at top); take 7.
    pitches = sc.getPitches(f"{tonic}4", f"{tonic}5")[:7]
    return [p.name.replace("-", "b") for p in pitches]


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


def midi_note_name(n: int, prefer_flats: bool = False) -> str:
    """Public alias for _midi_to_name."""
    return _midi_to_name(n, prefer_flats)


def key_prefers_flats(key: str | None) -> bool | None:
    """Does `key` use a flat signature?  None when the key gives no opinion.

    Accepts "F", "Bb major", "d minor", "Dm". Minor keys are resolved through
    their relative major, so D minor (one flat) prefers flats even though a bare
    D root would not.
    """
    if not key:
        return None
    raw = str(key).strip().replace("♭", "b").replace("♯", "#")
    if not raw:
        return None
    parts = raw.split()
    tonic = parts[0]
    rest = " ".join(parts[1:]).lower()
    # "Dm" / "dmin" with no space, versus "D minor".
    is_minor = rest.startswith("m") and not rest.startswith("maj")
    if not is_minor:
        suffix = tonic[2:] if len(tonic) > 1 and tonic[1] in "#b" else tonic[1:]
        if suffix.lower().startswith("m") and not suffix.lower().startswith("maj"):
            is_minor = True
            tonic = tonic[: len(tonic) - len(suffix)]
    tonic = tonic[:1].upper() + tonic[1:]

    accidental = tonic[1:]
    if accidental.startswith("b"):
        return True
    if accidental.startswith("#"):
        return False

    if is_minor:
        # Relative major sits a minor 3rd above; that's the key we know the
        # signature of. Only naturals reach here, so a lookup is enough.
        relative_major = {"A": "C", "B": "D", "C": "Eb", "D": "F",
                          "E": "G", "F": "Ab", "G": "Bb"}.get(tonic)
        tonic = relative_major or tonic
        if tonic.endswith("b"):
            return True

    return tonic in _FLAT_ROOTS


def _respell_root_for_key(root: str, prefer_flats: bool | None) -> str:
    """Swap a black-key root for its enharmonic partner when the key disagrees."""
    if prefer_flats is None:
        return root
    partner = _ENHARMONIC_PARTNER.get(root)
    if partner is None:
        return root
    if prefer_flats and root.endswith("#"):
        return partner
    if not prefer_flats and root.endswith("b"):
        return partner
    return root


def _chord_pc_spellings(root: str, quality: str) -> dict[int, str]:
    """Map each pitch class in the chord to its interval-derived spelling.

    `chord_note_names` already knows that C7alt is C E Bb D# Ab -- flats and
    sharps mixed inside one chord. Voicings reduce to pitch classes and would
    otherwise re-derive names from MIDI under a single flat/sharp flag, which no
    boolean can get right. Carrying this map through the reduction preserves the
    per-tone spelling.
    """
    try:
        names = chord_note_names(root, quality, octave=4)
    except Exception:
        return {}
    spellings: dict[int, str] = {}
    for name in names:
        m = _PITCH_RE.match(name)
        if not m:
            continue
        letter, acc, _ = m.groups()
        pitch_name = f"{letter.upper()}{_normalize_acc(acc)}"
        pc = NOTE_NAMES.get(pitch_name)
        if pc is not None:
            spellings.setdefault(pc, pitch_name)
    return spellings


def _name_midi_spelled(midi: int, spellings: dict[int, str], prefer_flats: bool) -> str:
    """Name a MIDI note using the chord's own spelling, falling back to the flag."""
    pitch_name = spellings.get(midi % 12)
    if pitch_name is None:
        return _midi_to_name(midi, prefer_flats)
    accidental = pitch_name[1:]
    alter = accidental.count("#") - accidental.count("b")
    # Same octave formula as _spell_interval_named: the *unwrapped* pitch value
    # (Cb = -1, B# = 12) is what puts B#3 and Cb5 in the right octave.
    named_pc_value = _LETTER_PC[pitch_name[0]] + alter
    octave = (midi - named_pc_value) // 12 - 1
    return f"{pitch_name}{octave}"


# ── Low-interval limits (LIL) ─────────────────────────────────────────────────
# Each entry: (interval_semitones, min_midi_for_lower_note)
# Below the min, the lower note must be pushed up an octave.
# Source: standard orchestration LIL tables.

_LIL: list[tuple[int, int]] = [
    (1,  None),  # m2  — avoid in low register entirely; push up to stay above E3
    (2,  None),  # M2
    (3,  52),    # m3  — no lower note below E3 (MIDI 52)
    (4,  48),    # M3  — no lower note below C3 (MIDI 48)
    (5,  47),    # P4  — no lower note below B2 (MIDI 47)
    (6,  None),  # A4/tritone
    (7,  None),  # P5  — open 5th, always OK
    (8,  None),
    (9,  None),
    (10, None),
    (11, None),
]

# Map interval semitones to LIL threshold MIDI for lower note
_LIL_THRESHOLD: dict[int, int] = {semi: threshold for semi, threshold in _LIL if threshold is not None}


def _enforce_lil(lower_midi: int, upper_midi: int) -> int:
    """Return (possibly octave-raised) lower_midi that satisfies LIL with upper."""
    interval = (upper_midi - lower_midi) % 12
    threshold = _LIL_THRESHOLD.get(interval)
    if threshold is not None:
        while lower_midi < threshold:
            lower_midi += 12
    return lower_midi


# ── Voicing helper ────────────────────────────────────────────────────────────

def _midi_to_abc_token(midi: int, name: str) -> str:
    """Convert MIDI + note name to an ABC note token (letter+acc+octave, no duration)."""
    m = re.match(r'^([A-G])(#{1,2}|b{1,2})?(-?\d+)$', name)
    if not m:
        octave = midi // 12 - 1
        pc = midi % 12
        letter = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][pc]
        acc_str = ''
    else:
        letter = m.group(1)
        acc_str = (m.group(2) or '').replace('#', '^').replace('b', '_')
        octave = int(m.group(3))
    if octave == 4:
        return f"{acc_str}{letter.upper()}"
    elif octave == 5:
        return f"{acc_str}{letter.lower()}"
    elif octave > 5:
        return f"{acc_str}{letter.lower()}{''.join([chr(39)] * (octave - 5))}"
    else:
        return f"{acc_str}{letter.upper()}{',' * (4 - octave)}"


def _place_close(pcs: list[int], bottom_midi: int) -> list[int]:
    """Stack pitch-classes into close voicing starting at or above bottom_midi."""
    result = []
    current = bottom_midi
    for i, pc in enumerate(pcs):
        diff = (pc - current % 12) % 12
        if i > 0 and diff == 0:
            diff = 12  # avoid unison with previous note
        note = current + diff
        result.append(note)
        current = note
    return result


def voice_chord(
    root: str,
    quality: str,
    melody_note: str | None = None,
    register: str = "mid",
    style: str = "close",
    omit_root: bool = False,
    key: str | None = None,
    _center_midi: int | None = None,
    _rotation: int = 0,
) -> dict:
    """Return a concrete chord voicing.

    Parameters
    ----------
    root : str
        Root note, e.g. "C", "F#", "Bb"
    quality : str
        Chord quality, e.g. "major7", "dominant7"
    melody_note : str | None
        Note name+octave, e.g. "G5".  When given, the top voicing note sits
        a 3rd-6th below the melody note and does not double it.
    register : str
        "low" | "mid" | "high" -- center target when no melody_note.
    style : str
        "close" | "drop2" | "shell" | "spread"
    omit_root : bool
        Drop the root (bass voice has it); keeps 3rd and 7th.
    key : str | None
        Key signature the chord sits in, e.g. "F", "Bb major", "d minor". Only
        affects spelling: it respells a black-key root to match the signature and
        decides accidentals for non-chord tones. Chord tones are spelled from
        their intervals regardless.

    Returns
    -------
    dict with keys:
        notes (list[str])  -- note names bottom to top, e.g. ["G3","B3","D4","F4"]
        midi  (list[int])  -- MIDI numbers bottom to top
        abc   (str)        -- ready-to-paste ABC chord token, e.g. "[GBdf]"
    """
    quality = normalize_chord_quality(quality, root=root)
    intervals = CHORD_INTERVALS.get(quality)
    if intervals is None:
        raise ValueError(f"Unknown chord quality: {quality!r}")

    melody_midi: int | None = None
    if melody_note:
        _, _, melody_midi = parse_pitch(melody_note)

    key_flats = key_prefers_flats(key)
    root = _respell_root_for_key(root, key_flats)
    prefer_flats = key_flats if key_flats is not None else (root in _FLAT_ROOTS)
    root_pc = NOTE_NAMES.get(root)
    if root_pc is None:
        raise ValueError(f"Unknown root: {root!r}")

    spellings = _chord_pc_spellings(root, quality)

    # Build tone pitch-classes in chord order
    tone_pcs = [(root_pc + s) % 12 for s in intervals]

    if omit_root and len(tone_pcs) > 2:
        tone_pcs = tone_pcs[1:]

    if style == "shell":
        if len(intervals) >= 4:
            keep = [0, 1, 3] if not omit_root else [0, 2]
        else:
            keep = list(range(min(3, len(tone_pcs))))
        tone_pcs = [tone_pcs[i] for i in keep if i < len(tone_pcs)]

    # Inversion: rotate so a non-root tone is at the bottom
    if _rotation and len(tone_pcs) > 1:
        rot = _rotation % len(tone_pcs)
        tone_pcs = tone_pcs[rot:] + tone_pcs[:rot]

    # Determine target register
    _REGISTER_CENTER = {"low": 45, "mid": 57, "high": 69}
    if _center_midi is not None:
        center = _center_midi
    else:
        center = _REGISTER_CENTER.get(register, 57)

    if melody_midi is not None:
        # Target: top note a M3 below melody (range 3-6 semitones below)
        top_target = melody_midi - 4
        bottom_target = top_target - (len(tone_pcs) - 1) * 4
    else:
        # Center the voicing centroid at `center`; span ≈ (n-1)*3.5 semitones
        half_span = int((len(tone_pcs) - 1) * 1.75)
        bottom_target = center - half_span
        top_target = center + half_span  # informational only

    # Snap bottom to nearest instance of first PC at or above bottom_target
    first_pc = tone_pcs[0]
    diff = (first_pc - bottom_target % 12) % 12
    bottom = bottom_target + diff
    if bottom < 24:
        bottom += 12

    # Build voicing
    if style == "drop2" and len(tone_pcs) >= 4:
        raw = _place_close(tone_pcs, bottom)
        raw[-2] -= 12
    elif style == "spread" and len(tone_pcs) >= 3:
        raw = _place_close(tone_pcs, bottom - 6)
    else:
        raw = _place_close(tone_pcs, bottom)

    placed = sorted(raw)

    # LIL enforcement from bottom up
    for i in range(len(placed) - 1):
        placed[i] = _enforce_lil(placed[i], placed[i + 1])
    placed.sort()

    # Melody avoidance: remove notes that double the melody PC, push top below melody
    if melody_midi is not None:
        melody_pc = melody_midi % 12
        filtered = [n for n in placed if n % 12 != melody_pc]
        if not filtered:
            filtered = placed  # fallback: skip doubling check

        # Push any notes at or above melody down by octave(s)
        filtered = [n - 12 * max(0, (n - melody_midi + 11) // 12 + 1)
                    if n >= melody_midi else n for n in filtered]
        placed = sorted(filtered)

        # Top note must be at least m3 below melody
        if placed and melody_midi - placed[-1] < 3:
            placed[-1] -= 12
            placed.sort()

    note_names = [_name_midi_spelled(m, spellings, prefer_flats) for m in placed]
    abc_inner = ''.join(_midi_to_abc_token(m, n) for m, n in zip(placed, note_names))

    return {
        "notes": note_names,
        "midi": placed,
        "abc": f"[{abc_inner}]",
    }


def voice_progression(
    chords: list[dict],
    melody: list[str | None] | None = None,
    style: str = "close",
    omit_root: bool = False,
    key: str | None = None,
) -> dict:
    """Voice a chord progression with minimal-motion voice leading.

    Parameters
    ----------
    chords : list[dict]
        Each entry: {symbol: "Cmaj7", beats: 4, melody_note: "e'4"} (melody_note optional).
        symbol is parsed as root+quality.
    melody : list[str|None] | None
        Optional per-chord melody note list (overrides melody_note in chords).
    style : str
        "close" | "drop2" | "shell" | "spread"
    omit_root : bool
        Drop the root from every voicing (rootless comping over a bass line).
    key : str | None
        Key signature the progression sits in, e.g. "F", "Bb major", "d minor".
        Spelling only -- see `voice_chord`.

    Returns
    -------
    dict with keys:
        voicings (list[dict])  -- per-chord: {symbol, beats, notes, midi, abc}
        abc_line (str)         -- one ABC line per chord, suitable for a [V:2] voice
    """
    if not chords:
        return {"voicings": [], "abc_line": ""}

    voicings: list[dict] = []
    prev_midis: list[int] | None = None

    for i, chord_entry in enumerate(chords):
        symbol = chord_entry.get("symbol", "Cmaj7")
        beats = chord_entry.get("beats", 4)
        mel = (melody[i] if melody and i < len(melody) else None) or chord_entry.get("melody_note")

        # Parse symbol into root + quality
        root, quality = _parse_chord_symbol_to_root_quality(symbol)

        # Try all inversions/registers; pick the one with minimum motion from prev
        if prev_midis is None:
            result = voice_chord(root, quality, melody_note=mel, style=style,
                                 omit_root=omit_root, key=key)
        else:
            # Voice near the centroid of the previous chord; try all
            # inversions (rotations) × octave offsets for minimal motion.
            prev_center = int(sum(prev_midis) / len(prev_midis))
            n_tones = len(CHORD_INTERVALS.get(
                normalize_chord_quality(quality, root=root), [0]))
            candidates = []
            for rotation in range(max(1, n_tones)):
                for center_offset in (0, 12, -12, 6, -6):
                    try:
                        r = voice_chord(
                            root, quality, melody_note=mel, style=style,
                            omit_root=omit_root, key=key,
                            _center_midi=prev_center + center_offset,
                            _rotation=rotation,
                        )
                        candidates.append(r)
                    except Exception:
                        pass

            def _motion_cost(candidate: dict) -> float:
                c_midi = sorted(candidate["midi"])
                p_midi = sorted(prev_midis)
                n = min(len(c_midi), len(p_midi))
                return sum(abs(c_midi[j] - p_midi[j]) for j in range(n))

            result = min(candidates, key=_motion_cost) if candidates else voice_chord(
                root, quality, melody_note=mel, style=style, omit_root=omit_root,
                key=key,
            )

        prev_midis = result["midi"]
        voicings.append({
            "symbol": symbol,
            "beats": beats,
            "notes": result["notes"],
            "midi": result["midi"],
            "abc": result["abc"],
        })

    # Build ABC line: chord token + duration
    abc_parts = []
    for v in voicings:
        dur_beats = v["beats"]
        # Duration as ABC multiplier (L:1/4 convention)
        from fractions import Fraction
        frac = Fraction(dur_beats).limit_denominator(64)
        num, den = frac.numerator, frac.denominator
        if den == 1:
            dur_str = '' if num == 1 else str(num)
        elif num == 1:
            dur_str = f'/{den}'
        else:
            dur_str = f'{num}/{den}'
        abc_parts.append(f"{v['abc']}{dur_str}")

    return {
        "voicings": voicings,
        "abc_line": " ".join(abc_parts),
    }


def _parse_chord_symbol_to_root_quality(symbol: str) -> tuple[str, str]:
    """Parse a chord symbol like 'Cmaj7', 'F#m', 'Bbdom7' into (root, quality)."""
    # Try longest root match first
    for root_len in (2, 1):
        candidate_root = symbol[:root_len]
        # Normalize: capitalize first letter
        if len(candidate_root) == 2:
            candidate_root = candidate_root[0].upper() + candidate_root[1]
        else:
            candidate_root = candidate_root.upper()
        if candidate_root in NOTE_NAMES:
            quality_str = symbol[root_len:].strip() or "major"
            quality = normalize_chord_quality(quality_str, root=candidate_root)
            # If normalize returned a dyad-interval key (e.g. 'm7' as minor-7th
            # interval), it's wrong in chord-symbol context. Fall back to music21.
            dyad_keys = {k for k, v in CHORD_INTERVALS.items() if len(v) <= 2}
            if quality in dyad_keys and quality_str:
                try:
                    # Neutral C root: a flat root followed by a degree ("Bb13")
                    # parses as B plus a b13 alteration and corrupts both halves.
                    cs = _parse_chord_symbol(f"C{quality_str}")
                    kind = cs.chordKind
                    canonical = _M21_KIND_TO_QUALITY.get(kind)
                    if canonical and not cs.chordStepModifications:
                        quality = canonical
                except Exception:
                    pass
            return candidate_root, quality
    # Fallback
    return "C", "major"
