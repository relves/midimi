"""Chord charts: bars, chord symbols, repeats, and a key — the form layer over the loop.

This is Slice B of docs/plans/jam-ready-tooling-proposal.md. A chart describes *the form*;
`sequencer.loop` plays it. The two stay separate: charts know nothing about audio, and the
loop knows nothing about roman numerals.

Charts are stored as **scale-degree roman numerals plus a key**, not as literal chord
symbols. Three of this slice's requirements fall out of that one choice:

  - transposition is "render with a different key", and spelling stays diatonic (IV in G-flat
    is C-flat, not B);
  - the roman-numeral overlay needs no analysis pass — the numeral *is* the stored form;
  - triad mode and dominant-7 mode are two renderings of the same chart, so Wk2's triad blues
    and Wk3's all-dominant blues cannot drift apart.

A chat-authored chart may instead carry literal `symbol` slots. Those still transpose, by
semitone shift plus key-aware respelling, but that path is lossier — it has to guess the
diatonic intent that a numeral states outright. Built-ins never take it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sequencer.theory import NOTE_NAMES, key_prefers_flats, midi_note_name

# ── Degrees & spelling ────────────────────────────────────────────────────────

_LETTERS = "CDEFGAB"
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Semitones above the tonic for scale degrees 1..7 of the major scale.
_MAJOR_DEGREE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]

_ROMAN_TO_DEGREE = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
}

_ACCIDENTAL_SHIFT = {"b": -1, "♭": -1, "#": 1, "♯": 1}


def _accidental_for(letter: str, target_pc: int) -> str:
    """The accidental that turns `letter` into `target_pc`, in -2..+2."""
    delta = (target_pc - _LETTER_PC[letter]) % 12
    if delta > 6:
        delta -= 12
    if delta < -2 or delta > 2:
        # Beyond a double accidental the diatomic spelling has stopped being useful;
        # fall back to a plain chromatic name at the call site.
        raise ValueError("no reasonable accidental")
    return {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[delta]


def spell_degree(tonic: str, degree: int, chromatic_alteration: int = 0) -> str:
    """Name the note `degree` steps above `tonic`, altered by `chromatic_alteration`.

    Spelling is by letter arithmetic, so the result keeps the diatonic letter the degree
    implies: the IV of Gb is Cb, not B.
    """
    tonic = normalize_root(tonic)
    tonic_letter = tonic[0].upper()
    if tonic_letter not in _LETTER_PC:
        raise ValueError(f"Bad tonic: {tonic!r}")
    tonic_pc = NOTE_NAMES[tonic]

    letter = _LETTERS[(_LETTERS.index(tonic_letter) + degree - 1) % 7]
    semitones = _MAJOR_DEGREE_SEMITONES[degree - 1] + chromatic_alteration
    target_pc = (tonic_pc + semitones) % 12
    try:
        return letter + _accidental_for(letter, target_pc)
    except ValueError:
        return midi_note_name(60 + target_pc, prefer_flats=bool(key_prefers_flats(tonic)))


def normalize_root(root: str) -> str:
    """'bb' -> 'Bb', 'f#' -> 'F#'. Raises on anything that isn't a note name."""
    raw = str(root).strip().replace("♭", "b").replace("♯", "#")
    if not raw:
        raise ValueError("Empty root")
    name = raw[0].upper() + raw[1:].lower().replace("s", "#")
    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown note name: {root!r}")
    return name


def key_tonic(key: str) -> str:
    """The tonic note of a key string like 'F', 'Bb major', 'd minor', 'Dm'."""
    raw = str(key).strip().replace("♭", "b").replace("♯", "#")
    tonic = raw.split()[0] if raw.split() else raw
    # "Dm" / "Dmin" — strip a quality suffix glued to the root.
    head = tonic[:2] if len(tonic) > 1 and tonic[1] in "b#" else tonic[:1]
    return normalize_root(head)


def key_is_minor(key: str) -> bool:
    raw = str(key).strip().lower().replace("♭", "b").replace("♯", "#")
    rest = raw[2:] if len(raw) > 1 and raw[1] in "b#" else raw[1:]
    rest = rest.strip()
    return rest.startswith("m") and not rest.startswith("maj")


# ── Numerals ──────────────────────────────────────────────────────────────────

_NUMERAL_RE = re.compile(r"^([b#♭♯]*)([ivIV]+)(.*)$")

# Suffix → (quality when the numeral is uppercase, quality when lowercase).
_SUFFIX_QUALITY: dict[str, tuple[str, str]] = {
    "":        ("major", "minor"),
    "7":       ("dominant7", "minor7"),
    "maj7":    ("major7", "minormajor7"),
    "M7":      ("major7", "minormajor7"),
    "Δ":       ("major7", "minormajor7"),
    "Δ7":      ("major7", "minormajor7"),
    "6":       ("major6", "minor6"),
    "9":       ("dominant9", "minor9"),
    "sus4":    ("sus4", "sus4"),
    "sus":     ("sus4", "sus4"),
    "sus2":    ("sus2", "sus2"),
    "+":       ("augmented", "augmented"),
    "aug":     ("augmented", "augmented"),
    "°":       ("diminished", "diminished"),
    "dim":     ("diminished", "diminished"),
    "o":       ("diminished", "diminished"),
    "°7":      ("diminished7", "diminished7"),
    "dim7":    ("diminished7", "diminished7"),
    "o7":      ("diminished7", "diminished7"),
    "ø":       ("halfdiminished7", "halfdiminished7"),
    "ø7":      ("halfdiminished7", "halfdiminished7"),
    "m7b5":    ("halfdiminished7", "halfdiminished7"),
    "7b5":     ("halfdiminished7", "halfdiminished7"),
}

# Chord-symbol suffix to append for each quality. `None` means "bare triad".
_QUALITY_SUFFIX: dict[str, str] = {
    "major": "",
    "minor": "m",
    "diminished": "dim",
    "augmented": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "dominant7": "7",
    "major7": "maj7",
    "minor7": "m7",
    "minormajor7": "mMaj7",
    "halfdiminished7": "m7b5",
    "diminished7": "dim7",
    "augmented7": "aug7",
    "major6": "6",
    "minor6": "m6",
    "dominant9": "9",
    "major9": "maj9",
    "minor9": "m9",
    "add9": "add9",
}

# Numeral suffix to display for each quality, mirroring _QUALITY_SUFFIX. Case carries
# major/minor, so the numeral suffix never repeats it.
_QUALITY_NUMERAL_SUFFIX: dict[str, str] = {
    "major": "", "minor": "",
    "diminished": "°", "augmented": "+",
    "sus2": "sus2", "sus4": "sus4",
    "dominant7": "7", "major7": "maj7", "minor7": "7", "minormajor7": "maj7",
    "halfdiminished7": "ø7", "diminished7": "°7", "augmented7": "+7",
    "major6": "6", "minor6": "6",
    "dominant9": "9", "major9": "maj9", "minor9": "9", "add9": "add9",
}

_MINOR_QUALITIES = {"minor", "minor7", "minor6", "minor9", "minormajor7",
                    "diminished", "diminished7", "halfdiminished7"}


@dataclass(frozen=True)
class Numeral:
    """A parsed roman numeral: which degree, how altered, and what quality."""
    degree: int              # 1..7
    alteration: int          # -1 for bIII, +1 for #IV, 0 otherwise
    quality: str             # a key of theory.CHORD_INTERVALS

    @property
    def is_minor_ish(self) -> bool:
        return self.quality in _MINOR_QUALITIES

    def text(self) -> str:
        roman = [k for k, v in _ROMAN_TO_DEGREE.items() if v == self.degree][0]
        if self.is_minor_ish:
            roman = roman.lower()
        prefix = "b" * -self.alteration if self.alteration < 0 else "#" * self.alteration
        return prefix + roman + _QUALITY_NUMERAL_SUFFIX.get(self.quality, "")


def parse_numeral(text: str) -> Numeral:
    """Parse 'I', 'bVII7', 'ii', 'viiø7', 'V7' into a Numeral."""
    raw = str(text).strip()
    match = _NUMERAL_RE.match(raw)
    if not match:
        raise ValueError(f"Not a roman numeral: {text!r}")
    accidentals, roman, suffix = match.groups()

    degree = _ROMAN_TO_DEGREE.get(roman.upper())
    if degree is None:
        raise ValueError(f"Not a roman numeral: {text!r}")

    alteration = sum(_ACCIDENTAL_SHIFT[c] for c in accidentals)
    is_upper = roman[0].isupper()

    suffix = suffix.strip()
    pair = _SUFFIX_QUALITY.get(suffix)
    if pair is None:
        pair = _SUFFIX_QUALITY.get(suffix.lower())
    if pair is None:
        raise ValueError(f"Unknown numeral quality {suffix!r} in {text!r}")
    quality = pair[0] if is_upper else pair[1]

    return Numeral(degree=degree, alteration=alteration, quality=quality)


def numeral_to_symbol(numeral: Numeral, key: str) -> str:
    """Render a numeral as a concrete chord symbol in `key`."""
    root = spell_degree(key_tonic(key), numeral.degree, numeral.alteration)
    return root + _QUALITY_SUFFIX.get(numeral.quality, "")


# ── Modes ─────────────────────────────────────────────────────────────────────

#: Rendering modes. `as_written` keeps whatever the chart says; the others re-quality the
#: *same* form, which is what lets Wk2 (triads) and Wk3 (all dominants) share one chart.
MODES = ("as_written", "triad", "dominant7", "seventh")

_TO_TRIAD = {
    "dominant7": "major", "major7": "major", "major6": "major", "dominant9": "major",
    "major9": "major", "add9": "major",
    "minor7": "minor", "minor6": "minor", "minor9": "minor", "minormajor7": "minor",
    "halfdiminished7": "diminished", "diminished7": "diminished",
    "augmented7": "augmented",
}

_TO_DOM7 = {
    "major": "dominant7", "major6": "dominant7", "major7": "dominant7",
    "minor": "minor7", "minor6": "minor7",
    "diminished": "halfdiminished7",
    "augmented": "augmented7",
}


def apply_mode(numeral: Numeral, mode: str) -> Numeral:
    """Re-quality a numeral for a rendering mode. Degree and alteration never change."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    if mode == "as_written":
        return numeral

    quality = numeral.quality
    if mode == "triad":
        quality = _TO_TRIAD.get(quality, quality)
    elif mode == "dominant7":
        quality = _TO_DOM7.get(quality, quality)
    elif mode == "seventh":
        # Diatonic sevenths: everything gets its own 7th, except that the dominant
        # degree keeps a dominant 7th — that's the whole point of calling it V7.
        triad = _TO_TRIAD.get(quality, quality)
        if triad == "major":
            quality = "dominant7" if numeral.degree == 5 and numeral.alteration == 0 else "major7"
        elif triad == "minor":
            quality = "minor7"
        elif triad == "diminished":
            quality = "halfdiminished7"
        elif triad == "augmented":
            quality = "augmented7"

    return Numeral(degree=numeral.degree, alteration=numeral.alteration, quality=quality)


# ── Symbol → numeral (for charts authored as literal symbols) ─────────────────

def symbol_root_quality(symbol: str) -> tuple[str, str]:
    """Split a chord symbol into (root, quality) using the shared theory parser."""
    from sequencer.theory import _parse_chord_symbol_to_root_quality
    return _parse_chord_symbol_to_root_quality(symbol)


def numeral_for_symbol(symbol: str, key: str) -> Numeral:
    """Analyse a literal chord symbol against `key`, for the roman overlay.

    Picks the degree whose diatonic spelling needs the smallest alteration, so C in the key
    of F reads as V rather than bVI. Ties — where one pitch is reachable as either #n or
    b(n+1) — are broken by how the *symbol itself* spells its root: Bb in C is bVII, while
    A# in C is #VI. When the root is natural, flats win, since bIII/bVI/bVII are the
    borrowings that actually turn up.
    """
    root, quality = symbol_root_quality(symbol)
    root = normalize_root(root)
    tonic = key_tonic(key)
    interval = (NOTE_NAMES[root] - NOTE_NAMES[tonic]) % 12

    preferred_sign = 1 if "#" in root else -1

    best: tuple[int, int, int, int] | None = None
    for degree in range(1, 8):
        alteration = interval - _MAJOR_DEGREE_SEMITONES[degree - 1]
        # Wrap into -6..+5 so degree 1 vs degree 7 compare fairly.
        alteration = ((alteration + 6) % 12) - 6
        if abs(alteration) > 1:
            continue
        mismatch = 0 if alteration == 0 or (alteration > 0) == (preferred_sign > 0) else 1
        candidate = (abs(alteration), mismatch, degree, alteration)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        # Unreachable for 12-TET roots, but be explicit rather than silently wrong.
        raise ValueError(f"Cannot place {symbol!r} in key {key!r}")

    _, _, degree, alteration = best
    return Numeral(degree=degree, alteration=alteration, quality=quality)


# ── Chart model ───────────────────────────────────────────────────────────────

@dataclass
class ChartSection:
    """A run of slots played `repeat` times.

    A slot is {"numeral": "I", "bars": 1} or {"symbol": "F7", "bars": 1}; `beats` may be
    given instead of `bars` for a mid-bar change.
    """
    slots: list[dict]
    repeat: int = 1
    label: str | None = None


@dataclass
class Chart:
    """A form: sections of chords, in a key, at a meter."""
    id: str
    name: str
    key: str
    sections: list[ChartSection]
    time_signature: str = "4/4"
    default_tempo_bpm: float = 100.0
    default_feel: str = "straight"
    default_mode: str = "as_written"
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def bars(self) -> int:
        from sequencer.loop import beats_per_bar
        bpb = beats_per_bar(self.time_signature)
        beats = sum(
            sum(_slot_beats(slot, bpb) for slot in section.slots) * section.repeat
            for section in self.sections
        )
        return int(round(beats / bpb))


def _slot_beats(slot: dict, bpb: float) -> float:
    if "beats" in slot:
        beats = float(slot["beats"])
    else:
        beats = float(slot.get("bars", 1)) * bpb
    if beats <= 0:
        raise ValueError(f"Chart slot has non-positive length: {slot!r}")
    return beats


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_chart(
    chart: Chart,
    key: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Render a chart into concrete bars in `key`, re-qualified for `mode`.

    Returns {key, mode, time_signature, bars, slots}, where each slot carries both the
    chord symbol and its roman numeral — the overlay is a display toggle over one render,
    not a second pass.
    """
    from sequencer.loop import beats_per_bar

    target_key = key or chart.key
    mode = mode or chart.default_mode
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {', '.join(MODES)}")

    bpb = beats_per_bar(chart.time_signature)
    source_tonic = key_tonic(chart.key)
    target_tonic = key_tonic(target_key)
    shift = (NOTE_NAMES[target_tonic] - NOTE_NAMES[source_tonic]) % 12

    slots: list[dict] = []
    at = 0.0
    for section in chart.sections:
        for pass_index in range(max(1, section.repeat)):
            for raw in section.slots:
                numeral = _slot_numeral(raw, chart.key, shift, target_key)
                numeral = apply_mode(numeral, mode)
                beats = _slot_beats(raw, bpb)
                slots.append({
                    "symbol": numeral_to_symbol(numeral, target_key),
                    "numeral": numeral.text(),
                    "degree": numeral.degree,
                    "quality": numeral.quality,
                    "beats": beats,
                    "bars": beats / bpb,
                    "bar": int(at // bpb) + 1,
                    "start_beat": at,
                    "section": section.label,
                    "pass": pass_index + 1,
                })
                at += beats

    return {
        "id": chart.id,
        "name": chart.name,
        "key": target_key,
        "mode": mode,
        "time_signature": chart.time_signature,
        "tempo_bpm": chart.default_tempo_bpm,
        "feel": chart.default_feel,
        "bars": int(round(at / bpb)),
        "slots": slots,
    }


def _slot_numeral(slot: dict, source_key: str, shift: int, target_key: str) -> Numeral:
    """The numeral a slot means, whether it was authored as a numeral or a symbol."""
    if "numeral" in slot:
        return parse_numeral(slot["numeral"])
    if "symbol" in slot:
        # Literal symbols carry no degree, so analyse them against the *source* key and
        # let the numeral do the transposing. This is why built-ins use numerals.
        return numeral_for_symbol(slot["symbol"], source_key)
    raise ValueError(f"Chart slot needs 'numeral' or 'symbol': {slot!r}")


def to_loop_chords(rendered: dict) -> list[dict]:
    """The `chords` list a LoopConfig wants, from a rendered chart."""
    return [{"symbol": slot["symbol"], "beats": slot["beats"]} for slot in rendered["slots"]]


def chart_text(rendered: dict, roman: bool = True) -> str:
    """A monospace grid of the rendered chart, four bars to a line.

    This is what the chat agent reads back, so the learner sees the form rather than a
    list of chord names.
    """
    from sequencer.loop import beats_per_bar

    bpb = beats_per_bar(rendered["time_signature"])
    total_bars = rendered["bars"]
    by_bar: dict[int, list[dict]] = {}
    for slot in rendered["slots"]:
        by_bar.setdefault(slot["bar"], []).append(slot)

    def cell(bar: int) -> str:
        entries = by_bar.get(bar)
        if not entries:
            return "%"  # a held chord — standard lead-sheet shorthand
        label = " ".join(
            f"{e['symbol']}({e['numeral']})" if roman else e["symbol"] for e in entries
        )
        return label

    width = max((len(cell(b)) for b in range(1, total_bars + 1)), default=4) + 2
    lines: list[str] = []
    for start in range(1, total_bars + 1, 4):
        row = "| " + " | ".join(
            cell(bar).ljust(width - 2) for bar in range(start, min(start + 4, total_bars + 1))
        ) + " |"
        lines.append(row)
    header = (
        f"{rendered['name']} — key of {rendered['key']}, {rendered['mode']} mode, "
        f"{rendered['time_signature']}, {total_bars} bars"
    )
    return header + "\n" + "\n".join(lines)


# ── Built-in charts ───────────────────────────────────────────────────────────

def _blues_slots(bars: list[str]) -> list[dict]:
    return [{"numeral": numeral, "bars": 1} for numeral in bars]


BUILTIN_CHARTS: dict[str, Chart] = {}


def _register(chart: Chart) -> Chart:
    BUILTIN_CHARTS[chart.id] = chart
    return chart


_register(Chart(
    id="blues-12-bar",
    name="12-Bar Blues",
    key="F",
    description=(
        "The plain 12-bar blues. Render it in triad mode for Wk2's form introduction and in "
        "dominant7 mode for Wk3's all-dominant blues — same twelve bars either way."
    ),
    tags=["blues", "form", "wk2", "wk3"],
    default_tempo_bpm=100.0,
    default_feel="shuffle",
    default_mode="dominant7",
    sections=[ChartSection(
        label="chorus",
        slots=_blues_slots([
            "I", "I", "I", "I",
            "IV", "IV", "I", "I",
            "V", "IV", "I", "V",
        ]),
    )],
))

_register(Chart(
    id="blues-12-bar-quick-change",
    name="12-Bar Blues (quick change)",
    key="F",
    description="The 12-bar blues with the bar-2 move to IV that most jam sessions expect.",
    tags=["blues", "form", "wk3"],
    default_tempo_bpm=100.0,
    default_feel="shuffle",
    default_mode="dominant7",
    sections=[ChartSection(
        label="chorus",
        slots=_blues_slots([
            "I", "IV", "I", "I",
            "IV", "IV", "I", "I",
            "V", "IV", "I", "V",
        ]),
    )],
))

_register(Chart(
    id="blues-12-bar-slow",
    name="Slow Blues",
    key="F",
    description="The 12-bar blues at a slow-blues tempo, with the bar-12 turnaround spelled out.",
    tags=["blues", "form", "wk3", "wk7"],
    default_tempo_bpm=62.0,
    default_feel="shuffle",
    default_mode="dominant7",
    sections=[ChartSection(
        label="chorus",
        slots=(
            _blues_slots(["I", "IV", "I", "I", "IV", "IV", "I", "I", "V", "IV"])
            + [
                {"numeral": "I", "beats": 2}, {"numeral": "VI", "beats": 2},
                {"numeral": "II", "beats": 2}, {"numeral": "V", "beats": 2},
            ]
        ),
    )],
))

_register(Chart(
    id="ii-v-i",
    name="ii–V–I",
    key="C",
    description="One bar each of ii, V and I, then a bar of rest on I. Wk4/Wk6 comping drill.",
    tags=["cadence", "wk4", "wk6"],
    default_tempo_bpm=90.0,
    default_mode="seventh",
    sections=[ChartSection(
        label="cadence",
        slots=[
            {"numeral": "ii", "bars": 1},
            {"numeral": "V", "bars": 1},
            {"numeral": "I", "bars": 2},
        ],
    )],
))


def get_chart(chart_id: str) -> Chart:
    chart = BUILTIN_CHARTS.get(chart_id)
    if chart is None:
        known = ", ".join(sorted(BUILTIN_CHARTS))
        raise KeyError(f"Unknown chart {chart_id!r}. Known charts: {known}")
    return chart


def list_charts() -> list[dict]:
    return [
        {
            "id": chart.id,
            "name": chart.name,
            "key": chart.key,
            "bars": chart.bars,
            "time_signature": chart.time_signature,
            "default_tempo_bpm": chart.default_tempo_bpm,
            "default_feel": chart.default_feel,
            "default_mode": chart.default_mode,
            "description": chart.description,
            "tags": list(chart.tags),
        }
        for chart in BUILTIN_CHARTS.values()
    ]


def chart_from_spec(spec: dict) -> Chart:
    """Build an ad-hoc chart from a plain dict, for charts the chat agent authors.

    Accepts either a flat `bars`/`slots` list or explicit `sections`, so the agent can emit
    the simple shape when the form has no repeats.
    """
    sections_spec = spec.get("sections")
    if sections_spec is None:
        slots = spec.get("slots") or spec.get("bars") or spec.get("chords")
        if not slots:
            raise ValueError("A chart needs 'slots' (or 'sections')")
        sections_spec = [{"slots": slots}]

    sections = []
    for raw in sections_spec:
        slots = raw.get("slots")
        if not slots:
            raise ValueError(f"Chart section has no slots: {raw!r}")
        sections.append(ChartSection(
            slots=[_normalize_slot(slot) for slot in slots],
            repeat=int(raw.get("repeat", 1)),
            label=raw.get("label"),
        ))

    return Chart(
        id=spec.get("id", "custom"),
        name=spec.get("name", "Custom chart"),
        key=spec.get("key", "C"),
        sections=sections,
        time_signature=spec.get("time_signature", "4/4"),
        default_tempo_bpm=float(spec.get("tempo_bpm", 100.0)),
        default_feel=spec.get("feel", "straight"),
        default_mode=spec.get("mode", "as_written"),
        description=spec.get("description", ""),
        tags=list(spec.get("tags", [])),
    )


def _normalize_slot(slot: Any) -> dict:
    """Accept a bare string slot ('I', 'F7') as well as the dict form."""
    if isinstance(slot, str):
        text = slot.strip()
        try:
            parse_numeral(text)
        except ValueError:
            return {"symbol": text, "bars": 1}
        return {"numeral": text, "bars": 1}
    if not isinstance(slot, dict):
        raise ValueError(f"Chart slot must be a string or an object: {slot!r}")
    return dict(slot)
