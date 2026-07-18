"""Tests for sequencer/theory.py — Phase 2 (music21-backed theory layer).

Golden test: sequencer.theory output must match tests/golden_theory.json,
which was snapshotted from the original hand-rolled server.py theory layer.
Any divergence from golden means a behavior change that must be reviewed.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from sequencer.theory import (
    build_chord, chord_note_names, normalize_chord_quality, parse_pitch, CHORD_INTERVALS,
    _parse_chord_symbol_to_root_quality,
)

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_theory.json")

ROOTS = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


class TestGoldenMidi:
    """MIDI note numbers must exactly match the golden snapshot."""

    def test_all_qualities_all_roots(self, golden):
        mismatches = []
        for q in sorted(CHORD_INTERVALS):
            for r in ROOTS:
                expected = golden.get(q, {}).get(r, {})
                if "error" in expected:
                    continue
                got = build_chord(r, q, 4)
                if got != expected.get("midi"):
                    mismatches.append(f"{q}/{r}: got {got} expected {expected.get('midi')}")
        assert not mismatches, "\n".join(mismatches)


class TestGoldenNames:
    """Spelled note names must exactly match the golden snapshot.

    If a test fails here it means music21 spells an enharmonic differently
    than the old code. Review the disagreement — don't just update the expected.
    """

    def test_all_qualities_all_roots(self, golden):
        mismatches = []
        for q in sorted(CHORD_INTERVALS):
            for r in ROOTS:
                expected = golden.get(q, {}).get(r, {})
                if "error" in expected:
                    continue
                got = chord_note_names(r, q, 4)
                if got != expected.get("names"):
                    mismatches.append(f"{q}/{r}: got {got} expected {expected.get('names')}")
        assert not mismatches, "\n".join(mismatches)


class TestNormalizeQuality:
    def test_aliases(self):
        assert normalize_chord_quality("7") == "dominant7"
        assert normalize_chord_quality("dom7") == "dominant7"
        assert normalize_chord_quality("7b9") == "dominant7b9"
        assert normalize_chord_quality("minor2") == "m2"
        assert normalize_chord_quality("tritone") == "A4"

    def test_passthrough(self):
        assert normalize_chord_quality("major") == "major"
        assert normalize_chord_quality("diminished7") == "diminished7"

    def test_root_stripping(self):
        # 'Cmaj' → strip root 'c' → 'maj'; music21 parsing maps it to 'major'
        result = normalize_chord_quality("Cmaj", root="C")
        assert result in ("major", "maj")


class TestParseChordSymbol:
    """Chord symbols must keep their alterations.

    music21 reports an altered chord as a plain chordKind plus a
    chordStepModification; reading only the kind silently downgrades the chord
    (m7b5 -> minor7), so the loop engine comps the wrong harmony with no error.
    """

    @pytest.mark.parametrize("symbol,expected", [
        # The reported bug: half-diminished must not collapse to minor7.
        ("F#m7b5", ("F#", "halfdiminished7")),
        ("Cm7b5", ("C", "halfdiminished7")),
        ("Cmin7b5", ("C", "halfdiminished7")),
        ("Cm7-5", ("C", "halfdiminished7")),
        ("Bbm7b5", ("Bb", "halfdiminished7")),
        # Other alterations lost down the same path.
        ("C7#5", ("C", "dominant7#5")),
        ("C7#9", ("C", "dominant7#9")),
        ("C7b13", ("C", "dominant7b13")),
        ("Cmaj7#11", ("C", "major7#11")),
        ("Ebmaj7#11", ("Eb", "major7#11")),
        # Suffixes music21 cannot parse at all.
        ("C7alt", ("C", "dominant7alt")),
        ("C6/9", ("C", "major69")),
        ("C69", ("C", "major69")),
        ("CmMaj7", ("C", "minormajor7")),
        ("Cmaj9", ("C", "major9")),
        ("C7sus4", ("C", "dominant7sus4")),
        ("Cm11", ("C", "minor11")),
        # A flat root followed by a degree used to be re-read by music21 as a
        # natural root plus an alteration ("Bb13" -> B + b13 -> B major).
        ("Bb13", ("Bb", "dominant13")),
        ("Bb7b13", ("Bb", "dominant7b13")),
        # Suffixes starting with a note letter must survive root-stripping.
        ("A7alt", ("A", "dominant7alt")),
        ("Aalt", ("A", "dominant7alt")),
        # Unaltered symbols keep working.
        ("Cmaj7", ("C", "major7")),
        ("CM7", ("C", "major7")),
        ("C7", ("C", "dominant7")),
        ("Cm", ("C", "minor")),
        ("Cdim7", ("C", "diminished7")),
        ("Cadd9", ("C", "add9")),
    ])
    def test_symbol_parses_to_quality(self, symbol, expected):
        assert _parse_chord_symbol_to_root_quality(symbol) == expected

    def test_every_parsed_quality_is_voiceable(self):
        """A parsed quality must be a real CHORD_INTERVALS key, not a leftover string."""
        for symbol in ("F#m7b5", "C7alt", "Bb13", "Cmaj7#11", "C6/9", "A7alt"):
            root, quality = _parse_chord_symbol_to_root_quality(symbol)
            assert quality in CHORD_INTERVALS, f"{symbol} -> unvoiceable {quality!r}"
            build_chord(root, quality, 4)

    def test_suffix_resolves_the_same_under_every_root(self):
        """A suffix must mean the same chord on every root.

        Root-vs-suffix ambiguity broke this in both directions: music21 read
        "Bb13" as B + b13, and stripping the root off "Ddim7" left "im7".
        """
        suffixes = [
            '', 'm', 'maj7', 'M7', 'm7', '7', 'dim', 'dim7', 'aug', '+',
            'm7b5', 'min7b5', '7alt', 'alt', '7#5', '7#9', '7b9', '7b13',
            'maj7#11', '6/9', '69', 'sus2', 'sus4', '7sus4', 'add9', '6',
            'm6', '9', '11', '13', 'm9', 'maj9', 'mMaj7', 'm11', '5',
        ]
        problems = []
        for suffix in suffixes:
            seen = {}
            for r in ROOTS:
                got_root, got_quality = _parse_chord_symbol_to_root_quality(r + suffix)
                if got_root != r:
                    problems.append(f"{r + suffix}: parsed root as {got_root}")
                if got_quality not in CHORD_INTERVALS:
                    problems.append(f"{r + suffix}: {got_quality!r} is not voiceable")
                seen.setdefault(got_quality, []).append(r)
            if len(seen) > 1:
                problems.append(f"{suffix!r} is root-dependent: {seen}")
        assert not problems, "\n".join(problems)

    def test_halfdiminished_differs_from_minor7(self):
        assert build_chord("F#", "halfdiminished7", 4) != build_chord("F#", "minor7", 4)

    def test_unhandled_alteration_fails_loudly(self):
        """An alteration we can't name must raise, not silently voice a wrong chord."""
        root, quality = _parse_chord_symbol_to_root_quality("C7b5")
        assert quality not in ("dominant7", "major"), "b5 silently dropped"
        if quality not in CHORD_INTERVALS:
            with pytest.raises(ValueError):
                build_chord(root, quality, 4)


class TestParsePitch:
    def test_c4(self):
        root, octave, midi = parse_pitch("C4")
        assert root == "C"
        assert octave == 4
        assert midi == 60

    def test_fsharp3(self):
        root, octave, midi = parse_pitch("F#3")
        assert root == "F#"
        assert octave == 3
        assert midi == 54

    def test_bb5(self):
        root, octave, midi = parse_pitch("Bb5")
        assert root == "Bb"
        assert octave == 5
        assert midi == 82


class TestBuildChord:
    def test_c_major(self):
        assert build_chord("C", "major", 4) == [60, 64, 67]

    def test_c_minor(self):
        assert build_chord("C", "minor", 4) == [60, 63, 67]

    def test_c_dominant7(self):
        assert build_chord("C", "dominant7", 4) == [60, 64, 67, 70]

    def test_unknown_root_raises(self):
        with pytest.raises(ValueError):
            build_chord("X", "major", 4)

    def test_unknown_quality_raises(self):
        with pytest.raises(ValueError):
            build_chord("C", "notaquality", 4)
