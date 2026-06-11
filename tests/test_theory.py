"""Tests for sequencer/theory.py — Phase 2 (music21-backed theory layer).

Golden test: sequencer.theory output must match tests/golden_theory.json,
which was snapshotted from the original hand-rolled server.py theory layer.
Any divergence from golden means a behavior change that must be reviewed.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from sequencer.theory import (
    build_chord, chord_note_names, normalize_chord_quality, parse_pitch, CHORD_INTERVALS
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
