"""Tests for sequencer/abc.py — Phase 1."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import mido
from fractions import Fraction
from sequencer.abc import parse_abc, to_abc, ABCParseError, per_bar_report

# ── Helpers ───────────────────────────────────────────────────────────────────

def _seq_simple(title="Test", tempo=120, ts="4/4", key="C"):
    """Minimal sequence dict for to_abc tests."""
    return {
        "title": title,
        "tempo_bpm": tempo,
        "time_signature": ts,
        "time_signature_parts": tuple(int(x) for x in ts.split("/")),
        "events": [],
        "duration_ms": 0,
        "total_beats": 0.0,
        "key": key,
    }

# ── Pitch/octave convention ───────────────────────────────────────────────────

class TestOctaveConvention:
    def test_middle_c_lowercase(self):
        """c (lowercase) = C4 = MIDI 60."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nc4 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["notes"] == [60]
        assert seq["events"][0]["note_names"] == ["C4"]

    def test_uppercase_c_octave3(self):
        """C (uppercase) = C3 = MIDI 48."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nC4 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["notes"] == [48]
        assert seq["events"][0]["note_names"] == ["C3"]

    def test_octave_raise(self):
        """c' = C5 = MIDI 72."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nc'4 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["notes"] == [72]

    def test_octave_lower(self):
        """C, = C2 = MIDI 36."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nC,4 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["notes"] == [36]

    def test_accidental_sharp(self):
        """^F in key C = F#4."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\n^f4 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["notes"] == [66]  # F#4

    def test_accidental_flat(self):
        """_b = Bb4 = MIDI 70."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\n_b4 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["notes"] == [70]  # Bb4


# ── Key signature ─────────────────────────────────────────────────────────────

class TestKeySignature:
    def test_g_major_f_sharp(self):
        """In G major, bare f → F#."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:G\nf4 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["notes"] == [66]  # F#4

    def test_natural_cancels_key_sig(self):
        """=f in G major → F natural."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:G\n=f4 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["notes"] == [65]  # F4

    def test_bar_accidental_persists(self):
        """Accidental in bar persists to end of bar."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\n^f f f f |"
        seq = parse_abc(abc)
        assert all(e["notes"] == [66] for e in seq["events"])  # all F#4

    def test_bar_accidental_resets(self):
        """Accidental resets at barline."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\n^f f f f | f4 |"
        seq = parse_abc(abc)
        # Bar 1: F# F# F# F#; bar 2: F natural
        assert seq["events"][4]["notes"] == [65]  # F4


# ── Duration ──────────────────────────────────────────────────────────────────

class TestDurations:
    def test_quarter_note_L_quarter(self):
        """With L:1/4, bare note = 1 beat."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nc c c c |"
        seq = parse_abc(abc)
        assert seq["events"][0]["duration_beats"] == pytest.approx(1.0)

    def test_half_note(self):
        """With L:1/4, '2' = 2 beats."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nc2 c2 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["duration_beats"] == pytest.approx(2.0)

    def test_eighth_note(self):
        """With L:1/8, bare note = 0.5 beats."""
        abc = "X:1\nT:test\nM:4/4\nL:1/8\nQ:120\nK:C\ncccccccc |"
        seq = parse_abc(abc)
        assert seq["events"][0]["duration_beats"] == pytest.approx(0.5)

    def test_dotted_quarter(self):
        """3/2 = dotted quarter, / = eighth."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nc3/2 c/ c3/2 c/ |"
        seq = parse_abc(abc)
        assert seq["events"][0]["duration_beats"] == pytest.approx(1.5)
        assert seq["events"][1]["duration_beats"] == pytest.approx(0.5)

    def test_unit_length_default_4_4(self):
        """Without L:, 4/4 meter → L=1/8."""
        abc = "X:1\nT:test\nM:4/4\nQ:120\nK:C\ncccccccc |"
        seq = parse_abc(abc)
        assert seq["events"][0]["duration_beats"] == pytest.approx(0.5)

    def test_unit_length_default_3_8(self):
        """Without L:, 3/8 meter → L=1/16; c2 = 2×1/16 = 0.5 beats."""
        # 3/8 < 3/4 → L defaults to 1/16; beats_per_bar = 1.5
        # c2 = 2 * 1/16 whole = 0.5 quarter beats
        abc = "X:1\nT:test\nM:3/8\nQ:120\nK:C\nc2 c2 c2 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["duration_beats"] == pytest.approx(0.5)

    def test_tie(self):
        """Tied notes merge duration."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nc2-c2 |"
        seq = parse_abc(abc)
        assert len(seq["events"]) == 1
        assert seq["events"][0]["duration_beats"] == pytest.approx(4.0)


# ── Bar accounting ────────────────────────────────────────────────────────────

class TestBarAccounting:
    def test_over_full_bar_raises(self):
        """Bar with 4.5 beats in 4/4 should raise with bar-precise message."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nc c c c c/ |"
        with pytest.raises(ABCParseError) as exc_info:
            parse_abc(abc)
        assert "bar 1" in str(exc_info.value)
        assert "4/4" in str(exc_info.value)

    def test_under_full_bar_raises(self):
        """Bar with 2 beats in 4/4 should raise."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nc c |"
        with pytest.raises(ABCParseError) as exc_info:
            parse_abc(abc)
        assert "bar 1" in str(exc_info.value)

    def test_correct_bar_no_error(self):
        """Correct 4/4 bar should parse without error."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nc c c c |"
        seq = parse_abc(abc)
        assert len(seq["events"]) == 4

    def test_3_4_meter(self):
        """3/4 bar with 3 quarter notes is correct."""
        abc = "X:1\nT:test\nM:3/4\nL:1/4\nQ:120\nK:C\nc c c |"
        seq = parse_abc(abc)
        assert len(seq["events"]) == 3

    def test_error_names_correct_bar(self):
        """Multi-bar: error message names the correct bar."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\nc c c c | c c c |"
        with pytest.raises(ABCParseError) as exc_info:
            parse_abc(abc)
        assert "bar 2" in str(exc_info.value)


# ── Chords ────────────────────────────────────────────────────────────────────

class TestChords:
    def test_chord_parsing(self):
        """[CEG] = C major chord."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\n[ceg]4 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["notes"] == [60, 64, 67]

    def test_chord_with_duration(self):
        """[CEG]2 = chord lasting 2 beats."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\n[ceg]2 [ceg]2 |"
        seq = parse_abc(abc)
        assert seq["events"][0]["duration_beats"] == pytest.approx(2.0)
        assert len(seq["events"]) == 2


# ── Repeats ───────────────────────────────────────────────────────────────────

class TestRepeats:
    def test_simple_repeat_expanded(self):
        """Simple |: ... :| expands to 2x playback."""
        abc = "X:1\nT:test\nM:4/4\nL:1/4\nQ:120\nK:C\n|: c c c c :|"
        seq = parse_abc(abc)
        # 4 events per repeat × 2 = 8 events
        assert len(seq["events"]) == 8
        # Second pass starts at beat 4
        assert seq["events"][4]["at_beat"] == pytest.approx(4.0)


# ── Round-trip ────────────────────────────────────────────────────────────────

class TestRoundTrip:
    def _round_trip(self, abc_in: str):
        seq = parse_abc(abc_in)
        abc_out = to_abc(seq)
        seq2 = parse_abc(abc_out)
        # Compare pitch, timing, duration (not velocity or label)
        assert len(seq["events"]) == len(seq2["events"]), \
            f"Event count mismatch: {len(seq['events'])} vs {len(seq2['events'])}"
        for i, (e1, e2) in enumerate(zip(seq["events"], seq2["events"])):
            assert e1["notes"] == e2["notes"], f"Event {i} notes mismatch"
            assert abs(e1["at_beat"] - e2["at_beat"]) < 0.01, f"Event {i} at_beat mismatch"
            assert abs(e1["duration_beats"] - e2["duration_beats"]) < 0.01, \
                f"Event {i} duration mismatch"

    def test_c_major_scale(self):
        abc = "X:1\nT:C Major Scale\nM:4/4\nL:1/4\nQ:120\nK:C\nc d e f | g a b c' |"
        self._round_trip(abc)

    def test_dotted_rhythms(self):
        abc = "X:1\nT:Dotted\nM:4/4\nL:1/4\nQ:120\nK:C\nc3/2 d/ e3/2 f/ |"
        self._round_trip(abc)

    def test_3_4_meter(self):
        abc = "X:1\nT:Waltz\nM:3/4\nL:1/4\nQ:120\nK:C\nc c c | d d d |"
        self._round_trip(abc)

    def test_chord_round_trip(self):
        abc = "X:1\nT:Chords\nM:4/4\nL:1/4\nQ:120\nK:C\n[ceg]2 [fac']2 |"
        self._round_trip(abc)

    def test_rests_round_trip(self):
        abc = "X:1\nT:Rests\nM:4/4\nL:1/4\nQ:120\nK:C\nc z c z |"
        self._round_trip(abc)


# ── Golden test: Ode to Joy (public domain) ───────────────────────────────────

ODE_TO_JOY_ABC = """\
X:1
T:Ode to Joy
M:4/4
L:1/4
Q:120
K:C
e e f g | g f e d | c c d e | e3/2 d/ d2 |
e e f g | g f e d | c c d e | d3/2 c/ c2 |
"""

# Expected MIDI note sequence for first 8 bars (pitch only, octave 4)
ODE_TO_JOY_NOTES = [
    64, 64, 65, 67,  # e e f g
    67, 65, 64, 62,  # g f e d
    60, 60, 62, 64,  # c c d e
    64, 62, 62,      # e. d d (dotted half: e3/2 d/ d2)
    64, 64, 65, 67,
    67, 65, 64, 62,
    60, 60, 62, 64,
    62, 60, 60,      # d. c c (d3/2 c/ c2)
]


class TestGolden:
    def test_ode_to_joy_notes(self):
        """Verify Ode to Joy parses to correct pitches."""
        seq = parse_abc(ODE_TO_JOY_ABC)
        notes = [e["notes"][0] for e in seq["events"]]
        assert notes == ODE_TO_JOY_NOTES

    def test_ode_to_joy_midi_ticks(self):
        """Verify MIDI output has correct tick values for bar 1."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from server import write_sequence_midi
        import tempfile, uuid

        seq = parse_abc(ODE_TO_JOY_ABC)
        seq_id = str(uuid.uuid4())[:8]
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily redirect GENERATED_DIR
            import server
            orig = server.GENERATED_DIR
            server.GENERATED_DIR = type(orig)(tmpdir)
            try:
                midi_path = write_sequence_midi(seq, seq_id)
                mid = mido.MidiFile(str(midi_path))
                track = mid.tracks[0]
                # Collect note_on events and their absolute ticks
                note_ons = []
                tick = 0
                for msg in track:
                    tick += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        note_ons.append((tick, msg.note))
                # First 4 note_on events in bar 1: e e f g at ticks 0, 480, 960, 1440
                expected_ticks = [0, 480, 960, 1440]
                expected_notes = [64, 64, 65, 67]
                for i, (etick, enote) in enumerate(zip(expected_ticks, expected_notes)):
                    assert note_ons[i][0] == etick, f"note {i} tick mismatch"
                    assert note_ons[i][1] == enote, f"note {i} pitch mismatch"
            finally:
                server.GENERATED_DIR = orig


# ── to_abc output format ──────────────────────────────────────────────────────

class TestToAbc:
    def test_headers_present(self):
        seq = parse_abc("X:1\nT:Hello\nM:4/4\nL:1/4\nQ:100\nK:G\nc c c c |")
        out = to_abc(seq)
        assert "T:Hello" in out
        assert "M:4/4" in out
        assert "L:1/4" in out
        assert "K:G" in out

    def test_four_bars_per_line(self):
        """8 bars → 2 body lines."""
        abc = "X:1\nT:t\nM:4/4\nL:1/4\nQ:120\nK:C\n" + "c c c c |\n" * 8
        seq = parse_abc(abc)
        out = to_abc(seq)
        body = [l for l in out.splitlines() if l and not l.startswith(('X:', 'T:', 'M:', 'L:', 'Q:', 'K:'))]
        assert len(body) == 2

    def test_6_8_meter(self):
        """6/8 with correct beats."""
        abc = "X:1\nT:t\nM:6/8\nL:1/8\nQ:120\nK:C\ncde cde |"
        seq = parse_abc(abc)
        assert len(seq["events"]) == 6
        out = to_abc(seq)
        seq2 = parse_abc(out)
        assert len(seq2["events"]) == 6
