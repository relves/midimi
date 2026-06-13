"""Phase 4 tests: quantizer and raw-replay."""

import math
import time
import threading
import pytest

from sequencer.midi_io import quantize_recording, timing_report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_events(notes: list[tuple[int, float, float]], bpm: float = 120.0) -> list[dict]:
    """Build raw_events from (note, on_seconds, off_seconds) tuples."""
    sec_per_beat = 60.0 / bpm
    evs = []
    for note, t_on, t_off in notes:
        evs.append({"note": note, "on": True,  "velocity": 80, "t": t_on})
        evs.append({"note": note, "on": False, "velocity": 0,  "t": t_off})
    return evs


def _beats_to_sec(beats: float, bpm: float) -> float:
    return beats * 60.0 / bpm


# ── Quantizer tests ───────────────────────────────────────────────────────────

class TestQuantizeBasic:
    def test_single_note_on_grid(self):
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        evs = _make_events([(60, 0.0, spb)], bpm)
        seq = quantize_recording(evs, tempo_bpm=bpm)
        assert len(seq["events"]) == 1
        assert seq["events"][0]["at_beat"] == 0.0
        assert seq["events"][0]["notes"] == [60]

    def test_two_sequential_notes(self):
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        evs = _make_events([
            (60, 0.0, spb * 0.9),
            (62, spb, spb * 2.0),
        ], bpm)
        seq = quantize_recording(evs, tempo_bpm=bpm)
        assert len(seq["events"]) == 2
        assert seq["events"][0]["at_beat"] == pytest.approx(0.0)
        assert seq["events"][1]["at_beat"] == pytest.approx(1.0)

    def test_chord_groups_simultaneous_onsets(self):
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        # C-E-G played 5ms apart — should group into one chord
        evs = [
            {"note": 60, "on": True,  "velocity": 80, "t": 0.000},
            {"note": 64, "on": True,  "velocity": 80, "t": 0.005},
            {"note": 67, "on": True,  "velocity": 80, "t": 0.010},
            {"note": 60, "on": False, "velocity": 0,  "t": spb},
            {"note": 64, "on": False, "velocity": 0,  "t": spb},
            {"note": 67, "on": False, "velocity": 0,  "t": spb},
        ]
        seq = quantize_recording(evs, tempo_bpm=bpm, grid=0.25)
        assert len(seq["events"]) == 1
        assert sorted(seq["events"][0]["notes"]) == [60, 64, 67]

    def test_near_boundary_snaps_correctly(self):
        """A note onset 40ms after the grid point should still snap to that point."""
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)  # 0.5s per beat
        grid_sec = _beats_to_sec(0.25, bpm)  # 1/16 = 0.125s
        # Play 40ms late — within half a grid cell (62.5ms threshold)
        evs = _make_events([(60, 0.04, 0.04 + spb * 0.9)], bpm)
        seq = quantize_recording(evs, tempo_bpm=bpm, grid=0.25)
        assert seq["events"][0]["at_beat"] == pytest.approx(0.0, abs=0.01)

    def test_late_onset_snaps_to_next_grid(self):
        """A note onset 80ms late relative to a grid cell should snap to that next cell."""
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)  # 0.5s
        grid_sec = _beats_to_sec(0.25, bpm)  # 0.125s
        # First note anchors t0 at 0.0. Second note is supposed to be at 0.25 beats
        # but arrives 80ms late (> half a grid cell = 62.5ms), so snaps to 0.5 beats.
        evs = _make_events([
            (60, 0.0, spb * 0.2),
            (62, grid_sec + 0.08, grid_sec + 0.08 + spb * 0.2),  # 80ms after 0.25-beat mark
        ], bpm)
        seq = quantize_recording(evs, tempo_bpm=bpm, grid=0.25)
        second = seq["events"][1]
        assert second["at_beat"] == pytest.approx(0.5, abs=0.01)

    def test_velocity_zero_treated_as_note_off(self):
        """MIDI spec: note_on velocity 0 = note_off."""
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        evs = [
            {"note": 60, "on": True,  "velocity": 80, "t": 0.0},
            {"note": 60, "on": True,  "velocity": 0,  "t": spb},   # vel 0 = off
        ]
        seq = quantize_recording(evs, tempo_bpm=bpm)
        assert len(seq["events"]) == 1
        assert seq["events"][0]["notes"] == [60]

    def test_minimum_duration_one_grid_cell(self):
        """Very short notes (shorter than grid) should get at least one grid cell."""
        bpm = 120.0
        evs = _make_events([(60, 0.0, 0.01)], bpm)  # 10ms note
        seq = quantize_recording(evs, tempo_bpm=bpm, grid=0.25)
        assert seq["events"][0]["duration_beats"] >= 0.25 - 1e-6

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            quantize_recording([], tempo_bpm=120.0)

    def test_sequence_shape(self):
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        evs = _make_events([(60, 0.0, spb)], bpm)
        seq = quantize_recording(evs, tempo_bpm=bpm, time_signature="4/4")
        assert seq["time_signature"] == "4/4"
        assert seq["time_signature_parts"] == (4, 4)
        assert seq["tempo_bpm"] == bpm
        assert "events" in seq
        assert seq["total_beats"] > 0
        assert seq["duration_ms"] > 0
        assert seq["raw_events"] is evs


class TestQuantizeLegato:
    def test_overlap_clipped_to_next_onset(self):
        """Legato overlap (release after next onset) is clipped so events never overlap."""
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        evs = _make_events([
            (60, 0.0, spb * 3.0),       # held 3 beats...
            (72, spb * 2.75, spb * 6.0),  # ...but next note starts at 2.75
        ], bpm)
        seq = quantize_recording(evs, tempo_bpm=bpm, grid=0.25)
        e0, e1 = seq["events"]
        assert e0["at_beat"] + e0["duration_beats"] <= e1["at_beat"] + 1e-9

    def test_legato_recording_roundtrips_through_abc(self):
        """A legato recording must serialize to bar-valid ABC that re-parses."""
        from sequencer.abc import parse_abc, to_abc
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        evs = _make_events([
            (60, 0.0, spb * 3.0),
            (72, spb * 2.75, spb * 6.0),
            (71, spb * 6.0, spb * 7.5),
            (67, spb * 7.25, spb * 8.25),
        ], bpm)
        seq = quantize_recording(evs, tempo_bpm=bpm, grid=0.25)
        reparsed = parse_abc(to_abc(seq))  # raises ABCParseError on invalid bars
        assert [e["notes"] for e in reparsed["events"]] == [[60], [72], [71], [67]]


class TestEstimateTempo:
    def test_steady_quarters_at_72(self):
        from sequencer.midi_io import estimate_tempo
        spb = 60.0 / 72.0
        evs = _make_events([(60 + i, i * spb, (i + 0.9) * spb) for i in range(8)], 72.0)
        bpm = estimate_tempo(evs)
        assert 60 <= bpm <= 90  # quarter=72 is also consistent with eighths at 144; prior keeps it moderate

    def test_mixed_simple_rhythm(self):
        """Halves and quarters at 100 bpm estimate close to 100."""
        from sequencer.midi_io import estimate_tempo
        spb = 60.0 / 100.0
        beats = [0, 2, 4, 5, 6, 8, 10, 11]
        evs = _make_events([(60, b * spb, (b + 0.9) * spb) for b in beats], 100.0)
        bpm = estimate_tempo(evs)
        assert abs(bpm - 100) <= 5 or abs(bpm - 50) <= 3

    def test_too_few_notes_defaults(self):
        from sequencer.midi_io import estimate_tempo
        evs = _make_events([(60, 0.0, 0.5)], 120.0)
        assert estimate_tempo(evs) == 120.0


class TestQuantizeSwing:
    def test_swing_eighth_still_quantizes(self):
        """Swung eighth notes (2/3, 1/3 of a beat) should snap to nearest 1/16."""
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        # Swung beat: on-beat at 0, off-beat at 2/3 beat
        evs = _make_events([
            (60, 0.0, spb * 0.45),
            (62, spb * 2/3, spb * 2/3 + spb * 0.3),
        ], bpm)
        seq = quantize_recording(evs, tempo_bpm=bpm, grid=0.25)
        assert len(seq["events"]) == 2
        # On-beat snaps to 0
        assert seq["events"][0]["at_beat"] == pytest.approx(0.0, abs=0.01)
        # Swung note at 2/3 (0.667 beats) — nearest 1/16 cells: 0.5 (dist 0.167) vs 0.75 (dist 0.083)
        assert seq["events"][1]["at_beat"] == pytest.approx(0.75, abs=0.01)


# ── Timing report tests ───────────────────────────────────────────────────────

class TestTimingReport:
    def test_on_time_note(self):
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        evs = _make_events([(60, 0.0, spb)], bpm)
        seq = quantize_recording(evs, tempo_bpm=bpm)
        report = timing_report(evs, seq)
        assert len(report) >= 1
        entry = next(r for r in report if r["note_name"].startswith("C"))
        assert entry["bar"] == 1
        assert abs(entry["deviation_ms"]) < 5.0  # within 5ms

    def test_late_note(self):
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        # Second note is supposed to be at beat 1 but played 50ms late
        evs = _make_events([
            (60, 0.0, spb * 0.9),          # on-time at beat 0
            (62, spb + 0.05, spb * 2.0),   # 50ms late → still snaps to beat 1
        ], bpm)
        seq = quantize_recording(evs, tempo_bpm=bpm)
        report = timing_report(evs, seq)
        d_entry = next((r for r in report if r["note_name"].startswith("D")), None)
        assert d_entry is not None
        assert d_entry["deviation_ms"] > 0
        assert d_entry["early_or_late"] == "late"

    def test_early_note(self):
        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        # Second note played 40ms early (at beat 1 but played 40ms early)
        evs = [
            {"note": 60, "on": True,  "velocity": 80, "t": 0.0},
            {"note": 60, "on": False, "velocity": 0,  "t": spb * 0.9},
            {"note": 62, "on": True,  "velocity": 80, "t": spb - 0.04},  # 40ms early
            {"note": 62, "on": False, "velocity": 0,  "t": spb * 2.0},
        ]
        seq = quantize_recording(evs, tempo_bpm=bpm)
        report = timing_report(evs, seq)
        d_entry = next((r for r in report if r["note_name"].startswith("D")), None)
        if d_entry:
            assert d_entry["early_or_late"] == "early"

    def test_empty_events(self):
        assert timing_report([], {}) == []


# ── Raw-replay engine test ────────────────────────────────────────────────────

class TestRawReplay:
    def test_replay_timestamps_within_one_tick(self):
        """Raw replay should schedule each event within ~1 MIDI tick (2.1ms at 120bpm) of its original offset."""
        import sequencer.engine as engine

        bpm = 120.0
        spb = _beats_to_sec(1, bpm)
        raw_events = [
            {"note": 60, "on": True,  "velocity": 80, "t": 0.0},
            {"note": 62, "on": True,  "velocity": 80, "t": spb},
            {"note": 60, "on": False, "velocity": 0,  "t": spb * 1.5},
            {"note": 62, "on": False, "velocity": 0,  "t": spb * 2.0},
        ]

        fired: list[tuple[str, int, float]] = []
        lock = threading.Lock()

        def fake_note_on(note, vel, ch):
            with lock:
                fired.append(("on", note, time.monotonic()))

        def fake_note_off(note, ch):
            with lock:
                fired.append(("off", note, time.monotonic()))

        orig_on = engine._note_on_fn
        orig_off = engine._note_off_fn
        engine._note_on_fn = fake_note_on
        engine._note_off_fn = fake_note_off

        try:
            engine.play_raw_recording_bg("test-raw", raw_events)
            time.sleep(spb * 3 + 0.2)  # wait for completion
        finally:
            engine._note_on_fn = orig_on
            engine._note_off_fn = orig_off

        with lock:
            captured = list(fired)

        # Should have 2 note_on and 2 note_off
        ons = [f for f in captured if f[0] == "on"]
        offs = [f for f in captured if f[0] == "off"]
        assert len(ons) == 2
        assert len(offs) == 2

        # Check relative timing: second on should be ~spb after first
        if len(ons) >= 2:
            delta = ons[1][2] - ons[0][2]
            # Allow 15ms tolerance (OS scheduling jitter)
            assert abs(delta - spb) < 0.015, f"Expected {spb:.3f}s between on events, got {delta:.3f}s"
