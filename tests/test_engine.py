"""Tests for sequencer/engine.py — fake-recorder scheduler, bar-range offsets."""

import threading
import time
from fractions import Fraction

import pytest

import sequencer.engine as engine


# ── Fake recorder ─────────────────────────────────────────────────────────────

class FakePlayer:
    def __init__(self):
        self.events: list[tuple[float, str, int, int]] = []  # (time, "on"/"off", note, velocity)
        self._t0 = time.monotonic()
        self._lock = threading.Lock()

    def note_on(self, note: int, velocity: int = 90, channel: int = 0) -> None:
        with self._lock:
            self.events.append((time.monotonic() - self._t0, "on", note, velocity))

    def note_off(self, note: int, channel: int = 0) -> None:
        with self._lock:
            self.events.append((time.monotonic() - self._t0, "off", note, 0))

    def play(self, notes: list[int], duration_ms: int) -> None:
        for n in notes:
            self.note_on(n)
        time.sleep(duration_ms / 1000)
        for n in notes:
            self.note_off(n)


def _make_sequence(events_spec, tempo_bpm=120, time_sig="4/4"):
    """Build a minimal sequence dict for engine tests."""
    ts_num, ts_den = (int(x) for x in time_sig.split("/"))
    evts = []
    for spec in events_spec:
        evts.append({
            "at_beat": spec["at_beat"],
            "duration_beats": spec["duration_beats"],
            "notes": spec["notes"],
            "velocity": spec.get("velocity", 90),
            "note_names": [],
        })
    total_beats = max((e["at_beat"] + e["duration_beats"]) for e in evts) if evts else 0
    spb = 60 / tempo_bpm
    return {
        "title": "Test",
        "tempo_bpm": tempo_bpm,
        "time_signature": time_sig,
        "time_signature_parts": (ts_num, ts_den),
        "total_beats": total_beats,
        "duration_ms": int(total_beats * spb * 1000),
        "events": evts,
    }


@pytest.fixture(autouse=True)
def fake_player():
    fp = FakePlayer()
    engine.set_note_fns(fp.note_on, fp.note_off, fp.play, None)
    yield fp
    engine.stop()


def _wait_for_silence(timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not engine._currently_playing:
            return True
        time.sleep(0.02)
    return False


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_single_note_plays_and_stops(fake_player):
    """A single quarter note at 240 bpm: note_on then note_off."""
    seq = _make_sequence(
        [{"at_beat": 0, "duration_beats": 1, "notes": [60]}],
        tempo_bpm=240,  # 0.25s per beat — fast for tests
    )
    engine.play_sequence_bg("t1", seq)
    assert _wait_for_silence()
    kinds = [e[1] for e in fake_player.events if e[2] == 60]
    assert kinds == ["on", "off"]


def test_event_order(fake_player):
    """Two sequential notes: C then D, note_on/off order preserved."""
    seq = _make_sequence(
        [
            {"at_beat": 0, "duration_beats": 1, "notes": [60]},
            {"at_beat": 1, "duration_beats": 1, "notes": [62]},
        ],
        tempo_bpm=480,  # 0.125s/beat
    )
    engine.play_sequence_bg("t2", seq)
    assert _wait_for_silence()
    notes_in_order = [e[2] for e in fake_player.events if e[1] == "on"]
    assert notes_in_order == [60, 62]


def test_bar_range_filters_events(fake_player):
    """bars='2-2' on a 4-bar sequence plays only bar 2 events."""
    # 4/4 at 480 bpm — each beat = 0.125s, each bar = 0.5s
    seq = _make_sequence(
        [
            {"at_beat": 0, "duration_beats": 1, "notes": [60]},   # bar 1
            {"at_beat": 4, "duration_beats": 1, "notes": [62]},   # bar 2
            {"at_beat": 8, "duration_beats": 1, "notes": [64]},   # bar 3
            {"at_beat": 12, "duration_beats": 1, "notes": [65]},  # bar 4
        ],
        tempo_bpm=480,
        time_sig="4/4",
    )
    engine.play_sequence_bg("t3", seq, bars="2-2")
    assert _wait_for_silence()
    played_notes = {e[2] for e in fake_player.events if e[1] == "on"}
    assert played_notes == {62}


def test_bar_range_offset_timing(fake_player):
    """bars='2-2' should start at t=0, not at the original bar offset."""
    seq = _make_sequence(
        [
            {"at_beat": 0, "duration_beats": 1, "notes": [60]},   # bar 1, should be skipped
            {"at_beat": 4, "duration_beats": 1, "notes": [62]},   # bar 2, should fire ~0s in
        ],
        tempo_bpm=480,
        time_sig="4/4",
    )
    t0 = time.monotonic()
    engine.play_sequence_bg("t4", seq, bars="2-2")
    assert _wait_for_silence()
    on_times = [e[0] for e in fake_player.events if e[1] == "on" and e[2] == 62]
    assert len(on_times) == 1
    # Bar 2 starts at beat 0 of the range; at 480 bpm that's ~0s — should fire within 0.3s
    assert on_times[0] < 0.3


def test_stop_silences_notes(fake_player):
    """Calling stop() during playback silences sounding notes."""
    seq = _make_sequence(
        [{"at_beat": 0, "duration_beats": 100, "notes": [60]}],  # very long note
        tempo_bpm=60,
    )
    engine.play_sequence_bg("t5", seq)
    time.sleep(0.05)
    engine.stop()
    assert _wait_for_silence(timeout=1.0)
    off_events = [e for e in fake_player.events if e[1] == "off" and e[2] == 60]
    assert len(off_events) >= 1
