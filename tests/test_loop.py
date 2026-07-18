"""Tests for sequencer/loop.py — chart layout, feel, rendered tracks, and the position feed."""

import threading
import time

import pytest

import sequencer.engine as engine
import sequencer.loop as loop
from sequencer.theory import NOTE_NAMES


class FakePlayer:
    def __init__(self):
        self.events: list[tuple[float, str, int, int, int]] = []
        self._t0 = time.monotonic()
        self._lock = threading.Lock()

    def note_on(self, note: int, velocity: int = 90, channel: int = 0) -> None:
        with self._lock:
            self.events.append((time.monotonic() - self._t0, "on", note, velocity, channel))

    def note_off(self, note: int, channel: int = 0) -> None:
        with self._lock:
            self.events.append((time.monotonic() - self._t0, "off", note, 0, channel))

    def play(self, notes: list[int], duration_ms: int) -> None:
        for n in notes:
            self.note_on(n)

    def ons(self, channel: int) -> list[int]:
        with self._lock:
            return [e[2] for e in self.events if e[1] == "on" and e[4] == channel]


@pytest.fixture(autouse=True)
def fake_player():
    fp = FakePlayer()
    engine.set_note_fns(fp.note_on, fp.note_off, fp.play, None)
    yield fp
    loop.stop()


BLUES_F = [
    {"symbol": "F7", "bars": 4},
    {"symbol": "Bb7", "bars": 2},
    {"symbol": "F7", "bars": 2},
    {"symbol": "C7", "bars": 1},
    {"symbol": "Bb7", "bars": 1},
    {"symbol": "F7", "bars": 2},
]


# ── Chart ─────────────────────────────────────────────────────────────────────

def test_build_chart_positions_bars():
    chart = loop.build_chart(BLUES_F, "4/4")
    assert len(chart) == 6
    assert loop.chart_total_beats(chart) == 48  # 12 bars of 4
    assert [s["bar"] for s in chart] == [1, 5, 7, 9, 10, 11]


def test_build_chart_accepts_beats():
    chart = loop.build_chart(
        [{"symbol": "Dm7", "beats": 2}, {"symbol": "G7", "beats": 2}], "4/4"
    )
    assert [s["start_beat"] for s in chart] == [0.0, 2.0]
    assert loop.chart_total_beats(chart) == 4.0


def test_build_chart_rejects_empty():
    with pytest.raises(ValueError):
        loop.build_chart([], "4/4")


def test_three_four_beats_per_bar():
    assert loop.beats_per_bar("3/4") == 3
    assert loop.beats_per_bar("4/4") == 4
    assert loop.beats_per_bar("6/8") == 3


def test_chord_index_at_finds_slot():
    chart = loop.build_chart(BLUES_F, "4/4")
    assert chart[loop.chord_index_at(chart, 0.0)]["symbol"] == "F7"
    assert chart[loop.chord_index_at(chart, 16.0)]["symbol"] == "Bb7"
    assert chart[loop.chord_index_at(chart, 36.0)]["symbol"] == "Bb7"
    assert chart[loop.chord_index_at(chart, 47.9)]["symbol"] == "F7"


# ── Feel ──────────────────────────────────────────────────────────────────────

def test_straight_feel_is_identity():
    for b in (0.0, 0.5, 1.25, 7.75):
        assert loop.apply_feel(b, "straight") == b


def test_shuffle_delays_the_offbeat():
    assert loop.apply_feel(0.0, "shuffle") == pytest.approx(0.0)
    assert loop.apply_feel(0.5, "shuffle") == pytest.approx(2 / 3)
    assert loop.apply_feel(1.0, "shuffle") == pytest.approx(1.0)
    assert loop.apply_feel(3.5, "shuffle") == pytest.approx(3 + 2 / 3)


def test_shuffle_preserves_order():
    positions = [loop.apply_feel(i / 16, "shuffle") for i in range(33)]
    assert positions == sorted(positions)


# ── Rendering ─────────────────────────────────────────────────────────────────

def _cfg(**kw):
    base = dict(chords=BLUES_F, tempo_bpm=120, time_signature="4/4", count_in_bars=0)
    base.update(kw)
    return loop.LoopConfig(**base)


def test_click_hits_every_beat_with_accented_downbeat():
    r = loop.render(_cfg(comp=False, bass=False))
    ons = [a for a in r.actions if a[1] == 1 and a[4] == loop.CLICK_CHANNEL]
    assert len(ons) == 48  # one per beat over 12 bars
    downbeats = [a for a in ons if a[2] == loop.CLICK_DOWNBEAT_NOTE]
    assert len(downbeats) == 12
    assert all(a[3] == loop.CLICK_DOWNBEAT_VELOCITY for a in downbeats)
    assert all(abs(a[0] % 4) < 1e-9 for a in downbeats)


def test_click_can_be_disabled():
    r = loop.render(_cfg(click=False, bass=False))
    assert not [a for a in r.actions if a[4] == loop.CLICK_CHANNEL]


def test_bass_plays_the_chord_root():
    r = loop.render(_cfg(click=False, comp=False))
    ons = [a for a in r.actions if a[1] == 1 and a[4] == loop.BASS_CHANNEL]
    assert ons, "expected bass notes"
    # Bar 5 (beat 16) is Bb7 — the bass should sound a Bb.
    at_16 = [a for a in ons if abs(a[0] - 16.0) < 1e-9]
    assert at_16
    assert at_16[0][2] % 12 == NOTE_NAMES["Bb"]


def test_bass_follows_a_mid_form_chord_change():
    r = loop.render(_cfg(click=False, comp=False))
    ons = [a for a in r.actions if a[1] == 1 and a[4] == loop.BASS_CHANNEL]
    at_36 = [a for a in ons if abs(a[0] - 36.0) < 1e-9]  # bar 10 → Bb7
    assert at_36 and at_36[0][2] % 12 == NOTE_NAMES["Bb"]


def test_charleston_comp_hits_one_and_the_and_of_two():
    r = loop.render(_cfg(click=False, bass=False, comp_style="charleston"))
    hit_beats = sorted({a[0] for a in r.actions if a[1] == 1})
    first_bar = [b for b in hit_beats if b < 4]
    assert first_bar == [0.0, 1.5]


def test_pad_comp_hits_only_chord_changes():
    r = loop.render(_cfg(click=False, bass=False, comp_style="pad"))
    hit_beats = sorted({a[0] for a in r.actions if a[1] == 1})
    assert hit_beats == [0.0, 16.0, 24.0, 32.0, 36.0, 40.0]


def test_rootless_comp_omits_the_root():
    voiced = loop.render(_cfg(click=False, bass=False, rootless=True))
    plain = loop.render(_cfg(click=False, bass=False, rootless=False))
    f7_root = NOTE_NAMES["F"]
    assert f7_root not in [m % 12 for m in voiced.voicings[0]["midi"]]
    assert f7_root in [m % 12 for m in plain.voicings[0]["midi"]]


def test_nothing_sounds_past_the_loop_point():
    r = loop.render(_cfg())
    assert all(a[0] <= r.total_beats + 1e-9 for a in r.actions)


def test_every_note_on_has_a_matching_off():
    r = loop.render(_cfg())
    ons = sorted((a[2], a[4]) for a in r.actions if a[1] == 1)
    offs = sorted((a[2], a[4]) for a in r.actions if a[1] == 0)
    assert ons == offs


def test_shuffle_moves_the_and_of_two_but_not_the_downbeat():
    straight = loop.render(_cfg(click=False, bass=False, feel="straight"))
    shuffled = loop.render(_cfg(click=False, bass=False, feel="shuffle"))
    s_hits = sorted({a[0] for a in straight.actions if a[1] == 1})[:2]
    h_hits = sorted({a[0] for a in shuffled.actions if a[1] == 1})[:2]
    assert s_hits[0] == h_hits[0] == 0.0
    assert s_hits[1] == 1.5
    assert h_hits[1] == pytest.approx(1 + 2 / 3)


def test_count_in_renders_a_bar_of_clicks():
    r = loop.render(_cfg(count_in_bars=1))
    ons = [a for a in r.count_in_actions if a[1] == 1]
    assert len(ons) == 4
    assert [a[0] for a in ons] == [-4.0, -3.0, -2.0, -1.0]
    assert ons[0][2] == loop.CLICK_DOWNBEAT_NOTE
    assert all(a[4] == loop.CLICK_CHANNEL for a in ons)


def test_count_in_clicks_even_when_click_is_off():
    r = loop.render(_cfg(count_in_bars=1, click=False))
    assert len([a for a in r.count_in_actions if a[1] == 1]) == 4


# ── Transport & position feed ─────────────────────────────────────────────────

def _wait_until(predicate, timeout=4.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_position_before_start_reports_not_playing():
    loop.stop()
    loop._config = None
    loop._rendered = None
    assert loop.position() == {"playing": False}


def test_position_advances_through_the_form():
    # 1-beat chords at a very fast tempo so the whole form takes ~0.5s.
    chords = [{"symbol": s, "beats": 1} for s in ("Dm7", "G7", "Cmaj7", "A7")]
    loop.start(loop.LoopConfig(
        chords=chords, tempo_bpm=960, time_signature="4/4",
        count_in_bars=0, click=False, bass=False, repeats=6,
    ))
    seen = set()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        p = loop.position()
        if p.get("playing") and not p.get("count_in"):
            seen.add(p["chord"])
        if len(seen) == 4:
            break
        time.sleep(0.002)
    assert seen == {"Dm7", "G7", "Cmaj7", "A7"}


def test_position_reports_next_chord_and_wraps():
    chords = [{"symbol": "Dm7", "bars": 1}, {"symbol": "G7", "bars": 1}]
    loop.start(loop.LoopConfig(
        chords=chords, tempo_bpm=60, time_signature="4/4",
        count_in_bars=0, click=False, comp=False, bass=False,
    ))
    p = loop.position()
    assert p["chord"] == "Dm7"
    assert p["next_chord"] == "G7"
    assert p["bar"] == 1
    assert p["total_bars"] == 2
    # The last chord's next_chord wraps to the top of the form.
    chart = loop.chart_view()
    assert chart[-1]["symbol"] == "G7"


def test_count_in_position_reports_count_in():
    loop.start(loop.LoopConfig(
        chords=[{"symbol": "C7", "bars": 1}], tempo_bpm=120,
        count_in_bars=1, click=False, comp=False, bass=False,
    ))
    p = loop.position()
    assert p["count_in"] is True
    assert p["count_in_beats"] == 4
    assert 1 <= p["count_in_beat"] <= 4
    assert p["bar"] == 0


def test_count_in_clicks_before_the_comp_enters(fake_player):
    loop.start(loop.LoopConfig(
        chords=[{"symbol": "C7", "bars": 1}], tempo_bpm=480,
        count_in_bars=1, click=False, bass=False, repeats=1,
    ))
    assert _wait_until(lambda: fake_player.ons(loop.COMP_CHANNEL))
    with fake_player._lock:
        first_click = next(e[0] for e in fake_player.events
                           if e[1] == "on" and e[4] == loop.CLICK_CHANNEL)
        first_comp = next(e[0] for e in fake_player.events
                          if e[1] == "on" and e[4] == loop.COMP_CHANNEL)
    assert first_click < first_comp


def test_loop_repeats_the_form(fake_player):
    loop.start(loop.LoopConfig(
        chords=[{"symbol": "C7", "bars": 1}], tempo_bpm=960,
        count_in_bars=0, comp=False, bass=False, repeats=3,
    ))
    assert _wait_until(lambda: len(fake_player.ons(loop.CLICK_CHANNEL)) >= 12)


def test_tracks_use_separate_channels(fake_player):
    loop.start(loop.LoopConfig(
        chords=[{"symbol": "C7", "bars": 1}], tempo_bpm=480,
        count_in_bars=0, repeats=2,
    ))
    assert _wait_until(lambda: fake_player.ons(loop.COMP_CHANNEL)
                       and fake_player.ons(loop.BASS_CHANNEL)
                       and fake_player.ons(loop.CLICK_CHANNEL))
    # One-shot playback lives on channel 0 and must stay clear.
    assert not fake_player.ons(0)


def test_stop_silences_sounding_notes(fake_player):
    loop.start(loop.LoopConfig(
        chords=[{"symbol": "C7", "bars": 4}], tempo_bpm=30,
        count_in_bars=0, click=False, bass=False, comp_style="pad",
    ))
    assert _wait_until(lambda: fake_player.ons(loop.COMP_CHANNEL))
    loop.stop()
    assert not loop.is_running()
    with fake_player._lock:
        ons = [(e[2], e[4]) for e in fake_player.events if e[1] == "on"]
        offs = [(e[2], e[4]) for e in fake_player.events if e[1] == "off"]
    assert sorted(ons) == sorted(offs), "every sounding note should be released on stop"


def test_start_replaces_a_running_loop():
    loop.start(loop.LoopConfig(
        chords=[{"symbol": "C7", "bars": 1}], tempo_bpm=120, count_in_bars=0,
    ))
    loop.start(loop.LoopConfig(
        chords=[{"symbol": "Fmaj7", "bars": 1}], tempo_bpm=120, count_in_bars=0,
    ))
    assert loop.position()["chord"] == "Fmaj7"
    assert loop.is_running()


def test_bad_config_does_not_kill_a_running_loop():
    loop.start(loop.LoopConfig(
        chords=[{"symbol": "C7", "bars": 1}], tempo_bpm=120, count_in_bars=0,
    ))
    with pytest.raises(ValueError):
        loop.start(loop.LoopConfig(chords=[], tempo_bpm=120))
    assert loop.is_running()
    assert loop.position()["chord"] == "C7"
