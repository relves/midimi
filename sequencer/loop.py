"""Looping comp engine: a bar/beat clock that plays a chord chart in time until stopped.

This is the jam-thread primitive (Slice A of docs/plans/jam-ready-tooling-proposal.md).
It runs its own scheduler thread rather than reusing `engine._run_sequence`, because a loop
holds the transport indefinitely and must not block one-shot playback.

Channels are kept disjoint from one-shot playback (which lives on channel 0) so drill
prompts and chord pills can sound *over* a running loop:

    ch 1 — comp (piano voicings)
    ch 2 — bass (chord roots)
    ch 9 — click (GM percussion)

The position feed (`position()`) is derived from wall-clock time rather than pushed from
the scheduler thread, so it stays accurate at any poll rate and costs O(len(chart)).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import sequencer.engine as engine
from sequencer.theory import (
    NOTE_NAMES,
    _parse_chord_symbol_to_root_quality,
    voice_progression,
)

# ── Channels & programs ───────────────────────────────────────────────────────

COMP_CHANNEL = 1
BASS_CHANNEL = 2
CLICK_CHANNEL = 9  # GM percussion

COMP_PROGRAM = 0    # Acoustic Grand Piano
BASS_PROGRAM = 32   # Acoustic Bass

CLICK_DOWNBEAT_NOTE = 76  # High Wood Block
CLICK_BEAT_NOTE = 77      # Low Wood Block
CLICK_DOWNBEAT_VELOCITY = 112
CLICK_BEAT_VELOCITY = 78
CLICK_GATE_BEATS = 0.1

COMP_VELOCITY = 68   # sits under the learner's own playing
BASS_VELOCITY = 84
COMP_GATE = 0.92
BASS_GATE = 0.88

BASS_OCTAVE_BASE = 36  # C2 — roots land in 36..47


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LoopConfig:
    """Everything needed to render and run one loop cycle.

    chords: [{"symbol": "C7", "bars": 1}, ...]; a slot may give `beats` instead of `bars`.
    """
    chords: list[dict]
    tempo_bpm: float = 120.0
    time_signature: str = "4/4"
    feel: str = "straight"          # "straight" | "shuffle"
    click: bool = True
    comp: bool = True
    bass: bool = True
    count_in_bars: int = 1
    comp_style: str = "charleston"  # "charleston" | "pad"
    voicing_style: str = "close"    # passed to voice_progression
    rootless: bool = False          # omit roots from comp voicings (Wk6)
    repeats: int | None = None      # None = loop forever, else stop after N cycles


def parse_time_signature(ts: str) -> tuple[int, int]:
    try:
        num, den = ts.split("/")
        return int(num), int(den)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Bad time signature: {ts!r}") from exc


def beats_per_bar(ts: str) -> float:
    """Beats per bar in quarter-note beats, matching engine.py's convention."""
    num, den = parse_time_signature(ts)
    return num * 4 / den


# ── Chart ─────────────────────────────────────────────────────────────────────

def build_chart(chords: list[dict], time_signature: str) -> list[dict]:
    """Expand chord entries into absolute-positioned slots.

    Returns [{symbol, start_beat, beats, bar}] where `bar` is the 1-indexed bar the
    slot begins in.
    """
    if not chords:
        raise ValueError("A loop needs at least one chord")
    bpb = beats_per_bar(time_signature)
    slots: list[dict] = []
    at = 0.0
    for entry in chords:
        symbol = entry.get("symbol")
        if not symbol:
            raise ValueError(f"Chord entry missing 'symbol': {entry!r}")
        if "beats" in entry:
            length = float(entry["beats"])
        else:
            length = float(entry.get("bars", 1)) * bpb
        if length <= 0:
            raise ValueError(f"Chord {symbol!r} has non-positive length")
        slots.append({
            "symbol": symbol,
            "start_beat": at,
            "beats": length,
            "bar": int(at // bpb) + 1,
        })
        at += length
    return slots


def chart_total_beats(chart: list[dict]) -> float:
    return chart[-1]["start_beat"] + chart[-1]["beats"]


def chord_index_at(chart: list[dict], beat: float) -> int:
    """Index of the slot sounding at `beat` (clamped into the chart)."""
    for i, slot in enumerate(chart):
        if slot["start_beat"] <= beat < slot["start_beat"] + slot["beats"] - 1e-9:
            return i
    return len(chart) - 1 if beat > 0 else 0


# ── Feel ──────────────────────────────────────────────────────────────────────

def apply_feel(beat: float, feel: str) -> float:
    """Map a straight beat position onto the performed grid.

    Shuffle pushes the offbeat 8th to the last third of the beat (a 2:1 triplet feel),
    piecewise-linear so positions between the anchors stay ordered.
    """
    if feel != "shuffle":
        return beat
    whole = int(beat // 1)
    frac = beat - whole
    if frac <= 0.5:
        frac = frac * 4 / 3
    else:
        frac = 2 / 3 + (frac - 0.5) * 2 / 3
    return whole + frac


# ── Track rendering ───────────────────────────────────────────────────────────

def _comp_hit_beats(chart: list[dict], bpb: float, total: float, style: str) -> list[float]:
    """Beats at which the comp track strikes a chord."""
    changes = {slot["start_beat"] for slot in chart}
    if style == "pad":
        return sorted(changes)
    # Charleston: beat 1 and the "and of 2" of every bar, plus any mid-bar chord change.
    hits = set(changes)
    bar_start = 0.0
    while bar_start < total - 1e-9:
        hits.add(bar_start)
        if bpb > 1.5:
            hits.add(bar_start + 1.5)
        bar_start += bpb
    return sorted(h for h in hits if h < total - 1e-9)


def _bass_hit_beats(chart: list[dict], bpb: float, total: float) -> list[float]:
    """Beats at which the bass sounds a root: every half-bar, plus every chord change."""
    step = 2.0 if bpb > 2 else 1.0
    hits = {slot["start_beat"] for slot in chart}
    at = 0.0
    while at < total - 1e-9:
        hits.add(at)
        at += step
    return sorted(h for h in hits if h < total - 1e-9)


def _root_midi(symbol: str) -> int | None:
    root, _quality = _parse_chord_symbol_to_root_quality(symbol)
    pc = NOTE_NAMES.get(root)
    if pc is None:
        return None
    return BASS_OCTAVE_BASE + pc


@dataclass
class RenderedLoop:
    """One cycle of the loop, ready to schedule."""
    chart: list[dict]
    actions: list[tuple[float, int, int, int, int]]  # (beat, 1=on/0=off, note, vel, channel)
    count_in_actions: list[tuple[float, int, int, int, int]]  # beats are negative
    total_beats: float
    beats_per_bar: float
    voicings: list[dict] = field(default_factory=list)


def render(config: LoopConfig) -> RenderedLoop:
    """Turn a LoopConfig into a schedulable set of note actions for one cycle.

    Pure — no audio, no threads. This is the seam the tests exercise.
    """
    chart = build_chart(config.chords, config.time_signature)
    bpb = beats_per_bar(config.time_signature)
    total = chart_total_beats(chart)

    voiced = voice_progression(
        [{"symbol": s["symbol"], "beats": s["beats"]} for s in chart],
        style=config.voicing_style,
        omit_root=config.rootless,
    )
    voicings = voiced["voicings"]

    actions: list[tuple[float, int, int, int, int]] = []

    def add(start: float, end: float, notes: list[int], velocity: int, channel: int) -> None:
        # Clamp inside the cycle so a held chord never leaks past the loop point.
        end = min(end, total)
        if end <= start:
            return
        t_on = apply_feel(start, config.feel)
        t_off = min(apply_feel(end, config.feel), total)
        if t_off <= t_on:
            return
        for note in notes:
            actions.append((t_on, 1, note, velocity, channel))
            actions.append((t_off, 0, note, 0, channel))

    if config.comp:
        hits = _comp_hit_beats(chart, bpb, total, config.comp_style)
        for i, hit in enumerate(hits):
            nxt = hits[i + 1] if i + 1 < len(hits) else total
            notes = voicings[chord_index_at(chart, hit)]["midi"]
            add(hit, hit + (nxt - hit) * COMP_GATE, notes, COMP_VELOCITY, COMP_CHANNEL)

    if config.bass:
        hits = _bass_hit_beats(chart, bpb, total)
        for i, hit in enumerate(hits):
            nxt = hits[i + 1] if i + 1 < len(hits) else total
            root = _root_midi(chart[chord_index_at(chart, hit)]["symbol"])
            if root is None:
                continue
            add(hit, hit + (nxt - hit) * BASS_GATE, [root], BASS_VELOCITY, BASS_CHANNEL)

    if config.click:
        at = 0.0
        while at < total - 1e-9:
            downbeat = abs(at % bpb) < 1e-9
            add(
                at, at + CLICK_GATE_BEATS,
                [CLICK_DOWNBEAT_NOTE if downbeat else CLICK_BEAT_NOTE],
                CLICK_DOWNBEAT_VELOCITY if downbeat else CLICK_BEAT_VELOCITY,
                CLICK_CHANNEL,
            )
            at += 1.0

    actions.sort(key=lambda a: (a[0], a[1]))

    # Count-in always clicks, even when the click track is off — that's its whole job.
    count_in: list[tuple[float, int, int, int, int]] = []
    count_in_beats = config.count_in_bars * bpb
    at = -count_in_beats
    while at < -1e-9:
        downbeat = abs((at + count_in_beats) % bpb) < 1e-9
        note = CLICK_DOWNBEAT_NOTE if downbeat else CLICK_BEAT_NOTE
        vel = CLICK_DOWNBEAT_VELOCITY if downbeat else CLICK_BEAT_VELOCITY
        count_in.append((at, 1, note, vel, CLICK_CHANNEL))
        count_in.append((at + CLICK_GATE_BEATS, 0, note, 0, CLICK_CHANNEL))
        at += 1.0
    count_in.sort(key=lambda a: (a[0], a[1]))

    return RenderedLoop(
        chart=chart,
        actions=actions,
        count_in_actions=count_in,
        total_beats=total,
        beats_per_bar=bpb,
        voicings=voicings,
    )


# ── Transport state ───────────────────────────────────────────────────────────

_state_lock = threading.Lock()
_stop_event = threading.Event()
_thread: threading.Thread | None = None

_config: LoopConfig | None = None
_rendered: RenderedLoop | None = None
_t0: float = 0.0        # monotonic time of loop beat 0 (i.e. after the count-in)
_cycle: int = 0
_running: bool = False


def is_running() -> bool:
    return _running


def start(config: LoopConfig) -> dict:
    """Start (or restart) the loop. Returns the initial position snapshot."""
    global _thread, _config, _rendered, _t0, _cycle, _running

    rendered = render(config)  # raises before we tear down a running loop

    stop()

    engine.set_program(COMP_CHANNEL, COMP_PROGRAM)
    engine.set_program(BASS_CHANNEL, BASS_PROGRAM)

    with _state_lock:
        _config = config
        _rendered = rendered
        _cycle = 0
        _t0 = time.monotonic() + config.count_in_bars * rendered.beats_per_bar * (
            60.0 / config.tempo_bpm
        )
        _running = True

    _stop_event.clear()
    _thread = threading.Thread(target=_run, args=(config, rendered), daemon=True)
    _thread.start()
    return position()


def stop() -> None:
    """Stop the loop and wait for the scheduler thread to release its notes."""
    global _thread, _running
    _stop_event.set()
    thread = _thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=2.0)
    _thread = None
    _running = False


def _fire(kind: int, note: int, velocity: int, channel: int) -> None:
    if kind:
        if engine._note_on_fn:
            engine._note_on_fn(note, velocity, channel)
    else:
        if engine._note_off_fn:
            engine._note_off_fn(note, channel)


def _run(config: LoopConfig, rendered: RenderedLoop) -> None:
    global _cycle, _running
    spb = 60.0 / config.tempo_bpm
    with _state_lock:
        t0 = _t0
    sounding: set[tuple[int, int]] = set()

    def play_action(beat: float, base: float, kind: int, note: int, vel: int, ch: int) -> bool:
        """Sleep until the action's moment, then fire it. False if we were stopped."""
        sleep_for = base + beat * spb - time.monotonic()
        if sleep_for > 0:
            _stop_event.wait(sleep_for)
        if _stop_event.is_set():
            return False
        _fire(kind, note, vel, ch)
        if kind:
            sounding.add((note, ch))
        else:
            sounding.discard((note, ch))
        return True

    try:
        for beat, kind, note, vel, ch in rendered.count_in_actions:
            if not play_action(beat, t0, kind, note, vel, ch):
                return

        cycle = 0
        while not _stop_event.is_set():
            if config.repeats is not None and cycle >= config.repeats:
                break
            with _state_lock:
                _cycle = cycle
            base = t0 + cycle * rendered.total_beats * spb
            for beat, kind, note, vel, ch in rendered.actions:
                if not play_action(beat, base, kind, note, vel, ch):
                    return
            cycle += 1
    finally:
        for note, ch in sounding:
            if engine._note_off_fn:
                engine._note_off_fn(note, ch)
        _running = False


# ── Position feed ─────────────────────────────────────────────────────────────

def position() -> dict[str, Any]:
    """Current transport position. Safe to poll at any rate; O(len(chart)).

    Bar/beat are reported on the *straight* grid even under shuffle — the feel changes
    where notes land, not where the learner counts.
    """
    with _state_lock:
        config, rendered, t0, cycle, running = _config, _rendered, _t0, _cycle, _running

    if config is None or rendered is None:
        return {"playing": False}

    spb = 60.0 / config.tempo_bpm
    now = time.monotonic()
    base = {
        "playing": running,
        "tempo_bpm": config.tempo_bpm,
        "time_signature": config.time_signature,
        "feel": config.feel,
        "total_bars": int(round(rendered.total_beats / rendered.beats_per_bar)),
        "click": config.click,
        "loop_count": cycle,
    }

    if now < t0:
        remaining_beats = (t0 - now) / spb
        count_in_beats = config.count_in_bars * rendered.beats_per_bar
        elapsed = count_in_beats - remaining_beats
        first = rendered.chart[0]
        return {
            **base,
            "count_in": True,
            "count_in_beat": int(elapsed) + 1,
            "count_in_beats": int(count_in_beats),
            "bar": 0,
            "beat": 0,
            "chord": first["symbol"],
            "next_chord": first["symbol"],
        }

    elapsed_beats = (now - t0) / spb
    pos = elapsed_beats % rendered.total_beats
    bpb = rendered.beats_per_bar
    idx = chord_index_at(rendered.chart, pos)
    slot = rendered.chart[idx]
    nxt = rendered.chart[(idx + 1) % len(rendered.chart)]

    return {
        **base,
        "count_in": False,
        "bar": int(pos // bpb) + 1,
        "beat": int(pos % bpb) + 1,
        "beat_float": pos % bpb,
        "position_beats": pos,
        "chord": slot["symbol"],
        "chord_bar": slot["bar"],
        "chord_beats_remaining": slot["start_beat"] + slot["beats"] - pos,
        "next_chord": nxt["symbol"],
    }


def chart_view() -> list[dict]:
    """The rendered chart with voicings, for a UI to draw a bar cursor against."""
    with _state_lock:
        rendered = _rendered
    if rendered is None:
        return []
    return [
        {**slot, "notes": voicing["notes"], "midi": voicing["midi"]}
        for slot, voicing in zip(rendered.chart, rendered.voicings)
    ]
