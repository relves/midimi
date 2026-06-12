"""Audio engine: FluidSynth / MIDI-out player with a scheduler thread and bar-range support."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

import fluidsynth
import mido

SOUNDFONT = os.environ.get("SOUNDFONT", str(Path.home() / "Music" / "GeneralUser-GS.sf2"))
DEFAULT_INSTRUMENT = 0
DEFAULT_CHANNEL = 0
DEFAULT_VELOCITY = 90

# ── Playback state ────────────────────────────────────────────────────────────

_play_lock = threading.Lock()
_stop_event = threading.Event()
_note_on_fn: Callable[[int, int, int], None] | None = None
_note_off_fn: Callable[[int, int], None] | None = None
_play_fn: Callable[[list[int], int], None] | None = None
_current_port: str | None = None
_current_fs: fluidsynth.Synth | None = None

_currently_playing: set[str] = set()
_currently_playing_lock = threading.Lock()


def current_port() -> str | None:
    return _current_port


# ── MIDI input / monitoring / recording ───────────────────────────────────────

_input_port_name: str | None = None
_input_port: mido.ports.BaseInput | None = None
_input_thread: threading.Thread | None = None
_input_stop = threading.Event()

# Recording state
_recording_lock = threading.Lock()
_recording: bool = False
_record_buf: list[dict] = []


def current_input_port() -> str | None:
    return _input_port_name


def start_input(port_name: str) -> None:
    """Open a MIDI input port. Monitoring (hear yourself) always on; recording is separate."""
    global _input_port_name, _input_port, _input_thread
    stop_input()
    try:
        port = mido.open_input(port_name)
    except Exception as exc:
        raise RuntimeError(f"Could not open MIDI input '{port_name}': {exc}") from exc
    _input_port = port
    _input_port_name = port_name
    _input_stop.clear()
    _input_thread = threading.Thread(target=_input_loop, args=(port,), daemon=True)
    _input_thread.start()
    print(f"MIDI in:  {port_name}")


def stop_input() -> None:
    """Close the current MIDI input port."""
    global _input_port_name, _input_port, _input_thread
    _input_stop.set()
    if _input_port is not None:
        try:
            _input_port.close()
        except Exception:
            pass
        _input_port = None
    _input_port_name = None
    _input_thread = None


def arm_recording() -> None:
    with _recording_lock:
        global _recording, _record_buf
        _record_buf = []
        _recording = True


def stop_recording() -> list[dict]:
    """Disarm recording and return captured events."""
    with _recording_lock:
        global _recording
        _recording = False
        events = list(_record_buf)
    return events


def _input_loop(port: mido.ports.BaseInput) -> None:
    """Listen on input port; play through to synth; capture when recording."""
    while not _input_stop.is_set():
        for msg in port.iter_pending():
            if msg.type == "note_on":
                is_on = msg.velocity > 0
                if is_on:
                    if _note_on_fn:
                        _note_on_fn(msg.note, msg.velocity, DEFAULT_CHANNEL)
                else:
                    if _note_off_fn:
                        _note_off_fn(msg.note, DEFAULT_CHANNEL)
                with _recording_lock:
                    if _recording:
                        _record_buf.append({
                            "note": msg.note,
                            "on": is_on,
                            "velocity": msg.velocity,
                            "t": time.monotonic(),
                        })
            elif msg.type == "note_off":
                if _note_off_fn:
                    _note_off_fn(msg.note, DEFAULT_CHANNEL)
                with _recording_lock:
                    if _recording:
                        _record_buf.append({
                            "note": msg.note,
                            "on": False,
                            "velocity": 0,
                            "t": time.monotonic(),
                        })
        time.sleep(0.002)  # 2 ms poll — low latency without busy-spin


# ── Initializers ──────────────────────────────────────────────────────────────

def set_note_fns(
    note_on: Callable[[int, int, int], None],
    note_off: Callable[[int, int], None],
    play: Callable[[list[int], int], None],
    port: str | None,
) -> None:
    """Wire engine to externally created note_on/note_off/play functions (e.g. from server.py)."""
    global _note_on_fn, _note_off_fn, _play_fn, _current_port
    _note_on_fn = note_on
    _note_off_fn = note_off
    _play_fn = play
    _current_port = port


def init_player() -> None:
    """Initialise from saved MIDI-out setting; fall back to FluidSynth."""
    # Import here to avoid circular at module load time
    from sequencer.model import _db_path
    midi_out = ""
    if _db_path is not None:
        try:
            import sqlite3
            conn = sqlite3.connect(_db_path)
            row = conn.execute("SELECT value FROM settings WHERE key='midi_out'").fetchone()
            conn.close()
            midi_out = (row[0] if row else "") or ""
        except Exception:
            pass

    if midi_out:
        available = mido.get_output_names()
        matches = [p for p in available if midi_out.lower() in p.lower()]
        if matches:
            _init_midi_port(matches[0])
            return
        print(f"Warning: saved MIDI_OUT='{midi_out}' matched no ports. Falling back to FluidSynth.")
    _init_fluidsynth()


def _init_midi_port(port_name: str) -> None:
    global _play_fn, _note_on_fn, _note_off_fn, _current_port
    port = mido.open_output(port_name)
    print(f"MIDI out: {port_name}")

    def note_on(note: int, velocity: int = DEFAULT_VELOCITY, channel: int = DEFAULT_CHANNEL) -> None:
        port.send(mido.Message("note_on", channel=channel, note=note, velocity=velocity))

    def note_off(note: int, channel: int = DEFAULT_CHANNEL) -> None:
        port.send(mido.Message("note_off", channel=channel, note=note, velocity=0))

    def play(notes: list[int], duration_ms: int) -> None:
        print(f"[midi-out:{port_name}] {notes}")
        with _play_lock:
            _stop_event.clear()
            for n in notes:
                note_on(n, DEFAULT_VELOCITY, DEFAULT_CHANNEL)
            _stop_event.wait(duration_ms / 1000)
            for n in notes:
                note_off(n, DEFAULT_CHANNEL)

    _note_on_fn = note_on
    _note_off_fn = note_off
    _play_fn = play
    _current_port = port_name


def _init_fluidsynth() -> None:
    global _play_fn, _note_on_fn, _note_off_fn, _current_port, _current_fs
    if _current_fs is not None:
        try:
            _current_fs.delete()
        except Exception:
            pass
        _current_fs = None

    fs = fluidsynth.Synth(gain=0.5)
    fs.start(driver="coreaudio")
    _current_fs = fs
    sfid = fs.sfload(SOUNDFONT)
    if sfid == -1:
        raise RuntimeError(f"Could not load soundfont: {SOUNDFONT}")
    fs.program_select(DEFAULT_CHANNEL, sfid, 0, DEFAULT_INSTRUMENT)
    fs.set_reverb(roomsize=0.5, damping=0.3, width=0.8, level=0.7)
    fs.set_chorus(nr=4, level=0.55, speed=0.36, depth=3.6, type=0)
    print("Audio out: FluidSynth")

    def note_on(note: int, velocity: int = DEFAULT_VELOCITY, channel: int = DEFAULT_CHANNEL) -> None:
        fs.noteon(channel, note, velocity)

    def note_off(note: int, channel: int = DEFAULT_CHANNEL) -> None:
        fs.noteoff(channel, note)

    def play(notes: list[int], duration_ms: int) -> None:
        print(f"[fluidsynth] {notes}")
        with _play_lock:
            _stop_event.clear()
            for n in notes:
                note_on(n, DEFAULT_VELOCITY, DEFAULT_CHANNEL)
            _stop_event.wait(duration_ms / 1000)
            for n in notes:
                note_off(n, DEFAULT_CHANNEL)

    _note_on_fn = note_on
    _note_off_fn = note_off
    _play_fn = play
    _current_port = None


def switch_to_port(port_name: str | None) -> None:
    """Switch output at runtime. None → FluidSynth."""
    if port_name is None:
        _init_fluidsynth()
    else:
        _init_midi_port(port_name)


# ── Simple chord playback ─────────────────────────────────────────────────────

def play_notes_bg(notes: list[int], duration_ms: int, note_id: str | None = None) -> None:
    """Play a chord in a background thread (non-blocking)."""
    def _run():
        if note_id:
            with _currently_playing_lock:
                _currently_playing.add(note_id)
        try:
            _play_fn(notes, duration_ms)
        finally:
            if note_id:
                with _currently_playing_lock:
                    _currently_playing.discard(note_id)

    threading.Thread(target=_run, daemon=True).start()


# ── Sequence scheduler ────────────────────────────────────────────────────────

def play_sequence_bg(sequence_id: str, sequence: dict, bars: str | None = None) -> None:
    """Play a sequence dict in a background scheduler thread.

    sequence must have: events (list of {at_beat, duration_beats, notes, velocity}),
    tempo_bpm, time_signature_parts.

    bars: optional "first-last" range string (1-indexed, inclusive), e.g. "1-4" or "3-6".
    If None, plays the whole sequence.
    """
    threading.Thread(target=_run_sequence, args=(sequence_id, sequence, bars), daemon=True).start()


def _parse_bars(bars_str: str | None, beats_per_bar: float) -> tuple[float, float]:
    """Return (start_beat, end_beat) for a bar range string, or (0, inf) for the whole piece."""
    if bars_str is None:
        return 0.0, float("inf")
    try:
        parts = bars_str.split("-")
        if len(parts) == 1:
            bar = int(parts[0])
            start = (bar - 1) * beats_per_bar
            end = bar * beats_per_bar
        else:
            start_bar = int(parts[0])
            end_bar = int(parts[1])
            start = (start_bar - 1) * beats_per_bar
            end = end_bar * beats_per_bar
    except (ValueError, IndexError):
        return 0.0, float("inf")
    return start, end


def _run_sequence(sequence_id: str, sequence: dict, bars: str | None) -> None:
    ts_num, ts_den = sequence["time_signature_parts"]
    beats_per_bar = ts_num * 4 / ts_den
    seconds_per_beat = 60 / sequence["tempo_bpm"]

    start_beat, end_beat = _parse_bars(bars, beats_per_bar)

    # Build sorted action list within the bar range
    actions: list[tuple[float, int, int, int]] = []  # (abs_time_s, kind 1=on/0=off, note, velocity)
    for event in sequence["events"]:
        eb = event["at_beat"]
        ee = eb + event["duration_beats"]
        if eb >= end_beat - 1e-9 or ee <= start_beat + 1e-9:
            continue
        # Clamp to range
        play_start = max(eb, start_beat)
        play_end = min(ee, end_beat)
        t_on = (play_start - start_beat) * seconds_per_beat
        t_off = (play_end - start_beat) * seconds_per_beat
        for note in event["notes"]:
            actions.append((t_on, 1, note, event["velocity"]))
            actions.append((t_off, 0, note, 0))

    actions.sort(key=lambda a: (a[0], a[1]))

    with _currently_playing_lock:
        _currently_playing.add(sequence_id)
    try:
        with _play_lock:
            _stop_event.clear()
            t0 = time.monotonic()
            sounding: list[int] = []
            try:
                for action_time, kind, note, velocity in actions:
                    if _stop_event.is_set():
                        break
                    sleep_for = t0 + action_time - time.monotonic()
                    if sleep_for > 0:
                        _stop_event.wait(sleep_for)
                    if _stop_event.is_set():
                        break
                    if kind:
                        _note_on_fn(note, velocity, DEFAULT_CHANNEL)
                        sounding.append(note)
                    else:
                        _note_off_fn(note, DEFAULT_CHANNEL)
                        if note in sounding:
                            sounding.remove(note)
            finally:
                for note in sounding:
                    _note_off_fn(note, DEFAULT_CHANNEL)
    finally:
        with _currently_playing_lock:
            _currently_playing.discard(sequence_id)


def play_raw_recording_bg(sequence_id: str, raw_events: list[dict]) -> None:
    """Replay raw capture timestamps verbatim (unquantized).

    raw_events: list of {note, on, velocity, t (monotonic seconds from capture)}
    """
    threading.Thread(target=_run_raw, args=(sequence_id, raw_events), daemon=True).start()


def _run_raw(sequence_id: str, raw_events: list[dict]) -> None:
    sorted_evs = sorted(raw_events, key=lambda e: e["t"])
    if not sorted_evs:
        return
    t0_cap = sorted_evs[0]["t"]

    with _currently_playing_lock:
        _currently_playing.add(sequence_id)
    try:
        with _play_lock:
            _stop_event.clear()
            t0_play = time.monotonic()
            sounding: list[int] = []
            try:
                for ev in sorted_evs:
                    if _stop_event.is_set():
                        break
                    target = t0_play + (ev["t"] - t0_cap)
                    sleep_for = target - time.monotonic()
                    if sleep_for > 0:
                        _stop_event.wait(sleep_for)
                    if _stop_event.is_set():
                        break
                    if ev.get("on", True) and ev.get("velocity", 1) > 0:
                        if _note_on_fn:
                            _note_on_fn(ev["note"], ev.get("velocity", DEFAULT_VELOCITY), DEFAULT_CHANNEL)
                        sounding.append(ev["note"])
                    else:
                        if _note_off_fn:
                            _note_off_fn(ev["note"], DEFAULT_CHANNEL)
                        if ev["note"] in sounding:
                            sounding.remove(ev["note"])
            finally:
                for note in sounding:
                    if _note_off_fn:
                        _note_off_fn(note, DEFAULT_CHANNEL)
    finally:
        with _currently_playing_lock:
            _currently_playing.discard(sequence_id)


def stop() -> None:
    _stop_event.set()
