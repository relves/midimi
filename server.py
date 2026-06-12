#!/usr/bin/env python3
"""midimi web server — FastAPI + Datastar SSE, server-side playback via FluidSynth or MIDI out"""

import base64
import html
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import anthropic
import fluidsynth
import mido
from sequencer.abc import parse_abc, to_abc, ABCParseError, per_bar_report
from sequencer.theory import (
    normalize_chord_quality, chord_note_names, build_chord, parse_pitch,
    NOTE_NAMES, CHORD_INTERVALS, midi_note_name as _theory_midi_note_name,
)
import sequencer.model as seq_model
import sequencer.engine as engine
from tools import TOOLS, SYSTEM_PROMPT, dispatch_tools
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field



midi_note_name = _theory_midi_note_name


SOUNDFONT = os.environ.get("SOUNDFONT", str(Path.home() / "Music" / "GeneralUser-GS.sf2"))
DB_PATH = Path(__file__).parent / "midimi.db"
STATIC_DIR = Path(__file__).parent / "static"
GENERATED_DIR = Path(__file__).parent / "generated" / "orchestrations"
DEFAULT_INSTRUMENT = 0
DEFAULT_CHANNEL = 0
DEFAULT_VELOCITY = 90
DEFAULT_DURATION_MS = 1500
MIDI_TICKS_PER_BEAT = 480
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
_anthropic_client: anthropic.Anthropic | None = None


def get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = db_get_setting("api_key")
        if not api_key:
            raise HTTPException(400, "Anthropic API key not configured. Open Settings (⚙) to add it.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Player (FluidSynth or MIDI out) ──────────────────────────────────────────

_play_lock = threading.Lock()
_stop_event = threading.Event()
_note_on = None
_note_off = None
_current_fs = None  # active FluidSynth instance, kept so it can be cleaned up on switch


def _init_player():
    midi_out = (db_get_setting("midi_out") or "").strip()
    if midi_out:
        available = mido.get_output_names()
        matches = [p for p in available if midi_out.lower() in p.lower()]
        if not matches:
            print(f"Warning: saved MIDI_OUT='{midi_out}' matched no ports. Falling back to FluidSynth.")
            return _init_fluidsynth()
        port = mido.open_output(matches[0])
        print(f"MIDI out: {matches[0]}")

        def play(notes: list[int], duration_ms: int) -> None:
            print(f"[midi-out:{matches[0]}] {notes}")
            with _play_lock:
                _stop_event.clear()
                for n in notes:
                    note_on(n, DEFAULT_VELOCITY, DEFAULT_CHANNEL)
                _stop_event.wait(duration_ms / 1000)
                for n in notes:
                    note_off(n, DEFAULT_CHANNEL)

        def note_on(note: int, velocity: int = DEFAULT_VELOCITY, channel: int = DEFAULT_CHANNEL) -> None:
            port.send(mido.Message("note_on", channel=channel, note=note, velocity=velocity))

        def note_off(note: int, channel: int = DEFAULT_CHANNEL) -> None:
            port.send(mido.Message("note_off", channel=channel, note=note, velocity=0))

        return play, matches[0], note_on, note_off
    else:
        return _init_fluidsynth()


def _init_fluidsynth():
    global _current_fs
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

    def play(notes: list[int], duration_ms: int) -> None:
        print(f"[fluidsynth] {notes}")
        with _play_lock:
            _stop_event.clear()
            for n in notes:
                note_on(n, DEFAULT_VELOCITY, DEFAULT_CHANNEL)
            _stop_event.wait(duration_ms / 1000)
            for n in notes:
                note_off(n, DEFAULT_CHANNEL)

    def note_on(note: int, velocity: int = DEFAULT_VELOCITY, channel: int = DEFAULT_CHANNEL) -> None:
        fs.noteon(channel, note, velocity)

    def note_off(note: int, channel: int = DEFAULT_CHANNEL) -> None:
        fs.noteoff(channel, note)

    return play, None, note_on, note_off


# In-memory registry: note_id → {notes, duration_ms} for replay
_note_registry: dict[str, dict] = {}
_sequence_registry: dict[str, dict] = {}


def _resolve_sequence(sequence_id: str) -> dict | None:
    """Return the normalized sequence dict for playback.

    Checks DB first (sequences table), then in-memory registry (legacy chat history rehydration).
    Returns None if not found.
    """
    row = seq_model.get_sequence(sequence_id)
    if row:
        try:
            return parse_abc(row["abc"])
        except ABCParseError:
            pass
    entry = _sequence_registry.get(sequence_id)
    if entry:
        return entry["sequence"]
    return None

# Set of note_ids currently being played
_currently_playing: set[str] = set()
_currently_playing_lock = threading.Lock()


def play_in_background(notes: list[int], duration_ms: int, note_id: str | None = None) -> None:
    def _run():
        if note_id:
            with _currently_playing_lock:
                _currently_playing.add(note_id)
        try:
            _play(notes, duration_ms)
        finally:
            if note_id:
                with _currently_playing_lock:
                    _currently_playing.discard(note_id)

    threading.Thread(target=_run, daemon=True).start()


from sequencer.midi_io import write_sequence_midi as _write_seq_midi


def write_sequence_midi(sequence: dict, sequence_id: str) -> Path:
    return _write_seq_midi(sequence, sequence_id, GENERATED_DIR)


def play_sequence_in_background(sequence_id: str, bars: str | None = None) -> None:
    """Play a sequence by id. Looks up DB first, then falls back to in-memory registry."""
    seq_dict = _resolve_sequence(sequence_id)
    if seq_dict is None:
        return
    engine.play_sequence_bg(sequence_id, seq_dict, bars=bars)


# ── Database ──────────────────────────────────────────────────────────────────

def db_get_setting(key: str) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def db_set_setting(key: str, value: str | None):
    conn = sqlite3.connect(DB_PATH)
    if value is None:
        conn.execute("DELETE FROM settings WHERE key=?", (key,))
    else:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, value))
    conn.commit()
    conn.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at INTEGER,
            title TEXT,
            modified_at INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at INTEGER
        )
    """)
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN starred INTEGER DEFAULT 0")
    except Exception:
        pass
    conn.commit()
    conn.close()


init_db()
seq_model.init_model(DB_PATH)
_play, _current_port, _note_on, _note_off = _init_player()
engine.set_note_fns(_note_on, _note_off, _play, _current_port)


def _init_input_from_settings():
    midi_in = (db_get_setting("midi_in") or "").strip()
    if midi_in:
        available = mido.get_input_names()
        matches = [p for p in available if midi_in.lower() in p.lower()]
        if matches:
            try:
                engine.start_input(matches[0])
            except Exception as e:
                print(f"Warning: could not open saved MIDI input '{matches[0]}': {e}")


_init_input_from_settings()


def db_update_session_modified(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE sessions SET modified_at=? WHERE id=?", (int(time.time()), session_id))
    conn.commit()
    conn.close()


def generate_title_async(session_id: str, first_user_message: str):
    def _gen():
        try:
            resp = get_anthropic_client().messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": (
                    f'Give a 3-5 word title for a music theory conversation that starts with: '
                    f'"{first_user_message}". Reply with just the title, no quotes or punctuation.'
                )}],
            )
            title = resp.content[0].text.strip()
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
            conn.commit()
            conn.close()
        except Exception:
            pass
    threading.Thread(target=_gen, daemon=True).start()


def db_save_message(session_id: str, role: str, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
        (session_id, role, json.dumps(content), int(time.time())),
    )
    conn.commit()
    conn.close()


def db_get_history(session_id: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": json.loads(r[1])} for r in rows]


# ── Datastar SSE helpers ──────────────────────────────────────────────────────

def ds_merge_fragment(fragment: str, selector: str = "", merge_mode: str = "append") -> str:
    lines = ["event: datastar-merge-fragments"]
    if selector:
        lines.append(f"data: selector {selector}")
    lines.append(f"data: mergeMode {merge_mode}")
    for line in fragment.splitlines():
        lines.append(f"data: fragments {line}")
    return "\n".join(lines) + "\n\n"


def ds_merge_signals(signals: dict) -> str:
    return f"event: datastar-merge-signals\ndata: signals {json.dumps(signals)}\n\n"


# ── HTML fragments ────────────────────────────────────────────────────────────

def user_bubble(text: str, msg_id: str, images: list[dict] | None = None) -> str:
    safe = html.escape(text)
    image_html = ""
    if images:
        image_html = '<div class="message-images">' + "".join(
            f'<img src="data:{html.escape(image["media_type"])};base64,{html.escape(image["data"])}" alt="Pasted sheet music" />'
            for image in images
        ) + "</div>"
    return (
        f'<div id="{msg_id}" class="msg msg-user">'
        f'<div class="bubble">{image_html}{safe}</div>'
        f'</div>'
    )


def assistant_bubble_open(msg_id: str) -> str:
    return (
        f'<div id="{msg_id}" class="msg msg-assistant">'
        f'<div class="bubble">'
        f'<span id="{msg_id}-spinner" class="thinking-spinner"></span>'
        f'</div>'
        f'</div>'
    )


def audio_pill(
    note_id: str,
    label: str,
    pill_id: str,
    notes: list[int] | None = None,
    root: str = "",
    quality: str = "",
) -> str:
    import json as _json
    safe_label = html.escape(label or "Play")
    chips_html = ""
    staff_btn = ""
    staff_panel = ""
    if notes:
        if root and quality:
            octave = notes[0] // 12 - 1
            names = chord_note_names(root, quality, octave)
        else:
            prefer_flats = prefer_flats_for(root, quality)
            names = [midi_note_name(n, prefer_flats) for n in notes]
        chips = "".join(
            f'<button class="note-chip" onclick="playMidiNote({n}, this)" title="{name}">'
            f'{html.escape(name)}</button>'
            for n, name in zip(notes, names)
        )
        chips_html = f'<div class="pill-notes">{chips}</div>'
        staff_panel_id = f"{pill_id}-staff"
        note_names_json = html.escape(_json.dumps(names), quote=True)
        staff_btn = (
            f'<button class="staff-btn" '
            f"onclick=\"toggleStaff('{staff_panel_id}', this, {note_names_json})\" "
            f'title="Show on staff">\U0001D11E</button>'
        )
        staff_panel = f'<div id="{staff_panel_id}" class="staff-panel"></div>'
    return (
        f'<div class="pill-wrap">'
        f'  <div id="{pill_id}" class="audio-pill">'
        f'    <button class="play-btn" onclick="replayNote(\'{note_id}\')" title="Replay">▶</button>'
        f'    <span class="pill-label">♪ {safe_label}</span>'
        f'    {chips_html}'
        f'    {staff_btn}'
        f'  </div>'
        f'  {staff_panel}'
        f'</div>'
    )


def sequence_pill(sequence_id: str, title: str, pill_id: str, duration_ms: int, midi_url: str, events: list[dict]) -> str:
    import json as _json
    safe_title = html.escape(title or "Orchestration")
    seconds = max(1, round(duration_ms / 1000))
    staff_panel_id = f"{pill_id}-staff"
    events_json = html.escape(_json.dumps(events), quote=True)
    return (
        f'<div class="pill-wrap">'
        f'  <div id="{pill_id}" class="audio-pill sequence-pill">'
        f'    <button class="play-btn" onclick="replaySequence(\'{sequence_id}\')" title="Replay sequence">▶</button>'
        f'    <span class="pill-label">♫ {safe_title}</span>'
        f'    <span class="sequence-meta">{seconds}s</span>'
        f'    <a class="download-btn" href="{html.escape(midi_url)}" title="Download MIDI">MIDI</a>'
        f'    <button class="staff-btn" onclick="toggleSequenceStaff(\'{staff_panel_id}\', this, {events_json})" title="Show on staff">\U0001D11E</button>'
        f'  </div>'
        f'  <div id="{staff_panel_id}" class="staff-panel sequence-staff-panel"></div>'
        f'</div>'
    )


# ── Config routes ─────────────────────────────────────────────────────────────

class ConfigRequest(BaseModel):
    port: str | None  # None = FluidSynth


@app.get("/config")
def get_config():
    return {
        "current": _current_port,
        "ports": mido.get_output_names(),
    }


@app.post("/config")
def set_config(req: ConfigRequest):
    global _play, _current_port, _note_on, _note_off

    if req.port is None:
        _play, _current_port, _note_on, _note_off = _init_fluidsynth()
        engine.set_note_fns(_note_on, _note_off, _play, None)
        db_set_setting("midi_out", None)
        return {"ok": True, "current": None}

    available = mido.get_output_names()
    if req.port not in available:
        raise HTTPException(400, f"Port '{req.port}' not found. Available: {available}")

    port = mido.open_output(req.port)

    def play(notes: list[int], duration_ms: int) -> None:
        print(f"[midi-out:{req.port}] {notes}")
        with _play_lock:
            _stop_event.clear()
            for n in notes:
                note_on(n, DEFAULT_VELOCITY, DEFAULT_CHANNEL)
            _stop_event.wait(duration_ms / 1000)
            for n in notes:
                note_off(n, DEFAULT_CHANNEL)

    def note_on(note: int, velocity: int = DEFAULT_VELOCITY, channel: int = DEFAULT_CHANNEL) -> None:
        port.send(mido.Message("note_on", channel=channel, note=note, velocity=velocity))

    def note_off(note: int, channel: int = DEFAULT_CHANNEL) -> None:
        port.send(mido.Message("note_off", channel=channel, note=note, velocity=0))

    _play = play
    _current_port = req.port
    _note_on = note_on
    _note_off = note_off
    engine.set_note_fns(note_on, note_off, play, req.port)
    db_set_setting("midi_out", req.port)
    return {"ok": True, "current": req.port}


class TestRequest(BaseModel):
    port: str | None


@app.post("/config/test")
def test_output(req: TestRequest):
    """Switch to the given port (without saving) and send a test chord."""
    global _play, _current_port
    # Temporarily apply the selected port so the test uses it
    result = set_config(ConfigRequest(port=req.port))
    play_in_background([60, 64, 67], 800)
    return result


@app.get("/settings")
def get_settings():
    api_key = db_get_setting("api_key")
    return {
        "api_key_set": bool(api_key),
        "current_port": _current_port,
        "ports": mido.get_output_names(),
        "current_input_port": engine.current_input_port(),
        "input_ports": mido.get_input_names(),
    }


class SettingsRequest(BaseModel):
    api_key: str | None = None
    port: str | None = None
    input_port: str | None = None  # None = no change, "" = disable


@app.post("/settings")
def save_settings(req: SettingsRequest):
    global _anthropic_client
    if req.api_key is not None:
        db_set_setting("api_key", req.api_key or None)
        _anthropic_client = None  # force re-init with new key
    if "port" in req.model_fields_set:
        set_config(ConfigRequest(port=req.port))
    if "input_port" in req.model_fields_set:
        if req.input_port == "" or req.input_port is None:
            engine.stop_input()
            db_set_setting("midi_in", None)
        else:
            available = mido.get_input_names()
            if req.input_port not in available:
                raise HTTPException(400, f"Input port '{req.input_port}' not found. Available: {available}")
            engine.start_input(req.input_port)
            db_set_setting("midi_in", req.input_port)
    return {"ok": True, "api_key_set": bool(db_get_setting("api_key"))}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/sessions")
def list_sessions():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title, modified_at, COALESCE(starred, 0) FROM sessions "
        "ORDER BY COALESCE(starred, 0) DESC, COALESCE(modified_at, created_at) DESC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1] or "New session", "modified_at": r[2], "starred": bool(r[3])} for r in rows]


@app.post("/session/{session_id}/star")
def toggle_star(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE sessions SET starred = CASE WHEN COALESCE(starred,0)=1 THEN 0 ELSE 1 END WHERE id=?",
        (session_id,)
    )
    conn.commit()
    starred = conn.execute("SELECT COALESCE(starred,0) FROM sessions WHERE id=?", (session_id,)).fetchone()[0]
    conn.close()
    return {"starred": bool(starred)}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/session/new")
def new_session():
    sid = str(uuid.uuid4())[:8]
    now = int(time.time())
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO sessions (id, created_at, title, modified_at) VALUES (?,?,?,?)", (sid, now, "New session", now))
    conn.commit()
    conn.close()
    return {"id": sid}


@app.get("/session/{session_id}/history")
def session_history(session_id: str):
    history = db_get_history(session_id)
    result = []
    for msg in history:
        if msg["role"] == "user":
            content = msg["content"]
            result.append({
                "role": "user",
                "text": user_content_text(content),
                "images": user_content_images(content),
            })
        elif msg["role"] == "assistant":
            parts = msg["content"] if isinstance(msg["content"], list) else []
            items = []
            for p in parts:
                if p.get("type") == "text":
                    items.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "audio":
                    nid = p["note_id"]
                    _note_registry[nid] = {
                        "notes": p["notes"],
                        "duration_ms": p["duration_ms"],
                        "root": p.get("root", ""),
                        "quality": p.get("quality", ""),
                    }
                    items.append({
                        "type": "audio",
                        "note_id": nid,
                        "label": p.get("label", ""),
                        "notes": p["notes"],
                        "root": p.get("root", ""),
                        "quality": p.get("quality", ""),
                        "prefer_flats": p.get("prefer_flats", prefer_flats_for(p.get("root", ""), p.get("quality", ""))),
                    })
                elif p.get("type") == "sequence":
                    sequence_id = p["sequence_id"]
                    midi_path = Path(p.get("midi_path", GENERATED_DIR / f"{sequence_id}.mid"))
                    if "sequence" in p:
                        _sequence_registry[sequence_id] = {"sequence": p["sequence"], "midi_path": midi_path}
                    items.append({
                        "type": "sequence",
                        "sequence_id": sequence_id,
                        "title": p.get("title", "Orchestration"),
                        "duration_ms": p.get("duration_ms", 1000),
                        "events": p.get("sequence", {}).get("events", []),
                        "midi_url": f"/sequence/{sequence_id}/download",
                    })
            if items:
                result.append({"role": "assistant", "items": items})
    return result


@app.post("/play/{note_id}")
def replay(note_id: str):
    entry = _note_registry.get(note_id)
    if not entry:
        raise HTTPException(404, "Note not found")
    play_in_background(entry["notes"], entry["duration_ms"], note_id)
    return {"ok": True}


@app.post("/sequence/{sequence_id}")
def replay_sequence(sequence_id: str):
    seq_dict = _resolve_sequence(sequence_id)
    if seq_dict is None:
        raise HTTPException(404, "Sequence not found")
    engine.play_sequence_bg(sequence_id, seq_dict)
    return {"ok": True}


@app.get("/sequence/{sequence_id}/download")
def download_sequence(sequence_id: str):
    # Try to regenerate MIDI from DB if the file is missing
    midi_path = GENERATED_DIR / f"{sequence_id}.mid"
    if not midi_path.exists():
        seq_dict = _resolve_sequence(sequence_id)
        if seq_dict is None:
            raise HTTPException(404, "Sequence not found")
        midi_path = write_sequence_midi(seq_dict, sequence_id)
    if not midi_path.exists():
        raise HTTPException(404, "MIDI file not found")
    return FileResponse(midi_path, media_type="audio/midi", filename=f"{sequence_id}.mid")


@app.post("/play_midi/{note}")
def play_single_midi(note: int):
    if not (0 <= note <= 127):
        raise HTTPException(400, "MIDI note must be 0–127")
    play_in_background([note], 800)
    return {"ok": True}


# ── Recording endpoints ───────────────────────────────────────────────────────

class RecordStartRequest(BaseModel):
    session_id: str
    tempo_bpm: float = 120.0
    time_signature: str = "4/4"


class RecordStopRequest(BaseModel):
    session_id: str
    tempo_bpm: float = 120.0
    time_signature: str = "4/4"
    title: str = "Recording"
    grid: float = 0.25  # quantization grid in beats (0.25 = 1/16th)
    quantize: bool = True


@app.post("/record/start")
def record_start(req: RecordStartRequest):
    if engine.current_input_port() is None:
        raise HTTPException(400, "No MIDI input port selected. Open Settings to choose one.")
    engine.arm_recording()
    return {"ok": True, "recording": True}


@app.post("/record/stop")
def record_stop(req: RecordStopRequest):
    from sequencer.midi_io import quantize_recording, timing_report
    from sequencer.abc import to_abc

    raw_events = engine.stop_recording()
    if not raw_events:
        raise HTTPException(400, "No notes were recorded.")

    try:
        seq_dict = quantize_recording(
            raw_events,
            tempo_bpm=req.tempo_bpm,
            time_signature=req.time_signature,
            grid=req.grid,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    seq_dict["title"] = req.title
    abc_text = to_abc(seq_dict)

    seq_id = seq_model.create_sequence(
        title=req.title,
        abc=abc_text,
        session_id=req.session_id,
        tempo_bpm=req.tempo_bpm,
        time_signature=req.time_signature,
        source="recording",
        raw_events=raw_events,
    )

    # Build pill for injection into chat
    midi_path = write_sequence_midi(seq_dict, seq_id)
    midi_url = f"/sequence/{seq_id}/download"
    duration_ms = seq_dict.get("duration_ms", 1000)
    pill_html = sequence_pill(seq_id, req.title, f"pill-{seq_id}", duration_ms, midi_url, seq_dict["events"])

    # Persist to chat history so it survives reload
    record_entry = {
        "type": "sequence",
        "sequence_id": seq_id,
        "title": req.title,
        "duration_ms": duration_ms,
        "sequence": seq_dict,
        "midi_path": str(midi_path),
        "source": "recording",
    }
    db_save_message(req.session_id, "assistant", [record_entry])
    db_update_session_modified(req.session_id)

    t_report = timing_report(raw_events, seq_dict)

    return {
        "ok": True,
        "sequence_id": seq_id,
        "abc": abc_text,
        "pill_html": pill_html,
        "timing_report": t_report,
        "duration_ms": duration_ms,
    }


@app.get("/playing")
def get_playing():
    with _currently_playing_lock:
        server_playing = set(_currently_playing)
    with engine._currently_playing_lock:
        engine_playing = set(engine._currently_playing)
    return list(server_playing | engine_playing)


@app.post("/stop")
def stop_playback():
    _stop_event.set()
    engine.stop()
    return {"ok": True}


ALLOWED_MODELS = {
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
}


class ImageAttachment(BaseModel):
    media_type: str
    data: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "claude-haiku-4-5-20251001"
    images: list[ImageAttachment] = Field(default_factory=list)


def normalize_image_attachments(images: list[ImageAttachment]) -> list[dict]:
    if len(images) > 3:
        raise HTTPException(400, "Attach at most 3 images per message.")

    normalized = []
    for idx, image in enumerate(images):
        media_type = image.media_type.strip().lower()
        if media_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, f"Unsupported image type for attachment {idx + 1}: {media_type}")
        try:
            raw = base64.b64decode(image.data, validate=True)
        except Exception:
            raise HTTPException(400, f"Attachment {idx + 1} is not valid base64.")
        if not raw:
            raise HTTPException(400, f"Attachment {idx + 1} is empty.")
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(400, f"Attachment {idx + 1} is larger than 5 MB.")
        normalized.append({"media_type": media_type, "data": image.data})
    return normalized


def user_content_blocks(message: str, images: list[dict]) -> str | list[dict]:
    text = message.strip()
    if not images:
        return text

    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    else:
        blocks.append({"type": "text", "text": "Please transcribe this sheet music and play it back as MIDI."})
    blocks.extend({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image["media_type"],
            "data": image["data"],
        },
    } for image in images)
    return blocks


def user_content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""


def user_content_images(content) -> list[dict]:
    if not isinstance(content, list):
        return []
    images = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image":
            continue
        source = part.get("source") or {}
        if source.get("type") != "base64":
            continue
        images.append({
            "media_type": source.get("media_type", "image/png"),
            "data": source.get("data", ""),
        })
    return images


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if req.model not in ALLOWED_MODELS:
        raise HTTPException(400, f"Unknown model: {req.model}")
    images = normalize_image_attachments(req.images)
    message_text = req.message.strip()
    if not message_text and not images:
        raise HTTPException(400, "Message or image required.")
    user_content = user_content_blocks(message_text, images)
    history = db_get_history(req.session_id)

    anthropic_history = []
    for msg in history:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                anthropic_history.append({"role": "user", "content": content})
            else:
                anthropic_history.append({"role": "user", "content": str(content)})
        elif msg["role"] == "assistant":
            parts = msg["content"]
            if isinstance(parts, list):
                api_parts = [p for p in parts if p.get("type") in ("text", "tool_use")]
                if api_parts:
                    anthropic_history.append({"role": "assistant", "content": api_parts})

    user_msg_id = f"u{str(uuid.uuid4())[:6]}"
    asst_msg_id = f"a{str(uuid.uuid4())[:6]}"

    db_save_message(req.session_id, "user", user_content)
    anthropic_history.append({"role": "user", "content": user_content})

    def generate():
        yield ds_merge_fragment(user_bubble(message_text, user_msg_id, images), selector="#transcript")
        yield ds_merge_fragment(assistant_bubble_open(asst_msg_id), selector="#transcript")
        yield ds_merge_signals({"loading": True})

        current_history = list(anthropic_history)
        assistant_record = []
        seg_idx = 0

        # Open first text segment inside the bubble
        seg_id = f"{asst_msg_id}-t{seg_idx}"
        yield ds_merge_fragment(
            f'<span id="{seg_id}"></span>',
            selector=f"#{asst_msg_id} .bubble",
        )
        seg_text = ""

        while True:
            api_content = []        # what we'll append to history as assistant turn
            pending_tools = []      # tool calls completed this turn

            # Per-block streaming state
            in_tool = False
            tool_id = tool_name = ""
            tool_input_buf = ""

            with get_anthropic_client().messages.stream(
                model=req.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=current_history,
            ) as stream:
                for event in stream:
                    etype = type(event).__name__

                    if etype == "RawContentBlockStartEvent":
                        blk = event.content_block
                        if blk.type == "tool_use":
                            in_tool = True
                            tool_id = blk.id
                            tool_name = blk.name
                            tool_input_buf = ""
                        else:
                            in_tool = False

                    elif etype == "RawContentBlockDeltaEvent":
                        delta = event.delta
                        if delta.type == "text_delta":
                            seg_text += delta.text
                            yield ds_merge_signals({"textUpdate": {"segId": seg_id, "raw": seg_text}})
                        elif delta.type == "input_json_delta":
                            tool_input_buf += delta.partial_json

                    elif etype == "ParsedContentBlockStopEvent" and in_tool:
                        inp = json.loads(tool_input_buf) if tool_input_buf else {}
                        api_content.append({
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": inp,
                        })
                        pending_tools.append({"id": tool_id, "name": tool_name, "input": inp})
                        in_tool = False

                final = stream.get_final_message()

            # Harvest text blocks from the final message
            for blk in final.content:
                if blk.type == "text":
                    api_content.insert(0, {"type": "text", "text": blk.text})

            current_history.append({"role": "assistant", "content": api_content})

            if final.stop_reason != "tool_use" or not pending_tools:
                for blk in final.content:
                    if blk.type == "text":
                        assistant_record.append({"type": "text", "text": blk.text})
                break

            # Save pre-tool text so it appears before the pills on restore
            for blk in final.content:
                if blk.type == "text" and blk.text.strip():
                    assistant_record.append({"type": "text", "text": blk.text})

            # Execute tools and emit pills inline
            tool_results = []
            for kind, payload in dispatch_tools(
                pending_tools,
                session_id=req.session_id,
                asst_msg_id=asst_msg_id,
                note_registry=_note_registry,
                sequence_registry=_sequence_registry,
                resolve_sequence=_resolve_sequence,
                sequence_pill_fn=sequence_pill,
                audio_pill_fn=audio_pill,
                generated_dir=GENERATED_DIR,
                play_notes_bg=play_in_background,
            ):
                if kind == "sse":
                    yield ds_merge_fragment(payload, selector=f"#{asst_msg_id} .bubble")
                elif kind == "result":
                    tool_results.append(payload)
                elif kind == "record":
                    assistant_record.append(payload)

            current_history.append({"role": "user", "content": tool_results})

            # New text segment after the pill(s)
            seg_idx += 1
            seg_id = f"{asst_msg_id}-t{seg_idx}"
            seg_text = ""
            yield ds_merge_fragment(
                f'<span id="{seg_id}"></span>',
                selector=f"#{asst_msg_id} .bubble",
            )
            yield ds_merge_fragment(
                f'<span class="thinking-spinner"></span>',
                selector=f"#{asst_msg_id} .bubble",
            )

        db_save_message(req.session_id, "assistant", assistant_record)
        db_update_session_modified(req.session_id)
        if not history:  # first exchange
            generate_title_async(req.session_id, message_text or "Pasted sheet music")
        yield ds_merge_signals({"loading": False})

    return StreamingResponse(generate(), media_type="text/event-stream")
