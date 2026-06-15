"""Tests for the scale-patch drill: spelling helpers, scheduling policy, endpoints."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from sequencer.theory import major_scale_notes, normalize_note_name
from sequencer import drill


# ── major_scale_notes — exact enharmonic spelling ────────────────────────────

@pytest.mark.parametrize("key,expected", [
    ("C",  ["C", "D", "E", "F", "G", "A", "B"]),
    ("G",  ["G", "A", "B", "C", "D", "E", "F#"]),
    ("F",  ["F", "G", "A", "Bb", "C", "D", "E"]),
    ("D",  ["D", "E", "F#", "G", "A", "B", "C#"]),
    ("Bb", ["Bb", "C", "D", "Eb", "F", "G", "A"]),
    ("A",  ["A", "B", "C#", "D", "E", "F#", "G#"]),
    ("Eb", ["Eb", "F", "G", "Ab", "Bb", "C", "D"]),
    ("B",  ["B", "C#", "D#", "E", "F#", "G#", "A#"]),
    ("Db", ["Db", "Eb", "F", "Gb", "Ab", "Bb", "C"]),
    ("F#", ["F#", "G#", "A#", "B", "C#", "D#", "E#"]),   # E# not F
    ("Gb", ["Gb", "Ab", "Bb", "Cb", "Db", "Eb", "F"]),   # Cb not B
    ("C#", ["C#", "D#", "E#", "F#", "G#", "A#", "B#"]),   # E# and B#
])
def test_major_scale_notes(key, expected):
    assert major_scale_notes(key) == expected


def test_major_scale_notes_accepts_unicode_input():
    assert major_scale_notes("F♯") == major_scale_notes("F#")


# ── normalize_note_name — tolerant, never enharmonic ─────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("f#", "F#"), ("F♯", "F#"), ("bb", "Bb"), ("B♭", "Bb"),
    ("c", "C"), ("Eb", "Eb"), ("e♭", "Eb"),
])
def test_normalize_note_name(raw, want):
    assert normalize_note_name(raw) == want


def test_normalize_does_not_remap_enharmonics():
    assert normalize_note_name("Db") == "Db"   # stays Db, never C#
    assert normalize_note_name("C#") == "C#"


def test_normalize_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_note_name("H")
    with pytest.raises(ValueError):
        normalize_note_name("")


# ── schedule_after ───────────────────────────────────────────────────────────

def test_schedule_after_correct_promotes_and_snaps_to_local_midnight():
    box, due = drill.schedule_after(1, True, 1000)
    assert box == 2
    assert due == drill._due_at_local_midnight(1000, drill.BOX_INTERVALS_DAYS[2])


def test_schedule_after_correct_caps_at_max_box():
    box, due = drill.schedule_after(5, True, 1000)
    assert box == 5
    assert due == drill._due_at_local_midnight(1000, drill.BOX_INTERVALS_DAYS[5])


def test_due_at_local_midnight_lands_on_calendar_day_start():
    import datetime
    # 9:26pm local on some day -> box-2 (1 day) card is due 00:00 the next day,
    # not 9:26pm the next day (the old rolling-24h behavior).
    now_local = datetime.datetime(2026, 6, 14, 21, 26, 40).astimezone()
    now = int(now_local.timestamp())
    due = drill._due_at_local_midnight(now, 1)
    due_local = datetime.datetime.fromtimestamp(due).astimezone()
    assert (due_local.hour, due_local.minute, due_local.second) == (0, 0, 0)
    assert due_local.date() == datetime.date(2026, 6, 15)
    # Strictly earlier than the old rolling-24h result.
    assert due < now + drill.DAY_SECONDS


def test_due_at_local_midnight_zero_days_is_now():
    assert drill._due_at_local_midnight(1000, 0) == 1000


def test_schedule_after_wrong_resets_to_box1_due_now():
    box, due = drill.schedule_after(4, False, 1000)
    assert box == 1
    assert due == 1000


# ── pick_next ────────────────────────────────────────────────────────────────

def test_pick_next_returns_most_overdue():
    rows = [
        {"key": "C", "box": 1, "due_at": 900},
        {"key": "G", "box": 1, "due_at": 500},
        {"key": "F", "box": 1, "due_at": 800},
    ]
    assert drill.pick_next(rows, 1000)["key"] == "G"


def test_pick_next_none_when_nothing_due():
    rows = [{"key": "C", "box": 1, "due_at": 2000}]
    assert drill.pick_next(rows, 1000) is None


# ── maybe_unlock ─────────────────────────────────────────────────────────────

def test_maybe_unlock_gated_below_box3():
    rows = [{"key": k, "box": 2} for k in drill.STARTER_KEYS]
    assert drill.maybe_unlock(rows, 1000) is None


def test_maybe_unlock_returns_next_rotation_key():
    rows = [{"key": k, "box": 3} for k in drill.STARTER_KEYS]
    assert drill.maybe_unlock(rows, 1000) == drill.ROTATION[7]  # "B"


def test_maybe_unlock_none_when_all_present():
    rows = [{"key": k, "box": 5} for k in drill.ROTATION]
    assert drill.maybe_unlock(rows, 1000) is None


# ── grade_played — pitch-class tolerant MIDI grading ─────────────────────────

def _on(note):
    return {"note": note, "on": True, "velocity": 80, "t": 0.0}


def test_grade_played_two_octave_run_passes():
    # C major over two octaves: 60..72..76 etc. De-dupe should read as the 7-note set.
    notes = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76]  # C D E F G A B C D E
    events = [_on(n) for n in notes]
    r = drill.grade_played(events, "C")
    assert r["correct"] is True
    assert r["wrong_notes"] == []
    assert r["missing_notes"] == []


def test_grade_played_one_wrong_note_fails_and_flags():
    notes = [60, 62, 64, 66, 67, 69, 71]  # F# (66) instead of F
    events = [_on(n) for n in notes]
    r = drill.grade_played(events, "C")
    assert r["correct"] is False
    assert "F#" in r["wrong_notes"]


def test_grade_played_missing_degree_listed():
    notes = [60, 62, 64, 65, 67, 69]  # missing B (11)
    r = drill.grade_played([_on(n) for n in notes], "C")
    assert r["correct"] is False
    assert "B" in r["missing_notes"]


def test_grade_played_order_tolerant_and_spelled_in_key():
    notes = [69, 67, 65, 64, 62, 71, 60]  # C major scrambled
    r = drill.grade_played([_on(n) for n in notes], "C")
    assert r["correct"] is True
    # Played names use the key's spelling (Gb major would spell F#'s PC as Gb).
    r2 = drill.grade_played([_on(66)], "Gb")
    assert "Gb" in r2["played_names"]


def test_grade_played_ignores_note_off_events():
    events = [_on(n) for n in [60, 62, 64, 65, 67, 69, 71]]
    events += [{"note": 60, "on": False, "velocity": 0, "t": 1.0}]
    r = drill.grade_played(events, "C")
    assert r["correct"] is True


# ── ear_choices ──────────────────────────────────────────────────────────────

class _NoShuffle:
    @staticmethod
    def shuffle(x):
        pass


def test_ear_choices_includes_key_and_distinct_distractors():
    choices = drill.ear_choices("C", n_distractors=2, rng=_NoShuffle())
    assert "C" in choices
    assert len(choices) == 3
    assert len(set(choices)) == 3  # no duplicates, answer not repeated
    assert choices.count("C") == 1


def test_ear_choices_three_distractors():
    choices = drill.ear_choices("G", n_distractors=3, rng=_NoShuffle())
    assert len(set(choices)) == 4
    assert "G" in choices


# ── direction-aware scheduling / unlock ──────────────────────────────────────

def test_ear_unlocks_only_at_spell_box3():
    rows = [{"key": "C", "direction": "spell", "box": 2}]
    assert drill.ear_unlocks(rows) == []
    rows = [{"key": "C", "direction": "spell", "box": 3}]
    assert drill.ear_unlocks(rows) == ["C"]


def test_ear_unlocks_skips_existing_ear_rows():
    rows = [
        {"key": "C", "direction": "spell", "box": 4},
        {"key": "C", "direction": "ear", "box": 1},
    ]
    assert drill.ear_unlocks(rows) == []


def test_maybe_unlock_ignores_ear_rows():
    # All spell rows at box>=3 -> unlocks next key even with ear rows present.
    rows = [{"key": k, "direction": "spell", "box": 3} for k in drill.STARTER_KEYS]
    rows += [{"key": "C", "direction": "ear", "box": 1}]
    assert drill.maybe_unlock(rows, 1000) == drill.ROTATION[7]


# ── Endpoints (TestClient against a temp DB) ─────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # let init_db create it fresh
    monkeypatch.setenv("MIDIMI_DB", path)
    # Import after env is set so module-level init_db() uses the temp DB.
    for mod in list(sys.modules):
        if mod == "server":
            del sys.modules[mod]
    import server  # noqa: F401
    from fastapi.testclient import TestClient
    with TestClient(server.app) as c:
        yield c
    if os.path.exists(path):
        os.unlink(path)


def test_next_returns_starter_on_fresh_db(client):
    r = client.get("/drill/next").json()
    assert r["due"] is True
    assert r["key"] in drill.STARTER_KEYS
    assert r["due_today"] == 7
    assert r["prompt"].startswith("Spell ")


def test_grade_correct_promotes_and_wrong_resets(client):
    key = client.get("/drill/next").json()["key"]
    expected = major_scale_notes(key)

    good = client.post("/drill/grade", json={"key": key, "answer": expected}).json()
    assert good["correct"] is True
    assert good["box"] == 2
    assert good["expected"] == expected

    # A wrong answer on some other due key resets it to box 1.
    other = client.get("/drill/next").json()["key"]
    bad = client.post("/drill/grade", json={"key": other, "answer": ["C"] * 7}).json()
    assert bad["correct"] is False
    assert bad["box"] == 1
    assert any(p["ok"] is False for p in bad["per_note"])


def test_status_counts_due(client):
    s = client.get("/drill/status").json()
    assert s["due_today"] == 7
    assert s["done_today"] is False
    assert len(s["keys"]) == 7
    assert s["streak_days"] == 0


def test_full_clear_sets_streak_and_done(client):
    # Grade every starter key correctly -> queue empties, streak advances to 1.
    for _ in range(len(drill.STARTER_KEYS)):
        nxt = client.get("/drill/next").json()
        if not nxt["due"]:
            break
        key = nxt["key"]
        client.post("/drill/grade", json={"key": key, "answer": major_scale_notes(key)})
    nxt = client.get("/drill/next").json()
    assert nxt["due"] is False
    assert nxt["streak_days"] == 1
    assert client.get("/drill/status").json()["done_today"] is True


# ── Slice 2 endpoints ────────────────────────────────────────────────────────

def test_grade_played_endpoint(client, monkeypatch):
    import server
    key = client.get("/drill/next").json()["key"]
    expected_midi = [60 + server.NOTE_NAMES[n] for n in major_scale_notes(key)]
    events = [{"note": m, "on": True, "velocity": 80, "t": 0.0} for m in expected_midi]
    monkeypatch.setattr(server.engine, "stop_recording", lambda: events)

    r = client.post("/drill/grade_played", json={"key": key}).json()
    assert r["correct"] is True
    assert r["box"] == 2
    # played_names rendered in the key's correct spelling
    assert r["played_names"] == major_scale_notes(key)


def test_record_start_requires_input_port(client, monkeypatch):
    import server
    monkeypatch.setattr(server.engine, "current_input_port", lambda: None)
    assert client.post("/drill/record/start").status_code == 400
    monkeypatch.setattr(server.engine, "current_input_port", lambda: "Fake Port")
    monkeypatch.setattr(server.engine, "arm_recording", lambda: None)
    assert client.post("/drill/record/start").json()["ok"] is True


def test_play_prompt_invokes_playback(client, monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_play", lambda notes, dur: calls.append((notes, dur)))
    r = client.post("/drill/play_prompt", json={"key": "C"}).json()
    assert r["ok"] is True
    # background thread; give it a moment
    import time as _t
    _t.sleep(0.2)
    assert len(calls) == 7  # 7 ascending notes


def test_ear_item_appears_with_choices(client):
    import server
    # Seed an ear row directly (as the unlock path would) and make it due.
    conn = __import__("sqlite3").connect(server.DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO scale_drill (key, direction, box, due_at, streak) "
        "VALUES ('C', 'ear', 1, 1, 0)",
    )
    conn.commit()
    conn.close()
    # 'C' ear is now the most-overdue (due_at=1).
    nxt = client.get("/drill/next").json()
    assert nxt["direction"] == "ear"
    assert nxt["key"] == "C"
    assert "C" in nxt["choices"]
    assert len(set(nxt["choices"])) == len(nxt["choices"])

    # Correct ear answer promotes the ear row.
    r = client.post("/drill/grade", json={"key": "C", "direction": "ear", "choice": "C"}).json()
    assert r["correct"] is True
    assert r["box"] == 2
    # Wrong ear answer resets.
    conn = __import__("sqlite3").connect(server.DB_PATH)
    conn.execute("UPDATE scale_drill SET due_at=1 WHERE key='C' AND direction='ear'")
    conn.commit()
    conn.close()
    bad = client.post("/drill/grade", json={"key": "C", "direction": "ear", "choice": "G"}).json()
    assert bad["correct"] is False
    assert bad["box"] == 1


def test_migration_preserves_slice1_rows(monkeypatch):
    """A Slice-1 scale_drill (PK key, no direction) migrates to spell rows."""
    import sqlite3
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    # Build a Slice-1 shaped table with one row.
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE scale_drill (
        key TEXT PRIMARY KEY, box INTEGER, due_at INTEGER, streak INTEGER,
        last_result TEXT, last_seen INTEGER)""")
    conn.execute("INSERT INTO scale_drill VALUES ('C', 4, 123, 2, 'correct', 100)")
    conn.commit()
    conn.close()

    monkeypatch.setenv("MIDIMI_DB", path)
    for mod in list(sys.modules):
        if mod == "server":
            del sys.modules[mod]
    import server  # triggers init_db migration

    rows = {(r["key"], r["direction"]): r for r in server._drill_rows()}
    assert ("C", "spell") in rows
    assert rows[("C", "spell")]["box"] == 4
    if os.path.exists(path):
        os.unlink(path)
