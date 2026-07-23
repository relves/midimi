"""Tests for the Slice-C harmony drill card types: decks, prompts, grading, gating."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from sequencer import drill, drill_cards as dc
from sequencer.theory import NOTE_NAMES


def _events(names_or_midi):
    """Note-on events from note names (middle-octave) or raw MIDI numbers."""
    out = []
    for x in names_or_midi:
        midi = x if isinstance(x, int) else 60 + NOTE_NAMES[x]
        out.append({"note": midi, "on": True})
    return out


def _rows(kind, box, items=None):
    return [
        {"kind": kind, "item": it, "key": dc.card_key(kind, it), "box": box, "due_at": 0}
        for it in (items or dc.seed_items(kind))
    ]


# ── Decks honour the plan's exclusions ───────────────────────────────────────

def test_no_extended_harmony_in_any_deck():
    banned = {"major9", "dominant9", "minor9", "add9", "dominant11", "minor11",
              "dominant13", "major7#11", "dominant7alt", "major69"}
    for kind, deck in dc.DECKS.items():
        assert not banned & set(deck), f"{kind} leaks extended harmony"


def test_deck_sizes():
    assert len(dc.DECKS["interval_spell"]) == 11
    assert len(dc.DECKS["triad_spell"]) == 4
    assert len(dc.DECKS["seventh_spell"]) == 5
    assert len(dc.DECKS["diatonic"]) == 8       # 7 numerals + "all"
    assert len(dc.DECKS["guide_tones"]) == 7    # starter keys


def test_prompt_roots_cover_all_pitch_classes():
    assert len({NOTE_NAMES[r] for r in dc.PROMPT_ROOTS}) == 12


# ── Interval cards ───────────────────────────────────────────────────────────

def test_interval_spell_prompt_and_played_grade():
    rng = random.Random(1)
    p = dc.make_prompt("interval_spell", "M3", rng=rng)
    assert p["text"] == f"Major 3rd above {p['root']}"
    assert len(p["expected"]) == 2
    good = dc.grade_prompt_played(p, _events(p["expected"]))
    assert good["correct"]
    bad = dc.grade_prompt_played(p, _events([p["expected"][0]]))
    assert not bad["correct"]


def test_interval_spelling_is_diatonic():
    rng = random.Random(0)
    while True:
        p = dc.make_prompt("interval_spell", "M3", rng=rng)
        if p["root"] == "Eb":
            break
    assert p["expected"] == ["Eb", "G"]
    assert p["answer_note"] == "G"


def test_interval_cards_never_spell_a_double_accidental():
    # Db+m2 (Ebb), Db+m6 (Bbb) and Ab+m2 (Bbb) are diatonically correct but
    # hostile to name; make_prompt respells the root to its enharmonic partner.
    forced = {
        ("Db", "m2"): ("C#", ["C#", "D"]),
        ("Db", "m6"): ("C#", ["C#", "A"]),
        ("Ab", "m2"): ("G#", ["G#", "A"]),
    }
    for root in dc.PROMPT_ROOTS:
        for item in dc.INTERVALS:
            p = dc.make_prompt("interval_spell", item, rng=_FixedRoot(root))
            assert not dc._has_double_accidental(p["expected"]), (root, item, p)
            if (root, item) in forced:
                exp_root, exp_notes = forced[(root, item)]
                assert p["root"] == exp_root and p["expected"] == exp_notes


class _FixedRoot:
    def __init__(self, root):
        self._root = root

    def choice(self, seq):
        return self._root


def test_interval_typed_grade_enharmonic_tolerant():
    assert dc.grade_typed_note("G", "G")["correct"]
    res = dc.grade_typed_note("Cb", "B")
    assert res["correct"] and not res["spelled_correctly"]
    assert not dc.grade_typed_note("Ab", "G")["correct"]
    assert not dc.grade_typed_note("garbage", "G")["correct"]


def test_interval_ear_prompt_choices_contain_answer():
    p = dc.make_prompt("interval_ear", "P4", rng=random.Random(3))
    assert p["answer"] == "P4"
    assert "P4" in p["choices"] and len(p["choices"]) == 3
    assert len(set(p["choices"])) == 3
    assert p["play_notes"]
    assert dc.grade_named(p, "P4")["correct"]
    assert not dc.grade_named(p, "A4")["correct"]


# ── Triads ───────────────────────────────────────────────────────────────────

def test_triad_spell_fsharp_diminished():
    rng = random.Random(0)
    while True:
        p = dc.make_prompt("triad_spell", "diminished", rng=rng)
        if p["root"] == "F#":
            break
    assert p["text"] == "F# diminished"
    assert p["expected"] == ["F#", "A", "C"]
    assert dc.grade_prompt_played(p, _events(["F#", "A", "C"]))["correct"]


def test_triad_inversion_checks_bass_note():
    rng = random.Random(0)
    while True:
        p = dc.make_prompt("triad_spell", "major", rng=rng, inversion=1)
        if p["root"] == "C":
            break
    assert p["bass"] == "E"
    assert "1st inversion" in p["text"]
    # E3-G3-C4: bass is the 3rd -> correct
    good = dc.grade_prompt_played(p, _events([52, 55, 60]))
    assert good["correct"] and good["bass_ok"]
    # Root position C4-E4-G4: same PC set, wrong bass
    bad = dc.grade_prompt_played(p, _events([60, 64, 67]))
    assert not bad["correct"] and not bad["bass_ok"]


def test_triad_inversion_rejects_bad_value():
    with pytest.raises(ValueError):
        dc.make_prompt("triad_spell", "major", rng=random.Random(0), inversion=3)


# ── Sevenths ─────────────────────────────────────────────────────────────────

def test_seventh_spell_m7b5_symbol_and_notes():
    rng = random.Random(0)
    while True:
        p = dc.make_prompt("seventh_spell", "halfdiminished7", rng=rng)
        if p["root"] == "C":
            break
    assert p["symbol"] == "Cm7b5"
    assert p["expected"] == ["C", "Eb", "Gb", "Bb"]
    assert dc.grade_prompt_played(p, _events(p["expected"]))["correct"]


def test_seventh_ear_round_trip():
    p = dc.make_prompt("seventh_ear", "dominant7", rng=random.Random(5))
    assert p["answer"] == "dominant7"
    assert "dominant7" in p["choices"]
    assert len(p["play_notes"]) == 4
    assert dc.grade_named(p, "dominant7")["correct"]


# ── Diatonic harmony ─────────────────────────────────────────────────────────

def test_diatonic_sevenths_of_c():
    chords = dc.diatonic_sevenths("C")
    assert [c["symbol"] for c in chords] == [
        "Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bm7b5"]
    assert [c["numeral"] for c in chords] == [
        "Imaj7", "ii7", "iii7", "IVmaj7", "V7", "vi7", "viiø7"]


def test_diatonic_numeral_prompt():
    rng = random.Random(0)
    while True:
        p = dc.make_prompt("diatonic", "ii7", rng=rng)
        if p["key"] == "Eb":
            break
    assert p["symbol"] == "Fm7"
    assert p["expected"] == ["F", "Ab", "C", "Eb"]
    assert dc.grade_prompt_played(p, _events(p["expected"]))["correct"]


def test_diatonic_list_grading_order_insensitive_and_enharmonic():
    p = dc.make_prompt("diatonic", "all", rng=random.Random(0))
    key = p["key"]
    symbols = [c["symbol"] for c in dc.diatonic_sevenths(key)]
    shuffled = list(reversed(symbols))
    assert dc.grade_diatonic_list(p, shuffled)["correct"]
    res = dc.grade_diatonic_list(p, symbols[:-1])
    assert not res["correct"] and len(res["missing"]) == 1
    res = dc.grade_diatonic_list(p, symbols[:-1] + ["Q7"])
    assert not res["correct"] and res["wrong"] == ["Q7"]


def test_diatonic_list_enharmonic_root_accepted():
    p = {"expected_chords": dc.diatonic_sevenths("F")}
    symbols = [c["symbol"] for c in p["expected_chords"]]
    assert "Bbmaj7" in symbols
    swapped = ["A#maj7" if s == "Bbmaj7" else s for s in symbols]
    assert dc.grade_diatonic_list(p, swapped)["correct"]


# ── Function ─────────────────────────────────────────────────────────────────

def test_function_of():
    assert [dc.function_of(n) for n in ["I", "ii", "iii", "IV", "V", "vi", "vii°"]] == [
        "T", "PD", "T", "PD", "D", "T", "D"]


def test_function_grading_with_resolution():
    p = dc.make_prompt("function", "I-vi-ii-V", rng=random.Random(0))
    assert p["expected_functions"] == ["T", "T", "PD", "D"]
    assert len(p["symbols"]) == 4
    good = dc.grade_function(p, ["t", "T", "pd", "d"], resolution="I")
    assert good["correct"]
    wrong_res = dc.grade_function(p, ["T", "T", "PD", "D"], resolution="IV")
    assert wrong_res["labels_correct"] and not wrong_res["correct"]
    no_res = dc.grade_function(p, ["T", "T", "PD", "D"])
    assert not no_res["correct"]


def test_function_grading_without_resolution_question():
    p = dc.make_prompt("function", "ii-V-I", rng=random.Random(0))
    assert p["resolves_to"] is None
    assert dc.grade_function(p, ["PD", "D", "T"])["correct"]
    assert not dc.grade_function(p, ["PD", "D", "D"])["correct"]


# ── Guide tones ──────────────────────────────────────────────────────────────

def test_guide_tones_in_d():
    chords = dc.guide_tones("D")
    assert [c["symbol"] for c in chords] == ["Em7", "A7", "Dmaj7"]
    assert chords[0]["guide_tones"] == ["G", "D"]
    assert chords[1]["guide_tones"] == ["C#", "G"]
    assert chords[2]["guide_tones"] == ["F#", "C#"]
    assert [c["avoid_root"] for c in chords] == ["E", "A", "D"]


def test_guide_tone_grading_rootless():
    p = dc.make_prompt("guide_tones", "D")
    segs = [_events(c["guide_tones"]) for c in p["chords"]]
    assert dc.grade_guide_tones(p, segs)["correct"]
    # Sneak the root into the first chord -> fails that chord
    segs_with_root = [_events(["E", "G", "D"])] + segs[1:]
    res = dc.grade_guide_tones(p, segs_with_root)
    assert not res["correct"] and not res["chords"][0]["correct"]
    assert res["chords"][1]["correct"]
    # Wrong segment count
    assert not dc.grade_guide_tones(p, segs[:2])["correct"]


# ── Gating (mirrors ear_unlocks) ─────────────────────────────────────────────

def test_first_kind_unlocks_from_nothing():
    assert dc.kind_unlocks([]) == ["interval_spell"]


def test_kind_unlocks_requires_prereq_at_box():
    rows = _rows("interval_spell", box=2)
    assert dc.kind_unlocks(rows) == []
    rows = _rows("interval_spell", box=drill.UNLOCK_BOX)
    assert dc.kind_unlocks(rows) == ["interval_ear"]


def test_kind_unlocks_partial_deck_blocks():
    rows = _rows("interval_spell", box=5)
    rows[0]["box"] = 1
    assert dc.kind_unlocks(rows) == []


def test_kind_unlocks_chain_in_week_order():
    rows = _rows("interval_spell", box=5) + _rows("interval_ear", box=5)
    assert dc.kind_unlocks(rows) == ["triad_spell"]
    rows += _rows("triad_spell", box=3) + _rows("seventh_spell", box=3)
    # Chain is linear in week order: diatonic waits on seventh_ear, not
    # seventh_spell.
    assert dc.kind_unlocks(rows) == ["seventh_ear"]
    rows += _rows("seventh_ear", box=3)
    assert dc.kind_unlocks(rows) == ["diatonic"]


def test_scale_drill_rows_do_not_confuse_kind_gating():
    scale_rows = [{"key": "C", "direction": "spell", "box": 5, "due_at": 0}]
    assert dc.kind_unlocks(scale_rows) == ["interval_spell"]


# ── Scheduler compatibility ──────────────────────────────────────────────────

def test_card_rows_work_with_pick_next_and_schedule_after():
    rows = _rows("interval_spell", box=1)
    rows[3]["due_at"] = -5
    picked = drill.pick_next(rows, now=0)
    assert picked is rows[3]
    new_box, new_due = drill.schedule_after(picked["box"], True, now=1000)
    assert new_box == 2 and new_due > 1000
    new_box, new_due = drill.schedule_after(picked["box"], False, now=1000)
    assert new_box == 1 and new_due == 1000


def test_grade_played_wrapper_unchanged():
    events = _events(["C", "D", "E", "F", "G", "A", "B"])
    res = drill.grade_played(events, "C")
    assert res["correct"] and res["expected"][0] == "C"


# ── Endpoints (TestClient against a temp DB) ─────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # let init_db create it fresh
    monkeypatch.setenv("MIDIMI_DB", path)
    # Import after env is set so module-level init_db() uses the temp DB.
    sys.modules.pop("server", None)
    import server  # noqa: F401
    from fastapi.testclient import TestClient
    with TestClient(server.app) as c:
        yield c
    if os.path.exists(path):
        os.unlink(path)


def _seed(kind, item, due_at=1):
    """Force a card row to exist and be the most-overdue one."""
    import sqlite3
    import server

    conn = sqlite3.connect(server.DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO card_drill (kind, item, box, due_at, streak) "
        "VALUES (?, ?, 1, ?, 0)", (kind, item, due_at))
    conn.commit()
    conn.close()


def test_fresh_db_seeds_only_the_first_kind(client):
    s = client.get("/drill/cards/status").json()
    by_kind = {k["kind"]: k for k in s["kinds"]}
    assert by_kind["interval_spell"]["unlocked"] is True
    assert by_kind["interval_ear"]["unlocked"] is False
    assert s["due_today"] == len(dc.INTERVALS)


def test_next_hides_the_answer(client):
    p = client.get("/drill/cards/next").json()["prompt"]
    assert p["kind"] == "interval_spell"
    # `root` and `text` are the question; `expected`/`answer_note` must not leak.
    assert "root" in p and "text" in p
    assert "expected" not in p and "answer_note" not in p


def test_typed_interval_grades_and_promotes(client):
    p = client.get("/drill/cards/next").json()["prompt"]
    answer = dc._chord_names(p["root"], p["item"])[1]
    r = client.post("/drill/cards/grade", json={"note": answer}).json()
    assert r["correct"] is True
    assert r["box"] == 2
    assert r["prompt"]["answer_note"] == answer  # revealed only after grading


def test_typed_spell_answer_is_pitch_class_tolerant(client):
    _seed("triad_spell", "major")
    p = client.get("/drill/cards/next").json()["prompt"]
    assert p["kind"] == "triad_spell"
    # Respell every note the sharp way. Different letters, same pitch classes —
    # a keyboard can't express enharmonic intent, so this must still grade correct.
    sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    names = [sharp[NOTE_NAMES[n] % 12] for n in dc._chord_names(p["root"], "major")]
    r = client.post("/drill/cards/grade", json={"notes": names}).json()
    assert r["correct"] is True
    assert r["missing_notes"] == [] and r["wrong_notes"] == []


def test_typed_spell_wrong_note_reported(client):
    _seed("triad_spell", "major")
    client.get("/drill/cards/next")
    r = client.post("/drill/cards/grade", json={"notes": ["C", "C", "C"]}).json()
    assert r["correct"] is False
    assert r["box"] == 1


def test_practice_mode_does_not_touch_the_schedule(client):
    p = client.get("/drill/cards/next").json()["prompt"]
    answer = dc._chord_names(p["root"], p["item"])[1]
    r = client.post("/drill/cards/grade",
                    json={"note": answer, "graded": False}).json()
    assert r["correct"] is True
    assert r["box"] == 1  # unchanged


def test_ear_card_choices_and_play_prompt(client, monkeypatch):
    import server

    _seed("interval_ear", "M3")
    nxt = client.get("/drill/cards/next").json()
    p = nxt["prompt"]
    assert p["kind"] == "interval_ear"
    assert "M3" in p["choices"] and "answer" not in p
    # The server sounds the card; shipping play_notes would hand over the answer.
    assert "play_notes" not in p and p["has_audio"] is True

    calls = []
    monkeypatch.setattr(server, "_play", lambda notes, dur: calls.append(notes))
    assert client.post("/drill/cards/play_prompt").json()["ok"] is True
    import time as _t
    _t.sleep(0.3)
    assert calls  # arpeggio then the chord together

    r = client.post("/drill/cards/grade", json={"choice": "M3"}).json()
    assert r["correct"] is True and r["box"] == 2


def test_diatonic_list_card(client):
    _seed("diatonic", "all")
    p = client.get("/drill/cards/next").json()["prompt"]
    assert p["expected_count"] == 7
    assert "expected_chords" not in p
    symbols = [c["symbol"] for c in dc.diatonic_sevenths(p["key"])]
    r = client.post("/drill/cards/grade", json={"symbols": symbols}).json()
    assert r["correct"] is True


def test_function_card_asks_for_resolution(client):
    _seed("function", "I-IV-V")
    p = client.get("/drill/cards/next").json()["prompt"]
    assert p["ask_resolution"] is True
    assert "expected_functions" not in p and "resolves_to" not in p
    r = client.post("/drill/cards/grade", json={
        "functions": ["T", "PD", "D"], "resolution": "I"}).json()
    assert r["correct"] is True

    # Right labels, wrong resolution -> not correct.
    _seed("function", "I-IV-V")
    client.get("/drill/cards/next")
    r = client.post("/drill/cards/grade", json={
        "functions": ["T", "PD", "D"], "resolution": "IV"}).json()
    assert r["labels_correct"] is True and r["correct"] is False


def test_guide_tones_segmented_capture(client, monkeypatch):
    import server

    _seed("guide_tones", "C")
    p = client.get("/drill/cards/next").json()["prompt"]
    assert p["kind"] == "guide_tones"
    assert p["symbols"] == ["Dm7", "G7", "Cmaj7"]
    assert "chords" not in p  # guide tones themselves stay server-side

    chords = dc.guide_tones("C")
    monkeypatch.setattr(server.engine, "arm_recording", lambda: None)
    for chord in chords:
        events = [{"note": 60 + NOTE_NAMES[n], "on": True} for n in chord["guide_tones"]]
        monkeypatch.setattr(server.engine, "stop_recording", lambda e=events: e)
        seg = client.post("/drill/cards/segment").json()
        assert seg["total"] == 3
    assert seg["complete"] is True

    r = client.post("/drill/cards/grade", json={}).json()
    assert r["correct"] is True
    assert [c["symbol"] for c in r["chords"]] == ["Dm7", "G7", "Cmaj7"]


def test_guide_tones_rejects_the_root(client, monkeypatch):
    import server

    _seed("guide_tones", "C")
    client.get("/drill/cards/next")
    monkeypatch.setattr(server.engine, "arm_recording", lambda: None)
    for chord in dc.guide_tones("C"):
        # Play the root too — that's exactly what "rootless" forbids.
        names = chord["guide_tones"] + [chord["avoid_root"]]
        events = [{"note": 60 + NOTE_NAMES[n], "on": True} for n in names]
        monkeypatch.setattr(server.engine, "stop_recording", lambda e=events: e)
        client.post("/drill/cards/segment")
    r = client.post("/drill/cards/grade", json={}).json()
    assert r["correct"] is False


def test_clearing_a_deck_unlocks_the_next_kind(client):
    import sqlite3
    import server

    # Put every interval_spell row one grade below UNLOCK_BOX, all due.
    conn = sqlite3.connect(server.DB_PATH)
    conn.execute("UPDATE card_drill SET box=?, due_at=1 WHERE kind='interval_spell'",
                 (drill.UNLOCK_BOX - 1,))
    conn.commit()
    conn.close()

    for _ in range(len(dc.INTERVALS)):
        nxt = client.get("/drill/cards/next").json()
        if not nxt["due"] or nxt["prompt"]["kind"] != "interval_spell":
            break
        p = nxt["prompt"]
        client.post("/drill/cards/grade",
                    json={"note": dc._chord_names(p["root"], p["item"])[1]})

    by_kind = {k["kind"]: k for k in client.get("/drill/cards/status").json()["kinds"]}
    assert by_kind["interval_ear"]["unlocked"] is True
    assert by_kind["triad_spell"]["unlocked"] is False  # chain advances one step


def test_grade_without_an_active_card_is_a_400(client):
    assert client.post("/drill/cards/grade", json={"note": "C"}).status_code == 400


def test_double_accidental_intervals_are_lookupable():
    """chord_note_names spells by letter distance, so double flats occur naturally.

    m2 above Db is Ebb; if NOTE_NAMES can't resolve that, every grader that maps
    expected names to pitch classes raises KeyError instead of grading.
    """
    assert dc._chord_names("Db", "m2") == ["Db", "Ebb"]
    assert NOTE_NAMES["Ebb"] == NOTE_NAMES["D"]
    assert NOTE_NAMES["Bbb"] == NOTE_NAMES["A"]

    # And the graders actually work on them, both typed and played.
    assert dc.grade_typed_note("D", "Ebb")["correct"] is True
    prompt = {"kind": "interval_spell", "item": "m2", "expected": ["Db", "Ebb"]}
    assert dc.grade_prompt_played(prompt, _events(["Db", "D"]))["correct"] is True


def test_triad_inversion_does_not_leak_the_bass_note(client):
    """"C major, 1st inversion" is the question; naming the bass would answer it."""
    import sqlite3
    import server

    conn = sqlite3.connect(server.DB_PATH)
    conn.execute("INSERT OR REPLACE INTO card_drill (kind, item, box, due_at, streak) "
                 "VALUES ('triad_spell', 'major', 4, 1, 0)")
    conn.commit()
    conn.close()

    for _ in range(30):
        p = client.get("/drill/cards/next").json()["prompt"]
        assert "bass" not in p and "expected" not in p
        if "inversion" in p:
            assert "inversion" in p["text"] or "inversion" in p["text"].lower()
            break
        client.post("/drill/cards/grade", json={"notes": [], "graded": False})
        conn = sqlite3.connect(server.DB_PATH)
        conn.execute("UPDATE card_drill SET due_at=1 WHERE kind='triad_spell'")
        conn.commit()
        conn.close()


def test_jump_to_locked_kind_bypasses_the_unlock_gate(client):
    """`?kind=` serves a locked kind directly (dev jump-to-drill)."""
    # guide_tones is the last kind and locked on a fresh DB.
    status = {k["kind"]: k for k in client.get("/drill/cards/status").json()["kinds"]}
    assert status["guide_tones"]["unlocked"] is False

    r = client.get("/drill/cards/next", params={"kind": "guide_tones"}).json()
    assert r["due"] is True
    assert r["prompt"]["kind"] == "guide_tones"


def test_jump_to_unknown_kind_is_a_400(client):
    r = client.get("/drill/cards/next", params={"kind": "nope"})
    assert r.status_code == 400


def test_jump_into_locked_kind_does_not_seed_or_unlock(client):
    """Practising a locked kind must not create rows that would unlock it."""
    client.get("/drill/cards/next", params={"kind": "guide_tones"})
    # Answer it (graded); a not-yet-unlocked kind must stay locked afterwards.
    client.post("/drill/cards/grade", json={"symbols": [], "graded": True})

    status = {k["kind"]: k for k in client.get("/drill/cards/status").json()["kinds"]}
    assert status["guide_tones"]["unlocked"] is False


def test_jump_prefers_a_due_item_of_that_kind(client):
    """When rows exist, the jump serves that kind (interval_ear is locked but seeded here)."""
    _seed("interval_ear", "P5", due_at=1)
    r = client.get("/drill/cards/next", params={"kind": "interval_ear"}).json()
    assert r["prompt"]["kind"] == "interval_ear"
    assert r["prompt"]["item"] == "P5"
