"""Scale-patch drill scheduling policy (pure, no I/O).

Leitner-box spaced repetition over a fixed circle-of-fifths rotation. Route
handlers in server.py own persistence; everything here is plain data in/out so
it can be unit-tested without a DB or audio.
"""

import datetime

# Circle-of-fifths order. The first 7 are the Slice-1 starter set; the rest stay
# dormant (absent from the DB) until maybe_unlock seeds them.
ROTATION = ["C", "G", "F", "D", "Bb", "A", "Eb", "B", "Db", "F#", "Gb", "Cb", "C#"]

STARTER_KEYS = ROTATION[:7]

DAY_SECONDS = 86400

# Days added to due_at on a correct answer, keyed by the NEW (post-promotion) box.
BOX_INTERVALS_DAYS = {1: 0, 2: 1, 3: 2, 4: 4, 5: 8}

MAX_BOX = 5
UNLOCK_BOX = 3  # all active keys must reach this box before the next key unlocks
EAR_UNLOCK_BOX = 3  # a key's spell box must reach this before its ear row unlocks

# Drill directions. "spell" = key->notes (Slice 1); "ear" = sound->key (Slice 2).
SPELL = "spell"
EAR = "ear"


def _due_at_local_midnight(now: int, days: int) -> int:
    """Epoch seconds for 00:00 local time, `days` calendar days after `now`.

    `days == 0` means "due immediately" (returns `now` unchanged). For days >= 1
    we snap to the start of the target local day so a card scheduled "tomorrow"
    is waiting the moment the next calendar day begins, regardless of what clock
    time the previous session ended at. This avoids the rolling-24h lockout where
    a 9pm session wouldn't come due again until 9pm the next day.
    """
    if days <= 0:
        return now
    local = datetime.datetime.fromtimestamp(now).astimezone()
    target = (local + datetime.timedelta(days=days)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return int(target.timestamp())


def schedule_after(box: int, correct: bool, now: int) -> tuple[int, int]:
    """Return (new_box, new_due_at) after grading a key.

    Correct -> promote one box (capped at MAX_BOX), push due_at out to the start
    of the local day the new box's interval lands on. Wrong -> reset to box 1,
    due immediately (re-drill this session via pick_next's most-overdue
    selection).
    """
    if correct:
        new_box = min(box + 1, MAX_BOX)
        return new_box, _due_at_local_midnight(now, BOX_INTERVALS_DAYS[new_box])
    return 1, now


def pick_next(rows: list[dict], now: int) -> dict | None:
    """Most-overdue active key whose due_at <= now; None if nothing is due.

    `rows` are scale_drill rows as dicts (key, box, due_at, ...). Ties broken by
    key name for determinism.
    """
    due = [r for r in rows if r.get("due_at") is not None and r["due_at"] <= now]
    if not due:
        return None
    return min(due, key=lambda r: (r["due_at"], r["key"]))


def _direction(row: dict) -> str:
    """Direction of a row, defaulting to SPELL for Slice-1 rows without the column."""
    return row.get("direction", SPELL)


def _spell_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if _direction(r) == SPELL]


def maybe_unlock(rows: list[dict], now: int) -> str | None:
    """Next ROTATION key to seed (as a `spell` row), or None.

    Unlocks only when every currently-active *spell* key is at box >= UNLOCK_BOX,
    keeping the plan's "start with the starter set, expand outward" pacing
    automatic. Returns the first ROTATION key without a spell row yet.
    """
    spell = _spell_rows(rows)
    if not spell:
        return None
    if any(r.get("box", 1) < UNLOCK_BOX for r in spell):
        return None
    present = {r["key"] for r in spell}
    for key in ROTATION:
        if key not in present:
            return key
    return None


def ear_unlocks(rows: list[dict]) -> list[str]:
    """Keys whose `spell` row has reached EAR_UNLOCK_BOX but have no `ear` row yet.

    "You should be able to spell it before you're asked to recognize it by ear."
    Returned in ROTATION order for determinism.
    """
    ear_present = {r["key"] for r in rows if _direction(r) == EAR}
    ready = {
        r["key"]
        for r in _spell_rows(rows)
        if r.get("box", 1) >= EAR_UNLOCK_BOX and r["key"] not in ear_present
    }
    return [k for k in ROTATION if k in ready]


def ear_choices(key: str, n_distractors: int = 2, rng=None) -> list[str]:
    """Multiple-choice options for an `ear` prompt: the true `key` plus distinct
    circle-of-fifths neighbours as distractors. Never repeats the answer.

    Neighbours (closest fifths first) make plausible distractors. Enharmonic
    equivalents of the answer (F#/Gb, B/Cb, Db/C#) are excluded: they sound
    identical, so offering one makes the prompt unanswerable by ear and grading
    would mark a musically correct hearing wrong.

    Pass an `rng` with `.shuffle` for deterministic tests; defaults to module
    `random`.
    """
    import random as _random

    from sequencer.theory import NOTE_NAMES

    rng = rng or _random
    seen_pcs = {NOTE_NAMES[key]}
    idx = ROTATION.index(key)
    neighbours: list[str] = []
    for off in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
        cand = ROTATION[(idx + off) % len(ROTATION)]
        pc = NOTE_NAMES[cand]
        if pc in seen_pcs:
            continue
        seen_pcs.add(pc)
        neighbours.append(cand)
    choices = [key] + neighbours[:max(0, n_distractors)]
    rng.shuffle(choices)
    return choices


def grade_played_notes(
    events: list[dict],
    expected_names: list[str],
    bass_name: str | None = None,
) -> dict:
    """Grade recorded MIDI note-on events against expected note names,
    pitch-class tolerant.

    Extracts note-on pitch classes in played order, de-dupes octaves (so a
    two-octave run still reads as the expected set), and compares to the
    expected pitch-class set. `correct` iff the played PC set == expected set
    AND no played note falls outside it. `played_names` re-spells played notes
    with the expected spelling so a right answer reinforces the spelling.

    `bass_name` (for inversion prompts) additionally requires the lowest played
    MIDI note to be that pitch class — the one part of voicing a PC set can't
    check.
    """
    from sequencer.theory import NOTE_NAMES

    name_by_pc = {NOTE_NAMES[n]: n for n in expected_names}
    expected_pcs = [NOTE_NAMES[n] for n in expected_names]
    expected_set = set(expected_pcs)

    _SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    played_midi = [e["note"] for e in events if e.get("on")]
    played_pcs: list[int] = []
    seen: set[int] = set()
    for m in played_midi:
        pc = m % 12
        if pc not in seen:
            seen.add(pc)
            played_pcs.append(pc)

    played_set = set(played_pcs)
    wrong_pcs = [pc for pc in played_pcs if pc not in expected_set]
    missing_pcs = [pc for pc in expected_pcs if pc not in played_set]
    correct = played_set == expected_set and not wrong_pcs

    result = {
        "correct": correct,
        "expected": expected_names,
        "played_midi": played_midi,
        "played_names": [name_by_pc.get(pc, _SHARP[pc]) for pc in played_pcs],
        "wrong_notes": [_SHARP[pc] for pc in wrong_pcs],
        "missing_notes": [name_by_pc[pc] for pc in missing_pcs],
    }

    if bass_name is not None:
        bass_pc = NOTE_NAMES[bass_name]
        bass_ok = bool(played_midi) and min(played_midi) % 12 == bass_pc
        result["expected_bass"] = bass_name
        result["bass_ok"] = bass_ok
        result["correct"] = correct and bass_ok

    return result


def grade_played(events: list[dict], key: str) -> dict:
    """Grade recorded MIDI note-on events against `key` major (Slice-1 scale drill)."""
    from sequencer.theory import major_scale_notes

    return grade_played_notes(events, major_scale_notes(key))
