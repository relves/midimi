"""Scale-patch drill scheduling policy (pure, no I/O).

Leitner-box spaced repetition over a fixed circle-of-fifths rotation. Route
handlers in server.py own persistence; everything here is plain data in/out so
it can be unit-tested without a DB or audio.
"""

# Circle-of-fifths order. The first 7 are the Slice-1 starter set; the rest stay
# dormant (absent from the DB) until maybe_unlock seeds them.
ROTATION = ["C", "G", "F", "D", "Bb", "A", "Eb", "B", "Db", "F#", "Gb", "Cb", "C#"]

STARTER_KEYS = ROTATION[:7]

DAY_SECONDS = 86400

# Days added to due_at on a correct answer, keyed by the NEW (post-promotion) box.
BOX_INTERVALS_DAYS = {1: 0, 2: 1, 3: 2, 4: 4, 5: 8}

MAX_BOX = 5
UNLOCK_BOX = 3  # all active keys must reach this box before the next key unlocks


def schedule_after(box: int, correct: bool, now: int) -> tuple[int, int]:
    """Return (new_box, new_due_at) after grading a key.

    Correct -> promote one box (capped at MAX_BOX), push due_at out by the new
    box's interval. Wrong -> reset to box 1, due immediately (re-drill this
    session via pick_next's most-overdue selection).
    """
    if correct:
        new_box = min(box + 1, MAX_BOX)
        return new_box, now + BOX_INTERVALS_DAYS[new_box] * DAY_SECONDS
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


def maybe_unlock(rows: list[dict], now: int) -> str | None:
    """Next ROTATION key to seed, or None.

    Unlocks only when every currently-active key is at box >= UNLOCK_BOX, keeping
    the plan's "start with the starter set, expand outward" pacing automatic.
    Returns the first ROTATION key not yet present in `rows`.
    """
    if not rows:
        return None
    if any(r.get("box", 1) < UNLOCK_BOX for r in rows):
        return None
    present = {r["key"] for r in rows}
    for key in ROTATION:
        if key not in present:
            return key
    return None
