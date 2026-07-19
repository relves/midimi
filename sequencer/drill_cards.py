"""Harmony drill card types (Slice C) — pure prompt/grade policy, no I/O.

New card kinds on the same Leitner scheduler as the scale drill (`drill.py`):
intervals → triads → sevenths → diatonic harmony → function → guide tones,
following the study plan's week order. Rows for these cards are dicts with
(kind, item, box, due_at); `drill.schedule_after` / `drill.pick_next` work
unchanged because they only look at box/due_at (pick_next ties break on "key",
so card rows carry a synthetic `key` of "kind:item").

Scope discipline (from the plan): no modes, no 9/11/13 extensions, no bebop
vocabulary, no advanced rootless voicings. The quality lists below are the
whole drill surface on purpose.
"""

import re

from sequencer.drill import ROTATION, UNLOCK_BOX, grade_played_notes

# ── Card kinds and the unlock graph ──────────────────────────────────────────
# Week order from the plan. Each kind unlocks when every row of its
# prerequisite kind has reached UNLOCK_BOX — the same "spell it before you're
# asked to recognise it" shape as drill.ear_unlocks, generalised to a chain.
KIND_ORDER = [
    "interval_spell",   # Wk1: "Major 3rd above Eb" -> play or type
    "interval_ear",     # Wk1: hear it -> name it
    "triad_spell",      # Wk2: "F# diminished" -> play; inversions as a variant
    "seventh_spell",    # Wk3: "Cm7b5" -> play
    "seventh_ear",      # Wk3: hear it -> name it
    "diatonic",         # Wk4: "ii7 in Ab" -> play; "list the diatonic 7ths in E"
    "function",         # Wk5: label T/PD/D; predict the resolution
    "guide_tones",      # Wk6: "ii-V-I in D, 3rds and 7ths only, rootless"
]

KIND_PREREQ = {kind: prev for prev, kind in zip(KIND_ORDER, KIND_ORDER[1:])}

# ── Decks ────────────────────────────────────────────────────────────────────
# Items are the Leitner unit; the prompt varies the root/key per rep so a box-5
# "M3" means "major thirds from any root", not "M3 above Eb memorised".

INTERVALS = ["m2", "M2", "m3", "M3", "P4", "A4", "P5", "m6", "M6", "m7", "M7"]

INTERVAL_LABELS = {
    "m2": "Minor 2nd", "M2": "Major 2nd", "m3": "Minor 3rd", "M3": "Major 3rd",
    "P4": "Perfect 4th", "A4": "Tritone", "P5": "Perfect 5th",
    "m6": "Minor 6th", "M6": "Major 6th", "m7": "Minor 7th", "M7": "Major 7th",
}

TRIAD_QUALITIES = ["major", "minor", "diminished", "augmented"]

TRIAD_LABELS = {
    "major": "major", "minor": "minor",
    "diminished": "diminished", "augmented": "augmented",
}

SEVENTH_QUALITIES = ["major7", "dominant7", "minor7", "halfdiminished7", "diminished7"]

SEVENTH_SUFFIX = {
    "major7": "maj7", "dominant7": "7", "minor7": "m7",
    "halfdiminished7": "m7b5", "diminished7": "dim7",
}

SEVENTH_LABELS = {
    "major7": "major 7th", "dominant7": "dominant 7th", "minor7": "minor 7th",
    "halfdiminished7": "half-diminished (m7b5)", "diminished7": "diminished 7th",
}

# Diatonic sevenths of a major key, in degree order. "all" is the
# list-them-all card from the plan.
DIATONIC_NUMERALS = ["Imaj7", "ii7", "iii7", "IVmaj7", "V7", "vi7", "viiø7"]
DIATONIC_ITEMS = DIATONIC_NUMERALS + ["all"]

# Roots for interval/triad/seventh prompts: every distinct pitch class once,
# spelled the way the rotation spells it (both F#- and flat-side names appear).
PROMPT_ROOTS = ["C", "G", "F", "D", "Bb", "A", "Eb", "B", "Db", "F#", "Ab", "E"]

# Keys for diatonic/guide-tone prompts: the scale-drill starter set, so these
# cards land on keys the learner has already spelled.
PROMPT_KEYS = ROTATION[:7]

# Wk5 progression bank. Functions: T (tonic: I, iii, vi), PD (predominant:
# ii, IV), D (dominant: V, vii°). `resolves_to` is the "predict the
# resolution" answer for the final chord.
FUNCTION_OF_DEGREE = {1: "T", 2: "PD", 3: "T", 4: "PD", 5: "D", 6: "T", 7: "D"}

FUNCTION_PROGRESSIONS = {
    "I-IV-V": {"numerals": ["I", "IV", "V"], "resolves_to": "I"},
    "I-vi-ii-V": {"numerals": ["I", "vi", "ii", "V"], "resolves_to": "I"},
    "ii-V-I": {"numerals": ["ii", "V", "I"], "resolves_to": None},
    "I-IV-I-V": {"numerals": ["I", "IV", "I", "V"], "resolves_to": "I"},
    "iii-vi-ii-V": {"numerals": ["iii", "vi", "ii", "V"], "resolves_to": "I"},
    "I-ii-iii-IV": {"numerals": ["I", "ii", "iii", "IV"], "resolves_to": "V"},
}

DECKS: dict[str, list[str]] = {
    "interval_spell": INTERVALS,
    "interval_ear": INTERVALS,
    "triad_spell": TRIAD_QUALITIES,
    "seventh_spell": SEVENTH_QUALITIES,
    "seventh_ear": SEVENTH_QUALITIES,
    "diatonic": DIATONIC_ITEMS,
    "function": list(FUNCTION_PROGRESSIONS),
    "guide_tones": PROMPT_KEYS,
}

# Guide-tone rows only appear over ii-V-I for now; the item is the key.
GUIDE_TONE_NUMERALS = ["ii7", "V7", "Imaj7"]


def _rng(rng):
    if rng is None:
        import random
        return random
    return rng


def _bare(name: str) -> str:
    """'Eb4' -> 'Eb' (chord_note_names returns octave-qualified names)."""
    return re.sub(r"-?\d+$", "", name)


def _chord_names(root: str, quality: str) -> list[str]:
    from sequencer.theory import chord_note_names

    return [_bare(n) for n in chord_note_names(root, quality)]


def card_key(kind: str, item: str) -> str:
    """Synthetic `key` for scheduler rows so drill.pick_next tie-breaks work."""
    return f"{kind}:{item}"


def seed_items(kind: str) -> list[str]:
    """The full deck for `kind`, seeded all at once when the kind unlocks."""
    return list(DECKS[kind])


def kind_rows(rows: list[dict], kind: str) -> list[dict]:
    return [r for r in rows if r.get("kind") == kind]


def kind_unlocks(rows: list[dict]) -> list[str]:
    """Kinds ready to be seeded, in week order (mirrors drill.ear_unlocks).

    The first kind unlocks unconditionally; each later kind unlocks once every
    row of its prerequisite has reached UNLOCK_BOX. Kinds that already have
    rows are never returned.
    """
    ready = []
    for kind in KIND_ORDER:
        if kind_rows(rows, kind):
            continue
        prereq = KIND_PREREQ.get(kind)
        if prereq is None:
            ready.append(kind)
            continue
        prereq_rows = kind_rows(rows, prereq)
        if prereq_rows and all(r.get("box", 1) >= UNLOCK_BOX for r in prereq_rows):
            ready.append(kind)
    return ready


# ── Diatonic helpers ─────────────────────────────────────────────────────────

_DIATONIC_TRIAD = ["major", "minor", "minor", "major", "major", "minor", "diminished"]


def _diatonic_numeral(degree: int):
    from sequencer.charts import Numeral, apply_mode

    triad = Numeral(degree=degree, alteration=0, quality=_DIATONIC_TRIAD[degree - 1])
    return apply_mode(triad, "seventh")


def diatonic_sevenths(key: str) -> list[dict]:
    """The seven diatonic 7th chords of `key` major, in degree order.

    Each entry: {numeral, symbol, notes} — e.g. in C, degree 5 is
    {"numeral": "V7", "symbol": "G7", "notes": ["G", "B", "D", "F"]}.
    """
    from sequencer.charts import numeral_to_symbol, symbol_root_quality

    out = []
    for degree in range(1, 8):
        numeral = _diatonic_numeral(degree)
        symbol = numeral_to_symbol(numeral, key)
        root, quality = symbol_root_quality(symbol)
        out.append({
            "numeral": numeral.text(),
            "symbol": symbol,
            "notes": _chord_names(root, quality),
        })
    return out


def guide_tones(key: str) -> list[dict]:
    """ii-V-I in `key`: per chord, the guide tones (3rd and 7th) and the root
    to leave out. E.g. in C: Dm7 -> F/C avoiding D."""
    from sequencer.charts import numeral_to_symbol, parse_numeral, symbol_root_quality

    out = []
    for text in GUIDE_TONE_NUMERALS:
        symbol = numeral_to_symbol(parse_numeral(text), key)
        root, quality = symbol_root_quality(symbol)
        names = _chord_names(root, quality)
        out.append({
            "numeral": text,
            "symbol": symbol,
            "guide_tones": [names[1], names[3]],  # 3rd and 7th of a stacked 7th chord
            "avoid_root": names[0],
        })
    return out


# ── Prompts ──────────────────────────────────────────────────────────────────

def make_prompt(kind: str, item: str, rng=None, inversion: int | None = None) -> dict:
    """Build one drill prompt for a card row.

    Returns a dict with `kind`, `item`, `text` (what to show/say), everything
    grading needs (expected notes, choices, ...), and for ear kinds the notes
    to sound. Pass `rng` for deterministic tests. `inversion` (1 or 2, triads
    only) is the Wk2 inversion variant; callers should offer it only once the
    row is established (e.g. box >= UNLOCK_BOX).
    """
    rng = _rng(rng)

    if kind in ("interval_spell", "interval_ear"):
        root = rng.choice(PROMPT_ROOTS)
        notes = _chord_names(root, item)
        if kind == "interval_spell":
            return {
                "kind": kind, "item": item,
                "text": f"{INTERVAL_LABELS[item]} above {root}",
                "root": root, "expected": notes, "answer_note": notes[1],
            }
        choices = _neighbour_choices(INTERVALS, item, rng)
        return {
            "kind": kind, "item": item,
            "text": "Name the interval you hear",
            "root": root, "play_notes": notes, "answer": item,
            "choices": choices,
            "choice_labels": {c: INTERVAL_LABELS[c] for c in choices},
        }

    if kind == "triad_spell":
        root = rng.choice(PROMPT_ROOTS)
        notes = _chord_names(root, item)
        prompt = {
            "kind": kind, "item": item,
            "text": f"{root} {TRIAD_LABELS[item]}",
            "root": root, "expected": notes,
        }
        if inversion:
            if inversion not in (1, 2):
                raise ValueError(f"Bad triad inversion: {inversion!r}")
            prompt["inversion"] = inversion
            prompt["bass"] = notes[inversion]
            ordinal = "1st" if inversion == 1 else "2nd"
            prompt["text"] += f", {ordinal} inversion"
        return prompt

    if kind in ("seventh_spell", "seventh_ear"):
        root = rng.choice(PROMPT_ROOTS)
        notes = _chord_names(root, item)
        symbol = root + SEVENTH_SUFFIX[item]
        if kind == "seventh_spell":
            return {
                "kind": kind, "item": item, "text": symbol,
                "root": root, "symbol": symbol, "expected": notes,
            }
        choices = _neighbour_choices(SEVENTH_QUALITIES, item, rng)
        return {
            "kind": kind, "item": item,
            "text": "Name the chord quality you hear",
            "root": root, "play_notes": notes, "answer": item,
            "choices": choices,
            "choice_labels": {c: SEVENTH_LABELS[c] for c in choices},
        }

    if kind == "diatonic":
        key = rng.choice(PROMPT_KEYS)
        if item == "all":
            return {
                "kind": kind, "item": item,
                "text": f"List the diatonic 7th chords in {key}",
                "key": key, "expected_chords": diatonic_sevenths(key),
            }
        from sequencer.charts import numeral_to_symbol, parse_numeral, symbol_root_quality

        symbol = numeral_to_symbol(parse_numeral(item), key)
        root, quality = symbol_root_quality(symbol)
        return {
            "kind": kind, "item": item,
            "text": f"{item} in {key}",
            "key": key, "symbol": symbol, "expected": _chord_names(root, quality),
        }

    if kind == "function":
        key = rng.choice(PROMPT_KEYS)
        prog = FUNCTION_PROGRESSIONS[item]
        from sequencer.charts import numeral_to_symbol, parse_numeral

        numerals = prog["numerals"]
        return {
            "kind": kind, "item": item,
            "text": f"Label each chord T, PD or D: {' - '.join(numerals)} in {key}",
            "key": key,
            "numerals": numerals,
            "symbols": [numeral_to_symbol(parse_numeral(n), key) for n in numerals],
            "expected_functions": [function_of(n) for n in numerals],
            "resolves_to": prog["resolves_to"],
        }

    if kind == "guide_tones":
        key = item  # the row's item *is* the key for guide-tone cards
        return {
            "kind": kind, "item": item,
            "text": f"ii-V-I in {key}, 3rds and 7ths only, rootless",
            "key": key, "chords": guide_tones(key),
        }

    raise ValueError(f"Unknown card kind: {kind!r}")


def _neighbour_choices(deck: list[str], answer: str, rng, n_distractors: int = 2) -> list[str]:
    """Answer plus deck-adjacent distractors, shuffled (cf. drill.ear_choices)."""
    idx = deck.index(answer)
    neighbours: list[str] = []
    for off in (1, -1, 2, -2, 3, -3, 4, -4):
        cand = deck[(idx + off) % len(deck)]
        if cand != answer and cand not in neighbours:
            neighbours.append(cand)
    choices = [answer] + neighbours[:max(0, n_distractors)]
    rng.shuffle(choices)
    return choices


def function_of(numeral_text: str) -> str:
    """T / PD / D for a diatonic numeral."""
    from sequencer.charts import parse_numeral

    return FUNCTION_OF_DEGREE[parse_numeral(numeral_text).degree]


# ── Grading ──────────────────────────────────────────────────────────────────
# Played answers all route through drill.grade_played_notes (pitch-class
# tolerant, which is right for a keyboard where enharmonic intent can't be
# expressed). Typed/named answers are graded here.

def grade_prompt_played(prompt: dict, events: list[dict]) -> dict:
    """Grade played MIDI events against a make_prompt result with `expected`."""
    return grade_played_notes(events, prompt["expected"], bass_name=prompt.get("bass"))


def grade_typed_note(answer: str, expected_name: str) -> dict:
    """Grade a typed note name, pitch-class tolerant but spelling-aware.

    `correct` on the right pitch class; `spelled_correctly` tells the UI
    whether to show "right note — but spelled Eb here" style feedback.
    """
    from sequencer.theory import NOTE_NAMES, normalize_note_name

    try:
        name = normalize_note_name(answer)
    except ValueError:
        return {"correct": False, "expected": expected_name, "answer": str(answer)}
    correct = NOTE_NAMES[name] == NOTE_NAMES[expected_name]
    return {
        "correct": correct,
        "expected": expected_name,
        "answer": name,
        "spelled_correctly": name == expected_name,
    }


def grade_named(prompt: dict, answer: str) -> dict:
    """Grade an ear card's multiple-choice/typed answer against prompt['answer']."""
    expected = prompt["answer"]
    picked = str(answer).strip()
    return {"correct": picked == expected, "expected": expected, "answer": picked}


def grade_diatonic_list(prompt: dict, answers: list[str]) -> dict:
    """Grade the Wk4 "list the diatonic 7ths" card: seven typed chord symbols.

    Order-insensitive; each answer is parsed and matched by root pitch class +
    quality, so 'Abmaj7' and 'G#maj7' both hit degree I of Ab (enharmonic
    tolerance, matching how played answers are graded).
    """
    from sequencer.charts import symbol_root_quality
    from sequencer.theory import NOTE_NAMES

    expected = prompt["expected_chords"]

    def _id(root: str, quality: str) -> tuple[int, str]:
        return NOTE_NAMES[root] % 12, quality

    expected_ids = {}
    for chord in expected:
        root, quality = symbol_root_quality(chord["symbol"])
        expected_ids[_id(root, quality)] = chord["symbol"]

    matched: set[tuple[int, str]] = set()
    wrong: list[str] = []
    for raw in answers:
        try:
            root, quality = symbol_root_quality(str(raw))
            cid = _id(root, quality)
        except Exception:
            wrong.append(str(raw))
            continue
        if cid in expected_ids and cid not in matched:
            matched.add(cid)
        else:
            wrong.append(str(raw))

    missing = [sym for cid, sym in expected_ids.items() if cid not in matched]
    return {
        "correct": not wrong and not missing,
        "expected": [c["symbol"] for c in expected],
        "wrong": wrong,
        "missing": missing,
    }


def grade_function(prompt: dict, labels: list[str], resolution: str | None = None) -> dict:
    """Grade Wk5 function labels (T/PD/D per chord) and, when the progression
    ends on a dominant, the predicted resolution numeral."""
    from sequencer.charts import parse_numeral

    expected = prompt["expected_functions"]
    normalized = [str(x).strip().upper() for x in labels]
    labels_ok = normalized == expected

    result = {
        "labels_correct": labels_ok,
        "expected_functions": expected,
        "answer_functions": normalized,
        "correct": labels_ok,
    }

    expected_res = prompt.get("resolves_to")
    if expected_res is not None:
        res_ok = False
        if resolution is not None:
            try:
                res_ok = (parse_numeral(str(resolution)).degree
                          == parse_numeral(expected_res).degree)
            except ValueError:
                res_ok = False
        result["expected_resolution"] = expected_res
        result["resolution_correct"] = res_ok
        result["correct"] = labels_ok and res_ok
    return result


def grade_guide_tones(prompt: dict, segments: list[list[dict]]) -> dict:
    """Grade Wk6 guide tones: one event segment per chord of the ii-V-I.

    Each segment must be exactly that chord's 3rd + 7th (pitch-class
    tolerant) with the root absent — grade_played_notes already fails any
    played note outside the expected set, which is what "rootless" means here.
    """
    chords = prompt["chords"]
    if len(segments) != len(chords):
        return {
            "correct": False,
            "error": f"expected {len(chords)} chords, got {len(segments)}",
            "chords": [c["symbol"] for c in chords],
        }
    per_chord = []
    for chord, events in zip(chords, segments):
        graded = grade_played_notes(events, chord["guide_tones"])
        graded["symbol"] = chord["symbol"]
        per_chord.append(graded)
    return {
        "correct": all(g["correct"] for g in per_chord),
        "chords": per_chord,
    }
