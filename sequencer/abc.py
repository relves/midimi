"""ABC notation parser/serializer (Phase 1 subset).

Supported subset:
  Headers: X: T: M: L: Q: K:
  Notes: A-G/a-g with octave marks (' ,), accidentals (^ _ = ^^ __),
         duration multipliers (N, /N, N/N)
  Chords: [CEG] with duration multiplier
  Chord symbols: "..." (annotation — ignored for playback)
  Rests: z/Z
  Barlines: | || [| |] (treated uniformly)
  Repeats: |: :| (expanded on parse — section played twice)
  Ties: - (extends note duration into the next note of same pitch)

Rejects tokens outside this subset with a bar/token-precise error message.

Octave convention (standard ABC):
  Uppercase letter = base octave 3 (C = C3 = MIDI 48)
  Lowercase letter = base octave 4 (c = C4 = MIDI 60 = middle C)
  ' raises by one octave, , lowers by one octave
"""

import re
import time
from fractions import Fraction

# ── Pitch helpers ─────────────────────────────────────────────────────────────

_LETTER_PC: dict[str, int] = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
}

# ABC octave base: uppercase = octave 3, lowercase = octave 4
_BASE_OCTAVE = {'upper': 3, 'lower': 4}

# Sharp/flat names for MIDI → note name
_SHARP_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_FLAT_NAMES  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# ── Key signatures ────────────────────────────────────────────────────────────

def _build_key_sigs() -> dict[str, dict[str, int]]:
    sharp_order = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    flat_order  = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
    major_entries = [
        ('C', 0), ('G', 1), ('D', 2), ('A', 3), ('E', 4), ('B', 5), ('F#', 6), ('C#', 7),
        ('F', -1), ('Bb', -2), ('Eb', -3), ('Ab', -4), ('Db', -5), ('Gb', -6), ('Cb', -7),
    ]
    sigs: dict[str, dict[str, int]] = {}
    for root, n in major_entries:
        if n >= 0:
            sig: dict[str, int] = {l: 1 for l in sharp_order[:n]}
        else:
            sig = {l: -1 for l in flat_order[:-n]}
        for alias in (root, root + 'maj', root + 'M', root + 'major'):
            sigs[alias] = sig
        # relative minor (down 3 semitones on the circle → same key sig)
        # e.g. Am is relative of C, Em of G, etc.
        # Build minor name from relative major
        _RELATIVE_MINOR = {
            'C': 'A', 'G': 'E', 'D': 'B', 'A': 'F#', 'E': 'C#', 'B': 'G#', 'F#': 'D#',
            'C#': 'A#', 'F': 'D', 'Bb': 'G', 'Eb': 'C', 'Ab': 'F', 'Db': 'Bb', 'Gb': 'Eb',
            'Cb': 'Ab',
        }
        minor_root = _RELATIVE_MINOR.get(root, '')
        if minor_root:
            for alias in (minor_root + 'm', minor_root + 'min', minor_root + 'minor'):
                sigs[alias] = sig
    return sigs

_KEY_SIGS = _build_key_sigs()


def _parse_key(k_value: str) -> dict[str, int]:
    """Parse K: header value and return accidental map {letter: +1/-1}."""
    k = k_value.strip()
    # Normalize: 'Hp'/'HP' = highland pipes, treat as D (mixolydian) – rare, ignore
    # Remove mode suffixes that don't affect our key sig lookup
    # Try direct lookup first
    if k in _KEY_SIGS:
        return _KEY_SIGS[k]
    # Try normalizing whitespace in things like "D maj"
    normalized = k.replace(' ', '')
    if normalized in _KEY_SIGS:
        return _KEY_SIGS[normalized]
    # Common modes: treat as major (dor, phr, lyd, mix, aeo, loc, exp)
    for suffix in ('dor', 'phr', 'lyd', 'mix', 'aeo', 'loc', 'exp', 'ion'):
        if normalized.endswith(suffix):
            root = normalized[:-len(suffix)]
            if root in _KEY_SIGS:
                return _KEY_SIGS[root]
    # No match → default to C
    return {}


# ── Header parsing ────────────────────────────────────────────────────────────

_HEADER_RE = re.compile(r'^([A-Za-z]):(.*)$')


def _parse_headers(lines: list[str]) -> tuple[dict[str, str], int]:
    """Parse ABC headers from lines. Returns (headers_dict, first_body_line_index)."""
    headers: dict[str, str] = {}
    i = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        m = _HEADER_RE.match(line)
        if m:
            field, value = m.group(1).upper(), m.group(2).strip()
            headers[field] = value
            # K: is always last header
            if field == 'K':
                return headers, i + 1
        else:
            # Non-header line in header section → body starts here
            return headers, i
    return headers, i + 1


def _parse_meter(m_value: str) -> tuple[int, int]:
    """Parse M: value like '4/4' or 'C' or 'C|'."""
    v = m_value.strip()
    if v in ('C', 'c'):
        return 4, 4
    if v in ('C|', 'c|'):
        return 2, 2
    try:
        num, den = v.split('/', 1)
        return int(num.strip()), int(den.strip())
    except ValueError:
        raise ABCParseError(f"Cannot parse meter {m_value!r}")


def _parse_unit_length(l_value: str) -> Fraction:
    """Parse L: value like '1/8' → Fraction(1, 8)."""
    v = l_value.strip()
    try:
        if '/' in v:
            num, den = v.split('/', 1)
            return Fraction(int(num.strip()), int(den.strip()))
        return Fraction(int(v))
    except (ValueError, ZeroDivisionError):
        raise ABCParseError(f"Cannot parse unit length {l_value!r}")


def _default_unit_length(numerator: int, denominator: int) -> Fraction:
    """Derive default L: from M: if L: not given.
    If meter >= 3/4, use 1/8; otherwise 1/16.
    """
    if Fraction(numerator, denominator) >= Fraction(3, 4):
        return Fraction(1, 8)
    return Fraction(1, 16)


def _parse_tempo(q_value: str) -> float:
    """Parse Q: value. Supports '120', '1/4=120', '3/8=80' etc."""
    v = q_value.strip()
    # Remove optional label in double quotes at start: Q:"Fast" 1/4=120
    v = re.sub(r'^"[^"]*"\s*', '', v)
    if '=' in v:
        # e.g. '1/4=120' or '3/8=80' – the note-length=bpm form
        note_part, bpm_part = v.split('=', 1)
        try:
            bpm = float(bpm_part.strip())
        except ValueError:
            raise ABCParseError(f"Cannot parse tempo {q_value!r}")
        # note_part could be '1/4', '3/8', etc. – this specifies the beat unit
        # We want tempo in quarter-note BPM; convert if needed
        try:
            if '/' in note_part:
                n, d = note_part.strip().split('/')
                beat_fraction = Fraction(int(n), int(d))
            else:
                beat_fraction = Fraction(int(note_part.strip()))
            # bpm is given in these note values per minute
            # Convert to quarter-note BPM: quarter = 1/4 note
            quarter_bpm = float(bpm * beat_fraction * 4)
        except (ValueError, ZeroDivisionError):
            quarter_bpm = bpm
        return quarter_bpm
    else:
        try:
            return float(v)
        except ValueError:
            raise ABCParseError(f"Cannot parse tempo {q_value!r}")


# ── Token-level parsing ───────────────────────────────────────────────────────

class ABCParseError(ValueError):
    pass


# Matches a single note (possibly with accidental, octave, duration)
# Group names: acc, letter, octave, num, den
_NOTE_TOKEN_RE = re.compile(
    r'(?P<acc>[_^=]{1,2})?'
    r'(?P<letter>[A-Ga-g])'
    r'(?P<octave>[,\']*)'
    r'(?P<num>\d+)?'
    r'(?P<slash>/)?'
    r'(?P<den>\d+)?'
)

# Matches a rest
_REST_TOKEN_RE = re.compile(
    r'[zZ]'
    r'(?P<num>\d+)?'
    r'(?P<slash>/)?'
    r'(?P<den>\d+)?'
)

# Matches a barline (various forms)
_BARLINE_RE = re.compile(r'\|\|?|\[[\|]|[\|]\]')

# Start/end repeat barlines
_START_REPEAT_RE = re.compile(r'\|:')
_END_REPEAT_RE   = re.compile(r':[\|]|::')
_FIRST_ENDING_RE = re.compile(r'\[1')
_SECOND_ENDING_RE = re.compile(r'\[2')


def _parse_duration(num_s: str | None, slash: str | None, den_s: str | None, unit: Fraction) -> Fraction:
    """Compute note duration in quarter-note beats given ABC duration tokens."""
    # Numerator
    num = int(num_s) if num_s else 1
    # Denominator
    if slash is None:
        den = 1
    elif den_s:
        den = int(den_s)
    else:
        den = 2  # bare '/' means halve
    duration_units = Fraction(num, den)
    # duration in beats (quarter notes): unit is fraction of a whole note, quarter = 1/4
    # unit in beats = unit * 4
    return duration_units * unit * 4


def _note_midi_and_name(
    letter: str,
    acc_str: str,
    octave_str: str,
    key_sig: dict[str, int],
    bar_accidentals: dict[str, int],
) -> tuple[int, str]:
    """Compute MIDI note number and note name string (e.g. 'C#4') from ABC note components."""
    upper = letter.upper()
    is_lower = letter.islower()

    # Base octave
    octave = _BASE_OCTAVE['lower'] if is_lower else _BASE_OCTAVE['upper']
    for ch in octave_str:
        if ch == "'":
            octave += 1
        elif ch == ',':
            octave -= 1

    # Accidental offset
    if acc_str:
        if acc_str == '^':
            acc_offset = 1
        elif acc_str == '^^':
            acc_offset = 2
        elif acc_str == '_':
            acc_offset = -1
        elif acc_str == '__':
            acc_offset = -2
        elif acc_str == '=':
            acc_offset = 0
        else:
            acc_offset = 0
        # Apply to bar_accidentals so it persists until barline
        bar_accidentals[upper] = acc_offset
    else:
        # Check bar accidentals first (they take priority over key sig)
        if upper in bar_accidentals:
            acc_offset = bar_accidentals[upper]
        else:
            acc_offset = key_sig.get(upper, 0)

    pc = _LETTER_PC[upper]
    midi = 12 * (octave + 1) + pc + acc_offset

    # Build note name string
    if acc_offset == 1:
        acc_name = '#'
    elif acc_offset == 2:
        acc_name = '##'
    elif acc_offset == -1:
        acc_name = 'b'
    elif acc_offset == -2:
        acc_name = 'bb'
    else:
        acc_name = ''
    note_name = f"{upper}{acc_name}{octave}"

    return midi, note_name


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize_body(body: str) -> list[tuple[str, str]]:
    """Tokenize ABC body into (type, raw) pairs.

    Types: 'note', 'rest', 'chord', 'barline', 'start_repeat', 'end_repeat',
           'first_ending', 'second_ending', 'chord_symbol', 'tie', 'unknown'
    """
    tokens = []
    pos = 0
    s = body
    while pos < len(s):
        ch = s[pos]

        # Skip whitespace and newlines
        if ch in ' \t\r\n':
            pos += 1
            continue

        # Comment to end of line
        if ch == '%':
            nl = s.find('\n', pos)
            pos = nl + 1 if nl != -1 else len(s)
            continue

        # Chord symbol annotation "..."
        if ch == '"':
            end = s.find('"', pos + 1)
            if end == -1:
                end = len(s) - 1
            tokens.append(('chord_symbol', s[pos:end + 1]))
            pos = end + 1
            continue

        # Tie
        if ch == '-':
            tokens.append(('tie', '-'))
            pos += 1
            continue

        # Repeat barlines (must check before plain barline)
        if s[pos:pos+2] == '|:':
            tokens.append(('start_repeat', '|:'))
            pos += 2
            continue
        if s[pos:pos+2] in ('::', ':|'):
            tokens.append(('end_repeat', s[pos:pos+2]))
            pos += 2
            continue
        if s[pos:pos+2] == '::':
            tokens.append(('end_repeat', '::'))
            pos += 2
            continue

        # First/second ending markers [1 [2
        if s[pos:pos+2] == '[1':
            tokens.append(('first_ending', '[1'))
            pos += 2
            continue
        if s[pos:pos+2] == '[2':
            tokens.append(('second_ending', '[2'))
            pos += 2
            continue

        # Chord [CEG...]
        if ch == '[':
            end = s.find(']', pos + 1)
            if end == -1:
                tokens.append(('unknown', ch))
                pos += 1
                continue
            # Get optional duration after ]
            dur_start = end + 1
            dur_m = re.match(r'(\d+)?(/)?(\d+)?', s[dur_start:])
            dur_raw = dur_m.group(0) if dur_m else ''
            tokens.append(('chord', s[pos:end + 1 + len(dur_raw)]))
            pos = end + 1 + len(dur_raw)
            continue

        # Rest z/Z
        if ch in 'zZ':
            m = _REST_TOKEN_RE.match(s, pos)
            if m:
                tokens.append(('rest', m.group(0)))
                pos = m.end()
            else:
                tokens.append(('rest', ch))
                pos += 1
            continue

        # Note (possibly with leading accidental)
        if ch in '^_=' or ch in 'ABCDEFGabcdefg':
            m = _NOTE_TOKEN_RE.match(s, pos)
            if m:
                tokens.append(('note', m.group(0)))
                pos = m.end()
            else:
                tokens.append(('unknown', ch))
                pos += 1
            continue

        # Barlines
        if ch == '|':
            # Look ahead for ||
            if s[pos:pos+2] == '||':
                tokens.append(('barline', '||'))
                pos += 2
            elif s[pos:pos+2] == '|]':
                tokens.append(('barline', '|]'))
                pos += 2
            else:
                tokens.append(('barline', '|'))
                pos += 1
            continue
        if ch == ']':
            pos += 1
            continue

        # Skip decorations +...+ or !...!
        if ch in '+!':
            end = s.find(ch, pos + 1)
            pos = end + 1 if end != -1 else len(s)
            continue

        # Skip staccato dots and other single-char ornaments
        if ch in '.~HLMOPSTuv':
            pos += 1
            continue

        tokens.append(('unknown', ch))
        pos += 1

    return tokens


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_abc(text: str) -> dict:
    """Parse ABC notation text and return a normalized sequence dict.

    Returns the same structure as build_sequence():
      title, tempo_bpm, time_signature, time_signature_parts,
      events (with notes, note_names, at_beat, duration_beats, velocity, label, root, quality, octave),
      duration_ms, total_beats, key

    Raises ABCParseError with bar-precise messages on invalid input.
    """
    lines = text.splitlines()
    headers, body_start = _parse_headers(lines)
    body = '\n'.join(lines[body_start:])

    # Parse headers
    title = headers.get('T', 'Untitled').strip()

    ts_num, ts_den = _parse_meter(headers['M']) if 'M' in headers else (4, 4)
    time_signature = f"{ts_num}/{ts_den}"
    beats_per_bar = Fraction(ts_num * 4, ts_den)  # in quarter-note beats

    unit_length = _parse_unit_length(headers['L']) if 'L' in headers else _default_unit_length(ts_num, ts_den)

    tempo_bpm = _parse_tempo(headers['Q']) if 'Q' in headers else 96.0

    key_sig = _parse_key(headers.get('K', 'C'))
    key_name = headers.get('K', 'C').strip()

    # Tokenize body
    tokens = _tokenize_body(body)

    # Expand repeats: find |: ... :| sections and duplicate
    tokens = _expand_repeats(tokens)

    # Process tokens into events
    events: list[dict] = []
    errors: list[str] = []

    bar_num = 1
    beat_in_bar = Fraction(0)
    at_beat = Fraction(0)
    bar_accidentals: dict[str, int] = {}  # reset at each barline
    tie_pending: dict[int, int] = {}  # midi_note -> event index; only active when tie_active
    tie_active: bool = False  # True only immediately after a '-' token

    def _reset_bar():
        nonlocal bar_num, beat_in_bar, bar_accidentals, tie_pending, tie_active
        bar_num += 1
        beat_in_bar = Fraction(0)
        bar_accidentals = {}
        # ties can span barlines in ABC, so keep tie_pending/tie_active

    for tok_type, tok_raw in tokens:
        if tok_type in ('barline', 'first_ending', 'second_ending'):
            if beat_in_bar > 0:
                _check_bar(bar_num, beat_in_bar, beats_per_bar, errors, time_signature)
            _reset_bar()
            continue

        if tok_type == 'start_repeat':
            if beat_in_bar > 0:
                _check_bar(bar_num, beat_in_bar, beats_per_bar, errors, time_signature)
            _reset_bar()
            continue

        if tok_type == 'end_repeat':
            if beat_in_bar > 0:
                _check_bar(bar_num, beat_in_bar, beats_per_bar, errors, time_signature)
            _reset_bar()
            continue

        if tok_type in ('chord_symbol', 'unknown'):
            continue

        if tok_type == 'tie':
            # Record which events are tied; next note of same pitch extends them
            tie_active = True
            for note_midi, evt_idx in tie_pending.items():
                if evt_idx < len(events):
                    events[evt_idx]['_tied'] = True
            continue

        if tok_type == 'rest':
            m = _REST_TOKEN_RE.match(tok_raw)
            dur = _parse_duration(
                m.group('num') if m else None,
                m.group('slash') if m else None,
                m.group('den') if m else None,
                unit_length,
            )
            at_beat += dur
            beat_in_bar += dur
            tie_pending = {}
            tie_active = False
            continue

        if tok_type == 'note':
            m = _NOTE_TOKEN_RE.match(tok_raw)
            if not m:
                raise ABCParseError(f"Bar {bar_num}: cannot parse note token {tok_raw!r}")
            dur = _parse_duration(
                m.group('num'), m.group('slash'), m.group('den'), unit_length
            )
            midi, note_name = _note_midi_and_name(
                m.group('letter'), m.group('acc') or '', m.group('octave') or '',
                key_sig, bar_accidentals,
            )
            # Handle tie continuation: only extend if tie was explicitly marked
            if tie_active and midi in tie_pending:
                prev_idx = tie_pending[midi]
                if prev_idx < len(events):
                    events[prev_idx]['duration_beats'] += float(dur)
                    tie_pending = {midi: prev_idx}
                    tie_active = False
                    at_beat += dur
                    beat_in_bar += dur
                    continue
            tie_active = False
            evt_idx = len(events)
            tie_pending = {midi: evt_idx}
            octave = midi // 12 - 1
            events.append({
                'at_beat': float(at_beat),
                'duration_beats': float(dur),
                'notes': [midi],
                'note_names': [note_name],
                'root': note_name[:-1] if note_name[-1].isdigit() else note_name,
                'quality': 'note',
                'octave': octave,
                'velocity': 90,
                'label': note_name,
            })
            at_beat += dur
            beat_in_bar += dur
            continue

        if tok_type == 'chord':
            # Parse [notes]dur
            inner_m = re.match(r'\[([^\]]+)\]((\d+)?(/)?(\d+)?)', tok_raw)
            if not inner_m:
                inner_m = re.match(r'\[([^\]]+)\]', tok_raw)
                dur_raw = ''
            else:
                dur_raw = inner_m.group(2)
            if not inner_m:
                raise ABCParseError(f"Bar {bar_num}: cannot parse chord token {tok_raw!r}")

            inner = inner_m.group(1)
            # Parse duration from after ]
            if dur_raw:
                dur_m = re.match(r'(\d+)?(/)?(\d+)?', dur_raw)
                dur = _parse_duration(
                    dur_m.group(1) if dur_m else None,
                    dur_m.group(2) if dur_m else None,
                    dur_m.group(3) if dur_m else None,
                    unit_length,
                )
            else:
                dur = unit_length * 4  # 1 unit note

            # Parse each note inside the chord
            chord_midis = []
            chord_names = []
            chord_bar_acc = dict(bar_accidentals)  # chord notes share bar_accidentals
            for nm in _NOTE_TOKEN_RE.finditer(inner):
                midi, note_name = _note_midi_and_name(
                    nm.group('letter'), nm.group('acc') or '', nm.group('octave') or '',
                    key_sig, chord_bar_acc,
                )
                chord_midis.append(midi)
                chord_names.append(note_name)
            if not chord_midis:
                raise ABCParseError(f"Bar {bar_num}: empty chord {tok_raw!r}")

            # Update bar_accidentals from chord
            bar_accidentals.update(chord_bar_acc)

            octave = chord_midis[0] // 12 - 1
            events.append({
                'at_beat': float(at_beat),
                'duration_beats': float(dur),
                'notes': chord_midis,
                'note_names': chord_names,
                'root': chord_names[0][:-1] if chord_names[0][-1].isdigit() else chord_names[0],
                'quality': 'note',
                'octave': octave,
                'velocity': 90,
                'label': '+'.join(chord_names),
            })
            at_beat += dur
            beat_in_bar += dur
            tie_pending = {}
            tie_active = False
            continue

    # Final bar check (if no trailing barline)
    if beat_in_bar > 0 and beat_in_bar != beats_per_bar:
        # Allow the last bar to be a pickup (partial) bar without error
        # But report if it's over-full
        if beat_in_bar > beats_per_bar + Fraction(1, 64):
            errors.append(
                f"bar {bar_num} contains {float(beat_in_bar):.4g} beats; "
                f"meter {time_signature} expects {float(beats_per_bar):.4g}"
            )

    # Clean up internal fields
    for evt in events:
        evt.pop('_tied', None)

    if errors:
        raise ABCParseError('\n'.join(errors))

    if not events:
        raise ABCParseError("No notes found in ABC")

    total_beats = max(e['at_beat'] + e['duration_beats'] for e in events)
    duration_ms = int(total_beats * 60 / tempo_bpm * 1000)

    return {
        'title': title,
        'tempo_bpm': float(tempo_bpm),
        'time_signature': time_signature,
        'time_signature_parts': (ts_num, ts_den),
        'events': events,
        'duration_ms': duration_ms,
        'total_beats': total_beats,
        'key': key_name,
        'abc_errors': [],  # populated only on warnings (not errors)
    }


def _check_bar(bar_num: int, beat_in_bar: Fraction, beats_per_bar: Fraction, errors: list[str], ts: str = "") -> None:
    diff = abs(beat_in_bar - beats_per_bar)
    if diff > Fraction(1, 64):
        meter_str = f"meter {ts} expects" if ts else "meter expects"
        errors.append(
            f"bar {bar_num} contains {float(beat_in_bar):.4g} beats; "
            f"{meter_str} {float(beats_per_bar):.4g}"
        )


def _expand_repeats(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Expand |: ... :| repeats by duplicating the section."""
    result = []
    i = 0
    while i < len(tokens):
        tok_type, tok_raw = tokens[i]
        if tok_type == 'start_repeat':
            # Find matching end_repeat, handling nested (simplified: find first :|)
            result.append(tokens[i])
            section_start = i + 1
            i += 1
            depth = 1
            section_tokens = []
            first_ending_tokens: list[tuple[str, str]] = []
            second_ending_tokens: list[tuple[str, str]] = []
            in_first = False
            in_second = False
            while i < len(tokens):
                t, r = tokens[i]
                if t == 'start_repeat':
                    depth += 1
                    section_tokens.append(tokens[i])
                elif t == 'end_repeat':
                    depth -= 1
                    if depth == 0:
                        # Emit section once with first ending, once without first ending but with second
                        if first_ending_tokens or second_ending_tokens:
                            # First pass: main + first ending
                            result.extend(section_tokens)
                            result.extend(first_ending_tokens)
                            result.append(('barline', '|'))
                            # Second pass: main + second ending
                            result.extend(section_tokens)
                            result.extend(second_ending_tokens)
                        else:
                            # Simple repeat: play section twice
                            result.extend(section_tokens)
                            result.append(('barline', '|'))
                            result.extend(section_tokens)
                        i += 1
                        break
                    else:
                        section_tokens.append(tokens[i])
                elif t == 'first_ending':
                    in_first = True
                    in_second = False
                elif t == 'second_ending':
                    in_second = True
                    in_first = False
                else:
                    if in_first:
                        first_ending_tokens.append(tokens[i])
                    elif in_second:
                        second_ending_tokens.append(tokens[i])
                    else:
                        section_tokens.append(tokens[i])
                i += 1
        else:
            result.append(tokens[i])
            i += 1
    return result


# ── Serializer ────────────────────────────────────────────────────────────────

# Note names used when serializing back to ABC
_NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_NOTE_NAMES_FLAT  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


def _midi_to_abc_note(midi: int, note_name: str | None = None) -> str:
    """Convert a MIDI note number (and optional spelled note name) to ABC note token.

    Uses L:1/4 convention (each unit = 1 quarter beat).
    Returns just the letter+accidental+octave part (no duration).
    """
    if note_name:
        # Parse the note name: letter + acc + octave
        m = re.match(r'^([A-G])(#{1,2}|b{1,2}|bb?)?(-?\d+)$', note_name)
        if m:
            letter = m.group(1)
            acc = m.group(2) or ''
            octave = int(m.group(3))
        else:
            octave = midi // 12 - 1
            pc = midi % 12
            letter = _NOTE_NAMES_SHARP[pc].rstrip('#')
            acc = '#' if '#' in _NOTE_NAMES_SHARP[pc] else ''
    else:
        octave = midi // 12 - 1
        pc = midi % 12
        name = _NOTE_NAMES_SHARP[pc]
        letter = name[0]
        acc = name[1:] if len(name) > 1 else ''

    # Accidental in ABC notation
    abc_acc = acc.replace('#', '^').replace('b', '_')

    # Octave in ABC notation
    # octave 3 → uppercase, no marks
    # octave 4 → lowercase, no marks
    # octave 5 → lowercase + '
    # octave 2 → uppercase + ,
    if octave == 3:
        abc_letter = letter.upper()
        oct_marks = ''
    elif octave == 4:
        abc_letter = letter.lower()
        oct_marks = ''
    elif octave > 4:
        abc_letter = letter.lower()
        oct_marks = "'" * (octave - 4)
    else:
        abc_letter = letter.upper()
        oct_marks = ',' * (3 - octave)

    return f"{abc_acc}{abc_letter}{oct_marks}"


def _duration_to_abc(duration_beats: float, unit_beats: float = 1.0) -> str:
    """Convert a duration in beats to ABC duration multiplier string.

    With L:1/4 (unit_beats=1.0), a quarter note has no multiplier,
    half note = '2', eighth note = '/2', dotted quarter = '3/2', etc.
    """
    frac = Fraction(duration_beats).limit_denominator(64) / Fraction(unit_beats).limit_denominator(64)
    # Simplify
    num = frac.numerator
    den = frac.denominator
    if den == 1:
        return '' if num == 1 else str(num)
    if num == 1:
        return f'/{den}'
    return f'{num}/{den}'


def to_abc(sequence: dict) -> str:
    """Serialize a sequence dict back to ABC notation.

    Uses L:1/4 (unit note = quarter), one bar per |, 4 bars per line.
    Produces deterministic output suitable for round-trip testing.
    """
    title = sequence.get('title', 'Untitled')
    tempo = sequence.get('tempo_bpm', 96.0)
    ts = sequence.get('time_signature', '4/4')
    key = sequence.get('key', 'C')

    ts_num, ts_den = map(int, ts.split('/'))
    beats_per_bar = ts_num * 4 / ts_den

    # Headers (L:1/4 so unit = 1 quarter beat)
    lines = [
        f"X:1",
        f"T:{title}",
        f"M:{ts}",
        f"L:1/4",
        f"Q:{int(round(tempo))}",
        f"K:{key}",
    ]

    events = sorted(sequence.get('events', []), key=lambda e: e['at_beat'])
    if not events:
        lines.append('')
        return '\n'.join(lines)

    total_beats = sequence.get('total_beats', 0.0)
    n_bars = int(total_beats / beats_per_bar) + (1 if total_beats % beats_per_bar > 1e-6 else 0)

    # Build a timeline: for each fractional beat, what's playing?
    # Group events by bar
    bar_lines: list[str] = []

    # Process bar by bar
    beat = 0.0
    evt_idx = 0
    all_events = list(events)

    for bar_i in range(max(n_bars, 1)):
        bar_start = bar_i * beats_per_bar
        bar_end = bar_start + beats_per_bar
        bar_evts = []
        for e in all_events:
            if e['at_beat'] >= bar_start - 1e-6 and e['at_beat'] < bar_end - 1e-6:
                bar_evts.append(e)

        bar_tokens = []
        prev_end = bar_start
        for e in bar_evts:
            # Gap = rest
            gap = e['at_beat'] - prev_end
            if gap > 1e-6:
                bar_tokens.append(f"z{_duration_to_abc(gap)}")
            # Event notes
            if len(e['notes']) == 1:
                note_names = e.get('note_names') or []
                nn = note_names[0] if note_names else None
                abc_note = _midi_to_abc_note(e['notes'][0], nn)
                dur = _duration_to_abc(e['duration_beats'])
                bar_tokens.append(f"{abc_note}{dur}")
            elif len(e['notes']) > 1:
                note_names = e.get('note_names') or []
                chord_parts = []
                for i, midi in enumerate(e['notes']):
                    nn = note_names[i] if i < len(note_names) else None
                    chord_parts.append(_midi_to_abc_note(midi, nn))
                dur = _duration_to_abc(e['duration_beats'])
                bar_tokens.append(f"[{''.join(chord_parts)}]{dur}")
            prev_end = e['at_beat'] + e['duration_beats']

        # Trailing rest to fill bar
        trailing = bar_end - prev_end
        if trailing > 1e-6:
            bar_tokens.append(f"z{_duration_to_abc(trailing)}")

        bar_lines.append(' '.join(bar_tokens) if bar_tokens else 'z')

    # Format: 4 bars per line, separated by |, final |
    body_lines = []
    for i in range(0, len(bar_lines), 4):
        chunk = bar_lines[i:i + 4]
        body_lines.append(' | '.join(chunk) + ' |')

    lines.extend(body_lines)
    return '\n'.join(lines)


# ── Per-bar validation ────────────────────────────────────────────────────────

def per_bar_report(sequence: dict) -> list[str]:
    """Return per-bar beat accounting messages for a sequence dict.

    Returns list of error strings (empty = all bars correct).
    The sequence must have time_signature_parts and events with at_beat/duration_beats.
    """
    ts_num, ts_den = sequence['time_signature_parts']
    beats_per_bar = ts_num * 4 / ts_den

    events = sorted(sequence.get('events', []), key=lambda e: e['at_beat'])
    if not events:
        return []

    total_beats = sequence.get('total_beats', 0.0)
    n_bars = int(total_beats / beats_per_bar) + (1 if total_beats % beats_per_bar > 1e-6 else 0)

    messages = []
    for bar_i in range(n_bars):
        bar_start = bar_i * beats_per_bar
        bar_end = bar_start + beats_per_bar
        bar_evts = [
            e for e in events
            if e['at_beat'] >= bar_start - 1e-6 and e['at_beat'] < bar_end - 1e-6
        ]
        # Count occupied beats in bar
        occupied = sum(
            min(e['at_beat'] + e['duration_beats'], bar_end) - e['at_beat']
            for e in bar_evts
        )
        if bar_evts and abs(occupied - beats_per_bar) > 0.01:
            messages.append(
                f"bar {bar_i + 1} contains {occupied:.4g} beats; "
                f"meter {sequence['time_signature']} expects {beats_per_bar:.4g}"
            )
    return messages
