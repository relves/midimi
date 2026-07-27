"""ABC notation parser/serializer (Phase 1 subset, Phase 5 multi-voice).

Supported subset:
  Headers: X: T: M: L: Q: K:
  Voice declarations: V: (in body, before first [V:id] line)
  Notes: A-G/a-g with octave marks (' ,), accidentals (^ _ = ^^ __),
         duration multipliers (N, /N, N/N)
  Chords: [CEG] with duration multiplier
  Chord symbols: "..." (annotation — ignored for playback)
  Rests: z/Z
  Barlines: | || [| |] (treated uniformly)
  Repeats: |: :| (expanded on parse — section played twice)
  Ties: - (extends note duration into the next note of same pitch)
  Inline fields: [Q:...] → skip for now; [V:...] mid-line → error

Rejects tokens outside this subset with a bar/token-precise error message.

Octave convention (standard ABC):
  Uppercase letter = base octave 4 (C = C4 = MIDI 60 = middle C)
  Lowercase letter = base octave 5 (c = C5 = MIDI 72)
  ' raises by one octave, , lowers by one octave

Multi-voice (Phase 5):
  V: declaration lines appear after K:, before first [V:id] body line.
  Body lines begin with [V:<id>] and contain whole bars for that voice.
  A tune with no V: lines parses exactly as before (single-voice backward compat).
"""

import re
import time
from fractions import Fraction

# ── Pitch helpers ─────────────────────────────────────────────────────────────

_LETTER_PC: dict[str, int] = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
}

# ABC octave base (standard): uppercase = octave 4 (middle C), lowercase = octave 5
_BASE_OCTAVE = {'upper': 4, 'lower': 5}

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
    if k in _KEY_SIGS:
        return _KEY_SIGS[k]
    normalized = k.replace(' ', '')
    if normalized in _KEY_SIGS:
        return _KEY_SIGS[normalized]
    for suffix in ('dor', 'phr', 'lyd', 'mix', 'aeo', 'loc', 'exp', 'ion'):
        if normalized.endswith(suffix):
            root = normalized[:-len(suffix)]
            if root in _KEY_SIGS:
                return _KEY_SIGS[root]
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
        try:
            if '/' in note_part:
                n, d = note_part.strip().split('/')
                beat_fraction = Fraction(int(n), int(d))
            else:
                beat_fraction = Fraction(int(note_part.strip()))
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
    num = int(num_s) if num_s else 1
    if slash is None:
        den = 1
    elif den_s:
        den = int(den_s)
    else:
        den = 2  # bare '/' means halve
    duration_units = Fraction(num, den)
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

    octave = _BASE_OCTAVE['lower'] if is_lower else _BASE_OCTAVE['upper']
    for ch in octave_str:
        if ch == "'":
            octave += 1
        elif ch == ',':
            octave -= 1

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
        bar_accidentals[upper] = acc_offset
    else:
        if upper in bar_accidentals:
            acc_offset = bar_accidentals[upper]
        else:
            acc_offset = key_sig.get(upper, 0)

    pc = _LETTER_PC[upper]
    midi = 12 * (octave + 1) + pc + acc_offset

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
           'first_ending', 'second_ending', 'chord_symbol', 'tie',
           'inline_field', 'unknown'
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

        # Inline field [X:...] — must check before chord branch
        # A chord is [note-letters]; an inline field is [letter:...]
        if ch == '[' and pos + 2 < len(s) and s[pos+1].isalpha() and s[pos+2] == ':':
            end = s.find(']', pos + 1)
            if end != -1:
                tokens.append(('inline_field', s[pos:end+1]))
                pos = end + 1
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

        # Decorations !...! — captured; unknown names are an error at parse time
        if ch == '!':
            end = s.find('!', pos + 1)
            if end != -1:
                tokens.append(('decoration', s[pos:end+1]))
                pos = end + 1
            else:
                pos += 1
            continue

        # +...+ decorations (old style) — skip silently
        if ch == '+':
            end = s.find('+', pos + 1)
            pos = end + 1 if end != -1 else len(s)
            continue

        # Single-char shorthand decorations: H=fermata, L=accent, .=staccato
        # M O P S T u v ~ = other ornaments, skip
        if ch in '.HLMOPSTuv~':
            tokens.append(('shorthand_decoration', ch))
            pos += 1
            continue

        tokens.append(('unknown', ch))
        pos += 1

    return tokens


# ── Repeat expansion ──────────────────────────────────────────────────────────

def _expand_repeats(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Expand |: ... :| repeats by duplicating the section."""
    result = []
    i = 0
    while i < len(tokens):
        tok_type, tok_raw = tokens[i]
        if tok_type == 'start_repeat':
            result.append(tokens[i])
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
                        if first_ending_tokens or second_ending_tokens:
                            result.extend(section_tokens)
                            result.extend(first_ending_tokens)
                            result.append(('barline', '|'))
                            result.extend(section_tokens)
                            result.extend(second_ending_tokens)
                        else:
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


# ── Body parser (single-voice, extracted for reuse) ───────────────────────────

def _parse_body_to_events(
    body: str,
    unit_length: Fraction,
    key_sig: dict[str, int],
    beats_per_bar: Fraction,
    time_signature: str,
) -> tuple[list[dict], list[str], float]:
    """Parse a single-voice ABC body string → (events, errors, total_beats).

    total_beats includes trailing rests (elapsed beats, not just last event end).

    This is the core event-builder, extracted so multi-voice parse can call it
    per-voice.  All existing logic is preserved exactly.
    """
    tokens = _tokenize_body(body)
    tokens = _expand_repeats(tokens)

    events: list[dict] = []
    errors: list[str] = []

    bar_num = 1
    beat_in_bar = Fraction(0)
    at_beat = Fraction(0)
    bar_accidentals: dict[str, int] = {}
    tie_pending: dict[int, int] = {}
    tie_active: bool = False
    pending_decs: list[str] = []  # decorations accumulated before next note
    running_velocity: int = 90

    def _reset_bar():
        nonlocal bar_num, beat_in_bar, bar_accidentals
        bar_num += 1
        beat_in_bar = Fraction(0)
        bar_accidentals = {}

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

        if tok_type == 'inline_field':
            # [Q:...] → tempo-change event; [V:...] mid-line → error
            if tok_raw.startswith('[V:') or tok_raw.startswith('[v:'):
                raise ABCParseError(
                    f"Bar {bar_num}: inline [V:] mid-line voice switching is not supported; "
                    "use stacked [V:id] lines at the start of each line instead."
                )
            if tok_raw.upper().startswith('[Q:'):
                q_val = tok_raw[3:-1].strip()
                try:
                    new_tempo = _parse_tempo(q_val)
                    events.append({
                        'at_beat': float(at_beat),
                        'duration_beats': 0.0,
                        'notes': [],
                        'note_names': [],
                        'root': '',
                        'quality': 'tempo_change',
                        'octave': 0,
                        'velocity': 0,
                        'label': f'Q:{q_val}',
                        'tempo_bpm': new_tempo,
                    })
                except ABCParseError:
                    pass
            continue

        if tok_type in ('decoration', 'shorthand_decoration'):
            # Accumulate pending decorations; applied to next note/chord
            pending_decs.append(tok_raw)
            continue

        if tok_type == 'tie':
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
            evt = {
                'at_beat': float(at_beat),
                'duration_beats': float(dur),
                'notes': [midi],
                'note_names': [note_name],
                'root': note_name[:-1] if note_name[-1].isdigit() else note_name,
                'quality': 'note',
                'octave': octave,
                'velocity': running_velocity,
                'label': note_name,
            }
            if pending_decs:
                running_velocity, _ = _apply_decorations(evt, pending_decs, running_velocity, bar_num, errors)
                pending_decs = []
            events.append(evt)
            at_beat += dur
            beat_in_bar += dur
            continue

        if tok_type == 'chord':
            inner_m = re.match(r'\[([^\]]+)\]((\d+)?(/)?(\d+)?)', tok_raw)
            if not inner_m:
                inner_m = re.match(r'\[([^\]]+)\]', tok_raw)
                dur_raw = ''
            else:
                dur_raw = inner_m.group(2)
            if not inner_m:
                raise ABCParseError(f"Bar {bar_num}: cannot parse chord token {tok_raw!r}")

            inner = inner_m.group(1)
            if dur_raw:
                dur_m = re.match(r'(\d+)?(/)?(\d+)?', dur_raw)
                dur = _parse_duration(
                    dur_m.group(1) if dur_m else None,
                    dur_m.group(2) if dur_m else None,
                    dur_m.group(3) if dur_m else None,
                    unit_length,
                )
            else:
                dur = unit_length * 4

            chord_midis = []
            chord_names = []
            chord_bar_acc = dict(bar_accidentals)
            for nm in _NOTE_TOKEN_RE.finditer(inner):
                midi, note_name = _note_midi_and_name(
                    nm.group('letter'), nm.group('acc') or '', nm.group('octave') or '',
                    key_sig, chord_bar_acc,
                )
                chord_midis.append(midi)
                chord_names.append(note_name)
            if not chord_midis:
                raise ABCParseError(f"Bar {bar_num}: empty chord {tok_raw!r}")

            bar_accidentals.update(chord_bar_acc)

            if tie_active and all(m in tie_pending for m in chord_midis):
                prev_idxs = {tie_pending[m] for m in chord_midis}
                if len(prev_idxs) == 1:
                    prev_idx = prev_idxs.pop()
                    if prev_idx < len(events) and sorted(events[prev_idx]['notes']) == sorted(chord_midis):
                        events[prev_idx]['duration_beats'] += float(dur)
                        tie_pending = {m: prev_idx for m in chord_midis}
                        tie_active = False
                        at_beat += dur
                        beat_in_bar += dur
                        continue
            tie_active = False

            octave = chord_midis[0] // 12 - 1
            evt = {
                'at_beat': float(at_beat),
                'duration_beats': float(dur),
                'notes': chord_midis,
                'note_names': chord_names,
                'root': chord_names[0][:-1] if chord_names[0][-1].isdigit() else chord_names[0],
                'quality': 'note',
                'octave': octave,
                'velocity': running_velocity,
                'label': '+'.join(chord_names),
            }
            if pending_decs:
                running_velocity, _ = _apply_decorations(evt, pending_decs, running_velocity, bar_num, errors)
                pending_decs = []
            evt_idx = len(events)
            events.append(evt)
            at_beat += dur
            beat_in_bar += dur
            tie_pending = {m: evt_idx for m in chord_midis}
            continue

    # Final bar check (if no trailing barline)
    if beat_in_bar > 0 and beat_in_bar != beats_per_bar:
        if beat_in_bar > beats_per_bar + Fraction(1, 64):
            errors.append(
                f"bar {bar_num} contains {float(beat_in_bar):.4g} beats; "
                f"meter {time_signature} expects {float(beats_per_bar):.4g}"
            )

    for evt in events:
        evt.pop('_tied', None)

    return events, errors, float(at_beat)


# ── Multi-voice helpers ───────────────────────────────────────────────────────

_V_DECL_RE = re.compile(r'^V:(\S+)(.*)')
_V_BODY_LINE_RE = re.compile(r'^\[V:([^\]]+)\](.*)')


def _parse_voice_declarations(body_lines: list[str]) -> list[dict]:
    """Parse V:id [name="..." octave=N] declaration lines from body region.

    Stops at the first [V:id] body line.  Only name= and octave= are accepted;
    other attributes are an error.
    """
    voices: list[dict] = []
    seen_ids: set[str] = set()
    for line in body_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        if stripped.startswith('[V:'):
            break  # into voice bodies — stop scanning declarations
        m = _V_DECL_RE.match(stripped)
        if not m:
            continue
        vid = m.group(1).strip()
        attrs_str = m.group(2).strip()
        if vid in seen_ids:
            continue
        seen_ids.add(vid)
        name = vid
        octave_shift = 0
        nm = re.search(r'name="([^"]*)"', attrs_str)
        if nm:
            name = nm.group(1)
        om = re.search(r'octave=(-?\d+)', attrs_str)
        if om:
            octave_shift = int(om.group(1))
        remainder = attrs_str
        remainder = re.sub(r'name="[^"]*"', '', remainder)
        remainder = re.sub(r'octave=-?\d+', '', remainder).strip()
        if remainder:
            raise ABCParseError(f"V:{vid}: unknown voice attributes: {remainder!r}")
        voices.append({'id': vid, 'name': name, 'octave_shift': octave_shift})
    return voices


def _split_voice_bodies(body_lines: list[str]) -> dict[str, list[str]]:
    """Split body lines into per-voice content based on [V:id] line prefixes.

    Lines before the first [V:id] marker (V: declarations, etc.) are skipped.
    Consecutive lines without a [V:id] prefix continue the current voice.
    """
    voice_bodies: dict[str, list[str]] = {}
    current_voice: str | None = None
    in_body = False

    for line in body_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        m = _V_BODY_LINE_RE.match(stripped)
        if m:
            in_body = True
            vid = m.group(1).strip()
            rest = m.group(2).strip()
            current_voice = vid
            if vid not in voice_bodies:
                voice_bodies[vid] = []
            if rest:
                voice_bodies[vid].append(rest)
        elif not in_body:
            # V: declarations or other pre-body content — skip
            continue
        elif current_voice is not None:
            voice_bodies[current_voice].append(stripped)

    return voice_bodies


def _assign_channels(voices: list[dict]) -> None:
    """Assign MIDI channels to voices in-place, skipping channel 9 (percussion)."""
    ch = 0
    for v in voices:
        if ch == 9:
            ch += 1
        v['channel'] = ch
        v.setdefault('program', 0)
        ch += 1


def _check_voice_interactions(events: list[dict], voices: list[dict]) -> list[str]:
    """Return informational messages about voice crossing and unisons."""
    if len(voices) < 2:
        return []
    warnings: list[str] = []
    voice_order = [v['id'] for v in voices]
    voice_name = {v['id']: v.get('name', v['id']) for v in voices}

    # Group simultaneous events by beat
    from collections import defaultdict
    beat_notes: dict[float, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for e in events:
        vid = e.get('voice', voice_order[0])
        beat_notes[e['at_beat']][vid].extend(e['notes'])

    seen_crossing: set[tuple[str, str]] = set()
    seen_unison: dict[tuple[str, str], set[int]] = {}

    for beat in sorted(beat_notes):
        vn = beat_notes[beat]
        for i, vid_hi in enumerate(voice_order):
            for vid_lo in voice_order[i+1:]:
                hi_notes = vn.get(vid_hi, [])
                lo_notes = vn.get(vid_lo, [])
                if not hi_notes or not lo_notes:
                    continue
                # Voice crossing: the lower-declared voice sounds above the higher-declared
                if min(lo_notes) > max(hi_notes):
                    key = (vid_hi, vid_lo)
                    if key not in seen_crossing:
                        seen_crossing.add(key)
                        warnings.append(
                            f"voice crossing at beat {beat:.4g}: "
                            f"{voice_name[vid_lo]} above {voice_name[vid_hi]}"
                        )
                # Unison: same pitch in both voices
                common = set(hi_notes) & set(lo_notes)
                if common:
                    key2 = (vid_hi, vid_lo)
                    already = seen_unison.get(key2, set())
                    new_common = common - already
                    if new_common:
                        seen_unison[key2] = already | new_common
                        from sequencer.theory import midi_note_name
                        note_strs = [midi_note_name(n) for n in sorted(new_common)]
                        warnings.append(
                            f"unison at beat {beat:.4g}: "
                            f"{voice_name[vid_hi]} and {voice_name[vid_lo]} share "
                            f"{', '.join(note_strs)}"
                        )

    return warnings


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_abc(text: str) -> dict:
    """Parse ABC notation text and return a normalized sequence dict.

    Single-voice ABC (no V: declarations) returns the same structure as before.
    Multi-voice ABC adds a 'voices' key and tags each event with 'voice'.

    Raises ABCParseError with bar-precise messages on invalid input.
    """
    lines = text.splitlines()
    headers, body_start = _parse_headers(lines)
    body_lines = lines[body_start:]

    # Parse headers
    title = headers.get('T', 'Untitled').strip()

    ts_num, ts_den = _parse_meter(headers['M']) if 'M' in headers else (4, 4)
    time_signature = f"{ts_num}/{ts_den}"
    beats_per_bar = Fraction(ts_num * 4, ts_den)

    unit_length = _parse_unit_length(headers['L']) if 'L' in headers else _default_unit_length(ts_num, ts_den)

    tempo_bpm = _parse_tempo(headers['Q']) if 'Q' in headers else 96.0

    key_sig = _parse_key(headers.get('K', 'C'))
    key_name = headers.get('K', 'C').strip()

    # Detect multi-voice: any body line starts with [V:
    is_multivoice = any(re.match(r'^\s*\[V:', l) for l in body_lines)

    if not is_multivoice:
        # ── Single-voice path (unchanged) ────────────────────────────────────
        body = '\n'.join(body_lines)
        events, errors, _ = _parse_body_to_events(body, unit_length, key_sig, beats_per_bar, time_signature)

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
            'abc_errors': [],
        }

    # ── Multi-voice path ──────────────────────────────────────────────────────
    voice_decls = _parse_voice_declarations(body_lines)
    voice_bodies = _split_voice_bodies(body_lines)

    # If no explicit V: declarations, infer from body markers
    if not voice_decls:
        for vid in voice_bodies:
            voice_decls.append({'id': vid, 'name': vid, 'octave_shift': 0})

    _assign_channels(voice_decls)

    all_events: list[dict] = []
    all_errors: list[str] = []
    voice_beat_totals: dict[str, float] = {}

    for v in voice_decls:
        vid = v['id']
        vname = v.get('name', vid)
        vbody = '\n'.join(voice_bodies.get(vid, []))
        evts, errs, vbeats = _parse_body_to_events(vbody, unit_length, key_sig, beats_per_bar, time_signature)

        # Apply octave shift (from octave= attribute)
        if v.get('octave_shift', 0):
            shift = v['octave_shift'] * 12
            for e in evts:
                e['notes'] = [n + shift for n in e['notes']]

        for e in evts:
            e['voice'] = vid
        all_events.extend(evts)
        all_errors.extend([f"voice {vname}, {err}" for err in errs])
        # Elapsed beats including trailing rests, so rest-padded voices count fully
        voice_beat_totals[vid] = vbeats

    # Voice bar count mismatch → error
    if len(voice_beat_totals) > 1:
        bpb = float(beats_per_bar)
        voice_bars = {
            vid: round(beats / bpb) for vid, beats in voice_beat_totals.items()
        }
        # Use voice names in error message
        vid_to_name = {v['id']: v.get('name', v['id']) for v in voice_decls}
        if len(set(voice_bars.values())) > 1:
            parts = [f"voice {vid_to_name[vid]} has {n} bars" for vid, n in voice_bars.items()]
            all_errors.append('; '.join(parts))

    if all_errors:
        raise ABCParseError('\n'.join(all_errors))

    # Informational: voice crossing, unisons (non-fatal)
    warnings = _check_voice_interactions(all_events, voice_decls)

    if not all_events:
        raise ABCParseError("No notes found in ABC")

    total_beats = max(e['at_beat'] + e['duration_beats'] for e in all_events)
    duration_ms = int(total_beats * 60 / tempo_bpm * 1000)

    return {
        'title': title,
        'tempo_bpm': float(tempo_bpm),
        'time_signature': time_signature,
        'time_signature_parts': (ts_num, ts_den),
        'events': sorted(all_events, key=lambda e: (e['at_beat'], e.get('voice', '1'))),
        'duration_ms': duration_ms,
        'total_beats': total_beats,
        'key': key_name,
        'voices': voice_decls,
        'abc_errors': warnings,
    }


# ── Decoration / articulation helpers ────────────────────────────────────────

# Known !decoration! names and shorthand mappings
_KNOWN_DECORATIONS: dict[str, str] = {
    '!fermata!': 'fermata', '!fermata2!': 'fermata',
    '!trill!': 'skip',     '!mordent!': 'skip',   '!turn!': 'skip',
    '!tenuto!': 'tenuto',
    '!accent!': 'accent',   '!emphasis!': 'accent',
    '!staccato!': 'staccato',
    '!p!': 'dynamic_p', '!pp!': 'dynamic_pp', '!ppp!': 'dynamic_ppp',
    '!mp!': 'dynamic_mp', '!mf!': 'dynamic_mf',
    '!f!': 'dynamic_f', '!ff!': 'dynamic_ff', '!fff!': 'dynamic_fff',
    '!sfz!': 'dynamic_sfz', '!sf!': 'dynamic_sfz',
    '!crescendo(!': 'skip', '!crescendo)!': 'skip',
    '!diminuendo(!': 'skip', '!diminuendo)!': 'skip',
    '!<(!': 'skip', '!<)!': 'skip', '!>(!': 'skip', '!>)!': 'skip',
    '!D.C.!': 'skip', '!D.S.!': 'skip', '!segno!': 'skip', '!coda!': 'skip',
    '!fine!': 'skip', '!>!': 'skip', '!<<!': 'skip', '!>>!': 'skip',
    '!8va!': 'skip', '!8vb!': 'skip',
    '!trem1!': 'skip', '!trem2!': 'skip', '!trem3!': 'skip', '!trem4!': 'skip',
}

_SHORTHAND_MAP: dict[str, str] = {
    'H': 'fermata',   # fermata shorthand
    'L': 'accent',    # accent shorthand
    '.': 'staccato',  # staccato shorthand
}

_DYNAMIC_VELOCITY: dict[str, int] = {
    'dynamic_ppp': 32, 'dynamic_pp': 40, 'dynamic_p': 48,
    'dynamic_mp': 64, 'dynamic_mf': 80,
    'dynamic_f': 96, 'dynamic_ff': 108, 'dynamic_fff': 120,
    'dynamic_sfz': 112,
}

FERMATA_FACTOR_DEFAULT = 1.8


def _apply_decorations(
    event: dict,
    decorations: list[str],
    running_velocity: int,
    bar_num: int,
    errors: list[str],
) -> tuple[int, list[str]]:
    """Apply decoration list to event in-place.  Returns (new running_velocity, remaining_unknown).

    Unknown !...! decorations are added to errors list.
    """
    for dec in decorations:
        # Shorthand (single char)
        if len(dec) == 1:
            kind = _SHORTHAND_MAP.get(dec)
        else:
            kind = _KNOWN_DECORATIONS.get(dec.lower())
            if kind is None:
                # Unknown !...! → error per plan
                errors.append(f"bar {bar_num}: unknown decoration {dec!r}")
                continue

        if kind == 'skip' or kind is None:
            continue
        elif kind == 'fermata':
            event['fermata'] = True
        elif kind == 'staccato':
            event['staccato'] = True
        elif kind == 'tenuto':
            event['tenuto'] = True
        elif kind == 'accent':
            event['accent'] = True
        elif kind.startswith('dynamic_'):
            vel = _DYNAMIC_VELOCITY.get(kind, running_velocity)
            event['dynamic'] = kind
            event['velocity'] = vel
            running_velocity = vel

    return running_velocity, []


def _check_bar(bar_num: int, beat_in_bar: Fraction, beats_per_bar: Fraction, errors: list[str], ts: str = "") -> None:
    diff = abs(beat_in_bar - beats_per_bar)
    if diff > Fraction(1, 64):
        meter_str = f"meter {ts} expects" if ts else "meter expects"
        errors.append(
            f"bar {bar_num} contains {float(beat_in_bar):.4g} beats; "
            f"{meter_str} {float(beats_per_bar):.4g}"
        )


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

    abc_acc = acc.replace('#', '^').replace('b', '_')

    if octave == 4:
        abc_letter = letter.upper()
        oct_marks = ''
    elif octave == 5:
        abc_letter = letter.lower()
        oct_marks = ''
    elif octave > 5:
        abc_letter = letter.lower()
        oct_marks = "'" * (octave - 5)
    else:
        abc_letter = letter.upper()
        oct_marks = ',' * (4 - octave)

    return f"{abc_acc}{abc_letter}{oct_marks}"


def _duration_to_abc(duration_beats: float, unit_beats: float = 1.0) -> str:
    """Convert a duration in beats to ABC duration multiplier string.

    With L:1/4 (unit_beats=1.0), a quarter note has no multiplier,
    half note = '2', eighth note = '/2', dotted quarter = '3/2', etc.
    """
    frac = Fraction(duration_beats).limit_denominator(64) / Fraction(unit_beats).limit_denominator(64)
    num = frac.numerator
    den = frac.denominator
    if den == 1:
        return '' if num == 1 else str(num)
    if num == 1:
        return f'/{den}'
    return f'{num}/{den}'


def _event_prefix(e: dict) -> str:
    """Build the decoration prefix string for an event (for serialization)."""
    parts = []
    if e.get('dynamic'):
        # dynamic_mp → !mp!
        dyn = e['dynamic'].replace('dynamic_', '!')
        parts.append(dyn + '!')
    if e.get('fermata'):
        parts.append('!fermata!')
    if e.get('staccato'):
        parts.append('.')
    if e.get('tenuto'):
        parts.append('!tenuto!')
    if e.get('accent'):
        parts.append('!accent!')
    return ''.join(parts)


def _split_events_at_bars(events: list[dict], beats_per_bar: float) -> list[dict]:
    """Split events that cross barlines into per-bar segments joined by ties.

    Each segment except the last carries '_tie': True so the renderer can emit
    the '-' tie token, preserving the full written duration across barlines.
    """
    out: list[dict] = []
    for e in events:
        if e.get('quality') == 'tempo_change' or not e.get('notes'):
            out.append(e)
            continue
        start = e['at_beat']
        end = start + e['duration_beats']
        bar_end = (int(start / beats_per_bar + 1e-9) + 1) * beats_per_bar
        if end <= bar_end + 1e-6:
            out.append(e)
            continue
        cur = start
        while cur < end - 1e-6:
            seg_end = min(end, bar_end)
            seg = dict(e)
            seg['at_beat'] = cur
            seg['duration_beats'] = seg_end - cur
            if seg_end < end - 1e-6:
                seg['_tie'] = True
            out.append(seg)
            cur = seg_end
            bar_end += beats_per_bar
    return out


def _render_bar_events(bar_evts: list[dict], bar_start: float, bar_end: float) -> str:
    """Render a list of events within a bar to an ABC token string."""
    bar_tokens: list[str] = []
    prev_end = bar_start
    for e in bar_evts:
        if e.get('quality') == 'tempo_change':
            # Inline [Q:...] tempo change event
            bpm = e.get('tempo_bpm', 120)
            bar_tokens.append(f"[Q:{int(round(bpm))}]")
            continue
        gap = e['at_beat'] - prev_end
        if gap > 1e-6:
            bar_tokens.append(f"z{_duration_to_abc(gap)}")
        prefix = _event_prefix(e)
        if not e.get('notes'):
            prev_end = e['at_beat'] + e.get('duration_beats', 0)
            continue
        # Clamp duration to bar boundary to prevent overfull bars
        capped_dur = min(e['duration_beats'], bar_end - e['at_beat'])
        if capped_dur <= 0:
            continue
        tie = '-' if e.get('_tie') else ''
        if len(e['notes']) == 1:
            note_names = e.get('note_names') or []
            nn = note_names[0] if note_names else None
            abc_note = _midi_to_abc_note(e['notes'][0], nn)
            dur = _duration_to_abc(capped_dur)
            bar_tokens.append(f"{prefix}{abc_note}{dur}{tie}")
        elif len(e['notes']) > 1:
            note_names = e.get('note_names') or []
            chord_parts = []
            for i, midi in enumerate(e['notes']):
                nn = note_names[i] if i < len(note_names) else None
                chord_parts.append(_midi_to_abc_note(midi, nn))
            dur = _duration_to_abc(capped_dur)
            bar_tokens.append(f"{prefix}[{''.join(chord_parts)}]{dur}{tie}")
        prev_end = e['at_beat'] + e['duration_beats']

    trailing = bar_end - prev_end
    if trailing > 1e-6:
        bar_tokens.append(f"z{_duration_to_abc(trailing)}")

    return ' '.join(bar_tokens) if bar_tokens else 'z'


def to_abc(sequence: dict) -> str:
    """Serialize a sequence dict back to ABC notation.

    Single-voice: uses L:1/4, one bar per |, 4 bars per line (unchanged).
    Multi-voice: emits V: declarations + interleaved [V:id] 4-bar systems.
    """
    voices = sequence.get('voices')
    if voices and len(voices) > 1:
        return _to_abc_multi(sequence)
    return _to_abc_single(sequence)


def _to_abc_single(sequence: dict) -> str:
    """Single-voice serializer — unchanged from Phase 1."""
    title = sequence.get('title', 'Untitled')
    tempo = sequence.get('tempo_bpm', 96.0)
    ts = sequence.get('time_signature', '4/4')
    key = sequence.get('key', 'C')

    ts_num, ts_den = map(int, ts.split('/'))
    beats_per_bar = ts_num * 4 / ts_den

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
    events = _split_events_at_bars(events, beats_per_bar)

    total_beats = sequence.get('total_beats', 0.0)
    n_bars = int(total_beats / beats_per_bar) + (1 if total_beats % beats_per_bar > 1e-6 else 0)

    bar_lines: list[str] = []
    all_events = list(events)

    for bar_i in range(max(n_bars, 1)):
        bar_start = bar_i * beats_per_bar
        bar_end = bar_start + beats_per_bar
        bar_evts = [
            e for e in all_events
            if e['at_beat'] >= bar_start - 1e-6 and e['at_beat'] < bar_end - 1e-6
        ]
        bar_lines.append(_render_bar_events(bar_evts, bar_start, bar_end))

    body_lines = []
    for i in range(0, len(bar_lines), 4):
        chunk = bar_lines[i:i + 4]
        body_lines.append(' | '.join(chunk) + ' |')

    lines.extend(body_lines)
    return '\n'.join(lines)


def _to_abc_multi(sequence: dict) -> str:
    """Multi-voice serializer: V: declarations + interleaved [V:id] 4-bar systems."""
    title = sequence.get('title', 'Untitled')
    tempo = sequence.get('tempo_bpm', 96.0)
    ts = sequence.get('time_signature', '4/4')
    key = sequence.get('key', 'C')
    voices = sequence['voices']

    ts_num, ts_den = map(int, ts.split('/'))
    beats_per_bar = ts_num * 4 / ts_den

    lines = [
        "X:1",
        f"T:{title}",
        f"M:{ts}",
        f"L:1/4",
        f"Q:{int(round(tempo))}",
        f"K:{key}",
    ]

    # Voice declaration lines
    for v in voices:
        vline = f"V:{v['id']}"
        if v.get('name') and v['name'] != v['id']:
            vline += f' name="{v["name"]}"'
        if v.get('octave_shift', 0):
            vline += f' octave={v["octave_shift"]}'
        lines.append(vline)

    # Group events by voice
    events_by_voice: dict[str, list[dict]] = {v['id']: [] for v in voices}
    for e in sequence.get('events', []):
        vid = e.get('voice', voices[0]['id'])
        if vid in events_by_voice:
            events_by_voice[vid].append(e)
    for vid in events_by_voice:
        events_by_voice[vid].sort(key=lambda e: e['at_beat'])
        events_by_voice[vid] = _split_events_at_bars(events_by_voice[vid], beats_per_bar)

    total_beats = sequence.get('total_beats', 0.0)
    n_bars = int(total_beats / beats_per_bar) + (1 if total_beats % beats_per_bar > 1e-6 else 0)

    # Build bar-strings per voice
    voice_bar_lines: dict[str, list[str]] = {}
    for v in voices:
        vid = v['id']
        vevents = events_by_voice[vid]
        vbars: list[str] = []
        for bar_i in range(max(n_bars, 1)):
            bar_start = bar_i * beats_per_bar
            bar_end = bar_start + beats_per_bar
            bar_evts = [
                e for e in vevents
                if e['at_beat'] >= bar_start - 1e-6 and e['at_beat'] < bar_end - 1e-6
            ]
            vbars.append(_render_bar_events(bar_evts, bar_start, bar_end))
        voice_bar_lines[vid] = vbars

    # Interleave voices in 4-bar systems
    for chunk_start in range(0, n_bars, 4):
        chunk_end = min(chunk_start + 4, n_bars)
        for v in voices:
            vid = v['id']
            chunk = voice_bar_lines[vid][chunk_start:chunk_end]
            body = ' | '.join(chunk) + ' |'
            lines.append(f"[V:{vid}] {body}")

    return '\n'.join(lines)


# ── Per-bar validation ────────────────────────────────────────────────────────

def per_bar_report(sequence: dict) -> list[str]:
    """Return per-bar beat accounting messages for a sequence dict.

    Multi-voice sequences prefix each message with 'voice <name>, '.
    Returns list of error strings (empty = all bars correct).
    """
    voices = sequence.get('voices')
    if not voices:
        return _per_bar_report_single(sequence, sequence.get('events', []))

    messages: list[str] = []
    for v in voices:
        vid = v['id']
        vname = v.get('name', vid)
        vevents = [e for e in sequence.get('events', []) if e.get('voice') == vid]
        voice_msgs = _per_bar_report_single(sequence, vevents)
        messages.extend([f"voice {vname}, {m}" for m in voice_msgs])
    return messages


def chord_report(sequence: dict) -> list[str]:
    """Name every written chord (3+ simultaneous notes) so the agent can verify it.

    Hand-written ABC accidentals are the easiest place to get a chord wrong —
    e.g. `[A,^CE^G]` in K:A is Amaj7, not the A7 the author meant. Echoing the
    identified chord name back makes that mistake visible before it is played.
    """
    from sequencer.theory import identify_chord

    ts_num, ts_den = sequence['time_signature_parts']
    beats_per_bar = ts_num * 4 / ts_den

    lines: list[str] = []
    voices = {v['id']: v.get('name', v['id']) for v in (sequence.get('voices') or [])}

    for e in sorted(sequence.get('events', []), key=lambda x: (x['at_beat'], x.get('voice') or '')):
        names = e.get('note_names') or []
        if len(names) < 3:
            continue
        name = identify_chord(names)
        if not name:
            continue
        bar = int(e['at_beat'] / beats_per_bar) + 1
        beat = e['at_beat'] % beats_per_bar + 1
        prefix = f"voice {voices[e['voice']]}, " if e.get('voice') in voices else ""
        lines.append(f"  {prefix}bar {bar} beat {beat:.4g}: {' '.join(names)} = {name}")
    return lines


def _per_bar_report_single(sequence: dict, events: list[dict]) -> list[str]:
    """Per-bar beat accounting for a single set of events."""
    ts_num, ts_den = sequence['time_signature_parts']
    beats_per_bar = ts_num * 4 / ts_den

    sorted_events = sorted(events, key=lambda e: e['at_beat'])
    if not sorted_events:
        return []

    total_beats = max(e['at_beat'] + e['duration_beats'] for e in sorted_events)
    n_bars = int(total_beats / beats_per_bar) + (1 if total_beats % beats_per_bar > 1e-6 else 0)

    messages = []
    for bar_i in range(n_bars):
        bar_start = bar_i * beats_per_bar
        bar_end = bar_start + beats_per_bar
        bar_evts = [
            e for e in sorted_events
            if e['at_beat'] >= bar_start - 1e-6 and e['at_beat'] < bar_end - 1e-6
        ]
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
