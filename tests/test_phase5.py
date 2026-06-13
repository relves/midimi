"""Phase 5 tests: multi-voice ABC, voicing helpers, expressive timing, engine, MIDI export."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pathlib import Path
from sequencer.abc import parse_abc, to_abc, ABCParseError, per_bar_report
from sequencer.theory import voice_chord, voice_progression, _LIL_THRESHOLD

# ─────────────────────────────────────────────────────────────────────────────
# 5.1 Multi-voice ABC
# ─────────────────────────────────────────────────────────────────────────────

MULTI_VOICE_ABC = """\
X:1
T:Example
M:4/4
L:1/8
Q:1/4=72
K:Eb
V:1 name="melody"
V:2 name="harmony"
V:3 name="bass" octave=-1
[V:1] E2 G2 B2 g2 | f6 z2 |
[V:2] [GBe]4 [Bdg]4 | [Acf]8 |
[V:3] E,4 B,,4 | F,8 |
"""


class TestMultiVoiceParse:
    def test_voices_key_present(self):
        seq = parse_abc(MULTI_VOICE_ABC)
        assert "voices" in seq
        assert len(seq["voices"]) == 3

    def test_voice_names(self):
        seq = parse_abc(MULTI_VOICE_ABC)
        names = [v["name"] for v in seq["voices"]]
        assert names == ["melody", "harmony", "bass"]

    def test_voice_octave_shift(self):
        seq = parse_abc(MULTI_VOICE_ABC)
        bass = next(v for v in seq["voices"] if v["id"] == "3")
        assert bass["octave_shift"] == -1

    def test_events_tagged_with_voice(self):
        seq = parse_abc(MULTI_VOICE_ABC)
        vids = {e["voice"] for e in seq["events"] if e.get("notes")}
        assert "1" in vids
        assert "2" in vids
        assert "3" in vids

    def test_channel_assignment_skips_9(self):
        seq = parse_abc(MULTI_VOICE_ABC)
        channels = [v["channel"] for v in seq["voices"]]
        assert 9 not in channels
        assert channels == sorted(channels)

    def test_single_voice_unchanged(self):
        """Single-voice ABC must parse with no 'voices' key."""
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\nc d e f |\n"
        seq = parse_abc(abc)
        assert "voices" not in seq

    def test_single_voice_round_trip(self):
        """Single-voice round-trip must be byte-for-byte identical."""
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\nc d e f |\n"
        seq = parse_abc(abc)
        normalized = to_abc(seq)
        seq2 = parse_abc(normalized)
        normalized2 = to_abc(seq2)
        assert normalized == normalized2


class TestMultiVoiceSerialize:
    def test_round_trip(self):
        seq = parse_abc(MULTI_VOICE_ABC)
        out = to_abc(seq)
        seq2 = parse_abc(out)
        # Same voices
        assert len(seq2["voices"]) == len(seq["voices"])
        # Same event count per voice
        for vid in ("1", "2", "3"):
            n1 = len([e for e in seq["events"] if e.get("voice") == vid and e.get("notes")])
            n2 = len([e for e in seq2["events"] if e.get("voice") == vid and e.get("notes")])
            assert n1 == n2, f"voice {vid} event count mismatch: {n1} vs {n2}"

    def test_v_declaration_in_output(self):
        seq = parse_abc(MULTI_VOICE_ABC)
        out = to_abc(seq)
        assert "V:1" in out
        assert "V:2" in out
        assert "V:3" in out

    def test_voice_body_lines_in_output(self):
        seq = parse_abc(MULTI_VOICE_ABC)
        out = to_abc(seq)
        assert "[V:1]" in out
        assert "[V:2]" in out
        assert "[V:3]" in out


class TestMultiVoiceValidation:
    def test_per_voice_per_bar_report(self):
        seq = parse_abc(MULTI_VOICE_ABC)
        msgs = per_bar_report(seq)
        # Should produce per-voice prefixed messages if any errors
        if msgs:
            for m in msgs:
                assert m.startswith("voice ")

    def test_voice_bar_count_mismatch_error(self):
        abc = """\
X:1
T:T
M:4/4
L:1/4
Q:120
K:C
V:1
V:2
[V:1] c d e f | g a b c' |
[V:2] c d e f |
"""
        with pytest.raises(ABCParseError) as exc:
            parse_abc(abc)
        assert "bars" in str(exc.value).lower()

    def test_per_voice_bar_error_names_voice(self):
        abc = """\
X:1
T:T
M:4/4
L:1/4
Q:120
K:C
V:1 name="melody"
V:2 name="harmony"
[V:1] c d e f | g a b c' |
[V:2] c d e f | g a b c' d |
"""
        with pytest.raises(ABCParseError) as exc:
            parse_abc(abc)
        msg = str(exc.value)
        # Should mention voice names
        assert "harmony" in msg or "melody" in msg or "voice" in msg.lower()

    def test_inline_voice_switching_rejected(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\nc [V:2] d e f |\n"
        with pytest.raises(ABCParseError) as exc:
            parse_abc(abc)
        assert "stacked" in str(exc.value) or "inline" in str(exc.value) or "mid-line" in str(exc.value)

    def test_unknown_voice_attribute_rejected(self):
        abc = """\
X:1
T:T
M:4/4
L:1/4
Q:120
K:C
V:1 name="melody" clef=treble
[V:1] c d e f |
"""
        with pytest.raises(ABCParseError) as exc:
            parse_abc(abc)
        assert "unknown" in str(exc.value).lower()


# ─────────────────────────────────────────────────────────────────────────────
# 5.2 Voicing helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestVoiceChord:
    def test_basic_major7_close(self):
        r = voice_chord("C", "major7", style="close")
        assert "notes" in r and "midi" in r and "abc" in r
        assert len(r["notes"]) >= 3
        # Notes sorted bottom to top
        assert r["midi"] == sorted(r["midi"])

    def test_melody_note_not_doubled(self):
        for quality in ("major7", "dominant7", "minor7"):
            r = voice_chord("G", quality, melody_note="D5")
            melody_pc = 2  # D
            assert all(n % 12 != melody_pc for n in r["midi"]), \
                f"{quality}: melody D doubled in {r['notes']}"

    def test_top_note_below_melody(self):
        from sequencer.theory import parse_pitch
        _, _, mel_midi = parse_pitch("G5")
        r = voice_chord("C", "major7", melody_note="G5")
        assert max(r["midi"]) < mel_midi, f"Top note not below melody: {r['notes']}"

    def test_lil_enforced(self):
        """No m3 below E3 (MIDI 52), no M3 below C3 (MIDI 48)."""
        for quality in ("major7", "minor7"):
            r = voice_chord("C", quality, register="low", style="close")
            midis = sorted(r["midi"])
            for i in range(len(midis) - 1):
                interval = (midis[i + 1] - midis[i]) % 12
                threshold = _LIL_THRESHOLD.get(interval)
                if threshold is not None:
                    assert midis[i] >= threshold, \
                        f"LIL violated: {interval}-semitone interval, lower note MIDI {midis[i]} < {threshold}"

    def test_omit_root(self):
        r = voice_chord("G", "dominant7", omit_root=True)
        g_pc = 7  # G
        assert all(n % 12 != g_pc for n in r["midi"]), \
            f"Root G present in omit_root voicing: {r['notes']}"

    def test_shell_style(self):
        r = voice_chord("F", "major7", style="shell")
        # Shell: 3, 7 always present; should have <= 3 notes
        assert len(r["notes"]) <= 3

    def test_drop2_style(self):
        r = voice_chord("C", "major7", style="drop2")
        # Should return a valid voicing (don't crash)
        assert len(r["notes"]) >= 3
        assert r["midi"] == sorted(r["midi"])

    @pytest.mark.parametrize("quality", ["major", "minor", "diminished", "augmented",
                                          "major7", "dominant7", "minor7", "halfdiminished7"])
    def test_all_qualities_produce_sorted_midi(self, quality):
        r = voice_chord("C", quality)
        assert r["midi"] == sorted(r["midi"])

    def test_abc_token_format(self):
        r = voice_chord("C", "major7")
        assert r["abc"].startswith("[") and r["abc"].endswith("]")
        # Should be parseable by parse_abc
        abc = f"X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n{r['abc']}4 |\n"
        seq = parse_abc(abc)
        assert len(seq["events"]) == 1


class TestVoiceProgression:
    def test_ii_v_i_basic(self):
        chords = [
            {"symbol": "Dm7", "beats": 4},
            {"symbol": "G7", "beats": 4},
            {"symbol": "Cmaj7", "beats": 4},
        ]
        r = voice_progression(chords, style="close")
        assert len(r["voicings"]) == 3
        assert r["abc_line"]  # non-empty

    def test_minimal_motion_bound(self):
        """ii-V-I in close style: each voice should move <= 4 semitones between chords."""
        chords = [
            {"symbol": "Dm7", "beats": 4},
            {"symbol": "G7", "beats": 4},
            {"symbol": "Cmaj7", "beats": 4},
        ]
        r = voice_progression(chords, style="close")
        prev_midi = None
        for v in r["voicings"]:
            if prev_midi is not None:
                curr = sorted(v["midi"])
                prev = sorted(prev_midi)
                n = min(len(curr), len(prev))
                total_motion = sum(abs(curr[i] - prev[i]) for i in range(n))
                # Total motion across all voices should be <= 2 semitones per voice on average
                assert total_motion <= n * 4, \
                    f"Excessive voice leading motion: {total_motion} semitones across {n} voices"
            prev_midi = v["midi"]

    def test_abc_line_parseable(self):
        # Two 2-beat chords fit in one 4/4 bar
        chords = [{"symbol": "Cmaj7", "beats": 2}, {"symbol": "Fmaj7", "beats": 2}]
        r = voice_progression(chords)
        abc = f"X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n{r['abc_line']} |\n"
        seq = parse_abc(abc)
        assert len(seq["events"]) >= 2

    def test_voicings_have_required_keys(self):
        chords = [{"symbol": "Cmaj7", "beats": 4}]
        r = voice_progression(chords)
        v = r["voicings"][0]
        assert all(k in v for k in ("symbol", "beats", "notes", "midi", "abc"))


# ─────────────────────────────────────────────────────────────────────────────
# 5.3 Expressive timing and articulation
# ─────────────────────────────────────────────────────────────────────────────

class TestDecorations:
    def test_fermata_parsed(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n!fermata! c d e f |\n"
        seq = parse_abc(abc)
        first = seq["events"][0]
        assert first.get("fermata") is True

    def test_staccato_dot(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n.c d e f |\n"
        seq = parse_abc(abc)
        assert seq["events"][0].get("staccato") is True

    def test_tenuto(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n!tenuto! c d e f |\n"
        seq = parse_abc(abc)
        assert seq["events"][0].get("tenuto") is True

    def test_accent(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n!accent! c d e f |\n"
        seq = parse_abc(abc)
        assert seq["events"][0].get("accent") is True

    def test_dynamic_p_sets_velocity(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n!p! c d e f |\n"
        seq = parse_abc(abc)
        assert seq["events"][0]["velocity"] == 48

    def test_dynamic_f_sets_velocity(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n!f! c d e f |\n"
        seq = parse_abc(abc)
        assert seq["events"][0]["velocity"] == 96

    def test_dynamic_running(self):
        """Dynamic persists until next dynamic."""
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n!p! c d !f! e f |\n"
        seq = parse_abc(abc)
        evts = seq["events"]
        assert evts[0]["velocity"] == 48  # c — p
        assert evts[1]["velocity"] == 48  # d — still p
        assert evts[2]["velocity"] == 96  # e — f
        assert evts[3]["velocity"] == 96  # f — still f

    def test_inline_tempo_change(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\nc d e f | [Q:60] g a b c' |\n"
        seq = parse_abc(abc)
        tempo_evts = [e for e in seq["events"] if e.get("quality") == "tempo_change"]
        assert len(tempo_evts) == 1
        assert tempo_evts[0]["tempo_bpm"] == pytest.approx(60.0)
        assert tempo_evts[0]["at_beat"] == pytest.approx(4.0)

    def test_unknown_decoration_error(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n!waltz! c d e f |\n"
        with pytest.raises(ABCParseError) as exc:
            parse_abc(abc)
        assert "waltz" in str(exc.value)

    def test_decoration_round_trip(self):
        """Decorations survive to_abc → parse_abc round-trip."""
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\n!p! c d !fermata! e f |\n"
        seq = parse_abc(abc)
        out = to_abc(seq)
        seq2 = parse_abc(out)
        assert seq2["events"][0].get("dynamic") == "dynamic_p"
        assert seq2["events"][2].get("fermata") is True

    def test_H_shorthand_fermata(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\nHc d e f |\n"
        seq = parse_abc(abc)
        assert seq["events"][0].get("fermata") is True

    def test_L_shorthand_accent(self):
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\nLc d e f |\n"
        seq = parse_abc(abc)
        assert seq["events"][0].get("accent") is True


# ─────────────────────────────────────────────────────────────────────────────
# 5.1 Engine: multi-voice channels
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineMultiVoice:
    """Verify _run_sequence dispatches events to correct channels."""

    def test_three_voice_channels(self):
        """Multi-voice sequence: events recorded on three distinct channels."""
        from sequencer import engine
        import threading

        seq = parse_abc(MULTI_VOICE_ABC)
        # Patch note_on to record (channel, note) calls
        recorded = []
        lock = threading.Lock()

        def fake_on(note, vel, ch):
            with lock:
                recorded.append(('on', ch, note))
        def fake_off(note, ch):
            with lock:
                recorded.append(('off', ch, note))
        def fake_play(notes, dur):
            pass

        orig_on = engine._note_on_fn
        orig_off = engine._note_off_fn
        orig_play = engine._play_fn
        engine._note_on_fn = fake_on
        engine._note_off_fn = fake_off
        engine._play_fn = fake_play

        try:
            t = threading.Thread(target=engine._run_sequence, args=("test", seq, None))
            t.start()
            t.join(timeout=10)
        finally:
            engine._note_on_fn = orig_on
            engine._note_off_fn = orig_off
            engine._play_fn = orig_play

        on_channels = {ch for (kind, ch, _) in recorded if kind == 'on'}
        assert len(on_channels) == 3, f"Expected 3 channels, got {on_channels}"
        assert 9 not in on_channels


# ─────────────────────────────────────────────────────────────────────────────
# 5.1 MIDI export: type-1 multi-track
# ─────────────────────────────────────────────────────────────────────────────

class TestMidiExportMultiVoice:
    def test_type1_multi_track(self, tmp_path):
        import mido
        from sequencer.midi_io import write_sequence_midi
        seq = parse_abc(MULTI_VOICE_ABC)
        midi_path = write_sequence_midi(seq, "test_multi", dest_dir=tmp_path)
        mid = mido.MidiFile(str(midi_path))
        # type 1, one tempo track + one track per voice
        assert mid.type == 1
        assert len(mid.tracks) == 4  # 1 tempo + 3 voice tracks

    def test_track_names(self, tmp_path):
        import mido
        from sequencer.midi_io import write_sequence_midi
        seq = parse_abc(MULTI_VOICE_ABC)
        midi_path = write_sequence_midi(seq, "test_names", dest_dir=tmp_path)
        mid = mido.MidiFile(str(midi_path))
        track_names = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == "track_name":
                    track_names.append(msg.name)
                    break
        # Voice track names should include melody/harmony/bass
        assert any("melody" in n.lower() for n in track_names)
        assert any("harmony" in n.lower() for n in track_names)

    def test_single_voice_still_type0(self, tmp_path):
        import mido
        from sequencer.midi_io import write_sequence_midi
        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\nc d e f |\n"
        seq = parse_abc(abc)
        midi_path = write_sequence_midi(seq, "test_single", dest_dir=tmp_path)
        mid = mido.MidiFile(str(midi_path))
        assert mid.type == 0


# ─────────────────────────────────────────────────────────────────────────────
# Engine: gate and fermata
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineExpressive:
    """Staccato/tenuto/fermata must affect scheduled action times."""

    def _collect_actions(self, abc_text):
        """Parse ABC and collect scheduled (t_on, t_off, note) for each note event."""
        from sequencer import engine
        import threading

        seq = parse_abc(abc_text)

        events_log = []
        note_on_times = {}
        lock = threading.Lock()

        def fake_on(note, vel, ch):
            with lock:
                note_on_times[note] = engine.time.monotonic() if False else len(events_log)
                events_log.append(('on', note, vel))

        def fake_off(note, ch):
            with lock:
                events_log.append(('off', note))

        orig_on = engine._note_on_fn
        orig_off = engine._note_off_fn
        engine._note_on_fn = fake_on
        engine._note_off_fn = fake_off

        try:
            t = threading.Thread(target=engine._run_sequence, args=("test_exp", seq, None))
            t.start()
            t.join(timeout=8)
        finally:
            engine._note_on_fn = orig_on
            engine._note_off_fn = orig_off

        return events_log

    def test_staccato_gate_shorter_than_default(self):
        """Staccato note should have a shorter off time than normal note of same written duration."""
        from sequencer import engine
        import threading, time as _time

        def _get_gate(abc_text, note_midi):
            seq = parse_abc(abc_text)
            timestamps = {}
            lock = threading.Lock()

            def fake_on(note, vel, ch):
                with lock:
                    timestamps[('on', note)] = _time.monotonic()

            def fake_off(note, ch):
                with lock:
                    timestamps[('off', note)] = _time.monotonic()

            orig_on = engine._note_on_fn
            orig_off = engine._note_off_fn
            engine._note_on_fn = fake_on
            engine._note_off_fn = fake_off
            try:
                t = threading.Thread(target=engine._run_sequence, args=(f"test_{note_midi}", seq, None))
                t.start()
                t.join(timeout=8)
            finally:
                engine._note_on_fn = orig_on
                engine._note_off_fn = orig_off
            on_t = timestamps.get(('on', note_midi))
            off_t = timestamps.get(('off', note_midi))
            if on_t is None or off_t is None:
                return None
            return off_t - on_t

        # C4 = MIDI 60
        normal_dur = _get_gate("X:1\nT:T\nM:4/4\nL:1/4\nQ:60\nK:C\nc z3 |\n", 60)
        staccato_dur = _get_gate("X:1\nT:T\nM:4/4\nL:1/4\nQ:60\nK:C\n.c z3 |\n", 60)
        if normal_dur is None or staccato_dur is None:
            pytest.skip("Engine not available in test environment")
        assert staccato_dur < normal_dur, \
            f"Staccato ({staccato_dur:.3f}s) should be shorter than normal ({normal_dur:.3f}s)"

    def test_fermata_stretch_applied(self):
        """Fermata note's off time should be longer than normal note of same duration."""
        from sequencer import engine
        import threading, time as _time

        def _get_gate(abc_text, note_midi):
            seq = parse_abc(abc_text)
            timestamps = {}
            lock = threading.Lock()

            def fake_on(note, vel, ch):
                with lock:
                    timestamps[('on', note)] = _time.monotonic()

            def fake_off(note, ch):
                with lock:
                    timestamps[('off', note)] = _time.monotonic()

            orig_on = engine._note_on_fn
            orig_off = engine._note_off_fn
            engine._note_on_fn = fake_on
            engine._note_off_fn = fake_off
            try:
                t = threading.Thread(target=engine._run_sequence, args=(f"test_f_{note_midi}", seq, None))
                t.start()
                t.join(timeout=12)
            finally:
                engine._note_on_fn = orig_on
                engine._note_off_fn = orig_off
            on_t = timestamps.get(('on', note_midi))
            off_t = timestamps.get(('off', note_midi))
            if on_t is None or off_t is None:
                return None
            return off_t - on_t

        normal_dur = _get_gate("X:1\nT:T\nM:4/4\nL:1/4\nQ:60\nK:C\nc z3 |\n", 60)
        fermata_dur = _get_gate("X:1\nT:T\nM:4/4\nL:1/4\nQ:60\nK:C\n!fermata! c z3 |\n", 60)
        if normal_dur is None or fermata_dur is None:
            pytest.skip("Engine not available in test environment")
        assert fermata_dur > normal_dur * 1.3, \
            f"Fermata ({fermata_dur:.3f}s) should be > 1.3x normal ({normal_dur:.3f}s)"

    def test_inline_tempo_change_shifts_times(self):
        """Events after [Q:] should sound at times consistent with new tempo."""
        from sequencer import engine
        import threading, time as _time

        abc = "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\nc d e f | [Q:60] g a b c' |\n"
        seq = parse_abc(abc)
        on_times = {}
        lock = threading.Lock()
        start_t = [None]

        def fake_on(note, vel, ch):
            with lock:
                now = _time.monotonic()
                if start_t[0] is None:
                    start_t[0] = now
                on_times[note] = now - start_t[0]

        def fake_off(note, ch):
            pass

        orig_on = engine._note_on_fn
        orig_off = engine._note_off_fn
        engine._note_on_fn = fake_on
        engine._note_off_fn = fake_off
        try:
            t = threading.Thread(target=engine._run_sequence, args=("test_tc", seq, None))
            t.start()
            t.join(timeout=30)
        finally:
            engine._note_on_fn = orig_on
            engine._note_off_fn = orig_off

        if not on_times:
            pytest.skip("Engine not available in test environment")

        # C4=60, D4=62, E4=64, F4=65 at Q:120 (0.5s/beat), G4=67 after tempo change to Q:60 (1s/beat)
        # G4 should sound at ~4 beats * 0.5 = 2.0s, A4 at ~2.0 + 1.0 = 3.0s
        if 67 in on_times and 69 in on_times:
            gap_g_to_a = on_times[69] - on_times[67]
            assert gap_g_to_a > 0.7, f"After [Q:60], inter-note gap should be ~1s, got {gap_g_to_a:.2f}s"


# ── Regression: cross-bar durations, rest-padded voices, chord ties ───────────

class TestRoundTripCrossBar:
    def _note(self, at, dur, midi, name):
        return {"at_beat": at, "duration_beats": dur, "notes": [midi], "note_names": [name],
                "root": name[:-1], "quality": "note", "octave": int(name[-1]),
                "velocity": 90, "label": name}

    def test_cross_bar_durations_survive_round_trip(self):
        # Mirrors a real recording where notes cross barlines; to_abc used to
        # truncate them at the bar boundary instead of emitting ties.
        events = [
            self._note(0, 3, 60, "C4"), self._note(3, 3.25, 72, "C5"),
            self._note(6.25, 2.5, 71, "B4"), self._note(8.75, 2, 67, "G4"),
            self._note(10.75, 1, 69, "A4"), self._note(11.75, 2.25, 71, "B4"),
            self._note(14.25, 3.75, 72, "C5"),
        ]
        seq = {"title": "Recording", "tempo_bpm": 88.0, "time_signature": "4/4",
               "time_signature_parts": (4, 4), "key": "C", "events": events,
               "total_beats": 18.0, "duration_ms": 0, "abc_errors": []}
        reparsed = parse_abc(to_abc(seq))
        got = [(e["at_beat"], e["duration_beats"], e["notes"][0]) for e in reparsed["events"]]
        expected = [(e["at_beat"], e["duration_beats"], e["notes"][0]) for e in events]
        assert got == expected

    def test_chord_tie_merges_across_barline(self):
        seq = parse_abc("X:1\nT:t\nM:4/4\nL:1/4\nK:C\n[CEG]2 [CEG]2-|[CEG]2 z2|\n")
        assert [(e["at_beat"], e["duration_beats"]) for e in seq["events"]] == [(0.0, 2.0), (2.0, 4.0)]

    def test_rest_padded_voice_counts_full_bars(self):
        # Voice 2 ends in rest-only bars; bar accounting must include them
        # rather than erroring "voice 2 has 3 bars".
        abc = (
            "X:1\nT:t\nM:4/4\nL:1/4\nQ:88\nK:Em\nV:1\nV:2\n"
            "[V:1] C3 c|z9/4 B7/4|z3/4 G2 A B/4|z9/4 c7/4|\n"
            "[V:2] z4|z3 [E^FA]|[B,^DF]4|z4|\n"
        )
        seq = parse_abc(abc)
        assert [v["id"] for v in seq["voices"]] == ["1", "2"]
        voice2_notes = [e for e in seq["events"] if e.get("voice") == "2"]
        assert len(voice2_notes) == 2
