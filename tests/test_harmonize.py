"""Tests for the harmonize_sequence tool: server-computed chord alignment."""

import pytest

import sequencer.model as seq_model
from sequencer.abc import parse_abc, to_abc


MELODY_ABC = """X:1
T:Recording
M:4/4
L:1/4
Q:72
K:C
C2 c2 | B G3/4 A/2 B5/4 c/2- | c7/4 z9/4 |
"""


@pytest.fixture
def session(tmp_path, monkeypatch):
    seq_model.init_model(tmp_path / "test.db")
    import tools
    monkeypatch.setattr(tools.engine, "play_sequence_bg", lambda *a, **k: None)
    monkeypatch.setattr(tools, "write_sequence_midi", lambda seq, sid, d=None: tmp_path / "x.mid")
    seq = parse_abc(MELODY_ABC)
    seq_id = seq_model.create_sequence(
        title="Recording", abc=to_abc(seq), session_id="s1",
        tempo_bpm=72, time_signature="4/4", source="recording",
    )
    return tools, seq_id, tmp_path


def _dispatch(tools, calls, tmp_path):
    return list(tools.dispatch_tools(
        calls, session_id="s1", asst_msg_id="a", note_registry={}, sequence_registry={},
        resolve_sequence=lambda i: None, sequence_pill_fn=lambda *a: "<pill>",
        audio_pill_fn=None, generated_dir=tmp_path, play_notes_bg=None,
    ))


def _result(out):
    return next(p for k, p in out if k == "result")


class TestHarmonizeSequence:
    def test_chords_align_to_melody_onsets(self, session):
        tools, seq_id, tmp_path = session
        out = _dispatch(tools, [{"name": "harmonize_sequence", "id": "t1", "input": {
            "sequence_id": seq_id,
            "anchors": [{"chord": "C", "at_event": 1},
                        {"chord": "Am", "at_event": 2},
                        {"chord": "Em", "at_event": 3}],
        }}], tmp_path)
        res = _result(out)
        assert not res.get("is_error"), res["content"]

        record = next(p for k, p in out if k == "record")
        seq = record["sequence"]
        mel = parse_abc(MELODY_ABC)
        mel_events = sorted(mel["events"], key=lambda e: e["at_beat"])
        chord_evts = sorted((e for e in seq["events"] if e.get("voice") == "2"),
                            key=lambda e: e["at_beat"])
        # Chord onsets land exactly on melody events 1, 2, 3
        assert [c["at_beat"] for c in chord_evts[:3]] == pytest.approx(
            [mel_events[i]["at_beat"] for i in range(3)])
        # Chords tile the piece with no gaps: each runs to the next anchor / the end
        ends = [c["at_beat"] + c["duration_beats"] for c in chord_evts]
        starts = [c["at_beat"] for c in chord_evts[1:]] + [seq["total_beats"]]
        assert ends == pytest.approx(starts)

    def test_melody_preserved_verbatim(self, session):
        tools, seq_id, tmp_path = session
        out = _dispatch(tools, [{"name": "harmonize_sequence", "id": "t1", "input": {
            "sequence_id": seq_id, "anchors": [{"chord": "C", "at_event": 1}],
        }}], tmp_path)
        record = next(p for k, p in out if k == "record")
        v1 = [(e["at_beat"], e["duration_beats"], e["notes"])
              for e in record["sequence"]["events"] if e.get("voice") == "1"]
        mel = [(e["at_beat"], e["duration_beats"], e["notes"])
               for e in sorted(parse_abc(MELODY_ABC)["events"], key=lambda e: e["at_beat"])]
        assert v1 == mel

    def test_result_abc_reparses(self, session):
        tools, seq_id, tmp_path = session
        out = _dispatch(tools, [{"name": "harmonize_sequence", "id": "t1", "input": {
            "sequence_id": seq_id,
            "anchors": [{"chord": "C", "at_event": 1}, {"chord": "Em", "at_beat": 4.0}],
        }}], tmp_path)
        res = _result(out)
        assert not res.get("is_error"), res["content"]
        record = next(p for k, p in out if k == "record")
        stored = seq_model.get_sequence(record["sequence_id"])
        parse_abc(stored["abc"])  # raises on invalid bars

    def test_unknown_sequence_errors(self, session):
        tools, _, tmp_path = session
        out = _dispatch(tools, [{"name": "harmonize_sequence", "id": "t1", "input": {
            "sequence_id": "nope", "anchors": [{"chord": "C", "at_event": 1}],
        }}], tmp_path)
        assert _result(out)["is_error"]

    def test_at_event_out_of_range_errors(self, session):
        tools, seq_id, tmp_path = session
        out = _dispatch(tools, [{"name": "harmonize_sequence", "id": "t1", "input": {
            "sequence_id": seq_id, "anchors": [{"chord": "C", "at_event": 99}],
        }}], tmp_path)
        assert _result(out)["is_error"]
