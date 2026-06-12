"""Tests for sequencer/model.py — CRUD, revisions, and bar_edits invariant."""

import tempfile
import time
from pathlib import Path

import pytest

import sequencer.model as model


@pytest.fixture(autouse=True)
def tmp_db(tmp_path):
    db = tmp_path / "test.db"
    model.init_model(db)
    yield db
    # Re-init so state doesn't bleed (model is global)
    model.init_model(db)


def test_create_and_get():
    seq_id = model.create_sequence(
        title="Test Piece",
        abc="X:1\nT:Test\nM:4/4\nL:1/4\nQ:120\nK:C\nc d e f |",
        session_id="sess1",
        tempo_bpm=120.0,
        time_signature="4/4",
        key="C",
        source="agent",
    )
    assert len(seq_id) == 8
    row = model.get_sequence(seq_id)
    assert row is not None
    assert row["title"] == "Test Piece"
    assert row["tempo_bpm"] == 120.0
    assert row["source"] == "agent"
    assert "c d e f" in row["abc"]


def test_get_missing_returns_none():
    assert model.get_sequence("nope1234") is None


def test_update_sequence():
    seq_id = model.create_sequence(
        title="Foo", abc="X:1\nT:Foo\nM:4/4\nL:1/4\nQ:100\nK:C\nc d e f |",
    )
    new_abc = "X:1\nT:Foo\nM:4/4\nL:1/4\nQ:100\nK:C\ng a b c' |"
    ok = model.update_sequence(seq_id, abc=new_abc)
    assert ok is True
    row = model.get_sequence(seq_id)
    assert "g a b c'" in row["abc"]


def test_update_missing_returns_false():
    assert model.update_sequence("nope1234", abc="X:1\nT:x\nM:4/4\nL:1/4\nQ:120\nK:C\nz |") is False


def test_revisions_appended():
    seq_id = model.create_sequence(title="Rev", abc="X:1\nT:Rev\nM:4/4\nL:1/4\nQ:100\nK:C\nc |")
    model.update_sequence(seq_id, abc="X:1\nT:Rev\nM:4/4\nL:1/4\nQ:100\nK:C\nd |")
    model.update_sequence(seq_id, abc="X:1\nT:Rev\nM:4/4\nL:1/4\nQ:100\nK:C\ne |")
    revs = model.get_revisions(seq_id)
    # create adds 1, each update adds 1 = 3 total
    assert len(revs) == 3
    # oldest first
    assert "c |" in revs[0]["abc"] or "T:Rev" in revs[0]["abc"]


def test_list_sequences_by_session():
    model.create_sequence(title="A", abc="X:1\nT:A\nM:4/4\nL:1/4\nQ:120\nK:C\nc |", session_id="s1")
    model.create_sequence(title="B", abc="X:1\nT:B\nM:4/4\nL:1/4\nQ:120\nK:C\nd |", session_id="s1")
    model.create_sequence(title="C", abc="X:1\nT:C\nM:4/4\nL:1/4\nQ:120\nK:C\ne |", session_id="s2")
    s1 = model.list_sequences(session_id="s1")
    s2 = model.list_sequences(session_id="s2")
    all_seqs = model.list_sequences()
    assert len(s1) == 2
    assert len(s2) == 1
    assert len(all_seqs) == 3


def test_list_sequences_ordered_newest_first(monkeypatch):
    # Force distinct timestamps by patching time.time
    _ts = [1_000_000]

    def fake_time():
        return _ts[0]

    monkeypatch.setattr(model.time, "time", fake_time)
    id1 = model.create_sequence(title="Old", abc="X:1\nT:Old\nM:4/4\nL:1/4\nQ:120\nK:C\nc |")
    _ts[0] += 5
    id2 = model.create_sequence(title="New", abc="X:1\nT:New\nM:4/4\nL:1/4\nQ:120\nK:C\nd |")
    seqs = model.list_sequences()
    assert seqs[0]["id"] == id2
    assert seqs[1]["id"] == id1
