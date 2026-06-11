"""Tests for sequencer/midi_io.py — Phase 2 (import/corpus tools)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from sequencer.midi_io import search_corpus, load_corpus_entry, ENABLE_FETCH_ABC, fetch_abc_url
from sequencer.abc import to_abc, parse_abc


class TestSearchCorpus:
    def test_bach_returns_results(self):
        results = search_corpus("bach")
        assert len(results) > 0
        for r in results:
            assert "title" in r
            assert "corpus_path" in r

    def test_no_results_returns_empty(self):
        results = search_corpus("xyzzy_no_such_piece_12345")
        assert results == []

    def test_max_results_respected(self):
        results = search_corpus("bach", max_results=2)
        assert len(results) <= 2


class TestLoadCorpusEntry:
    def test_bach_bwv1_6_loads(self):
        seq, dropped = load_corpus_entry("bach/bwv1.6")
        assert len(seq["events"]) > 0
        assert seq["total_beats"] > 0
        assert seq["tempo_bpm"] > 0
        assert "/" in seq["time_signature"]

    def test_abc_produced(self):
        seq, _ = load_corpus_entry("bach/bwv1.6")
        abc = to_abc(seq)
        # Must produce a non-empty ABC string with the required headers
        assert "X:1" in abc
        assert "M:" in abc
        assert "K:" in abc
        assert len(abc) > 100

    def test_not_found_raises(self):
        with pytest.raises((ValueError, Exception)):
            load_corpus_entry("this/does/not/exist.mxl")

    def test_sequence_structure(self):
        seq, _ = load_corpus_entry("bach/bwv1.6")
        required = {"title", "tempo_bpm", "time_signature", "time_signature_parts",
                    "key", "events", "total_beats", "duration_ms"}
        assert required.issubset(seq.keys())
        for evt in seq["events"][:5]:
            assert "at_beat" in evt
            assert "duration_beats" in evt
            assert "notes" in evt
            assert len(evt["notes"]) >= 1


class TestFetchAbcUrl:
    def test_disabled_by_default(self):
        if ENABLE_FETCH_ABC:
            pytest.skip("fetch_abc is enabled in this environment")
        with pytest.raises(RuntimeError, match="disabled"):
            fetch_abc_url("http://example.com/tune.abc")
