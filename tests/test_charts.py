"""Tests for the chord-chart model (Slice B)."""

import pytest

from sequencer import charts
from sequencer.charts import (
    Numeral,
    apply_mode,
    chart_from_spec,
    chart_text,
    get_chart,
    list_charts,
    numeral_for_symbol,
    numeral_to_symbol,
    parse_numeral,
    render_chart,
    spell_degree,
    to_loop_chords,
)


# ── Spelling ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tonic,degree,expected", [
    ("F", 1, "F"),
    ("F", 4, "Bb"),
    ("F", 5, "C"),
    ("C", 4, "F"),
    ("Bb", 4, "Eb"),
    ("Bb", 5, "F"),
    ("B", 4, "E"),
    ("B", 5, "F#"),
    ("Eb", 5, "Bb"),
])
def test_spell_degree_common_keys(tonic, degree, expected):
    assert spell_degree(tonic, degree) == expected


def test_spell_degree_keeps_diatonic_letter_at_the_edges():
    # The IV of Gb is Cb, not B — this is the case a pitch-class-plus-prefer-flats
    # approach gets wrong, and the reason spelling goes by letter arithmetic.
    assert spell_degree("Gb", 4) == "Cb"
    assert spell_degree("F#", 4) == "B"
    assert spell_degree("C#", 5) == "G#"


def test_spell_degree_with_alteration():
    assert spell_degree("C", 7, -1) == "Bb"
    assert spell_degree("C", 3, -1) == "Eb"
    assert spell_degree("C", 4, 1) == "F#"


def test_spell_degree_rejects_nonsense_tonic():
    with pytest.raises(ValueError):
        spell_degree("H", 1)


# ── Numerals ──────────────────────────────────────────────────────────────────

def test_parse_numeral_case_carries_quality():
    assert parse_numeral("I").quality == "major"
    assert parse_numeral("ii").quality == "minor"
    assert parse_numeral("V7").quality == "dominant7"
    assert parse_numeral("ii7").quality == "minor7"
    assert parse_numeral("Imaj7").quality == "major7"


def test_parse_numeral_alterations():
    assert parse_numeral("bVII").degree == 7
    assert parse_numeral("bVII").alteration == -1
    assert parse_numeral("#IV").alteration == 1
    assert parse_numeral("♭III").alteration == -1


def test_parse_numeral_symbols():
    assert parse_numeral("viiø7").quality == "halfdiminished7"
    assert parse_numeral("vii°7").quality == "diminished7"
    assert parse_numeral("vii°").quality == "diminished"


def test_parse_numeral_rejects_junk():
    with pytest.raises(ValueError):
        parse_numeral("H7")
    with pytest.raises(ValueError):
        parse_numeral("VIIIzz")


def test_numeral_text_round_trips():
    for text in ["I", "ii", "V7", "bVII", "Imaj7", "viiø7", "#IV"]:
        assert parse_numeral(parse_numeral(text).text()).quality == parse_numeral(text).quality


def test_numeral_to_symbol():
    assert numeral_to_symbol(parse_numeral("I"), "F") == "F"
    assert numeral_to_symbol(parse_numeral("IV7"), "F") == "Bb7"
    assert numeral_to_symbol(parse_numeral("ii7"), "C") == "Dm7"
    assert numeral_to_symbol(parse_numeral("bVII"), "C") == "Bb"


# ── Modes ─────────────────────────────────────────────────────────────────────

def test_dominant7_mode_makes_every_major_a_dominant():
    for text in ["I", "IV", "V"]:
        assert apply_mode(parse_numeral(text), "dominant7").quality == "dominant7"


def test_triad_mode_strips_sevenths():
    assert apply_mode(parse_numeral("V7"), "triad").quality == "major"
    assert apply_mode(parse_numeral("ii7"), "triad").quality == "minor"
    assert apply_mode(parse_numeral("viiø7"), "triad").quality == "diminished"


def test_seventh_mode_is_diatonic_but_keeps_V_dominant():
    assert apply_mode(parse_numeral("I"), "seventh").quality == "major7"
    assert apply_mode(parse_numeral("IV"), "seventh").quality == "major7"
    assert apply_mode(parse_numeral("V"), "seventh").quality == "dominant7"
    assert apply_mode(parse_numeral("ii"), "seventh").quality == "minor7"


def test_mode_never_changes_the_degree():
    numeral = parse_numeral("bVII7")
    for mode in charts.MODES:
        applied = apply_mode(numeral, mode)
        assert (applied.degree, applied.alteration) == (7, -1)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        apply_mode(parse_numeral("I"), "quartal")


# ── Symbol analysis ───────────────────────────────────────────────────────────

def test_numeral_for_symbol_prefers_the_smallest_alteration():
    # C in the key of F is V, not bVI.
    assert numeral_for_symbol("C7", "F").degree == 5
    assert numeral_for_symbol("C7", "F").alteration == 0
    assert numeral_for_symbol("Bb", "F").degree == 4
    assert numeral_for_symbol("Bb", "C").degree == 7
    assert numeral_for_symbol("Bb", "C").alteration == -1


def test_numeral_for_symbol_breaks_ties_on_the_symbol_spelling():
    # Same pitch, opposite spellings: the symbol's own accidental decides.
    assert numeral_for_symbol("A#", "C").text() == "#VI"
    assert numeral_for_symbol("Bb", "C").text() == "bVII"
    assert numeral_for_symbol("F#7", "C").text() == "#IV7"
    assert numeral_for_symbol("Gb7", "C").text() == "bV7"


def test_numeral_for_symbol_keeps_quality():
    assert numeral_for_symbol("Dm7", "C").quality == "minor7"


# ── The built-in blues ────────────────────────────────────────────────────────

def test_blues_is_twelve_bars():
    assert get_chart("blues-12-bar").bars == 12
    assert get_chart("blues-12-bar-quick-change").bars == 12
    assert get_chart("blues-12-bar-slow").bars == 12


def test_blues_in_f_dominant_mode():
    rendered = render_chart(get_chart("blues-12-bar"), key="F", mode="dominant7")
    assert [s["symbol"] for s in rendered["slots"]] == [
        "F7", "F7", "F7", "F7",
        "Bb7", "Bb7", "F7", "F7",
        "C7", "Bb7", "F7", "C7",
    ]


def test_blues_in_f_triad_mode_is_the_same_form():
    chart = get_chart("blues-12-bar")
    triads = render_chart(chart, key="F", mode="triad")
    sevenths = render_chart(chart, key="F", mode="dominant7")

    assert [s["symbol"] for s in triads["slots"]] == [
        "F", "F", "F", "F", "Bb", "Bb", "F", "F", "C", "Bb", "F", "C",
    ]
    # Wk2 and Wk3 must be looking at the same twelve bars.
    assert [s["degree"] for s in triads["slots"]] == [s["degree"] for s in sevenths["slots"]]
    assert [s["bar"] for s in triads["slots"]] == [s["bar"] for s in sevenths["slots"]]


@pytest.mark.parametrize("key,expected_i,expected_iv,expected_v", [
    ("F", "F7", "Bb7", "C7"),
    ("C", "C7", "F7", "G7"),
    ("Bb", "Bb7", "Eb7", "F7"),
    ("E", "E7", "A7", "B7"),
    ("Ab", "Ab7", "Db7", "Eb7"),
])
def test_blues_transposes_to_any_key(key, expected_i, expected_iv, expected_v):
    rendered = render_chart(get_chart("blues-12-bar"), key=key, mode="dominant7")
    symbols = [s["symbol"] for s in rendered["slots"]]
    assert symbols[0] == expected_i
    assert symbols[4] == expected_iv
    assert symbols[8] == expected_v


def test_roman_overlay_rides_along_with_every_render():
    rendered = render_chart(get_chart("blues-12-bar"), key="Bb", mode="dominant7")
    numerals = [s["numeral"] for s in rendered["slots"]]
    # The overlay is the same in every key — that's the point of the Wk4 lens.
    assert numerals == ["I7"] * 4 + ["IV7", "IV7", "I7", "I7", "V7", "IV7", "I7", "V7"]


def test_slow_blues_turnaround_splits_bars():
    rendered = render_chart(get_chart("blues-12-bar-slow"), key="F", mode="dominant7")
    last_four = rendered["slots"][-4:]
    assert [s["beats"] for s in last_four] == [2, 2, 2, 2]
    assert [s["bar"] for s in last_four] == [11, 11, 12, 12]
    assert [s["symbol"] for s in last_four] == ["F7", "D7", "G7", "C7"]


def test_unknown_chart_names_the_known_ones():
    with pytest.raises(KeyError, match="blues-12-bar"):
        get_chart("nope")


def test_list_charts_reports_bar_counts():
    listed = {c["id"]: c for c in list_charts()}
    assert listed["blues-12-bar"]["bars"] == 12
    assert listed["ii-v-i"]["bars"] == 4


# ── Repeats ───────────────────────────────────────────────────────────────────

def test_sections_repeat_inline():
    chart = chart_from_spec({
        "key": "C",
        "sections": [
            {"label": "A", "slots": ["I", "V"], "repeat": 2},
            {"label": "B", "slots": ["IV", "I"]},
        ],
    })
    rendered = render_chart(chart)
    assert [s["symbol"] for s in rendered["slots"]] == ["C", "G", "C", "G", "F", "C"]
    assert [s["section"] for s in rendered["slots"]] == ["A"] * 4 + ["B"] * 2
    assert [s["pass"] for s in rendered["slots"]] == [1, 1, 2, 2, 1, 1]
    assert rendered["bars"] == 6


# ── Chat-authored charts ──────────────────────────────────────────────────────

def test_bare_string_slots_accept_numerals_and_symbols():
    chart = chart_from_spec({"key": "F", "slots": ["I", "Bb7", "I"]})
    rendered = render_chart(chart, mode="as_written")
    assert [s["symbol"] for s in rendered["slots"]] == ["F", "Bb7", "F"]
    assert [s["numeral"] for s in rendered["slots"]] == ["I", "IV7", "I"]


def test_symbol_authored_chart_still_transposes():
    chart = chart_from_spec({"key": "C", "slots": ["C", "Am", "F", "G7"]})
    rendered = render_chart(chart, key="F")
    assert [s["symbol"] for s in rendered["slots"]] == ["F", "Dm", "Bb", "C7"]


def test_chart_from_spec_requires_slots():
    with pytest.raises(ValueError):
        chart_from_spec({"key": "C"})


def test_chart_from_spec_rejects_bad_slot_type():
    with pytest.raises(ValueError):
        chart_from_spec({"key": "C", "slots": [42]})


# ── Handoff to the loop ───────────────────────────────────────────────────────

def test_to_loop_chords_matches_the_loop_config_shape():
    from sequencer.loop import LoopConfig, build_chart

    rendered = render_chart(get_chart("blues-12-bar"), key="F", mode="dominant7")
    chords = to_loop_chords(rendered)
    assert chords[0] == {"symbol": "F7", "beats": 4.0}

    config = LoopConfig(chords=chords, key="F")
    slots = build_chart(config.chords, config.time_signature)
    assert len(slots) == 12
    assert slots[-1]["bar"] == 12


def test_loop_renders_a_transposed_blues_without_error():
    from sequencer.loop import LoopConfig, render

    rendered_chart = render_chart(get_chart("blues-12-bar"), key="Bb", mode="dominant7")
    result = render(LoopConfig(chords=to_loop_chords(rendered_chart), key="Bb", click=False))
    assert result.total_beats == 48
    assert [v["notes"] for v in result.voicings][0]  # comp actually voiced something


# ── Text view ─────────────────────────────────────────────────────────────────

def test_chart_text_lays_out_four_bars_a_line():
    rendered = render_chart(get_chart("blues-12-bar"), key="F", mode="triad")
    text = chart_text(rendered, roman=False)
    lines = text.splitlines()
    assert "key of F" in lines[0]
    assert len(lines) == 4  # header + three rows of four bars
    assert lines[1].count("|") == 5
    assert "Bb" in lines[2]


def test_chart_text_can_show_the_roman_overlay():
    rendered = render_chart(get_chart("blues-12-bar"), key="C", mode="dominant7")
    assert "C7(I7)" in chart_text(rendered, roman=True)
    assert "C7(I7)" not in chart_text(rendered, roman=False)


# ── Chat pill ─────────────────────────────────────────────────────────────────

def test_start_chart_loop_emits_a_pill_and_a_restorable_record(tmp_path):
    """The pill carries the whole rendered chart, so the client can redraw the form and
    restart the loop from history without re-resolving the chart id."""
    import json

    import sequencer.engine as engine
    import sequencer.loop as loop
    import tools as tools_mod

    engine.set_note_fns(lambda *a, **k: None, lambda *a, **k: None, lambda *a, **k: None, None)
    try:
        out = list(tools_mod.dispatch_tools(
            [{"name": "start_chart_loop", "id": "t1",
              "input": {"chart_id": "blues-12-bar", "key": "F", "mode": "dominant7",
                        "tempo_bpm": 80, "click": False}}],
            session_id="s1", asst_msg_id="a", note_registry={}, sequence_registry={},
            resolve_sequence=lambda i: None, sequence_pill_fn=None, audio_pill_fn=None,
            loop_pill_fn=lambda chart, options: json.dumps({"chart": chart, "options": options}),
            generated_dir=tmp_path, play_notes_bg=None,
        ))
    finally:
        loop.stop()

    pill = json.loads(next(p for k, p in out if k == "sse"))
    assert pill["chart"]["key"] == "F"
    assert pill["chart"]["bars"] == 12
    assert pill["chart"]["slots"][0]["symbol"] == "F7"
    assert pill["options"]["tempo_bpm"] == 80
    assert pill["options"]["repeats"] is None

    record = next(p for k, p in out if k == "record")
    assert record["type"] == "loop"
    assert record["chart"]["slots"] == pill["chart"]["slots"]

    result = next(p for k, p in out if k == "result")
    assert not result.get("is_error"), result["content"]
