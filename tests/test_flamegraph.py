"""Tests for analysis/flamegraph.py -- the dependency-free flame chart renderer.

GPU-free and network-free.  The two profiler-side cases named in the task
brief (``export_stacks=True`` without ``with_stack``; a missing CUDA metric at
export time) belong to ``utils/perf_profiler.py``, which is owned by another
agent and is being edited concurrently -- importing it here would make this
file red for reasons outside its subject.  The renderer-side analogues of
"CUDA stacks were requested but nothing came back" -- a missing file, an empty
file, and a file whose samples sum to zero -- are covered below instead.

Text-overflow is asserted programmatically, not by eye: every emitted
``<text>`` is measured with PIL against DejaVu Sans Mono (which ships with
matplotlib, no download) and must fit inside its own rectangle.
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.flamegraph import (  # noqa: E402
    FRAME_TEXT_PAD,
    FONT_SIZE,
    FOOTER_HEIGHT,
    HEADER_HEIGHT,
    ROW_HEIGHT,
    Frame,
    FoldedStackError,
    build_tree,
    fit_label,
    frame_color,
    layout,
    main,
    metric_from_filename,
    parse_folded,
    render_folded_text,
    render_svg,
    units_for_metric,
    NO_PYTHON_FRAME,
)

SVG_NS = "{http://www.w3.org/2000/svg}"

TINY_FOLDED = "a;b 10\na;c 20\na 5\n"


# --------------------------------------------------------------------------
# parsing policy
# --------------------------------------------------------------------------


def test_parses_well_formed_lines_and_sums_duplicates():
    parsed = parse_folded(["a;b 10", "a;b 5", "a;c 20"])

    assert parsed.issues == []
    assert parsed.accepted_lines == 3
    assert parsed.total == 35
    tree = build_tree(parsed.samples)
    assert tree.children["a"].children["b"].total == 15


def test_value_is_split_from_the_right_so_frames_may_contain_spaces():
    # Real torch frames look like this; splitting on the first space would
    # silently corrupt every one of them.
    parsed = parse_folded(["<built-in method mm of type object>;aten::mm 42"])

    assert parsed.samples == [(("<built-in method mm of type object>", "aten::mm"), 42.0)]


def test_blank_lines_and_comments_are_skipped_without_being_errors():
    parsed = parse_folded(["", "   ", "# a comment", "a 1"])

    assert parsed.issues == []
    assert parsed.accepted_lines == 1


@pytest.mark.parametrize(
    "line, reason_fragment",
    [
        ("a;b", "missing value"),
        ("justastackwithnovalue", "missing value"),
        ("a;b notanumber", "not a number"),
        ("a;b nan", "not finite"),
        ("a;b inf", "not finite"),
        ("a;b -5", "negative"),
        (";  7", "empty frame"),
        ("a;;b 7", "empty frame"),
    ],
)
def test_malformed_lines_raise_under_the_default_strict_policy(line, reason_fragment):
    with pytest.raises(FoldedStackError) as excinfo:
        parse_folded([line])

    assert reason_fragment in excinfo.value.reason
    assert excinfo.value.line_number == 1


def test_lenient_mode_records_every_rejected_line_instead_of_dropping_it_silently():
    parsed = parse_folded(["a 1", "a;b", "b;c -3", "d 2"], strict=False)

    assert parsed.accepted_lines == 2
    assert [issue.line_number for issue in parsed.issues] == [2, 3]
    assert "missing value" in parsed.issues[0].reason
    assert "negative" in parsed.issues[1].reason
    # ...and the count reaches the reader, in the rendered subtitle.
    svg = render_svg(build_tree(parsed.samples), issues=parsed.issues)
    assert "2 malformed line(s) skipped" in svg


def test_frame_names_are_whitespace_stripped_so_padded_stacks_fold_together():
    # Space sorts before every letter, so an unstripped " b" would both split
    # the subtree AND jump to the head of the alphabetical sibling order --
    # exactly the ordering two charts are meant to be compared by.
    parsed = parse_folded(["a; b;c 10", "a;b;c 10", " a ;b;c 5"])

    assert parsed.samples == [
        (("a", "b", "c"), 10.0),
        (("a", "b", "c"), 10.0),
        (("a", "b", "c"), 5.0),
    ]
    tree = build_tree(parsed.samples)
    assert list(tree.children) == ["a"]
    assert tree.children["a"].children["b"].children["c"].total == 25.0


def test_zero_valued_sample_is_accepted_and_adds_no_width():
    parsed = parse_folded(["a;b 0", "a;c 10"])

    assert parsed.issues == []
    tree = build_tree(parsed.samples)
    assert tree.children["a"].children["b"].total == 0.0
    frames = {f.name: f for f in layout(tree, width=1200)[0]}
    assert frames["b"].width == 0.0


def test_error_message_names_the_line_and_the_offending_text():
    with pytest.raises(FoldedStackError) as excinfo:
        parse_folded(["ok 1", "broken;stack"])

    message = str(excinfo.value)
    assert "line 2" in message
    assert "broken;stack" in message


# --------------------------------------------------------------------------
# inclusive width computation
# --------------------------------------------------------------------------


def test_inclusive_widths_on_a_hand_computed_tree():
    # root=100 : a=70 (a;b=30, a;c=20, a self=20), d=30
    folded = "a;b 30\na;c 20\na 20\nd 30\n"
    tree = build_tree(parse_folded(folded.splitlines()).samples)
    frames, _height = layout(tree, width=1010)  # plot width 990 after 2x10 margin
    by_name = {f.name: f for f in frames}

    assert tree.total == 100
    assert by_name["all"].width == pytest.approx(990.0)
    assert by_name["a"].width == pytest.approx(990.0 * 0.70)
    assert by_name["b"].width == pytest.approx(990.0 * 0.30)
    assert by_name["c"].width == pytest.approx(990.0 * 0.20)
    assert by_name["d"].width == pytest.approx(990.0 * 0.30)
    # children start at the parent's left edge; the parent's self value is the
    # uncovered remainder at its right.
    assert by_name["a"].x == pytest.approx(by_name["b"].x)
    assert by_name["c"].x == pytest.approx(10.0 + 990.0 * 0.30)
    assert by_name["d"].x == pytest.approx(10.0 + 990.0 * 0.70)


def test_depth_and_canvas_height_follow_the_deepest_stack():
    tree = build_tree(parse_folded(["a;b;c;d 1"]).samples)
    frames, height = layout(tree, width=1200)

    assert max(f.depth for f in frames) == 4  # synthetic root + 4 frames
    assert height == HEADER_HEIGHT + 5 * ROW_HEIGHT + FOOTER_HEIGHT


def test_every_child_is_geometrically_contained_in_its_parent():
    # A wide, deep, ragged tree: rounding drift would show up as a child
    # poking past its parent's right edge by a fraction of a pixel.
    #
    # Parent identity is carried from the tree, NOT inferred from x-interval
    # overlap.  An overlap heuristic would still pass if layout() attached a
    # child to the wrong parent, since a mis-parented box usually still lands
    # inside *some* box on the row below.
    lines: List[str] = []
    for i in range(23):
        for j in range(7):
            lines.append(f"root;branch_{i:02d};leaf_{j} {i * 7 + j + 1}")
    tree = build_tree(parse_folded(lines).samples)
    width = 1237
    frames, _ = layout(tree, width=width)

    # layout() emits depth-first with siblings alphabetical, so re-walking the
    # tree in the same order pairs each emitted Frame with its true parent.
    emitted = iter(frames)
    checked = 0

    def walk(node, depth: int, parent: Optional[Frame]) -> None:
        nonlocal checked
        frame = next(emitted)
        assert frame.name == node.name and frame.depth == depth, (
            f"emission order diverged from the tree at {node.name!r}"
        )
        if parent is not None:
            assert frame.x >= parent.x - 1e-9, f"{frame.name} starts left of {parent.name}"
            assert frame.x + frame.width <= parent.x + parent.width + 1e-9, (
                f"{frame.name} extends past the right edge of {parent.name}"
            )
            checked += 1
        for child in node.sorted_children():
            walk(child, depth + 1, frame)

    walk(tree, 0, None)
    assert next(emitted, None) is None, "layout() emitted a frame with no tree node"
    assert checked == 23 * 7 + 23 + 1  # leaves + branches + root's own child

    # ...and the same containment must survive the SVG's 3-decimal rounding,
    # which is what a viewer actually sees.
    svg = render_svg(tree, width=width)
    root_el = ET.fromstring(svg)
    rects = [r for r in root_el.iter(f"{SVG_NS}rect") if r.get("fill") != "#ffffff"]
    assert len(rects) == len(frames)
    for rect in rects:
        x, w = float(rect.get("x")), float(rect.get("width"))
        assert x >= 10.0 - 1e-9 and x + w <= width - 10.0 + 1e-9


def test_sibling_totals_never_exceed_the_parent_total():
    tree = build_tree(parse_folded(["a;b 3", "a;c 4", "a 5"]).samples)
    node = tree.children["a"]

    assert sum(c.total for c in node.children.values()) + node.self_value == node.total


# --------------------------------------------------------------------------
# alphabetical sibling ordering
# --------------------------------------------------------------------------


def test_siblings_are_ordered_alphabetically_not_by_size():
    # zebra is by far the largest; alphabetical order must still put it last.
    folded = "root;zebra 1000\nroot;alpha 1\nroot;middle 10\n"
    tree = build_tree(parse_folded(folded.splitlines()).samples)
    frames, _ = layout(tree, width=1200)
    row = sorted((f for f in frames if f.depth == 2), key=lambda f: f.x)

    assert [f.name for f in row] == ["alpha", "middle", "zebra"]


def test_sibling_order_is_independent_of_input_line_order():
    a = render_folded_text("root;zebra 5\nroot;alpha 5\nroot;middle 5\n")
    b = render_folded_text("root;middle 5\nroot;zebra 5\nroot;alpha 5\n")

    assert a == b


def test_sorted_children_helper_is_alphabetical():
    tree = build_tree(parse_folded(["r;b 1", "r;a 1", "r;C 1"]).samples)

    assert [c.name for c in tree.children["r"].sorted_children()] == ["C", "a", "b"]


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_rendering_twice_is_byte_identical():
    first = render_folded_text(TINY_FOLDED, title="t", metric="self_cpu_time_total")
    second = render_folded_text(TINY_FOLDED, title="t", metric="self_cpu_time_total")

    assert first == second


def test_output_is_identical_across_processes_with_different_hash_seeds(tmp_path: Path):
    # The real determinism risk is cross-process: a salted hash() or an
    # unsorted dict iteration only diverges between interpreter runs.
    folded = tmp_path / "stacks_self_cpu_time_total.txt"
    folded.write_text(
        "".join(f"root;mod_{i % 13};fn_{i} {i + 1}\n" for i in range(200)), encoding="utf-8"
    )

    # Each child also reports hash("probe"); if the two children did not in
    # fact end up with different hash salts, the identical-bytes assertion
    # below would prove nothing, so that is asserted first.
    driver = (
        "import sys, runpy\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "sys.stderr.write(repr(hash('probe')))\n"
        "sys.argv = ['flamegraph.py', sys.argv[1]]\n"
        f"runpy.run_path({str(REPO_ROOT / 'analysis' / 'flamegraph.py')!r}, run_name='__main__')\n"
    )

    outputs, salts = [], []
    for seed in ("1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run(
            [sys.executable, "-c", driver, str(folded)],
            capture_output=True,
            env=env,
        )
        outputs.append(proc.stdout)
        salts.append(proc.stderr)

    assert salts[0] != salts[1], "hash seeds did not actually differ; the test is vacuous"
    assert outputs[0] == outputs[1]
    assert b"<svg" in outputs[0]


def test_frame_colour_depends_only_on_the_name():
    assert frame_color("aten::mm") == frame_color("aten::mm")
    assert frame_color("aten::mm") != frame_color("aten::addmm")
    # Pinned so a future refactor of the hash is caught rather than silently
    # invalidating every archived before/after comparison.
    assert frame_color("aten::mm") == "#f3b02a"
    assert frame_color("") == "#f9cc30"


def test_label_colour_flips_to_white_on_the_dark_end_of_the_palette():
    from analysis.flamegraph import text_color_for

    assert text_color_for("#cd0000") == "#ffffff"  # near-pure red, dark
    assert text_color_for("#ffe637") == "#000000"  # yellow, light
    # still a pure function -> determinism is untouched
    assert text_color_for(frame_color("aten::mm")) == text_color_for(frame_color("aten::mm"))


@pytest.mark.parametrize("bad", ["", "#abc", "cd0000", "#gggggg", "#cd00000"])
def test_text_color_for_rejects_a_malformed_colour(bad):
    from analysis.flamegraph import text_color_for

    with pytest.raises(ValueError):
        text_color_for(bad)


def test_every_label_has_readable_contrast_against_its_own_frame():
    from analysis.flamegraph import text_color_for

    # few enough siblings that each box is wide enough to carry a label,
    # while the names still spread across the whole hash-derived palette.
    folded = "".join(f"frame_{i:02d} 100\n" for i in range(12))
    svg = render_folded_text(folded, width=1400)
    root = ET.fromstring(svg)

    checked = 0
    for group in root.iter(f"{SVG_NS}g"):
        rect = group.find(f"{SVG_NS}rect")
        text = group.find(f"{SVG_NS}text")
        if rect is None or text is None:
            continue
        assert text.get("fill") == text_color_for(rect.get("fill"))
        checked += 1
    assert checked > 5


def test_frame_colour_is_stable_across_processes():
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from analysis.flamegraph import frame_color; "
        "print(frame_color('aten::mm'), hash('probe'))" % str(REPO_ROOT)
    )
    colours, salts = set(), set()
    for seed in ("1", "999"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, check=True, text=True, env=env
        )
        colour, salt = proc.stdout.split()
        colours.add(colour)
        salts.add(salt)

    assert len(salts) == 2, "hash seeds did not actually differ; the test is vacuous"
    assert colours == {"#f3b02a"}


# --------------------------------------------------------------------------
# XML escaping / hostile names
# --------------------------------------------------------------------------


def test_hostile_frame_names_are_xml_escaped_and_the_svg_still_parses():
    hostile = 'std::vector<int, alloc&> "x" \'y\';<script>alert(1)</script> 100'
    svg = render_folded_text(hostile, title='a & b <c> "d" \'e\'')

    root = ET.fromstring(svg)  # would raise on malformed XML
    titles = [el.text for el in root.iter(f"{SVG_NS}title")]
    assert "std::vector<int, alloc&>" in " ".join(t or "" for t in titles)
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg
    assert "&amp;" in svg


def test_raw_ampersand_never_reaches_the_output_unescaped():
    svg = render_folded_text("a&b;c 1")

    assert "a&b" not in svg
    assert "a&amp;b" in svg
    ET.fromstring(svg)


# --------------------------------------------------------------------------
# labels, truncation, geometry bounds
# --------------------------------------------------------------------------


def test_narrow_frames_are_drawn_but_not_labelled():
    # one huge frame plus a sliver well under 0.1% of the total
    folded = "big 100000\nsliver 1\n"
    svg = render_folded_text(folded, width=1200)
    root = ET.fromstring(svg)

    rect_widths = [float(r.get("width")) for r in root.iter(f"{SVG_NS}rect")]
    assert min(w for w in rect_widths if w > 0) < 1.0  # the sliver is drawn
    labels = {t.text for t in root.iter(f"{SVG_NS}text")}
    assert "sliver" not in labels
    # ...but it is still discoverable via its tooltip
    assert any("sliver" in (t.text or "") for t in root.iter(f"{SVG_NS}title"))


def test_long_names_are_truncated_with_an_ellipsis_but_kept_whole_in_the_tooltip():
    long_name = "torch/nn/modules/module.py(1518): _call_impl_with_a_very_long_suffix"
    folded = f"{long_name} 50\nother 50\n"
    svg = render_folded_text(folded, width=400)
    root = ET.fromstring(svg)

    labels = [t.text for t in root.iter(f"{SVG_NS}text")]
    truncated = [t for t in labels if t and t.startswith("torch/nn")]
    assert truncated and "..." in truncated[0]
    assert len(truncated[0]) < len(long_name)
    assert any(long_name in (t.text or "") for t in root.iter(f"{SVG_NS}title"))


def test_on_frame_label_carries_a_percentage_when_the_box_has_room():
    # Two evenly-split siblings, each with a plain short name -- plenty of room
    # for "name (50.0%)" whole. fig-qa 2026-07-29: without this, ranking
    # same-width siblings requires decoding relative box widths by eye.
    folded = "left 50\nright 50\n"
    svg = render_folded_text(folded, width=400)
    labels = {t.text for t in ET.fromstring(svg).iter(f"{SVG_NS}text")}
    assert "left (50.0%)" in labels
    assert "right (50.0%)" in labels


def test_long_name_percentage_reserves_room_for_the_suffix_before_truncating():
    # A long repr-style name (real torch frames: "<built-in method mm of type
    # object at 0x...>") must still surface its percentage: the name truncates
    # around the suffix rather than the suffix being dropped because the whole
    # "name (pct%)" string didn't fit.
    long_name = "<built-in method mm of type object at 0x7f1869aa9b40>"
    folded = f"{long_name} 30\nother 70\n"
    svg = render_folded_text(folded, width=400)
    labels = [t.text for t in ET.fromstring(svg).iter(f"{SVG_NS}text")]
    mm_label = next(t for t in labels if t and t.startswith("<"))
    assert mm_label.endswith("(30.0%)")
    assert "..." in mm_label


def test_fit_label_returns_none_when_the_box_cannot_hold_a_readable_label():
    assert fit_label("anything", 8.0) is None
    assert fit_label("abc", 1000.0) == "abc"


def test_fit_label_prefers_no_label_over_an_all_ellipsis_stub():
    # 4 characters of room: "all" fits whole, but "a_long_name" would come out
    # as "a..." -- one real character, pure noise. Drop it instead.
    box = 4 * FONT_SIZE * 0.62 + 2 * FRAME_TEXT_PAD
    assert fit_label("all", box) == "all"
    assert fit_label("a_long_name", box) is None
    # ...whereas with 6 characters of room a truncation carries real signal.
    wider = 6 * FONT_SIZE * 0.62 + 2 * FRAME_TEXT_PAD
    assert fit_label("a_long_name", wider) == "a_l..."


def test_a_long_header_title_is_truncated_rather_than_clipped_by_the_viewbox():
    fm = pytest.importorskip("matplotlib.font_manager")
    ImageFont = pytest.importorskip("PIL.ImageFont")

    from analysis.flamegraph import MARGIN_X, TITLE_FONT_SIZE

    long_title = "capture_00/" + "very_long_run_name_" * 16
    width = 800
    svg = render_folded_text(TINY_FOLDED, title=long_title, width=width)
    root = ET.fromstring(svg)

    header = next(t for t in root.iter(f"{SVG_NS}text") if t.get("font-weight") == "bold")
    drawn = header.text or ""
    assert drawn.endswith("...")
    assert len(drawn) < len(long_title)

    # measured with the real face: centred at width/2, it must stay on canvas
    font = ImageFont.truetype(fm.findfont("DejaVu Sans Mono"), int(TITLE_FONT_SIZE))
    advance = font.getlength(drawn)
    assert advance <= width - 2 * MARGIN_X + 0.5
    assert width / 2 - advance / 2 >= 0.0

    # the untruncated title survives in a tooltip, as a sibling <title> in a
    # wrapping <g> -- never nested inside <text>, which would blank .text
    assert long_title in [t.text for t in root.iter(f"{SVG_NS}title")]
    assert header.text == drawn


def test_a_normal_length_title_emits_no_extra_tooltip_group():
    svg = render_folded_text(TINY_FOLDED, title="before: train step", width=1200)
    root = ET.fromstring(svg)

    header = next(t for t in root.iter(f"{SVG_NS}text") if t.get("font-weight") == "bold")
    assert header.text == "before: train step"
    # only the per-frame tooltips; the header adds none when nothing was cut
    assert all("before: train step" != (t.text or "") for t in root.iter(f"{SVG_NS}title"))


def test_no_frame_is_drawn_outside_the_canvas():
    folded = "".join(f"root;a{i};b{i};c{i} {i + 1}\n" for i in range(40))
    tree = build_tree(parse_folded(folded.splitlines()).samples)
    width = 1200
    frames, height = layout(tree, width=width)

    assert min(f.y for f in frames) >= HEADER_HEIGHT - 1e-9
    assert max(f.y + f.height for f in frames) <= height - 1e-9
    assert min(f.x for f in frames) >= 10.0 - 1e-9
    assert max(f.x + f.width for f in frames) <= width - 10.0 + 1e-9


def test_svg_declares_matching_width_height_and_viewbox():
    svg = render_folded_text(TINY_FOLDED, width=900)
    root = ET.fromstring(svg)

    assert root.get("width") == "900"
    assert root.get("viewBox") == f"0 0 900 {root.get('height')}"
    # The root must carry NO CSS style. The responsive idiom
    # style="max-width:100%;height:auto" works in browsers but makes cairosvg --
    # the only rasteriser available here -- emit a single flat colour, which was
    # caught on the real M3 capture: 1 distinct colour with the attribute, 7523
    # without. A chart that cannot be rasterised cannot be checked by looking at
    # it. width/height + viewBox already scale correctly where this is embedded.
    assert root.get("style") is None


def test_svg_is_self_contained_and_avoids_the_font_shorthand():
    svg = render_folded_text(TINY_FOLDED)

    assert "http://" not in svg.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in svg
    assert "<image" not in svg
    assert "@import" not in svg
    assert "font:" not in svg  # the CSS shorthand renders inconsistently here
    assert 'font-family="' in svg
    assert 'font-size="' in svg


def test_every_label_physically_fits_its_box_measured_with_a_real_font():
    """Independent check of the truncation arithmetic using PIL metrics."""
    fm = pytest.importorskip("matplotlib.font_manager")
    ImageFont = pytest.importorskip("PIL.ImageFont")

    font_path = fm.findfont("DejaVu Sans Mono")
    font = ImageFont.truetype(font_path, int(FONT_SIZE))

    names = [
        "aten::mm",
        "torch/nn/modules/module.py(1518): _call_impl",
        "<built-in method conv2d of type object at 0x7f0000000000>",
        "W" * 120,
        "i" * 120,
    ]
    folded = "".join(f"root;{n} {i * 3 + 1}\n" for i, n in enumerate(names))
    svg = render_folded_text(folded, width=700)
    root = ET.fromstring(svg)

    checked = 0
    for group in root.iter(f"{SVG_NS}g"):
        rect = group.find(f"{SVG_NS}rect")
        text = group.find(f"{SVG_NS}text")
        if rect is None or text is None:
            continue
        advance = font.getlength(text.text or "")
        assert advance <= float(rect.get("width")) - 2 * FRAME_TEXT_PAD + 0.5, (
            f"label {text.text!r} overflows its {rect.get('width')}px box"
        )
        checked += 1
    assert checked >= 3


# --------------------------------------------------------------------------
# degenerate / "asked for CUDA, got nothing" inputs
# --------------------------------------------------------------------------


def test_empty_input_renders_a_legible_empty_chart_without_dividing_by_zero():
    svg = render_folded_text("", title="stacks_self_cuda_time_total.txt")
    root = ET.fromstring(svg)

    assert "(no samples)" in svg
    assert root.get("height") is not None
    assert not list(root.iter(f"{SVG_NS}title"))


def test_all_zero_samples_render_an_empty_chart_rather_than_nan_widths():
    svg = render_folded_text("a;b 0\na;c 0\n")

    assert "(no samples)" in svg
    assert "nan" not in svg.lower()
    ET.fromstring(svg)


def test_cli_reports_a_missing_file_without_traceback(tmp_path, capsys):
    code = main([str(tmp_path / "stacks_self_cuda_time_total.txt")])

    assert code == 2
    assert "no such folded-stack file" in capsys.readouterr().err


def test_cli_on_an_empty_cuda_stack_file_still_produces_an_svg(tmp_path, capsys):
    src = tmp_path / "stacks_self_cuda_time_total.txt"
    src.write_text("", encoding="utf-8")
    out = tmp_path / "flame.svg"

    code = main([str(src), "-o", str(out)])
    err = capsys.readouterr().err

    assert code == 0
    assert "no usable samples" in err
    assert "(no samples)" in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# CLI behaviour
# --------------------------------------------------------------------------


def test_cli_writes_an_svg_and_infers_the_metric_from_the_filename(tmp_path, capsys):
    src = tmp_path / "stacks_self_cpu_time_total.txt"
    src.write_text(TINY_FOLDED, encoding="utf-8")
    out = tmp_path / "sub" / "flame.svg"

    assert main([str(src), "-o", str(out)]) == 0
    svg = out.read_text(encoding="utf-8")
    assert "self_cpu_time_total" in svg
    assert "total 35 us" in svg
    capsys.readouterr()


def test_cli_fails_loudly_on_malformed_input_and_succeeds_with_lenient(tmp_path, capsys):
    src = tmp_path / "stacks_self_cpu_time_total.txt"
    src.write_text("a;b 10\nbroken line here\n", encoding="utf-8")
    out = tmp_path / "flame.svg"

    assert main([str(src), "-o", str(out)]) == 2
    assert "--lenient" in capsys.readouterr().err
    assert not out.exists()

    assert main([str(src), "-o", str(out), "--lenient"]) == 0
    assert "skipped 1 malformed line" in capsys.readouterr().err
    assert "1 malformed line(s) skipped" in out.read_text(encoding="utf-8")


def test_metric_and_unit_inference():
    assert metric_from_filename("/x/stacks_self_cuda_time_total.txt") == "self_cuda_time_total"
    assert metric_from_filename("/x/whatever.txt") is None
    assert units_for_metric("self_cpu_time_total") == "us"
    assert units_for_metric(None) is None


def test_width_is_bounded():
    with pytest.raises(ValueError):
        render_folded_text(TINY_FOLDED, width=10)


# --------------------------------------------------------------------------
# scale
# --------------------------------------------------------------------------


def test_twenty_thousand_folded_lines_render_to_well_formed_xml():
    lines = []
    for i in range(20_000):
        lines.append(
            f"root;module_{i % 97};layer_{i % 311};aten::op_{i % 1009} {(i % 37) + 1}"
        )
    parsed = parse_folded(lines)
    tree = build_tree(parsed.samples)
    svg = render_svg(tree, title="scale", width=1600, metric="self_cpu_time_total", units="us")

    root = ET.fromstring(svg)  # well-formed even at this size
    assert len(svg) > 100_000
    assert root.tag == f"{SVG_NS}svg"


def test_very_deep_stacks_do_not_hit_a_recursion_limit():
    deep = ";".join(f"frame_{i:04d}" for i in range(2000))
    svg = render_folded_text(f"{deep} 1")

    root = ET.fromstring(svg)
    assert float(root.get("height")) > 2000 * ROW_HEIGHT


# --------------------------------------------------------------------------
# py-spy interop: samples with no attributable Python frame
# --------------------------------------------------------------------------


def test_bare_numeric_line_becomes_a_named_frame_not_a_rejection():
    """py-spy writes ' 1286' for samples with no Python frame.

    Rejecting it would drop real signal AND shrink the denominator, inflating
    every other frame's percentage.  It must survive as an explicit frame.
    """
    parsed = parse_folded(["a;b 10", " 1286"])

    assert parsed.issues == []
    assert ((NO_PYTHON_FRAME,), 1286.0) in parsed.samples
    assert parsed.total == pytest.approx(1296.0)

    root = build_tree(parsed.samples)
    assert root.total == pytest.approx(1296.0)
    assert NO_PYTHON_FRAME in root.children


def test_lone_non_numeric_token_is_still_a_missing_value_error():
    """The bare-value rule must not swallow a genuine stack-without-a-value."""
    with pytest.raises(FoldedStackError) as excinfo:
        parse_folded(["a;b"])
    assert "missing value" in excinfo.value.reason


def test_real_pyspy_output_parses_without_lenient_mode(tmp_path):
    """End-to-end shape of an actual `py-spy record --format raw` file."""
    text = (
        "<module> (train.py:8);outer (train.py:7);inner (train.py:4) 161\n"
        "<module> (train.py:8);outer (train.py:7) 12\n"
        " 40\n"
    )
    parsed = parse_folded(text.splitlines())

    assert parsed.issues == []
    assert parsed.total == pytest.approx(213.0)
    assert NO_PYTHON_FRAME in render_folded_text(text)
