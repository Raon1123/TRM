#!/usr/bin/env python3
"""Dependency-free folded-stack -> SVG flame chart renderer.

Input is the canonical *folded stack* format that ``torch.profiler``'s
``prof.export_stacks(path, metric)`` writes and that flamegraph.pl consumes::

    frame_a;frame_b;frame_c <value>

One line per unique stack; ``<value>`` is that stack's *self* cost (for torch
this is microseconds of ``self_cpu_time_total`` / ``self_cuda_time_total``).
A frame's drawn width is its *inclusive* value: its own self value plus every
descendant's.

Design constraints, all load-bearing for this repo:

* **Standard library only.**  No matplotlib, no d3, no SVG library, no
  network.  flamegraph.pl is not available here and cannot be fetched.
* **Deterministic.**  The same folded file always produces byte-identical SVG.
  Frame colours come from a SHA-256 of the frame name -- never ``hash()``
  (salted per process), never ``random``, never a timestamp.  Siblings are
  ordered **alphabetically**, not by size, so a before/after pair of charts
  lines up column-for-column and can be diffed.  That comparability is the
  entire point of the tool for optimization work.
* **No silent data loss.**  Malformed lines are an error by default; in
  ``--lenient`` mode they are counted, reported on stderr *and* stamped into
  the rendered subtitle.

Text is laid out against a monospace family on purpose: with a known advance
ratio the truncation arithmetic is checkable, so "does this label overflow its
box?" is a test assertion rather than an eyeball judgement.

CLI::

    uv run python analysis/flamegraph.py stacks_self_cpu_time_total.txt \\
        -o flame_cpu.svg --title "before: train step"
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "FoldedStackError",
    "ParseIssue",
    "FoldedStacks",
    "FrameNode",
    "Frame",
    "parse_folded",
    "build_tree",
    "layout",
    "render_svg",
    "render_folded_text",
    "frame_color",
    "text_color_for",
    "metric_from_filename",
    "units_for_metric",
    "main",
]

# --- geometry / typography -------------------------------------------------
# Monospace: the advance of DejaVu Sans Mono is 1233/2048 = 0.602 em.  We
# budget 0.62 em per character so the truncation estimate is conservative
# (labels come out narrower than their box, never wider) even when a viewer
# substitutes a slightly wider monospace face.
CHAR_WIDTH_RATIO = 0.62
FONT_FAMILY = "DejaVu Sans Mono, Menlo, Consolas, monospace"
FONT_SIZE = 12.0
TITLE_FONT_SIZE = 17.0
SUBTITLE_FONT_SIZE = 11.0

ROW_HEIGHT = 16.0
FRAME_HEIGHT = 15.0  # ROW_HEIGHT minus a 1px gutter between rows
FRAME_TEXT_PAD = 3.0
MARGIN_X = 10.0
HEADER_HEIGHT = 56.0
FOOTER_HEIGHT = 12.0

DEFAULT_WIDTH = 1200
MIN_WIDTH = 200
MAX_WIDTH = 20000

#: A label is dropped entirely below this share of the total, per the classic
#: flamegraph convention: the box is still drawn, it just carries no text.
MIN_LABEL_RATIO = 0.001
#: A box narrower than this many characters gets no label at all.
MIN_LABEL_CHARS = 3
#: A *truncated* label needs headroom for the "..." plus a few real
#: characters; below this, "t..." is noise and the box is left blank.
MIN_TRUNCATED_LABEL_CHARS = 6

BACKGROUND_COLOR = "#ffffff"
TEXT_COLOR = "#000000"
TEXT_COLOR_LIGHT = "#ffffff"
SUBTITLE_COLOR = "#555555"
FRAME_STROKE = "#ffffff"

ROOT_NAME = "all"


class FoldedStackError(ValueError):
    """A folded-stack line could not be parsed under the strict policy."""

    def __init__(self, reason: str, *, line_number: int, text: str) -> None:
        super().__init__(f"line {line_number}: {reason}: {text!r}")
        self.reason = reason
        self.line_number = line_number
        self.text = text


@dataclass(frozen=True)
class ParseIssue:
    """A rejected input line (lenient mode keeps these instead of raising)."""

    line_number: int
    reason: str
    text: str


@dataclass
class FoldedStacks:
    """Parsed folded stacks: accepted samples plus everything rejected."""

    samples: List[Tuple[Tuple[str, ...], float]] = field(default_factory=list)
    issues: List[ParseIssue] = field(default_factory=list)
    accepted_lines: int = 0

    @property
    def total(self) -> float:
        """Compensated (``math.fsum``) sum of every accepted sample value.

        Deliberately *not* bit-equal to ``build_tree(samples).total``, which
        accumulates naively (ten ``0.1`` samples: 1.0 here, 0.9999999999999999
        there).  ``build_tree`` is left naive on purpose -- switching it to a
        compensated sum would change the emitted SVG bytes for float-valued
        inputs and move the before/after diff baseline this tool exists for.
        All rendering uses ``FrameNode.total``; use this one for reporting.
        """
        return math.fsum(value for _, value in self.samples)


@dataclass
class FrameNode:
    """A node of the stack prefix tree.

    ``total`` is inclusive (self + descendants) and is what determines width.
    """

    name: str
    total: float = 0.0
    self_value: float = 0.0
    children: Dict[str, "FrameNode"] = field(default_factory=dict)

    def sorted_children(self) -> List["FrameNode"]:
        """Children in alphabetical order -- the only order this tool emits."""
        return [self.children[key] for key in sorted(self.children)]


@dataclass(frozen=True)
class Frame:
    """One laid-out rectangle, in SVG user units."""

    name: str
    depth: int
    x: float
    y: float
    width: float
    height: float
    value: float
    ratio: float


# --- parsing ---------------------------------------------------------------


#: Name given to samples a profiler could not attribute to any Python frame.
#: py-spy emits these as a bare value with no stack; for a training loop they are
#: mostly native code running with the GIL released (CUDA kernels, libtorch).
NO_PYTHON_FRAME = "(no python frame)"


def _looks_numeric(token: str) -> bool:
    """True when a lone token is a finite, non-negative number (a bare value)."""
    try:
        v = float(token)
    except ValueError:
        return False
    return math.isfinite(v) and v >= 0


def parse_folded(lines: Iterable[str], *, strict: bool = True) -> FoldedStacks:
    """Parse folded-stack text.

    Policy (explicit, and tested -- nothing is dropped quietly):

    * ``a;b;c 12.5``  -> accepted.  The value is the **last** whitespace
      separated token; everything before it is the stack.  Splitting from the
      right matters: real torch frames contain spaces
      (``<built-in method mm of type object>``).
    * blank / whitespace-only lines and ``#`` comments -> skipped, not an
      issue.  They carry no data.
    * a lone numeric token (``1286``, i.e. a value with **no** stack) ->
      accepted as a single frame named :data:`NO_PYTHON_FRAME`.  py-spy emits
      this for samples it cannot attribute to any Python frame -- native code
      holding no Python frame, which in a training loop is largely the CUDA
      kernels.  Rejecting it would shrink the denominator and silently inflate
      every other frame's percentage; naming it keeps the chart honest.
      A lone **non**-numeric token (``a;b``) is still a missing-value error.
    * missing value, non-numeric value, NaN/inf, **negative** value, or an
      empty frame inside the stack (``a;;b``) -> rejected.
      ``strict=True`` (the default) raises :class:`FoldedStackError` on the
      first one; ``strict=False`` records a :class:`ParseIssue` and continues.
    * a value of exactly ``0`` is accepted (torch emits these); it simply adds
      no width.
    * duplicate identical stacks are summed rather than overwriting.
    * surrounding whitespace is stripped from each frame name, so ``a; b;c``
      and ``a;b;c`` fold into one subtree.  Without this the emptiness check
      (which strips) and the stored name (which would not) disagree, and a
      stray leading space would sort a frame ahead of every letter -- silently
      breaking the alphabetical sibling order two charts are compared by.
    """
    result = FoldedStacks()
    for line_number, raw in enumerate(lines, start=1):
        text = raw.rstrip("\n").rstrip("\r")
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.rsplit(None, 1)
        if len(parts) == 2:
            stack_text, value_text = parts
        elif len(parts) == 1 and _looks_numeric(parts[0]):
            # py-spy emits a bare value with no stack (e.g. " 1286") for samples
            # it cannot attribute to any Python frame -- native code running with
            # the GIL released, which for a training loop is largely the CUDA
            # kernels themselves.  That is real signal, not corruption: dropping
            # it would silently shrink the denominator and inflate every other
            # frame's percentage.  Keep it under an explicit name instead.
            #
            # Gated on the token being numeric so that a genuine stack-without-a-
            # value ("a;b") stays a missing-value error rather than being
            # misread as an empty stack whose value happens to be "a;b".
            stack_text, value_text = NO_PYTHON_FRAME, parts[0]
        else:
            _reject(result, line_number, "missing value (expected 'stack <value>')", text, strict)
            continue

        try:
            value = float(value_text)
        except ValueError:
            _reject(result, line_number, f"value {value_text!r} is not a number", text, strict)
            continue
        if not math.isfinite(value):
            _reject(result, line_number, f"value {value_text!r} is not finite", text, strict)
            continue
        if value < 0:
            _reject(result, line_number, f"negative value {value_text!r}", text, strict)
            continue

        if not stack_text.strip():
            # Same case as the bare-value line above, reached when the stack is
            # whitespace rather than absent.  Name it rather than discard it.
            stack_text = NO_PYTHON_FRAME
        frames = [frame.strip() for frame in stack_text.split(";")]
        if any(not frame for frame in frames):
            _reject(result, line_number, "empty frame in stack", text, strict)
            continue

        result.samples.append((tuple(frames), value))
        result.accepted_lines += 1
    return result


def _reject(
    result: FoldedStacks, line_number: int, reason: str, text: str, strict: bool
) -> None:
    if strict:
        raise FoldedStackError(reason, line_number=line_number, text=text)
    result.issues.append(ParseIssue(line_number=line_number, reason=reason, text=text))


def build_tree(
    samples: Sequence[Tuple[Sequence[str], float]], *, root_name: str = ROOT_NAME
) -> FrameNode:
    """Fold samples into a prefix tree with inclusive ``total`` on every node.

    Iterative by construction: torch stacks are deep and recursion here would
    be a needless failure mode.
    """
    root = FrameNode(name=root_name)
    for path, value in samples:
        root.total += value
        node = root
        for frame in path:
            child = node.children.get(frame)
            if child is None:
                child = FrameNode(name=frame)
                node.children[frame] = child
            child.total += value
            node = child
        node.self_value += value
    return root


# --- layout ----------------------------------------------------------------


def layout(root: FrameNode, *, width: int = DEFAULT_WIDTH) -> Tuple[List[Frame], float]:
    """Place every node; return ``(frames, canvas_height)``.

    Every ``x`` is computed from a *global* cumulative value scale
    (``margin + cumulative / root.total * plot_width``) rather than by
    accumulating already-rounded child widths.  Accumulating rounded widths
    lets children drift past their parent's right edge by a fraction of a
    pixel -- invisible in a downscaled PNG, wrong in the SVG.

    Emission order is depth-first with siblings alphabetical, so the frame
    list (and therefore the SVG byte stream) is deterministic.
    """
    plot_width = float(width) - 2 * MARGIN_X
    total = root.total
    frames: List[Frame] = []
    max_depth = 0

    if total <= 0:
        canvas_height = HEADER_HEIGHT + ROW_HEIGHT + FOOTER_HEIGHT
        return frames, canvas_height

    # (node, depth, cumulative value to the left of this node)
    stack: List[Tuple[FrameNode, int, float]] = [(root, 0, 0.0)]
    while stack:
        node, depth, offset = stack.pop()
        max_depth = max(max_depth, depth)
        x = MARGIN_X + (offset / total) * plot_width
        w = (node.total / total) * plot_width
        frames.append(
            Frame(
                name=node.name,
                depth=depth,
                x=x,
                y=0.0,  # filled in below, once the canvas height is known
                width=w,
                height=FRAME_HEIGHT,
                value=node.total,
                ratio=node.total / total,
            )
        )
        # Children are laid out left-to-right alphabetically starting at the
        # parent's own left edge; the parent's self value is the uncovered
        # space that remains at its right, per flamegraph convention.
        child_offset = offset
        children = node.sorted_children()
        pending: List[Tuple[FrameNode, int, float]] = []
        for child in children:
            pending.append((child, depth + 1, child_offset))
            child_offset += child.total
        # Reverse before pushing so popping yields alphabetical order.
        stack.extend(reversed(pending))

    rows = max_depth + 1
    canvas_height = HEADER_HEIGHT + rows * ROW_HEIGHT + FOOTER_HEIGHT
    baseline = canvas_height - FOOTER_HEIGHT
    placed = [
        Frame(
            name=f.name,
            depth=f.depth,
            x=f.x,
            y=baseline - (f.depth + 1) * ROW_HEIGHT,
            width=f.width,
            height=f.height,
            value=f.value,
            ratio=f.ratio,
        )
        for f in frames
    ]
    return placed, canvas_height


# --- colour ----------------------------------------------------------------


def frame_color(name: str) -> str:
    """Stable warm-palette colour derived only from ``name``.

    SHA-256, not ``hash()``: CPython salts ``hash()`` per process, which would
    make two renders of the same file differ across runs and destroy the
    before/after diff this tool exists for.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    v = int.from_bytes(digest[:4], "big") / float(0xFFFFFFFF)
    r = 205 + int(50 * v)
    g = int(230 * v)
    b = int(55 * v)
    return f"#{r:02x}{g:02x}{b:02x}"


def text_color_for(fill: str) -> str:
    """Black or white label, whichever contrasts with ``fill``.

    The warm palette spans near-pure red (dark) to yellow (light), so a single
    fixed label colour is illegible at one end.  Same convention as
    ``analysis/viz_style.text_color_for``.  Still a pure function of the frame
    name, so determinism is untouched.

    ``fill`` must be ``#rrggbb``; this is public API, so a bad colour is a
    clear error rather than an ``IndexError`` from a bare slice.
    """
    if len(fill) != 7 or not fill.startswith("#"):
        raise ValueError(f"expected a '#rrggbb' colour, got {fill!r}")
    try:
        r, g, b = (int(fill[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    except ValueError as exc:
        raise ValueError(f"expected a '#rrggbb' colour, got {fill!r}") from exc

    def _linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)
    # Contrast ratio against white vs black, per WCAG relative luminance.
    return TEXT_COLOR if (luminance + 0.05) / 0.05 >= 1.05 / (luminance + 0.05) else TEXT_COLOR_LIGHT


# --- text ------------------------------------------------------------------


def _escape(text: str) -> str:
    """XML-escape text content and attribute values (all five entities)."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def fit_label(name: str, box_width: float, *, font_size: float = FONT_SIZE) -> Optional[str]:
    """Longest label that fits inside ``box_width``, or ``None`` if unreadable.

    Truncation uses an ASCII ``...`` (unambiguous under any encoding); the
    untruncated name always survives in the frame's ``<title>``.

    A short name that fits whole is always drawn; a name that must be cut is
    drawn only when at least three real characters survive the cut, so narrow
    boxes stay clean rather than filling with ``t...``.
    """
    usable = box_width - 2 * FRAME_TEXT_PAD
    char_width = font_size * CHAR_WIDTH_RATIO
    chars = int(usable / char_width)
    if chars < MIN_LABEL_CHARS:
        return None
    if len(name) <= chars:
        return name
    if chars < MIN_TRUNCATED_LABEL_CHARS:
        return None
    return name[: chars - 3] + "..."


def _format_value(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.3f}"


def metric_from_filename(path: str) -> Optional[str]:
    """``stacks_self_cpu_time_total.txt`` -> ``self_cpu_time_total``."""
    stem = Path(path).stem
    if stem.startswith("stacks_"):
        stem = stem[len("stacks_") :]
        return stem or None
    return None


def units_for_metric(metric: Optional[str]) -> Optional[str]:
    """torch reports ``*_time_total`` stack values in microseconds."""
    if metric and metric.endswith("time_total"):
        return "us"
    return None


# --- rendering -------------------------------------------------------------


def render_svg(
    root: FrameNode,
    *,
    title: str = "Flame chart",
    width: int = DEFAULT_WIDTH,
    metric: Optional[str] = None,
    units: Optional[str] = None,
    issues: Sequence[ParseIssue] = (),
) -> str:
    """Render a self-contained, deterministic SVG flame chart."""
    if width < MIN_WIDTH or width > MAX_WIDTH:
        raise ValueError(f"width must be within [{MIN_WIDTH}, {MAX_WIDTH}], got {width}")

    frames, canvas_height = layout(root, width=width)
    total = root.total

    subtitle_bits: List[str] = []
    if metric:
        subtitle_bits.append(metric)
    total_text = f"total {_format_value(total)}"
    if units:
        total_text += f" {units}"
    subtitle_bits.append(total_text)
    subtitle_bits.append(f"{len(frames)} frames")
    depth = max((f.depth for f in frames), default=0) + 1 if frames else 0
    subtitle_bits.append(f"depth {depth}")
    if issues:
        subtitle_bits.append(f"{len(issues)} malformed line(s) skipped")
    subtitle = "  |  ".join(subtitle_bits)

    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_num(width)}" '
        f'height="{_num(canvas_height)}" '
        f'viewBox="0 0 {_num(width)} {_num(canvas_height)}" '
        f'preserveAspectRatio="xMinYMin meet">'
    )
    # NOTE: do NOT put style="max-width:100%;height:auto" on the root <svg>.
    # It is the usual responsive-SVG idiom and works in browsers, but cairosvg
    # (our only available rasteriser) renders the whole canvas as a single flat
    # colour when height:auto is present -- verified on the real M3 capture:
    # with the attribute the PNG has exactly 1 distinct colour, without it 7523.
    # A chart that cannot be rasterised cannot be verified by looking at it, and
    # an unverifiable figure is worse than an ugly one.  width/height plus
    # viewBox already scale correctly wherever this is embedded.
    out.append(
        f'<rect x="0" y="0" width="{_num(width)}" height="{_num(canvas_height)}" '
        f'fill="{BACKGROUND_COLOR}"/>'
    )
    # The header is fitted through the same truncation path as a frame label.
    # Left unfitted, a long --title is centred at width/2 and clipped at both
    # ends by the viewBox: unreadable rather than degraded.  The full string is
    # preserved in a <title> tooltip, but only when it was actually cut, so a
    # chart with a normal-length header emits exactly the bytes it did before.
    title_label = fit_label(title, float(width) - 2 * MARGIN_X, font_size=TITLE_FONT_SIZE)
    title_text = (
        f'<text x="{_num(width / 2)}" y="26" text-anchor="middle" '
        f'font-family="{_escape(FONT_FAMILY)}" font-size="{_num(TITLE_FONT_SIZE)}" '
        f'font-weight="bold" fill="{TEXT_COLOR}">{_escape(title_label or "")}</text>'
    )
    if title_label == title:
        out.append(title_text)
    else:
        # Wrapped in a <g> rather than nesting <title> inside the <text>: a
        # child element would push the label into the parent's XML .tail and
        # readers that take ``text.text`` would see an empty header.
        out.append("<g>")
        out.append(f"<title>{_escape(title)}</title>")
        if title_label:
            out.append(title_text)
        out.append("</g>")
    out.append(
        f'<text x="{_num(width / 2)}" y="44" text-anchor="middle" '
        f'font-family="{_escape(FONT_FAMILY)}" font-size="{_num(SUBTITLE_FONT_SIZE)}" '
        f'fill="{SUBTITLE_COLOR}">{_escape(subtitle)}</text>'
    )

    if not frames:
        out.append(
            f'<text x="{_num(width / 2)}" y="{_num(HEADER_HEIGHT + ROW_HEIGHT)}" '
            f'text-anchor="middle" font-family="{_escape(FONT_FAMILY)}" '
            f'font-size="{_num(FONT_SIZE)}" fill="{SUBTITLE_COLOR}">'
            "(no samples)</text>"
        )
        out.append("</svg>")
        return "\n".join(out) + "\n"

    for frame in frames:
        pct = 100.0 * frame.ratio
        value_text = _format_value(frame.value)
        if units:
            value_text += f" {units}"
        tooltip = f"{frame.name} ({value_text}, {pct:.2f}%)"
        fill = frame_color(frame.name)
        out.append("<g>")
        out.append(f"<title>{_escape(tooltip)}</title>")
        out.append(
            f'<rect x="{_num(frame.x)}" y="{_num(frame.y)}" '
            f'width="{_num(frame.width)}" height="{_num(frame.height)}" '
            f'fill="{fill}" stroke="{FRAME_STROKE}" '
            f'stroke-width="0.5"/>'
        )
        label = None
        if frame.ratio >= MIN_LABEL_RATIO:
            # Try "name (pct%)" first -- the box width already encodes this same
            # ratio, so an on-frame number lets a reader rank same-looking
            # siblings without decoding relative widths by eye (fig-qa
            # 2026-07-29: two independent blind readers of a close multi-way
            # split could not do this from width alone). Only used whole: a
            # truncated "name (43.2..." carries no more signal than a bare
            # truncated name, so any cut falls back to the untruncated pct-free
            # path unchanged.
            annotated = f"{frame.name} ({pct:.1f}%)"
            fitted = fit_label(annotated, frame.width)
            label = fitted if fitted == annotated else fit_label(frame.name, frame.width)
        if label:
            out.append(
                f'<text x="{_num(frame.x + FRAME_TEXT_PAD)}" '
                f'y="{_num(frame.y + frame.height - 4.0)}" '
                f'font-family="{_escape(FONT_FAMILY)}" font-size="{_num(FONT_SIZE)}" '
                f'fill="{text_color_for(fill)}">{_escape(label)}</text>'
            )
        out.append("</g>")

    out.append("</svg>")
    return "\n".join(out) + "\n"


def _num(value: float) -> str:
    """Fixed-precision number formatting: identical bytes for identical input."""
    return f"{float(value):.3f}".rstrip("0").rstrip(".") or "0"


def render_folded_text(
    text: str,
    *,
    title: str = "Flame chart",
    width: int = DEFAULT_WIDTH,
    metric: Optional[str] = None,
    units: Optional[str] = None,
    strict: bool = True,
) -> str:
    """Convenience: folded-stack text in, SVG string out."""
    parsed = parse_folded(text.splitlines(), strict=strict)
    root = build_tree(parsed.samples)
    return render_svg(
        root,
        title=title,
        width=width,
        metric=metric,
        units=units or units_for_metric(metric),
        issues=parsed.issues,
    )


# --- CLI -------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flamegraph.py",
        description=(
            "Render a torch.profiler folded-stack file "
            "(prof.export_stacks) as a self-contained SVG flame chart."
        ),
    )
    parser.add_argument("folded", help="folded-stack file, or '-' for stdin")
    parser.add_argument("-o", "--output", default="-", help="output .svg (default: stdout)")
    parser.add_argument("--title", default=None, help="chart title (default: input filename)")
    parser.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH, help=f"canvas width in px (default {DEFAULT_WIDTH})"
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="metric name for the subtitle (default: inferred from the filename)",
    )
    parser.add_argument("--units", default=None, help="value units (default: inferred from metric)")
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="skip malformed lines instead of failing; counts go to stderr and into the subtitle",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.folded == "-":
        text = sys.stdin.read()
        source_name = "<stdin>"
    else:
        source = Path(args.folded)
        if not source.is_file():
            print(f"flamegraph: no such folded-stack file: {source}", file=sys.stderr)
            return 2
        text = source.read_text(encoding="utf-8", errors="replace")
        source_name = source.name

    metric = args.metric or (None if args.folded == "-" else metric_from_filename(args.folded))
    units = args.units or units_for_metric(metric)
    title = args.title or source_name

    try:
        parsed = parse_folded(text.splitlines(), strict=not args.lenient)
    except FoldedStackError as exc:
        print(
            f"flamegraph: malformed folded stack in {source_name}: {exc}\n"
            "            re-run with --lenient to skip bad lines (they will be counted).",
            file=sys.stderr,
        )
        return 2

    if parsed.issues:
        print(
            f"flamegraph: skipped {len(parsed.issues)} malformed line(s) in {source_name}",
            file=sys.stderr,
        )
        for issue in parsed.issues[:10]:
            print(f"  line {issue.line_number}: {issue.reason}", file=sys.stderr)
        if len(parsed.issues) > 10:
            print(f"  ... and {len(parsed.issues) - 10} more", file=sys.stderr)

    if not parsed.samples:
        print(
            f"flamegraph: {source_name} contained no usable samples; "
            "rendering an empty chart",
            file=sys.stderr,
        )

    root = build_tree(parsed.samples)
    svg = render_svg(
        root, title=title, width=args.width, metric=metric, units=units, issues=parsed.issues
    )

    if args.output == "-":
        sys.stdout.write(svg)
    else:
        out_path = Path(args.output)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(svg, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
