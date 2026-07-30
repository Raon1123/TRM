---
id: FIGQA-PERF001-M3-FLAME
parent: PERF-001-M3-FLAME
date: 2026-07-29
scope: "/fig-qa on the two M3 flame chart SVGs cited in lab/reports/2026-07-28_perf001-m3-flamegraph-findings.md"
---

# fig-qa — M3 flame charts (self_cpu / self_cuda)

## Target

- Figure A: `lab/figures/2026-07-28_perf001-m3-flamegraph/m3_tf_z_iter_k6_s1_cpu.svg` (`self_cpu_time_total`)
- Figure B: `lab/figures/2026-07-28_perf001-m3-flamegraph/m3_tf_z_iter_k6_s1_cuda.svg` (`self_cuda_time_total`)
- Owning report: `lab/reports/2026-07-28_perf001-m3-flamegraph-findings.md` (F1/F2/F3)
- No FIGSPEC pre-existed (report predates this skill's use on PERF-001); FIGSPEC drafted ad-hoc for
  this run from the report's §2/§3/§4 numbers and sealed before spawning interpreters.
- `problem_context` was written ad-hoc (flame-chart / torch-profiler mechanics) — the skill's
  canonical `sigma-k-min` paragraph does not cover profiling artifacts. Candidate for a new named
  canonical context if PERF-001 keeps producing flame charts.

## Renderer check (prerequisite, before packet build)

Both SVGs rasterized via `uvx cairosvg -W 2400` (no `cairosvg`/`rsvg-convert` in the project venv;
`uvx` pulled it standalone). Distinct-color counts: A=5653, B=7523 — B matches exactly the "7523
colours, correctly rendered" figure the renderer-fix commit (c4ee929) reported, confirming the
`max-width:100%` SVG-attribute fix is holding and neither chart is silently flattened.

## Panel

3 interpreters per figure, single-use, fresh contexts, no repo/tool access beyond the one neutral-named
PNG (+ caption for B2/B3): B1 = Sonnet image-only, B2 = Sonnet image+caption (shipping condition),
B3 = Haiku image+caption (rushed-reader proxy). Grader = fresh-context Sonnet with sealed FIGSPEC +
all 6 readings, no image access.

## Scores (0–2)

| Interpreter | R1 msg recovery | R2 encoding | R3 misreading | R4 self-contained | R5 overclaim |
|---|---|---|---|---|---|
| A/B1 (image only) | 2 | 2 | 2 | 2 | 2 |
| A/B2 (+caption) | 2 | 2 | 2 | 2 | 2 |
| A/B3 (Haiku) | 1 | 1 | 2 | 2 | 2 |
| B/B1 (image only) | 1 | 2 | 2 | 2 | 2 |
| B/B2 (+caption) | 1 | 2 | 2 | 2 | 2 |
| B/B3 (Haiku) | 0 | 0 | 1 | 0 | 0 |

## Figure A (self_cpu) — **PASS**

B1, with **no caption**, independently traced `optimizer.py → sparse_embedding.py(62):_step →
_sparse_emb_signsgd_dist → unique → <built-in method _unique2>` as the widest non-ancestor branch —
i.e. F1's headline fact (unique2 dominates self-CPU) survives image-only reading. B2 recovered the
same path with the caption. B3 (Haiku) *saw* the same evidence ("dominant red rectangle labeled
unique2... spans a large horizontal region") but buried it under generic bullets instead of naming it
as the main claim — a rushed-reader synthesis lapse, not a figure defect — and separately hallucinated
a truncated label (`pretrain.py(573):train_batched` for the actual `pretrain.py(373):train_batch_profiled`
— wrong line number *and* function name, not a plausible truncation guess).

The intended message's cross-figure clause ("more than the actual math shown in the companion CUDA
chart") is unrecoverable from Figure A alone by construction (no reader saw both images) — not charged
against any reader; flagged as a caption-writing lesson (CG3 below), not a figure-A defect.

No overclaims: all six readings (both figures) correctly kept self-time distinct from wall-clock,
declined to read the chart as a speedup number, and did not generalize past the single k=6/s=1 capture.

## Figure B (self_cuda) — **REVISE-FIGURE**

**Neither Sonnet reader (B1 or B2) recovered the intended three-way-comparable-split message**, with or
without the caption. Both noticed several top-region blocks look close in width and are "hard to rank
by eye," but neither named all four contributing leaves (`mm`, `_launch_kernel`/`triton_heuristics`,
`unique2`, `torch.stack`) or synthesized "non-math bookkeeping approaches matmul's share." This is the
single most load-bearing finding of this audit: the miss recurred identically across both information
conditions, which is evidence the gap is in the **figure's resolving power** (no per-frame numeric
labels, and the interesting leaves are narrow/truncated among many similarly-styled top-of-stack
blocks) rather than in either individual reader.

B3 (Haiku) failed severely and, unlike Figure A, the failure was actively harmful: it invented three
percentage-banded zones ("~40–50%", "~20–30%", "~20–30%") with no numeric labels in the image to derive
them from, and — despite explicitly listing "self-time excludes children" as a known misreading trap in
its own §5 — violated that exact rule in its own main claim by treating the near-full-width bottom
ancestor spine (near-zero *own* self-time by the chart's semantics) as a meaningful "~20–30% zone."
Figure B's genuinely close multi-way split is exactly the shape of chart where a rushed reader
manufactures false precision; Figure A's one overwhelmingly-wide branch is hard to miss even carelessly.

**Verdict is REVISE-FIGURE, not REVISE-CAPTION**, because the caption could not have fixed this without
becoming the finding itself (spelling out all four leaf names + magnitudes) — the more durable fix is
in the figure: either per-frame percentage annotations for top-N leaves, or the renderer highlighting
named frames of interest.

## Cross-condition diagnosis

- **B1 vs B2 (what the caption buys):** little, on both figures. Figure A's finding survives without a
  caption at all; Figure B's caption gives totals/frame-counts but no per-frame numbers or named-leaf
  hints, so it doesn't move either reader past "these look similar, can't rank them."
- **B2 vs B3 (rushed reading):** on Figure A, rushed reading degrades *synthesis* while raw perception
  stays intact, and introduces confident label hallucination. On Figure B, degradation is categorically
  worse — fabricated statistics and a self-contradicted misreading-trap violation — because Figure B's
  actual structure (a close multi-way split) offers no single visually-obvious answer for a rushed
  reader to correctly default to.

## Misreading-trap inventory

1. Color = name-hash misread as severity/heat scale — flagged by all 6, well-defended.
2. Alphabetical sibling order misread as time/execution order — flagged by all 6.
3. Frame width/depth misread as code complexity/call count — flagged by all 6.
4. Repeated same-looking labels at different depths misread as duplicate calls — flagged by 5/6.
5. **Self-time vs. cumulative-time confusion** (wide ancestor frames have ~0 own self-time despite
   width) — correctly self-taught by B1/B2 on Figure B, then violated by B3-FigB in its own main claim
   despite listing the rule. Highest-value trap to caption explicitly (CG4).
6. On-figure "(contended host)" title text misread as a quantified measurement — proactively flagged
   by B1-FigB; not observed as an actual failure in any reading.
7. **Reader-introduced (not figure-invited):** fabricating precise percentages absent numeric labels
   (B3-FigB) — the figure gives zero grounds for this; purely a rushed-reader failure.
8. **Reader-introduced:** hallucinating truncated-label text with high confidence (B3-FigA).

## Caption-gap list (writeback)

- **CG1 (Fig B, high priority):** no per-frame percentages or named top leaves in the caption —
  proximate cause of both Sonnet readers missing the three-way split. Fix: numeric top-N annotations on
  the figure itself, or explicit named callouts in the caption (compare `mm`, `_launch_kernel` +
  `triton_heuristics`, `unique2`, `torch.stack`).
- **CG2 (Fig A):** background never mentions `record_function`/profiler-overhead semantics, so KF2 was
  untestable from the image+background alone (graded generously, not counted against any reader). One
  added sentence would make it testable.
- **CG3 (both):** neither caption states the two charts are companions from the *same* capture, so no
  reader had a signal to look for `unique2` recurring in both. B1 (image-only) structurally cannot know
  this regardless of caption; B2/B3 had the caption and still had no such cue.
- **CG4 (Fig B, high priority):** no caption warning that wide bottom-of-stack ancestor frames carry
  near-zero self-time despite visual width — the exact trap B3 fell into despite naming it as a risk.
- **CG5 (Fig B, structural):** absence of numeric per-frame labels makes "which of several similar-width
  blocks is bigger" unreliable by design. Works fine for Fig A (one branch overwhelmingly dominates);
  breaks exactly the claim Fig B exists to make. Primary argument for REVISE-FIGURE over REVISE-CAPTION.

## Round 1 route (superseded — see Round 3 for final verdict)

- Figure A: **PASS** — citable as evidence for F1 as-is.
- Figure B: **REVISE-FIGURE** — do not cite as visual evidence for F2/F3's mm/launch/unique/stack
  comparison until re-rendered with per-frame magnitude annotations (or equivalent) and re-run through
  a fresh B panel (loop cap 2, per skill). The report's F2/F3 **numeric** claims (§2 table: mm 30.09%,
  `_launch_kernel` 15.34%, `triton_heuristics._run` 13.79%, `unique2` 14.12%, `stack` 13.35%) are drawn
  directly from the profiler's aggregate table, not from the flame-chart SVG, and are unaffected by this
  verdict — only the *chart-as-visual-evidence* citation was gated (now resolved, Round 3).

---

## Round 2 (revision 1 of 2) — Figure B only

Fix applied to `analysis/flamegraph.py`: on-frame labels now try `"name (pct%)"` first, reserving room
for the `(pct%)` suffix *before* truncating the name (previously the whole annotated string was tried
and dropped entirely if it didn't fit, which meant the percentage only ever appeared on short-named
frames — useless here, since the flagged frames carry long torch repr names). Both `_cpu.svg` and
`_cuda.svg` re-rendered and re-copied to both `reports/figures/` and `lab/figures/`; renderer check
re-run (cairosvg, non-flattened). New tests added to `tests/test_flamegraph.py` for the suffix-reservation
behaviour; one pre-existing test updated (`test_long_names_are_truncated...` — the ellipsis no longer
sits at the very end of the string once a suffix follows it).

Fresh 3-agent B panel (B1/B2/B3, all newly spawned — prior interpreters are contaminated) + fresh grader,
against the same sealed FIGSPEC as Round 1.

**Verdict: REVISE-FIGURE again.** R2 (encoding) went to 2/2/2 across all three readers — the suffix fix
fully solved the original problem (nobody failed to decode color/order/inclusive-share semantics this
round). But R1 (message recovery) still failed for 2 of 3 readers: **both Sonnet readers organized their
MAIN CLAIM around a repeated `57.4%` figure that appeared identically on ~8 consecutive stacked
pass-through/ancestor frames**, instead of recognizing the real four-way leaf split one level deeper in
the stack. Grader's diagnosis: this is a *new, second-order figure defect* introduced by the same
mechanism that fixed R2 — uniform percentage-labeling of every frame, including pure single-child
pass-through links, makes an identical, prominent numeral repeat across many stacked boxes, and a reader
building a "main claim" naturally reaches for the most repeated/largest labeled number on the chart. I
independently checked the grader's `torch.stack`-not-locatable concern against the raw folded-stack file
(`stacks_self_cuda_time_total.txt`): the leaf genuinely exists (`<built-in method stack of type object at
0x...>`, 13.35%) — not a data gap. The real cause is a *third* defect: truncation-from-the-right keeps the
`<built-in method ` prefix and cuts before the operation name (`stack`, `mm`, `_unique2`) ever appears,
since torch's own repr puts the boilerplate first and the identifying word in the middle.

Grader's proposed fixes for Round 3 (both applied, see below): (a) suppress the `(pct%)` suffix on pure
pass-through frames (single child, zero self-time) — only branch points and leaves keep one; (b) strip
the `<built-in method X of type object at 0x...>` / `<built-in function X>` boilerplate from the on-frame
*label only* (tooltip and colour hash keep the raw name) before truncating, so the operation name survives
the character budget instead of the hex address eating it.

---

## Round 3 (revision 2 of 2 — final, loop cap reached) — Figure B only

Both Round-2 fixes implemented in `analysis/flamegraph.py`: `Frame` gained an `is_passthrough` field
(`len(node.children) == 1 and node.self_value == 0`, computed in `layout()`); pass-through frames render
name-only (no suffix). New `display_name()` strips the builtin-repr boilerplate via regex (tolerant of
both the space- and underscore-separated conventions — torch's own `export_stacks` writes internal spaces
as underscores in real captures, but `parse_folded` also accepts literal spaces on input, so both had to
match). 8 new/updated tests in `tests/test_flamegraph.py` (63 total, all passing). Re-rendered, re-copied,
renderer-checked (non-flattened) as before. Effect on the real chart: the repeated `57.4%` now appears
**once** (down from ~8), and `stack`, `_launch_kernel`, `_unique2`, and both `mm` occurrences all render
cleanly with their percentage, none truncated below legibility.

Fresh 3-agent B panel + fresh grader, same sealed FIGSPEC (with an added `known_note_for_grader` flagging
a suspected caption/data mismatch — see below).

**Score table (0–2):**

| Interpreter | R1 msg recovery | R2 encoding | R3 misreading | R4 self-contained | R5 overclaim | Total |
|---|---|---|---|---|---|---|
| B1 (image only) | 0 | 2 | 2 | 2 | 0 | 6/10 |
| B2 (image+caption, shipping) | 2 | 2 | 2 | 2 | 2 | **10/10** |
| B3 (Haiku, rushed) | 1 | 2 | 1 | 0 | 1 | 5/10 |

**Did the Round 2 fixes work?** Yes, both, cleanly. No reader reported the repeated-57.4%-on-8-frames
confusion from Round 1; `stack` was correctly and unprompted located by B1 (no caption) and confirmed by
B2. R2 (encoding) stayed 2/2/2 — nothing regressed.

**Residual issue (not a defect the Round-2 fix class could address):** B1 (no-caption condition) still
headlined the largest single *branch-point* percentage (`57.4%`, the torchinductor-generated-kernel
subtree — legitimately a branch point, correctly kept its label) as "by far the largest single labeled
contributor," ranking it above the four-way leaf split rather than alongside it — a narrower recurrence
of the same KF4 trap (one honest node instead of eight repeated ghosts, but the same underlying
inferential error: an untutored reader gravitates to the single biggest on-frame number as "the answer,"
without a caption steering them). This is a structural property of inclusive-percentage flame charts, not
a boilerplate/repeat-suppression bug — suppressing this node's label too would hide genuinely correct
branch-point information, so it is not a targeted fix the way Round 2's two changes were.

**Genuine, independently-found caption defect (not a chart defect):** B2 flagged that the caption's
"reading the widest ... branches from left to right: kernel-launch/heuristics, matmul, unique, stack"
does not match the actual layout — alphabetical sibling order puts a `stack` frame **leftmost**, not last,
directly contradicting the caption's claimed order. This is a real caption bug, independently rediscovered
by a blind reader with no coaching, and is fixed here (see below) — it does not require touching the
render and does not consume a chart-revision round.

**B3's failures (mm misread as "rm"; could not locate the cleanly-labeled `stack`) are attributed to
rushed-reader conditions, not the figure** — both labels were correctly and easily read by both Sonnet
conditions in the same round.

## Final verdict — Figure B: **PASS (shipping condition)**, with a fixed caption and one escalated note

- The actual shipping condition (image + caption) scores 10/10 and cleanly recovers both KF1 and KF2
  with no KF4 violation. **Citable as visual evidence for F2/F3 as of this round, with the caption below.**
- **Caption fixed** (was: "...from left to right: kernel-launch/heuristics overhead, the matmul kernel
  itself, the sparse-embedding optimizer's per-step `unique` call, and the metrics-logging `torch.stack`
  call..."; the "from left to right" ordering claim is dropped as factually backwards against the actual
  alphabetical layout):
  > Self-CUDA-time flame chart for one M3 profiler capture (k=6, seed=1, contended host — other GPU jobs
  > were running on the same host during this capture). Four leaves — kernel-launch/heuristics overhead,
  > the matmul kernel itself, the sparse-embedding optimizer's per-step `unique` call, and the
  > metrics-logging `torch.stack` call — each occupy a comparable, non-trivial share of total self-CUDA
  > time (siblings are ordered alphabetically, not by size or position), unlike the companion
  > self-CPU-time chart for the same capture, where one call dominates overwhelmingly.
- **Escalated, not fixed (loop cap reached — 2 revision rounds used):** without a caption (e.g. slide
  reuse, social excerpt, or any future context where this chart is shown alone), a careful reader may
  still headline the largest *branch-point* subtree percentage instead of the four-way leaf split. Two
  unexecuted options for a future round, for the user to weigh rather than something applied here: (i)
  visually de-emphasize branch-point ancestor percentages (e.g. lighter weight, parentheses) so only leaf
  percentages read as "headline" numbers; or (ii) accept caption-dependency as a permanent constraint of
  this chart type and always ship it captioned, never bare.
- Figure A (self_cpu): unchanged from Round 1, **PASS** stands (the Round 2/3 renderer changes are
  additive-only — more information, never less — and were not expected to and did not require a fresh B
  round; Figure A was regenerated with the same renderer purely to keep both companion charts in sync with
  current code, not re-graded).

## Cost

Round 1: 6 blind-interpreter agents (Sonnet ×4, Haiku ×2) + 1 fresh-context Sonnet grader.
Round 2: 3 blind-interpreter agents (Sonnet ×2, Haiku ×1) + 1 fresh-context Sonnet grader, Figure B only.
Round 3: 3 blind-interpreter agents (Sonnet ×2, Haiku ×1) + 1 fresh-context Sonnet grader, Figure B only.
Total: 12 blind-interpreter agents + 3 graders across 3 rounds. No repo writes by any interpreter or
grader in any round (read-only packets).
