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

## Route

- Figure A: **PASS** — citable as evidence for F1 as-is.
- Figure B: **REVISE-FIGURE** — do not cite as visual evidence for F2/F3's mm/launch/unique/stack
  comparison until re-rendered with per-frame magnitude annotations (or equivalent) and re-run through
  a fresh B panel (loop cap 2, per skill). The report's F2/F3 **numeric** claims (§2 table: mm 30.09%,
  `_launch_kernel` 15.34%, `triton_heuristics._run` 13.79%, `unique2` 14.12%, `stack` 13.35%) are drawn
  directly from the profiler's aggregate table, not from the flame-chart SVG, and are unaffected by this
  verdict — only the *chart-as-visual-evidence* citation is gated.

## Cost

6 blind-interpreter agents (Sonnet x4, Haiku x2) + 1 fresh-context Sonnet grader. No repo writes by any
interpreter or grader (read-only packets).
