---
title: "Session Handoff — PERF-001/002: 계측 하네스 merge, flame chart 도구, 후보 P1-A 기각"
date: 2026-07-29
time: "07:25"
code: PFO
status: ready-for-resume
priority: P1
predecessor: null
next_action: "Read lab/reports/2026-07-29_perf002-p1a-refuted.md §5 → metrics 로깅 주기 후보를 사전등록(관측가능성 변경)한 뒤 A/B 측정"
tags: [handoff, session-end, perf, profiling]
---

# Session Handoff: 2026-07-29 → next session

> **TL;DR**: PERF-001 계측 하네스 + flame chart 도구를 PR #1로 merge했다. py-spy로 `unique()`가
> wall time의 37.6%임을 확인하고 후보 P1-A(sparse embedding `unique` 제거)를 구현했으나,
> **동등(bit-exact)하지만 빠르지 않아 기각**했다. 원인은 스텝이 GPU-bound라 sync 하나를 제거하면
> 대기가 다음 sync로 옮겨갈 뿐이라는 것 — per-update sync 제거 후보 한 부류가 통째로 정리됐다.
> 다음 목표는 host sync가 아니라 실제 GPU 작업량(metrics `torch.stack`, CUDA의 13.4%)이다.

---

## 1. Resume snippet (paste into next session)

PERF-001 계측 하네스(`utils/perf_benchmark.py`, `perf0` enqueue preset, disabled-path 회귀 테스트,
on/off 동등성 테스트)와 flame chart 도구(`utils/perf_profiler.py::export_stacks`,
`analysis/flamegraph.py`)는 PR #1으로 main에 merge됐다(`98e6c64`). 이후 브랜치
`perf-002-profiling-plan_claude`에 4커밋이 push돼 있으나 **PR 미생성**이다.

이번 사이클의 핵심 결과는 음성이다: flame chart와 py-spy가 `sparse_embedding.py:121`의
`unique(return_inverse=True)`를 wall time 37.6%로 지목했고, 이를 제거한 후보 P1-A는 CPU에서
bit-exact 동등성을 통과했지만 실측에서 이득이 0이었다(baseline 2.41/2.39 it/s vs candidate
2.37/2.38). py-spy로 세 상태를 연속 관측한 결과 `unique`는 0.00%로 사라졌으나 대기가
`new_tensor`(내가 실수로 넣은 H2D) → `metric_values.cpu().numpy()`로 옮겨갔을 뿐이고 wall time은
불변이었다. **스텝은 GPU-bound다.** 따라서 per-update sync를 하나만 제거하는 후보는 전부 무의미하다.

다음 후보는 metrics 로깅 주기다 — `torch.stack`은 대기가 아니라 CUDA의 13.4%를 쓰는 실제 GPU
작업이므로 줄이면 효과가 기대된다. 단 train 계열 per-step series가 바뀌므로 계획서의
observability 조항에 따라 **사전등록이 먼저**다. 그 다음은 커널 launch 축소(launch 15.3% +
triton heuristics 13.8% = 29% vs `mm` 30%)이며 계획서상 P3 고위험이다.

자원: aigpu0918 GPU 8개 전부 유휴, 과학 코호트는 unistml7로 이전됨. 우리 할당은 **4 5 6 7**.
큐는 `jobs=0 processing=1 done=55`이며 `0217` claim은 완주가 확인된 bookkeeping 결손(§3 Decision 1).

---

## 2. State of work

| Area | Change | File |
|---|---|---|
| 계측 수집기 | M2 타이밍 하네스(26컬럼 CSV·manifest·ledger·P0.0 캡처) | `utils/perf_benchmark.py` |
| 프로파일러 | `export_stacks` 추가 + PERF-DEV-18(dual export 충돌) 수정 | `utils/perf_profiler.py` |
| Flame chart | stdlib 전용 SVG 렌더러, py-spy interop, 프레임별 % 라벨 | `analysis/flamegraph.py` |
| 학습 배선 | span 호출부 + `puzzle_emb_dense_update` 롤백 스위치 | `pretrain.py` |
| Config | `perf_benchmark` 블록, `export_stacks`, `puzzle_emb_dense_update` (전부 기본 off) | `config/cfg_pretrain.yaml` |
| 후보 P1-A | dense scatter 분기(동등, **미승격**, 기본 off) | `models/sparse_embedding.py` |
| Launcher | `perf0` preset — M3 1 + M2 r1..r3, `PERF0_ALLOW_ENQUEUE` 펜스 | `scripts/sigma_enqueue.sh` |
| 테스트 | 계약·프로파일러·disabled-path·on/off 동등성·flamegraph·stacks (**378 passed**) | `tests/test_perf_*.py`, `tests/test_flamegraph.py`, `tests/test_profiler_stacks.py` |
| 비준 원장 | `PERF-DEV-00..19` + PI 결정 2건(26컬럼 개정, ledger→G1) | `lab/reports/2026-07-26_experiment-speed-action-plan.md` |
| 자원 계획 | GPU 예약 절차 + 동등성-by-test(T1/T2) | `lab/reports/2026-07-28_perf001-resource-plan.md` |
| M3 귀속 | flame chart 결과 + fig-qa 판정 반영 | `lab/reports/2026-07-28_perf001-m3-flamegraph-findings.md` |
| PERF-002 계획 | 다중 렌즈 설계 + 병렬화 경계 | `lab/reports/2026-07-29_perf002-profiling-plan.md` |
| **P1-A 기각** | 음성 결과와 병목 재해석 | `lab/reports/2026-07-29_perf002-p1a-refuted.md` |
| fig-qa 감사 | `_cuda.svg` REVISE-FIGURE 판정 | `lab/audits/2026-07-29_figqa-m3-flamecharts.md` |

**Git**: PR #1 merge → `98e6c64`. 브랜치 `perf-002-profiling-plan_claude`에 4커밋
(`a073024`, `09934ed`, `fd44c9e`, `fb94fe1`) push 완료, **PR 미생성**.

이번 세션에서 잡은 잠복 결함 3건: (i) 사전등록된 M3 명령이 dual trace export 충돌로 실행 불가였던
것, (ii) flame chart가 루트 CSS `height:auto` 때문에 백지로 렌더되던 것(색 1종 → 제거 후 7,523종),
(iii) py-spy의 빈-스택 라인을 거부해 분모가 17% 줄던 것. 모두 **실제 산출물을 렌더해서 보거나
실행해 본 결과** 드러났고 스모크 테스트로는 잡히지 않았다.

또한 07-28에 "queue supervisor가 죽었다"고 오진했던 것을 정정했다 — 원인은
`grep -E "queue_run|worker"`가 커널 `[kworker/*]`에 걸려 `head`에서 잘린 것이었다. 러너는 내내
살아 있었고, 그 전제 위에 내렸던 "수동 done 이동" 방침을 철회했다.

---

## 3. Pending decisions / open questions

### Decision 1 — 큐 claim `0217` 처리
**Source**: `scripts/queue/processing/0217_fig1_mlp_noz_iter_k8_s1.job.gpu5`
**사실**: 최종 체크포인트 `step_244100` 존재, 로그 07-28 15:10 정지, 대응 프로세스·GPU 점유 없음
= 완주한 job의 bookkeeping 결손. 다만 러너가 살아 있는데도 이동되지 않은 **원인이 미상**이다.
**Options**: (a) `done/`으로 수동 이동 (b) 원인 규명 후 이동 (c) 그대로 두고 drain
**Impact**: 07-28에 유사 claim을 오진한 전례가 있어 원인 불명 상태에서 반복하지 않고 보류했다.
큐 drain은 이것과 무관하게 가능하다.

### Decision 2 — `perf-002-profiling-plan_claude` PR 생성 여부
**Source**: 브랜치 4커밋 push 완료, PR 미생성
**Impact**: PR #1은 사용자가 merge했다. 이 브랜치도 같은 방식으로 리뷰할지, 아니면 후속 작업을
더 쌓은 뒤 한 번에 올릴지 결정 필요. PR 본문 초안 형식은 `lab/reports/2026-07-28_perf001-pr-body.md` 참조.
**주의**: `gh` CLI 미설치. GitHub REST API + `git credential fill`(VS Code askpass) 경로로 생성했다.

### Decision 3 — 다음 후보(metrics 로깅 주기)의 사전등록 범위
**Source**: `lab/reports/2026-07-26_experiment-speed-action-plan.md` Phase 1 후보 1
**Impact**: train 계열 per-step series가 바뀐다. 계획서는 이를 "허용된 observability 변경"으로
두되 **N과 aggregation schema를 후보 구현 전에 사전등록**하도록 요구한다. 무엇을 보존할지
(전 스텝 lossless local buffer 여부 포함) 결정이 선행돼야 한다.

---

## 4. Reference documents

- `lab/reports/2026-07-29_perf002-p1a-refuted.md` — 이번 사이클의 음성 결과와 §5 다음 목표. **먼저 읽을 것.**
- `lab/reports/2026-07-26_experiment-speed-action-plan.md` — 사전등록 실행계획 + `PERF-DEV-00..19` 비준 원장. 모든 이탈의 authority.
- `lab/reports/2026-07-29_perf002-profiling-plan.md` — 다중 렌즈(py-spy/line_profiler/sync-debug) 설계와 병렬화 경계. **"유휴 GPU 8개 ≠ 8배"** 표가 핵심.
- `lab/reports/2026-07-28_perf001-resource-plan.md` — GPU 예약 절차(R1–R5)와 동등성-by-test(T1/T2).
- `lab/reports/2026-07-28_perf001-m3-flamegraph-findings.md` — M3 귀속. **단, §3의 P1-A 처방은 기각됐다**(위 문서가 정정).
- `lab/audits/2026-07-29_figqa-m3-flamecharts.md` — `_cuda.svg` REVISE-FIGURE 판정. 재검증 전까지 시각 증거 인용 보류.

---

## 5. Next action

```
Read lab/reports/2026-07-29_perf002-p1a-refuted.md §5 →
metrics 로깅 주기 후보의 N·aggregation schema를 사전등록한 뒤,
GPU 5 단독에서 baseline/candidate A/B (각 3회, epochs=200) 측정
```

이유: P1-A 기각으로 host-side sync 제거 후보군이 정리됐고, 남은 것은 실제 GPU 작업량이다.
`torch.stack`(CUDA 13.4%)은 대기가 아니라 진짜 작업이라 줄이면 효과가 기대되는 유일한 저위험
후보다. 다만 관측가능성이 바뀌므로 측정보다 **사전등록이 먼저**다.
