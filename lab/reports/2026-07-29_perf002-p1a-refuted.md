---
id: PERF-002-P1A
parent: PERF-001
status: candidate REFUTED as a speedup — 승격 안 함
date: 2026-07-29
scope: "후보 P1-A(sparse embedding unique 제거) 실측 결과와 그로부터 나온 병목 재해석"
---

# P1-A는 기각한다 — 그리고 그 실패가 진짜 병목을 지목한다

**결론 먼저: P1-A는 동등하지만 빠르지 않다. 승격하지 않는다.**
`puzzle_emb_dense_update`는 기본 `false`로 남긴다.

## 1. 무엇을 했나

`models/sparse_embedding.py`의 `all_ids.unique(return_inverse=True)`를 제거하고,
고정 크기 `(P, D)` scatter-add + presence mask로 대체했다(스위치 `puzzle_emb_dense_update`).

동등성은 CPU에서 **bit-exact**로 확인했다 — `P=1`(σ^k), `P=8` 부분 등장, `P=4` 전원 등장
세 경우 모두 `maxdiff = 0.000e+00`. 함정 케이스(**batch에 등장하지 않은 row에 weight decay가
새지 않는가**)도 통과했다.

## 2. 측정: 이득이 없다

같은 GPU(5), 순차 실행, epochs=200(488 step), 다른 부하 없음.

| 반복 | baseline (`dense=False`) | candidate (`dense=True`) |
|---|---|---|
| 1 | 2.41 it/s | 2.37 it/s |
| 2 | 2.39 it/s | 2.38 it/s |

**개선 없음.** 계획서의 공통 `delta`(최소 3%) 근처에도 가지 못한다. 오히려 노이즈 안에서
미세하게 느리다.

## 3. 왜 안 되는가 — stall이 옮겨다닌다

py-spy(계측 없음, 100 Hz)로 세 상태를 연속 관측한 결과가 결정적이다.

| 상태 | 최상위 leaf (wall time) | 비중 |
|---|---|---|
| baseline | `_unique_impl (torch/functional.py:946)` | 37.59% |
| candidate v1 | `_sparse_emb_signsgd_dist:148` — 내가 넣은 `weights.new_tensor(...)` | 42.88% |
| candidate v2 | `train_batch (pretrain.py:369)` = `metric_values.cpu().numpy()` | 38.54% |

v1에서 `unique`는 **0.00%** 로 완전히 사라졌다. 목표는 확실히 제거됐다. 그런데 대기는
사라지지 않고 **다음 동기화 지점으로 이동**했다. v1의 `new_tensor`는 내가 실수로 넣은
host→device 복사였고(그것도 sync다), 그것을 스칼라 연산으로 없애자 이번엔 PERF-DEV-07이
이미 기록해 둔 **sync #1(`metric_values.cpu().numpy()`)** 이 그 자리를 차지했다.

세 상태 모두 wall time은 ~2.4 it/s로 같다.

**해석: 이 학습 스텝은 GPU-bound다.** host는 어차피 첫 번째 동기화 지점에서 GPU 큐가
비기를 기다린다. *어느* 지점에서 기다리는지는 총 시간에 영향을 주지 않는다. 따라서
**per-update sync를 하나만 제거하는 어떤 후보도 이득을 낼 수 없다.**

이는 py-spy 해석 규율을 한 단계 강화한다: py-spy는 host가 **어디서 막히는가**를 정확히
보여주지만, 그 지점이 **임계 경로의 원인**이라는 뜻은 아니다. GPU-bound 상태에서 host의
대기 위치는 증상이지 원인이 아니다.

## 4. 07-28 귀속의 정정

M3 flame chart 보고서는 `unique`를 "주 발견"으로 올리고 P1-A를 후보로 제안했다.
그 **귀속은 옳았고 처방은 틀렸다.** 정정 사항:

- `_unique_impl`이 wall time의 37.6%라는 관측은 재현된다(사실).
- 그러나 그것은 "제거하면 37.6%를 되찾는다"를 함의하지 않는다(추론 오류).
- M3가 준 `unique` CUDA self time 14.12%도 실제 커널 비용이 아니라 대기가 섞인 값으로
  읽어야 한다 — 제거해도 총 시간이 안 줄었기 때문이다.

## 5. 이 실패가 지목하는 다음 목표

host-side sync 제거가 무의미하다면 남는 것은 **GPU 작업량 자체**다. M3의 CUDA self time
분포가 그대로 우선순위가 된다:

| 후보 | 근거 | 성격 |
|---|---|---|
| **metrics 로깅 주기** | `torch.stack`이 CUDA의 **13.35%**. 매 update의 실제 GPU 작업이다 | 계획서 Phase 1 후보 1. 관측 가능성 변경으로 사전등록 필요 |
| 커널 launch 축소 | `_launch_kernel` 15.34% + `triton_heuristics._run` 13.79% = **29%**, 실제 수학 `mm`은 30% | 계획서 P3(고위험). CUDA graph/fusion 영역 |

즉 **다음 후보는 P1-A가 아니라 metrics 로깅 주기**이며, 이는 sync를 없애서가 아니라
**GPU 작업을 실제로 줄이기 때문에** 효과가 기대된다. 다만 이것도 측정 전에는 주장하지 않는다.

## 6. 코드 처분

`puzzle_emb_dense_update` 스위치와 dense 분기는 **남긴다**(기본 off). 근거:
(a) 동등성이 bit-exact로 확인됐고, (b) "모든 per-update sync를 동시에 제거"하는 후보를
언젠가 시험한다면 이 분기가 그 구성 요소이며, (c) 삭제하면 이 음성 결과를 재현할 수 없다.
**단독으로는 승격 대상이 아니다.**

## 7. 방법론 기록

이번 사이클은 계획서의 게이트가 실제로 작동한 사례다. flame chart가 강한 신호를 줬고,
후보는 동등성까지 통과했지만, **측정이 기각했다.** 사전등록이 "M2 없이 speedup 주장 금지"를
강제하지 않았다면 37.6%라는 수치를 근거로 승격됐을 것이다.
