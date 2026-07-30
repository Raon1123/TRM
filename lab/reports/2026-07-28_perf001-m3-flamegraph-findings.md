---
id: PERF-001-M3-FLAME
parent: PERF-001
status: attribution complete — 후보 제안, 승격 전
date: 2026-07-28
scope: "M3 profiler capture의 flame chart 귀속 결과와 Phase 1 후보 제안"
figqa: cpu=PASS, cuda=PASS (2 revision rounds, 2026-07-29 → lab/audits/2026-07-29_figqa-m3-flamecharts.md)
---

# PERF-001 M3 flame chart — 귀속 결과와 후보 제안

**속도 수치는 이 문서에 없다.** flame chart는 시간이 *어디로* 가는지를 보여주는 귀속 도구이며,
speedup은 M2 baseline(공통 `delta`) 없이 주장하지 않는다. 이 문서는 병목을 특정하고 Phase 1
후보 1개를 제안하는 데서 멈춘다.

기호: (n)=순열의 크기(n=10 고정), (k)=합성 깊이, (P)=`num_puzzle_identifiers`,
(D)=`puzzle_emb_ndim`, (N)=batch 내 sample 수.

## 1. 무엇을 돌렸나

사전등록 §4.2.1의 M3 명령에 **PERF-DEV-18 정정**(`export_tensorboard_trace=False`)과
`export_stacks=True`를 적용해 실행. `CUDA_VISIBLE_DEVICES=5`로 직접 pin.

| 항목 | 값 |
|---|---|
| 결과 | exit 0, 48 step / 69초 |
| GPU | 5 (할당 4–7 안, PI 승인분) |
| 산출물 | `reports/figures/2026-07-26_experiment-speed-profiling/profiles/m3_tf_z_iter_k6_s1/session_0000/capture_00/` |
| folded stacks | CPU 52,344줄 / CUDA 32,092줄, `stacks_manifest.json` 양쪽 `written` |
| flame chart | `reports/figures/2026-07-26_experiment-speed-profiling/flamegraphs/m3_tf_z_iter_k6_s1_{cpu,cuda}.svg` |

**큐 미사용 (기록된 이탈).** 계획서는 M2를 queue job body로만 실행하도록 요구한다. M3는
직접 실행했다. 이유: (a) `PERF0_ALLOW_ENQUEUE` 펜스는 미검토 enqueue를 막으려고 내가 세운
것이고 사용자 승인 없이 스스로 우회하지 않는다, (b) 큐에 넣으면 유휴 worker 중 어느 것이
집을지 비결정적인데 직접 pin은 GPU를 확정한다, (c) `scripts/queue/`를 전혀 건드리지 않는다.
M3는 사전등록상 performance-only diagnostic이며 과학적 증거가 아니다.

## 2. 귀속 (self time 상위)

`self_cuda_time_total` 총 20,238,091 us / `self_cpu_time_total` 총 7,343,650 us.

| leaf | CUDA | CPU |
|---|---|---|
| `mm` (matmul) | **30.09%** | 1.04% |
| `_launch_kernel` | 15.34% | — |
| **`_unique2`** | **14.12%** | **70.13%** |
| `triton_heuristics.py(1338):_run` | 13.79% | 1.06% |
| **`stack`** | **13.35%** | 6.43% |
| `_record_function_enter_new` | — | 19.17% |
| `models/ema.py(16):_update` | 0.14% | 0.15% |

**`_record_function_enter_new` 19.17%는 프로파일러 자신의 오버헤드다**(`train_batch_profiled`의
`profiler.record(...)` span). 귀속에서 제외해야 하며, M2 비계측 측정에는 존재하지 않는다.

## 3. F1 — `unique`가 상수를 매 step 재계산한다 (주 발견)

호출 지점: [`models/sparse_embedding.py:121`](../../models/sparse_embedding.py)
```python
grad_ids, inv = all_ids.unique(return_inverse=True)
```
`_sparse_emb_signsgd_dist` ← `CastedSparseEmbeddingSignSGD_Distributed.step` ← optimizer step.
이 optimizer는 [`pretrain.py:180,192`](../../pretrain.py)에서 정상 등록되므로
**프로파일 여부와 무관하게 매 update 실행된다.**

**결정적 사실**: σ^k 데이터셋은 `num_puzzle_identifiers: 1` 이고
`all__puzzle_identifiers.npy`의 고유값은 `[0]` 하나뿐이다(전 5000개 항목).
즉 `all_ids`는 **길이 N=2048의 0 벡터**이고, `unique(return_inverse=True)`는 매 step
GPU sort를 돌려 `grad_ids=[0]`, `inv=zeros(2048)`이라는 **상수**를 재발견한다.

CPU self time 70%가 여기 몰린 이유는 단순 연산량이 아니다. `unique`는 출력 크기를 알아야 하므로
**device→host 동기화**를 강제한다. 즉 이것은 PERF-DEV-07이 기록한 per-update 암묵 sync의
**세 번째이자 가장 큰 사례**다(앞의 둘: `metric_values.cpu().numpy()`, `non_blocking=False` H2D).

### 후보 P1-A (제안, 미구현)

`unique`를 제거하고 **고정 크기 (P×D) scatter-add + presence mask**로 대체한다:

```
counts   = zeros(P);      counts.scatter_add_(0, all_ids, ones)
grad_full= zeros(P, D);   grad_full.scatter_add_(0, ids.expand(-1,D), all_weights_grad)
mask     = (counts > 0).unsqueeze(-1)
# weight decay와 sign 갱신을 mask로 게이팅 — 등장하지 않은 row는 건드리지 않는다
```

**동등성 논거**: 원본은 등장한 row에만 `mul_(1-lr*wd)`와 `add_(sign(grad))`를 적용한다.
mask가 그 조건을 정확히 재현한다. **weight decay를 전 row에 적용하는 순진한 full-table 갱신은
동등하지 않다** — 등장하지 않은 embedding을 감쇠시켜 버린다. 이 함정이 이 후보의 핵심 위험이다.

**P=1(σ^k)에서는 bit-exact가 기대된다**: `inv`가 zeros이므로 원본의 scatter-add도 1×D 버퍼에
대한 동일 연산이다. **P>1 일반 데이터셋에서는 numerical 후보**로 다뤄야 한다(합산 순서 차이가
`sign()` 부호를 0 근처에서 뒤집을 수 있다). 비용은 O(U·D) → O(P·D)로 바뀌므로 P가 큰
데이터셋에서는 이득이 사라질 수 있다 — **σ^k 한정 이득으로 예약**한다.

## 4. F2 — metrics `torch.stack`이 CUDA의 13.35%

`stack`의 프로젝트 호출자는 `pretrain.py(373)`, 즉 매 update의
`torch.stack([metrics[k] for k in metric_keys])`다. 이는 계획서 Phase 1 후보 1(train logging
cadence)이 겨냥한 경로와 정확히 일치하며, §2 표의 수치(실측 근거)가 그 후보를 뒷받침한다.

**fig-qa 게이트 (2026-07-29, 2 revision rounds 후 PASS):** 최초 판정은 REVISE-FIGURE —
독립 blind reader 2명(caption 유무 무관) 모두 mm/launch-overhead/unique/stack **3~4갈래
비등 분할**이라는 의도된 메시지를 그림만으로 복구하지 못했다(프레임별 수치 라벨 부재가
원인). renderer에 프레임별 `(pct%)` 라벨을 추가(round 1)했으나 pass-through 체인 프레임에
동일 숫자가 반복 출력되어 오히려 그 숫자에 주의가 쏠리는 2차 결함이 드러났고, `torch.stack`
프레임은 truncation이 `<built-in method ` 접두부를 남기고 정작 식별어를 잘라내 caption을 쥐고도
찾지 못하는 3번째 결함도 발견됨(round 2 재검증). round 3에서 pass-through 프레임 라벨 억제 +
builtin-repr 접두부 제거(진짜 이름만 truncate) 적용 후 caption-informed 조건이 10/10으로 PASS.
caption의 "left to right" 순서 주장이 실제 alphabetical 배치(`stack`이 최좌측)와 어긋난다는
결함도 이 과정에서 독립적으로 발견되어 caption을 수정. no-caption 조건에서는 여전히 최대
branch-point 서브트리 %(57.4%)를 4갈래 분할보다 앞세워 읽는 잔여 위험이 있음 — caption 없이
단독 인용(슬라이드 등)될 경우 재검토 필요, 별도 renderer 작업으로 escalate(loop cap 2 도달).
위 §2 표 수치 자체는 profiler 집계 테이블에서 직접 뽑은 것이라 이 판정과 무관하다.
상세: `lab/audits/2026-07-29_figqa-m3-flamecharts.md`.

## 5. F3 — launch/heuristics 오버헤드가 CUDA의 29%

`_launch_kernel` 15.34% + `triton_heuristics._run` 13.79%. 실제 수학인 `mm`은 30.09%다.
즉 **커널 실행 오버헤드가 주 연산과 같은 크기**다. 이는 batch가 작거나 커널이 잘게 쪼개져
있음을 시사하나, 이 문서는 진단을 여기서 멈춘다 — 조치는 CUDA graph/컴파일 영역이고
계획서상 P3(고위험)이다.

## 6. 이 결과의 한계 (과대해석 금지)

1. **경합 호스트.** 측정 시 GPU 3·4·7에서 fig1 3개가 돌고 있었다. host CPU·PCIe 경합이
   host-side 수치를 부풀린다. 단 F1은 *어떤 코드가 실행되는가*에 대한 발견이며 경합이
   `unique` 호출을 만들어내지 않는다 — F1은 경합에 무관하게 성립한다.
2. **`self_cpu_time_total`은 sync 대기를 포함한다.** "CPU의 70%"는 "wall time의 70%"가 아니다.
   실제 wall 비중은 M2 비계측 측정에서만 나온다.
3. **프로파일된 경로**(`train_batch_profiled`)를 측정했고 `record_function` 오버헤드가 CPU의
   19%다. 다만 sparse-emb optimizer는 두 경로 공통이다.
4. **k=6, seed=1, 단일 capture.** 반복 없음. F1은 코드 구조상 k에 무관하나, 비율은 아니다.

## 7. 다음 단계 (승격 아님)

P1-A는 **제안 상태**다. 승격 조건은 계획서대로다: M2 baseline 3 repeat로 공통 `delta` 확정 →
G1 frozen-state 동등성(T2) → G2 paired canary. 지금 M2를 돌리면 경합으로 baseline이 오염되므로
[자원 계획](2026-07-28_perf001-resource-plan.md) §4의 R1–R5(코호트 완주 → 러너 drain →
예약 GPU 단일 worker)를 먼저 통과해야 한다.

코호트 완주 후 **경합 없는 상태에서 M3를 재측정**해 이 귀속을 재확인할 것을 권한다.
