---
id: PERF-002-PLAN
parent: PERF-001
status: planning — 실행 승인 전
date: 2026-07-29
scope: "유휴 자원(aigpu0918 전체)을 쓴 최적화 실행 계획. 다중 프로파일러 렌즈(py-spy / line_profiler / sync-debug / torch.profiler / M2) 배치와 병렬화 경계"
---

# PERF-002 — 유휴 자원 최적화 계획과 다중 렌즈 프로파일링

PERF-001이 계측 하네스와 첫 flame chart 귀속을 남겼다(PR #1 merged, `98e6c64`).
이 문서는 **비워진 aigpu0918 전체를 어떻게 쓸지**와 **py-spy / line_profiler를 기존
torch.profiler 위에 어떻게 얹을지**를 고정한다. **속도 수치는 여전히 0건이다.**

기호: (P)=`num_puzzle_identifiers`, (D)=`puzzle_emb_ndim`, (N)=batch sample 수,
(delta)=계획서가 정의한 공통 속도 게이트 임계값.

## 1. 자원 현황 (2026-07-29 06:00 KST, host `aigpu0918`)

| 항목 | 상태 |
|---|---|
| GPU 0–7 | **전부 유휴** (0%, 34–45 MiB). 외부 사용자 `ljsong7` 작업도 종료됨 |
| 과학 코호트 | `lab/monitoring/ACTIVE_COHORT.md` 4차 개정(07-28)으로 주 관찰 대상이 **unistml7**로 이전. fig1+module ablation 완료 |
| 큐 | `jobs=0 processing=1 done=55 failed=0` |
| 러너 | 2개 생존(A: 기본 `GPUS="4 5 6 7"` worker 4, B: `GPUS="2 3"` worker 2) = 유휴 worker 6 |

**할당 정책은 GPU `4 5 6 7`을 유지한다**(PI 2026-07-28). 0–3이 지금 비어 있다는 사실은
정책 변경 사유가 아니다. 정책을 넓히려면 PI 지시가 필요하다.

**미해결 claim 1건**: `0217_fig1_mlp_noz_iter_k8_s1.job.gpu5`. 판정 근거 —
최종 체크포인트 `step_244100` 존재, 로그가 07-28 15:10에 정지, 대응 프로세스 없음, GPU 전무.
**완주한 job의 bookkeeping 결손**이다. 러너가 살아 있는데도 이동되지 않은 이유는 미상이므로
원인 추정 없이 사실만 기록한다. 처리는 §4 R1에서 다룬다.

## 2. 병렬화 경계 — 유휴 GPU를 언제 쓸 수 있고 언제 못 쓰나

이 계획의 핵심 제약이다. **8개가 비었다고 8배로 돌릴 수 있는 게 아니다.**

| 작업 종류 | 병렬 가능? | 이유 |
|---|---|---|
| **M2 타이밍 측정** | **불가.** 반드시 단독·직렬 | 동시 실행은 host CPU·PCIe·데이터로더를 경합시켜 `data_wait_ms`·`h2d_cuda_ms`를 오염시킨다. r1..r3는 PERF-DEV-17에 따라 같은 예약 GPU에서 직렬 |
| **py-spy 샘플링** | **불가(수치용)** | wall-clock을 재므로 경합이 분포를 바꾼다. 헤드라인 수치는 조용한 호스트에서 |
| **line_profiler** | 불가(수치용) | 위와 동일 + 자체 오버헤드가 큼 |
| **sync-debug 열거** | 가능 | 어떤 sync가 *존재하는가*는 경합과 무관 |
| **G1 frozen-state 동등성** | **가능, 4 cell 병렬** | 결정론적 텐서 비교다. 정확성 판정이지 타이밍이 아니다 |
| **후보 구현·단위 테스트** | 가능(CPU) | GPU 불필요 |
| **G2 canary의 정확성 부분** | **가능, 4 cell 병렬** | 단, **속도 게이트 수치는 G2에서 뽑지 않는다** — 직렬 paired M2에서 뽑는다 |

즉 유휴 자원의 실질 이득은 **타이밍 단계가 아니라 정확성 단계(G1/G2 4-cell)** 에서 나온다.
타이밍은 어차피 1 GPU 직렬이므로, 나머지 3개는 그동안 **비워 둬야** 한다.

## 3. 다중 렌즈 프로파일링 설계 (PI 제안 반영)

PI가 개인적으로 쓰는 **line_profiler / py-spy**를 기존 torch.profiler 위에 얹는다.
이는 도구 취향 문제가 아니라, **M3가 답하지 못한 두 질문을 정확히 겨냥**한다:

1. M3의 `record_function` 오버헤드가 CPU self time의 **19.2%**였다 — 계측이 대상을 교란한다.
2. `self_cpu_time_total`은 sync 대기를 포함한다 — **"CPU의 70%"가 "wall의 70%"가 아니다.**
   그런데 `delta` 게이트가 실제로 쓰는 화폐는 wall time이다.

| 렌즈 | 답하는 질문 | 교란 | 코드 변경 |
|---|---|---|---|
| **py-spy** | **wall time이 어디로 가는가** | ~1–2% (외부 샘플링) | **없음** |
| torch.profiler (M3) | 어떤 CUDA 커널/op인가 | 큼 (19.2%) | 있음(span) |
| **line_profiler** | 의심 함수의 **어느 줄**이 막는가 | 매우 큼 (결정론적 추적) | `@profile` 데코레이션 |
| `set_sync_debug_mode("warn")` | 암묵 sync가 **정확히 어디에** 있는가 | 무시 가능 | 없음(런타임 호출) |
| M2 benchmark | 게이트가 쓰는 수치 | 없음 | 하네스 |

### 3.1 py-spy — 검증된 통합 경로

가용성 확인(2026-07-29): `/usr/local/bin/py-spy` **0.4.0 설치돼 있음**,
`/proc/sys/kernel/yama/ptrace_scope = 0` 이라 동일 사용자 프로세스에 attach 가능.

**결정적 사실**: `py-spy record --format raw`가 내는 것은 `frame;frame;frame <count>` —
**`analysis/flamegraph.py`의 입력 형식과 동일하다.** 실측 검증 완료(샘플 161개 → 정상 SVG 렌더).
따라서 py-spy flame chart와 torch.profiler flame chart가 **같은 렌더러·같은 알파벳순 정렬·
같은 이름-해시 색상**을 공유하고, 전/후 비교가 바이트 diff로 성립한다.

```bash
# 대상은 실제 python 프로세스여야 한다. `uv run` 래퍼 PID를 주면
# "Failed to find python version from target process"로 실패한다 (실측).
PID=$(pgrep -f "pretrain.py" | head -1)
py-spy record --pid $PID --format raw --rate 100 --duration 120 \
  --output <out>.folded            # --native 를 붙이면 C/C++ 프레임까지
uv run python analysis/flamegraph.py <out>.folded -o <out>.svg \
  --metric wall --units samples
```

`--native`가 특히 중요하다: `cudaStreamSynchronize`가 프레임으로 직접 보이므로
**PERF-DEV-07의 암묵 sync 주장을 파이썬 스택이 아니라 네이티브 스택으로 확증**할 수 있다.

부수 효과: `py-spy dump --pid`는 멈춘 run의 즉시 스냅샷을 준다 — queue 모니터링에도 쓸 수 있다.

### 3.2 line_profiler — 쓰되, 읽는 법을 고정한다

대상은 두 곳뿐이다: `models/sparse_embedding.py::_sparse_emb_signsgd_dist` (F1)와
`pretrain.py`의 metrics 블록 (F2).

**해석 규율 (중요).** CUDA는 비동기다. 따라서 line_profiler의 줄별 시간은
**"그 줄이 얼마나 비싼가"가 아니라 "그 줄에서 얼마나 막히는가"** 를 잰다. 그대로 읽으면
비용을 오귀속한다. 두 가지 중 하나로만 쓴다:

- **(a) sync 지점 탐지용** — 있는 그대로 읽는다. 막히는 줄 = sync 지점. F1 가설에는 이쪽이 맞다.
- **(b) 비용 측정용** — 구간을 `torch.cuda.synchronize()`로 감싸고 읽는다. 이때는 측정 자체가
  파이프라인을 직렬화하므로 **절대 M2 아티팩트 run에 쓰지 않는다.**

`@profile` 데코레이터는 `kernprof` 실행 시에만 정의되므로, 영구 코드에 남기지 않는다.
진단용 임시 패치로 적용하고 되돌린다(커밋하지 않음).

### 3.3 sync 열거 — 가장 싸고 가장 확정적

```python
torch.cuda.set_sync_debug_mode("warn")   # 한 update 감싸고 원복
```
PERF-DEV-07이 기록한 2건(`metric_values.cpu().numpy()`, `non_blocking=False` H2D)과
F1이 추가한 1건(`unique`의 출력 크기 sync)이 **전부인지** 여기서 확정한다.
지금까지의 목록은 "발견된 것"이지 "전부"가 아니다.

## 4. 실행 순서

병렬 표기: `∥` = 동시 실행 가능, `→` = 선행 필요.

```
R1 큐 정합 (0217 처리) → R2 러너 drain → R3 예약 GPU 5 단일 worker
   │
   ├─ D1 sync 열거 (set_sync_debug_mode)            ─┐ 진단, 짧음, 직렬
   ├─ S1 py-spy wall-clock 샘플링 (+--native)        ─┤ (조용한 호스트 필요)
   ├─ L1 line_profiler on sparse_embedding (모드 a)  ─┤
   └─ M3' 경합 없는 상태에서 M3 재측정               ─┘
        → M2 baseline r1..r3 (직렬, 단독) → delta 확정
             → C1 후보 P1-A 구현 + 단위 테스트 (CPU) ∥ 아무거나
                  → G1 frozen-state 4 cell  ∥ 병렬(GPU 4,5,6,7)
                       → paired M2 (직렬, 단독) = 속도 게이트
                            → G2 canary 4 cell ∥ 병렬(정확성만)
                                 → 승격 여부 판정
```

**유휴 자원이 실제로 절약해 주는 구간은 G1과 G2의 4-cell뿐이다.** 그 두 곳에서 4배,
나머지는 직렬이다. 이를 과장해 "8 GPU로 최적화 가속"이라고 쓰지 않는다.

### R1–R3 (자원 준비)

- **R1**: `0217` claim 처리. 완주가 확인됐으므로 `done/`으로 옮기는 것이 사실에 부합하나,
  **07-28에 내가 "supervisor 사망"을 오진하고 수동 이동을 권했다가 철회한 전례**가 있다.
  이번에는 러너가 살아 있는데도 이동되지 않은 원인이 미상이므로, **PI 확인 후 처리**한다.
  drain(R2)은 이 claim과 무관하게 진행 가능하다.
- **R2**: `touch scripts/queue/stop` → 두 러너의 worker 6개가 큐 빈 상태에서 자진 종료.
  `kill` 금지(진행 중 job을 죽인다). `ps -eo pid,cmd | grep -F "queue_run.sh"`로 0 확인.
  **주의**: `grep -E "queue_run|worker"`는 커널 `[kworker/*]`에 걸려 오판을 부른다(07-28 실측).
- **R3**: stop 제거 후 `GPUS="5" scripts/queue_run.sh`. worker가 1개이므로 r1..r3가 FIFO로
  직렬화되어 PERF-DEV-17이 구조적으로 충족된다.

### 예상 산출물

| 단계 | 산출물 |
|---|---|
| D1 | sync 목록 (완전성 판정 포함) |
| S1 | `flamegraphs/pyspy_<tag>_{wall,native}.svg` + folded 원본 |
| L1 | 줄별 표 + 해석 모드 명시 |
| M3' | 경합 없는 재귀속, 07-28 경합본과 대조 |
| M2 | `steady_state.csv` ×3 + `manifest.json` ×3 → `delta` |

## 5. 판정 기준 (선언, 데이터 보기 전)

- F1이 **wall time 기준으로도** 유의미해야 P1-A가 후보로 남는다. py-spy에서 `unique`
  관련 프레임이 유의미한 비중을 못 내면, CPU-self-time 70%는 **sync 대기의 착시**였다는
  뜻이고 후보는 강등된다. 이 조건을 데이터 보기 전에 고정한다.
- P1-A 승격은 계획서 그대로: G1 통과 + paired M2에서 median이 `delta` 이상 개선 + p95 미악화.
- P=1에서 bit-exact가 깨지면 즉시 rollback하고 원인 규명 전까지 재시도하지 않는다.

## 6. 이 계획이 하지 않는 것

- 과학 코호트 자원을 선점하지 않는다. unistml7로 이전됐지만 이 호스트에 과학 job이
  들어오면 **최적화 일정이 양보**한다.
- GPU 0–3을 쓰지 않는다(할당 정책).
- 어떤 speedup도 M2 전에 기록하지 않는다.
