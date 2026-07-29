---
id: PERF-001-RESOURCE
parent: PERF-001
status: planning — 실행 승인 전
date: 2026-07-28
scope: "PERF-001 M2/M3 실행을 위한 GPU 예약·큐 상호작용 계획 (측정 자원 계획; 최적화 후보 아님)"
---

# PERF-001 자원 계획 — GPU 예약과 큐 상호작용

이 문서는 [실행 계획서](2026-07-26_experiment-speed-action-plan.md)의 **P0.0 자원 확보 단계**를
실행 가능한 절차로 고정한다. 새로운 성능 가설이나 최적화 후보를 도입하지 않으며,
**속도 수치는 여전히 0건이고 승인된 M2 run 전에는 만들 수 없다.**

기호: (n) = 순열의 크기(본 문서 전 구간 n=10 고정), (k) = 합성 깊이.
"repeat" = 같은 조건의 반복 측정 r1..r3. "worker" = GPU 1개를 구조적으로 소유하는 실행 스레드.

## 1. 현재 자원 실측 (2026-07-28 ~11:50 KST, host `aigpu0918`)

**자원 정책 (PI 2026-07-28): 우리가 사용 가능한 GPU는 `4 5 6 7` 뿐이다.**
GPU 0·1은 외부 사용자 `ljsong7`, GPU 2·3은 우리 할당 밖이다. 아래 표의 "가용" 판단은
전부 이 정책 안에서만 이루어진다.

| GPU | 점유 | 소유자 / 정책 |
|---|---|---|
| 0, 1 | ~790 MiB, 68–71% | 외부 `ljsong7`. **사용 금지.** |
| 2, 3 | GPU 2 유휴 / GPU 3 33.5 GB | **할당 밖 — 사용 금지.** 러너 B(`GPUS="2 3"`)와 그 위에서 도는 `0221`은 이 정책 이전에 시작된 것이므로 완주만 시키고 이후 재사용하지 않는다. |
| 4, 7 | 33.5 GB, 100% | 우리 할당. fig1 잔여 run (`0217`, `0219`) |
| 5, 6 | 34 MiB, 0% | 우리 할당. **물리적으로 유휴 — 그러나 §2 때문에 "가용"이 아니다.** |

큐: `jobs=0 processing=3 done=53 failed=0`.

러너 2개가 살아 있다(등록 7일/6일 경과, disjoint GPU 집합 — `queue_run.sh:107-112`가 안전하다고
문서화한 패턴):

| 러너 | pid | 소유 GPU | worker 수 |
|---|---|---|---|
| A | 3104806 | env에 `GPUS` 없음 → 기본 `4 5 6 7` | 4 |
| B | 4010026 | `GPUS="2 3"` | 2 |

## 2. 핵심 제약 — 유휴 GPU ≠ 가용 GPU

**유휴 상태인 GPU 5·6은 이미 러너 A의 worker가 소유하고 있다.** `jobs/`가 비어 있어 그
worker들이 놀고 있을 뿐, 새 job 파일이 들어오면 **즉시 claim한다.**

따라서 M2 repeat 3개를 그냥 `jobs/`에 넣으면:

1. 유휴 worker들(GPU 5·6, 그리고 정책 밖이지만 러너 B의 GPU 2)이 **각각 하나씩 동시에 claim**한다.
2. → **PERF-DEV-17 위반**: 등록된 repeat 계약은 r1..r3가 *같은 예약 GPU*에서 돌 것을 요구한다.
3. → **타이밍 무효화**: 세 measured run이 같은 호스트에서 동시에 돌면 PCIe·호스트 CPU·
   데이터로더가 경합해 `data_wait_ms`와 `h2d_cuda_ms`가 서로를 오염시킨다. 측정 자체가 무의미해진다.

이것이 P0.0가 "유휴 GPU 확보"만으로 열리지 않는 이유다. 필요한 것은 **배타적 소유**다.

## 3. 추가 금지사항 — 세 번째 러너를 띄우지 않는다

`queue_run.sh:113-120`의 crash recovery는 **liveness(PID) 검사 없이 claim 파일만 보고**
`$GPUS` 범위의 claim을 `jobs/`로 되돌린다. 지금 `processing/`의 `0217_...job.gpu5`는 **살아 있는**
job이므로, `GPUS="5"`처럼 좁혀 띄워도 그 claim을 회수해 **중복 실행**을 만든다
(`checkpoints/Sigma_k_new/<run_name>/`·`Sigma_k_new` wandb run name 충돌).

또한 GPU 5·6은 러너 A의 집합 안에 있으므로, 같은 GPU를 소유하는 두 러너가 생기면 worker 2개가
같은 장치에서 경합한다.

**결론: 어떤 `GPUS` 범위로도 새 러너를 기동하지 않는다.** `status`/`--dry-run`은 안전하다(§5).

## 4. 절차 — 배타적 예약을 얻는 유일한 안전 경로

전제: 계획서가 M2를 **queue job body로만** 실행하도록 요구하므로, 큐를 우회한 직접 실행은
선택지가 아니다.

| 단계 | 행동 | 완료 조건 |
|---|---|---|
| R1 | fig1 잔여 3 run(`0217`,`0219`,`0221`) 완주 대기 | `processing/` 0개, `done/` 56, `failed/` 0. 러너가 자동 이동시킨다(수동 개입 금지). |
| R2 | 두 러너를 **drain**: `touch scripts/queue/stop` | 큐가 빈 상태에서 worker들이 스스로 종료(`queue_run.sh:148-152`). `ps -eo pid,cmd \| grep -F "queue_run.sh"`로 0개 확인. **`kill` 금지** — 진행 중 job을 죽인다. |
| R3 | stop 파일 제거 후 **단일 러너를 예약 GPU 하나로 기동**: `GPUS="5" scripts/queue_run.sh` | worker가 정확히 1개. 이 시점에는 `.gpu5` claim이 남아 있지 않아야 하므로 R1 완료가 선행 조건이다. |
| R4 | P0.0 자원 매니페스트 캡처 (§5) | `manifests/resource_<UTC>.json` 생성 |
| R5 | `scripts/sigma_enqueue.sh perf0` 로 M2 3 repeat + M3 job 생성 | 실제 enqueue는 `PERF0_ALLOW_ENQUEUE=1` 펜스를 넘어야 한다 |

**R3가 PERF-DEV-17을 자동으로 만족시킨다**: worker가 1개뿐이므로 r1·r2·r3가 FIFO로 **직렬**
실행되고, 전부 같은 물리 GPU에 떨어진다. 별도의 강제 장치가 필요 없다.

예약 GPU는 **GPU 5**로 한다 — 우리 할당(`4 5 6 7`) 안이고 2026-07-28 PI 승인분이다.
`GPUS="5"`는 할당 정책을 위반하지 않으며, 나머지 4·6·7은 이 기간 동안 비워 둔다
(다른 job을 넣지 않는다).

### 잔여 교란 요인 (제거 불가, 기록)

`ljsong7`의 작업이 GPU 0·1에서 계속 돌면 **호스트 수준 자원**(CPU 코어, 메모리 대역폭, PCIe)을
공유한다. 이는 예약으로 제거할 수 없다. 세 repeat의 분산(`CV`)과 repeat 간 CI가 이 교란을
흡수하며, manifest에 캡처 시점의 전체 GPU 점유가 남으므로 사후 판별이 가능하다.
**repeat 3회를 서로 다른 시간대에 돌리지 않는다** — 교란 조건을 repeat 간에 바꾸지 않기 위함이다.

## 5. P0.0 캡처 — `queue_run.sh status`는 안전하다

`capture_resource_manifest`(`utils/perf_benchmark.py:1436`)는 세 read-only probe를 캡처 시점에만
shell out한다: `queue_run.sh status`, `sigma_enqueue.sh --dry-run perf0`, `nvidia-smi --query-gpu`.

`status` 분기는 `queue_run.sh:100`의 `exit 0`으로 종료하며 **crash recovery 루프(`:113`)에
도달하지 않는다.** 따라서 살아 있는 job이 있어도 안전하다. (모듈 docstring은 이보다 보수적으로
경고하는데, 그 경고는 과하다 — 위험한 것은 **worker 기동**이지 `status`가 아니다.)

```bash
rtk uv run python -c "
from utils.perf_benchmark import capture_resource_manifest
print(capture_resource_manifest(repo_root='.'))"
```

출력: `reports/figures/2026-07-26_experiment-speed-profiling/manifests/resource_<UTC>.json`
(UTC, hostname, worktree git SHA + dirty flag, 세 probe의 stdout).

**커밋 상태 주의**: manifest는 `git.dirty`를 기록한다. 세 repeat에 깨끗한 provenance를 남기려면
M2 실행 전에 관련 변경이 커밋돼 있어야 한다(현재 PERF-001 코드는
`perf-001-m2-harness_claude` 브랜치에 커밋 완료).

## 5.5 동등성은 테스트로 보장한다 (PI 요구 2026-07-28)

계획서의 G1(frozen-state gate)은 지금까지 **수동 게이트**로만 기술돼 있었다. PI 지시에 따라
**"계측을 켜도/후보를 적용해도 실험 결과가 동등하다"를 자동 테스트가 보장**하도록 격상한다.
아래 T1/T2가 그 요구를 두 층으로 나눈 것이다. 근거: 사람이 한 번 확인하고 지나가는 게이트는
회귀를 못 막는다 — 동등성은 매 변경마다 재확인돼야 한다.

### T1 — 계측 on/off 동등성 (GPU 불필요, 상시 CI)

`tests/test_pretrain_perf_disabled_path.py`가 이미 **disabled** 경로를 검증한다. 빠진 것은 그
쌍(pair)이다: **`perf_benchmark.enabled=True`로 돌린 학습이 `False`와 동일한 결과를 내는가.**
이것이 "계측이 실험을 바꾸지 않는다"의 직접 진술이며, 기존 fake model/optimizer 하네스로
GPU 없이 검증 가능하다.

같은 seed·같은 고정 batch 시퀀스로 enabled/disabled 두 번 돌려 다음이 **정확히 일치**해야 한다:
model 호출 인자·횟수, optimizer step/zero_grad 횟수, 적용된 `lr` 시퀀스, 반환 `metrics` 전체,
batch 값의 `.cuda()` 호출 횟수, `train_state.step`/`carry` 진행. 부동소수 비교는 tolerance가
아니라 **bit-exact**여야 한다 — 계측은 산술을 건드리지 않으므로 근사 일치는 이미 결함이다.

추가로: 켠 상태에서 span 컨텍스트가 batch·metrics 텐서를 **변형하지 않음**을 확인한다.

### T2 — 후보 동등성 = G1 frozen-state (GPU 필요, `skipif` 게이트)

Phase 1 후보를 적용했을 때의 동등성. 고정 batch를 기록해 두고 candidate-off/on에서 비교한다:
sample index hash, logits/loss/predictions, gradients와 step 후 상태, EMA 텐서, z-probe 산출물,
checkpoint key와 재로딩. logging-only 후보는 **exact equality**, numerical/compiler 후보는
candidate 결과를 보기 **전에** baseline 반복으로 고정한 tolerance를 쓴다.

T2는 CUDA가 없으면 `skip`한다. **`skip`은 통과가 아니다** — 이 사실을 게이트 리포트에
`EVIDENCE-PENDING`으로 남기고, 승인된 GPU에서 한 번은 실제로 돌린다. 결과는
`equivalence_ledger.csv`(PERF-DEV-10에 따라 G1 산출물)에 기록한다.

### 지금 실행 가능 여부

T1은 **GPU 없이 지금 구현 가능**하며 P0.0/M2를 기다리지 않는다. T2는 후보가 존재해야 하므로
Phase 1 진입 시점에 구현한다. 단, T2의 **하네스**(고정 batch 기록·비교 유틸)는 후보보다 먼저
만들어 둘 수 있고, `append_equivalence_ledger`가 이미 그 소비처다.

### T1 결과 (2026-07-28, 독립 게이트 PASS)

`tests/test_perf_equivalence.py` (9 test). 전체 스위트 **369 passed**.
**test-only 추가** — `pretrain.py`·`utils/perf_benchmark.py`·config는 수정되지 않았다.
즉 테스트를 통과시키려 하네스를 굽히지 않았다는 것 자체가 증거다.

같은 고정 batch 시퀀스를 동일 초기 상태에서 **실제 `pretrain.train_batch`** 로 두 번
(enabled/disabled) 재생하고 전체 trace를 `==` 로 비교한다. 비교 대상: model 호출 순번·carry·
batch signature·`return_keys`, `initial_carry` 호출, optimizer step/zero_grad 횟수, `param_groups`에
실제로 기록된 **lr 시퀀스**(스케줄이 상수가 아님을 별도 단언 — 뒤섞인 스케줄이 우연히 같아질 수
없게), 반환 metrics 전 키·값, `train_state.step`/carry 진행, batch 값별 `.cuda()` 횟수, loader
1회 소비와 객체 동일성, 모델이 예외를 던질 때의 예외 동일성과 롤백. 부동소수는
`(type, dtype, float.hex())`로 정규화 — `-0.0`과 `0.0`을 구분하고 `NaN==NaN`을 성립시키므로
단순 `==`보다 강하다. 동등성 단언에 `pytest.approx`는 쓰이지 않는다.
window는 warmup/measured/post-window/`total_steps` early-return을 모두 가로지른다.

**비공허성은 독립 게이트가 mutation 주입으로 증명했다**(구현자 주장에 의존하지 않음):
계측이 batch 값마다 `.cuda()`를 한 번 더 부르게 하자 **9개 중 4개가 red**(`.cuda()` 횟수를
명시적으로 단언하지 않는 테스트까지 포함 — 전체-trace 비교가 작동한다는 뜻),
`_CudaSpan.__exit__`가 예외를 삼키게 하자 의도한 **1개가 red**.

**경계 (과대주장 금지).** 확립된 것은 **제어 흐름과 bookkeeping의 동등성**이지 실제 커널
수치의 동등성이 아니다. 또한 `train_batch`는 실물이지만 이를 감싸는 학습 루프는
`pretrain.launch()`의 span 호출부를 **재현한 mirror**다. 알려진 불일치 2건을 기록한다:
(a) mirror는 loop-level `wall_span("metrics_wandb")`/`cuda_span("ema")`를 무조건 여는 반면
프로덕션은 각각 `RANK==0 and metrics is not None`, `config.ema`로 가드한다 — `total_steps`
early-return 업데이트에서 프로덕션이 열지 않을 span을 mirror가 연다;
(b) mirror는 dict를, 프로덕션은 `(set_name, batch, global_batch_size)` 3-tuple을 순회한다
(`iter_batches`는 항목을 그대로 yield하므로 무해하나 loader 계약이 다르다).
따라서 **루프 수준 주장은 재현물에 근거**하며, 실물 `launch()`에 대한 것이 아니다.
bf16 커널·`torch.compile`·실제 CUDA event·DDP는 T2와 승인된 GPU run의 몫으로 남는다.

## 6. 이 계획이 열어주지 않는 것

- **M2 emitter가 존재한다는 사실은 실행 허가가 아니다.** `PERF0_ALLOW_ENQUEUE=1` 펜스가
  유일한 방지턱이며, 계획서 §4.1의 "no production enqueue in this registration"이 그 근거다.
- CUDA 통합은 여전히 **evidence-pending**이다. 모든 CUDA 진입점이 테스트에서 fake로 주입되므로,
  실제 event pair 해석·event pool 크기·비동기 stream 생존은 M2 run에서만 증명된다.
- 어떤 speedup도 기록하지 않는다. M2 artifact가 없으면 공통 `delta`조차 확정할 수 없다.

## 7. 의존 관계 요약

```
T1 계측 on/off 동등성 테스트  ─── GPU 불필요, 지금 착수 가능, 다른 어떤 것도 기다리지 않음
                                    │
R1 fig1 잔여 3 run 완주             │
  └─> R2 두 러너 drain (touch stop) │
        └─> R3 단일 러너 GPU 5 기동 │  ← PERF-DEV-17 자동 충족
              ├─> R4 P0.0 resource manifest
              └─> R5 perf0 enqueue (M3 + M2 r1..r3, 직렬)
                    └─> P0.2 귀속 ──> Phase 1 후보 1개
                                        └─> T2 G1 frozen-state 동등성 (GPU, ledger 기록)
```

즉 **T1은 R1–R5와 병렬**이며, 자원을 전혀 쓰지 않으므로 fig1 완주를 기다릴 이유가 없다.
