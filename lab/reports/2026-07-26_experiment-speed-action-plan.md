---
id: PERF-001-PLAN
parent: PERF-001
status: active-planning
date: 2026-07-26
scope: "Ordered execution backlog for semantics-preserving TRM baseline performance work"
---

# PERF-001 실행 계획 — 속도 측정 → 단일 후보 → 동등성 게이트

이 문서는 [PERF-001 사전등록](2026-07-26_experiment-speed-profiling.md)의 실행 순서를 고정한다. 새로운
가설·성능 수치·과학적 판정은 추가하지 않는다. 현재 **속도 향상 수치는 주장하지 않는다**.

## 현재 상태와 근거 인벤토리

| 상태 | 근거 | 다음에 할 수 있는 일 / 한계 |
|---|---|---|
| DONE | [사전등록](2026-07-26_experiment-speed-profiling.md) §1–6은 canonical `trm`, n=10, B=2048, H3/L6/L2, EMA, `probe/test_exact` 및 G1–G3를 고정한다. | 이 계획은 그 문서를 복제하지 않고 실행 상태만 갱신한다. |
| DONE | [profiler](../../utils/perf_profiler.py)는 opt-in `performance_only`, rank-0 trace ownership, bounded schedule, checkpoint/trace 분리 및 non-overwrite session을 가진다. [기본 config](../../config/cfg_pretrain.yaml)는 비활성이다. | M3 trace의 안전장치는 준비됐지만 CUDA/compile 통합은 미검증이다. |
| DONE | [training instrumentation](../../pretrain.py)은 `data_wait`, H2D, forward, backward, allreduce, optimizer, metrics-D2H, W&B, EMA 영역을 기록한다. | profile branch가 normal hot path를 대표하는지는 M2/M3 대조가 필요하다. |
| DONE | `rtk uv run pytest tests/test_perf_profiler.py -q` → `8 passed, 5 warnings` (2026-07-26). [테스트](../../tests/test_perf_profiler.py)는 opt-in, bounded trace, rank-0/경로 소유를 다룬다. | CUDA trace, DDP collective, `torch.compile` integration test는 없다. |
| DONE (CPU-only) | [raw loader artifact](../../reports/figures/2026-07-26_experiment-speed-profiling/dataloader_smoke.json): B=2048, n=19, mean 30.964 ms, p95 40.128 ms; `epochs_per_iter=20`. | canonical `epochs_per_iter=2000` startup도, H2D/GPU-idle 비율도 측정하지 않았으므로 병목/throughput 근거가 아니다. |
| BLOCKED-by-resource | [사전등록](2026-07-26_experiment-speed-profiling.md) §4.3의 **2026-07-26 snapshot**은 GPU 2–7 queue processing/98–100% utilization, GPU 0 external, GPU 1 reserved를 기록한다. 현재 시점의 timestamped resource manifest는 아직 없다. | **Phase 0은 유휴·승인된 queue slot 전까지 BLOCKED-by-resource**이다. 새 resource manifest 없이는 실행하지 않고, 이 과거 snapshot으로 현재 자원을 추정하지 않는다. **2026-07-28 갱신은 아래 §"2026-07-28 상태 갱신" 참조** — GPU 5는 PI 승인됐고 stale claim은 정리됐으나, manifest writer가 아직 없어 P0.0은 여전히 미실행이다. |

## 2026-07-28 상태 갱신 — 자원 승인과 queue 정합성 사건

측정 시각 2026-07-28 ~03:10 KST, host `aigpu0918`. 아래는 이 세션에서 직접 관측한 값이며,
아직 `manifests/resource_<UTC>.json` 로 기록되지 않았다 (P0.1a writer 미구현).

**GPU 점유 (nvidia-smi + `--query-compute-apps` 소유자 대조)**

| GPU | 상태 |
|---|---|
| 0, 1 | 외부 사용자 `ljsong7`, 각 ~37 GB. 본 plan 대상 아님 |
| 2, 3, 4, 6, 7 | fig1 코호트 5 run, 99–100% |
| **5** | **유휴 (0%, 34 MiB)** — PI가 2026-07-28 PERF-001 Phase 0 용도로 **승인** |

> **2026-07-28 11:35 정정 (중요).** 아래 "supervisor가 죽었다"는 최초 기술은 **틀렸다.**
> `scripts/queue_run.sh`는 내내 살아 있었다 — 러너 2개(등록 7일/6일 경과)가 disjoint GPU
> 집합에서 worker 6개를 돌리고 있다: 러너 A(pid 3104806, env에 `GPUS` 없음 → 기본
> `"4 5 6 7"`, worker 4개), 러너 B(pid 4010026, `GPUS="2 3"`, worker 2개). 이는 스크립트가
> `queue_run.sh:107-112`에서 안전하다고 문서화한 바로 그 패턴이다.
> **오진 원인**: 세션 초반 `ps -eo pid,etime,cmd | grep -E "queue_run|worker"` 의 `worker`
> 패턴이 커널 `[kworker/...]` 스레드 수십 개에 매칭됐고 `head`가 그 앞에서 잘라내 실제
> `queue_run.sh` 줄이 보이지 않았다. 프로세스 조회 패턴은 `grep -F "queue_run.sh"` 처럼
> 고정 문자열로 좁혀야 한다.
> **영향 평가**: 수동으로 `done/`에 옮긴 3건(0213/0216/0218)은 모두 최종 체크포인트
> `step_244100` 보유 + 로그 정지 + 대응 프로세스 부재로 **완주가 확인된 job**이었다. 진행 중인
> 학습을 끊거나 산출물을 잃은 바 없고, 중복 실행도 없다(claim이 `jobs/`로 돌아가지 않았고
> 새 러너를 기동하지 않았다). 즉 불필요했던 bookkeeping이지 손상은 아니다. 실제로 `0215`는
> 러너 B가 **자동으로** `done/`으로 옮겼다(수동 개입 없이 done 52→53).
> **따라서 앞으로 수동 `done/` 이동은 불필요하다** — 살아 있는 worker가 처리한다.
> 아래 (1)(2) 관찰과 `queue_run.sh` 기동 금지 규칙은 **여전히 유효**하다(러너가 이미 2개
> 떠 있으므로 세 번째 기동은 crash-recovery로 live claim을 회수할 수 있다).

**queue 정합성 관찰 (최초 기술; supervisor 판단만 위와 같이 정정됨).** 다음 두 사실이 확인됐다.

1. `.gpuN` claim suffix는 **신뢰할 수 없다**. claim상 GPU는 {2,3,4,5,7}인데 실제 점유는
   {2,3,4,6,7}이고, `gpu7` claim이 2개였다. 정합성 판정은 suffix가 아니라
   **job 이름 + 로그 mtime + 체크포인트 존재**로 해야 한다.
2. `0216_fig1_mlp_z_noiter_k8_s1`, `0218_fig1_mlp_noz_noiter_k8_s1`은 **정상 완주**했다.
   근거: 둘 다 `checkpoints/Sigma_k_new/<run>/step_244100` 보유(대조군 `fig1_mlp_z_iter_k3_s1`,
   `fig1_mlp_noz_iter_k3_s1` 도 동일하게 `step_244100` 종료), 로그 종료 패턴이 `done/` 상태
   run과 동일. supervisor 사망으로 `done/` rename만 누락된 bookkeeping 결손이었다.
   PI 승인 하에 2026-07-28 수동으로 `done/` 이동 → `processing: 5, done: 51, jobs: 0, failed: 0`.

**⛔ 대기 규칙 — claim 정합성 회복 전까지 `scripts/queue_run.sh` 를 어떤 `GPUS` 범위로도 기동하지 않는다.**
`jobs/` 가 비어 있어 기동 이득은 0인 반면, 기동 시 crash recovery
([queue_run.sh:113-120](../../scripts/queue_run.sh))가 claim을 `jobs/` 로 되돌려 **살아 있는
run을 재큐잉**한다. 기본 `GPUS="4 5 6 7"` 이면 live인 `0213(.gpu4)`, `0217(.gpu5)`,
`0219(.gpu7)` 이 중복 실행되어 `checkpoints/Sigma_k_new/<run_name>/` 와 `Sigma_k_new` wandb
run name에서 충돌한다. `GPUS="5"` 로 좁혀도 `0217` claim을 회수하므로 동일하게 위험하다.
위험의 근거는 crash-recovery가 **liveness(PID) 검사 없이 claim 파일만 보고** 회수한다는 점이다
(`queue_run.sh:113-120`). 이미 러너 2개가 떠 있으므로 **세 번째 기동이 곧 위험**이다.
~~남은 5개는 완주 후 수동으로 `done/` 이동한다~~ → **11:35 정정: 불필요.** 살아 있는 worker가
자동 처리하므로 수동 이동은 하지 않는다. PI 결정(2026-07-28 "수동 done")은 supervisor 사망이라는
잘못된 전제 위에서 내려진 것이므로 그 전제와 함께 철회한다.

**모니터링 (11:35 정정 반영).** 살아 있는 worker가 완주 시 `done/`, 실패 시 `failed/`로
자동 이동시킨다. `scripts/queue_run.sh status`는 `:100`의 `exit 0`으로 끝나고 crash-recovery
루프(`:113`)에 도달하지 않으므로 **읽기 전용으로 안전**하다 — 관찰에 사용해도 된다. 다만
worker를 **기동**하는 호출(인자 없는 실행)은 금지다. 교차 확인은 로그 mtime / `nvidia-smi`.

**따라서 P0.0의 차단 요인은 둘이다** — 승인된 유휴 GPU(해소됨: GPU 5)와 claim 정합성(부분
해소: stale 2건 정리, live 5건은 완주 대기). GPU 5가 물리적으로 비었다는 사실만으로는
충분조건이 아니며, plan이 M2/M3를 queue job body로 요구하는 한 queue를 안전하게 쓸 수 있어야
한다.

## 불변조건과 비목표

- 데이터는 `data/sigma_k_10/<k>`만 사용하며 순열 크기 (n=10)이다. (k)는 합성 깊이다.
- arch/`mlp_t`, H/L cycles, L layers, halt, B=2048, sample order, optimizer/LR/update 수, EMA rate/update 횟수, eval interval, 512-example probe, z logging 및 checkpoint 의미론을 유지한다.
- `probe/test_exact` (EMA 512-probe), `all/exact_accuracy` (raw online), recompute는 서로 대체하지 않는다.
- H1/L1, batch/accumulation, FP64 stablemax/loss cast, AMP/TF32, fused optimizer, eval/probe 축소는 baseline 가속이 아니다. 별도 preregistration 없이는 promotion하지 않는다. DDP는 별도 distributed-equivalence study다.

## 공통 수치 게이트

M2에서 같은 GPU/runtime으로 baseline 3회, 각 50 warm-up + 200 measured updates를 먼저 수집한다. 후보를 읽기 전에 다음 값을 `equivalence_ledger.csv`에 고정한다.

- `delta = max(0.03, 2 × baseline median의 relative 95% CI half-width)`.
- 후보 통과 속도 조건: 200개 이상 post-warm-up update에서 median `perf/update_ms`가 baseline보다 `delta` 이상 낮고, p95가 더 높아지지 않는다. 평균만으로 통과시키지 않는다.
- profiler trace는 attribution 전용이다. unprofiled M2 수치가 없으면 어떤 speedup도 기록하지 않는다.

## Phase 0 — baseline M2/M3 (상태: BLOCKED-by-resource)

### P0.1a: M2 unprofiled timing harness 구현 (PENDING; GPU 없이 선행 가능)

이 항목은 M3와 모든 speedup gate의 선행 산출물이다. 다음 이름과 책임은 **PLANNED-NAMES (미구현)** 으로 고정한다.

| planned owner | 책임 |
|---|---|
| [utils/perf_benchmark.py](../../utils/perf_benchmark.py) | `PerfBenchmarkConfig`, CUDA-event/CPU-wall collector, resource/config manifest와 CSV/ledger writer |
| [pretrain.py](../../pretrain.py) | normal (non-profiler) train path의 최소 benchmark hook; disabled path에는 timing/sync side effect 없음 |
| [config/cfg_pretrain.yaml](../../config/cfg_pretrain.yaml) | opt-in `perf_benchmark` key와 bounded performance-only acknowledgement |
| [tests/test_perf_benchmark.py](../../tests/test_perf_benchmark.py) | config bounds, disabled no-op, CSV/manifest schema, non-overwrite 및 fixed-schema tests |

**구현 후 runnable** canonical M2 preset은 profiler를 끈 normal path에서 50 warm-up + 200 measured update를 수집하고, canonical 2000-step eval event 하나를 따로 계측한다. queue job body로만 실행한다.

```bash
rtk uv run pretrain.py arch=trm global_batch_size=2048 epochs=2000 eval_interval=2000 min_eval_interval=0 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  ema=True ema_rate=0.999 evaluators="[]" checkpoint_every_eval=False \
  arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 arch.halt_max_steps=1 \
  arch.forward_dtype=bfloat16 data_paths="[data/sigma_k_10/6]" seed=1 +k=6 \
  +log_z_dynamics=True +z_snapshot=False +project_name=Sigma_k_perf \
  +run_name=perf0_m2_tf_z_iter_k6_s1_r1 perf_profiler.enabled=False \
  perf_benchmark.enabled=True perf_benchmark.performance_only=True \
  perf_benchmark.warmup_steps=50 perf_benchmark.measured_steps=200 \
  perf_benchmark.eval_event_step=2000 perf_benchmark.max_steps=2000 \
  perf_benchmark.output_dir=reports/figures/2026-07-26_experiment-speed-profiling/data/m2_tf_z_iter_k6_s1_r1
```

Timing contract: CUDA event pairs bracket H2D, forward+backward, optimizer, EMA and metric-device work; `time.perf_counter()` brackets loader `next()`, whole update, D2H/metrics-W&B, evaluation/z-probe/checkpoint events. The collector retains event pairs and calls `torch.cuda.synchronize()` only once after each measured 200-step window and at event boundaries—never per update. `perf_profiler` remains disabled. `eval_ms`, `zprobe_ms`, and `checkpoint_ms` are also written as `event_ms / eval_interval` amortized values.

Required output schema (**2026-07-28 PI 개정: 25 → 26컬럼, PERF-DEV-02**): `steady_state.csv` columns are `schema_version,condition_id,run_id,repeat,seed,step,global_effective_batch,input_tokens,target_tokens,data_wait_ms,update_wall_ms,h2d_cuda_ms,forward_backward_cuda_ms,optimizer_cuda_ms,ema_cuda_ms,metrics_device_cuda_ms,metrics_wandb_wall_ms,eval_event_ms,zprobe_event_ms,checkpoint_event_ms,eval_amortized_ms,zprobe_amortized_ms,checkpoint_amortized_ms,max_memory_allocated,max_memory_reserved,gpu_util_pct`; `manifest.json` holds command/resolved-config/data hash, git SHA/dirty state, Torch/CUDA/driver, GPU name/clock/power, hostname, start/end UTC. It records three repeats (`r1..r3`) with seed=1 and same reserved GPU; repeat/seed/hardware differences are provenance, not pooled silently.

`metrics_device_cuda_ms`는 timing contract가 요구하는 metric-device CUDA bracket의 정본 출력이다. `ema_cuda_ms` 뒤·`metrics_wandb_wall_ms` 앞에 놓여 CUDA 컬럼 5종이 contract의 열거 순서(H2D → forward+backward → optimizer → EMA → metric-device)와 같은 순서로 이어진다. 이 개정으로 `schema_version`을 올린다. manifest의 `extra_span_summary` 경로는 더 이상 이 span의 정본이 아니다.

`equivalence_ledger.csv`(candidate ID, control ID, fixed-batch/sample hashes, config diff, schema version, tolerance, G1/G2 status)는 **2026-07-28 PI 결정(PERF-DEV-10)으로 P0.1a 필수 산출물에서 G1 필수 산출물로 재배치**한다. candidate/control ID를 요구하므로 baseline-only인 M2 run은 이를 채울 수 없다. 따라서 M2 repeat 1회의 필수 산출물은 `steady_state.csv` + `manifest.json` 2종이다.

For each repeat, p50 is the median, p95 uses nearest rank `ceil(0.95N)-1`, mean uses all 200 rows, and `CV = sample_sd / mean`. Report a 95% CI for the median from 10,000 non-parametric resamples of the 200 rows; report the condition CI with a hierarchical bootstrap (resample repeat, then row) rather than treating 600 rows as independent. Freeze the common `delta` only after these baseline artifacts exist and before candidate data are read.

**구현 후 runnable** completion gate: benchmark config validates performance-only/bounds/output separation; disabled normal path is regression-tested; schema and non-overwrite tests pass; `rtk uv run pytest tests/test_perf_benchmark.py -q` and existing profiler tests pass; three complete M2 CSV+manifest sets contain 200 measured rows and one eval event. CUDA integration is evidence pending until a reserved GPU executes it. Any missing row/field, per-step synchronization, profiler-on capture, or manifest mismatch discards that repeat.

### P0.0: queue 안전 확인 (GPU 사용 전, PENDING)

P0.1a의 resource-manifest writer가 구현되기 전에는 이 단계도 실행하지 않는다. 아래는 **P0.1a 구현 후 runnable** read-only capture이고, 세 출력과 UTC/host/worktree SHA를 새 `manifests/resource_<UTC>.json`에 함께 저장한다. 실행 전에는 job을 만들거나 queue 순서를 바꾸지 않는다.

```bash
rtk scripts/queue_run.sh status
rtk scripts/queue_run.sh dry-run
rtk nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

성공: 승인된 GPU가 비어 있고, dry-run이 다른 processing claim을 회수/침범하지 않으며, 담당자가 그 GPU와 FIFO 우선순위를 명시적으로 승인한다. 중단: 어느 하나라도 타 job/예약 GPU를 보이거나, queue에 앞선 과학 job이 있으면 `BLOCKED-by-resource`를 유지한다. GPU 0–1 및 `FORCE_GPUS=1`은 사용하지 않는다.

### P0.1: profiler 전용 job 생성·검토 (PENDING, P0.0 뒤)

현 [launcher](../../scripts/sigma_enqueue.sh)는 PERF-M3 override를 발행하지 않는다. 다음 구현자는 profiler-only preset을 launcher에 추가하고, **구현 후 runnable** 아래 dry-run으로 job body를 검토한다. `rtk` 없는 직접 GPU launch는 하지 않는다.

```bash
rtk bash -n scripts/sigma_enqueue.sh scripts/queue_run.sh
rtk scripts/sigma_enqueue.sh --dry-run perf0
rtk scripts/queue_run.sh dry-run
```

job body는 아래 **구현 후 runnable** 등록된 M3 command와 바이트 단위로 registered fields가 같아야 한다. 단, queue worker가 `CUDA_VISIBLE_DEVICES`를 소유한다. `--cfg job` 출력도 같은 artifact directory에 저장한다.

```bash
rtk uv run pretrain.py arch=trm global_batch_size=2048 epochs=20 eval_interval=20 min_eval_interval=0 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  ema=True ema_rate=0.999 evaluators="[]" checkpoint_every_eval=False \
  arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 arch.halt_max_steps=1 \
  arch.forward_dtype=bfloat16 data_paths="[data/sigma_k_10/6]" seed=1 +k=6 \
  +log_z_dynamics=True +z_snapshot=False +project_name=Sigma_k_perf \
  +run_name=perf0_m3_tf_z_iter_k6_s1 perf_profiler.enabled=True \
  perf_profiler.performance_only=True \
  perf_profiler.output_dir=reports/figures/2026-07-26_experiment-speed-profiling/profiles/m3_tf_z_iter_k6_s1 \
  perf_profiler.wait=10 perf_profiler.warmup=10 perf_profiler.active=20 perf_profiler.repeat=1 \
  perf_profiler.max_steps=40 perf_profiler.record_shapes=True perf_profiler.profile_memory=True \
  perf_profiler.with_stack=True perf_profiler.export_chrome_trace=True \
  perf_profiler.export_tensorboard_trace=True
```

예상 출력: resolved config, unique `session_*/capture_*/trace.json`, TensorBoard trace 및 queue log. 실패/중단: profiler validation failure, trace가 checkpoint 아래에 쓰임, trace가 40-step schedule을 벗어남, config diff가 위 canonical fields를 바꿈. 이 M3는 performance-only이므로 accuracy/phase/checkpoint를 과학 evidence로 쓰지 않는다.

### P0.2: baseline 측정과 판정 입력 (PENDING, P0.1 뒤)

P0.2는 P0.1a의 `steady_state.csv`, `manifest.json`, `equivalence_ledger.csv` 세 baseline repeat를 먼저 소비한다. M2(비계측)는 `perf/update_ms`, samples/input-target tokens per second, H2D/forward/backward/optimizer/EMA, data wait/W&B/eval/zprobe/checkpoint amortized wall time, allocator peak, GPU util을 CSV로 낸다. 그 뒤 M3는 top operators, CPU↔CUDA gap, graph break/recompile 흔적을 남긴다. M3가 애매하게 남긴 top 1–2 kernel만 M4 Nsight 대상으로 올린다.

성공: 3×200 rows, warm-up/compile 분리, 동일 config/data/runtime manifest, trace와 CSV가 존재한다. 중단: 3회 중 하나의 환경/manifest가 달라지거나 M3가 runtime을 대표하지 못하면 baseline을 다시 수집한다. **M2 artifact가 없으면 speedup을 주장하지 않는다.** 결과가 intrinsic recurrence/FP64 loss 위주이거나 `delta` 미만이면 Phase 1 구현 없이 기록하고 멈춘다.

## Phase 1 — 하나씩만 실행 (모두 PENDING; P0.2의 관측 선택 후)

한 candidate의 G1/G2 결과가 닫히기 전 다른 candidate를 합치지 않는다. 각 항목의 대조군은 동일 commit의 candidate-off baseline이고, 동일 GPU/runtime/seed/fixed batches다.

| 순서·후보 | 가설과 소유 파일 | 구현/대조·테스트 | 수치 gate · rollback |
|---|---|---|---|
| 1. train logging cadence | 매-update `metrics.cpu().numpy()`/`wandb.log`가 M2/M3의 보존 가능 비용일 수 있다. 소유: [pretrain.py](../../pretrain.py), config, 전용 test. | primary invariant는 model trajectory와 eval/probe/z/final metrics다. per-step train series는 허용된 **observability 변경**이며 N과 aggregation schema를 candidate 전 preregister하고 보존한다. 어떤 run이 그 series를 evidence invariant로 지정하면 lossless local per-step buffer가 필수다. fixed-state logits/loss/grad/optimizer/EMA equality와 required-key/schema test를 한다. | M2 `wandb_ms` 또는 D2H가 actionability를 보이고 공통 `delta`를 통과해야 한다. required key/schema 누락 또는 model/eval 차이만 rollback 사유다; 정상적으로 preregister된 train aggregation 자체는 rollback 사유가 아니다. |
| 2. redundant z-probe forward | [_probe_forward](../../utils/z_logging.py)의 capture pass와 prediction pass가 eval event 비용일 수 있다. 소유: [utils/z_logging.py](../../utils/z_logging.py), probe test. | 한 forward history에서 final preds를 취하는 candidate 대 기존 two-pass control. frozen EMA checkpoint에서 probe input hash, preds, exact, z tensors/deltas, PCA input, phase output을 비교한다. | M2/M3에서 `zprobe_ms`가 actionability를 보이고 공통 `delta`를 통과해야 한다. 어느 probe artifact/metric이 불일치하면 즉시 old two-pass로 rollback한다. |
| 3. nonblocking H2D | pinned batch의 `{k: v.cuda()}`가 compute와 겹치지 않아 gap이 남을 수 있다. 소유: [pretrain.py](../../pretrain.py), dedicated H2D/order test. | `non_blocking=True` 및 전용 stream/prefetch는 별 candidate로 한다. 대조군은 현재 blocking transfer; sample-index trace, batch bytes, carry/update order를 기록한다. | trace가 H2D→compute gap을 보이고 공통 `delta`를 통과해야 한다. race, changed sample hash, allocator regression/p95 regression이면 rollback한다. |
| 4. compile graph-break 진단 | compile default-on path에 graph break/recompile가 있으면 원인을 제거할 여지가 있다. 소유: [pretrain.py](../../pretrain.py), model/call-site, regression test. | **구현 후 runnable** diagnostic-only job에서 `rtk env TORCH_LOGS=graph_breaks,recompiles TORCHDYNAMO_VERBOSE=1`를 M3 job body에 추가해 log를 보존한다. exact break site가 없으면 code change하지 않는다. | actionable break가 trace/log와 연결되고, compile-on G1 4-cell/`_orig_mod` checkpoint load-smoke 및 공통 `delta`를 모두 통과해야 한다. key/load or numerical failure면 prior compile setting으로 rollback한다. |

## Phase 2 — 고위험 후보 (모두 PENDING; Phase 1 종료 뒤만)

진입 조건은 (a) P0 attribution에서 해당 항목이 total/campaign wall time의 **10% 이상**, (b) 더 낮은 위험 후보가 불통과/비실행 가능, (c) 독립 verifier가 G1 protocol을 사전 검토한 경우다.

- EMA foreach/vectorization: [models/ema.py](../../models/ema.py)와 [pretrain.py](../../pretrain.py) 소유. EMA tensor, copied-model logits, checkpoint state-dict/load를 4 cell에서 검증한다. 산술 순서/cast가 달라지면 baseline-equivalent가 아니다.
- data pipeline: [puzzle_dataset.py](../../puzzle_dataset.py)와 loader path 소유. worker/vectorization/prefetch는 update별 sample-index trace 및 batch bytes가 같을 때만 계속한다. 그렇지 않으면 data-order protocol로 분리한다.
- casts/kernel changes: FP64 stablemax/loss, BF16/TF32/autocast의 어떤 변경도 이 plan의 promotion 대상이 아니다. 필요하면 별 EXP를 preregister한다.
- DDP: allreduce/topology/sharding/checkpoint/compile이 바뀌므로 이 plan에서는 실행하지 않는다. 별 DDP study의 G1/G2와 single-GPU comparator가 필요하다.

## 고정 상태 → canary → promotion

1. **G1 frozen-state (PENDING):** `trm`/`trm_singlez` × H3/L6/H1/L1에 기록한 fixed batches를 사용한다. sample index, logits/loss/preds, gradients/post-step state, EMA, z-probe, checkpoint keys/reload를 candidate-off와 비교한다. logging-only는 exact equality; numerical/compiler는 candidate를 보기 전 baseline 반복으로 tolerance를 고정한다.
2. **G2 paired canary (PENDING):** 하나의 predeclared k/seed와 네 cell에서 baseline/candidate를 짝지어 full 100000 epochs로 실행한다. 동일 eval step grid의 primary/corroboration/z keys, total optimizer/EMA count, final checkpoint, threshold-near recompute를 확인한다. 공통 speed gate와 independent local-W&B/disk ledger audit을 모두 통과해야 한다.
3. **G3 promotion cohort (PENDING):** G2 PASS 후에만 intended k grid를 새 prefix/runtime tag로 enqueue한다. legacy와 candidate를 pool하지 않는다. 실패하면 switch를 끄고 failed evidence를 유지한다.

## 의존성·병렬성·산출물

| work item | status | 의존성 | 병렬 가능 여부 | writeback |
|---|---|---|---|---|
| P0.1a static M2 harness + unit tests | PENDING | 없음; GPU 불필요 | **P0.0 전 병렬 가능** | planned-name implementation, test output |
| P0.0 timestamped queue/resource manifest | BLOCKED-by-resource | P0.1a resource writer + approved idle GPU | 아니오 | `resource_<UTC>.json` + 이 문서 status |
| P0.1 static profiler preset + unit tests | PENDING | 없음; GPU dry-run 전까지 | **P0.0 전 병렬 가능** | preset code/test output |
| P0.1 GPU dry-run/capture | PENDING | P0.0 approval + P0.1 static implementation | 아니오 | job body/resolved config/trace |
| P0.2 M2/M3/M4 baseline interpretation | PENDING | P0.1a M2 artifacts + P0.1 capture | 아니오 | CSV/trace/environment manifest |
| Phase 1 diagnosis prep | PENDING | P0.2 observed bottleneck | 읽기 전용 분석은 병렬 | candidate decision record |
| candidate implementation + G1/G2 | PENDING | 단 하나의 selected candidate | **후보 간 병렬 금지** | code/test + equivalence ledger |
| G3 cohort | PENDING | independent G2 PASS | 아니오 | separate prefix/W&B tags |

완료 체크리스트: dry-run job body; resolved config/data/runtime hashes; raw timing CSV; M3/M4 traces; frozen `delta`; G1 ledger; paired-canary ledger; independent verifier report; candidate-specific tests; 후보 구현 후 relevant `rtk bash -n scripts/*.sh`, `rtk scripts/sigma_enqueue.sh --dry-run <approved-prefix>`, `rtk uv run pytest tests/`, 그리고 checkpoint/model semantics를 건드린 경우 `rtk uv run python measure_rho.py ...` smoke. 커밋은 사용자만 한다.

## 2026-07-28 P0.1a 구현 비준 원장 (PERF-DEV-NN)

P0.1a/P0.1-static 구현이 랜딩됐다(`utils/perf_benchmark.py`, `tests/test_perf_benchmark.py`,
`pretrain.py` 배선, `config/cfg_pretrain.yaml`, `scripts/sigma_enqueue.sh` perf0). 독립 게이트
판정은 **FAIL**이며, 사유는 정확성 결함이 아니라 **미비준 이탈 + 미충족 gate clause**다.
계약 준수 자체는 확인됐다 — 25컬럼 CSV 헤더 이름·순서 일치, p95 nearest-rank
`ceil(0.95N)-1`, sample-SD CV, 10,000-resample median CI, 2단계 hierarchical bootstrap 모두
계약과 일치하고, `hierarchical_bootstrap_median_ci`는 flat sequence를 거부해 600 row pooling을
사고로 못 하게 막는다.

**라벨 규율.** 코드의 `R1/R2/R7/R8`은 모듈 로컬 작업 라벨이고 두 사전등록 문서 어디에도
정의된 risk register가 없다. **이 라벨을 사전등록 risk ID로 인용하지 않는다.** 아래
`PERF-DEV-NN`이 정본 번호다.

### 구현 수준 — 이 문서로 비준함 (사전등록 산출물 불변)

| ID | 이탈 | 비준 사유 |
|---|---|---|
| PERF-DEV-01 | `event_span`이 `==` 대신 `step >= eval_event_step`으로 arm | 등호는 결코 발화하지 않는다(M2 preset에서 eval 블록은 k=6 기준 step ~4.9k). 안전성의 실제 근거는 step 비교가 아니라 `_window_closed` 무조건 게이트이며, window가 열린 동안 `event_span`이 `_NULL_SPAN`을 반환해 mid-window drain이 도달 불가다. |
| PERF-DEV-03 | `data_wait`를 row 생성 전에 bracket 후 staging→승격 | loader `next()`가 `begin_update`보다 앞선다. "열린 row 밖 span 금지"를 문자 그대로 읽으면 frozen 컬럼 하나가 영구 `evidence_pending`이 된다. |
| PERF-DEV-04 | token count를 update-wall 타이머 시작 **전에** 계산 | label tensor scan이 `update_wall_ms`를 부풀리지 않게 한다. 측정 대상이 아닌 작업을 측정에서 제외하는 것이므로 계약 취지에 부합. |
| PERF-DEV-05 | `TrainingBenchmark`를 `profiler.start()` 앞에 생성 | mutual-exclusion ValueError가 profiler 자원 획득 전에, 그리고 `finally: profiler.stop()`이 덮지 않는 지점 밖에서 발생하도록. |

### 사전등록 변경 — **PI 결정 완료 (2026-07-28)**

| ID | 쟁점 | PI 결정 |
|---|---|---|
| PERF-DEV-02 | **계약 자체의 내부 모순.** timing contract는 "metric-device work"에 CUDA event pair를 요구하는데, byte-frozen 25컬럼 스키마에는 해당 컬럼이 없었다. 구현은 span을 수집하되 집계를 `manifest.extra_span_summary`로 보냈다. | **스키마를 26컬럼으로 개정한다.** `metrics_device_cuda_ms`를 `ema_cuda_ms` 뒤에 추가해 timing contract를 문자 그대로 충족시킨다. byte-freeze는 이 개정으로 대체되고 `schema_version`을 올린다. manifest `extra_span_summary`는 이 span의 정본이 아니게 된다. (참고: 메인 세션 권고는 freeze 유지였으나 PI가 계약 충실 쪽을 택했다.) |
| PERF-DEV-10 | `equivalence_ledger.csv`는 계약상 P0.1a 필수 산출물인데 `append_equivalence_ledger`에 **production caller가 없다.** | **G1 필수로 재배치한다.** candidate/control ID를 요구하므로 baseline-only인 M2 run은 채울 수 없다. M2 repeat 1회의 필수 산출물은 `steady_state.csv` + `manifest.json` 2종이다. writer·스키마는 이미 계약대로 구현돼 있어 G1 진입 시 호출만 하면 된다. |

### 미등록 부수 변경 — 기록만 (사고 비준 방지)

- **PERF-DEV-06**: `perf_benchmark.max_steps`는 **row 수집만 bound하고 학습 길이를 bound하지 않는다.** `perf_profiler.max_steps`와 다르다. M2 preset의 실제 상한은 Hydra `epochs=2000`(≈4.9k optimizer step)이다. 이름이 반대로 읽히므로 주의.
- **PERF-DEV-07**: 측정 window 안의 **두 번째 per-update 암묵 sync**. `pretrain.py:317`의 `{k: v.cuda()}`는 `non_blocking=False`라 H2D마다 stream을 동기화한다(HEAD에도 있던 기존 동작, 이번 변경이 도입한 게 아님). 모듈 docstring은 `metric_values.cpu().numpy()`만 열거한다. **따라서 `sync_count == 1`은 "측정 window에 device serialization이 없다"는 뜻이 아니라 "collector가 유발한 sync가 1회"라는 뜻으로만 읽어야 한다.** M2 run에서 `torch.cuda.set_sync_debug_mode("warn")`로 1줄 확인 가능. 이 항목은 Phase 1 후보 3(nonblocking H2D)의 근거와 직결된다.
- **PERF-DEV-08**: 학습 루프 전체를 `try:/finally:`로 감싸 teardown 순서가 disabled run 포함 모든 run에서 바뀐다(둘 다 no-op이지만 control flow 변경 자체는 미등록).
- **PERF-DEV-09**: `wandb.init(config=config.model_dump())`가 이후 **모든 과학 run의 wandb config에 `perf_profiler`/`perf_benchmark` 블록을 싣는다.** 다운스트림 figure 코드가 읽는 산출물을 건드리므로 명시 기록.
- **PERF-DEV-15**: `pyproject.toml`의 `matplotlib==3.10.8` 정확 pin과 `pandas>=2.2.0` 추가는 P0.1a 범위가 아니다. 이 원장으로 비준되지 않는다.
- **PERF-DEV-16**: perf0 job body가 `rtk` 접두어를 뺀다(사전등록 §4.2는 `rtk` 사용 규정). 코드 주석에만 기록돼 있고 `CUDA_VISIBLE_DEVICES` 생략과 달리 문서 비준이 없다. operator shell 필터라는 논거는 타당하나 기록 필요.

### 미충족 완료 게이트 (FAIL 사유, 후속 작업 대상)

| ID | clause | 상태 (2026-07-28 종료 시점) |
|---|---|---|
| PERF-DEV-12 | "disabled normal path is regression-tested" | **MET.** `tests/test_pretrain_perf_disabled_path.py` 신설(13 test) — `pretrain.py`를 실제 import해 `train_batch`를 구동한다. 커버리지는 2단이다: (i) autouse fixture가 **`torch.cuda` 속성 10개 + `torch.Tensor.cuda`** 를 raise로 막고, (ii) 같은 fixture가 테스트 전후 `torch.cuda.is_initialized()` **델타**를 단언한다. **(i)은 전 진입점을 덮지 않는다** — factory `device=` kwarg, `Tensor.to("cuda")`, `torch.device("cuda")` 컨텍스트(C method_descriptor라 패치 불가, `pretrain.py`가 disabled path에서 사용), `_lazy_init`(`torch.cuda.init` 패치는 장식일 뿐)이 모두 우회한다. 그 경로들을 잡는 것이 (ii)다. 수정 이력은 아래 참조. |
| PERF-DEV-14 | 실행 가능한 M2 경로 | **MET(코드 산출물).** `emit_perf0_m2_job` + `PERF0_M2_REPEATS=(1 2 3)`. `--dry-run perf0` → 4 job(M3 1 + M2 r1/r2/r3), 등록 필드 전부 일치, `_r<n>`이 `run_name`과 `output_dir` 양쪽에. `PERF0_ALLOW_ENQUEUE=1` 없이는 실제 enqueue 거부(근거: 사전등록 §4.1 "no production enqueue in this registration"). 실행은 evidence-pending. |
| PERF-DEV-00 | register 완전성 허위 주장 | **MET.** docstring이 스스로를 register로 주장하지 않고 이 원장을 authority로 지목하며, 불일치 시 원장이 이긴다고 명시. PERF-DEV-05/06/07 편입. 모듈 로컬 `R1/R2/R7/R8` 라벨은 제거됐고 잔존 언급은 "제거됨"을 설명하는 이력 문단뿐이다. |
| PERF-DEV-13 | "3× M2 CSV+manifest, 각 200 row + eval event 1회" | **EVIDENCE-PENDING.** 승인된 GPU M2 run 전까지 구조적으로 불가. 계획의 evidence-pending 조항이 이미 예견함. |
| PERF-DEV-17 | 등록된 repeat 계약의 "same reserved GPU" 절반 | **해소 경로 확정.** emitter는 강제하지 않으나, [자원 계획](2026-07-28_perf001-resource-plan.md) §4의 R2–R3(러너 drain 후 예약 GPU 1개에 단일 worker 기동)가 worker를 1개로 만들어 r1..r3를 직렬화하므로 구조적으로 충족된다. |
| **PERF-DEV-18** | **사전등록된 M3 명령이 실행 불가였다.** 사전등록 §4.2.1은 `export_chrome_trace=True`와 `export_tensorboard_trace=True`를 **둘 다** 설정한다. torch는 `on_trace_ready`당 kineto save를 1회만 허용하므로 두 번째가 `RuntimeError: Trace is already saved.`를 `profiler.step()`에서 던져 **학습 run이 중단된다.** 즉 등록된 M3는 한 번도 실행될 수 없는 명령이었다. | **정정.** (1) `export_tensorboard_trace` 기본값을 `true → false`로 바꾼다 — 기존 기본값 조합은 프로파일러를 사용 불가 상태로 두고 있었다. tensorboard handler도 chrome 형식 JSON을 쓰므로 손실 없음. (2) 두 export 동시 활성화를 **config 검증에서 거부**한다 — run 중간 중단을 즉시 읽히는 설정 오류로 바꾼다. (3) M3 실행 시 `perf_profiler.export_tensorboard_trace=False`를 명시한다. 독립 재현: CPU-only, `export_stacks` 미설정 상태에서도 동일 재현(선행 결함이며 이번 변경이 유발한 것이 아님). |
| PERF-DEV-19 | flame chart 도구 신설 | `utils/perf_profiler.py`에 `export_stacks`(folded stacks, `self_cpu_time_total`/`self_cuda_time_total`, `with_stack` 미설정 시 검증 오류), `analysis/flamegraph.py`에 표준 라이브러리 전용 SVG 렌더러. 형제 프레임 **알파벳순**·색은 이름 해시에서만 도출 → 동일 입력에 바이트 동일 출력이라 전/후 비교가 성립한다. `config/cfg_pretrain.yaml`에 `export_stacks` 키를 선언해 Hydra `+append` 없이 override 가능. |

**PERF-DEV-12 수정 이력 (self-graded → 독립 검증 → 정정).** 최초 랜딩분은 두 테스트에서
`assert torch.cuda.is_initialized() is False`를 인라인으로 걸었는데, 이는 **프로세스 전역**이라
`tests/test_perf_profiler.py`(profiler `step()`/`stop()`에서 kineto/CUPTI가 CUDA를 초기화)가
먼저 돌면 실패했다 — 격리 13 passed / 전체 2 failed. 1차 조치로 그 두 단언을 **삭제**했으나,
독립 verifier가 이를 **coverage regression으로 반증**했다: disabled path에
`torch.zeros(1, device="cuda")`와 `torch.empty(1).to("cuda")`를 주입해도 스위트가 13 passed로
통과했다(즉 tripwire가 그 경로를 못 잡는다). 최종 조치는 verifier가 제시·실측한 **fixture
baseline-delta** 패턴이다. 재현 확인: 같은 주입에 대해 이제
`AssertionError: disabled path created a CUDA context`로 실패하며, 적용 범위가 2개 테스트에서
파일 전체 13개로 넓어졌다. 작성자≠검증자 원칙에 따라 이 수정은 저자(메인 세션)가 아닌
독립 verifier의 실측으로 판정됐다.

**`queue_run.sh status`는 안전하다(확인).** status/dry-run 분기는 `queue_run.sh:100`의
`exit 0`으로 종료하고, crash recovery 루프는 `:113`이라 도달하지 않는다. 따라서 운영 규칙은
"`queue_run.sh`를 절대 실행 금지"가 아니라 **"worker를 기동하지 말 것; `status`/`dry-run`은
읽기 전용으로 안전"** 이다. P0.0의 `capture_resource_manifest`가 이 둘을 실제로 shell out한다.

**2026-07-28 종료 시점 게이트 실측**: `rtk uv run pytest tests/ -q` → **150 passed**(격리 13,
`test_perf_profiler.py` 선행 순서 21 passed로 순서 의존성 해소).
`rtk bash -n scripts/sigma_enqueue.sh scripts/queue_run.sh` OK. `--dry-run perf0` → 4 job,
`jobs/` 여전히 0. queue 무결성 `jobs=0 processing=4 done=52`.

**속도 수치는 여전히 0건이다.** CUDA 통합은 정의상 evidence-pending이며(모든 CUDA 진입점이
테스트에서 fake 주입), 승인된 M2 run 전에는 어떤 speedup도 기록하지 않는다.

## 기록·provenance 운영 규칙

- 결과 원본/figures는 `reports/figures/2026-07-26_experiment-speed-profiling/`에, 산문 상태/결정은 이 문서와 [사전등록](2026-07-26_experiment-speed-profiling.md) §6–7에 날짜를 붙여 append한다. 원본 trace/CSV를 overwrite하거나 사전등록의 prediction을 소급 편집하지 않는다.
- 코드 변경 PR/작업에는 candidate ID, candidate-off control, resolved diff, test output, rollback switch와 G1/G2 writeback 위치를 명시한다. independent verifier가 implementer와 다른 컨텍스트에서 ledger를 재계산한다.
- 이 계획과 다른 `lab/` 산문도 현재 ignore 규칙 아래에 있다. 커밋 정본으로 남길 때는 사용자가 의도적으로 `rtk git add -f lab/reports/2026-07-26_experiment-speed-action-plan.md`를 실행한다.
- [loader script](../../analysis/dataloader_smoke.py)와 [raw JSON](../../reports/figures/2026-07-26_experiment-speed-profiling/dataloader_smoke.json)은 현재 `.gitignore` 대상이다. provenance를 커밋에 포함해야 할 때만 사용자가 명시적으로 다음을 실행한다.

```bash
rtk git add -f analysis/dataloader_smoke.py reports/figures/2026-07-26_experiment-speed-profiling/dataloader_smoke.json
```
